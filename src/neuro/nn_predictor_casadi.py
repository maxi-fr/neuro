"""CasADi port of the Equinox MLP predictor in :mod:`neuro.prediction`.

Provides a symbolic, single-step equivalent of
:class:`neuro.prediction.AutoregressivePredictor` for embedding as a dynamics model in a
CasADi-based MPC. Reuses :class:`neuro.prediction.MLPArtifact` to load the artifact, so
the extracted weights are bit-identical to what JAX uses -- no retraining or
reimplementation of the network.

Unlike :mod:`neuro.prediction`, where the recursion over a horizon happens entirely in
normalized (z-scored) space, :func:`NNSymbolicModel.step`/:func:`NNSymbolicModel.output`
keep raw physical units at their boundary (matching how the project's previous CasADi
Jansen-Rit adapter kept ``f_step``/``f_out`` in raw state units); normalization is an
internal-only detail baked into the compiled graph as numeric constants.

The model has no free/symbolic parameters (:attr:`NNSymbolicModel.free_syms` is always
empty) -- it is purely a fitted numeric map, unlike a physics model with physically
meaningful coefficients to expose as system-identification decision variables.

Note that the ReLU activation has a non-smooth kink at zero; composing it across
``depth`` layers and many MPC horizon steps may warrant a smooth substitute (e.g.
softplus) if it causes convergence trouble in an IPOPT-based MPC. Not addressed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Self

import casadi as ca
import equinox as eqx
import numpy as np
import numpy.typing as npt

from neuro.prediction import MLPArtifact, unzscore, zscore

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

FloatArray = npt.NDArray[np.float64]


def _extract_mlp_layers(mlp: eqx.nn.MLP) -> list[tuple[FloatArray, FloatArray]]:
    """Pull plain numpy ``(weight, bias)`` pairs out of every ``Linear`` layer.

    Parameters
    ----------
    mlp : eqx.nn.MLP
        A trained Equinox MLP (any ``depth >= 0``).

    Returns
    -------
    list of (FloatArray, FloatArray)
        Per-layer ``(weight, bias)`` pairs, ``weight`` shaped ``(out, in)`` and ``bias``
        shaped ``(out,)``, in forward-pass order.
    """
    layers = []
    for layer in mlp.layers:
        if layer.bias is None:
            msg = "MLP layers without a bias are not supported"
            raise ValueError(msg)
        layers.append((np.asarray(layer.weight, dtype=np.float64), np.asarray(layer.bias, dtype=np.float64)))
    return layers


def _mlp_forward_ca(x: ca.SX | ca.MX, layers: Sequence[tuple[FloatArray, FloatArray]]) -> ca.SX | ca.MX:
    """Evaluate an MLP forward pass symbolically, replicating ``eqx.nn.MLP.__call__``.

    ReLU is applied after every layer except the last; the last layer has no activation
    (matching this project's MLPs, which never override ``final_activation``).

    Parameters
    ----------
    x : ca.SX | ca.MX
        Input column vector, shape ``(in_size, 1)``.
    layers : Sequence[(FloatArray, FloatArray)]
        Per-layer ``(weight, bias)`` pairs as returned by :func:`_extract_mlp_layers`.

    Returns
    -------
    ca.SX | ca.MX
        Output column vector, shape ``(out_size, 1)``.
    """
    for w, b in layers[:-1]:
        x = ca.fmax(ca.mtimes(w, x) + b.reshape(-1, 1), 0.0)
    w_last, b_last = layers[-1]
    return ca.mtimes(w_last, x) + b_last.reshape(-1, 1)


@dataclass(frozen=True)
class NNSymbolicModel:
    """CasADi-symbolic, single-step equivalent of the Equinox NN predictor.

    The "state" is the concatenation of the raw-unit y-window (``n_y x n_channels``)
    and u-window (``n_u x n_controls``), flattened row-major. All recurrence memory
    lives in this state vector, so :attr:`history_depth` is ``0`` and :meth:`step`
    reduces to a pure ``x_next = F(x, u)`` map.

    Attributes
    ----------
    artifact : MLPArtifact
        The loaded predictor, native dt, and scalers.
    """

    artifact: MLPArtifact

    @classmethod
    def from_artifact(cls, artifact: str | Path) -> Self:
        """Build the model by loading a 3-file artifact from disk."""
        return cls(MLPArtifact.load(artifact))

    @property
    def state_shape(self) -> tuple[int, int]:
        """Shape of the flattened state column vector."""
        n_y, n_u = self.artifact.n_y, self.artifact.n_u
        n_ch, n_ctrl = self.artifact.n_channels, self.artifact.n_controls
        return (n_y * n_ch + n_u * n_ctrl, 1)

    history_depth: int = 0

    @property
    def n_elec(self) -> int:
        """Number of control input channels."""
        return self.artifact.n_controls

    @property
    def n_channels(self) -> int:
        """Number of EEG output channels."""
        return self.artifact.n_channels

    @property
    def free_syms(self) -> dict[str, ca.MX]:
        """Free symbolic parameters; always empty -- this model is purely numeric."""
        return {}

    @cached_property
    def _layers(self) -> tuple[tuple[FloatArray, FloatArray], ...]:
        """Numpy ``(weight, bias)`` pairs extracted from the artifact's inner MLP."""
        mlp = self.artifact.model.model
        if not isinstance(mlp, eqx.nn.MLP):
            msg = f"expected AutoregressivePredictor.model to be an eqx.nn.MLP, got {type(mlp)}"
            raise TypeError(msg)
        return tuple(_extract_mlp_layers(mlp))

    def step(self, history: Sequence[ca.SX | ca.MX], u: ca.SX | ca.MX) -> ca.SX | ca.MX:
        """Advance one step: ``history == [x]`` (raw units), ``u`` raw control -> ``x_next``.

        The state packs the y-window/u-window as row-major flattens (matching
        ``jnp.ndarray.flatten()``), so "drop the oldest step, append the newest" is just
        slicing off/on a contiguous block of ``n_channels``/``n_controls`` entries --
        no reshape into a 2-D window is needed. Per-channel mean/scale constants are
        tiled across the window length to normalize the flat vector directly (CasADi,
        unlike numpy, does not broadcast a ``(1, n)`` constant against an ``(m, n)``
        symbolic matrix, so working in flat-vector form throughout avoids that
        entirely).
        """
        (x,) = history
        artifact = self.artifact
        n_y, n_u = artifact.n_y, artifact.n_u
        n_ch, n_ctrl = artifact.n_channels, artifact.n_controls

        y_flat = x[: n_y * n_ch]
        u_flat = x[n_y * n_ch :]

        new_u_flat = ca.vertcat(u_flat[n_ctrl:], u)

        y_mean_tiled = np.tile(artifact.y_mean, n_y).reshape(-1, 1)
        y_scale_tiled = np.tile(artifact.y_scale, n_y).reshape(-1, 1)
        u_mean_tiled = np.tile(artifact.u_mean, n_u).reshape(-1, 1)
        u_scale_tiled = np.tile(artifact.u_scale, n_u).reshape(-1, 1)

        y_scaled_flat = zscore(y_flat, y_mean_tiled, y_scale_tiled)
        new_u_scaled_flat = zscore(new_u_flat, u_mean_tiled, u_scale_tiled)

        mlp_in = ca.vertcat(y_scaled_flat, new_u_scaled_flat)
        y_next_scaled = _mlp_forward_ca(mlp_in, self._layers)
        y_mean = artifact.y_mean.reshape(-1, 1)
        y_scale = artifact.y_scale.reshape(-1, 1)
        y_next_raw = unzscore(y_next_scaled, y_mean, y_scale)

        new_y_flat = ca.vertcat(y_flat[n_ch:], y_next_raw)
        return ca.vertcat(new_y_flat, new_u_flat)

    def output(self, x: ca.SX | ca.MX) -> ca.SX | ca.MX:
        """Slice the most-recently-predicted raw EEG values out of a state vector."""
        n_y, n_ch = self.artifact.n_y, self.artifact.n_channels
        return x[(n_y - 1) * n_ch : n_y * n_ch]

    @cached_property
    def f_step(self) -> ca.Function:
        """Reusable compiled single-step function ``(x, u) -> x_next``."""
        x_sym = ca.MX.sym("x", *self.state_shape)
        u_sym = ca.MX.sym("u", self.n_elec, 1)
        return ca.Function("F_step_nn", [x_sym, u_sym], [self.step([x_sym], u_sym)])

    @cached_property
    def f_out(self) -> ca.Function:
        """Reusable compiled output function ``x -> y``."""
        x_sym = ca.MX.sym("x", *self.state_shape)
        return ca.Function("F_out_nn", [x_sym], [self.output(x_sym)])
