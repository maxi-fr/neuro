"""CasADi port of the Equinox MLP predictor in :mod:`neuro.prediction`.

Provides a symbolic, single-step equivalent of
:class:`neuro.prediction.AutoregressivePredictor` for embedding as a dynamics model in a
CasADi-based MPC. Reuses :class:`neuro.prediction.MLPArtifact` to load the artifact, so
the extracted weights are bit-identical to what JAX uses -- no retraining or
reimplementation of the network.

The state carries the y/u windows in model space -- standardized channels, or the latent
PCA components under a projection -- matching what :meth:`MLPArtifact.encode` produces, so
the MPC shooting state stays in the reduced latent space. :func:`NNSymbolicModel.step`
consumes the raw control, standardizing it into the graph as numeric constants, and
:func:`NNSymbolicModel.output` decodes the model-space state back to raw EEG (inverse PCA,
then inverse standardization).

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

from neuro.prediction import MLPArtifact
from neuro.transforms import unzscore, zscore

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from neuro.types import FloatArray


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


def _mlp_forward_ca(
    x: ca.SX | ca.MX, layers: Sequence[tuple[FloatArray, FloatArray]], activation: str
) -> ca.SX | ca.MX:
    """Evaluate an MLP forward pass symbolically, replicating ``eqx.nn.MLP.__call__``.

    ReLU is applied after every layer except the last; the last layer has no activation
    (matching this project's MLPs, which never override ``final_activation``).

    Parameters
    ----------
    x : ca.SX | ca.MX
        Input column vector, shape ``(in_size, 1)``.
    layers : Sequence[(FloatArray, FloatArray)]
        Per-layer ``(weight, bias)`` pairs as returned by :func:`_extract_mlp_layers`.
    activation : str
        The activation function name ("relu", "tanh", or "softplus").

    Returns
    -------
    ca.SX | ca.MX
        Output column vector, shape ``(out_size, 1)``.
    """
    for w, b in layers[:-1]:
        z = ca.mtimes(w, x) + b.reshape(-1, 1)
        if activation == "relu":
            x = ca.fmax(z, 0.0)
        elif activation == "tanh":
            x = ca.tanh(z)
        elif activation == "softplus":
            x = ca.log(1 + ca.exp(z))
        else:
            msg = f"Unsupported activation: {activation}"
            raise ValueError(msg)
    w_last, b_last = layers[-1]
    return ca.mtimes(w_last, x) + b_last.reshape(-1, 1)


@dataclass(frozen=True)
class NNSymbolicModel:
    """CasADi-symbolic, single-step equivalent of the Equinox NN predictor.

    The "state" is the concatenation of the y-window (``n_y x n_channels``) and u-window
    (``n_u x n_controls``), flattened row-major. All recurrence memory lives in this state
    vector, so :attr:`history_depth` is ``0`` and :meth:`step` reduces to a pure
    ``x_next = F(x, u)`` map. When the artifact carries a fixed PCA projection the y-window
    holds the ``k``-dimensional latent components rather than raw EEG (so ``n_channels`` is
    ``k``); :meth:`output` decodes them back to the ``n_eeg_channels`` raw EEG channels.

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
    def n_controls(self) -> int:
        """Number of control input channels."""
        return self.artifact.n_controls

    @property
    def n_channels(self) -> int:
        """Number of channels carried in the state per step (latent components with a projection)."""
        return self.artifact.n_channels

    @property
    def n_eeg_channels(self) -> int:
        """Number of raw EEG channels at the output boundary (equals ``n_channels`` without a projection)."""
        return self.artifact.n_eeg_channels

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

    def predict_output(self, y_flat: ca.SX | ca.MX, new_u_flat: ca.SX | ca.MX) -> ca.SX | ca.MX:
        """Predict the next model-space EEG sample from a flat y-window and (shifted) u-window.

        This is the network-evaluation core shared by :meth:`step` (which packs the result
        back into a full state) and the output-condensed MPC (which lifts only this output).
        The windows are row-major flattens (matching ``jnp.ndarray.flatten()``); ``new_u_flat``
        must already include the newest control as its last ``n_controls`` block, exactly as
        :meth:`step` builds it. The y-window already carries model-space values (standardized
        channels, or latent components under a projection), so it feeds the network directly;
        only the raw control window is standardized here. The u standardizer's mean/scale are
        broadcast to per-control length (handling both per-channel arrays and a single global
        scalar) then tiled across the window, since CasADi -- unlike numpy -- does not broadcast
        a ``(1, n)`` constant against an ``(m, n)`` symbolic matrix.

        Parameters
        ----------
        y_flat
            The model-space y-window flattened row-major, shape ``(n_y * n_channels, 1)``.
        new_u_flat
            The raw-unit u-window flattened row-major, shape ``(n_u * n_controls, 1)``, newest
            control last.

        Returns
        -------
        ca.SX | ca.MX
            The next model-space EEG prediction, shape ``(n_channels, 1)``.
        """
        n_u, n_ctrl = self.artifact.n_u, self.artifact.n_controls
        u_std = self.artifact.u_pipeline.standardizer
        if u_std is None:
            msg = "u-pipeline must carry a standardizer"
            raise ValueError(msg)

        u_mean_tiled = np.tile(np.broadcast_to(u_std.center, (n_ctrl,)), n_u).reshape(-1, 1)
        u_scale_tiled = np.tile(np.broadcast_to(u_std.scale, (n_ctrl,)), n_u).reshape(-1, 1)

        new_u_scaled_flat = zscore(new_u_flat, u_mean_tiled, u_scale_tiled)  # ty: ignore[invalid-argument-type]
        mlp_in = ca.vertcat(y_flat, new_u_scaled_flat)
        return _mlp_forward_ca(mlp_in, self._layers, self.artifact.model.activation)

    def step(self, history: Sequence[ca.SX | ca.MX], u: ca.SX | ca.MX) -> ca.SX | ca.MX:
        """Advance one step: ``history == [x]`` (model space), ``u`` raw control -> ``x_next``.

        The state packs the model-space y-window/u-window as row-major flattens, so "drop the
        oldest step, append the newest" is just slicing off/on a contiguous block of
        ``n_channels``/``n_controls`` entries -- no reshape into a 2-D window is needed. The
        network evaluation itself is delegated to :meth:`predict_output`.
        """
        (x,) = history
        n_y, n_ch = self.artifact.n_y, self.artifact.n_channels
        n_ctrl = self.artifact.n_controls

        y_flat = x[: n_y * n_ch]
        u_flat = x[n_y * n_ch :]

        new_u_flat = ca.vertcat(u_flat[n_ctrl:], u)
        y_next_model = self.predict_output(y_flat, new_u_flat)

        new_y_flat = ca.vertcat(y_flat[n_ch:], y_next_model)
        return ca.vertcat(new_y_flat, new_u_flat)

    def output(self, x: ca.SX | ca.MX) -> ca.SX | ca.MX:
        """Slice the most-recent model-space sample from the state and decode to raw EEG.

        The state holds model-space values, so the last vector is decoded by inverting the
        y-pipeline: undo the PCA projection (``y = E.T @ z + mean``) when present, then undo
        the channel standardization. This is the symbolic mirror of
        :meth:`neuro.prediction.MLPArtifact.decode`.
        """
        n_y, n_ch = self.artifact.n_y, self.artifact.n_channels
        z_last = x[(n_y - 1) * n_ch : n_y * n_ch]

        pca = self.artifact.y_pipeline.pca
        y_std = ca.mtimes(pca.basis.T, z_last) + pca.mean.reshape(-1, 1) if pca is not None else z_last

        std = self.artifact.y_pipeline.standardizer
        if std is None:
            msg = "y-pipeline must carry a standardizer"
            raise ValueError(msg)
        n_eeg = self.n_eeg_channels
        center = np.broadcast_to(std.center, (n_eeg,)).reshape(-1, 1)
        scale = np.broadcast_to(std.scale, (n_eeg,)).reshape(-1, 1)
        return unzscore(y_std, center, scale)  # ty: ignore[invalid-return-type]

    @cached_property
    def f_step(self) -> ca.Function:
        """Reusable compiled single-step function ``(x, u) -> x_next``."""
        x_sym = ca.MX.sym("x", *self.state_shape)
        u_sym = ca.MX.sym("u", self.n_controls, 1)
        return ca.Function("F_step_nn", [x_sym, u_sym], [self.step([x_sym], u_sym)])

    @cached_property
    def f_out(self) -> ca.Function:
        """Reusable compiled output function ``x -> y``."""
        x_sym = ca.MX.sym("x", *self.state_shape)
        return ca.Function("F_out_nn", [x_sym], [self.output(x_sym)])
