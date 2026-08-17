from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Self

import casadi as ca
import numpy as np

from neuro.predictor.artifact import MLPArtifact
from neuro.transforms import unzscore, zscore

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from neuro.predictor.artifact import Activation
    from neuro.types import FloatArray


def _mlp_forward_ca(
    x: ca.SX | ca.MX, layers: Sequence[tuple[FloatArray, FloatArray]], activation: Activation
) -> ca.SX | ca.MX:
    """Evaluate an MLP forward pass symbolically, replicating :meth:`MLPArtifact.forward_1step`.

    The activation is applied after every layer except the last; the last layer is affine. The
    name is validated once, in :meth:`MLPArtifact.load`, so the softplus branch is the fallthrough.

    Parameters
    ----------
    x : ca.SX | ca.MX
        Input column vector, shape ``(in_size, 1)``.
    layers : Sequence[(FloatArray, FloatArray)]
        Per-layer ``(weight, bias)`` pairs, ``weight`` shaped ``(out, in)``, in forward-pass order.
    activation : Activation
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
        else:
            # log(1 + exp(z)) overflows to inf for z > 709 and hands IPOPT a NaN gradient; this is
            # the shifted form np.logaddexp uses, exact for the same inputs and finite for all z.
            m = ca.fmax(z, 0.0)
            x = m + ca.log(ca.exp(-m) + ca.exp(z - m))
    w_last, b_last = layers[-1]
    return ca.mtimes(w_last, x) + b_last.reshape(-1, 1)


@dataclass(frozen=True)
class NNSymbolicModel:
    """CasADi-symbolic, single-step equivalent of the autoregressive MLP predictor.

    Attributes
    ----------
    artifact : MLPArtifact
        The loaded predictor, native dt, and scalers.
    """

    artifact: MLPArtifact

    @classmethod
    def from_artifact(cls, artifact: str | Path) -> Self:
        """Build the model by loading a single-``.npz`` artifact from disk."""
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
        """Number of EEG channels carried in the state per step."""
        return self.artifact.n_channels

    @property
    def native_horizon(self) -> int:
        """Native prediction horizon of the underlying artifact."""
        return self.artifact.horizon

    @property
    def is_linear(self) -> bool:
        """Whether the underlying MLP is linear (0 hidden layers)."""
        return self.artifact.is_linear

    @property
    def free_syms(self) -> dict[str, ca.MX]:
        """Free symbolic parameters; always empty -- this model is purely numeric."""
        return {}

    def initial_state(self) -> FloatArray:
        """Return initial state with NaN-padded EEG history and zero-padded control history."""
        n_y, n_ch = self.artifact.n_y, self.artifact.n_channels
        n_u, n_ctrl = self.artifact.n_u, self.artifact.n_controls
        y_buf = np.full(n_y * n_ch, np.nan, dtype=np.float64)
        u_buf = np.zeros(n_u * n_ctrl, dtype=np.float64)
        return np.concatenate([y_buf, u_buf])

    def absorb(self, state: FloatArray, y: FloatArray, u: FloatArray) -> FloatArray:
        """Absorb raw measurement y and control u into the shift-register state."""
        n_y, n_ch = self.artifact.n_y, self.artifact.n_channels
        n_u, n_ctrl = self.artifact.n_u, self.artifact.n_controls
        split = n_y * n_ch

        y_buf = state[:split].reshape(n_y, n_ch)
        u_buf = state[split:].reshape(n_u, n_ctrl)

        z = self.artifact.encode(np.asarray(y, dtype=np.float64).reshape(-1))
        new_y_buf = np.vstack([y_buf[1:], z.reshape(1, n_ch)])
        new_u_buf = np.vstack([u_buf[1:], np.asarray(u, dtype=np.float64).reshape(1, n_ctrl)])

        return np.concatenate([new_y_buf.reshape(-1), new_u_buf.reshape(-1)])

    def is_ready(self, state: FloatArray) -> bool:
        """Return True if the EEG history buffer has absorbed at least n_y samples."""
        n_y, n_ch = self.artifact.n_y, self.artifact.n_channels
        return not np.isnan(state[: n_y * n_ch]).any()

    def predict_output(self, y_flat: ca.SX | ca.MX, new_u_flat: ca.SX | ca.MX) -> ca.SX | ca.MX:
        """Predict the next model-space EEG sample.

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
        u_std = self.artifact.u_std

        u_mean_tiled = np.tile(np.broadcast_to(u_std.center, (n_ctrl,)), n_u).reshape(-1, 1)
        u_scale_tiled = np.tile(np.broadcast_to(u_std.scale, (n_ctrl,)), n_u).reshape(-1, 1)

        new_u_scaled_flat = zscore(new_u_flat, u_mean_tiled, u_scale_tiled)  # ty:ignore[invalid-argument-type]
        mlp_in = ca.vertcat(y_flat, new_u_scaled_flat)
        return _mlp_forward_ca(mlp_in, self.artifact.layers, self.artifact.activation)

    def step(self, history: Sequence[ca.SX | ca.MX], u: ca.SX | ca.MX) -> ca.SX | ca.MX:
        """Advance one step: ``history == [x]`` (model space), ``u`` raw control -> ``x_next``."""
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
        """Slice the most-recent model-space sample from the state and decode to raw EEG."""
        n_y, n_ch = self.artifact.n_y, self.artifact.n_channels
        z_last = x[(n_y - 1) * n_ch : n_y * n_ch]
        std = self.artifact.y_std
        center = np.broadcast_to(std.center, (n_ch,)).reshape(-1, 1)
        scale = np.broadcast_to(std.scale, (n_ch,)).reshape(-1, 1)
        return unzscore(z_last, center, scale)

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
