from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Self

import casadi as ca
import numpy as np

from neuro.checkpoint import ObservableCheckpoint, load_observable
from neuro.nn_predictor_casadi import mlp_forward_ca
from neuro.observable import control_means
from neuro.transforms import unzscore, zscore

if TYPE_CHECKING:
    from pathlib import Path

    from neuro.config import ObservableGeometry
    from neuro.types import FloatArray


def _tiled(values: FloatArray, width: int, reps: int) -> FloatArray:
    """Broadcast a per-channel standardizer parameter into a tiled column vector."""
    return np.tile(np.broadcast_to(values, (width,)), reps).reshape(-1, 1)


@dataclass(frozen=True)
class ObservableSymbolicModel:
    """CasADi-symbolic equivalent of the observable-space predictor: one shot over the Control Horizon.

    Deliberately carries no ``f_step`` / ``f_out``: there is no per-sample state to step, so the MPC
    branches on which model it received rather than guarding an optional member.

    Attributes
    ----------
    checkpoint : ObservableCheckpoint
        The torch-free weights and metadata the bridge is rebuilt from.
    """

    checkpoint: ObservableCheckpoint

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> Self:
        """Build the model by loading a single-``.npz`` checkpoint from disk, torch-free."""
        return cls(load_observable(path))

    @property
    def state_shape(self) -> tuple[int, int]:
        """Shape of the flattened history state column vector."""
        ckpt = self.checkpoint
        return (ckpt.n_y * ckpt.n_channels + ckpt.n_u * ckpt.n_controls, 1)

    @property
    def fs(self) -> float:
        """Sampling frequency the Frame grid is resolved at."""
        return self.checkpoint.fs

    @property
    def n_controls(self) -> int:
        """Number of control input channels."""
        return self.checkpoint.n_controls

    @property
    def n_channels(self) -> int:
        """Number of EEG channels the Observable is resolved per."""
        return self.checkpoint.n_channels

    @property
    def n_values(self) -> int:
        """Scored values a Frame carries per channel."""
        return self.checkpoint.n_values

    @property
    def native_horizon(self) -> int:
        """Control Horizon in samples the underlying checkpoint was fit at."""
        return self.checkpoint.horizon

    @property
    def geometry(self) -> ObservableGeometry:
        """The Observable grid the checkpoint was trained against."""
        return self.checkpoint.geometry

    def n_frames(self, horizon: int) -> int:
        """Frames the recursion emits over ``horizon`` samples."""
        return self.checkpoint.n_frames(horizon)

    def initial_state(self) -> FloatArray:
        """Return initial state with NaN-padded EEG history and zero-padded control history."""
        ckpt = self.checkpoint
        y_buf = np.full(ckpt.n_y * ckpt.n_channels, np.nan, dtype=np.float64)
        u_buf = np.zeros(ckpt.n_u * ckpt.n_controls, dtype=np.float64)
        return np.concatenate([y_buf, u_buf])

    def absorb(self, state: FloatArray, y: FloatArray, u: FloatArray) -> FloatArray:
        """Absorb raw measurement y and control u into the shift-register state."""
        ckpt = self.checkpoint
        n_ch, n_ctrl = ckpt.n_channels, ckpt.n_controls
        split = ckpt.n_y * n_ch

        y_buf = state[:split].reshape(ckpt.n_y, n_ch)
        u_buf = state[split:].reshape(ckpt.n_u, n_ctrl)

        z = ckpt.y_std.transform(np.asarray(y, dtype=np.float64).reshape(-1))
        new_y_buf = np.vstack([y_buf[1:], z.reshape(1, n_ch)])
        new_u_buf = np.vstack([u_buf[1:], np.asarray(u, dtype=np.float64).reshape(1, n_ctrl)])

        return np.concatenate([new_y_buf.reshape(-1), new_u_buf.reshape(-1)])

    def is_ready(self, state: FloatArray) -> bool:
        """Return True if the EEG history buffer has absorbed at least n_y samples."""
        ckpt = self.checkpoint
        return not np.isnan(state[: ckpt.n_y * ckpt.n_channels]).any()

    def forecast(self, x0: ca.SX | ca.MX, u_seq: ca.SX | ca.MX) -> ca.SX | ca.MX:
        """Forecast the log-Observable symbolically: ``(state, (horizon, m) controls) -> (C * F, M)``.

        Columns are Frames and rows are the ``(channel, value)`` grid flattened channel-major, in
        raw log units. Causality is structural: Frame ``m`` consumes only the Frame means that
        landed before its Segment ended, because a recursion cannot see forward.
        """
        ckpt = self.checkpoint
        horizon = u_seq.shape[0]
        n_z = ckpt.n_y * ckpt.n_channels

        u_past_std = zscore(
            x0[n_z:],
            _tiled(ckpt.u_std.center, ckpt.n_controls, ckpt.n_u),
            _tiled(ckpt.u_std.scale, ckpt.n_controls, ckpt.n_u),
        )
        z = mlp_forward_ca(ca.vertcat(x0[:n_z], u_past_std), ckpt.lift, ckpt.activation)

        u_bar = ca.mtimes(ca.MX(control_means(ckpt.geometry, horizon, ckpt.fs)), u_seq)
        u_center = np.broadcast_to(ckpt.u_std.center, (ckpt.n_controls,)).reshape(-1, 1)
        u_scale = np.broadcast_to(ckpt.u_std.scale, (ckpt.n_controls,)).reshape(-1, 1)
        l_center = ckpt.l_std.center.reshape(-1, 1)
        l_scale = ckpt.l_std.scale.reshape(-1, 1)
        readout_w, readout_b = ckpt.readout

        frames = []
        for m in range(u_bar.shape[0]):
            u_m = zscore(u_bar[m, :].T, u_center, u_scale)
            z = mlp_forward_ca(ca.vertcat(z, u_m), ckpt.transition, ckpt.activation)
            l_std_m = ca.mtimes(readout_w, z) + readout_b.reshape(-1, 1)
            frames.append(unzscore(l_std_m, l_center, l_scale))
        return ca.horzcat(*frames)

    @cached_property
    def f_forecast(self) -> ca.Function:
        """Reusable compiled forecast ``(x0, u_seq) -> l_hat`` at the checkpoint's native horizon."""
        x_sym = ca.MX.sym("x", *self.state_shape)
        u_sym = ca.MX.sym("u", self.native_horizon, self.n_controls)
        return ca.Function("F_observable", [x_sym, u_sym], [self.forecast(x_sym, u_sym)])
