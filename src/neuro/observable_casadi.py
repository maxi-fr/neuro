from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import casadi as ca
import numpy as np

from neuro.nn_predictor_casadi import mlp_forward_ca
from neuro.observable import control_means
from neuro.transforms import unzscore, zscore

if TYPE_CHECKING:
    from neuro.config import ObservableGeometry
    from neuro.observable import ObservableArtifact
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
    artifact : ObservableArtifact
        The loaded predictor, its Observable geometry, native dt, and scalers.
    """

    artifact: ObservableArtifact

    @property
    def state_shape(self) -> tuple[int, int]:
        """Shape of the flattened history state column vector."""
        art = self.artifact
        return (art.n_y * art.n_channels + art.n_u * art.n_controls, 1)

    @property
    def fs(self) -> float:
        """Sampling frequency the Frame grid is resolved at."""
        return self.artifact.fs

    @property
    def n_controls(self) -> int:
        """Number of control input channels."""
        return self.artifact.n_controls

    @property
    def n_channels(self) -> int:
        """Number of EEG channels the Observable is resolved per."""
        return self.artifact.n_channels

    @property
    def n_values(self) -> int:
        """Scored values a Frame carries per channel."""
        return self.artifact.n_values

    @property
    def native_horizon(self) -> int:
        """Control Horizon in samples the underlying artifact was fit at."""
        return self.artifact.horizon

    @property
    def geometry(self) -> ObservableGeometry:
        """The Observable grid the artifact was trained against."""
        return self.artifact.geometry

    def n_frames(self, horizon: int) -> int:
        """Frames the recursion emits over ``horizon`` samples."""
        return self.artifact.n_frames(horizon)

    def initial_state(self) -> FloatArray:
        """Return initial state with NaN-padded EEG history and zero-padded control history."""
        art = self.artifact
        y_buf = np.full(art.n_y * art.n_channels, np.nan, dtype=np.float64)
        u_buf = np.zeros(art.n_u * art.n_controls, dtype=np.float64)
        return np.concatenate([y_buf, u_buf])

    def absorb(self, state: FloatArray, y: FloatArray, u: FloatArray) -> FloatArray:
        """Absorb raw measurement y and control u into the shift-register state."""
        art = self.artifact
        n_ch, n_ctrl = art.n_channels, art.n_controls
        split = art.n_y * n_ch

        y_buf = state[:split].reshape(art.n_y, n_ch)
        u_buf = state[split:].reshape(art.n_u, n_ctrl)

        z = art.encode(np.asarray(y, dtype=np.float64).reshape(-1))
        new_y_buf = np.vstack([y_buf[1:], z.reshape(1, n_ch)])
        new_u_buf = np.vstack([u_buf[1:], np.asarray(u, dtype=np.float64).reshape(1, n_ctrl)])

        return np.concatenate([new_y_buf.reshape(-1), new_u_buf.reshape(-1)])

    def is_ready(self, state: FloatArray) -> bool:
        """Return True if the EEG history buffer has absorbed at least n_y samples."""
        art = self.artifact
        return not np.isnan(state[: art.n_y * art.n_channels]).any()

    def forecast(self, x0: ca.SX | ca.MX, u_seq: ca.SX | ca.MX) -> ca.SX | ca.MX:
        """Forecast the log-Observable symbolically: ``(state, (horizon, m) controls) -> (C * F, M)``.

        Columns are Frames and rows are the ``(channel, value)`` grid flattened channel-major, in
        raw log units. Causality is structural: Frame ``m`` consumes only the Frame means that
        landed before its Segment ended, because a recursion cannot see forward.
        """
        art = self.artifact
        horizon = u_seq.shape[0]
        n_z = art.n_y * art.n_channels

        u_past_std = zscore(
            x0[n_z:],
            _tiled(art.u_std.center, art.n_controls, art.n_u),
            _tiled(art.u_std.scale, art.n_controls, art.n_u),
        )
        z = mlp_forward_ca(ca.vertcat(x0[:n_z], u_past_std), art.lift, art.activation)

        u_bar = ca.mtimes(ca.MX(control_means(self.geometry, horizon, art.fs)), u_seq)
        u_center = np.broadcast_to(art.u_std.center, (art.n_controls,)).reshape(-1, 1)
        u_scale = np.broadcast_to(art.u_std.scale, (art.n_controls,)).reshape(-1, 1)
        l_center = art.l_std.center.reshape(-1, 1)
        l_scale = art.l_std.scale.reshape(-1, 1)
        readout_w, readout_b = art.readout

        frames = []
        for m in range(u_bar.shape[0]):
            u_m = zscore(u_bar[m, :].T, u_center, u_scale)
            z = mlp_forward_ca(ca.vertcat(z, u_m), art.transition, art.activation)
            l_std_m = ca.mtimes(readout_w, z) + readout_b.reshape(-1, 1)
            frames.append(unzscore(l_std_m, l_center, l_scale))
        return ca.horzcat(*frames)

    @cached_property
    def f_forecast(self) -> ca.Function:
        """Reusable compiled forecast ``(x0, u_seq) -> l_hat`` at the artifact's native horizon."""
        x_sym = ca.MX.sym("x", *self.state_shape)
        u_sym = ca.MX.sym("u", self.native_horizon, self.n_controls)
        return ca.Function("F_observable", [x_sym, u_sym], [self.forecast(x_sym, u_sym)])
