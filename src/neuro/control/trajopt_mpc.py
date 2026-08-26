from __future__ import annotations

import dataclasses
import importlib
from typing import TYPE_CHECKING, Any, Protocol, Self, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from simulate.controller import Controller
from trajopt.cones import ZeroCone
from trajopt.constraints.bounds import ControlBound
from trajopt.constraints.constraint_list import ConstraintList
from trajopt.constraints.linear import GoalConstraint, LinearConstraint
from trajopt.costs.objective import Objective
from trajopt.costs.quadratic import DiagonalCost
from trajopt.dynamics.base import DiscreteDynamics
from trajopt.problem import MPCState, Problem
from trajopt.transcription.ipopt import Ipopt
from trajopt.transcription.single_shooting import SingleShooting

from neuro.checkpoint import ESNCheckpoint, MLPCheckpoint, ObservableCheckpoint, load_esn, load_mlp, load_observable
from neuro.control.trajopt_costs import (
    ESNAutoRegressiveCost,
    ExcludeInitialKnotState,
    L1ControlCost,
    ObservableRolloutHinge,
    SpectralHingeCost,
    SumCost,
    has_whole_horizon_cost,
)
from neuro.observable import load_log_reference
from neuro.spectral import PsdEnvelope

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import ArrayLike
    from trajopt.costs.base import CostFunction
    from trajopt.transcription.result import Solver

    from neuro.config import ObservableGeometry
    from neuro.types import Activation, FloatArray


@runtime_checkable
class PredictorModel(Protocol):
    """A trajopt model that also exposes the Predictor's raw-units priming seam.

    The controller absorbs measurements into the model's opaque state (``absorb``), holds off
    until the state is primed (``is_ready``) and seeds its ``MPCState`` from the unprimed state
    (``initial_state``) -- the same seam the incumbent MPC used, hosted on the trajopt model
    adapter instead of a symbolic bridge. ``m`` and ``n_channels`` are the control and
    EEG channel counts the controller reads off the model directly.
    """

    m: int
    n_channels: int

    def absorb(self, state: FloatArray, y: FloatArray, u: FloatArray) -> FloatArray:
        """Append raw measurement ``y`` and applied control ``u`` into the model's opaque state."""
        ...

    def is_ready(self, state: FloatArray) -> bool:
        """Report whether the state has absorbed enough history to begin predicting."""
        ...

    def initial_state(self) -> FloatArray:
        """Return the unprimed state."""
        ...


def _build_problem(spec: dict[str, Any] | Problem) -> tuple[Problem, MPCState | None]:
    """Instantiate a Problem from an instance or a ``{class_path, ...}`` dict, plus an optional MPCState."""
    if isinstance(spec, Problem):
        return spec, None
    cfg = spec.copy()
    class_path: str = cfg.pop("class_path")
    module_name, func_name = class_path.rsplit(".", 1)
    target = getattr(importlib.import_module(module_name), func_name)
    res = target(**cfg) if cfg else target()
    if isinstance(res, tuple):
        state = res[1] if len(res) > 1 and isinstance(res[1], MPCState) else None
        return res[0], state
    return res, None


def _apply_activation(activation: Activation, z: jax.Array) -> jax.Array:
    """Apply the model's activation elementwise."""
    if activation == "relu":
        return jnp.maximum(z, 0.0)
    if activation == "tanh":
        return jnp.tanh(z)
    return jnp.logaddexp(z, 0.0)


class WaveformMLPModel(DiscreteDynamics):
    """trajopt ``DiscreteDynamics`` adapter for the waveform MLP predictor checkpoint.

    Holds the checkpoint's float64 weights and standardizer buffers as Equinox arrays, so the
    model rolls one MLP ``step`` per call with no torch in the loop. ``discrete_dynamics``
    reproduces the incumbent CasADi ``NNSymbolicModel.step`` state machine exactly: the newest
    control enters the control window *before* the prediction, so the predicted sample depends
    on the control applied at that step.
    """

    n_y: int = eqx.field(static=True)
    n_u: int = eqx.field(static=True)
    n_channels: int = eqx.field(static=True)
    n_controls: int = eqx.field(static=True)
    activation: Activation = eqx.field(static=True)
    residual: bool = eqx.field(static=True)
    y_center: jax.Array
    y_scale: jax.Array
    u_center: jax.Array
    u_scale: jax.Array
    weights: tuple[jax.Array, ...]
    biases: tuple[jax.Array, ...]

    def __init__(self, checkpoint: MLPCheckpoint) -> None:
        """Copy the checkpoint's float64 buffers into Equinox arrays.

        Parameters
        ----------
        checkpoint
            The torch-free MLP checkpoint whose weights and standardizers become this model.
        """
        super().__init__(
            n=checkpoint.n_y * checkpoint.n_channels + checkpoint.n_u * checkpoint.n_controls,
            m=checkpoint.n_controls,
            ne=checkpoint.n_y * checkpoint.n_channels + checkpoint.n_u * checkpoint.n_controls,
        )
        self.n_y = checkpoint.n_y
        self.n_u = checkpoint.n_u
        self.n_channels = checkpoint.n_channels
        self.n_controls = checkpoint.n_controls
        self.activation = checkpoint.activation
        self.residual = checkpoint.residual
        self.y_center = jnp.asarray(checkpoint.y_std.center)
        self.y_scale = jnp.asarray(checkpoint.y_std.scale)
        self.u_center = jnp.asarray(checkpoint.u_std.center)
        self.u_scale = jnp.asarray(checkpoint.u_std.scale)
        self.weights = tuple(jnp.asarray(weight) for weight, _ in checkpoint.layers)
        self.biases = tuple(jnp.asarray(bias) for _, bias in checkpoint.layers)

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> Self:
        """Rebuild the model from a numpy-readable MLP checkpoint on disk (a suffix-less stem)."""
        return cls(load_mlp(path))

    def _activate(self, z: jax.Array) -> jax.Array:
        """Apply the model's activation elementwise."""
        return _apply_activation(self.activation, z)

    def _predict(self, y_window: jax.Array, u_window: jax.Array) -> jax.Array:
        """One MLP forward pass on standardized windows -> the next standardized sample ``(n_channels,)``.

        With the residual skip the MLP output adds the window's last sample, so the layers fit
        the one-step delta exactly as the torch module does.
        """
        z = jnp.concatenate([y_window.reshape(-1), u_window.reshape(-1)])
        for i, weight in enumerate(self.weights[:-1]):
            z = self._activate(z @ weight.T + self.biases[i])
        weight_last, bias_last = self.weights[-1], self.biases[-1]
        z = z @ weight_last.T + bias_last
        if self.residual:
            z = z + y_window[-1]
        return z

    def output(self, x: jax.Array) -> jax.Array:
        """Decode the newest standardized y-window row into the raw predicted sample ``(n_channels,)``.

        The prediction the state carries is always its newest row, so the raw sample is a state
        component -- the decode the spectral hinge reads off any sample-grid model.
        """
        n_z = self.n_y * self.n_channels
        z_last = x[n_z - self.n_channels : n_z]
        return z_last * self.y_scale + self.y_center

    def discrete_dynamics(
        self,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array,
        dt: float | jax.Array,
    ) -> jax.Array:
        """Advance one sample: shift ``u`` into the control window, predict, and shift both windows.

        Mirrors the incumbent MPC's ``NNSymbolicModel.f_step``: the predicted ``y_{t+1}`` is the
        MLP output on the y-window ending at ``t`` and the u-window ending at ``t + 1`` after
        ``u`` is shifted in, and the returned state's control window ends with ``u``.
        """
        del t, dt
        n_z = self.n_y * self.n_channels
        y_window = x[:n_z].reshape(self.n_y, self.n_channels)
        u_window_raw = x[n_z:].reshape(self.n_u, self.n_controls)
        u_window = jnp.concatenate([u_window_raw[1:], u.reshape(1, -1)], axis=0)
        z_next = self._predict(y_window, (u_window - self.u_center) / self.u_scale)
        y_window = jnp.concatenate([y_window[1:], z_next[None, :]], axis=0)
        return jnp.concatenate([y_window.reshape(-1), u_window.reshape(-1)])

    def absorb(self, state: FloatArray, y: FloatArray, u: FloatArray) -> FloatArray:
        """Append raw measurement ``y`` and applied control ``u`` into the shift-register state."""
        n_z = self.n_y * self.n_channels
        state_arr = np.asarray(state, dtype=np.float64)
        y_window = state_arr[:n_z].reshape(self.n_y, self.n_channels)
        u_window = state_arr[n_z:].reshape(self.n_u, self.n_controls)
        z = (np.asarray(y, dtype=np.float64).reshape(-1) - np.asarray(self.y_center)) / np.asarray(self.y_scale)
        y_window = np.concatenate([y_window[1:], z[None, :]], axis=0)
        u_window = np.concatenate([u_window[1:], np.asarray(u, dtype=np.float64).reshape(1, -1)], axis=0)
        return np.concatenate([y_window.reshape(-1), u_window.reshape(-1)])

    def is_ready(self, state: FloatArray) -> bool:
        """Report whether the EEG window holds no NaN, i.e. at least ``n_y`` samples were absorbed."""
        n_z = self.n_y * self.n_channels
        return not bool(np.isnan(np.asarray(state, dtype=np.float64)[:n_z]).any())

    def initial_state(self) -> FloatArray:
        """NaN-padded EEG window and zero-padded control window: nothing absorbed yet."""
        y_buf = np.full(self.n_y * self.n_channels, np.nan, dtype=np.float64)
        u_buf = np.zeros(self.n_u * self.n_controls, dtype=np.float64)
        return np.concatenate([y_buf, u_buf])


class ESNModel(DiscreteDynamics):
    """trajopt ``DiscreteDynamics`` adapter for the ESN predictor checkpoint.

    The opaque state is the reservoir vector followed by the absorbed-step counter, exactly the
    ``ESNModule`` state; ``discrete_dynamics`` advances one free-running sample -- the readout of
    the current reservoir replaces the measurement in the input, matching the torch module's
    ``step``/``rollout`` -- and ``output`` decodes that readout into the raw predicted sample.
    The checkpoint's float64 weights and standardizer buffers become Equinox arrays, with the
    sparse reservoir densified for JAX.
    """

    reservoir_size: int = eqx.field(static=True)
    n_channels: int = eqx.field(static=True)
    n_controls: int = eqx.field(static=True)
    priming_steps: int = eqx.field(static=True)
    leak_rate: float = eqx.field(static=True)
    y_center: jax.Array
    y_scale: jax.Array
    u_center: jax.Array
    u_scale: jax.Array
    w_res: jax.Array
    w_in: jax.Array
    w_out: jax.Array

    def __init__(self, checkpoint: ESNCheckpoint) -> None:
        """Copy the checkpoint's float64 buffers into Equinox arrays, densifying the reservoir.

        Parameters
        ----------
        checkpoint
            The torch-free ESN checkpoint whose weights and standardizers become this model.
        """
        super().__init__(
            n=checkpoint.reservoir_size + 1,
            m=checkpoint.n_controls,
            ne=checkpoint.reservoir_size + 1,
        )
        self.reservoir_size = int(checkpoint.reservoir_size)
        self.n_channels = int(checkpoint.n_channels)
        self.n_controls = int(checkpoint.n_controls)
        self.priming_steps = int(checkpoint.priming_steps)
        self.leak_rate = float(checkpoint.leak_rate)
        self.y_center = jnp.asarray(checkpoint.y_std.center)
        self.y_scale = jnp.asarray(checkpoint.y_std.scale)
        self.u_center = jnp.asarray(checkpoint.u_std.center)
        self.u_scale = jnp.asarray(checkpoint.u_std.scale)
        self.w_res = jnp.asarray(checkpoint.w_res.toarray())
        self.w_in = jnp.asarray(checkpoint.w_in)
        self.w_out = jnp.asarray(checkpoint.w_out)

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> Self:
        """Rebuild the model from a numpy-readable ESN checkpoint on disk (a suffix-less stem)."""
        return cls(load_esn(path))

    def _readout(self, h: jax.Array) -> jax.Array:
        """Standardized one-step-ahead readout ``(n_channels,)`` from the reservoir ``h``."""
        return h @ self.w_out[:, : self.reservoir_size].T + self.w_out[:, self.reservoir_size]

    def _absorb_reservoir(self, h: jax.Array, z: jax.Array, v: jax.Array) -> jax.Array:
        """Advance the reservoir one step under the standardized input ``(z, v)``."""
        x_in = jnp.concatenate([z, v, jnp.ones(1)])
        alpha = self.leak_rate
        return (1.0 - alpha) * h + alpha * jnp.tanh(h @ self.w_res.T + x_in @ self.w_in.T)

    def output(self, x: jax.Array) -> jax.Array:
        """Decode the readout of the reservoir part of the state into the raw sample ``(n_channels,)``."""
        return self._readout(x[: self.reservoir_size]) * self.y_scale + self.y_center

    def discrete_dynamics(
        self,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array,
        dt: float | jax.Array,
    ) -> jax.Array:
        """Advance one free-running sample: readout the current reservoir, then absorb it with ``u``.

        Matches the torch module's ``step``: the readout of the pre-step reservoir is the emitted
        prediction, and the same standardized readout feeds the update. The step counter advances.
        """
        del t, dt
        h = x[: self.reservoir_size]
        v = (u - self.u_center) / self.u_scale
        h_next = self._absorb_reservoir(h, self._readout(h), v)
        return jnp.concatenate([h_next, x[self.reservoir_size :] + 1.0])

    def absorb(self, state: FloatArray, y: FloatArray, u: FloatArray) -> FloatArray:
        """Append raw measurement ``y`` and applied control ``u``, advancing the step counter."""
        state_arr = np.asarray(state, dtype=np.float64)
        z = jnp.asarray(
            (np.asarray(y, dtype=np.float64).reshape(-1) - np.asarray(self.y_center)) / np.asarray(self.y_scale)
        )
        v = jnp.asarray(
            (np.asarray(u, dtype=np.float64).reshape(-1) - np.asarray(self.u_center)) / np.asarray(self.u_scale)
        )
        h = jnp.asarray(state_arr[: self.reservoir_size])
        h_next = np.asarray(self._absorb_reservoir(h, z, v), dtype=np.float64)
        return np.concatenate([h_next, state_arr[self.reservoir_size :] + 1.0])

    def is_ready(self, state: FloatArray) -> bool:
        """Report whether the state has absorbed at least ``priming_steps`` samples."""
        return np.asarray(state, dtype=np.float64)[self.reservoir_size] >= self.priming_steps

    def initial_state(self) -> FloatArray:
        """Zero reservoir and zero absorbed steps: nothing absorbed yet."""
        return np.zeros(self.reservoir_size + 1, dtype=np.float64)


class ObservableModel(DiscreteDynamics):
    """trajopt ``DiscreteDynamics`` adapter for the one-Frame-per-step Observable predictor.

    The opaque state is the history register -- ``[standardized EEG window | raw control
    window]`` -- followed by the lifted Frame state (and the residual carry block, when the
    checkpoint's skip is on), exactly the ``StepwiseObservableMLP`` state. One step is one
    Frame: ``discrete_dynamics`` runs the shared transition under the Frame-mean control ``u``
    and carries the register unchanged, matching the torch module's ``step``; the lift
    (register -> Frame state) happens once at ``absorb``/``initial_state``. The controller
    therefore drives this adapter on the Frame grid, and the problem's decision variables are
    Frame-mean controls rather than per-sample ones.
    """

    n_y: int = eqx.field(static=True)
    n_u: int = eqx.field(static=True)
    n_channels: int = eqx.field(static=True)
    n_controls: int = eqx.field(static=True)
    z_dim: int = eqx.field(static=True)
    n_values: int = eqx.field(static=True)
    fs: float = eqx.field(static=True)
    activation: Activation = eqx.field(static=True)
    residual: bool = eqx.field(static=True)
    geometry: ObservableGeometry = eqx.field(static=True)
    lift_weights: tuple[jax.Array, ...]
    lift_biases: tuple[jax.Array, ...]
    transition_weights: tuple[jax.Array, ...]
    transition_biases: tuple[jax.Array, ...]
    readout_w: jax.Array
    readout_b: jax.Array
    y_center: jax.Array
    y_scale: jax.Array
    u_center: jax.Array
    u_scale: jax.Array
    l_center: jax.Array
    l_scale: jax.Array

    def __init__(self, checkpoint: ObservableCheckpoint) -> None:
        """Copy the checkpoint's float64 buffers into Equinox arrays.

        Parameters
        ----------
        checkpoint
            The torch-free Observable checkpoint whose weights and standardizers become this model.
        """
        n_hist = checkpoint.n_y * checkpoint.n_channels + checkpoint.n_u * checkpoint.n_controls
        carry_size = checkpoint.n_channels * checkpoint.n_values if checkpoint.residual else 0
        super().__init__(
            n=n_hist + checkpoint.z_dim + carry_size, m=checkpoint.n_controls, ne=n_hist + checkpoint.z_dim + carry_size
        )
        self.n_y = int(checkpoint.n_y)
        self.n_u = int(checkpoint.n_u)
        self.n_channels = int(checkpoint.n_channels)
        self.n_controls = int(checkpoint.n_controls)
        self.z_dim = int(checkpoint.z_dim)
        self.n_values = int(checkpoint.n_values)
        self.fs = float(checkpoint.fs)
        self.activation = checkpoint.activation
        self.residual = checkpoint.residual
        self.geometry = checkpoint.geometry
        self.lift_weights = tuple(jnp.asarray(weight) for weight, _ in checkpoint.lift)
        self.lift_biases = tuple(jnp.asarray(bias) for _, bias in checkpoint.lift)
        self.transition_weights = tuple(jnp.asarray(weight) for weight, _ in checkpoint.transition)
        self.transition_biases = tuple(jnp.asarray(bias) for _, bias in checkpoint.transition)
        self.readout_w, self.readout_b = (jnp.asarray(checkpoint.readout[0]), jnp.asarray(checkpoint.readout[1]))
        self.y_center = jnp.asarray(checkpoint.y_std.center)
        self.y_scale = jnp.asarray(checkpoint.y_std.scale)
        self.u_center = jnp.asarray(checkpoint.u_std.center)
        self.u_scale = jnp.asarray(checkpoint.u_std.scale)
        self.l_center = jnp.asarray(checkpoint.l_std.center)
        self.l_scale = jnp.asarray(checkpoint.l_std.scale)

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> Self:
        """Rebuild the model from a numpy-readable Observable checkpoint on disk (a stem)."""
        return cls(load_observable(path))

    @property
    def _n_hist(self) -> int:
        """Register width: the standardized EEG window plus the raw control window."""
        return self.n_y * self.n_channels + self.n_u * self.n_controls

    def n_frames(self, horizon: int) -> int:
        """Frames the recursion emits over ``horizon`` samples at the checkpoint's geometry."""
        return self.geometry.n_frames(horizon, self.fs)

    def _activate(self, z: jax.Array) -> jax.Array:
        """Apply the model's activation elementwise."""
        return _apply_activation(self.activation, z)

    def _mlp(self, weights: tuple[jax.Array, ...], biases: tuple[jax.Array, ...], z: jax.Array) -> jax.Array:
        """One MLP block forward pass; the activation follows every layer except the last."""
        for i, weight in enumerate(weights[:-1]):
            z = self._activate(z @ weight.T + biases[i])
        weight_last, bias_last = weights[-1], biases[-1]
        return z @ weight_last.T + bias_last

    def _lift(self, register: jax.Array) -> jax.Array:
        """Lift a history register to the Frame state via the checkpoint's lift block."""
        n_z = self.n_y * self.n_channels
        u_window = register[n_z:].reshape(self.n_u, self.n_controls)
        lift_in = jnp.concatenate([register[:n_z], ((u_window - self.u_center) / self.u_scale).reshape(-1)])
        return self._mlp(self.lift_weights, self.lift_biases, lift_in)

    def output(self, x: jax.Array) -> jax.Array:
        """Decode the Frame level into the raw log-Observable frame ``(n_channels * n_values,)``.

        With the residual skip the level is the accumulated carry block; otherwise it is the
        readout of the lifted Frame state.
        """
        if self.residual:
            l_std = x[self._n_hist + self.z_dim :]
        else:
            z = x[self._n_hist :]
            l_std = z @ self.readout_w.T + self.readout_b
        return l_std * self.l_scale + self.l_center

    def discrete_dynamics(
        self,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array,
        dt: float | jax.Array,
    ) -> jax.Array:
        """Advance one Frame: run the shared transition on the carried lifted state under Frame-mean ``u``.

        The register is carried unchanged -- only the lifted Frame state advances, matching the
        torch module's ``step``.
        """
        del t, dt
        z = x[self._n_hist : self._n_hist + self.z_dim]
        v = (u - self.u_center) / self.u_scale
        z_next = self._mlp(self.transition_weights, self.transition_biases, jnp.concatenate([z, v]))
        if self.residual:
            carry = x[self._n_hist + self.z_dim :] + (z_next @ self.readout_w.T + self.readout_b)
            return jnp.concatenate([x[: self._n_hist], z_next, carry])
        return jnp.concatenate([x[: self._n_hist], z_next])

    def absorb(self, state: FloatArray, y: FloatArray, u: FloatArray) -> FloatArray:
        """Append raw measurement ``y`` and applied control ``u`` to the register, re-lifting."""
        state_arr = np.asarray(state, dtype=np.float64)
        n_z = self.n_y * self.n_channels
        y_window = state_arr[:n_z].reshape(self.n_y, self.n_channels)
        u_window = state_arr[n_z : self._n_hist].reshape(self.n_u, self.n_controls)
        z_new = (np.asarray(y, dtype=np.float64).reshape(-1) - np.asarray(self.y_center)) / np.asarray(self.y_scale)
        y_window = np.concatenate([y_window[1:], z_new[None, :]], axis=0)
        u_window = np.concatenate([u_window[1:], np.asarray(u, dtype=np.float64).reshape(1, -1)], axis=0)
        register = np.concatenate([y_window.reshape(-1), u_window.reshape(-1)])
        z = np.asarray(self._lift(jnp.asarray(register, dtype=jnp.float64)), dtype=np.float64)
        state = np.concatenate([register, z])
        if self.residual:
            state = np.concatenate([state, np.zeros(self.n_channels * self.n_values, dtype=np.float64)])
        return state

    def is_ready(self, state: FloatArray) -> bool:
        """Report whether the EEG window holds no NaN, i.e. at least ``n_y`` samples were absorbed."""
        return not bool(np.isnan(np.asarray(state, dtype=np.float64)[: self.n_y * self.n_channels]).any())

    def initial_state(self) -> FloatArray:
        """NaN-padded EEG window, zero-padded control window, their (NaN) lifted Frame state, zero carry."""
        y_buf = np.full(self.n_y * self.n_channels, np.nan, dtype=np.float64)
        u_buf = np.zeros(self.n_u * self.n_controls, dtype=np.float64)
        register = np.concatenate([y_buf, u_buf])
        state = np.concatenate([register, self._lift(jnp.asarray(register, dtype=jnp.float64))])
        if self.residual:
            state = np.concatenate([state, np.zeros(self.n_channels * self.n_values, dtype=np.float64)])
        return state


@dataclasses.dataclass(frozen=True)
class TrajOptMPCLog:
    """Per-step diagnostics: the applied control, optimal cost, solver success, and warm-up flag."""

    u: FloatArray
    cost: float
    success: bool
    warmup: bool


def kirchhoff_constraint(n: int, m: int) -> LinearConstraint:
    """Kirchhoff current-law equality: the per-electrode currents sum to zero at each step.

    The incumbent NLP's ``_sum_to_zero`` equality, expressed with
    ``trajopt.constraints.linear``: ``A @ u - b = 0`` with ``A = ones(1, m)`` and ``b = 0``
    on the control block of ``z = [x; u]`` at every non-terminal knot.
    """
    return LinearConstraint(
        n=n,
        m=m,
        A=jnp.ones((1, m)),
        b=jnp.zeros(1),
        sense=ZeroCone(),
        inds=range(n, n + m),
    )


def _combine_costs(costs: list[CostFunction]) -> CostFunction:
    """Return the single stage cost, wrapping several sub-costs in a :class:`SumCost`."""
    return SumCost(costs) if len(costs) > 1 else costs[0]


def _spectral_envelope(psd_ref: str | Path | None, w_psd: float) -> PsdEnvelope | None:
    """Load the healthy PSD envelope when ``w_psd`` enables the spectral hinge, else ``None``."""
    if w_psd <= 0:
        return None
    if psd_ref is None:
        msg = "psd_ref must be provided when w_psd > 0"
        raise ValueError(msg)
    return PsdEnvelope.load(psd_ref)


def _assemble_problem(
    model: DiscreteDynamics,
    objective: Objective,
    *,
    N: int,
    u_max: ArrayLike,
    kirchhoff: bool,
) -> Problem:
    """Add the control box bounds (and optional Kirchhoff equality) and build the ``Problem``."""
    n, m = model.n, model.m
    u_max_arr = np.broadcast_to(np.atleast_1d(np.asarray(u_max, dtype=np.float64)), (m,))
    constraints = ConstraintList(n=n, m=m, N=N)
    constraints.add_constraint(ControlBound(n=n, m=m, u_min=-u_max_arr, u_max=u_max_arr), range(N - 1))
    if kirchhoff:
        constraints.add_constraint(kirchhoff_constraint(n, m), range(N - 1))
    return Problem(model=model, obj=objective, constraints=constraints, N=N)


def _native_expansion_only(solver: Solver) -> bool:
    """Whether ``solver`` is a native JAX backend that Taylor-expands ``evaluate`` per knot."""
    return type(solver).__module__.startswith("trajopt.solvers")


def _has_non_goal_knot_constraint(problem: Problem) -> bool:
    """Whether a knot constraint beyond box bounds remains, which single shooting rejects.

    Box bounds are hoisted into primal limits, so the knot evaluators hold only non-box
    constraints; any that is not a ``GoalConstraint`` (the Kirchhoff linear equality, say) is
    exactly what ``SingleShooting._validate_supported_constraints`` refuses at solve time.
    """
    return any(
        not isinstance(con, GoalConstraint)
        for evaluator in problem.constraints.knot_evaluators
        for con in evaluator.constraints
    )


def _default_solver(problem: Problem) -> Ipopt | SingleShooting:
    """Select the default solver: general Ipopt for non-box knots, single shooting otherwise.

    ``SingleShooting`` expresses only ``ControlBound`` and ``GoalConstraint``, so a problem
    carrying the Kirchhoff linear equality must use the general Ipopt transcription -- the
    incumbent's ``solver="ipopt"`` choice, reproduced here with its limited-memory Hessian.
    """
    ipopt = Ipopt(options={"print_level": 0, "hessian_approximation": "limited-memory"})
    if _has_non_goal_knot_constraint(problem):
        return ipopt
    return SingleShooting(solver=ipopt)


def ensure_solver_supports_constraints(problem: Problem, solver: Solver) -> None:
    """Raise when an injected solver cannot carry the problem's knot constraints.

    ``SingleShooting`` rejects any non-box knot constraint beyond a ``GoalConstraint`` at solve
    time; fail at construction instead, under the same criterion.
    """
    if isinstance(solver, SingleShooting) and _has_non_goal_knot_constraint(problem):
        msg = (
            "SingleShooting supports only ControlBound and GoalConstraint; this problem carries "
            "a knot constraint it cannot express. Use the general Ipopt transcription instead."
        )
        raise ValueError(msg)


def ensure_solver_supports_objective(problem: Problem, solver: Solver) -> None:
    """Raise when a native expansion-only solver would silently drop a whole-horizon cost."""
    if not _native_expansion_only(solver):
        return
    if has_whole_horizon_cost(problem.obj.stage_cost):
        msg = (
            f"{type(solver).__name__} expands costs per knot and cannot score the whole-horizon "
            "hinge cost; use a transcription solver (e.g. SingleShooting(Ipopt(...)))."
        )
        raise ValueError(msg)


def build_waveform_problem(  # noqa: PLR0913 -- checkpoint plus the nine MPC cost/bound knobs
    artifact: str | Path,
    *,
    horizon: int,
    u_max: ArrayLike,
    w_y: float = 1.0,
    w_u: float = 0.0,
    w_y_terminal: float | None = None,
    w_u_l1: float = 0.0,
    w_psd: float = 0.0,
    psd_ref: str | Path | None = None,
    kirchhoff: bool = False,
) -> Problem:
    """Assemble the waveform MPC problem: model adapter, objective, box and Kirchhoff bounds.

    The state cost encodes ``w_y * ||decode(z_last)||^2`` -- the incumbent's EEG-power term,
    a quadratic in the newest standardized y-window row ``z_last`` -- plus ``w_u * ||u||^2`` on
    every control; ``w_y_terminal`` (when given) replaces ``w_y`` on the final knot, exactly as
    the incumbent's horizon-mean stage cost does. All three are scaled by ``1 / horizon``,
    reproducing the incumbent's ``cost / horizon`` reduction so that the spectral hinge (which
    is not horizon-meaned) trades against them exactly as it does in the CasADi graph. The
    L1 sparsity penalty ``(w_u_l1 / horizon) * sum(|u|)`` and the spectral PSD hinge
    (``w_psd`` against ``psd_ref``'s healthy envelope) join the quadratic as custom
    ``CostFunction`` subclasses. The constraints are the control box bounds
    ``-u_max <= u <= u_max``, plus the Kirchhoff sum-to-zero equality when ``kirchhoff`` is
    set. The default single-shooting transcription rejects the controls-block linear equality,
    so the full constraint set is solved with the general Ipopt transcription.

    Parameters
    ----------
    artifact
        Suffix-less stem of the numpy-readable MLP checkpoint.
    horizon
        Control Horizon in steps; the trajopt horizon is ``horizon + 1`` knot points.
    u_max
        Per-electrode amplitude bound: a scalar shared by every electrode or a
        length-``n_controls`` vector.
    w_y
        Weight on predicted EEG power in the cost.
    w_u
        Weight on control effort (quadratic) in the cost.
    w_y_terminal
        Weight on predicted EEG power at the final horizon step, replacing ``w_y`` there;
        ``None`` (default) keeps ``w_y`` uniform over the horizon.
    w_u_l1
        Weight on the L1 norm of the control effort (a sparse-stimulation penalty); ``0``
        disables it (default), leaving the pure-quadratic problem unchanged.
    w_psd
        Weight on the spectral cost: the mean squared amount by which the predicted EEG
        spectrum exceeds ``psd_ref``'s healthy envelope, in log power. ``0`` (default)
        disables it.
    psd_ref
        Path to the healthy reference envelope npz written by ``scripts/build_healthy_psd.py``.
        Required when ``w_psd > 0``; its stored window geometry drives the cost.
    kirchhoff
        Add the Kirchhoff sum-to-zero equality on the controls. Off by default; the incumbent
        applies it unconditionally, so full parity sets it.
    """
    model = WaveformMLPModel.from_checkpoint(artifact)
    n, m = model.n, model.m
    N = horizon + 1

    z_last = slice((model.n_y - 1) * model.n_channels, model.n_y * model.n_channels)
    Q = jnp.zeros(n).at[z_last].set(2.0 * w_y * model.y_scale**2 / horizon)
    xf = jnp.zeros(n).at[z_last].set(-model.y_center / model.y_scale)
    stage = DiagonalCost.tracking(Q, jnp.full(m, 2.0 * w_u / horizon), xf, jnp.zeros(m))
    costs: list[CostFunction] = [ExcludeInitialKnotState(stage)]
    if w_u_l1 > 0:
        costs.append(L1ControlCost(n=n, m=m, w_l1=w_u_l1, horizon=horizon))
    envelope = _spectral_envelope(psd_ref, w_psd)
    if envelope is not None:
        costs.append(SpectralHingeCost(model=model, envelope=envelope, w_psd=w_psd, horizon=horizon))
    stage_cost: CostFunction = _combine_costs(costs)
    # The terminal knot carries the horizon's final output, weighted by ``w_y_terminal`` when
    # given else ``w_y`` -- the incumbent's last-step stage cost. Always explicit, because the
    # composite's derived terminal (``SumCost.as_terminal``) would otherwise carry the
    # control-only L1 and whole-horizon hinge into a knot that has no control.
    w_y_final = w_y_terminal if w_y_terminal is not None else w_y
    Q_f = jnp.zeros(n).at[z_last].set(2.0 * w_y_final * model.y_scale**2 / horizon)
    terminal = DiagonalCost.terminal_tracking(Q_f, xf, m)
    objective = Objective(stage_cost=stage_cost, terminal_cost=terminal, N=N)

    return _assemble_problem(model, objective, N=N, u_max=u_max, kirchhoff=kirchhoff)


def build_esn_problem(  # noqa: PLR0913 -- checkpoint plus the nine MPC cost/bound knobs
    artifact: str | Path,
    *,
    horizon: int,
    u_max: ArrayLike,
    w_y: float = 1.0,
    w_u: float = 0.0,
    w_y_terminal: float | None = None,
    w_u_l1: float = 0.0,
    w_psd: float = 0.0,
    psd_ref: str | Path | None = None,
    kirchhoff: bool = False,
) -> Problem:
    """Assemble the ESN MPC problem: model adapter, objective, box and Kirchhoff bounds.

    The same shape as :func:`build_waveform_problem` on the ESN: the EEG-power term reads the
    raw readout output off the reservoir state (a linear map of the state, not a component, so
    it cannot be a diagonal state weight), the smooth L1 surrogate and the spectral PSD hinge
    join the quadratic as custom ``CostFunction`` subclasses, and the constraints are the
    control box bounds plus the optional Kirchhoff equality.

    Parameters
    ----------
    artifact
        Suffix-less stem of the numpy-readable ESN checkpoint.
    horizon
        Control Horizon in steps; the trajopt horizon is ``horizon + 1`` knot points.
    u_max
        Per-electrode amplitude bound: a scalar shared by every electrode or a
        length-``n_controls`` vector.
    w_y
        Weight on predicted EEG power in the cost.
    w_u
        Weight on control effort (quadratic) in the cost.
    w_y_terminal
        Weight on predicted EEG power at the final horizon step, replacing ``w_y`` there;
        ``None`` (default) keeps ``w_y`` uniform over the horizon.
    w_u_l1
        Weight on the L1 norm of the control effort (a sparse-stimulation penalty); ``0``
        disables it (default), leaving the pure-quadratic problem unchanged.
    w_psd
        Weight on the spectral cost: the mean squared amount by which the predicted EEG
        spectrum exceeds ``psd_ref``'s healthy envelope, in log power. ``0`` (default)
        disables it.
    psd_ref
        Path to the healthy reference envelope npz written by ``scripts/build_healthy_psd.py``.
        Required when ``w_psd > 0``; its stored window geometry drives the cost.
    kirchhoff
        Add the Kirchhoff sum-to-zero equality on the controls. Off by default, matching the
        quadratic/box subset; the incumbent applies it unconditionally, so full parity sets it.
    """
    model = ESNModel.from_checkpoint(artifact)
    n, m = model.n, model.m
    N = horizon + 1

    costs: list[CostFunction] = [
        ExcludeInitialKnotState(ESNAutoRegressiveCost(model=model, w_y=w_y, w_u=w_u, horizon=horizon))
    ]
    if w_u_l1 > 0:
        costs.append(L1ControlCost(n=n, m=m, w_l1=w_u_l1, horizon=horizon))
    envelope = _spectral_envelope(psd_ref, w_psd)
    if envelope is not None:
        costs.append(SpectralHingeCost(model=model, envelope=envelope, w_psd=w_psd, horizon=horizon))
    stage_cost: CostFunction = _combine_costs(costs)
    # The terminal knot carries the horizon's final output, weighted by ``w_y_terminal`` when
    # given else ``w_y`` -- the incumbent's last-step stage cost, with no control term.
    w_y_final = w_y_terminal if w_y_terminal is not None else w_y
    terminal = ESNAutoRegressiveCost(model=model, w_y=w_y_final, w_u=0.0, horizon=horizon, terminal=True)
    objective = Objective(stage_cost=stage_cost, terminal_cost=terminal, N=N)

    return _assemble_problem(model, objective, N=N, u_max=u_max, kirchhoff=kirchhoff)


def build_observable_problem(  # noqa: PLR0913 -- checkpoint plus the nine MPC cost/bound knobs
    artifact: str | Path,
    *,
    horizon: int,
    u_max: ArrayLike,
    w_y: float = 0.0,
    w_u: float = 0.0,
    w_y_terminal: float | None = None,
    w_u_l1: float = 0.0,
    w_psd: float = 0.0,
    psd_ref: str | Path | None = None,
    kirchhoff: bool = False,
) -> Problem:
    """Assemble the observable MPC problem on the Frame grid.

    The Observable predictor steps one Frame per position, so the trajopt horizon is the Frame
    count ``n_frames(horizon)`` rather than the sample count and every decision variable is a
    Frame-mean control. ``w_y``/``w_y_terminal`` are rejected -- the Rollout produces no
    per-sample EEG outputs -- and ``w_psd > 0`` with a ``psd_ref`` is required, exactly the
    incumbent's observable branch. The log reference is reduced from the envelope onto the
    checkpoint's ``ObservableGeometry`` grid: the shared source of truth with the training-time
    Loss.

    Parameters
    ----------
    artifact
        Suffix-less stem of the numpy-readable Observable checkpoint.
    horizon
        Control Horizon in samples; the trajopt horizon is ``n_frames(horizon) + 1`` knots.
    u_max
        Per-electrode amplitude bound: a scalar shared by every electrode or a
        length-``n_controls`` vector, applied to the Frame-mean controls.
    w_y
        Weight on predicted EEG power; rejected on this path (it predicts the Observable
        directly and produces no per-sample outputs).
    w_u
        Weight on control effort (quadratic) in the cost.
    w_y_terminal
        Rejected on this path, alongside ``w_y``.
    w_u_l1
        Weight on the L1 norm of the Frame-mean controls; ``0`` (default) disables it.
    w_psd
        Weight on the Observable hinge: the mean squared amount by which the rolled-out Frames
        exceed ``psd_ref``'s healthy envelope, in log units. Required on this path.
    psd_ref
        Path to the healthy reference envelope npz; required when ``w_psd > 0`` (always here).
    kirchhoff
        Add the Kirchhoff sum-to-zero equality on the Frame-mean controls. Off by default.
    """
    model = ObservableModel.from_checkpoint(artifact)
    if w_y > 0 or w_y_terminal is not None:
        msg = (
            f"w_y ({w_y}) and w_y_terminal ({w_y_terminal}) have no meaning on the observable "
            "path: it predicts the Observable directly and produces no per-sample EEG outputs."
        )
        raise ValueError(msg)
    if w_psd <= 0 or psd_ref is None:
        msg = "the observable path requires w_psd > 0 and a psd_ref; it has no other output term."
        raise ValueError(msg)
    frames = model.n_frames(horizon)
    if frames < 1:
        msg = (
            f"horizon ({horizon}) holds no {model.geometry.kind} frame at the checkpoint's "
            "geometry; it must cover at least one Segment."
        )
        raise ValueError(msg)
    n, m = model.n, model.m
    N = frames + 1
    log_reference = load_log_reference(psd_ref, model.geometry, model.fs)

    costs: list[CostFunction] = [
        ObservableRolloutHinge(model=model, log_reference=log_reference, w_psd=w_psd, n_frames=frames),
        DiagonalCost(Q=jnp.zeros(n), R=jnp.full(m, 2.0 * w_u / frames)),
    ]
    if w_u_l1 > 0:
        costs.append(L1ControlCost(n=n, m=m, w_l1=w_u_l1, horizon=frames))
    stage_cost: CostFunction = _combine_costs(costs)
    # The terminal knot scores nothing of its own: the last Frame's hinge value already landed
    # in the final stage entry, and there is no control at the terminal.
    terminal = DiagonalCost.terminal_tracking(jnp.zeros(n), jnp.zeros(n), m)
    objective = Objective(stage_cost=stage_cost, terminal_cost=terminal, N=N)

    return _assemble_problem(model, objective, N=N, u_max=u_max, kirchhoff=kirchhoff)


class TrajOptMPCController(Controller[TrajOptMPCLog]):
    """Receding-horizon MPC for the waveform predictor, built directly on trajopt primitives.

    Owns the true absorbed predictor state and ``u_last`` as private instance attributes, and
    reproduces the incumbent MPC ``update`` loop shape -- ``absorb``, then
    ``with_measurement``, ``problem.solve``, extract the first control, ``shift`` -- without
    instantiating ``TrajOptMPC``. ``TrajOptMPC.update`` is the reference for how ``Problem``,
    ``MPCState`` and the solver compose, not a dependency.
    """

    def __init__(
        self,
        dt: float,
        problem: Problem,
        solver: Solver | None = None,
        initial_state: MPCState | None = None,
    ) -> None:
        """Initialize the controller and its persistent MPC state.

        Parameters
        ----------
        dt
            Controller update step in seconds; should equal the predictor's native dt.
        problem
            The trajopt optimal-control problem: model adapter + objective + constraint list.
        solver
            Solver backend (e.g. ``SingleShooting(Ipopt(...))``); overrides the default. When
            omitted, the default is the general Ipopt transcription if the constraint list
            carries a non-box knot constraint (the Kirchhoff linear equality), else
            single-shooting Ipopt -- both with printing off.
        initial_state
            Initial MPC warm-start state; defaults to one built from the unprimed model state,
            with the NaN padding of the unprimed EEG window replaced by zeros so the seed is
            finite for every solver (the multiple-shooting transcription starts from the full
            state trajectory, not just the controls).
        """
        super().__init__(dt)
        self.problem = problem
        model = problem.model
        if not isinstance(model, PredictorModel):
            msg = f"problem.model ({type(model).__name__}) does not implement the Predictor priming seam"
            raise TypeError(msg)
        self.model = model
        self.solver = solver if solver is not None else _default_solver(problem)
        ensure_solver_supports_constraints(problem, self.solver)
        ensure_solver_supports_objective(problem, self.solver)
        unprimed = jnp.nan_to_num(jnp.asarray(self.model.initial_state()), nan=0.0)
        self.state = initial_state if initial_state is not None else MPCState.initial(problem, x0=unprimed, dt=dt)
        self._state = np.asarray(self.model.initial_state(), dtype=np.float64)
        self._u_last = np.zeros(problem.model.m, dtype=np.float64)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate from a config dict, dispatching the Problem to a ``{class_path, ...}`` factory.

        Follows ``TrajOptMPC.from_config``'s pattern: ``problem`` is either a Problem instance
        or a ``{class_path, ...}`` dict naming a Problem-building factory (e.g.
        ``neuro.control.trajopt_mpc.build_waveform_problem``); ``solver`` and ``initial_state``
        are optional. When ``solver`` is omitted the default is chosen from the problem's
        constraints (see ``__init__``), so a migrated ``kirchhoff: true`` config needs no
        injected solver.
        """
        problem, initial_state = _build_problem(config["problem"])
        return cls(
            dt=float(config["dt"]),
            problem=problem,
            solver=config.get("solver"),
            initial_state=initial_state or config.get("initial_state"),
        )

    def update(
        self,
        t: float,
        ref: FloatArray,  # noqa: ARG002 -- the goal is baked into the objective
        x_hat: FloatArray,
    ) -> tuple[FloatArray, TrajOptMPCLog]:
        """Ingest the current EEG measurement, solve the receding-horizon problem, emit the first control."""
        self._state = np.asarray(self.model.absorb(self._state, np.asarray(x_hat).reshape(-1), self._u_last))

        if not self.model.is_ready(self._state):
            u_zero = np.zeros(self.model.m, dtype=np.float64)
            self._u_last = u_zero
            return u_zero, TrajOptMPCLog(u=u_zero, cost=0.0, success=True, warmup=True)

        state = self.state.with_measurement(jnp.asarray(self._state), t=t)
        solved = self.problem.solve(state, solver=self.solver)
        u_cmd = np.asarray(solved.controls[0], dtype=np.float64)
        self.state = solved.shift(self.dt)
        self._u_last = u_cmd
        return u_cmd, TrajOptMPCLog(
            u=u_cmd.copy(),
            cost=float(self.problem.cost(solved)),
            success=solved.status == "converged",
            warmup=False,
        )
