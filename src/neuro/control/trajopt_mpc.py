from __future__ import annotations

import dataclasses
import importlib
from typing import TYPE_CHECKING, Any, Protocol, Self, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from simulate.controller import Controller
from trajopt.cones import ZeroCone
from trajopt.constraints.bounds import ControlBound
from trajopt.constraints.constraint_list import ConstraintList
from trajopt.constraints.linear import LinearConstraint
from trajopt.costs.objective import Objective
from trajopt.costs.quadratic import DiagonalCost
from trajopt.dynamics.base import DiscreteDynamics
from trajopt.problem import MPCState, Problem
from trajopt.transcription.ipopt import Ipopt
from trajopt.transcription.single_shooting import SingleShooting

from neuro.checkpoint import MLPCheckpoint, load_mlp
from neuro.control.trajopt_costs import L1ControlCost, SpectralHingeCost, SumCost
from neuro.spectral import PsdEnvelope

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import ArrayLike
    from trajopt.costs.base import CostFunction
    from trajopt.transcription.result import Solver

    from neuro.types import Activation, FloatArray


class PredictorModel(Protocol):
    """A trajopt model that also exposes the Predictor's raw-units priming seam.

    The controller absorbs measurements into the model's opaque state (``absorb``), holds off
    until the state is primed (``is_ready``) and seeds its ``MPCState`` from the unprimed state
    (``initial_state``) -- the same seam ``MPCController`` uses today, hosted on the trajopt
    model adapter instead of the CasADi bridge. ``m`` and ``n_channels`` are the control and
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
        if self.activation == "relu":
            return jnp.maximum(z, 0.0)
        if self.activation == "tanh":
            return jnp.tanh(z)
        return jnp.logaddexp(z, 0.0)

    def _predict(self, y_window: jax.Array, u_window: jax.Array) -> jax.Array:
        """One MLP forward pass on standardized windows -> the next standardized sample ``(n_channels,)``."""
        z = jnp.concatenate([y_window.reshape(-1), u_window.reshape(-1)])
        for i, weight in enumerate(self.weights[:-1]):
            z = self._activate(z @ weight.T + self.biases[i])
        weight_last, bias_last = self.weights[-1], self.biases[-1]
        return z @ weight_last.T + bias_last

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
    set -- ticket 01's quadratic/box subset leaves the equality out, because the single-shooting
    transcription cannot carry it; the full set is solved with the general Ipopt transcription.

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
        Add the Kirchhoff sum-to-zero equality on the controls. Off by default, matching
        ticket 01's quadratic/box subset; the incumbent applies it unconditionally, so the
        full parity test sets it.
    """
    model = WaveformMLPModel.from_checkpoint(artifact)
    n, m = model.n, model.m
    N = horizon + 1
    u_max_arr = np.broadcast_to(np.atleast_1d(np.asarray(u_max, dtype=np.float64)), (m,))

    z_last = slice((model.n_y - 1) * model.n_channels, model.n_y * model.n_channels)
    Q = jnp.zeros(n).at[z_last].set(2.0 * w_y * model.y_scale**2 / horizon)
    xf = jnp.zeros(n).at[z_last].set(-model.y_center / model.y_scale)
    stage = DiagonalCost.tracking(Q, jnp.full(m, 2.0 * w_u / horizon), xf, jnp.zeros(m))
    costs: list[CostFunction] = [stage]
    if w_u_l1 > 0:
        costs.append(L1ControlCost(n=n, m=m, w_l1=w_u_l1, horizon=horizon))
    if w_psd > 0:
        if psd_ref is None:
            msg = "psd_ref must be provided when w_psd > 0"
            raise ValueError(msg)
        envelope = PsdEnvelope.load(psd_ref)
        costs.append(
            SpectralHingeCost(
                model=model,
                envelope=envelope,
                w_psd=w_psd,
                horizon=horizon,
            )
        )
    stage_cost: CostFunction = SumCost(costs) if len(costs) > 1 else costs[0]
    # The terminal knot carries the horizon's final output, weighted by ``w_y_terminal`` when
    # given else ``w_y`` -- the incumbent's last-step stage cost. Always explicit, because the
    # composite's derived terminal (``SumCost.as_terminal``) would otherwise carry the
    # control-only L1 and whole-horizon hinge into a knot that has no control.
    w_y_final = w_y_terminal if w_y_terminal is not None else w_y
    Q_f = jnp.zeros(n).at[z_last].set(2.0 * w_y_final * model.y_scale**2 / horizon)
    terminal = DiagonalCost.terminal_tracking(Q_f, xf, m)
    objective = Objective(stage_cost=stage_cost, terminal_cost=terminal, N=N)

    constraints = ConstraintList(n=n, m=m, N=N)
    constraints.add_constraint(ControlBound(n=n, m=m, u_min=-u_max_arr, u_max=u_max_arr), range(N - 1))
    if kirchhoff:
        constraints.add_constraint(kirchhoff_constraint(n, m), range(N - 1))
    return Problem(model=model, obj=objective, constraints=constraints, N=N)


class TrajOptMPCController(Controller[TrajOptMPCLog]):
    """Receding-horizon MPC for the waveform predictor, built directly on trajopt primitives.

    Owns the true absorbed predictor state and ``u_last`` as private instance attributes, and
    reproduces the incumbent ``MPCController.update`` loop shape -- ``absorb``, then
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
            Solver backend (e.g. ``SingleShooting``); defaults to Ipopt single shooting with
            printing off.
        initial_state
            Initial MPC warm-start state; defaults to one built from the unprimed model state,
            with the NaN padding of the unprimed EEG window replaced by zeros so the seed is
            finite for every solver (the multiple-shooting transcription starts from the full
            state trajectory, not just the controls).
        """
        super().__init__(dt)
        self.problem = problem
        self.model = cast("PredictorModel", problem.model)
        self.solver = solver if solver is not None else SingleShooting(solver=Ipopt(options={"print_level": 0}))
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
        are optional.
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
