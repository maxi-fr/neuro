from __future__ import annotations

import dataclasses
import importlib
from typing import TYPE_CHECKING, Any, Self

import jax.numpy as jnp
import numpy as np
from simulate.controller import Controller
from trajopt.cones import ZeroCone
from trajopt.constraints.bounds import ControlBound
from trajopt.constraints.constraint_list import ConstraintList
from trajopt.constraints.linear import GoalConstraint, LinearConstraint
from trajopt.costs.objective import Objective
from trajopt.costs.quadratic import DiagonalCost
from trajopt.problem import MPCState, Problem
from trajopt.transcription.ipopt import Ipopt
from trajopt.transcription.single_shooting import SingleShooting

from neuro.control.costs import (
    ExcludeInitialKnotState,
    L1ControlCost,
    ObservableHingeCost,
    SpectralHingeCost,
    SumCost,
    has_whole_horizon_cost,
)
from neuro.predictor.inference import InferencePredictor, ObservableMLPModel, WaveformMLPModel
from neuro.spectral import ObservableEnvelope, PsdEnvelope

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import ArrayLike
    from trajopt.costs.base import CostFunction
    from trajopt.dynamics.base import DiscreteDynamics
    from trajopt.transcription.result import Solver

    from neuro.types import FloatArray


@dataclasses.dataclass(frozen=True)
class TrajOptMPCLog:
    """Per-step diagnostics: the applied control, optimal cost, solver success, and warm-up flag."""

    u: FloatArray
    cost: float
    success: bool
    warmup: bool


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


def _observable_envelope(psd_ref: str | Path | None, w_psd: float) -> ObservableEnvelope | None:
    """Load the healthy Observable envelope when ``w_psd`` enables the hinge, else ``None``."""
    if w_psd <= 0:
        return None
    if psd_ref is None:
        msg = "psd_ref must be provided when w_psd > 0"
        raise ValueError(msg)
    return ObservableEnvelope.load(psd_ref)


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
    model = WaveformMLPModel.load(artifact)
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
        costs.append(
            SpectralHingeCost(
                n=n,
                m=m,
                n_y=model.n_y,
                n_channels=model.n_channels,
                y_center=model.y_center,
                y_scale=model.y_scale,
                envelope=envelope,
                w_psd=w_psd,
                horizon=horizon,
            )
        )
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


def build_observable_problem(  # noqa: PLR0913 -- checkpoint plus the MPC cost/bound knobs
    artifact: str | Path,
    *,
    horizon: int,
    u_max: ArrayLike,
    w_u: float = 0.0,
    w_u_l1: float = 0.0,
    w_psd: float = 0.0,
    psd_ref: str | Path | None = None,
    kirchhoff: bool = False,
) -> Problem:
    """Assemble the observable MPC problem: model adapter, objective, box and Kirchhoff bounds.

    The model steps one Frame per call on the hop grid. The objective minimizes the one-sided
    log-power hinge against the healthy Observable envelope, plus quadratic and L1 control effort
    penalties. The constraints are the control box bounds ``-u_max <= u <= u_max``, plus the
    Kirchhoff sum-to-zero equality when ``kirchhoff`` is set.

    Parameters
    ----------
    artifact
        Suffix-less stem of the numpy-readable Observable MLP checkpoint.
    horizon
        Control Horizon counted in Frames; the trajopt horizon is ``horizon + 1`` knot points.
    u_max
        Per-electrode amplitude bound: a scalar shared by every electrode or a
        length-``n_controls`` vector.
    w_u
        Weight on control effort (quadratic) in the cost.
    w_u_l1
        Weight on the L1 norm of the control effort (a sparse-stimulation penalty); ``0``
        disables it (default).
    w_psd
        Weight on the spectral hinge cost: the mean squared amount by which predicted log-power
        Frames exceed ``psd_ref``'s healthy envelope. ``0`` (default) disables it.
    psd_ref
        Path to the healthy reference envelope npz written by ``scripts/build_healthy_psd.py``.
        Required when ``w_psd > 0``; its stored geometry drives the cost.
    kirchhoff
        Add the Kirchhoff sum-to-zero equality on the controls.
    """
    model = ObservableMLPModel.load(artifact)
    n, m = model.n, model.m
    N = horizon + 1

    stage = DiagonalCost.tracking(jnp.zeros(n), jnp.full(m, 2.0 * w_u / horizon), jnp.zeros(n), jnp.zeros(m))
    costs: list[CostFunction] = [ExcludeInitialKnotState(stage)]
    if w_u_l1 > 0:
        costs.append(L1ControlCost(n=n, m=m, w_l1=w_u_l1, horizon=horizon))
    envelope = _observable_envelope(psd_ref, w_psd)
    if envelope is not None:
        costs.append(
            ObservableHingeCost(
                n=n,
                m=m,
                n_y=model.n_y,
                n_outputs=model.n_outputs,
                y_center=model.y_center,
                y_scale=model.y_scale,
                envelope=envelope,
                w_psd=w_psd,
                horizon=horizon,
            )
        )
    stage_cost: CostFunction = _combine_costs(costs)
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
        if not isinstance(model, InferencePredictor):
            msg = f"problem.model ({type(model).__name__}) does not implement the InferencePredictor priming seam"
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
        ``neuro.control.mpc.build_waveform_problem``); ``solver`` and ``initial_state``
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
