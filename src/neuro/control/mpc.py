from __future__ import annotations

import dataclasses
import importlib
import time
from typing import TYPE_CHECKING, Any, Self, cast

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
from trajopt.mpc import MPC
from trajopt.problem import Problem
from trajopt.solvers.altro import ALTRO
from trajopt.solvers.boxqp import BoxQP
from trajopt.solvers.options import SolverOptions
from trajopt.transcription.ipopt import Ipopt
from trajopt.transcription.result import constraint_row_count
from trajopt.transcription.single_shooting import SingleShooting

from neuro.control.costs import (
    ExcludeInitialKnotState,
    L1ControlCost,
    ObservableHingeCost,
    ReducedEffortCost,
    SpectralHingeCost,
    StateOutputs,
    SumCost,
    has_whole_horizon_cost,
)
from neuro.predictor.inference import InferencePredictor, ObservableMLPModel, WaveformMLPModel
from neuro.spectral import ObservableEnvelope, PsdEnvelope

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import ArrayLike
    from trajopt.costs.base import CostFunction
    from trajopt.problem import BoundaryConditions
    from trajopt.program import Program, WarmStart
    from trajopt.transcription.result import Solver, SolverResult

    from neuro.types import FloatArray


@dataclasses.dataclass(frozen=True)
class TrajOptMPCLog:
    """Per-step diagnostics: the applied control, optimal cost, solver success, solve time, warm-up flag."""

    u: FloatArray
    cost: float
    success: bool
    warmup: bool
    solve_time: float


def _build_problem(spec: dict[str, Any] | Problem) -> Problem:
    """Instantiate a Problem from an instance or a ``{class_path, ...}`` dict naming a factory."""
    if isinstance(spec, Problem):
        return spec
    cfg = spec.copy()
    class_path: str = cfg.pop("class_path")
    module_name, func_name = class_path.rsplit(".", 1)
    target = getattr(importlib.import_module(module_name), func_name)
    return target(**cfg) if cfg else target()


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


def _observable_envelope(envelope_ref: str | Path | None, w_hinge: float) -> ObservableEnvelope | None:
    """Load the healthy Observable envelope when ``w_hinge`` enables the hinge, else ``None``."""
    if w_hinge <= 0:
        return None
    if envelope_ref is None:
        msg = "envelope_ref must be provided when w_hinge > 0"
        raise ValueError(msg)
    return ObservableEnvelope.load(envelope_ref)


def _reduced_control_constraint(n: int, basis: jax.Array, u_max_arr: FloatArray) -> LinearConstraint:
    """Express the per-electrode limit ``|Z v| <= u_max`` as the polytope ``[Z; -Z] v - u_max <= 0``.

    The limit is a box on the electrode currents ``u``, but ``u = Z v`` shears it into a polytope
    on the reduced controls, so it is carried as coupled inequality rows rather than as bounds.
    """
    m_red = int(basis.shape[1])
    return LinearConstraint(
        n=n,
        m=m_red,
        A=jnp.concatenate([basis, -basis], axis=0),
        b=jnp.concatenate([jnp.asarray(u_max_arr), jnp.asarray(u_max_arr)]),
        inds=range(n, n + m_red),
    )


def _assemble_problem(  # noqa: PLR0913 -- the horizon's grid joins the five it already took
    model: DiscreteDynamics,
    objective: Objective,
    *,
    N: int,
    dt: float,
    u_max: ArrayLike,
    kirchhoff: bool,
    reduce_kirchhoff: bool = False,
) -> Problem:
    """Add the control bounds (and optional Kirchhoff equality) and build the ``Problem``.

    ``dt`` is the horizon's time grid, which trajopt now carries structurally on the Problem
    rather than per-step on the driver, so it is the Predictor's own step. Under
    ``reduce_kirchhoff`` the model is already a :class:`NullspaceReducedModel`, so ``u_max``
    is broadcast over the *electrode* count rather than over the reduced control count.
    """
    n, m = model.n, model.m
    if reduce_kirchhoff:
        if not isinstance(model, NullspaceReducedModel):
            msg = "reduce_kirchhoff requires the model to be wrapped in NullspaceReducedModel"
            raise TypeError(msg)
        u_max_arr = np.broadcast_to(np.atleast_1d(np.asarray(u_max, dtype=np.float64)), (m + 1,))
        constraints = ConstraintList(n=n, m=m, N=N)
        constraints.add_constraint(_reduced_control_constraint(n, model.basis, u_max_arr), range(N - 1))
        return Problem(model=model, obj=objective, constraints=constraints, N=N, dt=dt)
    u_max_arr = np.broadcast_to(np.atleast_1d(np.asarray(u_max, dtype=np.float64)), (m,))
    constraints = ConstraintList(n=n, m=m, N=N)
    constraints.add_constraint(ControlBound(n=n, m=m, u_min=-u_max_arr, u_max=u_max_arr), range(N - 1))
    if kirchhoff:
        constraints.add_constraint(kirchhoff_constraint(n, m), range(N - 1))
    return Problem(model=model, obj=objective, constraints=constraints, N=N, dt=dt)


def _default_solver(problem: Problem) -> Solver:
    """Select SingleShooting(Ipopt), the only backend every deployed formulation admits.

    Both Kirchhoff formulations couple the controls -- the hard equality as an equality row, the
    null-space reduction as the polytope carrying the per-electrode limit -- so the DDP backends,
    which clamp elementwise, are eligible for neither.
    """
    del problem
    return SingleShooting(solver=Ipopt(options={"print_level": 0, "hessian_approximation": "limited-memory"}))


def ensure_solver_supports_objective(problem: Problem, solver: Solver) -> None:
    """Raise when a solver does not support the problem's objective or constraints."""
    if has_whole_horizon_cost(problem.obj.stage_cost) and not isinstance(solver, (SingleShooting, Ipopt)):
        msg = (
            f"{type(solver).__name__} expands costs per knot and cannot score the whole-horizon "
            "hinge cost; use a transcription solver (e.g. SingleShooting(Ipopt(...)))."
        )
        raise ValueError(msg)
    if isinstance(solver, BoxQP) and sum(problem.constraints.p) > 0:
        msg = (
            f"BoxQP only supports uncoupled box bounds, but problem has non-box constraints "
            f"(p={problem.constraints.p}); use ALTRO or Ipopt for Kirchhoff's law in either formulation."
        )
        raise ValueError(msg)


def _instantiate_solver(class_path: str, cfg: dict[str, Any]) -> Solver:
    """Import and instantiate a solver from its class_path and config."""
    if not isinstance(class_path, str) or "." not in class_path:
        msg = f"solver 'class_path' must be a dot-separated import path, got {class_path!r}"
        raise ValueError(msg)

    module_name, class_name = class_path.rsplit(".", 1)
    target_cls = getattr(importlib.import_module(module_name), class_name)

    if "solver" in cfg and isinstance(cfg["solver"], dict):
        cfg["solver"] = _build_solver(cfg["solver"])

    if "options" in cfg and isinstance(cfg["options"], dict) and issubclass(target_cls, (ALTRO, BoxQP)):
        cfg["options"] = SolverOptions(**cfg["options"])

    res = target_cls(**cfg) if cfg else target_cls()
    if not isinstance(res, (ALTRO, BoxQP, SingleShooting, Ipopt)) and not hasattr(res, "solve"):
        msg = f"Target class {class_path} does not implement Solver interface"
        raise TypeError(msg)
    return res


def _build_solver(spec: dict[str, Any] | Solver | None, problem: Problem | None = None) -> Solver:
    """Instantiate a Solver from a ``{class_path, ...}`` dict or return the benchmark default.

    Parameters
    ----------
    spec
        Solver instance, config dict with ``'class_path'``, or ``None``.
    problem
        Optional Problem instance used to select the benchmark-winning default when ``spec`` is None.
    """
    if spec is None:
        if problem is None:
            msg = "Cannot select default solver without a Problem instance"
            raise ValueError(msg)
        return _default_solver(problem)

    if not isinstance(spec, dict):
        if not hasattr(spec, "solve"):
            msg = f"Invalid solver object of type {type(spec).__name__}: must be a dict with 'class_path' or implement .solve()"
            raise TypeError(msg)
        if problem is not None:
            ensure_solver_supports_objective(problem, spec)
        return spec

    cfg = spec.copy()
    if "class_path" not in cfg:
        msg = f"solver config must contain 'class_path', got keys: {list(cfg.keys())}"
        raise ValueError(msg)

    solver_instance = _instantiate_solver(cfg.pop("class_path"), cfg)
    if problem is not None:
        ensure_solver_supports_objective(problem, solver_instance)
    return solver_instance


_NO_DUALS = np.zeros(0, dtype=np.float64)


@dataclasses.dataclass(frozen=True)
class CanonicalDuals:
    """Solver adapter dropping duals a backend reports in a layout the warm start cannot shift.

    ``SingleShooting`` eliminates the states, so its primal is the control trajectory alone and
    Ipopt hands back bound duals of length ``(N - 1) * m`` and constraint duals over the user rows
    only. ``WarmStart`` is defined over the full Primal Vector and the canonical row order, and
    trajopt folds the backend's duals in without checking, so the next ``MPC.shift`` fails on the
    bound duals' shape and would mis-gather the constraint duals. Replacing a mismatched vector
    with the empty one puts the backend in the case the driver already handles: it returned no
    duals, so the warm start keeps its own.

    Only the backends that actually report a shifted layout are wrapped, because the adapter hides
    the backend's own type: trajopt classifies a solver by ``isinstance`` -- for the benchmark's
    ``linearizing`` column and for `ensure_solver_supports_objective` -- and a wrapper it cannot
    see through would misreport every solver it covers.
    """

    solver: Solver

    def solve(self, program: Program, bc: BoundaryConditions, ws: WarmStart) -> SolverResult:
        """Solve through the wrapped backend, blanking duals whose length is not the canonical one."""
        res = self.solver.solve(program, bc, ws)
        problem = program.problem
        N, n, m = int(problem.N), int(problem.model.n), int(problem.model.m)
        lam_ok = len(res.lam) in (0, constraint_row_count(problem))
        mu_ok = len(res.mu) in (0, N * n + (N - 1) * m)
        if lam_ok and mu_ok:
            return res
        # ``_replace`` carries every other field of the backend's own NamedTuple through untouched,
        # including ones the SolverResult Protocol does not name -- ALTRO's ``al`` among them, which
        # the driver reads back off the result to restore its multipliers.
        # ``SolverResult`` is a Protocol, so ``_replace`` is not on it; every backend satisfying
        # it is a NamedTuple, which is what makes the copy possible.
        return cast("Any", res)._replace(
            lam=res.lam if lam_ok else _NO_DUALS,
            mu=res.mu if mu_ok else _NO_DUALS,
        )


def canonicalize_duals(solver: Solver) -> Solver:
    """Wrap `solver` in :class:`CanonicalDuals` when its backend reports duals in a shifted layout.

    Only the Ipopt-backed transcriptions do. Every other backend is returned as it came, so trajopt
    can still classify it by ``isinstance`` -- which is what decides the benchmark's ``linearizing``
    column and what `ensure_solver_supports_objective` reads.
    """
    return CanonicalDuals(solver) if isinstance(solver, (SingleShooting, Ipopt)) else solver


def _validate_waveform_envelope(envelope: PsdEnvelope, model: WaveformMLPModel) -> None:
    """Ensure the healthy spectral envelope matches the predictor's channels and sampling rate."""
    if envelope.power.shape[0] != model.n_channels:
        msg = f"envelope channel count ({envelope.power.shape[0]}) does not match model channel count ({model.n_channels})."
        raise ValueError(msg)
    model_fs = 1.0 / model.dt
    if not np.isclose(envelope.fs, model_fs, rtol=1e-9):
        msg = f"envelope sampling rate ({envelope.fs:g} Hz) does not match model sampling rate ({model_fs:g} Hz)."
        raise ValueError(msg)


def _validate_observable_envelope(envelope: ObservableEnvelope, model: ObservableMLPModel) -> None:
    """Ensure the healthy Observable envelope matches the predictor's channels, rate, and geometry."""
    if envelope.power.shape[0] != model.n_channels:
        msg = f"envelope channel count ({envelope.power.shape[0]}) does not match model channel count ({model.n_channels})."
        raise ValueError(msg)
    envelope_frame_rate = model.geometry.frame_rate(envelope.fs)
    model_frame_rate = 1.0 / model.dt
    if not np.isclose(envelope_frame_rate, model_frame_rate, rtol=1e-9):
        msg = (
            f"envelope sampling rate ({envelope.fs:g} Hz) is a Frame rate of {envelope_frame_rate:g} Hz "
            f"at hop {model.geometry.n_hop}, but the model steps at {model_frame_rate:g} Hz."
        )
        raise ValueError(msg)
    if envelope.geometry != model.geometry:
        differing = ", ".join(
            f"{field} ({getattr(envelope.geometry, field)!r} vs {getattr(model.geometry, field)!r})"
            for field in type(model.geometry).model_fields
            if getattr(envelope.geometry, field) != getattr(model.geometry, field)
        )
        msg = f"envelope geometry does not match model geometry: {differing}."
        raise ValueError(msg)


_MIN_ELECTRODES = 2


def kirchhoff_basis(m: int) -> jax.Array:
    """Basis Z of shape ``(m, m - 1)`` spanning ``null(1^T)``, so ``u = Z v`` satisfies ``sum(u) = 0``.

    Eliminates the last electrode: ``v`` carries the first ``m - 1`` currents and
    ``u[m - 1] = -sum(v)``. At ``m = 2`` this is the bipolar pair ``[v, -v]^T``.
    """
    if m < _MIN_ELECTRODES:
        msg = f"Kirchhoff reduction needs at least {_MIN_ELECTRODES} electrodes, got m={m}"
        raise ValueError(msg)
    return jnp.concatenate([jnp.eye(m - 1), -jnp.ones((1, m - 1))], axis=0)


class NullspaceReducedModel(DiscreteDynamics, InferencePredictor):
    """DiscreteDynamics adapter mapping reduced controls ``v`` to electrode currents ``u = Z v``.

    ``Z`` is `kirchhoff_basis`, a basis of ``null(1^T)``, so Kirchhoff's Current Law holds
    identically for every ``v``: the equality leaves the constraint set entirely and the control
    dimension drops from ``m`` to ``m - 1``. The box bounds do not survive the map, since
    ``|Z v| <= u_max`` is a polytope in ``v``, which is what `_reduced_control_constraint` resolves.
    """

    base_model: InferencePredictor
    basis: jax.Array
    n_y: int = eqx.field(static=True)
    n_u: int = eqx.field(static=True)
    n_channels: int = eqx.field(static=True)
    n_controls: int = eqx.field(static=True)
    n_outputs: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)

    def __init__(self, base_model: InferencePredictor) -> None:
        """Wrap a base InferencePredictor over the last-electrode elimination basis."""
        super().__init__(n=base_model.n, m=base_model.m - 1, ne=base_model.ne)
        self.base_model = base_model
        self.basis = kirchhoff_basis(base_model.m)
        self.n_y = int(base_model.n_y)
        self.n_u = int(base_model.n_u)
        self.n_channels = int(base_model.n_channels)
        self.n_controls = int(base_model.m - 1)
        self.n_outputs = int(base_model.n_outputs)
        self.dt = float(base_model.dt)

    def discrete_dynamics(
        self,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array,
        dt: float | jax.Array,
    ) -> jax.Array:
        """Advance one step with the expanded currents u_full = Z v."""
        del t, dt
        u_full = self.basis @ jnp.atleast_1d(u)
        return self.base_model.discrete_dynamics(x, u_full, 0.0, self.dt)

    def absorb(self, state: FloatArray, y: FloatArray, u: FloatArray) -> FloatArray:
        """Absorb with the expanded currents u_full = Z v."""
        u_full = np.asarray(self.basis) @ np.atleast_1d(np.asarray(u, dtype=np.float64))
        return self.base_model.absorb(state, y, u_full)

    def is_ready(self, state: FloatArray) -> bool:
        """Report readiness from base model."""
        return self.base_model.is_ready(state)

    def initial_state(self) -> FloatArray:
        """Return base model's initial state."""
        return self.base_model.initial_state()

    def free_run(
        self,
        y_hists: FloatArray,
        u_hists: FloatArray,
        u_futures: FloatArray,
    ) -> jax.Array:
        """Free-run with the reduced controls expanded through Z; histories may already be full."""
        Z_T = np.asarray(self.basis).T
        u_f_full = np.asarray(u_futures, dtype=np.float64) @ Z_T
        u_h_arr = np.asarray(u_hists, dtype=np.float64)
        u_h_full = u_h_arr @ Z_T if u_h_arr.shape[-1] == self.m else u_h_arr
        return self.base_model.free_run(y_hists, u_h_full, u_f_full)

    def to_checkpoint(self) -> tuple[dict[str, Any], dict[str, FloatArray]]:
        """Return base model checkpoint."""
        return self.base_model.to_checkpoint()

    @classmethod
    def from_checkpoint(cls, meta: dict[str, Any], arrays: dict[str, FloatArray]) -> Self:
        """Rebuild base model then wrap."""
        if "geometry" in meta:
            base: InferencePredictor = ObservableMLPModel.from_checkpoint(meta, arrays)
        else:
            base = WaveformMLPModel.from_checkpoint(meta, arrays)
        return cls(base)


def build_waveform_problem(  # noqa: PLR0913 -- checkpoint plus the ten MPC cost/bound knobs
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
    reduce_kirchhoff: bool = False,
) -> Problem:
    """Assemble the waveform MPC problem: model adapter, objective, box and Kirchhoff bounds.

    The objective minimizes tracking deviation from zero, quadratic and L1 control effort, and
    one-sided log-power PSD hinges against ``psd_ref``. The constraints are the control box
    bounds ``-u_max <= u <= u_max``, plus the Kirchhoff sum-to-zero equality when ``kirchhoff``
    is set.

    Parameters
    ----------
    artifact
        Suffix-less stem of the numpy-readable MLP checkpoint.
    horizon
        Control Horizon counted in model steps; the trajopt horizon is ``horizon + 1`` knot
        points.
    u_max
        Per-electrode amplitude bound: a scalar shared by every electrode or a
        length-``n_controls`` vector.
    w_y
        Weight on state tracking error in the stage cost.
    w_u
        Weight on control effort (quadratic) in the stage cost.
    w_y_terminal
        Weight on the terminal knot state tracking error. When ``None`` (default), inherits
        ``w_y``.
    w_u_l1
        Weight on the L1 norm of the control effort (a sparse-stimulation penalty); ``0``
        disables it (default).
    w_psd
        Weight on the spectral hinge cost: the mean squared amount by which predicted log-power
        exceeds ``psd_ref``'s healthy envelope. ``0`` (default) disables it.
    psd_ref
        Path to the healthy reference envelope npz written by ``scripts/build_healthy_psd.py``.
        Required when ``w_psd > 0``.
    kirchhoff
        Add the Kirchhoff sum-to-zero equality on the controls. Off by default; the incumbent
        applies it unconditionally, so full parity sets it.
    reduce_kirchhoff
        Satisfy Kirchhoff's law by construction instead, parameterizing the currents as ``u = Z v``
        over `kirchhoff_basis`. The per-electrode limit is carried exactly, as the polytope
        ``[Z; -Z] v <= u_max``. Excludes ``kirchhoff``.
    """
    base = WaveformMLPModel.load(artifact)
    if reduce_kirchhoff:
        if kirchhoff:
            msg = "reduce_kirchhoff satisfies Kirchhoff by construction; drop kirchhoff"
            raise ValueError(msg)
        if w_u_l1 > 0:
            # ||Z v||_1 is not a weighted ||v||_1, so the L1 term has no reduced-coordinate form.
            msg = "reduce_kirchhoff does not support w_u_l1"
            raise ValueError(msg)
    model: DiscreteDynamics = NullspaceReducedModel(base) if reduce_kirchhoff else base
    n, m = model.n, model.m
    N = horizon + 1

    z_last = slice((base.n_y - 1) * base.n_channels, base.n_y * base.n_channels)
    Q = jnp.zeros(n).at[z_last].set(2.0 * w_y * base.y_scale**2 / horizon)
    xf = jnp.zeros(n).at[z_last].set(-base.y_center / base.y_scale)
    # Under reduction the effort is coupled (``||u||^2 = v^T Z^T Z v``), so it moves out of the
    # quadratic's ``R`` and into a control-only cost; folding it into ``R`` would promote the
    # diagonal state weight to a dense ``(n, n)`` matrix for no gain.
    R = jnp.zeros(m) if reduce_kirchhoff else jnp.full(m, 2.0 * w_u / horizon)
    stage = DiagonalCost.tracking(Q, R, xf, jnp.zeros(m))
    costs: list[CostFunction] = [ExcludeInitialKnotState(stage)]
    if reduce_kirchhoff:
        costs.append(ReducedEffortCost(n=n, m=m, w_u=w_u, horizon=horizon))
    if w_u_l1 > 0:
        costs.append(L1ControlCost(n=n, m=m, w_l1=w_u_l1, horizon=horizon))
    envelope = _spectral_envelope(psd_ref, w_psd)
    if envelope is not None:
        _validate_waveform_envelope(envelope, base)
        outputs = StateOutputs(
            n=n,
            m=m,
            n_y=base.n_y,
            n_outputs=base.n_channels,
            center=base.y_center,
            scale=base.y_scale,
        )
        costs.append(SpectralHingeCost(outputs, envelope, w_psd=w_psd, horizon=horizon))
    stage_cost: CostFunction = _combine_costs(costs)
    # The terminal knot carries the horizon's final output, weighted by ``w_y_terminal`` when
    # given else ``w_y`` -- the incumbent's last-step stage cost. Always explicit, because the
    # composite's derived terminal (``SumCost.as_terminal``) would otherwise carry the
    # control-only L1 and whole-horizon hinge into a knot that has no control.
    w_y_final = w_y_terminal if w_y_terminal is not None else w_y
    Q_f = jnp.zeros(n).at[z_last].set(2.0 * w_y_final * base.y_scale**2 / horizon)
    terminal = DiagonalCost.terminal_tracking(Q_f, xf, m)
    objective = Objective(stage_cost=stage_cost, terminal_cost=terminal, N=N)

    return _assemble_problem(
        model,
        objective,
        N=N,
        dt=base.dt,
        u_max=u_max,
        kirchhoff=kirchhoff,
        reduce_kirchhoff=reduce_kirchhoff,
    )


def build_observable_problem(  # noqa: PLR0913 -- checkpoint plus the MPC cost/bound knobs
    artifact: str | Path,
    *,
    horizon: int,
    u_max: ArrayLike,
    w_u: float = 0.0,
    w_u_l1: float = 0.0,
    w_hinge: float = 0.0,
    envelope_ref: str | Path | None = None,
    kirchhoff: bool = False,
    reduce_kirchhoff: bool = False,
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
    w_hinge
        Weight on the hinge cost: the mean squared amount by which the predicted log-power
        Frames exceed ``envelope_ref``'s healthy envelope. ``0`` (default) disables it.
    envelope_ref
        Path to the healthy reference Observable envelope npz written by
        ``scripts/build_healthy_psd.py``. Required when ``w_hinge > 0``; its stored geometry
        drives the cost.
    kirchhoff
        Add the Kirchhoff sum-to-zero equality on the controls.
    reduce_kirchhoff
        Satisfy Kirchhoff's law by construction instead, parameterizing the currents as ``u = Z v``
        over `kirchhoff_basis`. Excludes ``kirchhoff``.
    """
    base = ObservableMLPModel.load(artifact)
    if reduce_kirchhoff and kirchhoff:
        msg = "reduce_kirchhoff satisfies Kirchhoff by construction; drop kirchhoff"
        raise ValueError(msg)
    model = NullspaceReducedModel(base) if reduce_kirchhoff else base
    n, m = model.n, model.m
    N = horizon + 1

    R = jnp.zeros(m) if reduce_kirchhoff else jnp.full(m, 2.0 * w_u / horizon)
    stage = DiagonalCost.tracking(jnp.zeros(n), R, jnp.zeros(n), jnp.zeros(m))
    costs: list[CostFunction] = [ExcludeInitialKnotState(stage)]
    if reduce_kirchhoff:
        costs.append(ReducedEffortCost(n=n, m=m, w_u=w_u, horizon=horizon))
    if w_u_l1 > 0:
        costs.append(L1ControlCost(n=n, m=m, w_l1=w_u_l1, horizon=horizon))
    envelope = _observable_envelope(envelope_ref, w_hinge)
    # The stage trajectory carries every Frame of the Control Horizon but the last, which lives
    # only in the terminal knot; the terminal Cost scores it so no predicted Frame goes unpriced.
    terminal: CostFunction = DiagonalCost.terminal_tracking(jnp.zeros(n), jnp.zeros(n), m)
    if envelope is not None:
        _validate_observable_envelope(envelope, base)
        outputs = StateOutputs(
            n=n,
            m=m,
            n_y=base.n_y,
            n_outputs=base.n_outputs,
            center=base.y_center,
            scale=base.y_scale,
        )
        costs.append(ObservableHingeCost(outputs, envelope, w_hinge=w_hinge, horizon=horizon))
        terminal = ObservableHingeCost(outputs, envelope, w_hinge=w_hinge, horizon=horizon, terminal=True)
    stage_cost: CostFunction = _combine_costs(costs)
    objective = Objective(stage_cost=stage_cost, terminal_cost=terminal, N=N)

    return _assemble_problem(
        model,
        objective,
        N=N,
        dt=base.dt,
        u_max=u_max,
        kirchhoff=kirchhoff,
        reduce_kirchhoff=reduce_kirchhoff,
    )


class TrajOptMPCController(Controller[TrajOptMPCLog]):
    """Receding-horizon MPC for the Waveform Predictor, driven by trajopt's :class:`~trajopt.mpc.MPC`.

    The driver owns the horizon's boundary conditions and warm start, and holds one compiled
    ``Program`` for the life of the run, so the solver's cores are built once rather than per
    step. This controller owns what the driver does not: the true State-Absorbed Predictor state and
    ``u_last``, which are what turn a measurement into the driver's ``x0``.
    """

    def __init__(
        self,
        dt: float,
        problem: Problem,
        solver: Solver | dict[str, Any] | None = None,
    ) -> None:
        """Initialize the controller and its persistent MPC driver.

        Parameters
        ----------
        dt
            Controller update step in seconds; should equal the Predictor's native dt.
        problem
            The trajopt optimal-control problem: model adapter + objective + constraint list.
        solver
            Solver backend instance or ``{class_path, ...}`` config dict. When omitted,
            `_default_solver` picks ``SingleShooting(Ipopt)``.
        """
        super().__init__(dt)
        model = problem.model
        if not isinstance(model, InferencePredictor):
            msg = f"problem.model ({type(model).__name__}) does not implement the InferencePredictor priming seam"
            raise TypeError(msg)
        self.model = model
        self.solver = _build_solver(solver, problem)

        # The unprimed EEG window is NaN padding; the driver seeds a full state trajectory from
        # x0, not just the controls, so it is zeroed to keep that seed finite for every solver.
        unprimed = jnp.nan_to_num(jnp.asarray(self.model.initial_state()), nan=0.0)
        self.mpc = MPC(problem, canonicalize_duals(self.solver), x0=unprimed)
        self._state = np.asarray(self.model.initial_state(), dtype=np.float64)
        self._u_last = np.zeros(model.m, dtype=np.float64)
        # Under reduction the decision variable is ``v``, one shorter than the montage; the Plant
        # takes electrode currents, so the basis is kept here to expand at the boundary.
        self._basis = np.asarray(model.basis, dtype=np.float64) if isinstance(model, NullspaceReducedModel) else None
        self.n_electrodes = model.m + 1 if self._basis is not None else model.m

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate from a config dict, dispatching the Problem and Solver factories.

        Follows the standard ``{class_path, ...}`` pattern: ``problem`` is either a Problem instance
        or a ``{class_path, ...}`` dict naming a Problem-building factory (e.g.
        ``neuro.control.mpc.build_waveform_problem``); ``solver`` is an optional
        ``{class_path, ...}`` dict naming a solver backend (e.g. ``trajopt.solvers.altro.ALTRO``).
        When ``solver`` is omitted, the benchmark-winning default is selected automatically.
        """
        problem = _build_problem(config["problem"])
        return cls(
            dt=float(config["dt"]),
            problem=problem,
            solver=_build_solver(config.get("solver"), problem),
        )

    def update(
        self,
        t: float,
        ref: FloatArray,  # noqa: ARG002 -- the goal is baked into the objective
        x_hat: FloatArray,
    ) -> tuple[FloatArray, TrajOptMPCLog]:
        """Ingest the EEG measurement, solve the receding-horizon problem, emit the first electrode currents.

        The emitted control is always the ``(n_electrodes,)`` physical currents: under a Nullspace
        Frame the solver decides in the reduced ``v``, which is expanded through ``Z`` here.
        """
        self._state = np.asarray(self.model.absorb(self._state, np.asarray(x_hat).reshape(-1), self._u_last))

        if not self.model.is_ready(self._state):
            self._u_last = np.zeros(self.model.m, dtype=np.float64)
            u_zero = np.zeros(self.n_electrodes, dtype=np.float64)
            return u_zero, TrajOptMPCLog(u=u_zero, cost=0.0, success=True, warmup=True, solve_time=0.0)

        self.mpc.measure(jnp.asarray(self._state), t)
        started = time.perf_counter()
        solved = self.mpc.solve()
        solve_time = time.perf_counter() - started
        u_solved = np.asarray(self.mpc.controls[0], dtype=np.float64)
        cost = float(self.mpc.cost())
        self.mpc.shift(self.dt)
        # ``_u_last`` feeds ``model.absorb``, which expands for itself, so the state keeps the
        # solver's own coordinates while the Plant and the log get the electrode currents.
        self._u_last = u_solved
        u_cmd = u_solved if self._basis is None else self._basis @ u_solved
        return u_cmd, TrajOptMPCLog(
            u=u_cmd.copy(),
            cost=cost,
            success=bool(solved.success),
            warmup=False,
            solve_time=solve_time,
        )
