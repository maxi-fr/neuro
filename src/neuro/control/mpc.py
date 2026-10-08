from __future__ import annotations

import dataclasses
import importlib
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from simulate.controller import Controller
from trajopt.cones import NegativeOrthant, ZeroCone
from trajopt.constraints.bounds import ControlBound
from trajopt.constraints.constraint_list import ConstraintList
from trajopt.constraints.horizon import LinearHorizonConstraint
from trajopt.constraints.linear import LinearConstraint
from trajopt.costs.objective import Objective
from trajopt.costs.output import OutputCost
from trajopt.costs.quadratic import DiagonalCost
from trajopt.dynamics.base import DiscreteDynamics
from trajopt.mpc import MPC
from trajopt.problem import BoundaryConditions, Problem, retarget_problem
from trajopt.program import WarmStart
from trajopt.solvers.altro import ALTRO
from trajopt.solvers.boxqp import BoxQP
from trajopt.solvers.options import SolverOptions
from trajopt.trajectory import Trajectory
from trajopt.transcription.ipopt import _SUCCESS_STATUSES, Ipopt, IpoptResult
from trajopt.transcription.layout import _trajectory_to_z, parse_solver_initial_state
from trajopt.transcription.result import constraint_row_count, split_bound_duals
from trajopt.transcription.single_shooting import (
    SingleShooting,
    _compute_constraint_violation,
    _constraint_bounds,
    _dense_jacobian_pattern,
    _primal_bounds,
    _validate_supported_constraints,
    eval_f,
    eval_g,
    eval_grad_f,
    eval_jac_g,
    rollout_states,
    single_shooting_dimensions,
)

from neuro.control.costs import (
    ExcludeInitialKnotState,
    L1ControlCost,
    ObservableFrameHingeCost,
    ObservableHingeCost,
    ReducedEffortCost,
    SumCost,
    has_whole_horizon_cost,
)
from neuro.predictor.inference import (
    InferencePredictor,
    ObservableCNNModel,
    ObservableMLPModel,
    WaveformCNNModel,
    WaveformMLPModel,
    inference_from_checkpoint,
)
from neuro.spectral import HealthyReference, ObservableEnvelope

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike
    from trajopt.constraints.constraint_list import BuiltConstraintList
    from trajopt.costs.base import CostFunction
    from trajopt.dynamics.base import IntegratorCallable
    from trajopt.dynamics.integrators import Integrator
    from trajopt.program import Program
    from trajopt.transcription.result import Solver, SolverResult

    from neuro.types import FloatArray


ACTIVE_CURRENT_THRESHOLD: float = 1e-3


@dataclasses.dataclass(frozen=True)
class TrajOptMPCLog:
    """Decision diagnostics and unshifted plans: outputs ``(H+1, p)``, Control Currents ``(H, m)``."""

    u: FloatArray
    cost: float
    success: bool
    warmup: bool
    solve_time: float
    predicted_y: FloatArray
    planned_u: FloatArray
    status: str = "Solve_Succeeded"
    planned_active: FloatArray = dataclasses.field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    active_count: float = 0.0
    cost_active: float = 0.0
    cost_spectral: float = 0.0
    cost_quadratic_effort: float = 0.0
    cost_sparse_effort: float = 0.0
    cost_tracking: float = 0.0
    normalization: str = "channel_mean"


@eqx.filter_jit
def planned_outputs(
    model: InferencePredictor, states: jax.Array, controls: jax.Array, t: float, dt: float
) -> jax.Array:
    """Project every solved state to physical outputs, including the initial and terminal knots."""
    padded = jnp.concatenate((controls, controls[-1:]), axis=0)
    times = t + jnp.arange(states.shape[0]) * dt
    return jax.vmap(model.output)(states, padded, times)


def _build_problem(spec: dict[str, Any] | Problem) -> Problem:
    """Instantiate a Problem from an instance or a ``{class_path, ...}`` dict naming a factory."""
    if isinstance(spec, Problem):
        return spec
    cfg = spec.copy()
    if "reference" in cfg and isinstance(cfg["reference"], (str, Path)):
        cfg["reference"] = HealthyReference.load(cfg["reference"])
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


def _observable_envelope(
    reference: HealthyReference | None,
    w_hinge: float,
) -> ObservableEnvelope | None:
    """Extract the healthy Observable envelope when ``w_hinge`` enables the hinge, else ``None``."""
    if w_hinge <= 0:
        return None
    if reference is None:
        msg = "reference must be provided when w_hinge > 0"
        raise ValueError(msg)
    if reference.observable is None:
        msg = "reference contains no Observable envelope"
        raise ValueError(msg)
    return reference.observable


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


def _amplitude_coupling_constraint(n: int, m_elec: int, u_max_arr: FloatArray) -> LinearConstraint:
    """Express |u_i| <= u_max_i * delta as [I; -I] u - u_max delta <= 0."""
    m = m_elec + 1
    A = np.zeros((2 * m_elec, m), dtype=np.float64)
    u_max_vec = np.asarray(u_max_arr, dtype=np.float64)
    for i in range(m_elec):
        A[2 * i, i] = 1.0
        A[2 * i, -1] = -u_max_vec[i]
        A[2 * i + 1, i] = -1.0
        A[2 * i + 1, -1] = -u_max_vec[i]
    b = np.zeros(2 * m_elec, dtype=np.float64)
    return LinearConstraint(
        n=n,
        m=m,
        A=jnp.asarray(A),
        b=jnp.asarray(b),
        sense=NegativeOrthant(),
        inds=range(n, n + m),
    )


def _augmented_kirchhoff_constraint(n: int, m_elec: int) -> LinearConstraint:
    """Express sum(u_i) = 0 for augmented controls [u; delta]."""
    m = m_elec + 1
    A = np.zeros((1, m), dtype=np.float64)
    A[0, :m_elec] = 1.0
    b = np.zeros(1, dtype=np.float64)
    return LinearConstraint(
        n=n,
        m=m,
        A=jnp.asarray(A),
        b=jnp.asarray(b),
        sense=ZeroCone(),
        inds=range(n, n + m),
    )


class NeuroProblem(Problem):
    """Optimal control Problem carrying neurostimulation weights, bounds, and Observable reference."""

    horizon: int = eqx.field(static=True)
    w_active: float | None = eqx.field(static=True, default=None)
    w_hinge: float = eqx.field(static=True, default=0.0)
    w_u: float = eqx.field(static=True, default=0.0)
    w_u_l1: float = eqx.field(static=True, default=0.0)
    u_max: jax.Array = eqx.field(default=None)  # type: ignore[assignment]
    kirchhoff: bool = eqx.field(static=True, default=False)
    envelope: ObservableEnvelope | None = eqx.field(static=True, default=None)
    max_active_intervals: int | None = eqx.field(static=True, default=None)
    continuous_problem: Problem | None = eqx.field(default=None)

    def __init__(  # noqa: PLR0913, PLR0917 -- carries problem definition plus domain evaluation metadata
        self,
        model: DiscreteDynamics,
        obj: Objective,
        constraints: BuiltConstraintList | ConstraintList | None = None,
        N: int | None = None,
        dt: float | jax.Array = 0.05,
        integrator: Integrator | IntegratorCallable | None = None,
        binary_control_indices: Sequence[int] = (),
        *,
        horizon: int,
        u_max: ArrayLike,
        w_active: float | None = None,
        w_hinge: float = 0.0,
        w_u: float = 0.0,
        w_u_l1: float = 0.0,
        kirchhoff: bool = False,
        envelope: ObservableEnvelope | None = None,
        max_active_intervals: int | None = None,
        continuous_problem: Problem | None = None,
    ) -> None:
        """Construct a NeuroProblem carrying domain weights, envelope, and bounds."""
        super().__init__(
            model=model,
            obj=obj,
            constraints=constraints,
            N=N,
            dt=dt,
            integrator=integrator,
            binary_control_indices=binary_control_indices,
        )
        self.horizon = int(horizon)
        self.u_max = jnp.asarray(u_max, dtype=jnp.float64)
        self.w_active = w_active
        self.w_hinge = float(w_hinge)
        self.w_u = float(w_u)
        self.w_u_l1 = float(w_u_l1)
        self.kirchhoff = bool(kirchhoff)
        self.envelope = envelope
        self.max_active_intervals = max_active_intervals
        self.continuous_problem = continuous_problem


def _assemble_problem(  # noqa: PLR0913 -- the horizon's grid joins the five it already took
    model: DiscreteDynamics,
    objective: Objective,
    *,
    N: int,
    dt: float,
    u_max: ArrayLike,
    kirchhoff: bool,
    reduce_kirchhoff: bool = False,
    binary_control_indices: Sequence[int] = (),
    max_active_intervals: int | None = None,
    w_active: float | None = None,
    w_hinge: float = 0.0,
    w_u: float = 0.0,
    w_u_l1: float = 0.0,
    envelope: ObservableEnvelope | None = None,
    continuous_problem: Problem | None = None,
) -> NeuroProblem:
    """Add the control bounds (and optional Kirchhoff equality) and build the ``NeuroProblem``."""
    n, m = model.n, model.m
    horizon = N - 1
    if binary_control_indices:
        m_elec = m - len(binary_control_indices)
        u_max_arr = np.broadcast_to(np.atleast_1d(np.asarray(u_max, dtype=np.float64)), (m_elec,)).copy()
        constraints = ConstraintList(n=n, m=m, N=N)
        u_min = np.concatenate([-u_max_arr, [0.0] * len(binary_control_indices)])
        u_max_bound = np.concatenate([u_max_arr, [1.0] * len(binary_control_indices)])
        constraints.add_constraint(ControlBound(n=n, m=m, u_min=u_min, u_max=u_max_bound), range(N - 1))
        constraints.add_constraint(_amplitude_coupling_constraint(n, m_elec, u_max_arr), range(N - 1))
        if kirchhoff:
            constraints.add_constraint(_augmented_kirchhoff_constraint(n, m_elec), range(N - 1))
        if max_active_intervals is not None:
            inds = [k * (n + m) + n + m - 1 for k in range(N - 1)]
            A_cap = jnp.ones((1, len(inds)))
            b_cap = jnp.array([float(max_active_intervals)])
            constraints.add_horizon_constraint(
                LinearHorizonConstraint(A=A_cap, b=b_cap, sense=NegativeOrthant(), inds=inds)
            )
        return NeuroProblem(
            model=model,
            obj=objective,
            constraints=constraints,
            N=N,
            dt=dt,
            binary_control_indices=binary_control_indices,
            horizon=horizon,
            u_max=u_max_arr,
            w_active=w_active,
            w_hinge=w_hinge,
            w_u=w_u,
            w_u_l1=w_u_l1,
            kirchhoff=kirchhoff,
            envelope=envelope,
            max_active_intervals=max_active_intervals,
            continuous_problem=continuous_problem,
        )

    if reduce_kirchhoff:
        if not isinstance(model, NullspaceReducedModel):
            msg = "reduce_kirchhoff requires the model to be wrapped in NullspaceReducedModel"
            raise TypeError(msg)
        u_max_arr = np.broadcast_to(np.atleast_1d(np.asarray(u_max, dtype=np.float64)), (m + 1,))
        constraints = ConstraintList(n=n, m=m, N=N)
        constraints.add_constraint(_reduced_control_constraint(n, model.basis, u_max_arr), range(N - 1))
        return NeuroProblem(
            model=model,
            obj=objective,
            constraints=constraints,
            N=N,
            dt=dt,
            horizon=horizon,
            u_max=u_max_arr,
            w_active=w_active,
            w_hinge=w_hinge,
            w_u=w_u,
            w_u_l1=w_u_l1,
            kirchhoff=True,
            envelope=envelope,
            max_active_intervals=max_active_intervals,
        )
    u_max_arr = np.broadcast_to(np.atleast_1d(np.asarray(u_max, dtype=np.float64)), (m,))
    constraints = ConstraintList(n=n, m=m, N=N)
    constraints.add_constraint(ControlBound(n=n, m=m, u_min=-u_max_arr, u_max=u_max_arr), range(N - 1))
    if kirchhoff:
        constraints.add_constraint(kirchhoff_constraint(n, m), range(N - 1))
    return NeuroProblem(
        model=model,
        obj=objective,
        constraints=constraints,
        N=N,
        dt=dt,
        horizon=horizon,
        u_max=u_max_arr,
        w_active=w_active,
        w_hinge=w_hinge,
        w_u=w_u,
        w_u_l1=w_u_l1,
        kirchhoff=kirchhoff,
        envelope=envelope,
        max_active_intervals=max_active_intervals,
    )


IPOPT_DEFAULTS: dict[str, Any] = {
    "print_level": 0,
    "hessian_approximation": "limited-memory",
    "tol": 1e-3,
    "acceptable_tol": 1e-2,
    "acceptable_iter": 5,
    "max_iter": 300,
}


_BINARY_WARM_START_THRESHOLD = 1e-2
_CONTINUOUS_POLISH_THRESHOLD = 0.20


class _SingleShootingBinaryCallback:
    """Callback wrapper connecting Single-Shooting JAX kernels to cyipopt with binary penalty and cap."""

    def __init__(  # noqa: PLR0913, PLR0917 -- Callback needs problem, boundary conditions, and binary configuration
        self,
        problem: Problem,
        x0: jax.Array,
        t0: jax.Array,
        dt: jax.Array,
        xf: jax.Array | None = None,
        max_active_intervals: int | None = None,
    ) -> None:
        """Initialize single-shooting callback and build sparsity pattern."""
        self.problem = problem
        self.x0 = x0
        self.t0 = t0
        self.dt = dt
        self.xf = xf
        self.max_active_intervals = max_active_intervals

        N = int(problem.N)
        m = int(problem.model.m)
        n_u, p_user = single_shooting_dimensions(problem)
        self.n_u = n_u
        self.p_base = p_user
        self.p_user = p_user + (1 if max_active_intervals is not None else 0)

        base_jac_rows, base_jac_cols = _dense_jacobian_pattern(p_user, n_u)
        if max_active_intervals is not None:
            cap_row = np.full(n_u, p_user, dtype=np.int32)
            cap_col = np.arange(n_u, dtype=np.int32)
            self.jac_rows = np.concatenate([base_jac_rows, cap_row])
            self.jac_cols = np.concatenate([base_jac_cols, cap_col])
        else:
            self.jac_rows = base_jac_rows
            self.jac_cols = base_jac_cols

        self.binary_u_indices = np.asarray(
            [k * m + i for k in range(N - 1) for i in problem.binary_control_indices], dtype=np.int32
        )
        self.beta = 0.0
        self.iteration_count = 0

    def intermediate(self, *args: object) -> bool:
        """Track solver iteration counter from cyipopt."""
        if len(args) > 1 and isinstance(args[1], (int, float, str)):
            self.iteration_count = int(args[1])
        return True

    def objective(self, u: np.ndarray) -> float:
        """Evaluate scalar objective value plus binary penalty."""
        cost = float(eval_f(self.problem, jnp.asarray(u), self.x0, t0=self.t0, dt=self.dt))
        if len(self.binary_u_indices) and self.beta > 0:
            a = u[self.binary_u_indices]
            cost += self.beta * float(np.sum(a * (1.0 - a)))
        return cost

    def gradient(self, u: np.ndarray) -> np.ndarray:
        """Evaluate objective gradient nabla J(u) plus binary penalty gradient."""
        grad = np.asarray(
            eval_grad_f(self.problem, jnp.asarray(u), self.x0, t0=self.t0, dt=self.dt),
            dtype=np.float64,
        ).copy()
        if len(self.binary_u_indices) and self.beta > 0:
            grad[self.binary_u_indices] += self.beta * (1.0 - 2.0 * u[self.binary_u_indices])
        return grad

    def constraints(self, u: np.ndarray) -> np.ndarray:
        """Evaluate constraint vector c(u) and optional active-interval cap."""
        c_base = np.asarray(
            eval_g(self.problem, jnp.asarray(u), self.x0, t0=self.t0, dt=self.dt, xf=self.xf),
            dtype=np.float64,
        )
        if self.max_active_intervals is not None:
            a = u[self.binary_u_indices]
            cap_val = np.array([float(np.sum(a) - self.max_active_intervals)], dtype=np.float64)
            return np.concatenate([c_base, cap_val])
        return c_base

    def jacobian(self, u: np.ndarray) -> np.ndarray:
        """Evaluate dense constraint Jacobian values and optional cap row."""
        j_base = np.asarray(
            eval_jac_g(self.problem, jnp.asarray(u), self.x0, t0=self.t0, dt=self.dt, xf=self.xf),
            dtype=np.float64,
        )
        if self.max_active_intervals is not None:
            grad_cap = np.zeros(self.n_u, dtype=np.float64)
            grad_cap[self.binary_u_indices] = 1.0
            return np.concatenate([j_base, grad_cap])
        return j_base

    def jacobianstructure(self) -> tuple[np.ndarray, np.ndarray]:
        """Return build-time constraint Jacobian sparsity pattern (rows, cols)."""
        return self.jac_rows, self.jac_cols


def _run_continuation(  # noqa: PLR0913, PLR0917 -- Homotopy continuation requires problem options and loop states
    nlp: Any,  # noqa: ANN401 -- cyipopt Problem instance is an untyped C-extension
    cb: _SingleShootingBinaryCallback,
    u0: np.ndarray,
    lam0: np.ndarray | None,
    mu0: np.ndarray | None,
    n_passes: int,
    start_pass: int,
    beta_initial: float,
    beta_growth: float,
    binary_tolerance: float,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict[str, Any], int, float]:
    """Execute multi-pass binary penalty continuation loop."""
    u_curr = u0
    info: dict[str, Any] = {}
    pass_count = 0
    binary_distance = 0.0
    has_binary = len(cb.binary_u_indices) > 0

    for pass_index in range(start_pass, n_passes):
        pass_count += 1
        cb.beta = 0.0 if pass_index == 0 else beta_initial * (beta_growth ** (pass_index - 1))
        if lam0 is not None and mu0 is not None:
            nlp.add_option("warm_start_init_point", "yes")
            nlp.add_option("warm_start_bound_push", 1e-9)
            nlp.add_option("warm_start_mult_bound_push", 1e-9)
            mult_x_l, mult_x_u = split_bound_duals(mu0)
            u_curr, info = nlp.solve(u_curr, lagrange=lam0, zl=mult_x_l, zu=mult_x_u)
        else:
            u_curr, info = nlp.solve(u_curr)

        u_curr = np.asarray(u_curr, dtype=np.float64)
        lam0 = np.asarray(info.get("mult_g", _NO_DUALS), dtype=np.float64)
        mu0 = np.asarray(info.get("mult_x_U", _NO_DUALS), dtype=np.float64) - np.asarray(
            info.get("mult_x_L", _NO_DUALS), dtype=np.float64
        )
        if has_binary:
            a = u_curr[cb.binary_u_indices]
            binary_distance = float(np.max(np.minimum(np.abs(a), np.abs(1.0 - a))))
        if int(info.get("status", -1)) not in _SUCCESS_STATUSES:
            break
        if pass_index > 0 and binary_distance <= binary_tolerance:
            break

    return u_curr, lam0, mu0, info, pass_count, binary_distance


class SingleShootingBinaryIpopt(Ipopt):
    """Single-shooting transcription solver with binary penalty homotopy continuation."""

    def __init__(
        self,
        solver: Ipopt | None = None,
        *,
        beta_initial: float = 1.0,
        beta_growth: float = 10.0,
        max_passes: int = 8,
        binary_tolerance: float = 1e-4,
    ) -> None:
        """Initialize single-shooting solver with binary continuation settings."""
        opts = solver.options if solver is not None else IPOPT_DEFAULTS
        super().__init__(
            options=opts,
            beta_initial=beta_initial,
            beta_growth=beta_growth,
            max_passes=max_passes,
            binary_tolerance=binary_tolerance,
        )

    def solve(self, program: Program, bc: BoundaryConditions, ws: WarmStart) -> IpoptResult:
        """Solve using single shooting with binary penalty homotopy continuation."""
        import cyipopt  # noqa: PLC0415 -- cyipopt is an optional solver dependency

        problem = program.problem
        _validate_supported_constraints(problem)

        N = int(problem.N)
        m = int(problem.model.m)

        x0_arr, t0_arr, dt_arr, xf_val, _ = parse_solver_initial_state(problem, bc, ws)
        problem = retarget_problem(problem, bc)
        dt_arr = jnp.broadcast_to(dt_arr, (N - 1,))

        u0 = np.asarray(ws.unpack(problem)[1], dtype=np.float64).reshape(-1)
        z_lower, z_upper = _primal_bounds(problem)
        g_lower, g_upper = _constraint_bounds(problem)

        max_cap = getattr(problem, "max_active_intervals", None)
        if max_cap is not None:
            g_lower = np.append(g_lower, -np.inf)
            g_upper = np.append(g_upper, 0.0)

        cb = _SingleShootingBinaryCallback(problem, x0_arr, t0_arr, dt_arr, xf_val, max_active_intervals=max_cap)

        problem_cls: Any = getattr(cyipopt, "Problem")  # noqa: B009 -- cyipopt is an untyped C-extension
        nlp = problem_cls(
            n=len(u0),
            m=len(g_lower),
            problem_obj=cb,
            lb=z_lower,
            ub=z_upper,
            cl=g_lower,
            cu=g_upper,
        )

        options: dict[str, Any] = dict(self.options)
        options.setdefault("hessian_approximation", "limited-memory")
        for key, value in options.items():
            nlp.add_option(key, value)

        has_binary = len(cb.binary_u_indices) > 0
        n_passes = self.max_passes if has_binary else 1
        lam0 = np.asarray(ws.lam, dtype=np.float64) if len(ws.lam) == len(g_lower) else None
        mu0 = np.asarray(ws.mu, dtype=np.float64) if len(ws.mu) == len(u0) else None

        a0 = u0[cb.binary_u_indices] if has_binary else np.empty(0)
        start_dist = float(np.max(np.minimum(np.abs(a0), np.abs(1.0 - a0)))) if len(a0) else 0.0
        start_pass = 1 if (has_binary and start_dist <= _BINARY_WARM_START_THRESHOLD) else 0

        u_curr, lam0, mu0, info, pass_count, binary_distance = _run_continuation(
            nlp,
            cb,
            u0,
            lam0,
            mu0,
            n_passes,
            start_pass,
            self.beta_initial,
            self.beta_growth,
            self.binary_tolerance,
        )

        status = int(info.get("status", -1))
        success = status in _SUCCESS_STATUSES and (not has_binary or binary_distance <= self.binary_tolerance)
        message = str(info.get("status_msg", ""))

        u_opt = jnp.asarray(u_curr, dtype=jnp.float64).reshape((N - 1, m))
        t_opt = t0_arr + jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), jnp.cumsum(dt_arr)])
        x_opt = rollout_states(problem.model, x0_arr, u_opt, t=t_opt, dt=dt_arr)

        opt_traj = Trajectory(X=x_opt, U=u_opt, t=t_opt, dt=dt_arr)
        z_opt_jax = _trajectory_to_z(x_opt, u_opt)
        cost_val = float(eval_f(problem, u_opt.reshape(-1), x0_arr, t0=t0_arr, dt=dt_arr))
        viol = _compute_constraint_violation(problem, u_opt.reshape(-1), x0_arr, t0_arr, dt_arr, xf_val)

        info.update(
            {
                "pass_count": pass_count,
                "binary_distance": binary_distance,
                "integrality_violation": binary_distance,
            }
        )

        mu_canonical = _NO_DUALS
        if mu0 is not None and len(mu0) == (N - 1) * m:
            mu_u_arr = jnp.asarray(mu0, dtype=jnp.float64).reshape((N - 1, m))
            mu_x_arr = jnp.zeros((N, int(problem.model.n)), dtype=jnp.float64)
            mu_canonical = np.asarray(_trajectory_to_z(mu_x_arr, mu_u_arr), dtype=np.float64)

        return IpoptResult(
            trajectory=opt_traj,
            success=success,
            status=status,
            message=message,
            cost=cost_val,
            Z=z_opt_jax,
            info=info,
            iterations=int(cb.iteration_count),
            constraint_violation=viol,
            lam=_NO_DUALS,
            mu=mu_canonical,
        )


def _default_solver(problem: Problem) -> Solver:
    """Select single-shooting Ipopt with binary homotopy continuation for all MPC problems."""
    if problem.binary_control_indices or problem.constraints.horizon_constraints:
        return SingleShootingBinaryIpopt(Ipopt(options=IPOPT_DEFAULTS))
    return SingleShooting(solver=Ipopt(options=IPOPT_DEFAULTS))


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
    """Import and instantiate a Solver from its class_path and config, defaulting Ipopt options."""
    if not isinstance(class_path, str) or "." not in class_path:
        msg = f"solver 'class_path' must be a dot-separated import path, got {class_path!r}"
        raise ValueError(msg)

    module_name, class_name = class_path.rsplit(".", 1)
    target_cls = getattr(importlib.import_module(module_name), class_name)

    if "solver" in cfg and isinstance(cfg["solver"], dict):
        cfg["solver"] = _build_solver(cfg["solver"])

    if issubclass(target_cls, Ipopt):
        cfg["options"] = {**IPOPT_DEFAULTS, **cfg.get("options", {})}
    elif issubclass(target_cls, SingleShooting) and "solver" not in cfg:
        cfg["solver"] = Ipopt(options=dict(IPOPT_DEFAULTS))
    elif "options" in cfg and isinstance(cfg["options"], dict) and issubclass(target_cls, (ALTRO, BoxQP)):
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


def _validate_waveform_envelope(envelope: ObservableEnvelope, model: WaveformMLPModel | WaveformCNNModel) -> None:
    """Ensure the healthy Observable envelope matches the waveform predictor's channels and sample rate."""
    if envelope.power.shape[0] != model.n_channels:
        msg = f"envelope channel count ({envelope.power.shape[0]}) does not match model channel count ({model.n_channels})."
        raise ValueError(msg)
    model_fs = 1.0 / model.dt
    if not np.isclose(envelope.fs, model_fs, rtol=1e-9):
        msg = f"envelope sampling rate ({envelope.fs:g} Hz) does not match model sampling rate ({model_fs:g} Hz)."
        raise ValueError(msg)


def _validate_observable_envelope(envelope: ObservableEnvelope, model: ObservableMLPModel | ObservableCNNModel) -> None:
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
        """Wrap a base InferencePredictor over the last-electrode elimination basis, inheriting output dimension ``p``."""
        super().__init__(n=base_model.n, m=base_model.m - 1, ne=base_model.ne, p=base_model.p)
        self.base_model = base_model
        self.basis = kirchhoff_basis(base_model.m)
        self.n_y = int(base_model.n_y)
        self.n_u = int(base_model.n_u)
        self.n_channels = int(base_model.n_channels)
        self.n_controls = int(base_model.m - 1)
        self.n_outputs = int(base_model.n_outputs)
        self.dt = float(base_model.dt)

    def output(
        self,
        x: jax.Array,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate physical output, projecting reduced Control Currents when feedthrough is present."""
        u_full = self.basis @ jnp.atleast_1d(u) if u is not None and self.base_model.has_control_feedthrough() else None
        return self.base_model.output(x, u_full, t)

    def has_control_feedthrough(self) -> bool:
        """Whether the base model's output function depends directly on Control Current ``u``."""
        return self.base_model.has_control_feedthrough()

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

    @property
    def n_history(self) -> int:
        """History buffer length of the base model if defined, else n_y."""
        return getattr(self.base_model, "n_history", getattr(self.base_model, "n_y", 1))

    def past_outputs(self, x: jax.Array, count: int) -> jax.Array:
        """Extract physical past outputs through the base model."""
        return self.base_model.past_outputs(x, count)

    def with_history(self, n_history: int) -> NullspaceReducedModel:
        """Return a copy wrapping base_model with extended history."""
        return NullspaceReducedModel(self.base_model.with_history(n_history))

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
        base = inference_from_checkpoint(meta, arrays)
        return cls(base)


class AugmentedActivationModel(DiscreteDynamics, InferencePredictor):
    """DiscreteDynamics adapter adding an activation variable as the last control coordinate."""

    base_model: InferencePredictor
    n_y: int = eqx.field(static=True)
    n_u: int = eqx.field(static=True)
    n_channels: int = eqx.field(static=True)
    n_controls: int = eqx.field(static=True)
    n_outputs: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)

    def __init__(self, base_model: InferencePredictor) -> None:
        """Wrap a base InferencePredictor adding an activation coordinate as the last control."""
        super().__init__(n=base_model.n, m=base_model.m + 1, ne=base_model.ne, p=base_model.p)
        self.base_model = base_model
        self.n_y = int(base_model.n_y)
        self.n_u = int(base_model.n_u)
        self.n_channels = int(base_model.n_channels)
        self.n_controls = int(base_model.n_controls)
        self.n_outputs = int(base_model.n_outputs)
        self.dt = float(base_model.dt)

    def output(
        self,
        x: jax.Array,
        u: jax.Array | None = None,
        t: float | jax.Array = 0.0,
    ) -> jax.Array:
        """Evaluate physical output, dropping the activation coordinate when feedthrough is present."""
        u_phys = u[: self.base_model.m] if u is not None and self.base_model.has_control_feedthrough() else None
        return self.base_model.output(x, u_phys, t)

    def has_control_feedthrough(self) -> bool:
        """Whether the base model's output function depends directly on Control Current."""
        return self.base_model.has_control_feedthrough()

    def discrete_dynamics(
        self,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array,
        dt: float | jax.Array,
    ) -> jax.Array:
        """Advance one step with physical controls u[:base_model.m]."""
        del t, dt
        return self.base_model.discrete_dynamics(x, u[: self.base_model.m], 0.0, self.dt)

    def absorb(self, state: FloatArray, y: FloatArray, u: FloatArray) -> FloatArray:
        """Absorb measurement and physical control history."""
        return self.base_model.absorb(state, y, u[: self.base_model.m])

    def is_ready(self, state: FloatArray) -> bool:
        """Report readiness from base model."""
        return self.base_model.is_ready(state)

    @property
    def n_history(self) -> int:
        """History buffer length of the base model if defined, else n_y."""
        return getattr(self.base_model, "n_history", getattr(self.base_model, "n_y", 1))

    @property
    def y_scale(self) -> FloatArray | None:
        """Physical observation scale vector from base model."""
        return getattr(self.base_model, "y_scale", None)

    @property
    def y_center(self) -> FloatArray | None:
        """Physical observation centering vector from base model."""
        return getattr(self.base_model, "y_center", None)

    def past_outputs(self, x: jax.Array, count: int) -> jax.Array:
        """Extract physical past outputs through the base model."""
        return self.base_model.past_outputs(x, count)

    def with_history(self, n_history: int) -> AugmentedActivationModel:
        """Return a copy wrapping base_model with extended history."""
        return AugmentedActivationModel(self.base_model.with_history(n_history))

    def initial_state(self) -> FloatArray:
        """Return base model's initial state."""
        return self.base_model.initial_state()

    def free_run(
        self,
        y_hists: FloatArray,
        u_hists: FloatArray,
        u_futures: FloatArray,
    ) -> jax.Array:
        """Free-run with the physical controls u[..., :base_model.m]."""
        u_f = np.asarray(u_futures, dtype=np.float64)
        u_h = np.asarray(u_hists, dtype=np.float64)
        u_f_phys = u_f[..., : self.base_model.m]
        u_h_phys = u_h[..., : self.base_model.m] if u_h.shape[-1] == self.m else u_h
        return self.base_model.free_run(y_hists, u_h_phys, u_f_phys)

    def to_checkpoint(self) -> tuple[dict[str, Any], dict[str, FloatArray]]:
        """Return base model checkpoint."""
        return self.base_model.to_checkpoint()

    @classmethod
    def from_checkpoint(cls, meta: dict[str, Any], arrays: dict[str, FloatArray]) -> Self:
        """Rebuild base model then wrap."""
        base = inference_from_checkpoint(meta, arrays)
        return cls(base)


def _resolve_waveform_target(
    ref: HealthyReference | None,
    base: WaveformMLPModel | WaveformCNNModel,
    *,
    tracking_active: bool,
    w_hinge: float,
) -> FloatArray | None:
    """Resolve and validate the physical tracking target from a HealthyReference."""
    if tracking_active and ref is None:
        msg = "reference must be provided when w_y > 0"
        raise ValueError(msg)
    if w_hinge > 0 and ref is None:
        msg = "reference must be provided when w_hinge > 0"
        raise ValueError(msg)
    if not tracking_active or ref is None:
        return None
    if ref.eeg_mean is not None and len(ref.eeg_mean) == base.n_channels:
        y_ref = ref.eeg_mean
    elif ref.lfp_mean is not None and len(ref.lfp_mean) == base.n_channels:
        y_ref = ref.lfp_mean
    else:
        msg = f"reference mean vector does not match model channel count ({base.n_channels})"
        raise ValueError(msg)
    return np.asarray(y_ref, dtype=np.float64)


def _load_waveform_runtime(artifact: str | Path) -> WaveformMLPModel | WaveformCNNModel:
    """Load and validate a waveform MLP or CNN runtime checkpoint."""
    base = InferencePredictor.load(artifact)
    if not isinstance(base, (WaveformMLPModel, WaveformCNNModel)):
        msg = f"waveform checkpoint required, got {type(base).__name__}"
        raise TypeError(msg)
    return base


def _waveform_terminal_cost(
    base_terminal: CostFunction,
    frame_hinge: ObservableFrameHingeCost | None,
) -> CostFunction:
    """Combine output tracking and optional terminal Frame hinge into the terminal Cost."""
    if frame_hinge is None:
        return base_terminal
    return _combine_costs([base_terminal, frame_hinge.as_terminal()])


def build_waveform_problem(  # noqa: PLR0913 -- checkpoint plus the ten MPC cost/bound knobs
    artifact: str | Path,
    *,
    horizon: int,
    u_max: ArrayLike,
    w_y: float = 1.0,
    w_u: float = 0.0,
    w_y_terminal: float | None = None,
    w_u_l1: float = 0.0,
    w_hinge: float = 0.0,
    reference: HealthyReference | None = None,
    kirchhoff: bool = False,
    reduce_kirchhoff: bool = False,
) -> Problem:
    """Assemble waveform MPC with stage and terminal spectral Frames and the Control Budget.

    The objective minimizes tracking deviation from the healthy reference operating point,
    quadratic and L1 control effort, and one-sided log-power Frame hinges against ``reference``.
    The constraints are the control box bounds ``-u_max <= u <= u_max``, plus the Kirchhoff
    sum-to-zero equality when ``kirchhoff`` is set.

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
        Weight on Raw EEG output tracking error in the stage Cost.
    w_u
        Weight on Control Current effort (quadratic) in the stage Cost.
    w_y_terminal
        Weight on the terminal knot Raw EEG output tracking error. When ``None`` (default), inherits
        ``w_y``.
    w_u_l1
        Weight on the L1 norm of the Control Current effort (a sparse-stimulation penalty); ``0``
        disables it (default).
    w_hinge
        Weight on the spectral hinge Cost: the mean squared amount by which predicted log-power Frames
        exceeds ``reference``'s healthy envelope. ``0`` (default) disables it.
    reference
        :class:`~neuro.spectral.HealthyReference` container carrying empirical channel
        means and/or Observable envelope. Required when ``w_y > 0`` or ``w_hinge > 0``.
    kirchhoff
        Add the Kirchhoff sum-to-zero equality on the controls. Off by default; the incumbent
        applies it unconditionally, so full parity sets it.
    reduce_kirchhoff
        Satisfy Kirchhoff's law by construction instead, parameterizing the currents as ``u = Z v``
        over `kirchhoff_basis`. The per-electrode limit is carried exactly, as the polytope
        ``[Z; -Z] v <= u_max``. Excludes ``kirchhoff``.
    """
    base = _load_waveform_runtime(artifact)
    envelope = _observable_envelope(reference, w_hinge)
    if envelope is not None:
        _validate_waveform_envelope(envelope, base)
        support = envelope.geometry.sample_support_steps(envelope.fs)
        if base.n_history < support:
            base = base.with_history(support)
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

    w_y_final = w_y_terminal if w_y_terminal is not None else w_y
    target_val = _resolve_waveform_target(
        reference,
        base,
        tracking_active=(w_y > 0 or w_y_final > 0),
        w_hinge=w_hinge,
    )

    target = jnp.zeros(model.p) if target_val is None else jnp.asarray(target_val)
    Q = jnp.full(model.p, 2.0 * w_y / horizon)
    # Under reduction the effort is coupled (``||u||^2 = v^T Z^T Z v``), so it moves out of the
    # quadratic's ``R`` and into a control-only cost; folding it into ``R`` would promote the
    # diagonal state weight to a dense ``(n, n)`` matrix for no gain.
    R = jnp.zeros(m) if reduce_kirchhoff else jnp.full(m, 2.0 * w_u / horizon)
    stage = DiagonalCost.tracking(Q, R, target, jnp.zeros(m))
    output_stage = OutputCost(model, stage)
    costs: list[CostFunction] = [ExcludeInitialKnotState(output_stage)]
    if reduce_kirchhoff:
        costs.append(ReducedEffortCost(n=n, m=m, w_u=w_u, horizon=horizon))
    if w_u_l1 > 0:
        costs.append(L1ControlCost(n=n, m=m, w_l1=w_u_l1, horizon=horizon))
    frame_hinge: ObservableFrameHingeCost | None = None
    if envelope is not None:
        frame_hinge = ObservableFrameHingeCost(model, envelope, w_hinge=w_hinge, horizon=horizon)
        costs.append(frame_hinge)
    stage_cost: CostFunction = _combine_costs(costs)
    if w_y_terminal is not None and w_y_terminal != w_y:
        Q_f = jnp.full(model.p, 2.0 * w_y_final / horizon)
        base_terminal: CostFunction = OutputCost(model, DiagonalCost.terminal_tracking(Q_f, target, m=m))
    else:
        base_terminal = output_stage.as_terminal()
    terminal_cost = _waveform_terminal_cost(base_terminal, frame_hinge)
    objective = Objective(stage_cost=stage_cost, terminal_cost=terminal_cost, N=N)

    return _assemble_problem(
        model,
        objective,
        N=N,
        dt=base.dt,
        u_max=u_max,
        kirchhoff=kirchhoff,
        reduce_kirchhoff=reduce_kirchhoff,
        w_u=w_u,
        w_u_l1=w_u_l1,
        w_hinge=w_hinge,
        envelope=envelope,
    )


def _validate_integer_observable_problem_args(
    horizon: int,
    *,
    w_active: float | None,
    max_active_intervals: int | None,
    kirchhoff: bool,
    reduce_kirchhoff: bool,
) -> bool:
    """Validate mutual exclusivity, bounds, and formulation support for integer problem configurations."""
    if reduce_kirchhoff and kirchhoff:
        msg = "reduce_kirchhoff satisfies Kirchhoff by construction; drop kirchhoff"
        raise ValueError(msg)
    if max_active_intervals is not None and (
        isinstance(max_active_intervals, bool)
        or not isinstance(max_active_intervals, int)
        or not 0 <= max_active_intervals <= horizon
    ):
        msg = "max_active_intervals must be an integer between zero and horizon"
        raise ValueError(msg)
    if w_active is not None and (not np.isfinite(w_active) or w_active < 0):
        msg = "w_active must be finite and nonnegative"
        raise ValueError(msg)
    if max_active_intervals is not None and w_active is not None:
        msg = "max_active_intervals and w_active are mutually exclusive"
        raise ValueError(msg)
    integer_mode = w_active is not None or max_active_intervals is not None
    if integer_mode and reduce_kirchhoff:
        msg = "integer mode does not currently support reduce_kirchhoff; use kirchhoff=True"
        raise NotImplementedError(msg)
    return integer_mode


def build_observable_problem(  # noqa: PLR0913 -- checkpoint plus the MPC cost/bound knobs
    artifact: str | Path | ObservableMLPModel | ObservableCNNModel,
    *,
    horizon: int,
    u_max: ArrayLike,
    w_u: float = 0.0,
    w_u_l1: float = 0.0,
    w_hinge: float = 0.0,
    reference: HealthyReference | None = None,
    kirchhoff: bool = False,
    reduce_kirchhoff: bool = False,
    w_active: float | None = None,
    max_active_intervals: int | None = None,
) -> NeuroProblem:
    """Assemble the observable MPC problem: model adapter, objective, box and Kirchhoff bounds.

    The model steps one Frame per call on the hop grid. The objective minimizes the one-sided
    log-power hinge against the healthy Observable envelope, plus quadratic and L1 control effort
    penalties. The constraints are the control box bounds ``-u_max <= u <= u_max``, plus the
    Kirchhoff sum-to-zero equality when ``kirchhoff`` is set.

    Parameters
    ----------
    artifact
        Observable Predictor checkpoint path or loaded model instance.
    horizon
        Control Horizon counted in Frames; the trajopt horizon is ``horizon + 1`` knot points.
    u_max
        Per-electrode amplitude bound: a scalar shared by every electrode or a
        length-``n_controls`` vector.
    w_u
        Weight on Control Current effort (quadratic) in the Cost.
    w_u_l1
        Weight on the L1 norm of the Control Current effort (a sparse-stimulation penalty); ``0``
        disables it (default).
    w_hinge
        Weight on the hinge Cost: the mean squared amount by which the predicted log-power
        Frames exceed ``reference``'s healthy envelope. ``0`` (default) disables it.
    reference
        :class:`~neuro.spectral.HealthyReference` container carrying the healthy Observable envelope.
        Required when ``w_hinge > 0``.
    kirchhoff
        Add the Kirchhoff sum-to-zero equality on the controls.
    reduce_kirchhoff
        Satisfy Kirchhoff's law by construction instead, parameterizing the currents as ``u = Z v``
        over `kirchhoff_basis`. Excludes ``kirchhoff``.
    w_active
        Weight on enabled Control Horizon intervals (activation penalty). Mutually exclusive with
        ``max_active_intervals``.
    max_active_intervals
        Upper bound on the number of enabled intervals per Control Horizon (activation cap).
        Mutually exclusive with ``w_active``.
    """
    base = (
        artifact
        if isinstance(artifact, (ObservableMLPModel, ObservableCNNModel))
        else InferencePredictor.load(artifact)
    )
    if not isinstance(base, (ObservableMLPModel, ObservableCNNModel)):
        msg = f"Observable checkpoint required, got {type(base).__name__}"
        raise TypeError(msg)
    integer_mode = _validate_integer_observable_problem_args(
        horizon,
        w_active=w_active,
        max_active_intervals=max_active_intervals,
        kirchhoff=kirchhoff,
        reduce_kirchhoff=reduce_kirchhoff,
    )

    raw_model = NullspaceReducedModel(base) if reduce_kirchhoff else base
    model: DiscreteDynamics = AugmentedActivationModel(raw_model) if integer_mode else raw_model
    n, m = model.n, model.m
    N = horizon + 1

    if w_hinge > 0 and reference is None:
        msg = "reference must be provided when w_hinge > 0"
        raise ValueError(msg)

    R = jnp.zeros(m)
    r = jnp.zeros(m)
    if not reduce_kirchhoff:
        m_phys = base.m
        R = R.at[:m_phys].set(2.0 * w_u / horizon)
    if w_active is not None:
        r = r.at[-1].set(w_active / horizon)
    stage = DiagonalCost(Q=jnp.zeros(n), R=R, q=jnp.zeros(n), r=r)
    costs: list[CostFunction] = [ExcludeInitialKnotState(stage)]
    if reduce_kirchhoff:
        costs.append(ReducedEffortCost(n=n, m=m, w_u=w_u, horizon=horizon))
    if w_u_l1 > 0:
        costs.append(L1ControlCost(n=n, m=m, w_l1=w_u_l1, horizon=horizon))
    envelope = _observable_envelope(reference, w_hinge)
    # The stage trajectory carries every Frame of the Control Horizon but the last, which lives
    # only in the terminal knot; the terminal Cost scores it so no predicted Frame goes unpriced.
    terminal: CostFunction = DiagonalCost.terminal_tracking(jnp.zeros(n), jnp.zeros(n), m)
    if envelope is not None:
        _validate_observable_envelope(envelope, base)
        hinge = ObservableHingeCost(envelope, w_hinge=w_hinge, horizon=horizon)
        output_hinge = OutputCost(model, hinge)
        costs.append(ExcludeInitialKnotState(output_hinge))
        terminal = output_hinge.as_terminal()
    stage_cost: CostFunction = _combine_costs(costs)
    objective = Objective(stage_cost=stage_cost, terminal_cost=terminal, N=N)

    u_max_arr = np.broadcast_to(np.atleast_1d(np.asarray(u_max, dtype=np.float64)), (base.m,)).copy()
    cont_prob = None
    if integer_mode:
        cont_prob = build_observable_problem(
            artifact=base,
            horizon=horizon,
            u_max=u_max,
            w_u=w_u,
            w_u_l1=w_u_l1,
            w_hinge=w_hinge,
            reference=reference,
            kirchhoff=kirchhoff,
            reduce_kirchhoff=reduce_kirchhoff,
            max_active_intervals=None,
            w_active=None,
        )

    return _assemble_problem(
        model,
        objective,
        N=N,
        dt=base.dt,
        u_max=u_max_arr,
        kirchhoff=kirchhoff,
        reduce_kirchhoff=reduce_kirchhoff,
        binary_control_indices=(m - 1,) if integer_mode else (),
        max_active_intervals=max_active_intervals,
        w_active=w_active,
        w_hinge=w_hinge,
        w_u=w_u,
        w_u_l1=w_u_l1,
        envelope=envelope,
        continuous_problem=cont_prob,
    )


def _collect_leaf_costs(cost: CostFunction) -> list[CostFunction]:
    """Flatten SumCost hierarchies into individual leaf CostFunction instances."""
    if isinstance(cost, SumCost):
        leaves: list[CostFunction] = []
        for c in cost.costs:
            leaves.extend(_collect_leaf_costs(c))
        return leaves
    return [cost]


def _diagonal_r(c: CostFunction) -> jax.Array | None:
    """Extract control penalty weight R from a diagonal or wrapped diagonal cost."""
    target = c.inner if isinstance(c, ExcludeInitialKnotState) else c
    diag = target.cost if isinstance(target, OutputCost) else target
    return diag.R if isinstance(diag, DiagonalCost) else None


def _diagonal_linear_r(c: CostFunction) -> jax.Array | None:
    """Extract linear control weight r from a diagonal or wrapped diagonal cost."""
    target = c.inner if isinstance(c, ExcludeInitialKnotState) else c
    diag = target.cost if isinstance(target, OutputCost) else target
    return diag.r if isinstance(diag, DiagonalCost) else None


def _is_spectral_cost(c: CostFunction) -> bool:
    """Identify whether a CostFunction represents an Observable or spectral hinge."""
    return (
        isinstance(c, (ObservableFrameHingeCost, ObservableHingeCost))
        or (isinstance(c, OutputCost) and isinstance(c.cost, (ObservableFrameHingeCost, ObservableHingeCost)))
        or (isinstance(c, ExcludeInitialKnotState) and _is_spectral_cost(c.inner))
    )


def _stage_leaf_contributions(
    c: CostFunction,
    states: jax.Array,
    controls: jax.Array,
    t_stage: jax.Array,
) -> tuple[float, float, float, float, float]:
    """Break one stage Cost leaf into (spectral, quadratic_effort, sparse_effort, tracking, active)."""
    full_val = float(jnp.sum(c.stage_costs(states[:-1], controls, t_stage)))
    if _is_spectral_cost(c):
        return full_val, 0.0, 0.0, 0.0, 0.0
    if isinstance(c, L1ControlCost):
        return 0.0, 0.0, full_val, 0.0, 0.0
    if isinstance(c, ReducedEffortCost):
        return 0.0, full_val, 0.0, 0.0, 0.0
    r = _diagonal_r(c)
    if r is not None:
        r_val = float(0.5 * jnp.sum(r * (controls**2)))
        lin = _diagonal_linear_r(c)
        lin_val = float(jnp.sum(lin * controls)) if lin is not None else 0.0
        return 0.0, r_val, 0.0, full_val - r_val - lin_val, lin_val
    return 0.0, 0.0, 0.0, full_val, 0.0


def decompose_cost(
    problem: Problem,
    states: jax.Array,
    controls: jax.Array,
    t: float = 0.0,
    dt: float = 1.0,
) -> dict[str, Any]:
    """Decompose the total optimal control cost into spectral, effort, and tracking contributions.

    Parameters
    ----------
    problem
        The optimal control Problem containing the objective.
    states
        State trajectory of shape ``(N, n)``.
    controls
        Control trajectory of shape ``(N - 1, m)``.
    t
        Initial time of the trajectory.
    dt
        Sampling period.

    Returns
    -------
    dict[str, Any]
        Dictionary containing ``cost_spectral``, ``cost_quadratic_effort``,
        ``cost_sparse_effort``, ``cost_tracking``, ``cost_active``, ``active_count``,
        ``cost_total``, and ``normalization``.
    """
    N = states.shape[0]
    t_stage = t + jnp.arange(N - 1) * dt
    t_term = t + (N - 1) * dt

    spectral_val = 0.0
    quadratic_effort_val = 0.0
    sparse_effort_val = 0.0
    tracking_val = 0.0
    active_val = 0.0

    for c in _collect_leaf_costs(problem.obj.stage_cost):
        s, qe, se, tr, act = _stage_leaf_contributions(c, states, controls, t_stage)
        spectral_val += s
        quadratic_effort_val += qe
        sparse_effort_val += se
        tracking_val += tr
        active_val += act

    for c in _collect_leaf_costs(problem.obj.terminal_cost):
        term_val = float(c.evaluate(states[-1], None, t_term))
        if _is_spectral_cost(c):
            spectral_val += term_val
        else:
            tracking_val += term_val

    cost_total = spectral_val + quadratic_effort_val + sparse_effort_val + tracking_val + active_val
    if problem.binary_control_indices:
        active_count = float(np.sum(np.round(np.asarray(controls)[:, problem.binary_control_indices])))
    else:
        ctrls_arr = np.asarray(controls)
        active_count = float(np.sum(np.any(np.abs(ctrls_arr) >= ACTIVE_CURRENT_THRESHOLD, axis=-1)))

    return {
        "cost_spectral": spectral_val,
        "cost_quadratic_effort": quadratic_effort_val,
        "cost_sparse_effort": sparse_effort_val,
        "cost_tracking": tracking_val,
        "cost_active": active_val,
        "active_count": active_count,
        "cost_total": cost_total,
        "normalization": "channel_mean",
    }


def _rollout_states(model: DiscreteDynamics, x0: jax.Array, U: jax.Array, dt: float) -> jax.Array:
    """Forward-simulate DiscreteDynamics states from an initial state and control trajectory."""

    def step(x: jax.Array, u: jax.Array) -> tuple[jax.Array, jax.Array]:
        x_next = model.discrete_dynamics(x, u, 0.0, dt)
        return x_next, x_next

    _, xs = jax.lax.scan(step, x0, U)
    return jnp.concatenate([x0[None, :], xs], axis=0)


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
        init_mode: str = "shift",
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
        init_mode
            Warm-start initialization mode: ``"shift"`` (default receding horizon) or ``"cont_polish"``.
        """
        super().__init__(dt)
        model = problem.model
        if not isinstance(model, InferencePredictor):
            msg = f"problem.model ({type(model).__name__}) does not implement the InferencePredictor priming seam"
            raise TypeError(msg)
        self.problem = problem
        self.model = model
        self.solver = _build_solver(solver, problem)
        self.horizon = int(problem.N - 1)
        self._integer_mode = bool(problem.binary_control_indices)
        self.init_mode = str(init_mode)

        # The unprimed EEG window is NaN padding; the driver seeds a full state trajectory from
        # x0, not just the controls, so it is zeroed to keep that seed finite for every solver.
        unprimed = jnp.nan_to_num(jnp.asarray(self.model.initial_state()), nan=0.0)
        solver_driver = (
            self.solver
            if self._integer_mode or problem.constraints.horizon_constraints
            else canonicalize_duals(self.solver)
        )
        self.mpc = MPC(problem, solver_driver, x0=unprimed)
        self._state = np.asarray(self.model.initial_state(), dtype=np.float64)
        self._u_last = np.zeros(model.m, dtype=np.float64)
        # Under reduction the decision variable is ``v``, one shorter than the montage; the Plant
        # takes electrode currents, so the basis is kept here to expand at the boundary.
        self._basis = np.asarray(model.basis, dtype=np.float64) if isinstance(model, NullspaceReducedModel) else None
        if self._integer_mode:
            self.n_controls = int(self.model.n_controls)
            self.n_electrodes = self.n_controls
        else:
            self.n_controls = int(model.m)
            self.n_electrodes = model.m + 1 if self._basis is not None else model.m
        self._u_guess = np.zeros((self.horizon, self.n_controls), dtype=np.float64)
        self._active_guess = np.zeros(self.horizon, dtype=np.float64) if self._integer_mode else None

        cont_prob = getattr(problem, "continuous_problem", None)
        if self._integer_mode and self.init_mode == "cont_polish" and cont_prob is not None:
            self._cont_mpc: MPC | None = MPC(
                cont_prob, canonicalize_duals(SingleShooting(solver=Ipopt(options=IPOPT_DEFAULTS))), x0=unprimed
            )
            self._cont_u_max = float(getattr(problem, "u_max", [2.0])[0])
            self._max_cap = getattr(problem, "max_active_intervals", None)
        else:
            self._cont_mpc = None

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
            init_mode=str(config.get("init_mode", "shift")),
        )

    @property
    def is_integer_mode(self) -> bool:
        """Whether this controller solves mixed-integer activation decisions."""
        return self._integer_mode

    def update(
        self,
        t: float,
        ref: FloatArray,  # noqa: ARG002 -- the goal is baked into the objective
        x_hat: FloatArray,
    ) -> tuple[FloatArray, TrajOptMPCLog]:
        """Absorb the measurement and record the solved plan and Cost contributions before shifting.

        The emitted control is always the ``(n_electrodes,)`` physical currents: under a Nullspace
        Frame the solver decides in the reduced ``v``, which is expanded through ``Z`` here.
        """
        self._state = np.asarray(self.model.absorb(self._state, np.asarray(x_hat).reshape(-1), self._u_last))

        if not self.model.is_ready(self._state):
            self._u_last = np.zeros(self.model.m, dtype=np.float64)
            u_zero = np.zeros(self.n_electrodes, dtype=np.float64)
            knots = self.mpc.problem.N
            planned_active = np.full(knots - 1, np.nan, dtype=np.float64)
            active_count = float("nan")
            return u_zero, TrajOptMPCLog(
                u=u_zero,
                cost=0.0,
                success=True,
                warmup=True,
                status="warmup",
                solve_time=0.0,
                predicted_y=np.full((knots, self.model.p), np.nan),
                planned_u=np.full((knots - 1, self.n_electrodes), np.nan),
                planned_active=planned_active,
                active_count=active_count,
                cost_active=0.0,
                cost_spectral=0.0,
                cost_quadratic_effort=0.0,
                cost_sparse_effort=0.0,
                cost_tracking=0.0,
                normalization="channel_mean",
            )

        self.mpc.measure(jnp.asarray(self._state), t)
        if self._cont_mpc is not None:
            self._cont_mpc.measure(jnp.asarray(self._state), t)
            res_cont = self._cont_mpc.solve()
            self._cont_mpc.shift(self.dt)
            u_cont = np.asarray(res_cont.trajectory.U)
            envelope = np.max(np.abs(u_cont), axis=1) / self._cont_u_max
            delta_thresh = np.where(envelope > _CONTINUOUS_POLISH_THRESHOLD, 1.0, 0.0)
            if self._max_cap is not None and np.sum(delta_thresh) > self._max_cap:
                top_k = np.argsort(envelope)[-self._max_cap :]
                delta_thresh = np.zeros(self.horizon)
                delta_thresh[top_k] = 1.0
            u_full = np.hstack([u_cont, delta_thresh[:, None]])
            u_full_jax = jnp.asarray(u_full, dtype=jnp.float64)
            x0_arr = jnp.asarray(self._state, dtype=jnp.float64)
            x_rollout = _rollout_states(self.problem.model, x0_arr, u_full_jax, float(self.problem.dt[0]))
            self.mpc._ws = self.mpc._ws.with_primal(  # noqa: SLF001 -- inject continuous relaxation primal guess into driver warm start
                self.problem, X=x_rollout, U=u_full_jax
            )
        started = time.perf_counter()
        solved = self.mpc.solve()
        solve_time = time.perf_counter() - started
        u_solved = np.asarray(self.mpc.controls[0], dtype=np.float64)
        cost = float(self.mpc.cost())
        predicted_y = np.asarray(planned_outputs(self.model, self.mpc.states, self.mpc.controls, t, self.dt)).copy()
        plan = np.asarray(self.mpc.controls)
        if self._integer_mode:
            planned_u = plan[:, :-1].copy()
            planned_active = plan[:, -1].copy()
            u_cmd = planned_u[0].copy() if solved.success else np.zeros(self.n_electrodes, dtype=np.float64)
        else:
            planned_u = (plan if self._basis is None else plan @ self._basis.T).copy()
            planned_active = np.any(np.abs(planned_u) >= ACTIVE_CURRENT_THRESHOLD, axis=-1).astype(np.float64)
            u_cmd = u_solved if self._basis is None else self._basis @ u_solved
        costs_decomp = decompose_cost(self.mpc.problem, self.mpc.states, self.mpc.controls, t, self.dt)
        self.mpc.shift(self.dt)
        # ``_u_last`` feeds ``model.absorb``, which expands for itself, so the state keeps the
        # solver's own coordinates while the Plant and the log get the electrode currents.
        self._u_last = u_solved
        if solved.success:
            self._u_guess = np.vstack((planned_u[1:], planned_u[-1:]))
            if self._integer_mode and planned_active is not None:
                self._active_guess = np.concatenate((planned_active[1:], planned_active[-1:]))
        return u_cmd, TrajOptMPCLog(
            u=u_cmd.copy(),
            cost=cost,
            success=bool(solved.success),
            warmup=False,
            status=str(solved.status),
            solve_time=solve_time,
            predicted_y=predicted_y,
            planned_u=planned_u,
            planned_active=planned_active,
            active_count=costs_decomp["active_count"],
            cost_active=costs_decomp["cost_active"],
            cost_spectral=costs_decomp["cost_spectral"],
            cost_quadratic_effort=costs_decomp["cost_quadratic_effort"],
            cost_sparse_effort=costs_decomp["cost_sparse_effort"],
            cost_tracking=costs_decomp["cost_tracking"],
            normalization=costs_decomp["normalization"],
        )

    def solve_state(
        self,
        x0: FloatArray,
        initial_u: FloatArray | None = None,
        initial_active: FloatArray | None = None,
        time_limit_s: float | None = None,  # noqa: ARG002 -- interface compatibility with MINLP controllers
    ) -> TrajOptMPCLog:
        """Solve a fixed Predictor state with an optional primal guess, without advancing controller history.

        Parameters
        ----------
        x0
            Predictor state vector of shape ``(n,)``.
        initial_u
            Control guess of shape ``(horizon, n_controls)``.
        initial_active
            Binary activation guess of shape ``(horizon,)`` for integer mode.
        time_limit_s
            Optional solver execution time limit in seconds.
        """
        x0_arr = jnp.asarray(x0, dtype=jnp.float64).reshape(self.model.n)
        bc = BoundaryConditions(x0=x0_arr, t0=jnp.asarray(0.0, dtype=jnp.float64))
        if initial_u is not None:
            u_init = np.asarray(initial_u, dtype=np.float64).reshape(self.horizon, -1)
            if self._integer_mode:
                if initial_active is not None:
                    act_init = np.asarray(initial_active, dtype=np.float64).reshape(self.horizon, 1)
                else:
                    act_init = (np.any(u_init != 0, axis=1, keepdims=True)).astype(np.float64)
                u_full = np.hstack([u_init, act_init])
            elif self._basis is not None and u_init.shape[1] == self.n_electrodes:
                u_full = u_init @ self._basis
            else:
                u_full = u_init
            ws = WarmStart.cold(self.problem, x0_arr)
            u_full_jax = jnp.asarray(u_full, dtype=jnp.float64)
            x_rollout = _rollout_states(self.problem.model, x0_arr, u_full_jax, float(self.problem.dt[0]))
            ws = ws.with_primal(self.problem, X=x_rollout, U=u_full_jax)
        else:
            ws = WarmStart.cold(self.problem, x0_arr)

        started = time.perf_counter()
        try:
            res = self.mpc.program.solve(bc, ws)
            solve_time = time.perf_counter() - started
            success = bool(res.success)
            status = str(res.status)
        except Exception as exc:  # noqa: BLE001 -- report solver failure cleanly
            solve_time = time.perf_counter() - started
            success = False
            status = str(exc)
            res = None

        if res is None:
            planned_u = np.full((self.horizon, self.n_controls), np.nan)
            planned_active = np.full(self.horizon, np.nan, dtype=np.float64)
            predicted_y = np.full((self.horizon + 1, self.model.p), np.nan)
            u_cmd = np.zeros(self.n_controls, dtype=np.float64)
            return TrajOptMPCLog(
                u=u_cmd,
                cost=float("inf"),
                success=False,
                warmup=False,
                status=status,
                solve_time=solve_time,
                predicted_y=predicted_y,
                planned_u=planned_u,
                planned_active=planned_active,
                active_count=float("nan"),
                cost_active=0.0,
                cost_spectral=0.0,
                cost_quadratic_effort=0.0,
                cost_sparse_effort=0.0,
                cost_tracking=0.0,
                normalization="channel_mean",
            )

        plan = np.asarray(res.trajectory.U)
        if self._integer_mode:
            planned_u = plan[:, :-1].copy()
            planned_active = plan[:, -1].copy()
            u_cmd = planned_u[0].copy() if success else np.zeros(self.n_controls, dtype=np.float64)
        else:
            planned_u = (plan if self._basis is None else plan @ self._basis.T).copy()
            planned_active = np.any(np.abs(planned_u) >= ACTIVE_CURRENT_THRESHOLD, axis=-1).astype(np.float64)
            u_cmd = planned_u[0].copy() if success else np.zeros(self.n_controls, dtype=np.float64)

        predicted_y = np.asarray(planned_outputs(self.model, res.trajectory.X, res.trajectory.U, 0.0, self.dt)).copy()
        costs_decomp = decompose_cost(self.problem, res.trajectory.X, res.trajectory.U, 0.0, self.dt)
        return TrajOptMPCLog(
            u=u_cmd,
            cost=float(res.cost),
            success=success,
            warmup=False,
            status=status,
            solve_time=solve_time,
            predicted_y=predicted_y,
            planned_u=planned_u,
            planned_active=planned_active,
            active_count=costs_decomp["active_count"],
            cost_active=costs_decomp["cost_active"],
            cost_spectral=costs_decomp["cost_spectral"],
            cost_quadratic_effort=costs_decomp["cost_quadratic_effort"],
            cost_sparse_effort=costs_decomp["cost_sparse_effort"],
            cost_tracking=costs_decomp["cost_tracking"],
            normalization=costs_decomp["normalization"],
        )


class _CandidateCallback:
    """Callback for single-shooting IPOPT evaluation of a fixed candidate schedule."""

    def __init__(
        self,
        problem: Problem,
        x0_p: jax.Array,
        t0_arr: jax.Array,
        dt_arr: jax.Array,
        xf_val: jax.Array | None,
    ) -> None:
        self.problem = problem
        self.x0_p = x0_p
        self.t0_arr = t0_arr
        self.dt_arr = dt_arr
        self.xf_val = xf_val
        n_u, p_user = single_shooting_dimensions(problem)
        self.jac_rows, self.jac_cols = _dense_jacobian_pattern(p_user, n_u)

    def objective(self, u: np.ndarray) -> float:
        """Evaluate single-shooting objective for candidate currents."""
        return float(eval_f(self.problem, jnp.asarray(u), self.x0_p, t0=self.t0_arr, dt=self.dt_arr))

    def gradient(self, u: np.ndarray) -> np.ndarray:
        """Evaluate objective gradient nabla J(u)."""
        return np.asarray(
            eval_grad_f(self.problem, jnp.asarray(u), self.x0_p, t0=self.t0_arr, dt=self.dt_arr),
            dtype=np.float64,
        )

    def constraints(self, u: np.ndarray) -> np.ndarray:
        """Evaluate constraint residual vector c(u)."""
        return np.asarray(
            eval_g(self.problem, jnp.asarray(u), self.x0_p, t0=self.t0_arr, dt=self.dt_arr, xf=self.xf_val),
            dtype=np.float64,
        )

    def jacobian(self, u: np.ndarray) -> np.ndarray:
        """Evaluate constraint Jacobian values."""
        return np.asarray(
            eval_jac_g(self.problem, jnp.asarray(u), self.x0_p, t0=self.t0_arr, dt=self.dt_arr, xf=self.xf_val),
            dtype=np.float64,
        )

    def jacobianstructure(self) -> tuple[np.ndarray, np.ndarray]:
        """Return constraint Jacobian sparsity pattern."""
        return self.jac_rows, self.jac_cols

    def intermediate(self, *args: object) -> bool:  # noqa: ARG002 -- cyipopt iteration callback protocol
        """Accept iteration progress unconditionally."""
        return True


class CandidateMPCController(Controller[TrajOptMPCLog]):
    """Combinatorial control-blocking MPC controller evaluating discrete candidate schedules."""

    def __init__(
        self,
        dt: float,
        problem: Problem,
        blocks: list[list[int]] | None = None,
        w_active: float = 0.3,
    ) -> None:
        """Initialize candidate controller and compile schedule permutations.

        Parameters
        ----------
        dt
            Controller update step in seconds.
        problem
            Continuous optimal-control problem (without binary variables).
        blocks
            List of Control Horizon knot indices partitioned into decision blocks.
        w_active
            Sparsity weight on active Control Horizon intervals.
        """
        super().__init__(dt)
        model = problem.model
        if not isinstance(model, InferencePredictor):
            msg = f"problem.model ({type(model).__name__}) does not implement InferencePredictor"
            raise TypeError(msg)
        self.problem = problem
        self.model = model
        self.horizon = int(problem.N - 1)
        self.blocks = blocks if blocks is not None else [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]
        self.w_active = float(w_active)
        self.schedules = self._build_schedules(self.blocks, self.horizon)
        self._state = np.asarray(self.model.initial_state(), dtype=np.float64)
        self._u_last = np.zeros(self.model.m, dtype=np.float64)
        self._basis = np.asarray(model.basis, dtype=np.float64) if isinstance(model, NullspaceReducedModel) else None
        self.n_controls = int(self.model.m)
        self.n_electrodes = self.model.m + 1 if self._basis is not None else self.model.m
        self._prev_plan_u: np.ndarray | None = None

    @staticmethod
    def _build_schedules(blocks: list[list[int]], horizon: int) -> list[np.ndarray]:
        """Generate all 2^B candidate binary schedules from block definitions."""
        n_blocks = len(blocks)
        schedules: list[np.ndarray] = []
        for code in range(2**n_blocks):
            bits = [(code >> (n_blocks - 1 - b)) & 1 for b in range(n_blocks)]
            sched = np.zeros(horizon, dtype=int)
            for b, bit in enumerate(bits):
                for idx in blocks[b]:
                    if idx < horizon:
                        sched[idx] = bit
            schedules.append(sched)
        return schedules

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> CandidateMPCController:
        """Instantiate CandidateMPCController from a configuration dictionary."""
        problem = _build_problem(config["problem"])
        blocks = config.get("blocks")
        w_active = float(config.get("w_active", 0.3))
        return cls(float(config["dt"]), problem, blocks=blocks, w_active=w_active)

    def _evaluate_candidates(
        self,
        cb: _CandidateCallback,
        bounds: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        m_dim: int,
    ) -> tuple[float, np.ndarray, np.ndarray, int, bool]:
        """Evaluate candidate binary schedules via bounded single shooting."""
        import cyipopt  # noqa: PLC0415 -- cyipopt is an optional solver dependency

        z_lower, z_upper, g_lower, g_upper = bounds
        problem_cls: Any = getattr(cyipopt, "Problem")  # noqa: B009 -- cyipopt Problem
        best_cost = float("inf")
        best_sol = np.zeros(len(z_lower))
        best_sched = self.schedules[0]
        best_status = -1
        best_success = False

        for sched in self.schedules:
            act_sum = int(np.sum(sched))
            sparsity_cost = self.w_active * act_sum / self.horizon

            if act_sum == 0:
                sol = np.zeros(len(z_lower))
                raw_obj = cb.objective(sol)
                unified_cost = raw_obj + sparsity_cost
                status = 0
                success = True
            else:
                lb = z_lower.copy()
                ub = z_upper.copy()
                for k, act in enumerate(sched):
                    if act == 0:
                        lb[k * m_dim : (k + 1) * m_dim] = 0.0
                        ub[k * m_dim : (k + 1) * m_dim] = 0.0

                nlp = problem_cls(
                    n=len(lb),
                    m=len(g_lower),
                    problem_obj=cb,
                    lb=lb,
                    ub=ub,
                    cl=g_lower,
                    cu=g_upper,
                )
                nlp.add_option("print_level", 0)
                nlp.add_option("hessian_approximation", "limited-memory")
                nlp.add_option("tol", 1e-4)

                init_guess = self._prev_plan_u.reshape(-1) if self._prev_plan_u is not None else np.zeros(len(lb))
                init_guess = np.clip(init_guess, lb, ub)
                sol, info = nlp.solve(init_guess)
                raw_obj = cb.objective(sol)
                unified_cost = raw_obj + sparsity_cost
                status = int(info.get("status", -1))
                success = status in _SUCCESS_STATUSES

            if success and unified_cost < best_cost:
                best_cost = unified_cost
                best_sol = sol
                best_sched = sched
                best_status = status
                best_success = True

        return best_cost, best_sol, best_sched, best_status, best_success

    def update(
        self,
        t: float,
        ref: FloatArray,  # noqa: ARG002 -- goal is baked into objective
        x_hat: FloatArray,
    ) -> tuple[FloatArray, TrajOptMPCLog]:
        """Absorb measurement, evaluate candidate schedules, and apply the best first-step current."""
        self._state = np.asarray(self.model.absorb(self._state, np.asarray(x_hat).reshape(-1), self._u_last))
        knots = self.problem.N
        m_dim = self.n_controls

        if not self.model.is_ready(self._state):
            self._u_last = np.zeros(m_dim, dtype=np.float64)
            u_zero = np.zeros(self.n_electrodes, dtype=np.float64)
            return u_zero, TrajOptMPCLog(
                u=u_zero,
                cost=0.0,
                success=True,
                warmup=True,
                status="warmup",
                solve_time=0.0,
                predicted_y=np.full((knots, self.model.p), np.nan),
                planned_u=np.full((knots - 1, self.n_electrodes), np.nan),
                planned_active=np.full(knots - 1, np.nan),
                active_count=float("nan"),
                cost_active=0.0,
                cost_spectral=0.0,
                cost_quadratic_effort=0.0,
                cost_sparse_effort=0.0,
                cost_tracking=0.0,
                normalization="channel_mean",
            )

        started = time.perf_counter()
        x0_arr = jnp.asarray(self._state, dtype=jnp.float64)
        bc = BoundaryConditions(x0=x0_arr, t0=jnp.asarray(t, dtype=jnp.float64))
        ws = WarmStart.cold(self.problem, x0_arr)
        x0_p, t0_arr, dt_arr, xf_val, _ = parse_solver_initial_state(self.problem, bc, ws)
        problem_ret = retarget_problem(self.problem, bc)
        dt_arr = jnp.broadcast_to(dt_arr, (self.horizon,))

        z_lower, z_upper = _primal_bounds(problem_ret)
        g_lower, g_upper = _constraint_bounds(problem_ret)
        cb = _CandidateCallback(problem_ret, x0_p, t0_arr, dt_arr, xf_val)

        bounds = (z_lower, z_upper, g_lower, g_upper)
        best_cost, best_sol, best_sched, best_status, best_success = self._evaluate_candidates(cb, bounds, m_dim)

        solve_time = time.perf_counter() - started
        plan_u_red = best_sol.reshape(self.horizon, m_dim)
        if self._basis is not None:
            plan_u_phys = plan_u_red @ self._basis.T
            u_cmd = self._basis @ plan_u_red[0]
        else:
            plan_u_phys = plan_u_red
            u_cmd = plan_u_red[0]

        self._u_last = plan_u_red[0]
        self._prev_plan_u = np.vstack((plan_u_red[1:], plan_u_red[-1:]))

        u_jax = jnp.asarray(plan_u_red, dtype=jnp.float64)
        x_rollout = _rollout_states(self.problem.model, x0_p, u_jax, float(self.problem.dt[0]))
        predicted_y = np.asarray(planned_outputs(self.model, x_rollout, u_jax, t, self.dt)).copy()
        costs_decomp = decompose_cost(self.problem, x_rollout, u_jax, t, self.dt)

        return u_cmd, TrajOptMPCLog(
            u=u_cmd.copy(),
            cost=best_cost,
            success=best_success,
            warmup=False,
            status=str(best_status),
            solve_time=solve_time,
            predicted_y=predicted_y,
            planned_u=plan_u_phys,
            planned_active=best_sched.astype(np.float64),
            active_count=float(np.sum(best_sched)),
            cost_active=self.w_active * float(np.sum(best_sched)) / self.horizon,
            cost_spectral=costs_decomp["cost_spectral"],
            cost_quadratic_effort=costs_decomp["cost_quadratic_effort"],
            cost_sparse_effort=costs_decomp["cost_sparse_effort"],
            cost_tracking=costs_decomp["cost_tracking"],
            normalization="channel_mean",
        )
