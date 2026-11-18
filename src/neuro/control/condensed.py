from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import trajopt  # noqa: F401 -- registers the Windows IPOPT DLL directory before importing cyipopt
from trajopt.cones import NegativeOrthant, ZeroCone
from trajopt.constraints.linear import LinearConstraint
from trajopt.costs.quadratic import DiagonalCost
from trajopt.costs.sum import cost_terms
from trajopt.problem import _EpigraphModel
from trajopt.trajectory import Trajectory
from trajopt.transcription.homotopy import (
    _SUCCESS_STATUSES,
    solve_continuation_passes,
    validate_homotopy_parameters,
)
from trajopt.transcription.ipopt import Ipopt, IpoptResult
from trajopt.transcription.layout import _trajectory_to_z
from trajopt.transcription.osqp import OSQPResult
from trajopt.transcription.penalties import IndexedConstraint, _PenaltyObjective

from neuro.control.costs import AffineObservableHinge, CondensedL1ControlCost
from neuro.control.diagnostics import _rollout_states
from neuro.control.model_wrappers import AugmentedActivationModel
from neuro.predictor.inference import ObservableMLPModel

if TYPE_CHECKING:
    from collections.abc import Callable

    import osqp
    from trajopt.dynamics.base import DiscreteDynamics
    from trajopt.problem import BoundaryConditions, Problem
    from trajopt.program import Program, WarmStart

    from neuro.types import BoolArray, FloatArray, IntArray


@dataclass(frozen=True)
class _CondensedObjective:
    """Validated physical Predictor, Cost terms, and effort metadata for condensed solvers."""

    base_model: ObservableMLPModel
    effort: DiagonalCost
    hinge: AffineObservableHinge
    sparse_weight: float
    active_weight: float
    integer_mode: bool


def _validate_objective(problem: Problem, *, allow_binary: bool = False) -> _CondensedObjective:
    """Extract the base Predictor, affine hinge, and physical Control Current Costs."""
    obj = problem.obj.physical if allow_binary and isinstance(problem.obj, _PenaltyObjective) else problem.obj
    model = problem.model.original if allow_binary and isinstance(problem.model, _EpigraphModel) else problem.model
    terms = cost_terms(obj.stage_cost)
    sparse = next((term for term in terms if isinstance(term, CondensedL1ControlCost)), None)
    physical_terms = tuple(term for term in terms if term is not sparse)
    terminal = obj.terminal_cost
    integer_mode = bool(problem.binary_control_indices)
    base_model = model.base_model if allow_binary and isinstance(model, AugmentedActivationModel) else model
    binary_ok = (
        (not integer_mode)
        or (
            allow_binary
            and isinstance(model, AugmentedActivationModel)
            and tuple(problem.binary_control_indices) == (model.m - 1,)
        )
    )
    if (
        binary_ok
        and isinstance(base_model, ObservableMLPModel)
        and len(base_model.weights) == 1
        and len(physical_terms) == 2  # noqa: PLR2004 -- one quadratic term and one hinge
        and isinstance(terminal, DiagonalCost)
        and not np.any(terminal.Q)
        and not np.any(terminal.q)
        and not np.any(terminal.c)
    ):
        effort, hinge = physical_terms
        if isinstance(effort, DiagonalCost) and isinstance(hinge, AffineObservableHinge):
            r_quad = np.asarray(effort.R, dtype=np.float64)
            r_lin = np.asarray(effort.r, dtype=np.float64)
            r_phys = r_lin[..., :-1] if integer_mode else r_lin
            active_slice = r_lin[..., -1]
            active_weight = float(np.mean(active_slice)) if integer_mode else 0.0
            r_gate_ok = (
                bool(
                    np.all(active_slice >= 0.0)
                    and np.allclose(active_slice, active_weight)
                    and not np.any(r_quad[..., -1])
                )
                if integer_mode
                else True
            )
            if r_gate_ok and (sparse is None or sparse.m_phys == base_model.m) and not (
                np.any(effort.Q) or np.any(effort.q) or np.any(effort.c) or np.any(r_phys) or np.any(r_quad < 0)
            ):
                return _CondensedObjective(
                    base_model=base_model,
                    effort=effort,
                    hinge=hinge,
                    sparse_weight=float(sparse.weight) if sparse is not None else 0.0,
                    active_weight=active_weight,
                    integer_mode=integer_mode,
                )
    msg = (
        "Condensed solver requires a depth-zero Observable MLP Problem with affine_hinge=True"
        " and convex Control Current effort"
    )
    raise ValueError(msg)


def condense_affine_observable(
    model: DiscreteDynamics, hinge: AffineObservableHinge, horizon: int, dt: float
) -> tuple[Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]], FloatArray]:
    """Return the affine residual Rollout and its fixed Jacobian in planned Control Currents."""
    zero_state = jnp.zeros(model.n, dtype=jnp.float64)
    zero_controls = jnp.zeros((horizon, model.m), dtype=jnp.float64)

    @jax.jit
    def evaluate(x0: jax.Array, controls: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return the states and flattened unrectified spectral residuals."""
        states = _rollout_states(model, x0, controls, dt)
        z = jnp.concatenate([states[:-1], controls], axis=1)
        return states, (z @ hinge.A.T - hinge.b).ravel()

    response = jax.jacfwd(lambda u: evaluate(zero_state, u)[1])(zero_controls)
    return evaluate, np.asarray(response).reshape(horizon * len(hinge.b), horizon * model.m)


def _l1_epigraph_blocks(hm: int, mid_cols: int) -> tuple[sp.spmatrix, FloatArray, FloatArray]:
    """Build exact L1 epigraph blocks ``[I, 0, -I]`` and ``[-I, 0, -I] <= 0``."""
    i_hm = sp.eye(hm, format="csc")
    mid_pad = sp.csc_matrix((hm, mid_cols))
    rows = sp.vstack((sp.hstack((i_hm, mid_pad, -i_hm)), sp.hstack((-i_hm, mid_pad, -i_hm))), format="csc")
    return rows, np.full(2 * hm, -np.inf), np.zeros(2 * hm)


def _extract_active_cap(problem: Problem, horizon: int, m_phys: int, *, integer_mode: bool) -> float | None:
    """Validate and return the optional active-interval Control Horizon cap."""
    horizon_constraints = problem.constraints.horizon_constraints
    if not horizon_constraints:
        return None
    if not integer_mode or len(horizon_constraints) != 1:
        msg = "CondensedIPOPT supports at most one active-interval cap Control Horizon constraint"
        raise ValueError(msg)
    cap_con = horizon_constraints[0]
    n, m = problem.model.n, problem.model.m
    expected_inds = tuple(k * (n + m) + n + m_phys for k in range(horizon))
    if (
        not isinstance(cap_con.cone, NegativeOrthant)
        or tuple(cap_con.inds) != expected_inds
        or not np.allclose(cap_con.A, np.ones((1, horizon)))
    ):
        msg = "CondensedIPOPT encountered an unsupported Control Horizon constraint"
        raise ValueError(msg)
    return float(cap_con.b[0])


def _extract_constraints(
    problem: Problem,
    horizon: int,
    m_phys: int,
    *,
    allow_binary: bool,
    solver_name: str,
) -> tuple[sp.csc_matrix, float | None]:
    """Validate linear Control Current constraints and extract Kirchhoff rows and active cap."""
    constraints = problem.constraints
    x_lower, x_upper, _, _ = constraints.primal_bounds()
    integer_mode = bool(problem.binary_control_indices)
    lifted_epigraph = allow_binary and isinstance(problem.model, _EpigraphModel)
    m_aug = m_phys + 1 if allow_binary and integer_mode else m_phys
    if (
        problem.control_block_steps != 1
        or (integer_mode and not allow_binary)
        or (not allow_binary and constraints.horizon_constraints)
        or np.any(np.isfinite(x_lower))
        or np.any(np.isfinite(x_upper))
    ):
        msg = (
            f"{solver_name} supports unblocked Control Currents with only box, Kirchhoff, and activation constraints"
            if allow_binary
            else f"{solver_name} supports continuous unblocked Control Currents with only boxes and Kirchhoff equalities"
        )
        raise ValueError(msg)

    n = problem.model.n
    expected_kirchhoff = np.zeros((1, m_aug), dtype=np.float64)
    expected_kirchhoff[0, :m_phys] = 1.0
    kirchhoff_steps: list[int] = []
    for k, evaluator in enumerate(constraints.knot_evaluators):
        for row in evaluator.constraints:
            if lifted_epigraph and isinstance(row, IndexedConstraint):
                if max(row.iu, default=-1) >= m_aug:
                    continue
                con = row.constraint
            else:
                con = row
            if k == problem.N - 1 or not isinstance(con, LinearConstraint) or con.inds != tuple(range(n, n + m_aug)):
                msg = (
                    f"{solver_name} supports only linear Control Current and gate constraints at Control Horizon knots"
                    if allow_binary
                    else f"{solver_name} supports only Kirchhoff equalities at Control Horizon knots"
                )
                raise ValueError(msg)
            if isinstance(con.cone, ZeroCone):
                if not np.allclose(con.A, expected_kirchhoff) or not np.array_equal(con.b, np.zeros(1)):
                    msg = f"{solver_name} supports only Kirchhoff equalities at Control Horizon knots"
                    raise ValueError(msg)
                kirchhoff_steps.append(k)
            elif (
                allow_binary
                and integer_mode
                and isinstance(con.cone, NegativeOrthant)
                and con.A.shape == (2 * m_phys, m_aug)
            ):
                continue
            else:
                msg = f"{solver_name} encountered an unsupported knot constraint"
                raise ValueError(msg)

    kirchhoff = sp.kron(sp.eye(horizon), np.ones((1, m_phys)), format="csr")[kirchhoff_steps].tocsc()
    active_cap = _extract_active_cap(problem, horizon, m_phys, integer_mode=integer_mode) if allow_binary else None
    return kirchhoff, active_cap


@dataclass
class _CondensedHandle:
    """One prepared OSQP factorization and its affine residual Rollout."""

    solver: osqp.OSQP
    evaluate: Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]
    upper: FloatArray
    lower: FloatArray
    matrix: sp.csc_matrix
    response: FloatArray
    objective: Callable[[Trajectory], tuple[jax.Array, jax.Array]]
    current_rows: slice
    sparse_weight: float


def _evaluate_at_boundary(
    solver: CondensedOSQP | CondensedIPOPT,
    program: Program,
    bc: BoundaryConditions,
    handle_key: str,
) -> tuple[Any, jax.Array, FloatArray]:
    """Evaluate and validate the zero-Control Current Rollout for the measured state."""
    solver_name = type(solver).__name__
    if bc.X_ref is not None or bc.U_ref is not None or bc.xf is not None:
        msg = f"{solver_name} requires a fixed Objective and no terminal goal"
        raise ValueError(msg)
    if handle_key not in program.handles:
        solver.prepare(program)
    handle = program.handles[handle_key]
    problem = program.problem
    zero = jnp.zeros((problem.N - 1, problem.model.m), dtype=jnp.float64)
    states, baseline = handle.evaluate(bc.x0, zero)
    baseline_np = np.asarray(baseline)
    if not np.isfinite(baseline_np).all():
        msg = f"{solver_name} requires finite predicted Frame residuals"
        raise ValueError(msg)
    return handle, states, baseline_np


def _zero_cost(
    program: Program,
    bc: BoundaryConditions,
    zero_states: jax.Array,
    objective: Callable[[Trajectory], tuple[jax.Array, jax.Array]],
) -> float:
    """Compute the unstimulated spectral Objective Cost at zero Control Current."""
    problem = program.problem
    zero = jnp.zeros((problem.N - 1, problem.model.m), dtype=jnp.float64)
    times = bc.t0 + jnp.r_[0.0, jnp.cumsum(problem.dt)]
    return float(objective(Trajectory(X=zero_states, U=zero, t=times, dt=problem.dt))[0])


def _select_condensed_schedule(
    schedules: list[IntArray],
    weight: float,
    zero_shape: tuple[int, int],
    zero_cost: float,
    solve_step: Callable[[BoolArray], OSQPResult | IpoptResult],
) -> tuple[FloatArray, IntArray, int, bool]:
    """Price all-off directly and select the lowest penalized fixed schedule."""
    best_plan = np.zeros(zero_shape, dtype=np.float64)
    best_schedule = schedules[0]
    best_cost = zero_cost
    status = 0
    solved_any = False
    for schedule in schedules[1:]:
        solved = solve_step(schedule.astype(bool))
        if not solved.success:
            continue
        solved_any = True
        score = float(solved.cost) + weight * float(schedule.mean())
        if score < best_cost:
            best_cost, best_schedule, status = score, schedule, solved.status
            best_plan = np.asarray(solved.trajectory.U)
    return best_plan.ravel(), best_schedule, status, solved_any


class CondensedOSQP:
    """Solve the affine Observable Objective in Control Currents and epigraph slacks."""

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        """Set OSQP options with the campaign's replay tolerances."""
        self.options = {"verbose": False, "eps_abs": 1e-4, "eps_rel": 1e-4, **(options or {})}

    def prepare(self, program: Program) -> None:
        """Compile the Rollout and allocate the fixed QP before the first control release."""
        import osqp  # noqa: PLC0415 -- import the backend only when selected

        problem = program.problem
        spec = _validate_objective(problem)
        model = problem.model
        horizon = problem.N - 1
        kirchhoff, _ = _extract_constraints(
            problem, horizon, model.m, allow_binary=False, solver_name=type(self).__name__
        )
        currents = horizon * model.m
        residuals = horizon * len(spec.hinge.b)
        zero_state = jnp.zeros(model.n, dtype=jnp.float64)
        zero_controls = jnp.zeros((horizon, model.m), dtype=jnp.float64)
        dt = float(problem.dt[0])

        evaluate, response = condense_affine_observable(model, spec.hinge, horizon, dt)
        zero_states, zero_residual = evaluate(zero_state, zero_controls)
        baseline = np.asarray(zero_residual)
        slack_identity = sp.eye(residuals)
        matrix = sp.bmat(
            [
                [sp.csc_matrix(response), -slack_identity],
                [None, slack_identity],
                [sp.eye(currents), None],
                [kirchhoff, None],
            ],
            format="csc",
        )
        _, _, u_lower, u_upper = problem.constraints.primal_bounds()
        lower = np.r_[np.full(residuals, -np.inf), np.zeros(residuals), u_lower.ravel(), np.zeros(kirchhoff.shape[0])]
        upper = np.r_[-baseline, np.full(residuals, np.inf), u_upper.ravel(), np.zeros(kirchhoff.shape[0])]
        hessian = sp.diags(
            np.r_[np.asarray(spec.effort.R).ravel(), np.tile(2 * np.asarray(spec.hinge.weight), horizon)],
            format="csc",
        )
        linear = np.zeros(currents + residuals)
        if spec.sparse_weight:
            l1_rows, l1_cl, l1_cu = _l1_epigraph_blocks(currents, residuals)
            matrix = sp.vstack(
                [sp.hstack((matrix, sp.csc_matrix((matrix.shape[0], currents)))), l1_rows],
                format="csc",
            )
            hessian = sp.block_diag((hessian, sp.csc_matrix((currents, currents))), format="csc")
            lower = np.r_[lower, l1_cl]
            upper = np.r_[upper, l1_cu]
            linear = np.r_[linear, np.full(currents, spec.sparse_weight)]
        solver = osqp.OSQP()
        solver.setup(P=hessian, q=linear, A=matrix, l=lower, u=upper, **self.options)

        @jax.jit
        def objective(trajectory: Trajectory) -> tuple[jax.Array, jax.Array]:
            """Evaluate physical spectral and Control Current effort subtotals."""
            inputs = trajectory.X[:-1], trajectory.U, trajectory.t[:-1]
            return jnp.sum(spec.hinge.stage_costs(*inputs)), jnp.sum(spec.effort.stage_costs(*inputs))

        jax.block_until_ready(
            objective(
                Trajectory(
                    X=zero_states,
                    U=zero_controls,
                    t=jnp.r_[0.0, jnp.cumsum(problem.dt)],
                    dt=problem.dt,
                )
            )
        )
        program.handles["condensed_osqp"] = _CondensedHandle(
            solver,
            evaluate,
            upper,
            lower,
            matrix,
            response,
            objective,
            slice(2 * residuals, 2 * residuals + currents),
            spec.sparse_weight,
        )

    def solve(
        self, program: Program, bc: BoundaryConditions, ws: WarmStart, *, active: BoolArray | None = None
    ) -> OSQPResult:
        """Update the envelope and optional fixed schedule, then return a physical Trajectory."""
        handle, _, baseline = _evaluate_at_boundary(self, program, bc, "condensed_osqp")
        return self._solve_at_boundary(program, bc, ws, handle, baseline, active=active)

    def select_schedule(
        self, program: Program, bc: BoundaryConditions, ws: WarmStart, schedules: list[IntArray], weight: float
    ) -> tuple[FloatArray, IntArray, int, bool]:
        """Price all-off directly and reuse one measured free response across fixed schedules."""
        handle, zero_states, baseline = _evaluate_at_boundary(self, program, bc, "condensed_osqp")
        problem = program.problem
        return _select_condensed_schedule(
            schedules,
            weight,
            (problem.N - 1, problem.model.m),
            _zero_cost(program, bc, zero_states, handle.objective),
            lambda active: self._solve_at_boundary(program, bc, ws, handle, baseline, active=active),
        )

    def _solve_at_boundary(  # noqa: PLR0913 -- share the prepared handle and residual across fixed schedules
        self,
        program: Program,
        bc: BoundaryConditions,
        ws: WarmStart,
        handle: _CondensedHandle,
        baseline_np: FloatArray,
        *,
        active: BoolArray | None,
    ) -> OSQPResult:
        """Solve a fixed schedule at an already evaluated measured state."""
        problem = program.problem
        _, seed = ws.unpack(problem)
        rows = baseline_np.size
        lower, upper = handle.lower.copy(), handle.upper.copy()
        upper[:rows] = -baseline_np
        if active is not None:
            disabled = np.repeat(~active, problem.model.m)
            lower[handle.current_rows][disabled] = 0.0
            upper[handle.current_rows][disabled] = 0.0
            seed = jnp.where(active[:, None], seed, 0.0)
        handle.solver.update(l=lower, u=upper)
        seed_u = np.asarray(seed).ravel()
        seed_slacks = np.maximum(0.0, baseline_np + handle.response @ seed_u)
        seed_primal = np.r_[seed_u, seed_slacks]
        if handle.sparse_weight:
            seed_primal = np.r_[seed_primal, np.abs(seed_u)]
        handle.solver.warm_start(x=seed_primal, y=np.zeros(handle.upper.size))
        result = handle.solver.solve(raise_error=False)
        primal = result.x
        success = result.info.status == "solved"
        if primal is None or not np.isfinite(primal).all():
            msg = f"Condensed OSQP returned no finite plan ({result.info.status})"
            raise RuntimeError(msg)
        controls = jnp.asarray(primal[: seed_u.size].reshape(seed.shape)) if success else seed
        if active is not None:
            controls = jnp.where(active[:, None], controls, 0.0)
        states, _ = handle.evaluate(bc.x0, controls)
        times = bc.t0 + jnp.r_[0.0, jnp.cumsum(problem.dt)]
        trajectory = Trajectory(X=states, U=controls, t=times, dt=problem.dt)
        values = handle.matrix @ (primal if success else seed_primal)
        violation = float(np.maximum(lower - values, values - upper).max(initial=0.0))
        spectral, effort = (float(cost) for cost in handle.objective(trajectory))
        sparse = handle.sparse_weight * float(jnp.sum(jnp.abs(controls)))
        return OSQPResult(
            trajectory=trajectory,
            success=success,
            status=result.info.status_val,
            message=result.info.status,
            cost=spectral + effort + sparse,
            Z=_trajectory_to_z(states, controls),
            info={
                "status": result.info.status,
                "prim_res": result.info.prim_res,
                "dual_res": result.info.dual_res,
                "cost_spectral": spectral,
                "cost_quadratic_effort": effort,
                "cost_sparse_effort": sparse,
            },
            constraint_violation=violation,
            iterations=result.info.iter,
        )


class _CondensedIPOPTCB:
    """Exact-Hessian cyipopt callback for the condensed direct-hinge Observable Objective."""

    def __init__(  # noqa: PLR0913 -- shares compiled JAX kernels and constant linear Jacobian
        self,
        *,
        hm: int,
        horizon: int,
        integer_mode: bool,
        sparse_weight: float,
        active_weight: float,
        obj_fn: Callable[[jax.Array, jax.Array], jax.Array],
        grad_fn: Callable[[jax.Array, jax.Array], jax.Array],
        hess_fn: Callable[[jax.Array, jax.Array], jax.Array],
        linear_matrix: sp.coo_matrix,
    ) -> None:
        """Bind compiled physical Objective kernels and constant sparse constraint structure."""
        self.hm = hm
        self.horizon = horizon
        self.integer_mode = integer_mode
        self.sparse_weight = sparse_weight
        self.active_weight = active_weight
        self.obj_fn = obj_fn
        self.grad_fn = grad_fn
        self.hess_fn = hess_fn
        self.linear_matrix = linear_matrix.tocsr()
        self.jac_vals = np.asarray(linear_matrix.data, dtype=np.float64)
        self.jac_rows: IntArray = linear_matrix.row.astype(np.intp)
        self.jac_cols: IntArray = linear_matrix.col.astype(np.intp)
        self.u_tril_r, self.u_tril_c = np.tril_indices(hm)
        if integer_mode:
            a_diag = np.arange(hm, hm + horizon, dtype=np.intp)
            self.hess_rows: IntArray = np.r_[self.u_tril_r, a_diag]
            self.hess_cols: IntArray = np.r_[self.u_tril_c, a_diag]
        else:
            self.hess_rows = self.u_tril_r.astype(np.intp)
            self.hess_cols = self.u_tril_c.astype(np.intp)
        self.baseline = jnp.zeros(0, dtype=jnp.float64)
        self.beta = 0.0

    def objective(self, x: FloatArray) -> float:
        """Evaluate condensed direct-hinge spectral, quadratic, L1, activation, and homotopy Costs."""
        u = jnp.asarray(x[: self.hm], dtype=jnp.float64)
        val = float(self.obj_fn(u, self.baseline))
        offset = self.hm
        if self.integer_mode:
            a = x[offset : offset + self.horizon]
            val += self.active_weight * float(np.sum(a))
            if self.beta:
                val += self.beta * float(np.sum(a * (1.0 - a)))
            offset += self.horizon
        if self.sparse_weight:
            z = x[offset : offset + self.hm]
            val += self.sparse_weight * float(np.sum(z))
        return val

    def gradient(self, x: FloatArray) -> FloatArray:
        """Evaluate exact gradient for physical Control Currents, gates, and L1 slacks."""
        parts = [np.asarray(self.grad_fn(jnp.asarray(x[: self.hm], dtype=jnp.float64), self.baseline), dtype=np.float64)]
        offset = self.hm
        if self.integer_mode:
            a = x[offset : offset + self.horizon]
            ga = np.full(self.horizon, self.active_weight, dtype=np.float64)
            if self.beta:
                ga += self.beta * (1.0 - 2.0 * a)
            parts.append(ga)
            offset += self.horizon
        if self.sparse_weight:
            parts.append(np.full(self.hm, self.sparse_weight, dtype=np.float64))
        return np.concatenate(parts)

    def constraints(self, x: FloatArray) -> FloatArray:
        """Evaluate linear Kirchhoff, gate coupling, active cap, and L1 epigraph constraints."""
        return np.asarray(self.linear_matrix @ x, dtype=np.float64)

    def jacobian(self, x: FloatArray) -> FloatArray:  # noqa: ARG002 -- linear constraints have constant Jacobian
        """Return constant sparse linear constraint Jacobian values."""
        return self.jac_vals

    def jacobianstructure(self) -> tuple[IntArray, IntArray]:
        """Return sparse constraint Jacobian coordinates."""
        return self.jac_rows, self.jac_cols

    def hessian(self, x: FloatArray, lagrange: FloatArray, obj_factor: float) -> FloatArray:  # noqa: ARG002 -- linear constraints have zero Hessian
        """Evaluate exact lower-triangular Lagrangian Hessian."""
        h_u = np.asarray(
            self.hess_fn(jnp.asarray(x[: self.hm], dtype=jnp.float64), self.baseline), dtype=np.float64
        )[self.u_tril_r, self.u_tril_c]
        if self.integer_mode:
            h_a = (
                np.full(self.horizon, -2.0 * self.beta, dtype=np.float64)
                if self.beta
                else np.zeros(self.horizon, dtype=np.float64)
            )
            return obj_factor * np.r_[h_u, h_a]
        return obj_factor * h_u

    def hessianstructure(self) -> tuple[IntArray, IntArray]:
        """Return sparse lower-triangular Hessian coordinates."""
        return self.hess_rows, self.hess_cols


@dataclass
class _CondensedIPOPTHandle:
    """One prepared condensed IPOPT callback, bounds, and cached cyipopt Problem handles."""

    cb: _CondensedIPOPTCB
    evaluate: Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]
    objective: Callable[[Trajectory], tuple[jax.Array, jax.Array]]
    lbx: FloatArray
    ubx: FloatArray
    cl: FloatArray
    cu: FloatArray
    nlps: dict[bytes, Any]
    m_phys: int
    binary_indices: IntArray


@dataclass(frozen=True)
class CondensedIPOPT(Ipopt):
    """Solve the condensed direct-hinge Observable Problem with exact-Hessian IPOPT."""

    def __post_init__(self) -> None:
        """Validate inherited binary homotopy continuation parameters."""
        validate_homotopy_parameters(self.beta_initial, self.beta_growth, self.max_passes, self.binary_tolerance)

    def prepare(self, program: Program) -> None:  # noqa: PLR0915 -- compiles Rollout, constraints, and callback once
        """Compile the condensed Rollout, derivatives, and linear constraint matrix once."""
        problem = program.problem
        spec = _validate_objective(problem, allow_binary=True)
        horizon = problem.N - 1
        m_phys = spec.base_model.m
        hm = horizon * m_phys
        kirchhoff, active_cap = _extract_constraints(
            problem, horizon, m_phys, allow_binary=True, solver_name=type(self).__name__
        )
        hinge_scalar_weight = float(np.asarray(spec.hinge.weight).ravel()[0])
        hinge_phys = (
            AffineObservableHinge(
                spec.hinge.A[:, : spec.base_model.n + m_phys],
                spec.hinge.b,
                hinge_scalar_weight,
                n=spec.base_model.n,
                epigraph=False,
            )
            if spec.hinge.A.shape[1] > spec.base_model.n + m_phys
            else spec.hinge
        )
        dt = float(problem.dt[0])
        evaluate_base, response = condense_affine_observable(spec.base_model, hinge_phys, horizon, dt)
        response_jax = jnp.asarray(response, dtype=jnp.float64)
        hinge_weight = jnp.tile(jnp.asarray(spec.hinge.weight, dtype=jnp.float64), horizon)
        r_quad_arr = np.broadcast_to(np.asarray(spec.effort.R, dtype=np.float64), (horizon, spec.effort.m))
        r_quad = jnp.asarray(r_quad_arr[:, :m_phys].ravel(), dtype=jnp.float64)

        @jax.jit
        def u_obj(u_flat: jax.Array, baseline: jax.Array) -> jax.Array:
            """Score condensed direct spectral hinge and quadratic Control Current effort."""
            res = baseline + response_jax @ u_flat
            return jnp.sum(hinge_weight * (jnp.maximum(res, 0.0) ** 2)) + 0.5 * jnp.sum(r_quad * (u_flat**2))

        u_grad = jax.jit(jax.grad(u_obj))
        u_hess = jax.jit(jax.hessian(u_obj))

        _, _, u_lower, u_upper = problem.constraints.primal_bounds()
        u_lower_np = np.asarray(u_lower[:, :m_phys], dtype=np.float64).ravel()
        u_upper_np = np.asarray(u_upper[:, :m_phys], dtype=np.float64).ravel()
        u_max_vec = np.asarray(u_upper[0, :m_phys], dtype=np.float64)

        n_a = horizon if spec.integer_mode else 0
        n_z = hm if spec.sparse_weight else 0
        n_vars = hm + n_a + n_z
        blocks: list[sp.spmatrix] = []
        cl_parts: list[FloatArray] = []
        cu_parts: list[FloatArray] = []

        if kirchhoff.shape[0]:
            blocks.append(sp.hstack((kirchhoff, sp.csc_matrix((kirchhoff.shape[0], n_a + n_z)))))
            cl_parts.append(np.zeros(kirchhoff.shape[0]))
            cu_parts.append(np.zeros(kirchhoff.shape[0]))

        if spec.integer_mode:
            i_hm = sp.eye(hm, format="csc")
            g_a = sp.kron(sp.eye(horizon), u_max_vec[:, None], format="csc")
            pad_z = sp.csc_matrix((hm, n_z))
            blocks.append(sp.hstack((i_hm, -g_a, pad_z)))
            blocks.append(sp.hstack((-i_hm, -g_a, pad_z)))
            cl_parts.extend((np.full(hm, -np.inf), np.full(hm, -np.inf)))
            cu_parts.extend((np.zeros(hm), np.zeros(hm)))
            if active_cap is not None:
                cap_row = sp.hstack((sp.csc_matrix((1, hm)), sp.csr_matrix(np.ones((1, horizon))), sp.csc_matrix((1, n_z))))
                blocks.append(cap_row)
                cl_parts.append(np.array([-np.inf]))
                cu_parts.append(np.array([active_cap]))

        if spec.sparse_weight:
            l1_rows, l1_cl, l1_cu = _l1_epigraph_blocks(hm, n_a)
            blocks.append(l1_rows)
            cl_parts.append(l1_cl)
            cu_parts.append(l1_cu)

        linear_matrix = sp.vstack(blocks, format="coo") if blocks else sp.coo_matrix((0, n_vars))
        cl = np.concatenate(cl_parts) if cl_parts else np.zeros(0, dtype=np.float64)
        cu = np.concatenate(cu_parts) if cu_parts else np.zeros(0, dtype=np.float64)
        lbx_parts = [u_lower_np]
        ubx_parts = [u_upper_np]
        if spec.integer_mode:
            lbx_parts.append(np.zeros(horizon, dtype=np.float64))
            ubx_parts.append(np.ones(horizon, dtype=np.float64))
        if spec.sparse_weight:
            lbx_parts.append(np.zeros(hm, dtype=np.float64))
            ubx_parts.append(np.maximum(np.abs(u_lower_np), np.abs(u_upper_np)))

        cb = _CondensedIPOPTCB(
            hm=hm,
            horizon=horizon,
            integer_mode=spec.integer_mode,
            sparse_weight=spec.sparse_weight,
            active_weight=spec.active_weight,
            obj_fn=u_obj,
            grad_fn=u_grad,
            hess_fn=u_hess,
            linear_matrix=linear_matrix,
        )

        @jax.jit
        def evaluate(x0: jax.Array, controls: jax.Array) -> tuple[jax.Array, jax.Array]:
            """Evaluate physical states and unrectified spectral residuals from full Control Currents."""
            return evaluate_base(x0, controls[:, :m_phys])

        @jax.jit
        def objective(trajectory: Trajectory) -> tuple[jax.Array, jax.Array]:
            """Evaluate physical spectral and quadratic Control Current effort subtotals."""
            u_phys = trajectory.U[:, :m_phys]
            _, res = evaluate_base(trajectory.X[0], u_phys)
            spectral = jnp.sum(hinge_weight * (jnp.maximum(res, 0.0) ** 2))
            quad = 0.5 * jnp.sum(r_quad * (u_phys.ravel() ** 2))
            return spectral, quad

        zero_state = jnp.zeros(spec.base_model.n, dtype=jnp.float64)
        zero_controls = jnp.zeros((horizon, spec.effort.m), dtype=jnp.float64)
        zero_states, zero_res = evaluate(zero_state, zero_controls)
        jax.block_until_ready(
            objective(
                Trajectory(
                    X=zero_states,
                    U=zero_controls,
                    t=jnp.r_[0.0, jnp.cumsum(problem.dt)],
                    dt=problem.dt,
                )
            )
        )
        cb.baseline = zero_res
        _ = cb.objective(np.zeros(n_vars))
        _ = cb.gradient(np.zeros(n_vars))
        _ = cb.hessian(np.zeros(n_vars), np.zeros(cl.size), 1.0)

        program.handles["condensed_ipopt"] = _CondensedIPOPTHandle(
            cb=cb,
            evaluate=evaluate,
            objective=objective,
            lbx=np.concatenate(lbx_parts),
            ubx=np.concatenate(ubx_parts),
            cl=cl,
            cu=cu,
            nlps={},
            m_phys=m_phys,
            binary_indices=np.arange(hm, hm + n_a, dtype=np.intp),
        )

    def _get_nlp(self, handle: _CondensedIPOPTHandle, active: BoolArray | None) -> tuple[Any, FloatArray, FloatArray]:
        """Retrieve or compile a cyipopt Problem handle for the given fixed schedule mask."""
        import cyipopt  # noqa: PLC0415 -- optional solver dependency registered via trajopt

        key = b"" if active is None else active.tobytes()
        lbx, ubx = handle.lbx.copy(), handle.ubx.copy()
        if active is not None:
            disabled = np.repeat(~active, handle.m_phys)
            hm = handle.cb.hm
            lbx[:hm][disabled] = 0.0
            ubx[:hm][disabled] = 0.0
            if handle.cb.sparse_weight:
                z_start = hm + (handle.cb.horizon if handle.cb.integer_mode else 0)
                ubx[z_start : z_start + hm][disabled] = 0.0
        nlp = handle.nlps.get(key)
        if nlp is None:
            problem_cls: Any = getattr(cyipopt, "Problem")  # noqa: B009 -- cyipopt Problem
            nlp = problem_cls(
                n=lbx.size,
                m=handle.cl.size,
                problem_obj=handle.cb,
                lb=lbx,
                ub=ubx,
                cl=handle.cl,
                cu=handle.cu,
            )
            default_tol = 1e-4 if handle.cb.integer_mode else 1e-8
            opts = {"print_level": 0, "sb": "yes", "tol": default_tol, **self.options}
            for opt_key, opt_val in opts.items():
                nlp.add_option(opt_key, opt_val)
            handle.nlps[key] = nlp
        return nlp, lbx, ubx

    def solve(
        self, program: Program, bc: BoundaryConditions, ws: WarmStart, *, active: BoolArray | None = None
    ) -> IpoptResult:
        """Solve the condensed direct-hinge Problem from the measured state and optional mask."""
        handle, _, baseline = _evaluate_at_boundary(self, program, bc, "condensed_ipopt")
        return self._solve_at_boundary(program, bc, ws, handle, baseline, active=active)

    def select_schedule(
        self, program: Program, bc: BoundaryConditions, ws: WarmStart, schedules: list[IntArray], weight: float
    ) -> tuple[FloatArray, IntArray, int, bool]:
        """Price all-off directly and reuse one measured free response across fixed schedules."""
        handle, zero_states, baseline = _evaluate_at_boundary(self, program, bc, "condensed_ipopt")
        problem = program.problem
        return _select_condensed_schedule(
            schedules,
            weight,
            (problem.N - 1, problem.model.m),
            _zero_cost(program, bc, zero_states, handle.objective),
            lambda active: self._solve_at_boundary(program, bc, ws, handle, baseline, active=active),
        )

    def _solve_at_boundary(  # noqa: PLR0913 -- share the prepared handle and residual across fixed schedules
        self,
        program: Program,
        bc: BoundaryConditions,
        ws: WarmStart,
        handle: _CondensedIPOPTHandle,
        baseline_np: FloatArray,
        *,
        active: BoolArray | None,
    ) -> IpoptResult:
        """Solve continuous or homotopy subproblem at the evaluated measured state."""
        problem = program.problem
        _, seed = ws.unpack(problem)
        m_phys = handle.m_phys
        seed_u_arr = np.asarray(seed[:, :m_phys], dtype=np.float64)
        if active is not None:
            seed_u_arr = np.where(active[:, None], seed_u_arr, 0.0)
        seed_u = seed_u_arr.ravel()
        seed_parts = [seed_u]
        if handle.cb.integer_mode:
            seed_parts.append(np.asarray(seed[:, m_phys], dtype=np.float64).ravel())
        if handle.cb.sparse_weight:
            seed_parts.append(np.abs(seed_u))
        nlp, lbx, ubx = self._get_nlp(handle, active)
        z0 = np.clip(np.concatenate(seed_parts), lbx, ubx)

        handle.cb.baseline = jnp.asarray(baseline_np, dtype=jnp.float64)
        handle.cb.beta = 0.0
        if handle.cb.integer_mode:
            z_opt, info, pass_count, binary_distance = solve_continuation_passes(
                nlp,
                handle.cb,
                z0,
                None,
                None,
                binary_indices=handle.binary_indices,
                beta_initial=self.beta_initial,
                beta_growth=self.beta_growth,
                max_passes=self.max_passes,
                binary_tolerance=self.binary_tolerance,
            )
        else:
            nlp.add_option("warm_start_init_point", "no")
            z_opt, raw_info = nlp.solve(z0)
            info = dict(raw_info)
            z_opt = np.asarray(z_opt, dtype=np.float64)
            pass_count, binary_distance = 1, 0.0

        status = int(info.get("status", -1))
        g_val = handle.cb.constraints(z_opt)
        con_viol = float(np.maximum(handle.cl - g_val, g_val - handle.cu).max(initial=0.0))
        bnd_viol = float(np.maximum(lbx - z_opt, z_opt - ubx).max(initial=0.0))
        violation = max(con_viol, bnd_viol)
        feasibility_tol = float(self.options.get("constr_viol_tol", 1e-4))
        success = (
            status in _SUCCESS_STATUSES
            and (not handle.cb.integer_mode or (pass_count > 1 and binary_distance <= self.binary_tolerance))
            and violation <= feasibility_tol
        )

        u_opt = z_opt[: handle.cb.hm].reshape(handle.cb.horizon, m_phys)
        if active is not None:
            u_opt = np.where(active[:, None], u_opt, 0.0)
        if handle.cb.integer_mode:
            a_opt = z_opt[handle.cb.hm : handle.cb.hm + handle.cb.horizon]
            controls = jnp.asarray(np.hstack((u_opt, a_opt[:, None])), dtype=jnp.float64)
            active_cost = handle.cb.active_weight * float(np.sum(a_opt))
        else:
            controls = jnp.asarray(u_opt, dtype=jnp.float64)
            active_cost = 0.0
        states, _ = handle.evaluate(bc.x0, controls)
        times = bc.t0 + jnp.r_[0.0, jnp.cumsum(problem.dt)]
        trajectory = Trajectory(X=states, U=controls, t=times, dt=problem.dt)
        spectral, effort = (float(cost) for cost in handle.objective(trajectory))
        sparse = handle.cb.sparse_weight * float(np.sum(np.abs(u_opt)))
        return IpoptResult(
            trajectory=trajectory,
            success=success,
            status=status,
            message=str(info.get("status_msg", "")),
            cost=spectral + effort + sparse + active_cost,
            Z=_trajectory_to_z(states, controls),
            info={
                **info,
                "cost_spectral": spectral,
                "cost_quadratic_effort": effort,
                "cost_sparse_effort": sparse,
                "cost_active": active_cost,
                "homotopy_beta": handle.cb.beta,
                "binary_distance": binary_distance,
                "homotopy_passes": pass_count,
            },
            constraint_violation=violation,
            iterations=int(info.get("iter", 0)),
        )
