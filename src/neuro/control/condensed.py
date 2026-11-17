from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from trajopt.cones import ZeroCone
from trajopt.constraints.linear import LinearConstraint
from trajopt.costs.quadratic import DiagonalCost
from trajopt.costs.sum import cost_terms
from trajopt.trajectory import Trajectory
from trajopt.transcription.layout import _trajectory_to_z
from trajopt.transcription.osqp import OSQPResult

from neuro.control.costs import AffineObservableHinge, CondensedL1ControlCost
from neuro.control.diagnostics import _rollout_states
from neuro.predictor.inference import ObservableMLPModel

if TYPE_CHECKING:
    from collections.abc import Callable

    import osqp
    from trajopt.dynamics.base import DiscreteDynamics
    from trajopt.problem import BoundaryConditions, Problem
    from trajopt.program import Program, WarmStart

    from neuro.types import BoolArray, FloatArray, IntArray


def _validate_objective(problem: Problem) -> tuple[DiagonalCost, AffineObservableHinge, float]:
    """Extract the affine hinge and supported physical current Costs."""
    terms = cost_terms(problem.obj.stage_cost)
    sparse = next((term for term in terms if isinstance(term, CondensedL1ControlCost)), None)
    physical_terms = tuple(term for term in terms if term is not sparse)
    terminal = problem.obj.terminal_cost
    model = problem.model
    if (
        isinstance(model, ObservableMLPModel)
        and len(model.weights) == 1
        and len(physical_terms) == 2  # noqa: PLR2004 -- one quadratic term and one hinge
        and isinstance(terminal, DiagonalCost)
        and not np.any(terminal.Q)
        and not np.any(terminal.q)
        and not np.any(terminal.c)
    ):
        effort, hinge = physical_terms
        if (
            isinstance(effort, DiagonalCost)
            and isinstance(hinge, AffineObservableHinge)
            and not (
                np.any(effort.Q) or np.any(effort.q) or np.any(effort.c) or np.any(effort.r) or np.any(effort.R < 0)
            )
        ):
            return effort, hinge, float(sparse.weight) if sparse is not None else 0.0
    msg = "CondensedOSQP requires a depth-zero Observable MLP Problem with affine_hinge=True and convex current effort"
    raise ValueError(msg)


def condense_affine_observable(
    model: DiscreteDynamics, hinge: AffineObservableHinge, horizon: int, dt: float
) -> tuple[Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]], FloatArray]:
    """Return the affine residual Rollout and its fixed Jacobian in planned currents."""
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


def _kirchhoff_rows(problem: Problem) -> sp.csc_matrix:
    """Validate physical constraints and place each Kirchhoff equality on its stage currents."""
    constraints = problem.constraints
    x_lower, x_upper, _, _ = constraints.primal_bounds()
    if (
        problem.control_block_steps != 1
        or problem.binary_control_indices
        or constraints.horizon_constraints
        or np.any(np.isfinite(x_lower))
        or np.any(np.isfinite(x_upper))
    ):
        msg = "CondensedOSQP supports continuous unblocked currents with only boxes and Kirchhoff equalities"
        raise ValueError(msg)
    n, m = problem.model.n, problem.model.m
    steps = []
    for k, evaluator in enumerate(constraints.knot_evaluators):
        for row in evaluator.constraints:
            if (
                k == problem.N - 1
                or not isinstance(row, LinearConstraint)
                or not isinstance(row.cone, ZeroCone)
                or row.inds != tuple(range(n, n + m))
                or not np.array_equal(row.A, np.ones((1, m)))
                or not np.array_equal(row.b, np.zeros(1))
            ):
                msg = "CondensedOSQP supports only Kirchhoff equalities at control knots"
                raise ValueError(msg)
            steps.append(k)
    all_rows = sp.kron(sp.eye(problem.N - 1), np.ones((1, m)), format="csr")
    return all_rows[steps].tocsc()


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


class CondensedOSQP:
    """Solve the affine Observable objective in Control Currents and epigraph slacks."""

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        """Set OSQP options with the campaign's replay tolerances."""
        self.options = {"verbose": False, "eps_abs": 1e-4, "eps_rel": 1e-4, **(options or {})}

    def prepare(self, program: Program) -> None:
        """Compile the Rollout and allocate the fixed QP before the first control release."""
        import osqp  # noqa: PLC0415 -- import the backend only when selected

        problem = program.problem
        effort, hinge, sparse_weight = _validate_objective(problem)
        kirchhoff = _kirchhoff_rows(problem)
        model = problem.model
        horizon = problem.N - 1
        currents = horizon * model.m
        residuals = horizon * len(hinge.b)
        zero_state = jnp.zeros(model.n, dtype=jnp.float64)
        zero_controls = jnp.zeros((horizon, model.m), dtype=jnp.float64)
        dt = float(problem.dt[0])

        evaluate, response = condense_affine_observable(model, hinge, horizon, dt)
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
            np.r_[np.asarray(effort.R).ravel(), np.tile(2 * np.asarray(hinge.weight), horizon)], format="csc"
        )
        linear = np.zeros(currents + residuals)
        if sparse_weight:
            identity = sp.eye(currents)
            matrix = sp.vstack(
                [
                    sp.hstack((matrix, sp.csc_matrix((matrix.shape[0], currents)))),
                    sp.hstack((identity, sp.csc_matrix((currents, residuals)), -identity)),
                    sp.hstack((-identity, sp.csc_matrix((currents, residuals)), -identity)),
                ],
                format="csc",
            )
            hessian = sp.block_diag((hessian, sp.csc_matrix((currents, currents))), format="csc")
            lower = np.r_[lower, np.full(2 * currents, -np.inf)]
            upper = np.r_[upper, np.zeros(2 * currents)]
            linear = np.r_[linear, np.full(currents, sparse_weight)]
        solver = osqp.OSQP()
        solver.setup(P=hessian, q=linear, A=matrix, l=lower, u=upper, **self.options)

        @jax.jit
        def objective(trajectory: Trajectory) -> tuple[jax.Array, jax.Array]:
            """Evaluate physical spectral and Control Current effort subtotals."""
            inputs = trajectory.X[:-1], trajectory.U, trajectory.t[:-1]
            return jnp.sum(hinge.stage_costs(*inputs)), jnp.sum(effort.stage_costs(*inputs))

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
            sparse_weight,
        )

    def solve(
        self, program: Program, bc: BoundaryConditions, ws: WarmStart, *, active: BoolArray | None = None
    ) -> OSQPResult:
        """Update the envelope and optional fixed schedule, then return a physical Trajectory."""
        handle, _, baseline = self._at_boundary(program, bc)
        return self._solve_at_boundary(program, bc, ws, handle, baseline, active=active)

    def select_schedule(
        self, program: Program, bc: BoundaryConditions, ws: WarmStart, schedules: list[IntArray], weight: float
    ) -> tuple[FloatArray, IntArray, int, bool]:
        """Price all-off directly and reuse one measured free response across fixed schedules."""
        problem = program.problem
        handle, zero_states, baseline = self._at_boundary(program, bc)
        zero = jnp.zeros((problem.N - 1, problem.model.m), dtype=jnp.float64)
        times = bc.t0 + jnp.r_[0.0, jnp.cumsum(problem.dt)]
        best_plan = np.asarray(zero)
        best_schedule = schedules[0]
        best_cost = float(handle.objective(Trajectory(X=zero_states, U=zero, t=times, dt=problem.dt))[0])
        status = 0
        solved_any = False
        for schedule in schedules[1:]:
            solved = self._solve_at_boundary(program, bc, ws, handle, baseline, active=schedule.astype(bool))
            if not solved.success:
                continue
            solved_any = True
            score = float(solved.cost) + weight * float(schedule.mean())
            if score < best_cost:
                best_cost, best_schedule, status = score, schedule, solved.status
                best_plan = np.asarray(solved.trajectory.U)
        return best_plan.ravel(), best_schedule, status, solved_any

    def _at_boundary(self, program: Program, bc: BoundaryConditions) -> tuple[_CondensedHandle, jax.Array, FloatArray]:
        """Evaluate and validate the zero-current rollout for the measured state."""
        if bc.X_ref is not None or bc.U_ref is not None or bc.xf is not None:
            msg = "CondensedOSQP requires a fixed Objective and no terminal goal"
            raise ValueError(msg)
        if "condensed_osqp" not in program.handles:
            self.prepare(program)
        handle: _CondensedHandle = program.handles["condensed_osqp"]
        problem = program.problem
        zero = jnp.zeros((problem.N - 1, problem.model.m), dtype=jnp.float64)
        states, baseline = handle.evaluate(bc.x0, zero)
        baseline_np = np.asarray(baseline)
        if not np.isfinite(baseline_np).all():
            msg = "CondensedOSQP requires finite predicted Frame residuals"
            raise ValueError(msg)
        return handle, states, baseline_np

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
