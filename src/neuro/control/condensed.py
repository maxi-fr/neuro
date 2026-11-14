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

from neuro.control.costs import AffineObservableHinge
from neuro.control.diagnostics import _rollout_states
from neuro.predictor.inference import ObservableMLPModel

if TYPE_CHECKING:
    from collections.abc import Callable

    import osqp
    from trajopt.problem import BoundaryConditions, Problem
    from trajopt.program import Program, WarmStart

    from neuro.types import FloatArray

_STAGE_TERMS = 2


def _validate_objective(problem: Problem) -> tuple[DiagonalCost, AffineObservableHinge]:
    """Extract the two Costs this convex QP can represent exactly."""
    terms = cost_terms(problem.obj.stage_cost)
    terminal = problem.obj.terminal_cost
    model = problem.model
    if (
        isinstance(model, ObservableMLPModel)
        and len(model.weights) == 1
        and len(terms) == _STAGE_TERMS
        and isinstance(terminal, DiagonalCost)
        and not np.any(terminal.Q)
        and not np.any(terminal.q)
        and not np.any(terminal.c)
    ):
        effort, hinge = terms
        if (
            isinstance(effort, DiagonalCost)
            and isinstance(hinge, AffineObservableHinge)
            and not (
                np.any(effort.Q) or np.any(effort.q) or np.any(effort.c) or np.any(effort.r) or np.any(effort.R < 0)
            )
        ):
            return effort, hinge
    msg = "CondensedOSQP requires a depth-zero Observable MLP Problem with affine_hinge=True and convex current effort"
    raise ValueError(msg)


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


class CondensedOSQP:
    """Solve the affine Observable objective in Control Currents and hinge slacks."""

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        """Set OSQP options with the campaign's replay tolerances."""
        self.options = {"verbose": False, "eps_abs": 1e-4, "eps_rel": 1e-4, **(options or {})}

    def prepare(self, program: Program) -> None:
        """Compile the Rollout and allocate the fixed QP before the first control release."""
        import osqp  # noqa: PLC0415 -- import the backend only when selected

        problem = program.problem
        effort, hinge = _validate_objective(problem)
        kirchhoff = _kirchhoff_rows(problem)
        model = problem.model
        horizon = problem.N - 1
        currents = horizon * model.m
        residuals = horizon * len(hinge.b)
        zero_state = jnp.zeros(model.n, dtype=jnp.float64)
        zero_controls = jnp.zeros((horizon, model.m), dtype=jnp.float64)
        dt = float(problem.dt[0])

        @jax.jit
        def evaluate(x0: jax.Array, controls: jax.Array) -> tuple[jax.Array, jax.Array]:
            """Return physical states and unrectified affine residuals along the Rollout."""
            states = _rollout_states(model, x0, controls, dt)
            z = jnp.concatenate([states[:-1], controls], axis=1)
            return states, (z @ hinge.A.T - hinge.b).ravel()

        zero_states, zero_residual = evaluate(zero_state, zero_controls)
        baseline = np.asarray(zero_residual)
        response = np.asarray(jax.jacfwd(lambda u: evaluate(zero_state, u)[1])(zero_controls))
        response = response.reshape(residuals, currents)
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
            solver, evaluate, upper, lower, matrix, response, objective
        )

    def solve(self, program: Program, bc: BoundaryConditions, ws: WarmStart) -> OSQPResult:
        """Update the envelope offset, seed the shifted plan and return a physical Trajectory."""
        if bc.X_ref is not None or bc.U_ref is not None or bc.xf is not None:
            msg = "CondensedOSQP requires a fixed Objective and no terminal goal"
            raise ValueError(msg)
        if "condensed_osqp" not in program.handles:
            self.prepare(program)
        handle: _CondensedHandle = program.handles["condensed_osqp"]
        problem = program.problem
        _, seed = ws.unpack(problem)
        _, baseline = handle.evaluate(bc.x0, jnp.zeros_like(seed))
        baseline_np = np.asarray(baseline)
        if not np.isfinite(baseline_np).all():
            msg = "CondensedOSQP requires finite predicted Frame residuals"
            raise ValueError(msg)
        rows = baseline_np.size
        handle.upper[:rows] = -baseline_np
        handle.solver.update(u=handle.upper)
        seed_u = np.asarray(seed).ravel()
        seed_slacks = np.maximum(0.0, baseline_np + handle.response @ seed_u)
        seed_primal = np.r_[seed_u, seed_slacks]
        handle.solver.warm_start(x=seed_primal, y=np.zeros(handle.upper.size))
        result = handle.solver.solve(raise_error=False)
        primal = result.x
        success = result.info.status == "solved"
        if primal is None or not np.isfinite(primal).all():
            msg = f"Condensed OSQP returned no finite plan ({result.info.status})"
            raise RuntimeError(msg)
        controls = jnp.asarray(primal[: seed_u.size].reshape(seed.shape)) if success else seed
        states, _ = handle.evaluate(bc.x0, controls)
        times = bc.t0 + jnp.r_[0.0, jnp.cumsum(problem.dt)]
        trajectory = Trajectory(X=states, U=controls, t=times, dt=problem.dt)
        values = handle.matrix @ (primal if success else seed_primal)
        violation = float(np.maximum(handle.lower - values, values - handle.upper).max(initial=0.0))
        spectral, effort = (float(cost) for cost in handle.objective(trajectory))
        return OSQPResult(
            trajectory=trajectory,
            success=success,
            status=result.info.status_val,
            message=result.info.status,
            cost=spectral + effort,
            Z=_trajectory_to_z(states, controls),
            info={
                "status": result.info.status,
                "prim_res": result.info.prim_res,
                "dual_res": result.info.dual_res,
                "cost_spectral": spectral,
                "cost_quadratic_effort": effort,
            },
            constraint_violation=violation,
            iterations=result.info.iter,
        )
