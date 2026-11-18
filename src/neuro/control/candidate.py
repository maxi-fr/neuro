from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from simulate.controller import Controller
from trajopt.costs.l1 import L1ControlCost
from trajopt.costs.sum import cost_terms
from trajopt.dynamics.rollout import rollout_states
from trajopt.problem import (
    BoundaryConditions,
    Problem,
    retarget_problem,
)
from trajopt.program import Program, WarmStart
from trajopt.transcription.ipopt import _SUCCESS_STATUSES
from trajopt.transcription.layout import parse_solver_initial_state
from trajopt.transcription.penalties import PenaltyLayout
from trajopt.transcription.single_shooting import (
    _constraint_bounds,
    _dense_jacobian_pattern,
    _primal_bounds,
    eval_f,
    eval_g,
    eval_grad_f,
    eval_jac_g,
    single_shooting_dimensions,
)

from neuro.control.condensed import CondensedIPOPT, CondensedOSQP
from neuro.control.diagnostics import (
    _knot_times,
    decompose_cost,
    planned_outputs,
)
from neuro.control.execution import CommandPlayback
from neuro.control.mpc import TrajOptMPCLog
from neuro.control.problems import _build_problem
from neuro.predictor.inference import InferencePredictor

if TYPE_CHECKING:
    from neuro.types import FloatArray, IntArray


_rollout_plan = eqx.filter_jit(rollout_states)


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
        """Bind the single-shooting callbacks to one measured state and solver layout."""
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

    def __init__(  # noqa: PLR0913 -- solver options join the configured schedule and release
        self,
        dt: float,
        problem: Problem,
        blocks: list[list[int]] | None = None,
        w_active: float = 0.3,
        control_start_s: float | None = None,
        *,
        condensed_osqp: dict[str, Any] | None = None,
        condensed_ipopt: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the schedules and prepare their selected continuous solver.

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
        control_start_s
            Common release on the native solve grid.
        condensed_osqp
            OSQP options for the affine spectral QP.
        condensed_ipopt
            IPOPT options for the condensed direct-hinge solver.
        """
        super().__init__(dt)
        if condensed_osqp is not None and condensed_ipopt is not None:
            msg = "condensed_osqp and condensed_ipopt are mutually exclusive"
            raise ValueError(msg)
        if problem.binary_control_indices:
            msg = "candidate activation schedules require a continuous Problem"
            raise ValueError(msg)
        if problem.control_block_steps > 1:
            msg = "candidate activation schedules do not support structural Control Current blocking"
            raise ValueError(msg)
        model = problem.model
        if not isinstance(model, InferencePredictor):
            msg = f"problem.model ({type(model).__name__}) does not implement InferencePredictor"
            raise TypeError(msg)
        self.problem = problem
        stage_terms = cost_terms(problem.obj.stage_cost)
        self._penalties = (
            PenaltyLayout(
                problem,
                (
                    tuple(term.reformulate() for term in stage_terms),
                    (None,) * len(cost_terms(problem.obj.terminal_cost)),
                ),
                reuse_transformed_violation=False,
            )
            if any(isinstance(term, L1ControlCost) for term in stage_terms)
            else None
        )
        self.model = model
        self.horizon = int(problem.N - 1)
        self._release = (
            None
            if control_start_s is None
            else CommandPlayback(dt, dt, self.horizon, control_start_s, max(model.priming_steps, model.n_history))
        )
        self.blocks = blocks if blocks is not None else [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]
        self.w_active = float(w_active)
        self.schedules = self._build_schedules(self.blocks, self.horizon)
        self._condensed: CondensedOSQP | CondensedIPOPT | None = (
            CondensedOSQP(condensed_osqp)
            if condensed_osqp is not None
            else CondensedIPOPT(condensed_ipopt)
            if condensed_ipopt is not None
            else None
        )
        self._condensed_program = None
        if self._condensed is not None:
            if sorted(index for block in self.blocks for index in block) != list(range(self.horizon)):
                msg = "Candidate blocks must partition the Control Horizon"
                raise ValueError(msg)
            self._condensed_program = Program(problem, self._condensed)
            self._condensed.prepare(self._condensed_program)
        self._state = np.asarray(self.model.initial_state(), dtype=np.float64)
        self._u_last = np.zeros(self.model.m, dtype=np.float64)
        self.n_controls = int(self.model.m)
        self.n_electrodes = self.model.m
        self._prev_plan_u: FloatArray | None = None

    @staticmethod
    def _build_schedules(blocks: list[list[int]], horizon: int) -> list[IntArray]:
        """Generate all binary block schedules, starting with all-off."""
        n_blocks = len(blocks)
        schedules: list[IntArray] = []
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
        """Instantiate schedules, solver options and release from configuration."""
        problem = _build_problem(config["problem"])
        blocks = config.get("blocks")
        w_active = float(config.get("w_active", 0.3))
        return cls(
            float(config["dt"]),
            problem,
            blocks=blocks,
            w_active=w_active,
            control_start_s=config.get("control_start_s"),
            condensed_osqp=config.get("condensed_osqp"),
            condensed_ipopt=config.get("condensed_ipopt"),
        )

    def _evaluate_candidates(
        self,
        cb: _CandidateCallback,
        bounds: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        seed: FloatArray,
    ) -> tuple[np.ndarray, np.ndarray, int, bool]:
        """Fix physical currents on inactive stages and solve with any L1 slacks free."""
        import cyipopt  # noqa: PLC0415 -- cyipopt is an optional solver dependency

        z_lower, z_upper, g_lower, g_upper = bounds
        problem_cls: Any = getattr(cyipopt, "Problem")  # noqa: B009 -- cyipopt Problem
        best_cost = float("inf")
        best_sol = np.zeros(len(z_lower))
        best_sched = self.schedules[0]
        best_status = -1
        best_success = False
        width = cb.problem.model.m

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
                lb.reshape(self.horizon, width)[sched == 0, : self.n_controls] = 0.0
                ub.reshape(self.horizon, width)[sched == 0, : self.n_controls] = 0.0

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

                init_guess = np.clip(seed, lb, ub)
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

        return best_sol, best_sched, best_status, best_success

    def _solve_condensed_candidates(
        self, bc: BoundaryConditions, ws: WarmStart
    ) -> tuple[FloatArray, IntArray, int, bool]:
        """Search the configured schedules against the condensed solver."""
        condensed, program = self._condensed, self._condensed_program
        if condensed is None or program is None:
            msg = "The condensed candidate solver has not been prepared"
            raise RuntimeError(msg)
        return condensed.select_schedule(program, bc, ws, self.schedules, self.w_active)

    def _solve_nlp_candidates(self, bc: BoundaryConditions, ws: WarmStart) -> tuple[np.ndarray, np.ndarray, int, bool]:
        """Lower the generic objective and enumerate its single-shooting subproblems."""
        solver_problem = self.problem
        if self._penalties is not None:
            solver_problem = self._penalties.lowered
            ws = self._penalties.lift_warm_start(ws)
        x0, t0, dt, xf, _ = parse_solver_initial_state(solver_problem, bc, ws)
        retargeted = retarget_problem(solver_problem, bc)
        cb = _CandidateCallback(retargeted, x0, t0, jnp.broadcast_to(dt, (self.horizon,)), xf)
        bounds = (*_primal_bounds(retargeted), *_constraint_bounds(retargeted))
        _, seed = ws.unpack(solver_problem)
        return self._evaluate_candidates(cb, bounds, np.asarray(seed).ravel())

    def update(
        self,
        t: float,
        ref: FloatArray,  # noqa: ARG002 -- goal is baked into objective
        x_hat: FloatArray,
    ) -> tuple[FloatArray, TrajOptMPCLog]:
        """Absorb physical history and log the selected plan from the configured schedule solver."""
        update_started = time.perf_counter()
        self._state = np.asarray(self.model.absorb(self._state, np.asarray(x_hat).reshape(-1), self._u_last))
        knots = self.problem.N
        m_dim = self.n_controls

        ready = self.model.is_ready(self._state)
        due = ready if self._release is None else self._release.phase(t, ready=ready)[2]
        if not due:
            self._u_last = np.zeros(m_dim, dtype=np.float64)
            u_zero = np.zeros(self.n_electrodes, dtype=np.float64)
            return u_zero, TrajOptMPCLog(
                u=u_zero,
                cost=float("nan"),
                success=False,
                warmup=True,
                status="warmup",
                solve_time=float("nan"),
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
                solve_attempted=False,
                plan_index=-1,
                controller_time=time.perf_counter() - update_started,
            )

        started = time.perf_counter()
        x0_arr = jnp.asarray(self._state, dtype=jnp.float64)
        bc = BoundaryConditions(x0=x0_arr, t0=jnp.asarray(t, dtype=jnp.float64))
        ws = WarmStart.cold(self.problem, x0_arr)
        if self._prev_plan_u is not None:
            ws = ws.with_primal(self.problem, U=jnp.asarray(self._prev_plan_u))
        solve = self._solve_nlp_candidates if self._condensed is None else self._solve_condensed_candidates
        best_sol, best_sched, best_status, best_success = solve(bc, ws)
        solve_time = time.perf_counter() - started
        planned_u = best_sol.reshape(self.horizon, -1)[:, :m_dim].copy()
        u_cmd = planned_u[0]

        self._u_last = planned_u[0]
        self._prev_plan_u = np.vstack((planned_u[1:], planned_u[-1:]))

        u_jax = jnp.asarray(planned_u, dtype=jnp.float64)
        t0_arr = bc.t0
        dt_arr = jnp.broadcast_to(self.problem.dt, (self.horizon,))
        times = _knot_times(t0_arr, dt_arr, knots)
        x_rollout = _rollout_plan(self.problem.model, x0_arr, u_jax, t=times, dt=dt_arr)
        predicted_y = np.asarray(planned_outputs(self.model, x_rollout, u_jax, t0_arr, dt_arr)).copy()
        costs_decomp = decompose_cost(self.problem, x_rollout, u_jax, t0_arr, dt_arr)

        activation_cost = self.w_active * float(np.sum(best_sched)) / self.horizon
        return u_cmd, TrajOptMPCLog(
            u=u_cmd.copy(),
            cost=costs_decomp["cost_total"] + activation_cost if best_success else float("inf"),
            success=best_success,
            warmup=False,
            status=str(best_status),
            solve_time=solve_time,
            predicted_y=predicted_y,
            planned_u=planned_u,
            planned_active=best_sched.astype(np.float64),
            active_count=float(np.sum(best_sched)),
            cost_active=activation_cost,
            cost_spectral=costs_decomp["cost_spectral"],
            cost_quadratic_effort=costs_decomp["cost_quadratic_effort"],
            cost_sparse_effort=costs_decomp["cost_sparse_effort"],
            cost_tracking=costs_decomp["cost_tracking"],
            normalization="channel_mean",
            fallback=not best_success,
            controller_time=time.perf_counter() - update_started,
            replan_time=time.perf_counter() - started,
        )
