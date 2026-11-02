from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from simulate.controller import Controller
from trajopt.problem import (
    BoundaryConditions,
    Problem,
    retarget_problem,
)
from trajopt.program import WarmStart
from trajopt.transcription.ipopt import _SUCCESS_STATUSES
from trajopt.transcription.layout import parse_solver_initial_state
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

from neuro.control.diagnostics import (
    _rollout_states,
    decompose_cost,
    planned_outputs,
)
from neuro.control.mpc import TrajOptMPCLog
from neuro.control.problems import _build_problem
from neuro.predictor.inference import InferencePredictor

if TYPE_CHECKING:
    from neuro.types import FloatArray


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
        """Initialize candidate schedules and physical Control Current history.

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
        self.n_controls = int(self.model.m)
        self.n_electrodes = self.model.m
        self._prev_plan_u: FloatArray | None = None

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
        """Absorb measurement and apply the best candidate schedule's first physical Control Current."""
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
        planned_u = best_sol.reshape(self.horizon, m_dim)
        u_cmd = planned_u[0]

        self._u_last = planned_u[0]
        self._prev_plan_u = np.vstack((planned_u[1:], planned_u[-1:]))

        u_jax = jnp.asarray(planned_u, dtype=jnp.float64)
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
            planned_u=planned_u,
            planned_active=best_sched.astype(np.float64),
            active_count=float(np.sum(best_sched)),
            cost_active=self.w_active * float(np.sum(best_sched)) / self.horizon,
            cost_spectral=costs_decomp["cost_spectral"],
            cost_quadratic_effort=costs_decomp["cost_quadratic_effort"],
            cost_sparse_effort=costs_decomp["cost_sparse_effort"],
            cost_tracking=costs_decomp["cost_tracking"],
            normalization="channel_mean",
        )
