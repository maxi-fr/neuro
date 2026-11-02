from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from trajopt.problem import (
    BoundaryConditions,
    Problem,
    retarget_problem,
)
from trajopt.trajectory import Trajectory
from trajopt.transcription.ipopt import _SUCCESS_STATUSES, Ipopt, IpoptResult
from trajopt.transcription.layout import _trajectory_to_z, parse_solver_initial_state
from trajopt.transcription.result import split_bound_duals
from trajopt.transcription.single_shooting import (
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

from neuro.control.solvers import IPOPT_DEFAULTS

if TYPE_CHECKING:
    from trajopt.program import Program, WarmStart


_BINARY_WARM_START_THRESHOLD = 1e-2


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


_NO_DUALS = np.zeros(0, dtype=np.float64)
