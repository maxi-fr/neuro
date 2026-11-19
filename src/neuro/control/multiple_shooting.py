from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import trajopt  # noqa: F401 -- registers the Windows IPOPT DLL directory before importing cyipopt
from trajopt.problem import BoundaryConditions, Problem, retarget_problem
from trajopt.trajectory import Trajectory
from trajopt.transcription.homotopy import _SUCCESS_STATUSES
from trajopt.transcription.ipopt import Ipopt, IpoptResult
from trajopt.transcription.layout import (
    _cone_bounds,
    _evaluator_bounds,
    _z_to_trajectory,
    compute_constraint_violation,
    parse_solver_initial_state,
    retained_knot_evaluators,
)
from trajopt.transcription.result import constraint_row_count

from neuro.predictor.base import AutoregressiveModel

if TYPE_CHECKING:
    from trajopt.program import Program, WarmStart

    from neuro.types import FloatArray


def _require_ar(problem: Problem) -> AutoregressiveModel:
    """Return ``problem.model`` typed as ``AutoregressiveModel`` or raise ``TypeError``."""
    model = problem.model
    if not isinstance(model, AutoregressiveModel):
        msg = f"MultipleShooting requires an AutoregressiveModel, got {type(model).__name__}."
        raise TypeError(msg)
    return model


def _validate_ms_problem(problem: Problem) -> AutoregressiveModel:
    """Validate that ``problem`` matches full-state multiple shooting assumptions."""
    model = _require_ar(problem)
    if int(problem.control_block_steps) != 1:
        msg = f"MultipleShooting requires control_block_steps=1, got {problem.control_block_steps}."
        raise ValueError(msg)
    if problem.binary_control_indices:
        msg = "MultipleShooting does not support binary control indices."
        raise ValueError(msg)
    return model


def _cost_fn(
    problem: Problem,
    z: jax.Array,
    t0: float | jax.Array,
    dt: float | jax.Array,
) -> jax.Array:
    """Evaluate total Objective at canonical Primal Vector ``z``."""
    model = _require_ar(problem)
    horizon = int(problem.N) - 1
    X, U = _z_to_trajectory(z, horizon + 1, int(model.n), int(model.m))
    dt_arr = jnp.broadcast_to(jnp.asarray(dt), (horizon,))
    t_arr = t0 + jnp.concatenate([jnp.zeros(1, dtype=z.dtype), jnp.cumsum(dt_arr)])
    obj = problem.obj
    stage_costs = obj.stage_cost.stage_costs(X[:-1], U, t_arr[:-1])
    term_cost = obj.terminal_cost.evaluate(X[-1], None, t_arr[-1])
    return jnp.sum(stage_costs) + term_cost


def _predict_from_xu(model: AutoregressiveModel, x_k: jax.Array, u_k: jax.Array) -> jax.Array:
    """Evaluate the newest standardized prediction from ``(x_k, u_k)``."""
    n_hist = int(model.n_history)
    n_o = int(model.n_outputs)
    n_y = int(model.n_y)
    n_u = int(model.n_u)
    m = int(model.m)
    n_z = n_hist * n_o
    y_win = x_k[:n_z].reshape((n_hist, n_o))[-n_y:]
    u_prev = x_k[n_z:].reshape((n_u, m))
    u_win = jnp.concatenate([u_prev[1:], u_k.reshape((1, m))], axis=0)
    return model._predict(y_win, (u_win - model.u_center) / model.u_scale)  # noqa: SLF001 -- AutoregressiveModel defines _predict for transcriptions


def _constraints_fn(  # noqa: PLR0913, PLR0917 -- constraint callback takes 6 arguments
    problem: Problem,
    z: jax.Array,
    x0: jax.Array,
    t0: float | jax.Array,
    dt: float | jax.Array,
    xf: jax.Array | None = None,
) -> jax.Array:
    """Evaluate initial condition, full-state shift-register defects, and user constraints."""
    model = _require_ar(problem)
    N = int(problem.N)
    horizon = N - 1
    n = int(model.n)
    m = int(model.m)
    X, U = _z_to_trajectory(z, N, n, m)

    init_defect = X[0] - x0
    next_states = jax.vmap(lambda xk, uk: model.discrete_dynamics(xk, uk, 0.0, model.dt))(X[:-1], U)
    dyn_defects = (X[1:] - next_states).reshape(-1)

    dt_arr = jnp.broadcast_to(jnp.asarray(dt), (horizon,))
    t_arr = t0 + jnp.concatenate([jnp.zeros(1, dtype=z.dtype), jnp.cumsum(dt_arr)])
    retained = retained_knot_evaluators(problem)
    parts: list[jax.Array] = [init_defect, dyn_defects]
    for k in range(horizon):
        evaluator = retained[k]
        if evaluator.p > 0:
            parts.append(evaluator.evaluate(X[k], U[k], t_arr[k], xf=xf))
    if len(retained) > horizon and retained[horizon].p > 0:
        parts.append(retained[horizon].evaluate(X[-1], None, t_arr[-1], xf=xf))
    if problem.constraints.horizon_constraints:
        parts.extend(con.evaluate(z) for con in problem.constraints.horizon_constraints)
    return jnp.concatenate(parts)


def _user_constraints_fn(
    problem: Problem,
    z: jax.Array,
    t0: float | jax.Array,
    dt: float | jax.Array,
    xf: jax.Array | None = None,
) -> jax.Array:
    """Evaluate retained knot and horizon user constraints."""
    model = _require_ar(problem)
    N = int(problem.N)
    horizon = N - 1
    X, U = _z_to_trajectory(z, N, int(model.n), int(model.m))
    dt_arr = jnp.broadcast_to(jnp.asarray(dt), (horizon,))
    t_arr = t0 + jnp.concatenate([jnp.zeros(1, dtype=z.dtype), jnp.cumsum(dt_arr)])
    retained = retained_knot_evaluators(problem)
    parts: list[jax.Array] = []
    for k in range(horizon):
        evaluator = retained[k]
        if evaluator.p > 0:
            parts.append(evaluator.evaluate(X[k], U[k], t_arr[k], xf=xf))
    if len(retained) > horizon and retained[horizon].p > 0:
        parts.append(retained[horizon].evaluate(X[-1], None, t_arr[-1], xf=xf))
    if problem.constraints.horizon_constraints:
        parts.extend(con.evaluate(z) for con in problem.constraints.horizon_constraints)
    if not parts:
        return jnp.zeros(0, dtype=z.dtype)
    return jnp.concatenate(parts)


def _ms_primal_bounds(problem: Problem) -> tuple[FloatArray, FloatArray]:
    """Build canonical primal bounds ``zL <= z <= zU``."""
    N = int(problem.N)
    x_lo, x_hi, u_lo, u_hi = problem.constraints.primal_bounds()
    stage_lo = np.concatenate([np.asarray(x_lo[:-1], dtype=np.float64), np.asarray(u_lo, dtype=np.float64)], axis=1)
    stage_hi = np.concatenate([np.asarray(x_hi[:-1], dtype=np.float64), np.asarray(u_hi, dtype=np.float64)], axis=1)
    return (
        np.concatenate([stage_lo.reshape(-1), np.asarray(x_lo[N - 1], dtype=np.float64)]),
        np.concatenate([stage_hi.reshape(-1), np.asarray(x_hi[N - 1], dtype=np.float64)]),
    )


def _ms_constraint_bounds(problem: Problem) -> tuple[FloatArray, FloatArray]:
    """Build lower and upper bounds for initial condition, defects, and user constraints."""
    model = _require_ar(problem)
    horizon = int(problem.N) - 1
    n_eq = (horizon + 1) * int(model.n)
    g_lo: list[FloatArray] = [np.zeros(n_eq, dtype=np.float64)]
    g_hi: list[FloatArray] = [np.zeros(n_eq, dtype=np.float64)]

    retained = retained_knot_evaluators(problem)
    for k in range(horizon):
        lo_k, hi_k = _evaluator_bounds(retained[k])
        g_lo.extend(lo_k)
        g_hi.extend(hi_k)
    if len(retained) > horizon:
        lo_t, hi_t = _evaluator_bounds(retained[horizon])
        g_lo.extend(lo_t)
        g_hi.extend(hi_t)
    for con in problem.constraints.horizon_constraints:
        lo, hi = _cone_bounds(con.cone, con.p)
        g_lo.append(lo)
        g_hi.append(hi)
    return np.concatenate(g_lo), np.concatenate(g_hi)


class _MsGather(eqx.Module):
    """Static metadata and user-constraint indices for multiple-shooting Jacobian."""

    n_const_ones: int = eqx.field(static=True)
    n_const_neg_ones: int = eqx.field(static=True)
    y_start_col: int = eqx.field(static=True)
    y_end_col: int = eqx.field(static=True)
    u_prev_start_col: int = eqx.field(static=True)
    u_prev_end_col: int = eqx.field(static=True)
    user_rows: jax.Array
    user_cols: jax.Array


@dataclass(frozen=True)
class _MsSparsity:
    """Compact int32 COO sparsity pattern for autoregressive full-state multiple shooting."""

    rows: np.ndarray
    cols: np.ndarray
    gather: _MsGather


def _defect_coo(horizon: int, model: AutoregressiveModel) -> tuple[list[np.ndarray], list[np.ndarray], int, int]:
    """Build constant (+1 / -1) and prediction COO indices for initial and defect rows."""
    n = int(model.n)
    m = int(model.m)
    n_o = int(model.n_outputs)
    n_hist = int(model.n_history)
    n_y = int(model.n_y)
    n_u = int(model.n_u)
    n_z = n_hist * n_o
    d_stage = n + m
    k_ids = np.arange(horizon, dtype=np.int32)

    # 1. All +1.0 entries: X[0] identity (n) + X[k+1] identity for every step (H * n).
    pos_rows: list[np.ndarray] = [np.arange(n, dtype=np.int32)]
    pos_cols: list[np.ndarray] = [np.arange(n, dtype=np.int32)]
    step_row_base = n + k_ids * n
    step_xnext_col = (k_ids + 1) * d_stage
    n_idx = np.arange(n, dtype=np.int32)[None, :]
    pos_rows.append((step_row_base[:, None] + n_idx).reshape(-1))
    pos_cols.append((step_xnext_col[:, None] + n_idx).reshape(-1))
    n_const_ones = n + horizon * n

    # 2. All -1.0 entries from shift-register copies in X[k+1] - F(X[k], U[k]).
    neg_rows: list[np.ndarray] = []
    neg_cols: list[np.ndarray] = []
    step_xk_col = k_ids * d_stage

    # Output shift rows: (n_hist - 1) * n_o.
    n_y_shift = (n_hist - 1) * n_o
    if n_y_shift > 0:
        ys_idx = np.arange(n_y_shift, dtype=np.int32)[None, :]
        neg_rows.append((step_row_base[:, None] + ys_idx).reshape(-1))
        neg_cols.append((step_xk_col[:, None] + n_o + ys_idx).reshape(-1))

    # Control shift rows: (n_u - 1) * m.
    n_u_shift = (n_u - 1) * m
    if n_u_shift > 0:
        us_idx = np.arange(n_u_shift, dtype=np.int32)[None, :]
        neg_rows.append((step_row_base[:, None] + n_z + us_idx).reshape(-1))
        neg_cols.append((step_xk_col[:, None] + n_z + m + us_idx).reshape(-1))

    # Newest control copy rows: m entries wrt U[k].
    m_idx = np.arange(m, dtype=np.int32)[None, :]
    neg_rows.append((step_row_base[:, None] + n_z + n_u_shift + m_idx).reshape(-1))
    neg_cols.append((step_xk_col[:, None] + n + m_idx).reshape(-1))

    n_const_neg_ones = horizon * (n_y_shift + n_u_shift + m)

    # 3. Prediction Jacobian blocks -d(pred_k)/d(y_win_k, u_prev_k, u_k) of shape (H, n_o, pred_in).
    y_start = (n_hist - n_y) * n_o
    y_end = n_z
    u_prev_start = n_z + m
    u_prev_end = n
    pred_cols_local = np.concatenate(
        [
            np.arange(y_start, y_end, dtype=np.int32),
            np.arange(u_prev_start, u_prev_end, dtype=np.int32),
            np.arange(n, n + m, dtype=np.int32),
        ]
    )
    pred_row_local = np.arange((n_hist - 1) * n_o, n_z, dtype=np.int32)
    r_grid = step_row_base[:, None] + pred_row_local[None, :]  # (H, n_o)
    c_grid = step_xk_col[:, None] + pred_cols_local[None, :]  # (H, pred_in)
    pred_rows = np.repeat(r_grid[:, :, None], len(pred_cols_local), axis=2).reshape(-1)
    pred_cols = np.repeat(c_grid[:, None, :], n_o, axis=1).reshape(-1)

    rows = [*pos_rows, *neg_rows, pred_rows]
    cols = [*pos_cols, *neg_cols, pred_cols]
    return rows, cols, n_const_ones, n_const_neg_ones


def _build_ms_sparsity(problem: Problem, model: AutoregressiveModel) -> _MsSparsity:
    """Build compact COO sparsity pattern exploiting shift-register structure."""
    horizon = int(problem.N) - 1
    n = int(model.n)
    m = int(model.m)
    n_o = int(model.n_outputs)
    n_hist = int(model.n_history)
    n_y = int(model.n_y)
    d_stage = n + m
    n_eq = (horizon + 1) * n

    row_parts, col_parts, n_ones, n_neg_ones = _defect_coo(horizon, model)

    retained = retained_knot_evaluators(problem)
    user_row_list: list[np.ndarray] = []
    user_col_list: list[np.ndarray] = []
    curr_row = n_eq
    for k in range(horizon):
        p_k = int(retained[k].p)
        if p_k > 0:
            col_start = k * d_stage
            cols_k = np.arange(col_start, col_start + d_stage, dtype=np.int32)
            user_row_list.append(np.repeat(np.arange(curr_row, curr_row + p_k, dtype=np.int32), len(cols_k)))
            user_col_list.append(np.tile(cols_k, p_k))
            curr_row += p_k
    if len(retained) > horizon and int(retained[horizon].p) > 0:
        p_t = int(retained[horizon].p)
        col_start = horizon * d_stage
        cols_t = np.arange(col_start, col_start + n, dtype=np.int32)
        user_row_list.append(np.repeat(np.arange(curr_row, curr_row + p_t, dtype=np.int32), len(cols_t)))
        user_col_list.append(np.tile(cols_t, p_t))
        curr_row += p_t
    n_vars = (horizon + 1) * n + horizon * m
    all_cols = np.arange(n_vars, dtype=np.int32)
    for con in problem.constraints.horizon_constraints:
        if con.p > 0:
            user_row_list.append(np.repeat(np.arange(curr_row, curr_row + con.p, dtype=np.int32), n_vars))
            user_col_list.append(np.tile(all_cols, con.p))
            curr_row += con.p

    user_rows = np.concatenate(user_row_list) if user_row_list else np.zeros(0, dtype=np.int32)
    user_cols = np.concatenate(user_col_list) if user_col_list else np.zeros(0, dtype=np.int32)
    if len(user_rows):
        row_parts.append(user_rows)
        col_parts.append(user_cols)

    n_z = n_hist * n_o
    gather = _MsGather(
        n_const_ones=n_ones,
        n_const_neg_ones=n_neg_ones,
        y_start_col=(n_hist - n_y) * n_o,
        y_end_col=n_z,
        u_prev_start_col=n_z + m,
        u_prev_end_col=n,
        user_rows=jnp.asarray(user_rows - n_eq if len(user_rows) else user_rows, dtype=jnp.int32),
        user_cols=jnp.asarray(user_cols, dtype=jnp.int32),
    )
    return _MsSparsity(
        rows=np.concatenate(row_parts).astype(np.int32),
        cols=np.concatenate(col_parts).astype(np.int32),
        gather=gather,
    )


@eqx.filter_jit
def eval_f(
    problem: Problem,
    z: jax.Array,
    t0: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.05,
) -> jax.Array:
    """Evaluate scalar Objective ``J(z)``."""
    return _cost_fn(problem, z, t0, dt)


@eqx.filter_jit
def eval_grad_f(
    problem: Problem,
    z: jax.Array,
    t0: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.05,
) -> jax.Array:
    """Evaluate Objective gradient ``nabla J(z)``."""
    return jax.grad(lambda z_: _cost_fn(problem, z_, t0, dt))(z)


@eqx.filter_jit
def eval_g(  # noqa: PLR0913, PLR0917 -- constraint evaluation takes 6 arguments
    problem: Problem,
    z: jax.Array,
    x0: jax.Array,
    t0: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.05,
    xf: jax.Array | None = None,
) -> jax.Array:
    """Evaluate full-state multiple shooting constraints ``g(z)``."""
    return _constraints_fn(problem, z, x0, t0, dt, xf=xf)


@eqx.filter_jit
def eval_jac_g(  # noqa: PLR0913 -- Jacobian callback takes 6 arguments plus static gather
    problem: Problem,
    z: jax.Array,
    t0: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.05,
    xf: jax.Array | None = None,
    *,
    gather: _MsGather,
) -> jax.Array:
    """Evaluate nonzero COO values of the shift-sparse multiple-shooting Jacobian."""
    model = _require_ar(problem)
    horizon = int(problem.N) - 1
    X, U = _z_to_trajectory(z, horizon + 1, int(model.n), int(model.m))

    def step_jac(xk: jax.Array, uk: jax.Array) -> tuple[jax.Array, jax.Array]:
        return jax.jacrev(_predict_from_xu, argnums=(1, 2))(model, xk, uk)

    jac_x, jac_u = jax.vmap(step_jac)(X[:-1], U)
    jac_y = jac_x[:, :, gather.y_start_col : gather.y_end_col]
    jac_up = jac_x[:, :, gather.u_prev_start_col : gather.u_prev_end_col]
    pred_vals = -jnp.concatenate([jac_y, jac_up, jac_u], axis=2).reshape(-1)

    parts: list[jax.Array] = [
        jnp.ones(gather.n_const_ones, dtype=z.dtype),
        -jnp.ones(gather.n_const_neg_ones, dtype=z.dtype),
        pred_vals,
    ]
    if gather.user_rows.size:
        jac_user = jax.jacrev(lambda z_: _user_constraints_fn(problem, z_, t0, dt, xf=xf))(z)
        parts.append(jac_user[gather.user_rows, gather.user_cols])
    return jnp.concatenate(parts)


class _MsCallback:
    """Callback wrapper connecting sparse autoregressive multiple-shooting kernels to cyipopt."""

    def __init__(
        self,
        problem: Problem,
        x0: jax.Array,
        t0: float | jax.Array,
        dt: float | jax.Array,
        xf: jax.Array | None = None,
    ) -> None:
        model = _validate_ms_problem(problem)
        self.problem = problem
        self.x0 = x0
        self.t0 = t0
        self.dt = dt
        self.xf = xf
        self.sparsity = _build_ms_sparsity(problem, model)
        self.iteration_count = 0

    def update(
        self,
        problem: Problem,
        x0: jax.Array,
        t0: float | jax.Array,
        dt: float | jax.Array,
        xf: jax.Array | None = None,
    ) -> None:
        """Update dynamic Problem and Boundary Conditions and reset iteration counter."""
        self.problem = problem
        self.x0 = x0
        self.t0 = t0
        self.dt = dt
        self.xf = xf
        self.iteration_count = 0

    def intermediate(self, *args: object) -> bool:
        """Track solver iteration counter from Ipopt."""
        if len(args) > 1 and isinstance(args[1], (int, float, str)):
            self.iteration_count = int(args[1])
        return True

    def objective(self, z: np.ndarray) -> float:
        """Evaluate Objective ``J(z)``."""
        return float(eval_f(self.problem, jnp.asarray(z), self.t0, self.dt))

    def gradient(self, z: np.ndarray) -> FloatArray:
        """Evaluate Objective gradient ``nabla J(z)``."""
        return np.asarray(eval_grad_f(self.problem, jnp.asarray(z), self.t0, self.dt), dtype=np.float64)

    def constraints(self, z: np.ndarray) -> FloatArray:
        """Evaluate multiple-shooting constraint vector ``g(z)``."""
        return np.asarray(eval_g(self.problem, jnp.asarray(z), self.x0, self.t0, self.dt, xf=self.xf), dtype=np.float64)

    def jacobian(self, z: np.ndarray) -> FloatArray:
        """Evaluate sparse constraint Jacobian values."""
        return np.asarray(
            eval_jac_g(self.problem, jnp.asarray(z), self.t0, self.dt, xf=self.xf, gather=self.sparsity.gather),
            dtype=np.float64,
        )

    def jacobianstructure(self) -> tuple[np.ndarray, np.ndarray]:
        """Return compact int32 COO Jacobian sparsity pattern ``(rows, cols)``."""
        return self.sparsity.rows, self.sparsity.cols


@dataclass
class _MsHandle:
    """Cached cyipopt Problem and callback stored on ``Program.handles``."""

    nlp: Any
    cb: _MsCallback


@dataclass(frozen=True)
class MultipleShooting:
    """Shift-sparse full-state multiple-shooting Transcription for AutoregressiveModels using Ipopt."""

    solver: Ipopt

    def __post_init__(self) -> None:
        """Refuse non-Ipopt solvers at construction."""
        if not isinstance(self.solver, Ipopt):
            msg = f"MultipleShooting supports only Ipopt, got {type(self.solver).__name__}."
            raise TypeError(msg)

    def solve(self, program: Program, bc: BoundaryConditions, ws: WarmStart) -> IpoptResult:
        """Solve the shift-sparse multiple-shooting Transcription using Ipopt."""
        problem = program.problem
        model = _validate_ms_problem(problem)
        try:
            import cyipopt  # noqa: PLC0415 -- cyipopt is an optional solver dependency
        except ImportError as e:
            msg = "cyipopt is not installed. Install trajopt[solvers] to run MultipleShooting."
            raise ImportError(msg) from e

        N = int(problem.N)
        x0_arr, t0_arr, dt_arr, xf_val, z0_jax = parse_solver_initial_state(problem, bc, ws)
        problem = retarget_problem(problem, bc)
        dt_arr = jnp.broadcast_to(dt_arr, (N - 1,))

        z0 = np.asarray(z0_jax, dtype=np.float64)
        handle = cast("_MsHandle | None", program.handles.get("ar_multiple_shooting"))
        if handle is None:
            zL, zU = _ms_primal_bounds(problem)
            gL, gU = _ms_constraint_bounds(problem)
            cb = _MsCallback(problem, x0_arr, t0_arr, dt_arr, xf_val)
            problem_cls: Any = getattr(cyipopt, "Problem")  # noqa: B009 -- cyipopt is an untyped C-extension
            nlp = problem_cls(
                n=len(z0),
                m=len(gL),
                problem_obj=cb,
                lb=zL,
                ub=zU,
                cl=gL,
                cu=gU,
            )
            options = dict(self.solver.options)
            options.setdefault("hessian_approximation", "limited-memory")
            for k, val in options.items():
                nlp.add_option(k, val)
            handle = _MsHandle(nlp=nlp, cb=cb)
            program.handles["ar_multiple_shooting"] = cast("Any", handle)
        else:
            cb = handle.cb
            nlp = handle.nlp
            cb.update(problem, x0_arr, t0_arr, dt_arr, xf_val)

        z_opt, info = nlp.solve(z0)
        return self._build_result(
            problem,
            model,
            cb,
            np.asarray(z_opt, dtype=np.float64),
            info,
            x0_arr=x0_arr,
            t0_arr=t0_arr,
            dt_arr=dt_arr,
            xf_val=xf_val,
        )

    def _build_result(  # noqa: PLR0913 -- packaging result needs solve metadata
        self,
        problem: Problem,
        model: AutoregressiveModel,
        cb: _MsCallback,
        z_opt: FloatArray,
        info: dict[str, Any],
        *,
        x0_arr: jax.Array,
        t0_arr: jax.Array,
        dt_arr: jax.Array,
        xf_val: jax.Array | None,
    ) -> IpoptResult:
        """Package optimal canonical Primal Vector into ``IpoptResult``."""
        N = int(problem.N)
        n = int(model.n)
        m = int(model.m)
        horizon = N - 1

        status = int(info.get("status", -1))
        success = status in _SUCCESS_STATUSES
        message = str(info.get("status_msg", ""))

        z_opt_jax = jnp.asarray(z_opt, dtype=jnp.float64)
        X_opt, U_opt = _z_to_trajectory(z_opt_jax, N, n, m)
        t_opt = t0_arr + jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), jnp.cumsum(dt_arr)])
        opt_traj = Trajectory(X=X_opt, U=U_opt, t=t_opt, dt=dt_arr)

        cost_val = float(eval_f(problem, z_opt_jax, t0=t0_arr, dt=dt_arr))
        iter_count = int(cb.iteration_count)
        viol = compute_constraint_violation(problem, z_opt_jax, x0_arr, t0=t0_arr, dt=dt_arr, xf=xf_val)

        P_total = constraint_row_count(problem)
        nz_primal = N * n + horizon * m
        return IpoptResult(
            trajectory=opt_traj,
            success=success,
            status=status,
            message=message,
            cost=cost_val,
            Z=z_opt_jax,
            info=info,
            iterations=iter_count,
            constraint_violation=viol,
            lam=np.zeros(P_total, dtype=np.float64),
            mu=np.zeros(nz_primal, dtype=np.float64),
        )
