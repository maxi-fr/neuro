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
    _trajectory_to_z,
    compute_constraint_violation,
    parse_solver_initial_state,
    retained_knot_evaluators,
)
from trajopt.transcription.result import constraint_row_count

from neuro.predictor.base import AutoregressiveModel

if TYPE_CHECKING:
    from trajopt.program import Program, WarmStart

    from neuro.types import FloatArray, IntArray


def _require_autoregressive(problem: Problem) -> AutoregressiveModel:
    """Return ``problem.model`` typed as ``AutoregressiveModel`` or raise ``TypeError``."""
    model = problem.model
    if not isinstance(model, AutoregressiveModel):
        msg = f"NarxShooting requires an AutoregressiveModel, got {type(model).__name__}."
        raise TypeError(msg)
    return model


def _validate_narx_problem(problem: Problem) -> AutoregressiveModel:
    """Validate that ``problem`` matches the 1-step NARX Transcription assumptions."""
    model = _require_autoregressive(problem)
    if int(problem.control_block_steps) != 1:
        msg = f"NarxShooting requires control_block_steps=1, got {problem.control_block_steps}."
        raise ValueError(msg)
    if problem.binary_control_indices:
        msg = "NarxShooting does not support binary control indices."
        raise ValueError(msg)
    x_lower, x_upper, _, _ = problem.constraints.primal_bounds()
    if np.any(np.isfinite(x_lower)) or np.any(np.isfinite(x_upper)):
        msg = "NarxShooting does not support finite state primal bounds; use full-state Ipopt instead."
        raise ValueError(msg)
    return model


def narx_dimensions(problem: Problem) -> tuple[int, int, int]:
    """Return NARX NLP dimensions ``(n_vars, n_defects, n_user)``."""
    model = _validate_narx_problem(problem)
    horizon = int(problem.N) - 1
    n_vars = horizon * (int(model.m) + int(model.n_outputs))
    n_defects = horizon * int(model.n_outputs)
    retained = retained_knot_evaluators(problem)
    n_user = int(sum(k.p for k in retained)) + sum(con.p for con in problem.constraints.horizon_constraints)
    return n_vars, n_defects, n_user


def _narx_primal_bounds(problem: Problem) -> tuple[FloatArray, FloatArray]:
    """Build lower and upper bounds ``zL <= v <= zU`` for interleaved ``(u_j, z_{j+1})`` blocks."""
    model = _validate_narx_problem(problem)
    horizon = int(problem.N) - 1
    n_o = int(model.n_outputs)
    _, _, u_lower, u_upper = problem.constraints.primal_bounds()
    u_lo = np.asarray(u_lower, dtype=np.float64)
    u_hi = np.asarray(u_upper, dtype=np.float64)
    z_lo = np.full((horizon, n_o), -np.inf, dtype=np.float64)
    z_hi = np.full((horizon, n_o), np.inf, dtype=np.float64)
    return np.concatenate([u_lo, z_lo], axis=1).reshape(-1), np.concatenate([u_hi, z_hi], axis=1).reshape(-1)


def _narx_constraint_bounds(problem: Problem) -> tuple[FloatArray, FloatArray]:
    """Build lower and upper bounds for NARX defects followed by user constraints."""
    model = _validate_narx_problem(problem)
    horizon = int(problem.N) - 1
    n_defects = horizon * int(model.n_outputs)
    g_lo: list[FloatArray] = [np.zeros(n_defects, dtype=np.float64)]
    g_hi: list[FloatArray] = [np.zeros(n_defects, dtype=np.float64)]

    retained = retained_knot_evaluators(problem)
    for k in range(horizon):
        lo_k, hi_k = _evaluator_bounds(retained[k])
        g_lo.extend(lo_k)
        g_hi.extend(hi_k)
    if len(retained) > horizon:
        lo_term, hi_term = _evaluator_bounds(retained[horizon])
        g_lo.extend(lo_term)
        g_hi.extend(hi_term)
    for con in problem.constraints.horizon_constraints:
        lo, hi = _cone_bounds(con.cone, con.p)
        g_lo.append(lo)
        g_hi.append(hi)
    return np.concatenate(g_lo), np.concatenate(g_hi)


def _unpack_trajectory(
    model: AutoregressiveModel,
    v: jax.Array,
    x0: jax.Array,
    horizon: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Reconstruct ``(X, U, Z_pred, y_wins, u_wins)`` from the NARX Primal Vector ``v``.

    Returns
    -------
    X
        Canonical state trajectory of shape ``(H + 1, n)``.
    U
        Control trajectory of shape ``(H, m)``.
    Z_pred
        Lifted standardized future outputs of shape ``(H, n_o)``.
    y_wins
        Standardized output history windows at each step, shape ``(H, n_y, n_o)``.
    u_wins
        Raw Control Current windows at each step, shape ``(H, n_u, m)``.
    """
    m = int(model.m)
    n_o = int(model.n_outputs)
    n_y = int(model.n_y)
    n_u = int(model.n_u)
    n_hist = int(model.n_history)

    v_blk = v.reshape((horizon, m + n_o))
    U = v_blk[:, :m]
    Z_pred = v_blk[:, m:]

    n_z0 = n_hist * n_o
    z_hist0 = x0[:n_z0].reshape((n_hist, n_o))
    u_hist0 = x0[n_z0:].reshape((n_u, m))
    z_all = jnp.concatenate([z_hist0, Z_pred], axis=0)
    u_all = jnp.concatenate([u_hist0, U], axis=0)

    def assemble_state(k: jax.Array) -> jax.Array:
        z_slice = jax.lax.dynamic_slice(z_all, (k, 0), (n_hist, n_o)).reshape(-1)
        u_slice = jax.lax.dynamic_slice(u_all, (k, 0), (n_u, m)).reshape(-1)
        return jnp.concatenate([z_slice, u_slice])

    X = jax.vmap(assemble_state)(jnp.arange(horizon + 1))

    y_offset = n_hist - n_y

    def step_windows(j: jax.Array) -> tuple[jax.Array, jax.Array]:
        yw = jax.lax.dynamic_slice(z_all, (y_offset + j, 0), (n_y, n_o))
        uw = jax.lax.dynamic_slice(u_all, (j + 1, 0), (n_u, m))
        return yw, uw

    y_wins, u_wins = jax.vmap(step_windows)(jnp.arange(horizon))
    return X, U, Z_pred, y_wins, u_wins


def _cost_fn(
    problem: Problem,
    v: jax.Array,
    x0: jax.Array,
    t0: float | jax.Array,
    dt: float | jax.Array,
) -> jax.Array:
    """Evaluate total Objective at the NARX Primal Vector ``v``."""
    model = _require_autoregressive(problem)
    horizon = int(problem.N) - 1
    X, U, _, _, _ = _unpack_trajectory(model, v, x0, horizon)
    dt_arr = jnp.broadcast_to(jnp.asarray(dt), (horizon,))
    t_arr = t0 + jnp.concatenate([jnp.zeros(1, dtype=v.dtype), jnp.cumsum(dt_arr)])
    obj = problem.obj
    stage_costs = obj.stage_cost.stage_costs(X[:-1], U, t_arr[:-1])
    term_cost = obj.terminal_cost.evaluate(X[-1], None, t_arr[-1])
    return jnp.sum(stage_costs) + term_cost


def _defect_step(model: AutoregressiveModel, y_win: jax.Array, u_win: jax.Array) -> jax.Array:
    """Evaluate one-step standardized output prediction from ``y_win`` and raw ``u_win``."""
    return model._predict(y_win, (u_win - model.u_center) / model.u_scale)  # noqa: SLF001 -- AutoregressiveModel defines _predict for its transcriptions


def _user_constraints_from_traj(  # noqa: PLR0913 -- takes unpacked trajectory arrays plus boundary parameters
    problem: Problem,
    X: jax.Array,
    U: jax.Array,
    *,
    t0: float | jax.Array,
    dt: float | jax.Array,
    xf: jax.Array | None = None,
) -> list[jax.Array]:
    """Evaluate retained knot and horizon constraints given unpacked ``(X, U)``."""
    horizon = int(problem.N) - 1
    dt_arr = jnp.broadcast_to(jnp.asarray(dt), (horizon,))
    t_arr = t0 + jnp.concatenate([jnp.zeros(1, dtype=U.dtype), jnp.cumsum(dt_arr)])
    retained = retained_knot_evaluators(problem)

    parts: list[jax.Array] = []
    for k in range(horizon):
        evaluator = retained[k]
        if evaluator.p > 0:
            parts.append(evaluator.evaluate(X[k], U[k], t_arr[k], xf=xf))
    if len(retained) > horizon and retained[horizon].p > 0:
        parts.append(retained[horizon].evaluate(X[-1], None, t_arr[-1], xf=xf))
    if problem.constraints.horizon_constraints:
        Z_canon = _trajectory_to_z(X, U)
        parts.extend(con.evaluate(Z_canon) for con in problem.constraints.horizon_constraints)
    return parts


def _constraints_fn(  # noqa: PLR0913, PLR0917 -- constraint callback takes 6 arguments
    problem: Problem,
    v: jax.Array,
    x0: jax.Array,
    t0: float | jax.Array,
    dt: float | jax.Array,
    xf: jax.Array | None = None,
) -> jax.Array:
    """Evaluate NARX defect equalities followed by retained knot and horizon constraints."""
    model = _require_autoregressive(problem)
    horizon = int(problem.N) - 1
    X, U, Z_pred, y_wins, u_wins = _unpack_trajectory(model, v, x0, horizon)
    preds = jax.vmap(lambda yw, uw: _defect_step(model, yw, uw))(y_wins, u_wins)
    defects = (Z_pred - preds).reshape(-1)
    user_parts = _user_constraints_from_traj(problem, X, U, t0=t0, dt=dt, xf=xf)
    return jnp.concatenate([defects, *user_parts])


def _user_constraints_fn(  # noqa: PLR0913, PLR0917 -- user constraint evaluator mirrors _constraints_fn
    problem: Problem,
    v: jax.Array,
    x0: jax.Array,
    t0: float | jax.Array,
    dt: float | jax.Array,
    xf: jax.Array | None = None,
) -> jax.Array:
    """Evaluate retained knot and horizon constraints without dynamics defects."""
    model = _require_autoregressive(problem)
    horizon = int(problem.N) - 1
    X, U, _, _, _ = _unpack_trajectory(model, v, x0, horizon)
    parts = _user_constraints_from_traj(problem, X, U, t0=t0, dt=dt, xf=xf)
    if not parts:
        return jnp.zeros(0, dtype=v.dtype)
    return jnp.concatenate(parts)


@eqx.filter_jit
def eval_f(
    problem: Problem,
    v: jax.Array,
    x0: jax.Array,
    t0: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.05,
) -> jax.Array:
    """Evaluate scalar Objective ``J(v)`` for the NARX Primal Vector."""
    return _cost_fn(problem, v, x0, t0, dt)


@eqx.filter_jit
def eval_grad_f(
    problem: Problem,
    v: jax.Array,
    x0: jax.Array,
    t0: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.05,
) -> jax.Array:
    """Evaluate Objective gradient ``nabla J(v)``."""
    return jax.grad(lambda v_: _cost_fn(problem, v_, x0, t0, dt))(v)


@eqx.filter_jit
def eval_g(  # noqa: PLR0913, PLR0917 -- constraint evaluation takes 6 arguments
    problem: Problem,
    v: jax.Array,
    x0: jax.Array,
    t0: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.05,
    xf: jax.Array | None = None,
) -> jax.Array:
    """Evaluate combined NARX defects and user constraint vector ``g(v)``."""
    return _constraints_fn(problem, v, x0, t0, dt, xf=xf)


class _NarxGatherIndices(eqx.Module):
    """Static JAX index arrays for gathering sparse Jacobian blocks."""

    n_defects: int = eqx.field(static=True)
    y_step_idx: jax.Array
    y_slot_idx: jax.Array
    u_step_idx: jax.Array
    u_slot_idx: jax.Array
    user_rows: jax.Array
    user_cols: jax.Array


@dataclass(frozen=True)
class _NarxSparsity:
    """Precomputed COO sparsity pattern and JAX gather indices for NARX constraints."""

    rows: IntArray
    cols: IntArray
    gather: _NarxGatherIndices


def _knot_active_columns(k: int, model: AutoregressiveModel, *, include_u: bool) -> IntArray:
    """Return Primal Vector column indices contributing to ``(X[k], U[k])``."""
    m = int(model.m)
    n_o = int(model.n_outputs)
    n_hist = int(model.n_history)
    n_u = int(model.n_u)
    blk = m + n_o
    out_ids = np.arange(n_o, dtype=np.intp)
    ctrl_ids = np.arange(m, dtype=np.intp)
    cols = [(s - 1) * blk + m + out_ids for s in range(max(1, k - n_hist + 1), k + 1)]
    cols.extend(r * blk + ctrl_ids for r in range(max(0, k - n_u), k))
    if include_u:
        cols.append(k * blk + ctrl_ids)
    return np.concatenate(cols) if cols else np.zeros(0, dtype=np.intp)


def _defect_sparsity(
    horizon: int,
    model: AutoregressiveModel,
) -> tuple[list[IntArray], list[IntArray], list[int], list[int], list[int], list[int]]:
    """Build COO sparsity coordinates and step/slot gather lists for NARX defect rows."""
    m = int(model.m)
    n_o = int(model.n_outputs)
    n_y = int(model.n_y)
    n_u = int(model.n_u)
    blk = m + n_o
    step_ids = np.arange(horizon, dtype=np.intp)
    out_ids = np.arange(n_o, dtype=np.intp)
    ctrl_ids = np.arange(m, dtype=np.intp)

    row_parts: list[IntArray] = [(step_ids[:, None] * n_o + out_ids[None, :]).reshape(-1)]
    col_parts: list[IntArray] = [(step_ids[:, None] * blk + m + out_ids[None, :]).reshape(-1)]

    y_steps: list[int] = []
    y_slots: list[int] = []
    for j in range(horizon):
        for s in range(max(1, j - n_y + 1), j + 1):
            y_steps.append(j)
            y_slots.append(n_y - 1 - (j - s))
            row_parts.append(np.repeat(j * n_o + out_ids, n_o))
            col_parts.append(np.tile((s - 1) * blk + m + out_ids, n_o))

    u_steps: list[int] = []
    u_slots: list[int] = []
    for j in range(horizon):
        for r in range(max(0, j - n_u + 1), j + 1):
            u_steps.append(j)
            u_slots.append(n_u - 1 - (j - r))
            row_parts.append(np.repeat(j * n_o + out_ids, m))
            col_parts.append(np.tile(r * blk + ctrl_ids, n_o))

    return row_parts, col_parts, y_steps, y_slots, u_steps, u_slots


def _user_sparsity(problem: Problem, model: AutoregressiveModel, n_defects: int) -> tuple[IntArray, IntArray]:
    """Build COO sparsity coordinates for retained knot and horizon user constraints."""
    horizon = int(problem.N) - 1
    blk = int(model.m) + int(model.n_outputs)
    retained = retained_knot_evaluators(problem)
    user_row_list: list[IntArray] = []
    user_col_list: list[IntArray] = []
    curr_row = n_defects

    for k in range(horizon):
        p_k = int(retained[k].p)
        if p_k > 0:
            active_cols = _knot_active_columns(k, model, include_u=True)
            user_row_list.append(np.repeat(np.arange(curr_row, curr_row + p_k, dtype=np.intp), len(active_cols)))
            user_col_list.append(np.tile(active_cols, p_k))
            curr_row += p_k

    if len(retained) > horizon and int(retained[horizon].p) > 0:
        p_t = int(retained[horizon].p)
        active_cols = _knot_active_columns(horizon, model, include_u=False)
        user_row_list.append(np.repeat(np.arange(curr_row, curr_row + p_t, dtype=np.intp), len(active_cols)))
        user_col_list.append(np.tile(active_cols, p_t))
        curr_row += p_t

    n_vars = horizon * blk
    all_cols = np.arange(n_vars, dtype=np.intp)
    for con in problem.constraints.horizon_constraints:
        if con.p > 0:
            user_row_list.append(np.repeat(np.arange(curr_row, curr_row + con.p, dtype=np.intp), n_vars))
            user_col_list.append(np.tile(all_cols, con.p))
            curr_row += con.p

    user_rows = np.concatenate(user_row_list) if user_row_list else np.zeros(0, dtype=np.intp)
    user_cols = np.concatenate(user_col_list) if user_col_list else np.zeros(0, dtype=np.intp)
    return user_rows, user_cols


def _build_narx_sparsity(problem: Problem, model: AutoregressiveModel) -> _NarxSparsity:
    """Build exact COO sparsity structure for NARX defects and user constraints."""
    horizon = int(problem.N) - 1
    n_defects = horizon * int(model.n_outputs)
    row_parts, col_parts, y_steps, y_slots, u_steps, u_slots = _defect_sparsity(horizon, model)
    user_rows, user_cols = _user_sparsity(problem, model, n_defects)
    if len(user_rows):
        row_parts.append(user_rows)
        col_parts.append(user_cols)

    gather = _NarxGatherIndices(
        n_defects=n_defects,
        y_step_idx=jnp.asarray(y_steps, dtype=jnp.int32),
        y_slot_idx=jnp.asarray(y_slots, dtype=jnp.int32),
        u_step_idx=jnp.asarray(u_steps, dtype=jnp.int32),
        u_slot_idx=jnp.asarray(u_slots, dtype=jnp.int32),
        user_rows=jnp.asarray(user_rows - n_defects if len(user_rows) else user_rows, dtype=jnp.int32),
        user_cols=jnp.asarray(user_cols, dtype=jnp.int32),
    )
    return _NarxSparsity(
        rows=np.concatenate(row_parts).astype(np.intp),
        cols=np.concatenate(col_parts).astype(np.intp),
        gather=gather,
    )


@eqx.filter_jit
def eval_jac_g(  # noqa: PLR0913, PLR0917 -- Jacobian callback takes 6 arguments plus gather indices
    problem: Problem,
    v: jax.Array,
    x0: jax.Array,
    t0: float | jax.Array = 0.0,
    dt: float | jax.Array = 0.05,
    xf: jax.Array | None = None,
    *,
    gather: _NarxGatherIndices,
) -> jax.Array:
    """Evaluate nonzero COO values of the NARX constraint Jacobian."""
    model = _require_autoregressive(problem)
    horizon = int(problem.N) - 1
    _, _, _, y_wins, u_wins = _unpack_trajectory(model, v, x0, horizon)

    def jac_step(yw: jax.Array, uw: jax.Array) -> tuple[jax.Array, jax.Array]:
        return jax.jacrev(_defect_step, argnums=(1, 2))(model, yw, uw)

    jac_y, jac_u = jax.vmap(jac_step)(y_wins, u_wins)
    id_vals = jnp.ones(gather.n_defects, dtype=v.dtype)
    parts: list[jax.Array] = [id_vals]

    if gather.y_step_idx.size:
        y_vals = -jac_y[gather.y_step_idx, :, gather.y_slot_idx, :]
        parts.append(y_vals.reshape(-1))
    if gather.u_step_idx.size:
        u_vals = -jac_u[gather.u_step_idx, :, gather.u_slot_idx, :]
        parts.append(u_vals.reshape(-1))
    if gather.user_rows.size:
        jac_user = jax.jacrev(lambda v_: _user_constraints_fn(problem, v_, x0, t0, dt, xf=xf))(v)
        parts.append(jac_user[gather.user_rows, gather.user_cols])

    return jnp.concatenate(parts)


class _NarxCallback:
    """Callback wrapper connecting NARX JAX kernels to cyipopt."""

    def __init__(
        self,
        problem: Problem,
        x0: jax.Array,
        t0: float | jax.Array,
        dt: float | jax.Array,
        xf: jax.Array | None = None,
    ) -> None:
        model = _validate_narx_problem(problem)
        self.problem = problem
        self.x0 = x0
        self.t0 = t0
        self.dt = dt
        self.xf = xf
        self.sparsity = _build_narx_sparsity(problem, model)
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

    def objective(self, v: np.ndarray) -> float:
        """Evaluate NARX Objective ``J(v)``."""
        return float(eval_f(self.problem, jnp.asarray(v), self.x0, self.t0, self.dt))

    def gradient(self, v: np.ndarray) -> FloatArray:
        """Evaluate Objective gradient ``nabla J(v)``."""
        return np.asarray(eval_grad_f(self.problem, jnp.asarray(v), self.x0, self.t0, self.dt), dtype=np.float64)

    def constraints(self, v: np.ndarray) -> FloatArray:
        """Evaluate NARX defect and user constraint vector ``g(v)``."""
        return np.asarray(
            eval_g(self.problem, jnp.asarray(v), self.x0, self.t0, self.dt, xf=self.xf), dtype=np.float64
        )

    def jacobian(self, v: np.ndarray) -> FloatArray:
        """Evaluate sparse constraint Jacobian values."""
        return np.asarray(
            eval_jac_g(
                self.problem,
                jnp.asarray(v),
                self.x0,
                self.t0,
                self.dt,
                xf=self.xf,
                gather=self.sparsity.gather,
            ),
            dtype=np.float64,
        )

    def jacobianstructure(self) -> tuple[IntArray, IntArray]:
        """Return build-time constraint Jacobian sparsity pattern ``(rows, cols)``."""
        return self.sparsity.rows, self.sparsity.cols


@dataclass
class _NarxHandle:
    """Cached cyipopt Problem and NARX callback stored on ``Program.handles``."""

    nlp: Any
    cb: _NarxCallback


def _warm_start_to_narx_v0(problem: Problem, ws: WarmStart) -> FloatArray:
    """Pack canonical ``WarmStart`` ``(X, U)`` into interleaved NARX Primal Vector ``v0``."""
    model = _validate_narx_problem(problem)
    X_guess, U_guess = ws.unpack(problem)
    n_o = int(model.n_outputs)
    n_hist = int(model.n_history)
    z_guess = X_guess[1:, (n_hist - 1) * n_o : n_hist * n_o]
    v_blk = jnp.concatenate([U_guess, z_guess], axis=1)
    return np.asarray(v_blk.reshape(-1), dtype=np.float64)


@dataclass(frozen=True)
class NarxShooting:
    """Output-lifted 1-step NARX Transcription using Ipopt."""

    solver: Ipopt

    def __post_init__(self) -> None:
        """Refuse non-Ipopt solvers at construction."""
        if not isinstance(self.solver, Ipopt):
            msg = f"NarxShooting supports only Ipopt, got {type(self.solver).__name__}."
            raise TypeError(msg)

    def solve(self, program: Program, bc: BoundaryConditions, ws: WarmStart) -> IpoptResult:
        """Solve the 1-step NARX Transcription of ``program``'s Problem using Ipopt."""
        problem = program.problem
        model = _validate_narx_problem(problem)
        try:
            import cyipopt  # noqa: PLC0415 -- cyipopt is an optional solver dependency
        except ImportError as e:
            msg = "cyipopt is not installed. Install trajopt[solvers] to run NarxShooting."
            raise ImportError(msg) from e

        N = int(problem.N)
        x0_arr, t0_arr, dt_arr, xf_val, _ = parse_solver_initial_state(problem, bc, ws)
        problem = retarget_problem(problem, bc)
        dt_arr = jnp.broadcast_to(dt_arr, (N - 1,))

        v0 = _warm_start_to_narx_v0(problem, ws)
        handle = cast("_NarxHandle | None", program.handles.get("narx_shooting"))
        if handle is None:
            zL, zU = _narx_primal_bounds(problem)
            gL, gU = _narx_constraint_bounds(problem)
            cb = _NarxCallback(problem, x0_arr, t0_arr, dt_arr, xf_val)
            problem_cls: Any = getattr(cyipopt, "Problem")  # noqa: B009 -- cyipopt is an untyped C-extension
            nlp = problem_cls(
                n=len(v0),
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
            handle = _NarxHandle(nlp=nlp, cb=cb)
            program.handles["narx_shooting"] = cast("Any", handle)
        else:
            cb = handle.cb
            nlp = handle.nlp
            cb.update(problem, x0_arr, t0_arr, dt_arr, xf_val)

        v_opt, info = nlp.solve(v0)
        return self._build_result(
            problem,
            model,
            cb,
            np.asarray(v_opt, dtype=np.float64),
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
        cb: _NarxCallback,
        v_opt: FloatArray,
        info: dict[str, Any],
        *,
        x0_arr: jax.Array,
        t0_arr: jax.Array,
        dt_arr: jax.Array,
        xf_val: jax.Array | None,
    ) -> IpoptResult:
        """Unpack NARX Primal Vector into canonical Trajectory and IpoptResult."""
        N = int(problem.N)
        n = int(model.n)
        m = int(model.m)
        horizon = N - 1

        status = int(info.get("status", -1))
        success = status in _SUCCESS_STATUSES
        message = str(info.get("status_msg", ""))

        v_opt_jax = jnp.asarray(v_opt, dtype=jnp.float64)
        X_opt, U_opt, _, _, _ = _unpack_trajectory(model, v_opt_jax, x0_arr, horizon)
        Z_opt_jax = _trajectory_to_z(X_opt, U_opt)
        t_opt = t0_arr + jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), jnp.cumsum(dt_arr)])
        opt_traj = Trajectory(X=X_opt, U=U_opt, t=t_opt, dt=dt_arr)

        cost_val = float(eval_f(problem, v_opt_jax, x0_arr, t0=t0_arr, dt=dt_arr))
        iter_count = int(cb.iteration_count)
        viol = compute_constraint_violation(problem, Z_opt_jax, x0_arr, t0=t0_arr, dt=dt_arr, xf=xf_val)

        P_total = constraint_row_count(problem)
        nz_primal = N * n + horizon * m
        return IpoptResult(
            trajectory=opt_traj,
            success=success,
            status=status,
            message=message,
            cost=cost_val,
            Z=Z_opt_jax,
            info=info,
            iterations=iter_count,
            constraint_violation=viol,
            lam=np.zeros(P_total, dtype=np.float64),
            mu=np.zeros(nz_primal, dtype=np.float64),
        )
