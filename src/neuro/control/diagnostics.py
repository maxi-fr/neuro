from __future__ import annotations

from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from trajopt.costs.output import OutputCost
from trajopt.costs.pseudo_huber import PseudoHuberControlCost
from trajopt.costs.quadratic import DiagonalCost
from trajopt.problem import (
    Problem,
    _EpigraphStageCost,
)

from neuro.control.costs import (
    ExcludeInitialKnotState,
    ObservableFrameHingeCost,
    ObservableHingeCost,
    SumCost,
)

if TYPE_CHECKING:
    from trajopt.costs.base import CostFunction
    from trajopt.dynamics.base import DiscreteDynamics

    from neuro.predictor.inference import (
        InferencePredictor,
    )


ACTIVE_CURRENT_THRESHOLD: float = 1e-3


@eqx.filter_jit
def planned_outputs(
    model: InferencePredictor, states: jax.Array, controls: jax.Array, t: float, dt: float
) -> jax.Array:
    """Project every solved state to physical outputs, including the initial and terminal knots."""
    padded = jnp.concatenate((controls, controls[-1:]), axis=0)
    times = t + jnp.arange(states.shape[0]) * dt
    return jax.vmap(model.output)(states, padded, times)


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
    """Split a physical-current stage Cost into spectral, quadratic, sparse, tracking and activation terms."""
    if isinstance(c, _EpigraphStageCost):
        m_orig = c.original.m
        sparse_val = float(jnp.sum(c.weight * jnp.sum(controls[:, m_orig:], axis=1)))
        spectral_acc = 0.0
        quad_acc = 0.0
        sparse_acc = sparse_val
        track_acc = 0.0
        act_acc = 0.0
        for sub_c in _collect_leaf_costs(c.original):
            s, qe, se, tr, act = _stage_leaf_contributions(sub_c, states, controls[:, :m_orig], t_stage)
            spectral_acc += s
            quad_acc += qe
            sparse_acc += se
            track_acc += tr
            act_acc += act
        return spectral_acc, quad_acc, sparse_acc, track_acc, act_acc
    full_val = float(jnp.sum(c.stage_costs(states[:-1], controls, t_stage)))
    if _is_spectral_cost(c):
        return full_val, 0.0, 0.0, 0.0, 0.0
    if isinstance(c, PseudoHuberControlCost):
        return 0.0, 0.0, full_val, 0.0, 0.0
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
        m_phys = getattr(problem.model, "n_controls", ctrls_arr.shape[-1])
        if ctrls_arr.shape[-1] == 2 * m_phys:
            ctrls_arr = ctrls_arr[:, :m_phys]
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
