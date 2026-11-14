from __future__ import annotations

from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from trajopt.costs.l1 import L1ControlCost
from trajopt.costs.output import OutputCost
from trajopt.costs.pseudo_huber import PseudoHuberControlCost
from trajopt.costs.quadratic import DiagonalCost
from trajopt.costs.sum import cost_terms

from neuro.control.costs import (
    AffineObservableHinge,
    ExcludeInitialKnotState,
    ObservableFrameHingeCost,
    ObservableHingeCost,
)

if TYPE_CHECKING:
    from trajopt.costs.base import CostFunction
    from trajopt.dynamics.base import DiscreteDynamics
    from trajopt.problem import Problem

    from neuro.predictor.inference import (
        InferencePredictor,
    )


ACTIVE_CURRENT_THRESHOLD: float = 1e-3


def _knot_times(t: float | jax.Array, dt: float | jax.Array, n_knots: int) -> jax.Array:
    """Build knot times from a uniform step or one duration per interval."""
    steps = jnp.asarray(dt)
    if steps.ndim == 0:
        return t + jnp.arange(n_knots) * steps
    return t + jnp.concatenate((jnp.zeros(1, dtype=steps.dtype), jnp.cumsum(steps)))


@eqx.filter_jit
def planned_outputs(
    model: InferencePredictor, states: jax.Array, controls: jax.Array, t: float | jax.Array, dt: float | jax.Array
) -> jax.Array:
    """Project solved states to physical outputs on the uniform or nonuniform knot grid."""
    padded = jnp.concatenate((controls, controls[-1:]), axis=0)
    times = _knot_times(t, dt, states.shape[0])
    return jax.vmap(model.output)(states, padded, times)


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
    """Identify direct and affine Spectral Hinge Costs."""
    return (
        isinstance(c, (ObservableFrameHingeCost, ObservableHingeCost, AffineObservableHinge))
        or (isinstance(c, OutputCost) and isinstance(c.cost, (ObservableFrameHingeCost, ObservableHingeCost)))
        or (isinstance(c, ExcludeInitialKnotState) and _is_spectral_cost(c.inner))
    )


def _stage_leaf_contributions(
    c: CostFunction,
    states: jax.Array,
    controls: jax.Array,
    t_stage: jax.Array,
) -> tuple[float, float, float, float, float]:
    """Score spectral, quadratic, sparse, tracking and activation terms in physical coordinates."""
    full_val = float(jnp.sum(c.stage_costs(states[:-1], controls, t_stage)))
    if _is_spectral_cost(c):
        return full_val, 0.0, 0.0, 0.0, 0.0
    if isinstance(c, (L1ControlCost, PseudoHuberControlCost)):
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
    t: float | jax.Array = 0.0,
    dt: float | jax.Array = 1.0,
) -> dict[str, Any]:
    """Decompose the physical objective into spectral, effort, tracking and activation costs.

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
        Uniform sampling period or interval durations of shape ``(N - 1,)``.

    Returns
    -------
    dict[str, Any]
        Dictionary containing ``cost_spectral``, ``cost_quadratic_effort``,
        ``cost_sparse_effort``, ``cost_tracking``, ``cost_active``, ``active_count``,
        ``cost_total``, and ``normalization``.
    """
    times = _knot_times(t, dt, states.shape[0])
    t_stage, t_term = times[:-1], times[-1]

    spectral_val = 0.0
    quadratic_effort_val = 0.0
    sparse_effort_val = 0.0
    tracking_val = 0.0
    active_val = 0.0

    for c in cost_terms(problem.obj.stage_cost):
        s, qe, se, tr, act = _stage_leaf_contributions(c, states, controls, t_stage)
        spectral_val += s
        quadratic_effort_val += qe
        sparse_effort_val += se
        tracking_val += tr
        active_val += act

    for c in cost_terms(problem.obj.terminal_cost):
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
