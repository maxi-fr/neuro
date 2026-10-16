from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neuro.config import EegMsGeometry, TrainingConfig
from neuro.predictor import gradient
from neuro.predictor.cnn import WaveformCNNModel
from neuro.predictor.losses import CurriculumMSE, EegMsLoss, LossContext, total_loss
from neuro.predictor.mlp import WaveformMLPModel

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    import optax

    from neuro.predictor.base import AutoregressiveModel
    from neuro.predictor.losses import Loss


def _model(architecture: str = "mlp") -> AutoregressiveModel:
    """Build a tiny Predictor with nontrivial Standardizers."""
    cls = WaveformMLPModel if architecture == "mlp" else WaveformCNNModel
    return cls(
        n_y=2,
        n_u=2,
        horizon=3,
        n_channels=1,
        n_controls=1,
        n_outputs=1,
        hidden_size=2,
        depth=1,
        y_center=np.array([2.0]),
        y_scale=np.array([3.0]),
        u_center=np.array([4.0]),
        u_scale=np.array([5.0]),
    )


def _batch() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return one batch with a three-step target."""
    return (jnp.ones((2, 2, 1)), jnp.ones((2, 2, 1)), jnp.ones((2, 3, 1)), jnp.zeros((2, 3, 1)))


@pytest.mark.parametrize("architecture", ["mlp", "cnn"])
@pytest.mark.parametrize("custom", [False, True])
def test_weight_decay_preserves_standardizers(architecture: str, *, custom: bool) -> None:
    model = _model(architecture)
    names = ("y_center", "y_scale", "u_center", "u_scale")
    before = [np.asarray(getattr(model, name)).copy() for name in names]

    def mse(
        mod: AutoregressiveModel,
        y: jax.Array,
        uh: jax.Array,
        uf: jax.Array,
        target: jax.Array,
        epoch: int | None,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Score the Predictor using the custom Loss entry point."""
        del epoch
        return jnp.mean((mod.rollout(y, uh, uf) - target) ** 2), {}

    fit = gradient.fit_gradient_descent(
        model,
        [_batch()],
        [_batch()],
        TrainingConfig(epochs=2, learning_rate=0.05, weight_decay=0.2, eval_horizon_s=0.1),
        seed=1,
        losses=[CurriculumMSE(1.0, 3, 0, 0, 0)],
        loss_fn=mse if custom else None,
    )
    for name, expected in zip(names, before, strict=True):
        np.testing.assert_array_equal(np.asarray(getattr(fit.best_model, name)), expected)
    weights_name = "weights" if architecture == "mlp" else "conv_weights"
    assert any(
        not np.array_equal(before, after)
        for before, after in zip(getattr(model, weights_name), getattr(fit.best_model, weights_name), strict=True)
    )


def test_curriculum_and_delayed_loss_do_not_retrace(monkeypatch: pytest.MonkeyPatch) -> None:
    traces = []
    original = gradient.total_loss

    def record(
        terms: Sequence[Loss],
        pred: jax.Array,
        true: jax.Array,
        ctx: LossContext,
        *,
        step_mask: jax.Array | None = None,
        active_mask: jax.Array | None = None,
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Count Loss traces while forwarding dynamic schedule masks."""
        traces.append(step_mask is not None)
        return original(terms, pred, true, ctx, step_mask=step_mask, active_mask=active_mask)

    monkeypatch.setattr(gradient, "total_loss", record)
    losses = [
        CurriculumMSE(1.0, 3, 0, 0, 4),
        EegMsLoss(0.1, 3, 2, EegMsGeometry(window_s=0.002, hop_s=0.001)),
    ]
    fit = gradient.fit_gradient_descent(
        _model(),
        [_batch()],
        [_batch()],
        TrainingConfig(epochs=5, patience=10, eval_horizon_s=0.1),
        seed=1,
        losses=losses,
        ctx=LossContext(jnp.array([2.0]), jnp.array([3.0]), 1000.0),
    )
    assert traces.count(True) == 1
    assert fit.train_components["eeg_ms"][:2] == [None, None]
    assert all(value is not None for value in fit.train_components["eeg_ms"][2:])
    assert fit.train_components["L"][0] == 1
    assert fit.train_components["L"][-1] == 3


def test_prebatched_schedule_counts_every_update(monkeypatch: pytest.MonkeyPatch) -> None:
    budgets = []
    original = gradient.lr_schedule

    def record(*, warmup_steps: int, total_steps: int, learning_rate: float) -> optax.Schedule:
        """Record the update budget that controls the actual learning rate."""
        budgets.append((warmup_steps, total_steps))
        return original(warmup_steps=warmup_steps, total_steps=total_steps, learning_rate=learning_rate)

    monkeypatch.setattr(gradient, "lr_schedule", record)
    gradient.fit_gradient_descent(
        _model(),
        [_batch()] * 7,
        [_batch()],
        TrainingConfig(epochs=3, warmup_epochs=1, batch_size=32, patience=10, eval_horizon_s=0.1),
        seed=1,
        losses=[CurriculumMSE(1.0, 3, 0, 0, 0)],
    )
    assert budgets == [(7, 21)]


@pytest.mark.parametrize("active", [False, True])
def test_dynamic_activation_preserves_loss_values_and_gradients(*, active: bool) -> None:
    losses = [
        CurriculumMSE(1.0, 3, 0, 0, 0),
        EegMsLoss(0.1, 3, 2, EegMsGeometry(window_s=0.002, hop_s=0.001)),
    ]
    ctx = LossContext(jnp.array([2.0]), jnp.array([3.0]), 1000.0)
    pred = jnp.full((2, 3, 1), 0.5)
    target = jnp.zeros_like(pred)

    def masked(p: jax.Array) -> jax.Array:
        """Score the dynamically selected Loss terms."""
        return total_loss(losses, p, target, ctx, active_mask=jnp.array([True, active]))[0]

    def reference(p: jax.Array) -> jax.Array:
        """Score only the Loss terms active at this epoch."""
        return total_loss(losses if active else losses[:1], p, target, ctx)[0]

    actual_value, actual_grad = jax.jit(jax.value_and_grad(masked))(pred)
    expected_value, expected_grad = jax.value_and_grad(reference)(pred)
    np.testing.assert_allclose(actual_value, expected_value, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual_grad, expected_grad, rtol=1e-6, atol=1e-6)
