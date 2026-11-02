from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neuro.config import EegMsGeometry, TrainingConfig
from neuro.predictor import gradient
from neuro.predictor.cnn import WaveformCNNModel
from neuro.predictor.data import TrajectoryWindowDataset
from neuro.predictor.losses import CurriculumMSE, EegMsLoss, LossContext, total_loss
from neuro.predictor.mlp import WaveformMLPModel

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    import optax

    from neuro.predictor.base import AutoregressiveModel
    from neuro.predictor.losses import Loss


def _model(architecture: str = "mlp") -> AutoregressiveModel:
    """Build a tiny Predictor with valid temporal support and nontrivial Standardizers."""
    cls = WaveformMLPModel if architecture == "mlp" else WaveformCNNModel
    return cls(
        n_y=3,
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


def _dataset(n_windows: int = 2) -> TrajectoryWindowDataset:
    """Build standardized continuous data with three history samples and three-step targets."""
    n_steps = n_windows + 5
    y = np.zeros((n_steps, 1))
    y[:3] = 1.0
    return TrajectoryWindowDataset([(np.ones((n_steps, 1)), y)], n_y=3, n_u=2, horizon=3)


def _context() -> LossContext:
    """Use the tiny Predictor's physical output scaling for configured Losses."""
    return LossContext(jnp.array([2.0]), jnp.array([3.0]), 1000.0)


def test_training_time_projection_stops_after_three_epochs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Abort a window-based fit after three epochs when it projects beyond seven hours."""
    clock = iter([0.0, 0.0, 10.0, 10.0, 210.0, 210.0, 410.0])
    monkeypatch.setattr(gradient, "perf_counter", lambda: next(clock))
    cfg = TrainingConfig(epochs=150, max_training_hours=7.0, eval_horizon_s=0.1)

    with pytest.raises(gradient.TrainingTimeLimitError) as caught:
        gradient.fit_gradient_descent(
            _model(),
            _dataset(),
            _dataset(),
            cfg,
            seed=1,
            ctx=_context(),
            losses=[CurriculumMSE(1.0, 3, 0, 0, 0)],
            eligibility_start=80,
        )

    assert caught.value.completed_epochs == 3
    assert caught.value.elapsed_s == 410.0
    assert caught.value.projected_s == 29810.0
    assert caught.value.limit_s == 25200.0


def test_training_time_projection_excludes_first_epoch_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exclude compilation time when projecting the remaining window-based fit duration."""
    clock = iter([0.0, 0.0, 20000.0, 20000.0, 20010.0, 20010.0, 20020.0, 20020.0, 20030.0])
    monkeypatch.setattr(gradient, "perf_counter", lambda: next(clock))
    fit = gradient.fit_gradient_descent(
        _model(),
        _dataset(),
        _dataset(),
        TrainingConfig(epochs=4, max_training_hours=7.0, eval_horizon_s=0.1),
        seed=1,
        ctx=_context(),
        losses=[CurriculumMSE(1.0, 3, 0, 0, 0)],
    )
    assert len(fit.train_losses) == 4


def test_training_time_projection_rechecks_after_initial_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Recheck the duration limit when window-based training slows after the initial probe."""
    clock = iter([0.0, 0.0, 10.0, 10.0, 20.0, 20.0, 30.0, 30.0, 430.0])
    monkeypatch.setattr(gradient, "perf_counter", lambda: next(clock))
    with pytest.raises(gradient.TrainingTimeLimitError) as caught:
        gradient.fit_gradient_descent(
            _model(),
            _dataset(),
            _dataset(),
            TrainingConfig(epochs=150, max_training_hours=7.0, eval_horizon_s=0.1),
            seed=1,
            ctx=_context(),
            losses=[CurriculumMSE(1.0, 3, 0, 0, 0)],
            eligibility_start=80,
        )
    assert caught.value.completed_epochs == 4
    assert caught.value.projected_s == 30360.0


@pytest.mark.parametrize("architecture", ["mlp", "cnn"])
def test_weight_decay_preserves_standardizers(architecture: str) -> None:
    """Train each supported architecture while retaining its fixed Standardizer buffers."""
    model = _model(architecture)
    names = ("y_center", "y_scale", "u_center", "u_scale")
    before = [np.asarray(getattr(model, name)).copy() for name in names]

    fit = gradient.fit_gradient_descent(
        model,
        _dataset(),
        _dataset(),
        TrainingConfig(epochs=2, learning_rate=0.05, weight_decay=0.2, eval_horizon_s=0.1),
        seed=1,
        ctx=_context(),
        losses=[CurriculumMSE(1.0, 3, 0, 0, 0)],
    )
    for name, expected in zip(names, before, strict=True):
        np.testing.assert_array_equal(np.asarray(getattr(fit.best_model, name)), expected)
    weights_name = "weights" if architecture == "mlp" else "conv_weights"
    assert any(
        not np.array_equal(before, after)
        for before, after in zip(getattr(model, weights_name), getattr(fit.best_model, weights_name), strict=True)
    )


def test_curriculum_and_delayed_loss_do_not_retrace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Train real windows without retracing curriculum or delayed Loss activation."""
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
        _dataset(),
        _dataset(),
        TrainingConfig(epochs=5, patience=10, eval_horizon_s=0.1),
        seed=1,
        ctx=_context(),
        losses=losses,
    )
    assert traces.count(True) == 1
    assert fit.train_components["eeg_ms"][:2] == [None, None]
    assert all(value is not None for value in fit.train_components["eeg_ms"][2:])
    assert fit.train_components["L"][0] == 1
    assert fit.train_components["L"][-1] == 3


def test_window_dataset_schedule_counts_every_update(monkeypatch: pytest.MonkeyPatch) -> None:
    """Count all minibatch updates when building the warm-up and cosine schedules."""
    budgets = []
    original = gradient.lr_schedule

    def record(*, warmup_steps: int, total_steps: int, learning_rate: float) -> optax.Schedule:
        """Record the update budget that controls the actual learning rate."""
        budgets.append((warmup_steps, total_steps))
        return original(warmup_steps=warmup_steps, total_steps=total_steps, learning_rate=learning_rate)

    monkeypatch.setattr(gradient, "lr_schedule", record)
    gradient.fit_gradient_descent(
        _model(),
        _dataset(14),
        _dataset(),
        TrainingConfig(epochs=3, warmup_epochs=1, batch_size=2, patience=10, eval_horizon_s=0.1),
        seed=1,
        ctx=_context(),
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
