from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neuro.config import (
    CurriculumMSESpec,
    EegMsSpec,
    LossSpecs,
    ModelConfig,
    NNPredictorConfig,
    SimulationConfig,
    StftGeometry,
    StftSpec,
    TrainingConfig,
)
from neuro.predictor import gradient
from neuro.predictor.data import TrajectoryWindowDataset
from neuro.predictor.gradient import fit_gradient_descent
from neuro.predictor.losses import (
    CurriculumMSE,
    EegMsLoss,
    LossContext,
    StftLoss,
    build_losses,
    eligibility_start_epoch,
    total_loss,
)
from neuro.predictor.mlp import WaveformMLPModel
from neuro.predictor.train import TrainingResult, train

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from neuro.predictor.losses import Loss

_SEED = 42
_DT = 1e-3
_T, _N_EEG, _N_CONTROLS = 200, 3, 2
_N_Y, _N_U, _HORIZON = 2, 2, 4


def _write_trajectories(tmp_path: Path) -> list[str]:
    """Write synthetic trajectories for Trainer tests."""
    rng = np.random.default_rng(_SEED)
    files = []
    for i in range(2):
        u = np.cumsum(rng.standard_normal((_T, _N_CONTROLS)) * 0.1, axis=0)
        phase = np.arange(_T)[:, None] * _DT * 2 * np.pi * np.array([7.0, 11.0, 13.0])
        y = np.sin(phase) + 0.3 * u[:, :1] + 0.05 * rng.standard_normal((_T, _N_EEG))
        path = tmp_path / f"sim_{i:03d}.npz"
        np.savez(path, **{"sensor_0.y_mea": y, "controller.u": u})  # ty: ignore[invalid-argument-type]
        files.append(str(path))
    return files


def test_reproducing_regression_delayed_loss_and_curriculum_with_early_minimum(tmp_path: Path) -> None:
    """A run with delayed Loss, curriculum, and early minimum must select an eligible checkpoint.

    Under legacy behavior:
    1. Early stopping patience started at epoch 0, so early non-improvement aborted training
       before the delayed Loss ever ran.
    2. Final checkpoint selection restored epoch 0, which only received 1-step MSE training.
    Under fixed behavior:
    - Patience only starts in the eligible phase.
    - Checkpoint selection is restricted to eligible epochs.
    - The restored model and selected_epoch come from the eligible phase.
    """
    files = _write_trajectories(tmp_path)
    fs = 1.0 / _DT
    span_s = _HORIZON / fs

    curr_start, curr_end, delayed_start = 0, 4, 4
    losses = LossSpecs(
        curriculum_mse=CurriculumMSESpec(
            weight=1.0, span_s=span_s, curr_start=curr_start, curr_end=curr_end, start_epoch=0
        ),
        eeg_ms=EegMsSpec(weight=0.5, span_s=span_s, window_s=2.0 / fs, hop_s=1.0 / fs, start_epoch=delayed_start),
    )
    cfg = NNPredictorConfig(
        simulation=SimulationConfig(dt=_DT, downsample=1),
        model=ModelConfig(n_y=_N_Y, n_u=_N_U, hidden_size=4, depth=1),
        training=TrainingConfig(
            epochs=8,
            patience=2,
            batch_size=64,
            learning_rate=1e-2,
            weight_decay=0.0,
            train_split=0.5,
            seed=_SEED,
            losses=losses,
        ),
    )

    result = train(cfg, files)
    assert isinstance(result, TrainingResult)

    # 1. Eligibility start must be epoch 4
    assert result.eligibility_start == 4

    # 2. Selected epoch must be in the eligible phase (>= 4)
    assert result.selected_epoch >= 4

    # 3. The run must have reached at least epoch 4 (trained through the eligible phase)
    assert len(result.val_losses) > 4

    # 4. Training stats record all required fields and distinguish zero from not-run
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    result.save(artifact_dir)
    stats = json.loads((artifact_dir / "training_stats.json").read_text())

    assert stats["eligibility_start"] == 4
    assert stats["selected_epoch"] == result.selected_epoch
    assert stats["stopping_reason"] in {"early_stopping", "max_epochs"}

    # Ineligible epochs for delayed loss must be None (distinguishing not-run from 0.0)
    for epoch_idx in range(4):
        assert stats["train_components"]["eeg_ms"][epoch_idx] is None
    # From epoch 4 onwards, it ran, so it must be a float
    for epoch_idx in range(4, len(result.val_losses)):
        val = stats["train_components"]["eeg_ms"][epoch_idx]
        assert isinstance(val, float)


def test_eligibility_start_epoch_uses_effective_schedules_and_disabled_terms() -> None:
    """Eligibility calculation accounts for rounding in curriculum and ignores disabled terms."""
    # Curriculum: span_steps=4, curr_start=0, curr_end=10
    # trusted_length(epoch): round(1 + 3 * epoch / 10)
    # epoch 8: round(1 + 2.4) = 3
    # epoch 9: round(1 + 2.7) = 4 == span_steps -> reaches full span at epoch 9 due to rounding!
    curr = CurriculumMSE(weight=1.0, span_steps=4, start_epoch=0, curr_start=0, curr_end=10)
    assert eligibility_start_epoch([curr]) == 9

    # Disabled term (weight=0.0) with start_epoch=50 must NOT delay eligibility
    disabled = StftLoss(
        weight=0.0,
        span_steps=4,
        start_epoch=50,
        geometry=StftGeometry(n_segment=4, n_hop=4),
    )
    assert eligibility_start_epoch([curr, disabled]) == 9

    # Enabled delayed term (weight=1.0, start_epoch=15) pushes eligibility to 15
    enabled = StftLoss(
        weight=1.0,
        span_steps=4,
        start_epoch=15,
        geometry=StftGeometry(n_segment=4, n_hop=4),
    )
    assert eligibility_start_epoch([curr, disabled, enabled]) == 15

    # Zero-span / 1-step curriculum reaches full span immediately at start_epoch
    single_step = CurriculumMSE(weight=1.0, span_steps=1, start_epoch=0, curr_start=0, curr_end=10)
    assert eligibility_start_epoch([single_step]) == 0


def test_final_selection_restores_lowest_val_loss_among_eligible_epochs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore the eligible minimum's standardized Rollout despite a lower pre-eligibility score."""
    # Disable JIT only to supply a controlled validation curve through the real Loss scorer.
    monkeypatch.setattr(gradient.eqx, "filter_jit", lambda function: function)
    val_loss_curve = [0.05, 0.20, 0.30, 0.15, 0.10, 0.25]
    predictions = []
    original = gradient.total_loss

    def score(
        terms: Sequence[Loss],
        pred: jax.Array,
        true: jax.Array,
        ctx: LossContext,
        *,
        step_mask: jax.Array | None = None,
        active_mask: jax.Array | None = None,
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Run configured Losses, replacing only validation scores with the declared curve."""
        value, parts = original(terms, pred, true, ctx, step_mask=step_mask, active_mask=active_mask)
        if step_mask is None:
            value = jnp.asarray(val_loss_curve[len(predictions)])
            predictions.append(np.asarray(pred).copy())
        return value, parts

    monkeypatch.setattr(gradient, "total_loss", score)
    model = WaveformMLPModel(
        n_y=1,
        n_u=1,
        horizon=1,
        n_channels=1,
        n_controls=1,
        n_outputs=1,
        hidden_size=1,
        depth=0,
        residual=False,
        key=jax.random.PRNGKey(1),
    )
    dataset = TrajectoryWindowDataset([(np.ones((3, 1)), np.ones((3, 1)))], n_y=1, n_u=1, horizon=1)
    fit = fit_gradient_descent(
        model,
        dataset,
        dataset,
        TrainingConfig(epochs=6, patience=3, learning_rate=0.1, warmup_epochs=0, weight_decay=0.0),
        seed=1,
        losses=[CurriculumMSE(1.0, 1, 0, 0, 0)],
        ctx=LossContext(model.y_center, model.y_scale, 1000.0),
        eligibility_start=3,
    )
    assert fit.selected_epoch == 4
    y_hist, u_hist, u_future, _ = dataset.get_batch([0])
    restored = np.asarray(fit.best_model.rollout_standardized(y_hist, u_hist, u_future))
    np.testing.assert_allclose(restored, predictions[4])
    assert not np.allclose(restored, predictions[-1])


def test_incompatible_schedule_budget_rejected_before_training() -> None:
    """A run whose epoch budget cannot reach eligibility raises ValueError before training."""
    fs = 1.0 / _DT
    span_s = _HORIZON / fs

    # Delayed loss scheduled for epoch 20, but budget is only 10 epochs
    losses = LossSpecs(
        curriculum_mse=CurriculumMSESpec(weight=1.0, span_s=span_s, curr_start=0, curr_end=2),
        eeg_ms=EegMsSpec(weight=0.5, span_s=span_s, window_s=2.0 / fs, hop_s=1.0 / fs, start_epoch=20),
    )
    with pytest.raises(ValueError, match=r"cannot reach.*eligibility"):
        NNPredictorConfig(
            simulation=SimulationConfig(dt=_DT, downsample=1),
            model=ModelConfig(n_y=_N_Y, n_u=_N_U, hidden_size=4, depth=1),
            training=TrainingConfig(
                epochs=10,
                patience=5,
                losses=losses,
            ),
        )


def test_run_with_no_delayed_terms_retains_ordinary_behavior(tmp_path: Path) -> None:
    """A run with no delayed terms or curriculum retains ordinary best-validation checkpoint selection and patience."""
    files = _write_trajectories(tmp_path)
    fs = 1.0 / _DT
    span_s = 1.0 / fs  # 1 step, no curriculum ramp

    losses = LossSpecs(
        curriculum_mse=CurriculumMSESpec(weight=1.0, span_s=span_s, curr_start=0, curr_end=0, start_epoch=0)
    )
    cfg = NNPredictorConfig(
        simulation=SimulationConfig(dt=_DT, downsample=1),
        model=ModelConfig(n_y=_N_Y, n_u=_N_U, hidden_size=4, depth=1),
        training=TrainingConfig(
            epochs=10,
            patience=3,
            losses=losses,
        ),
    )
    result = train(cfg, files)
    assert isinstance(result, TrainingResult)
    assert result.eligibility_start == 0
    assert result.selected_epoch >= 0
    # Minimum validation loss among all epochs matches selected_epoch
    assert result.val_losses[result.selected_epoch] == min(result.val_losses)
