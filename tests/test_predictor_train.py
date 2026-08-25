"""Cover the torch training loop end to end: convergence, the best-model snapshot and seeding."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from neuro.artifacts import load_any_artifact
from neuro.config import (
    CurriculumMSESpec,
    LossSpecs,
    ModelConfig,
    NNPredictorConfig,
    SimulationConfig,
    TrainingConfig,
)
from neuro.predictor.artifact import MLPArtifact
from neuro.predictor.data import fit_standardizers, prepare_datasets
from neuro.predictor.losses import LossContext, build_losses, total_loss
from neuro.predictor.module import AutoregressiveMLP
from neuro.predictor.ridge import RidgeTrainer
from neuro.predictor.train import lr_schedule, train

if TYPE_CHECKING:
    from pathlib import Path

_SEED = 11
_DT = 1e-3
_T, _N_EEG, _N_CONTROLS = 200, 3, 2
_N_Y, _N_U, _HORIZON = 2, 2, 3


def _write_trajectories(tmp_path: Path) -> list[str]:
    """Two synthetic trajectories -- sinusoids plus a control-driven drift -- in the loader's npz layout."""
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


def _config(depth: int = 1, curr_start: int = 0, curr_end: int = 2, **training: object) -> NNPredictorConfig:
    """A tiny but complete predictor config; ``training`` overrides the optimisation defaults."""
    fs = 1.0 / (_DT * 1)
    span_s = _HORIZON / fs
    losses = LossSpecs(
        curriculum_mse=CurriculumMSESpec(weight=1.0, span_s=span_s, curr_start=curr_start, curr_end=curr_end)
    )
    defaults = {
        "epochs": 3,
        "batch_size": 64,
        "learning_rate": 1e-2,
        "weight_decay": 0.0,
        "train_split": 0.5,
        "seed": _SEED,
        "patience": 50,
        "eval_horizon_s": span_s,
        "losses": losses,
    }
    return NNPredictorConfig(
        simulation=SimulationConfig(dt=_DT, downsample=1),
        model=ModelConfig(n_y=_N_Y, n_u=_N_U, hidden_size=4, depth=depth),
        training=TrainingConfig.model_validate({**defaults, **training}),
    )


def _weights(art: MLPArtifact) -> list[np.ndarray]:
    """Flat list of every weight and bias array, in forward order."""
    return [a for layer in art.layers for a in layer]


def _validation_loss(cfg: NNPredictorConfig, files: list[str], art: MLPArtifact) -> float:
    """Re-score ``art`` on the validation windows exactly as the training loop does."""
    mdl = cfg.model
    fs = cfg.fs
    assert cfg.training.losses is not None
    losses = build_losses(cfg.training.losses, fs)
    horizon = max(loss_obj.span_steps for loss_obj in losses)

    data = prepare_datasets(
        files,
        None,
        1,
        mdl.n_y,
        mdl.n_u,
        horizon,
        _DT,
        cfg.training.train_split,
        scaler=cfg.training.scaler,
        global_scaling=cfg.training.global_scaling,
    )

    model = AutoregressiveMLP.from_artifact(art)
    pred = model(torch.as_tensor(data.X_val, dtype=torch.float32)).reshape(-1, horizon, art.n_channels)
    target = torch.as_tensor(data.Y_val, dtype=torch.float32).reshape(-1, horizon, data.n_channels)
    ctx = LossContext(
        y_center=torch.as_tensor(art.y_std.center, dtype=torch.float32),
        y_scale=torch.as_tensor(art.y_std.scale, dtype=torch.float32),
        fs=fs,
        epoch=None,
    )
    loss, _ = total_loss(losses, pred, target, ctx)
    return float(loss.detach())


@pytest.fixture
def files(tmp_path: Path) -> list[str]:
    return _write_trajectories(tmp_path)


def test_training_converges_and_scores_the_rollout(files: list[str]) -> None:
    """The smoke test: the loss goes down and the free-run rollout produces finite numbers."""
    result = train(_config(), files)

    assert len(result.train_losses) == len(result.val_losses) == 3
    assert result.train_losses[-1] < result.train_losses[0]
    assert np.isfinite(result.rollout.pooled)
    assert np.all(np.isfinite(result.rollout.per_step))
    assert result.rollout.per_step.shape == (_HORIZON,)
    assert np.isfinite(result.log_energy.pooled)
    assert np.all(np.isfinite(result.log_energy.per_position))
    assert np.isfinite(result.du_sensitivity)
    assert result.du_sensitivity > 0.0
    assert len(result.val_trajs) == 1


def test_save_round_trip_predicts_identically(files: list[str], tmp_path: Path) -> None:
    """``save`` writes both files and the reloaded artifact rolls out bit-identically."""
    result = train(_config(), files)
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()

    result.save(artifact_dir)
    stats = json.loads((artifact_dir / "training_stats.json").read_text())
    assert set(stats) == {
        "train_loss",
        "val_loss",
        "train_components",
        "val_components",
        "nmse_rollout",
        "nmse_rollout_per_step",
        "log_energy",
        "log_energy_per_position",
        "du_sensitivity",
    }
    assert stats["nmse_rollout"] == result.rollout.pooled
    assert stats["log_energy"] == result.log_energy.pooled

    loaded = load_any_artifact(artifact_dir / "model")
    assert isinstance(loaded, MLPArtifact)

    u, y = result.val_trajs[0]
    priming = result.artifact.priming_steps
    state = result.artifact.prime(y[:priming], u[:priming])
    u_future = u[priming : priming + _HORIZON]
    np.testing.assert_array_equal(loaded.rollout(state, u_future), result.artifact.rollout(state, u_future))


def test_returned_artifact_is_the_best_epoch_not_the_last(files: list[str]) -> None:
    """Early stopping must return the best epoch's weights, not an alias of the live module.

    A torch module is mutable, so snapshotting it by reference would silently hand back the last
    epoch. Stopping on ``patience`` guarantees the run ends on non-improving epochs, so the best
    epoch cannot be the last; re-scoring the returned artifact then pins which epoch it came from.
    """
    cfg = _config(epochs=20, learning_rate=0.3, curr_end=1, patience=2)
    result = train(cfg, files)

    assert len(result.val_losses) < 20, "the run must stop on patience, not on the epoch budget"
    assert int(np.argmin(result.val_losses)) == len(result.val_losses) - 3
    assert min(result.val_losses) < result.val_losses[-1]

    assert _validation_loss(cfg, files, result.artifact) == pytest.approx(min(result.val_losses))


def test_depth0_ridge_fit_reproduces_the_warm_start(files: list[str]) -> None:
    """The Ridge Trainer on a depth-0 MLP reproduces the incumbent warm-start least-squares.

    The incumbent warm start is the exact 1-step ``lstsq`` over the standardized windows; the
    ridge fit folds the same features and targets into normal equations from the raw trajectories
    and must land on the same single layer at lambda = 0.
    """
    cfg = _config(depth=0)
    mdl, trn = cfg.model, cfg.training
    horizon = _HORIZON
    data = prepare_datasets(
        files,
        None,
        1,
        mdl.n_y,
        mdl.n_u,
        horizon,
        _DT,
        trn.train_split,
        scaler=trn.scaler,
        global_scaling=trn.global_scaling,
    )
    split = fit_standardizers(
        files,
        n_steps_cfg=None,
        downsample=1,
        dt=_DT,
        train_split=trn.train_split,
        scaler=trn.scaler,
        global_scaling=trn.global_scaling,
    )

    def build() -> AutoregressiveMLP:
        return AutoregressiveMLP(
            n_y=mdl.n_y,
            n_u=mdl.n_u,
            horizon=horizon,
            n_channels=data.n_channels,
            n_controls=data.n_controls,
            hidden_size=mdl.hidden_size,
            depth=0,
            activation=mdl.activation,
            dt=_DT,
            y_std=data.y_std,
            u_std=data.u_std,
        )

    model = build()
    RidgeTrainer(ridge_lambda=0.0).fit(model, split.train_trajs)

    # The incumbent warm-start solution, re-derived on the same standardized windows.
    y_len = mdl.n_y * data.n_channels
    m = data.n_controls
    X_1step = np.hstack([data.X_train[:, :y_len], data.X_train[:, y_len + m : y_len + (mdl.n_u + 1) * m]])
    weight_bias, *_ = np.linalg.lstsq(
        np.hstack([X_1step, np.ones((X_1step.shape[0], 1))]), data.Y_train[:, : data.n_channels], rcond=None
    )
    layer = model.layers[0]
    assert isinstance(layer, torch.nn.Linear)
    # The module's standardizers live in float32 buffers, so the ridge features differ from the
    # pipeline's float64 windows at ~1e-7; the single layer must match at float32 precision.
    np.testing.assert_allclose(layer.weight.detach().numpy(), weight_bias[:-1].T, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(layer.bias.detach().numpy(), weight_bias[-1], rtol=1e-5, atol=1e-6)


def test_same_seed_reproduces_and_offset_decorrelates(files: list[str]) -> None:
    """Both the initialisation and the epoch shuffle follow ``training.seed + seed_offset``."""
    first = train(_config(), files)
    again = train(_config(), files)
    shifted = train(_config(), files, seed_offset=1)

    assert first.train_losses == again.train_losses
    for got, want in zip(_weights(again.artifact), _weights(first.artifact), strict=True):
        np.testing.assert_array_equal(got, want)

    assert shifted.train_losses != first.train_losses
    assert any(
        not np.array_equal(got, want)
        for got, want in zip(_weights(shifted.artifact), _weights(first.artifact), strict=True)
    )


@pytest.mark.parametrize("warmup_steps", [0, 4])
def test_lr_schedule_ramps_in_then_anneals_to_zero(warmup_steps: int) -> None:
    """Warm-up climbs to the peak at ``warmup_steps``; the cosine still reaches 0 on the last step."""
    total_steps = 20
    optimizer = torch.optim.AdamW(torch.nn.Linear(2, 2).parameters(), lr=1.0)
    scheduler = lr_schedule(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)

    trace = []
    for _ in range(total_steps):
        trace.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    assert np.argmax(trace) == warmup_steps
    assert trace[warmup_steps] == pytest.approx(1.0)
    assert trace[: warmup_steps + 1] == sorted(trace[: warmup_steps + 1])
    assert trace[warmup_steps:] == sorted(trace[warmup_steps:], reverse=True)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-12)


def test_warmup_shortens_the_first_epochs_without_changing_the_epoch_count(files: list[str]) -> None:
    """``warmup_epochs`` reshapes the schedule only -- the loop still runs every configured epoch."""
    plain = train(_config(epochs=4), files)
    warmed = train(_config(epochs=4, warmup_epochs=2), files)

    assert len(warmed.train_losses) == len(plain.train_losses) == 4
    assert warmed.train_losses != plain.train_losses


def test_linear_model_is_warm_started_and_skips_the_one_step_epochs(files: list[str]) -> None:
    """A depth-0 model starts from the exact 1-step solution, so the L = 1 epochs are skipped."""
    linear = train(_config(depth=0, epochs=3, curr_start=2, curr_end=2), files)
    nonlinear = train(_config(depth=1, epochs=3, curr_start=2, curr_end=2), files)

    assert linear.artifact.is_linear
    assert len(linear.train_losses) == 1
    assert len(nonlinear.train_losses) == 3
    assert np.isfinite(linear.rollout.pooled)
