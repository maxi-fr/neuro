"""Cover the torch training loop end to end: convergence, the best-model snapshot and seeding."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from neuro.artifacts import load_any_artifact
from neuro.config import ModelConfig, NNPredictorConfig, SimulationConfig, TrainingConfig
from neuro.predictor.artifact import MLPArtifact
from neuro.predictor.data import apply_to_blocks, prepare_datasets, transform_features
from neuro.predictor.losses import predictor_loss
from neuro.predictor.module import AutoregressiveMLP
from neuro.predictor.train import train
from neuro.transforms import Pipeline

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


def _config(depth: int = 1, **training: object) -> NNPredictorConfig:
    """A tiny but complete predictor config; ``training`` overrides the optimisation defaults."""
    defaults = {
        "epochs": 3,
        "batch_size": 64,
        "learning_rate": 1e-2,
        "weight_decay": 0.0,
        "train_split": 0.5,
        "curriculum_end_epoch": 2,
        "seed": _SEED,
        "patience": 50,
    }
    return NNPredictorConfig(
        simulation=SimulationConfig(dt=_DT, downsample=1),
        model=ModelConfig(n_y=_N_Y, n_u=_N_U, horizon=_HORIZON, hidden_size=4, depth=depth),
        training=TrainingConfig.model_validate({**defaults, **training}),
    )


def _weights(art: MLPArtifact) -> list[np.ndarray]:
    """Flat list of every weight and bias array, in forward order."""
    return [a for layer in art.layers for a in layer]


def _validation_loss(cfg: NNPredictorConfig, files: list[str], art: MLPArtifact) -> float:
    """Re-score ``art`` on the validation windows exactly as the training loop does."""
    mdl = cfg.model
    data = prepare_datasets(files, None, 1, mdl.n_y, mdl.n_u, mdl.horizon, _DT, cfg.training.train_split)
    X = transform_features(data.X_val, art.y_pipeline, art.u_pipeline, mdl.n_y, data.n_channels, data.n_controls)
    Y = apply_to_blocks(data.Y_val, Pipeline(art.y_pipeline.steps[:1]), data.n_channels)

    model = AutoregressiveMLP.from_artifact(art)
    pred = model(torch.as_tensor(X)).reshape(-1, mdl.horizon, art.n_channels)
    target = torch.as_tensor(Y).reshape(-1, mdl.horizon, data.n_channels)
    loss, _ = predictor_loss(model, pred, target, torch.ones(mdl.horizon, dtype=torch.float64), cfg.training.w_psd)
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
    assert set(stats) == {"train_loss", "val_loss", "nmse_rollout", "nmse_rollout_per_step", "du_sensitivity"}
    assert stats["nmse_rollout"] == result.rollout.pooled

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
    cfg = _config(epochs=20, learning_rate=0.3, curriculum_end_epoch=1, patience=2)
    result = train(cfg, files)

    assert len(result.val_losses) < 20, "the run must stop on patience, not on the epoch budget"
    assert int(np.argmin(result.val_losses)) == len(result.val_losses) - 3
    assert min(result.val_losses) < result.val_losses[-1]

    assert _validation_loss(cfg, files, result.artifact) == pytest.approx(min(result.val_losses))


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


def test_linear_model_is_warm_started_and_skips_the_one_step_epochs(files: list[str]) -> None:
    """A depth-0 model starts from the exact 1-step solution, so the L = 1 epochs are skipped."""
    settings: dict[str, object] = {"epochs": 3, "curriculum_start_epoch": 2, "curriculum_end_epoch": 2}
    linear = train(_config(depth=0, **settings), files)
    nonlinear = train(_config(depth=1, **settings), files)

    assert linear.artifact.is_linear
    assert len(linear.train_losses) == 1
    assert len(nonlinear.train_losses) == 3
    assert np.isfinite(linear.rollout.pooled)
