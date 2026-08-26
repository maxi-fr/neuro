"""Seam 2 -- the unified Trainer entry point: fit dispatch, candidates, and the save round-trip."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from torch import nn

from neuro.config import (
    CurriculumMSESpec,
    LossSpecs,
    ModelConfig,
    NNPredictorConfig,
    NNSweepConfig,
    SimulationConfig,
    TrainingConfig,
)
from neuro.predictor.gradient import fit_gradient_descent
from neuro.predictor.module import AutoregressiveMLP, TrainingPredictor
from neuro.predictor.ridge import RidgeTrainingResult
from neuro.predictor.train import TrainingResult, train

if TYPE_CHECKING:
    from pathlib import Path

    from neuro.types import FloatArray

_SEED = 21
_WAVE_DT, _T = 1e-3, 200
_N_EEG, _N_CONTROLS = 3, 2
_WAVE_HORIZON = 3


def _write_trajectories(tmp_path: Path, *, dt: float, t: int) -> list[str]:
    """Two synthetic trajectories -- sinusoids plus a control-driven drift -- in the loader's npz layout."""
    rng = np.random.default_rng(_SEED)
    files = []
    for i in range(2):
        u = np.cumsum(rng.standard_normal((t, _N_CONTROLS)) * 0.1, axis=0)
        phase = np.arange(t)[:, None] * dt * 2 * np.pi * np.array([7.0, 11.0, 13.0])
        y = np.sin(phase) + 0.3 * u[:, :1] + 0.05 * rng.standard_normal((t, _N_EEG))
        path = tmp_path / f"sim_{i:03d}.npz"
        np.savez(path, **{"sensor_0.y_mea": y, "controller.u": u})  # ty: ignore[invalid-argument-type]
        files.append(str(path))
    return files


def _wave_config(**training: object) -> NNPredictorConfig:
    """A tiny but complete waveform config; ``training`` overrides the optimisation defaults."""
    fs = 1.0 / _WAVE_DT
    span_s = _WAVE_HORIZON / fs
    losses = LossSpecs(curriculum_mse=CurriculumMSESpec(weight=1.0, span_s=span_s, curr_start=0, curr_end=2))
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
        simulation=SimulationConfig(dt=_WAVE_DT, downsample=1),
        model=ModelConfig(n_y=2, n_u=2, hidden_size=4, depth=1),
        training=TrainingConfig.model_validate({**defaults, **training}),
    )


def _checkpoint_arrays(model: nn.Module) -> list[np.ndarray]:
    """Every trainable parameter, in forward order, as plain NumPy arrays."""
    return [p.detach().cpu().numpy().astype(np.float64) for p in model.parameters()]


def _wave_train(cfg: NNPredictorConfig, files: list[str]) -> TrainingResult:
    """Train the waveform arm, narrowing the union the dispatcher returns."""
    result = train(cfg, files)
    assert isinstance(result, TrainingResult)
    return result


def _wave_model(depth: int) -> ModelConfig:
    """The waveform model block the shared helpers use, at an explicit depth."""
    return ModelConfig(n_y=2, n_u=2, hidden_size=4, depth=depth)


def test_waveform_candidates_match_the_config_kind(tmp_path: Path) -> None:
    """The waveform run records exactly ``{log_energy, val_loss, rollout_nmse}``, consistently."""
    result = _wave_train(_wave_config(), _write_trajectories(tmp_path, dt=_WAVE_DT, t=_T))

    assert set(result.candidates) == {"log_energy", "val_loss", "rollout_nmse"}
    assert result.candidates["log_energy"] == result.log_energy.pooled
    assert result.candidates["val_loss"] == min(result.val_losses)
    assert result.candidates["rollout_nmse"] == result.rollout.pooled
    assert all(np.isfinite(value) for value in result.candidates.values())


def test_candidates_contain_the_config_named_objective(tmp_path: Path) -> None:
    """A ``sweep.objective`` named in config is always among the recorded candidates."""
    wave = _wave_train(
        _wave_config().model_copy(update={"sweep": NNSweepConfig(objective="rollout_nmse")}),
        _write_trajectories(tmp_path, dt=_WAVE_DT, t=_T),
    )
    assert wave.candidates["rollout_nmse"] == wave.rollout.pooled


def test_waveform_save_round_trips_weights_standardizers_and_metadata(tmp_path: Path) -> None:
    """``save`` writes the checkpoint; ``load`` restores weights, standardizers and recorded metadata."""
    result = _wave_train(_wave_config(), _write_trajectories(tmp_path, dt=_WAVE_DT, t=_T))
    artifact_dir = tmp_path / "wave"
    result.save(artifact_dir)

    loaded = AutoregressiveMLP.load(artifact_dir / "model")

    for got, want in zip(_checkpoint_arrays(loaded), _checkpoint_arrays(result.predictor), strict=True):
        np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(loaded.y_std.center, result.predictor.y_std.center)
    np.testing.assert_array_equal(loaded.y_std.scale, result.predictor.y_std.scale)
    np.testing.assert_array_equal(loaded.u_std.center, result.predictor.u_std.center)
    np.testing.assert_array_equal(loaded.u_std.scale, result.predictor.u_std.scale)
    assert loaded.activation == result.predictor.activation
    assert loaded.horizon == result.predictor.horizon
    assert loaded.dt == result.predictor.dt
    assert loaded.downsample == result.predictor.downsample
    assert loaded.provenance == result.predictor.provenance


class _TinyNet(nn.Module):
    """A plain two-layer MLP, deliberately not one of the repo's modules."""

    def __init__(self, n_in: int, n_out: int) -> None:
        """Build a two-layer MLP."""
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 8), nn.ReLU(), nn.Linear(8, n_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, n_in)`` to ``(B, n_out)``."""
        return self.net(x)


def test_gradient_descent_serves_any_torch_module() -> None:
    """The shared fit regresses a foreign ``nn.Module``: the loss descends and stays finite."""
    rng = np.random.default_rng(_SEED + 1)
    n_in, n_out, n_samples = 4, 2, 256
    x = torch.as_tensor(rng.standard_normal((n_samples, n_in)), dtype=torch.float32)
    w = rng.standard_normal((n_in, n_out))
    y = torch.as_tensor(x.numpy() @ w + 0.05 * rng.standard_normal((n_samples, n_out)), dtype=torch.float32)
    cfg = TrainingConfig.model_validate(
        {
            "epochs": 20,
            "batch_size": 32,
            "learning_rate": 1e-2,
            "weight_decay": 0.0,
            "patience": 50,
            "eval_horizon_s": 0.1,
            "seed": _SEED,
        }
    )

    def mse(
        model: nn.Module, xb: torch.Tensor, yb: torch.Tensor, epoch: int | None
    ) -> tuple[torch.Tensor, dict[str, float]]:
        return torch.mean((model(xb) - yb) ** 2), {}

    model = _TinyNet(n_in, n_out)
    train_losses, val_losses, _, _ = fit_gradient_descent(
        model, x[:200], y[:200], x[200:], y[200:], cfg, seed=_SEED, loss_fn=mse
    )

    assert len(train_losses) == len(val_losses) == 20
    assert train_losses[-1] < train_losses[0]
    assert val_losses[-1] < val_losses[0]
    assert all(np.isfinite(value) for value in train_losses + val_losses)


def test_ridge_fit_on_depth2_mlp_fails_at_build_time(tmp_path: Path) -> None:
    """``training.fit: ridge`` on a depth-2 MLP fails at build time, before any fit runs."""
    cfg = _wave_config(fit="ridge").model_copy(update={"model": _wave_model(depth=2)})
    with pytest.raises(ValueError, match="depth-0 MLP"):
        train(cfg, _write_trajectories(tmp_path, dt=_WAVE_DT, t=_T))


def test_ridge_fit_through_train_on_depth0_mlp(tmp_path: Path) -> None:
    """``training.fit: ridge`` on a depth-0 MLP routes to the Ridge Trainer and fits end-to-end."""
    cfg = _wave_config(fit="ridge").model_copy(update={"model": _wave_model(depth=0)})
    result = train(cfg, _write_trajectories(tmp_path, dt=_WAVE_DT, t=_T))

    assert isinstance(result, RidgeTrainingResult)
    assert isinstance(result.predictor, AutoregressiveMLP)
    assert result.predictor.depth == 0
    assert set(result.candidates) == {"rollout_nmse", "log_energy"}
    assert result.candidates["rollout_nmse"] == result.rollout.pooled
    assert result.candidates["log_energy"] == result.log_energy.pooled
    assert all(np.isfinite(value) for value in result.candidates.values())
