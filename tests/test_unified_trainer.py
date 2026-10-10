"""Seam 2 -- the unified Trainer entry point: fit dispatch, candidates, and the save round-trip."""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neuro.config import (
    CurriculumMSESpec,
    LossSpecs,
    ModelConfig,
    NNPredictorConfig,
    NNSweepConfig,
    SimulationConfig,
    StftGeometry,
    TrainingConfig,
)
from neuro.predictor.gradient import fit_gradient_descent
from neuro.predictor.mlp import ObservableMLPModel, WaveformMLPModel
from neuro.predictor.ridge import RidgeTrainingResult
from neuro.predictor.train import TrainingResult, train

if TYPE_CHECKING:
    from pathlib import Path

    from neuro.predictor.base import AutoregressiveModel
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


def _checkpoint_arrays(model: AutoregressiveModel) -> list[np.ndarray]:
    """Every trainable parameter, in forward order, as plain NumPy arrays."""
    weights = getattr(model, "weights", ())
    biases = getattr(model, "biases", ())
    return [np.asarray(w, dtype=np.float64) for w in weights] + [np.asarray(b, dtype=np.float64) for b in biases]


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
    assert result.log_energy is not None  # the waveform arm always scores it
    assert result.candidates["log_energy"] == result.log_energy.pooled
    assert result.candidates["val_loss"] == min(result.val_losses)
    assert result.candidates["rollout_nmse"] == result.free_run.pooled
    assert all(np.isfinite(value) for value in result.candidates.values())


def test_candidates_contain_the_config_named_objective(tmp_path: Path) -> None:
    """A ``sweep.objective`` named in config is always among the recorded candidates."""
    wave = _wave_train(
        _wave_config().model_copy(update={"sweep": NNSweepConfig(objective="rollout_nmse")}),
        _write_trajectories(tmp_path, dt=_WAVE_DT, t=_T),
    )
    assert wave.candidates["rollout_nmse"] == wave.free_run.pooled


def test_waveform_save_round_trips_weights_standardizers_and_metadata(tmp_path: Path) -> None:
    """``save`` writes the checkpoint; ``load`` restores weights, standardizers and recorded metadata."""
    result = _wave_train(_wave_config(), _write_trajectories(tmp_path, dt=_WAVE_DT, t=_T))
    artifact_dir = tmp_path / "wave"
    result.save(artifact_dir)

    loaded = WaveformMLPModel.load(artifact_dir / "model")

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


class _TinyNet(eqx.Module):
    """A plain two-layer MLP, deliberately not one of the repo's modules."""

    w1: jax.Array
    b1: jax.Array
    w2: jax.Array
    b2: jax.Array

    def __init__(self, n_in: int, n_out: int, key: jax.Array) -> None:
        """Build a two-layer MLP."""
        k1, k2 = jax.random.split(key)
        bound1 = 1.0 / np.sqrt(n_in)
        self.w1 = jax.random.uniform(k1, (n_in, 8), minval=-bound1, maxval=bound1)
        self.b1 = jnp.zeros(8)
        bound2 = 1.0 / np.sqrt(8)
        self.w2 = jax.random.uniform(k2, (8, n_out), minval=-bound2, maxval=bound2)
        self.b2 = jnp.zeros(n_out)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Map ``(B, n_in)`` to ``(B, n_out)``."""
        h = jax.nn.relu(x @ self.w1 + self.b1)
        return h @ self.w2 + self.b2


def test_gradient_descent_serves_any_module() -> None:
    """The shared fit regresses a foreign module: the loss descends and stays finite."""
    rng = np.random.default_rng(_SEED + 1)
    n_in, n_out, n_samples = 4, 2, 256
    x = rng.standard_normal((n_samples, n_in)).astype(np.float32)
    w = rng.standard_normal((n_in, n_out)).astype(np.float32)
    y = x @ w + 0.05 * rng.standard_normal((n_samples, n_out)).astype(np.float32)
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
        model: _TinyNet,
        y_hist: jax.Array,
        u_hist: jax.Array,
        u_future: jax.Array,
        y_target: jax.Array,
        epoch: int | None,
    ) -> tuple[jax.Array, dict[str, float]]:
        del u_hist, u_future, epoch
        return jnp.mean((model(y_hist) - y_target) ** 2), {}

    dummy_u = np.zeros((n_samples, 1), dtype=np.float32)
    train_batches = [
        (
            x[i : i + cfg.batch_size],
            dummy_u[i : i + cfg.batch_size],
            dummy_u[i : i + cfg.batch_size],
            y[i : i + cfg.batch_size],
        )
        for i in range(0, 200, cfg.batch_size)
    ]
    val_batches = [
        (
            x[i : i + cfg.batch_size],
            dummy_u[i : i + cfg.batch_size],
            dummy_u[i : i + cfg.batch_size],
            y[i : i + cfg.batch_size],
        )
        for i in range(200, n_samples, cfg.batch_size)
    ]

    key = jax.random.PRNGKey(_SEED)
    model = _TinyNet(n_in, n_out, key)
    fit = fit_gradient_descent(model, train_batches, val_batches, cfg, seed=_SEED, loss_fn=mse)
    train_losses, val_losses = fit.train_losses, fit.val_losses

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
    assert isinstance(result.predictor, WaveformMLPModel)
    assert result.predictor.depth == 0
    assert set(result.candidates) == {"rollout_nmse", "log_energy"}
    assert result.candidates["rollout_nmse"] == result.free_run.pooled
    assert result.log_energy is not None  # the waveform arm always scores it
    assert result.candidates["log_energy"] == result.log_energy.pooled
    assert all(np.isfinite(value) for value in result.candidates.values())


def _obs_config(**training: object) -> NNPredictorConfig:
    """A tiny observable config for testing the observable training arms."""
    dt = 0.004  # 250 Hz
    fs = 1.0 / dt
    geometry = StftGeometry(n_segment=64, n_hop=16, band_hz=[4.0, 30.0], n_bin_pool=2, kernel_width=5)
    fs_frame = fs / geometry.n_hop
    span_s = 4 / fs_frame
    losses = LossSpecs(curriculum_mse=CurriculumMSESpec(weight=1.0, span_s=span_s, curr_start=0, curr_end=2))
    defaults = {
        "epochs": 3,
        "batch_size": 32,
        "learning_rate": 1e-2,
        "weight_decay": 0.0,
        "train_split": 0.5,
        "seed": _SEED,
        "patience": 50,
        "eval_horizon_s": span_s,
        "losses": losses,
    }
    return NNPredictorConfig(
        simulation=SimulationConfig(dt=dt, downsample=1),
        model=ModelConfig(n_y=2, n_u=8, hidden_size=4, depth=1),
        training=TrainingConfig.model_validate({**defaults, **training}),
        observable=geometry,
    )


def test_observable_candidates_match_the_config_kind(tmp_path: Path) -> None:
    """The observable gradient-descent run records exactly {val_loss, val_log_mse}."""
    files = _write_trajectories(tmp_path, dt=0.004, t=600)
    cfg = _obs_config()
    result = train(cfg, files)

    assert isinstance(result, TrainingResult)
    assert set(result.candidates) == {"val_loss", "val_log_mse"}
    assert result.candidates["val_loss"] == result.val_losses[result.selected_epoch]
    assert result.candidates["val_log_mse"] == result.free_run.pooled
    assert np.isfinite(result.du_sensitivity)
    assert result.du_sensitivity > 0.0
    assert all(np.isfinite(value) for value in result.candidates.values())

    # Verify save round-trip
    art_dir = tmp_path / "obs_art"
    result.save(art_dir)
    assert (art_dir / "model.npz").exists()
    assert (art_dir / "training_stats.json").exists()


def test_observable_ridge_fit_through_train_on_depth0_mlp(tmp_path: Path) -> None:
    """training.fit: ridge on an observable depth-0 MLP fits in closed form and records candidates."""
    files = _write_trajectories(tmp_path, dt=0.004, t=600)
    cfg = _obs_config(fit="ridge").model_copy(update={"model": ModelConfig(n_y=2, n_u=8, hidden_size=4, depth=0)})
    result = train(cfg, files)

    assert isinstance(result, RidgeTrainingResult)
    assert isinstance(result.predictor, ObservableMLPModel)
    assert result.predictor.depth == 0
    assert set(result.candidates) == {"val_loss", "val_log_mse"}
    assert result.candidates["val_log_mse"] == result.free_run.pooled
    assert all(np.isfinite(value) for value in result.candidates.values())

    art_dir = tmp_path / "obs_ridge_art"
    result.save(art_dir)
    assert (art_dir / "model.npz").exists()
    assert (art_dir / "training_stats.json").exists()


def test_observable_ridge_fit_on_depth2_mlp_fails_at_build_time(tmp_path: Path) -> None:
    """training.fit: ridge on an observable depth-2 MLP fails at build time."""
    files = _write_trajectories(tmp_path, dt=0.004, t=600)
    cfg = _obs_config(fit="ridge").model_copy(update={"model": ModelConfig(n_y=2, n_u=8, hidden_size=4, depth=2)})
    with pytest.raises(ValueError, match="depth-0 MLP"):
        train(cfg, files)
