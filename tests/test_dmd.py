from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import optuna
import pytest

from neuro.config import (
    CurriculumMSESpec,
    FloatParam,
    LogUniformParam,
    LossSpecs,
    ModelConfig,
    NNPredictorConfig,
    NNSweepConfig,
    SimulationConfig,
    StftGeometry,
    TrainingConfig,
)
from neuro.predictor.dmd import DmdTrainer, dmd
from neuro.predictor.inference import ObservableMLPModel, WaveformMLPModel
from neuro.predictor.module import AutoregressiveMLP
from neuro.predictor.ridge import RidgeTrainingResult
from neuro.predictor.sweep import OptunaSweep
from neuro.predictor.train import train
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from pathlib import Path


def test_dmd_linear_recovery() -> None:
    """dmd recovers the exact linear operator W and affine bias b on noiseless affine data."""
    rng = np.random.default_rng(42)
    n_samples, n_in, n_out = 100, 5, 3
    w_true = rng.standard_normal((n_out, n_in))
    b_true = rng.standard_normal((n_out,))
    x = rng.standard_normal((n_samples, n_in))
    y = x @ w_true.T + b_true

    w_est, b_est = dmd(x, y, rank=n_in)
    np.testing.assert_allclose(w_est, w_true, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(b_est, b_true, rtol=1e-10, atol=1e-10)


def test_dmd_energy_thresholding() -> None:
    """dmd energy thresholding selects the expected rank on decaying singular values."""
    rng = np.random.default_rng(42)
    n_samples = 200
    # Create rank-2 dominant features with small noise
    u_base = rng.standard_normal((n_samples, 2))
    noise = rng.standard_normal((n_samples, 10)) * 1e-4
    x = np.hstack([u_base, noise])
    y = u_base @ rng.standard_normal((2, 3))

    w_est, b_est = dmd(x, y, energy=0.999)
    assert w_est.shape == (3, 12)
    assert b_est.shape == (3,)


def test_dmd_rank_truncation_and_lambda() -> None:
    """dmd truncates to explicit rank and applies Tikhonov damping."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal((50, 10))
    y = rng.standard_normal((50, 4))

    w_est, b_est = dmd(x, y, rank=3, dmd_lambda=1e-2)
    assert w_est.shape == (4, 10)
    assert b_est.shape == (4,)


def test_dmd_trainer_rejects_depth_gt_0() -> None:
    """DmdTrainer raises TypeError if model depth > 0."""
    model = AutoregressiveMLP(
        n_y=2,
        n_u=2,
        horizon=3,
        n_channels=2,
        n_controls=1,
        n_outputs=2,
        hidden_size=8,
        depth=1,
    )
    trainer = DmdTrainer()
    trajs = [(np.zeros((10, 1)), np.zeros((10, 2)))]
    with pytest.raises(TypeError, match="DmdTrainer requires a depth-0 model"):
        trainer.fit(model, trajs)


def _write_trajectories(tmp_path: Path, n_samples: int = 150) -> list[str]:
    """Write synthetic multi-channel simulation trajectories."""
    rng = np.random.default_rng(123)
    files = []
    for i in range(2):
        u = rng.standard_normal((n_samples, 2)) * 0.1
        phase = np.arange(n_samples)[:, None] * 1e-3 * 2 * np.pi * np.array([8.0, 12.0, 16.0])
        y = np.sin(phase) + 0.2 * u[:, :1] + 0.02 * rng.standard_normal((n_samples, 3))
        path = tmp_path / f"sim_{i:03d}.npz"
        np.savez(path, **{"sensor_0.y_mea": y, "controller.u": u})  # ty: ignore[invalid-argument-type]
        files.append(str(path))
    return files


def test_dmd_observable_end_to_end(tmp_path: Path) -> None:
    """DMD Observable training fits, scores, and round-trips to JAX inference."""
    files = _write_trajectories(tmp_path, n_samples=250)
    dt = 1e-3
    downsample = 1
    fs = 1.0 / (dt * downsample)
    geom = StftGeometry(n_segment=32, n_hop=8, kernel_width=1)
    fs_frame = geom.frame_rate(fs)
    min_n_u = geom.min_past_controls()

    cfg = NNPredictorConfig(
        simulation=SimulationConfig(dt=dt, downsample=downsample),
        model=ModelConfig(n_y=2, n_u=min_n_u, depth=0),
        training=TrainingConfig(
            fit="dmd",
            dmd_rank=10,
            dmd_lambda=1e-3,
            eval_horizon_s=4 / fs_frame,
            losses=LossSpecs(
                curriculum_mse=CurriculumMSESpec(weight=1.0, span_s=4 / fs_frame, curr_start=0, curr_end=1)
            ),
        ),
        observable=geom,
    )

    result = train(cfg, files)
    assert isinstance(result, RidgeTrainingResult)
    assert "val_loss" in result.candidates
    assert "val_log_mse" in result.candidates
    assert result.predictor.depth == 0

    # Save and round-trip to JAX ObservableMLPModel
    ckpt_path = tmp_path / "model"
    result.predictor.save(ckpt_path)
    inference = ObservableMLPModel.load(ckpt_path)
    assert isinstance(inference, ObservableMLPModel)
    assert inference.n_y == 2
    assert inference.n_u == min_n_u

    # Test free-run on JAX side
    u_h = np.zeros((1, min_n_u, 2))
    y_h = np.zeros((1, 2, inference.n_outputs))
    u_f = np.zeros((1, 4, 2))
    pred = inference.free_run(y_h, u_h, u_f)
    assert pred.shape == (1, 4, inference.n_outputs)


def test_dmd_waveform_end_to_end(tmp_path: Path) -> None:
    """DMD waveform training fits, scores, and round-trips to JAX inference."""
    files = _write_trajectories(tmp_path, n_samples=100)
    dt = 1e-3
    fs = 1.0 / dt
    span_s = 5 / fs

    cfg = NNPredictorConfig(
        simulation=SimulationConfig(dt=dt, downsample=1),
        model=ModelConfig(n_y=2, n_u=2, depth=0),
        training=TrainingConfig(
            fit="dmd",
            dmd_energy=0.98,
            eval_horizon_s=span_s,
            losses=LossSpecs(curriculum_mse=CurriculumMSESpec(weight=1.0, span_s=span_s, curr_start=0, curr_end=1)),
        ),
    )

    result = train(cfg, files)
    assert isinstance(result, RidgeTrainingResult)
    assert "rollout_nmse" in result.candidates
    assert "log_energy" in result.candidates

    # Save and round-trip to JAX WaveformMLPModel
    ckpt_path = tmp_path / "wave_model"
    result.predictor.save(ckpt_path)
    inference = WaveformMLPModel.load(ckpt_path)
    assert isinstance(inference, WaveformMLPModel)


def test_dmd_observable_optuna_sweep_end_to_end(tmp_path: Path) -> None:
    """An Optuna trial suggesting dmd_energy/dmd_lambda reaches a real DmdTrainer fit and scores it.

    Unlike ``test_sweep.py``, which stubs ``train`` to test the objective-selection wiring, this
    exercises the real closed-form fit through ``OptunaSweep`` -- the one path no other test
    covers for the ``fit: dmd`` arm.
    """
    files = _write_trajectories(tmp_path, n_samples=250)
    dt = 1e-3
    downsample = 1
    fs = 1.0 / (dt * downsample)
    geom = StftGeometry(n_segment=32, n_hop=8, kernel_width=1)
    fs_frame = geom.frame_rate(fs)
    min_n_u = geom.min_past_controls()

    cfg = NNPredictorConfig(
        simulation=SimulationConfig(dt=dt, downsample=downsample),
        model=ModelConfig(n_y=2, n_u=min_n_u, depth=0),
        training=TrainingConfig(
            fit="dmd",
            eval_horizon_s=4 / fs_frame,
            losses=LossSpecs(
                curriculum_mse=CurriculumMSESpec(weight=1.0, span_s=4 / fs_frame, curr_start=0, curr_end=1)
            ),
        ),
        observable=geom,
        sweep=NNSweepConfig(
            objective="val_log_mse",
            training={
                "dmd_energy": FloatParam(type="float", low=0.9, high=0.999),
                "dmd_lambda": LogUniformParam(type="loguniform", low=1e-6, high=1e-1),
            },
        ),
    )
    trial = optuna.trial.FixedTrial({"dmd_energy": 0.95, "dmd_lambda": 1e-3})

    value = OptunaSweep(cfg, files, tmp_path).objective(trial)

    assert isinstance(value, float)
    assert trial.user_attrs.keys() == {"val_loss", "val_log_mse"}
