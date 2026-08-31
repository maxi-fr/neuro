from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from neuro.config import (
    CurriculumMSESpec,
    LossSpecs,
    ModelConfig,
    NNPredictorConfig,
    SimulationConfig,
    StftGeometry,
    TrainingConfig,
)
from neuro.predictor.plotting import (
    plot_rollout_comparison,
    plot_training_curves,
)
from neuro.predictor.train import train

if TYPE_CHECKING:
    from pathlib import Path

_DT = 1e-3
_HORIZON = 3


def _write_mock_trajectories(tmp_path: Path, n_samples: int = 100) -> list[str]:
    """Write synthetic multi-channel simulation trajectories for testing."""
    rng = np.random.default_rng(42)
    files: list[str] = []
    for i in range(2):
        u = rng.standard_normal((n_samples, 2)) * 0.1
        y = rng.standard_normal((n_samples, 3)) * 0.5
        path = tmp_path / f"sim_{i:03d}.npz"
        np.savez(path, **{"sensor_0.y_mea": y, "controller.u": u})  # ty: ignore[invalid-argument-type]
        files.append(str(path))
    return files


def test_plot_training_curves_generates_image(tmp_path: Path) -> None:
    """plot_training_curves writes a valid loss_curve.png for a gradient-descent training result."""
    data_files = _write_mock_trajectories(tmp_path, n_samples=100)
    fs = 1.0 / _DT
    span_s = _HORIZON / fs
    losses = LossSpecs(curriculum_mse=CurriculumMSESpec(weight=1.0, span_s=span_s, curr_start=0, curr_end=2))
    cfg = NNPredictorConfig(
        simulation=SimulationConfig(dt=_DT, downsample=1),
        model=ModelConfig(n_y=2, n_u=2, hidden_size=4, depth=1),
        training=TrainingConfig.model_validate(
            {
                "epochs": 2,
                "batch_size": 32,
                "learning_rate": 1e-2,
                "train_split": 0.5,
                "seed": 42,
                "patience": 10,
                "eval_horizon_s": span_s,
                "losses": losses,
            }
        ),
    )

    result = train(cfg, data_files)
    plot_path = tmp_path / "loss_curve.png"
    plot_training_curves(result, plot_path)

    assert plot_path.exists()
    assert plot_path.stat().st_size > 0


def test_plot_rollout_comparison_waveform_generates_image(tmp_path: Path) -> None:
    """plot_rollout_comparison writes comparison.png for a waveform predictor result."""
    data_files = _write_mock_trajectories(tmp_path, n_samples=100)
    fs = 1.0 / _DT
    span_s = _HORIZON / fs
    losses = LossSpecs(curriculum_mse=CurriculumMSESpec(weight=1.0, span_s=span_s, curr_start=0, curr_end=2))
    cfg = NNPredictorConfig(
        simulation=SimulationConfig(dt=_DT, downsample=1),
        model=ModelConfig(n_y=2, n_u=2, hidden_size=4, depth=1),
        training=TrainingConfig.model_validate(
            {
                "epochs": 2,
                "batch_size": 32,
                "learning_rate": 1e-2,
                "train_split": 0.5,
                "seed": 42,
                "patience": 10,
                "eval_horizon_s": span_s,
                "losses": losses,
            }
        ),
    )

    result = train(cfg, data_files)
    plot_path = tmp_path / "comparison.png"
    plot_rollout_comparison(result, plot_path)

    assert plot_path.exists()
    assert plot_path.stat().st_size > 0


def test_plot_rollout_comparison_observable_generates_image(tmp_path: Path) -> None:
    """plot_rollout_comparison writes comparison.png for an observable predictor result."""
    data_files = _write_mock_trajectories(tmp_path, n_samples=250)
    geometry = StftGeometry(n_segment=32, n_hop=8, band_hz=[4.0, 30.0], n_bin_pool=2, kernel_width=3)
    fs_frame = 250.0 / geometry.n_hop
    span_s = 3 / fs_frame
    losses = LossSpecs(curriculum_mse=CurriculumMSESpec(weight=1.0, span_s=span_s, curr_start=0, curr_end=2))
    cfg = NNPredictorConfig(
        simulation=SimulationConfig(dt=0.004, downsample=1),
        model=ModelConfig(n_y=2, n_u=8, hidden_size=4, depth=1),
        training=TrainingConfig.model_validate(
            {
                "epochs": 2,
                "batch_size": 16,
                "learning_rate": 1e-2,
                "train_split": 0.5,
                "seed": 42,
                "patience": 10,
                "eval_horizon_s": span_s,
                "losses": losses,
            }
        ),
        observable=geometry,
    )

    result = train(cfg, data_files)
    plot_path = tmp_path / "comparison_obs.png"
    plot_rollout_comparison(result, plot_path)

    assert plot_path.exists()
    assert plot_path.stat().st_size > 0


def test_plotting_handles_empty_or_ridge_gracefully(tmp_path: Path) -> None:
    """Plotting functions do not raise errors when results lack training curves or validation trajs."""

    class EmptyResult:
        def __init__(self) -> None:
            self.candidates: dict[str, float] = {}

    empty = EmptyResult()
    # Should safely return without error
    plot_training_curves(empty, tmp_path / "dummy_loss.png")  # ty: ignore[invalid-argument-type]
    plot_rollout_comparison(empty, tmp_path / "dummy_comp.png")  # ty: ignore[invalid-argument-type]

    assert not (tmp_path / "dummy_loss.png").exists()
    assert not (tmp_path / "dummy_comp.png").exists()
