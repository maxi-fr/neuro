from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from experiments.comparison import Cell, validate_grid
from neuro.predictor.mlp import WaveformMLPModel
from neuro.transforms import Standardizer
from neuro.validation import ConfigConsistencyError

if TYPE_CHECKING:
    from pathlib import Path


def _create_synthetic_model(
    *,
    gain: float = 0.5,
    n_y: int = 2,
    n_u: int = 2,
    n_channels: int = 2,
    n_controls: int = 2,
    horizon: int = 75,
    dt: float = 0.02,
    downsample: int = 200,
) -> WaveformMLPModel:
    """Create a deterministic linear MLP with state matrix scaled by ``gain``."""
    module = WaveformMLPModel(
        n_y=n_y,
        n_u=n_u,
        horizon=horizon,
        n_channels=n_channels,
        n_controls=n_controls,
        n_outputs=n_channels,
        hidden_size=8,
        depth=0,
        activation="relu",
        residual=False,
        dt=dt,
        y_std=Standardizer(center=np.zeros(n_channels), scale=np.ones(n_channels)),
        u_std=Standardizer(center=np.zeros(n_controls), scale=np.ones(n_controls)),
    )
    module.downsample = downsample
    weight = np.zeros(module.weights[0].shape, dtype=np.float64)
    y_t_start = (n_y - 1) * n_channels
    for c in range(n_channels):
        weight[c, y_t_start + c] = gain
    module = module.with_readout(weight, np.zeros(n_channels))
    meta, arrays = module.to_checkpoint()
    return WaveformMLPModel.from_checkpoint(meta, arrays)


def test_comparison_grid_validates_config_without_empirical_gate(tmp_path: Path) -> None:
    """The grid checks config consistency without screening Predictor rollouts."""
    unstable_model = _create_synthetic_model(gain=1.1, horizon=75, dt=0.02)
    art_path = tmp_path / "unstable_model"
    unstable_model.save(art_path)

    cfg = {
        "dynamics": {"class_path": "neuro.plant.jansen_rit.JansenRitDynamics", "dt": 1e-4, "params": {"K": 0.6}},
        "sensors": {"class_path": "simulate.sensor.GaussianSensor", "dt": 1e-4, "std_dev": 0.0},
        "estimator": {"class_path": "neuro.filtering.AntiAliasEstimator", "dt": 1e-4, "downsample": 200},
        "controller": {
            "class_path": "neuro.control.mpc.TrajOptMPCController",
            "dt": 0.02,
            "problem": {
                "class_path": "neuro.control.problems.build_waveform_problem",
                "artifact": str(art_path),
                "horizon": 75,
                "u_max": 1.0,
                "w_y": 0.0,
            },
        },
    }
    cell = Cell(run="test_arm", arm="unstable_arm", seed=7000, config=cfg)
    validate_grid([cell])

    cfg["controller"]["dt"] = 0.04
    with pytest.raises(ConfigConsistencyError, match=r"controller\.dt"):
        validate_grid([cell])
