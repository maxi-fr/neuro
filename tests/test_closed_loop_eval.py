from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml

from neuro.closed_loop_eval import evaluate_closed_loop_suppression
from neuro.config import ClosedLoopEvalConfig

if TYPE_CHECKING:
    from pathlib import Path


def _eval_cfg(simulation_config: Path) -> ClosedLoopEvalConfig:
    return ClosedLoopEvalConfig(
        simulation_config=str(simulation_config),
        seeds=[69],
        t_end=5.0,
        seizure_ptp_mv=5.0,
        max_seizing_regions=5,
    )


def test_missing_simulation_config_raises(tmp_path: Path) -> None:
    """Test evaluate_closed_loop_suppression raises when the simulation config does not exist."""
    trial_dir = tmp_path / "trial_0"
    trial_dir.mkdir()
    (trial_dir / "model.npz").touch()

    with pytest.raises(FileNotFoundError, match="Closed-loop simulation config not found"):
        evaluate_closed_loop_suppression(trial_dir, _eval_cfg(tmp_path / "absent.yaml"))


def test_missing_model_checkpoint_raises(tmp_path: Path) -> None:
    """Test evaluate_closed_loop_suppression raises when the trial has no trained predictor checkpoint."""
    sim_config_path = tmp_path / "sim.yaml"
    with sim_config_path.open("w") as f:
        yaml.dump({"t_end": 5.0}, f)

    trial_dir = tmp_path / "trial_0"
    trial_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Trained predictor checkpoint not found"):
        evaluate_closed_loop_suppression(trial_dir, _eval_cfg(sim_config_path))
