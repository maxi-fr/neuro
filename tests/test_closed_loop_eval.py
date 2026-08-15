from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import yaml

from neuro.closed_loop_eval import (
    _spread_profile_from_logs,
    evaluate_closed_loop_suppression,
)
from neuro.config import ClosedLoopEvalConfig
from neuro.seizure import spread_profile_from_lfp

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
    (trial_dir / "model.eqx").touch()

    with pytest.raises(FileNotFoundError, match="Closed-loop simulation config not found"):
        evaluate_closed_loop_suppression(trial_dir, _eval_cfg(tmp_path / "absent.yaml"))


def test_missing_model_artifact_raises(tmp_path: Path) -> None:
    """Test evaluate_closed_loop_suppression raises when the trial has no trained predictor."""
    sim_config_path = tmp_path / "sim.yaml"
    with sim_config_path.open("w") as f:
        yaml.dump({"t_end": 5.0}, f)

    trial_dir = tmp_path / "trial_0"
    trial_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Trained predictor artifact not found"):
        evaluate_closed_loop_suppression(trial_dir, _eval_cfg(sim_config_path))


def _logs(n_samples: int, ptp_mv: np.ndarray) -> list[dict[str, Any]]:
    """Fake core logs whose per-region LFP (``x[1] - x[2]``) has the given peak-to-peak amplitudes."""
    wave = np.sign(np.sin(np.linspace(0.0, 8.0 * np.pi, n_samples)))
    logs = []
    for i in range(n_samples):
        x = np.zeros((6, len(ptp_mv)))
        x[1] = 0.5 * ptp_mv * wave[i]
        logs.append({"x": x})
    return logs


def test_spread_profile_counts_regions_above_threshold() -> None:
    """Test the windowed envelope thresholds each region against seizure_ptp_mv."""
    ptp_mv = np.array([20.0, 8.0, 1.0, 0.0])
    profile = _spread_profile_from_logs(_logs(200, ptp_mv), dt=0.01, threshold=5.0)

    assert profile.ptp.shape[0] == len(ptp_mv)
    assert profile.ptp[:, -1] == pytest.approx(ptp_mv)
    assert int(profile.n_seizing()[-1]) == 2


def test_spread_profile_from_lfp() -> None:
    """Test building a spread profile directly from a logged 2D region LFP trajectory."""
    ptp_mv = np.array([20.0, 8.0, 1.0, 0.0])
    wave = np.sign(np.sin(np.linspace(0.0, 8.0 * np.pi, 200)))
    y_traj = 0.5 * ptp_mv[None, :] * wave[:, None]

    profile = spread_profile_from_lfp(y_traj.T, dt=0.01, threshold=5.0)
    assert profile.ptp.shape[0] == len(ptp_mv)
    assert profile.ptp[:, -1] == pytest.approx(ptp_mv)
    assert int(profile.n_seizing()[-1]) == 2


def test_spread_profile_rejects_run_shorter_than_window() -> None:
    """Test a run too short to fill one spread window raises rather than scoring nothing."""
    with pytest.raises(ValueError, match="shorter than the"):
        _spread_profile_from_logs(_logs(50, np.array([20.0])), dt=0.01, threshold=5.0)
