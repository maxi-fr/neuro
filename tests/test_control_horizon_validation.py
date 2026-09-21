from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from neuro.comparison import Cell, check_arm_eligibility, validate_grid
from neuro.config import StftGeometry
from neuro.predictor.evaluation import (
    EligibilityStatus,
    NumericalTolerances,
    ScientificThresholds,
    check_candidate_eligibility,
    evaluate_control_horizon,
)
from neuro.predictor.inference import WaveformMLPModel
from neuro.predictor.module import AutoregressiveMLP
from neuro.run_view import Run
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from neuro.types import FloatArray

_INVESTIGATED_RUN = Path("artifacts/canonical_predictors_comparison/2_waveform_mlp/runs/waveform_curriculum_stft_s7000")


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
    module = AutoregressiveMLP(
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
    layer = module.layers[0]
    assert isinstance(layer, torch.nn.Linear)
    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.zero_()
        y_t_start = (n_y - 1) * n_channels
        for c in range(n_channels):
            layer.weight[c, y_t_start + c] = gain
    meta, arrays = module.to_checkpoint()
    return WaveformMLPModel.from_checkpoint(meta, arrays)


def _synthetic_trajectory(
    *,
    n_steps: int = 200,
    n_channels: int = 2,
    n_controls: int = 2,
    seed: int = 42,
    flat_zero: bool = False,
) -> tuple[FloatArray, FloatArray]:
    """Generate synthetic controls and measurements."""
    if flat_zero:
        return np.zeros((n_steps, n_controls), dtype=np.float64), np.zeros((n_steps, n_channels), dtype=np.float64)
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n_steps, n_controls)) * 0.1
    # Bounded AR(1) target
    y = np.zeros((n_steps, n_channels), dtype=np.float64)
    y[0] = rng.standard_normal(n_channels)
    for t in range(1, n_steps):
        y[t] = 0.5 * y[t - 1] + 0.1 * rng.standard_normal(n_channels)
    return u, y


def test_investigated_checkpoint_rejected_on_saved_75_step_actual_input_replay() -> None:
    """The investigated checkpoint fails eligibility on its saved 75-step actual-input replay."""
    if not _INVESTIGATED_RUN.exists():
        pytest.skip(f"Artifact run not found: {_INVESTIGATED_RUN}")

    run = Run.load(_INVESTIGATED_RUN)
    model = WaveformMLPModel.load(Path(run.config["controller"]["problem"]["artifact"]))

    report = evaluate_control_horizon(
        model,
        run,
        horizon=75,
        dt=0.02,
        stride=5,
        conditioning="actual",
        thresholds=ScientificThresholds(max_growth_ratio=3.0, max_nmse=1.25),
        tolerances=NumericalTolerances(reconstruction_atol=1e-5),
    )

    # 1. Complete Control Horizon evaluated through terminal predicted output
    assert report.metadata.horizon == 75
    assert len(report.lookahead_steps) == 75
    assert report.lookahead_steps[-1] == 75
    assert report.n_evaluated_windows > 0
    assert report.n_excluded_windows > 0
    assert report.total_windows == report.n_evaluated_windows + report.n_excluded_windows

    # 2. Per-lookahead metrics populated
    assert report.rmse.shape == (75,)
    assert report.nmse.shape == (75,)
    assert report.persistence_rmse.shape == (75,)
    assert report.amplitude_growth.shape == (75,)

    # 3. One-step results separately visible
    assert "rmse" in report.one_step_metrics
    assert "amplitude_growth" in report.one_step_metrics
    assert report.one_step_metrics["rmse"] == pytest.approx(report.rmse[0])
    assert report.terminal_metrics["amplitude_growth"] == pytest.approx(report.amplitude_growth[-1])

    # 4. Investigated checkpoint is rejected without hardcoded checkpoint name
    assert report.eligibility.status == EligibilityStatus.FAIL
    assert not report.eligibility.passed
    assert report.max_growth_ratio > 10.0
    assert any("growth ratio" in reason for reason in report.eligibility.reasons)

    # 5. Saved-plan reconstruction passes under planned conditioning
    plan_report = evaluate_control_horizon(
        model,
        run,
        horizon=75,
        dt=0.02,
        stride=5,
        conditioning="planned",
    )
    assert plan_report.conditioning == "planned"
    assert plan_report.reconstruction is not None
    assert plan_report.reconstruction["is_reconstructed"]
    assert plan_report.reconstruction["max_error"] < 1e-5


def test_stable_fixture_passes_eligibility() -> None:
    """A stable fixture passes declared scientific acceptance thresholds."""
    model = _create_synthetic_model(gain=0.5, horizon=75, dt=0.02)
    trajectories = [_synthetic_trajectory(n_steps=200, seed=101)]

    report = evaluate_control_horizon(
        model,
        trajectories,
        horizon=75,
        dt=0.02,
        conditioning="actual",
        thresholds=ScientificThresholds(max_growth_ratio=3.0, max_nmse=1.25),
    )

    assert report.eligibility.status == EligibilityStatus.PASS
    assert report.eligibility.passed is True
    assert report.n_evaluated_windows > 0
    assert report.max_growth_ratio <= 3.0
    growth = report.terminal_metrics["amplitude_growth"]
    assert growth is not None
    assert growth <= 3.0
    term_nmse = report.terminal_metrics["nmse"]
    assert term_nmse is not None
    assert term_nmse <= 1.25


def test_unstable_fixture_fails_eligibility() -> None:
    """An unstable fixture fails declared eligibility criteria due to signal growth."""
    model = _create_synthetic_model(gain=1.1, horizon=75, dt=0.02)
    trajectories = [_synthetic_trajectory(n_steps=200, seed=202)]

    report = evaluate_control_horizon(
        model,
        trajectories,
        horizon=75,
        dt=0.02,
        conditioning="actual",
        thresholds=ScientificThresholds(max_growth_ratio=3.0),
    )

    assert report.eligibility.status == EligibilityStatus.FAIL
    assert not report.eligibility.passed
    assert any("growth ratio" in reason for reason in report.eligibility.reasons)


def test_insufficient_horizon_coverage_yields_insufficient_evidence() -> None:
    """Windows without a full recorded future are excluded, and insufficient coverage never passes."""
    model = _create_synthetic_model(gain=0.5, horizon=75, dt=0.02)
    # Trajectory of 50 steps is shorter than horizon of 75
    short_trajectories = [_synthetic_trajectory(n_steps=50, seed=303)]

    report = evaluate_control_horizon(
        model,
        short_trajectories,
        horizon=75,
        dt=0.02,
        conditioning="actual",
        thresholds=ScientificThresholds(min_windows=1),
    )

    assert report.n_evaluated_windows == 0
    assert report.n_excluded_windows > 0
    assert report.eligibility.status == EligibilityStatus.INSUFFICIENT_EVIDENCE
    assert not report.eligibility.passed
    assert any("coverage" in reason.lower() for reason in report.eligibility.reasons)


def test_near_zero_reference_energy_yields_rejected_or_insufficient_evidence() -> None:
    """Near-zero reference energy produces explicit undefined/rejected result rather than misleading ratio."""
    model = _create_synthetic_model(gain=0.5, horizon=75, dt=0.02)
    flat_zero_trajectories = [_synthetic_trajectory(n_steps=200, flat_zero=True)]

    report = evaluate_control_horizon(
        model,
        flat_zero_trajectories,
        horizon=75,
        dt=0.02,
        conditioning="actual",
        thresholds=ScientificThresholds(min_reference_energy=1e-8),
    )

    assert report.eligibility.status == EligibilityStatus.INSUFFICIENT_EVIDENCE
    assert not report.eligibility.passed
    assert any("reference energy" in reason.lower() for reason in report.eligibility.reasons)
    term_growth = report.terminal_metrics["growth_ratio"]
    assert term_growth is not None
    assert np.isnan(term_growth)


def test_spectral_error_under_observable_geometry() -> None:
    """Spectral error is computed under intended Observable geometry."""
    model = _create_synthetic_model(gain=0.5, horizon=75, dt=0.02)
    trajectories = [_synthetic_trajectory(n_steps=200, seed=404)]
    geometry = StftGeometry(n_segment=50, n_hop=25, band_hz=[4.0, 20.0], n_bin_pool=2, kernel_width=1)

    report = evaluate_control_horizon(
        model,
        trajectories,
        horizon=75,
        dt=0.02,
        conditioning="actual",
        geometry=geometry,
    )

    assert report.spectral_error is not None
    assert len(report.spectral_error) > 0
    assert report.one_step_metrics["spectral_error"] is not None


def test_comparison_workflow_candidate_eligibility_check() -> None:
    """The comparison workflow checks candidate eligibility and rejects failing models."""
    unstable_model = _create_synthetic_model(gain=1.1, horizon=75, dt=0.02)
    trajectories = [_synthetic_trajectory(n_steps=200, seed=505)]

    eligibility = check_candidate_eligibility(
        unstable_model,
        trajectories,
        horizon=75,
        dt=0.02,
        thresholds=ScientificThresholds(max_growth_ratio=3.0),
    )

    assert eligibility.status == EligibilityStatus.FAIL
    assert not eligibility.passed


def test_comparison_validate_grid_rejects_ineligible_arm(tmp_path: Path) -> None:
    """validate_grid rejects an arm whose predictor fails Control Horizon eligibility."""
    unstable_model = _create_synthetic_model(gain=1.1, horizon=75, dt=0.02)
    art_path = tmp_path / "unstable_model"
    unstable_model.save(art_path)

    cfg = {
        "dynamics": {"class_path": "neuro.jansen_rit.JansenRitDynamics", "dt": 1e-4, "params": {"K": 0.6}},
        "sensors": {"class_path": "simulate.sensor.GaussianSensor", "dt": 1e-4, "std_dev": 0.0},
        "estimator": {"class_path": "neuro.filtering.AntiAliasEstimator", "dt": 1e-4, "downsample": 200},
        "controller": {
            "class_path": "neuro.control.mpc.TrajOptMPCController",
            "dt": 0.02,
            "problem": {
                "class_path": "neuro.control.mpc.build_waveform_problem",
                "artifact": str(art_path),
                "horizon": 75,
                "w_y": 0.0,
            },
        },
    }
    cell = Cell(run="test_arm", arm="unstable_arm", seed=7000, u_max=1.0, config=cfg)
    trajectories = [_synthetic_trajectory(n_steps=200, seed=606)]

    eligibility = check_arm_eligibility(cell, trajectories)
    assert not eligibility.passed
    assert eligibility.status == EligibilityStatus.FAIL

    # When no trajectories provided: yields INSUFFICIENT_EVIDENCE, never PASS
    no_calib = check_arm_eligibility(cell, None)
    assert not no_calib.passed
    assert no_calib.status == EligibilityStatus.INSUFFICIENT_EVIDENCE

    # validate_grid with trajectories raises ValueError
    with pytest.raises(ValueError, match="failed Control Horizon eligibility"):
        validate_grid([cell], trajectories)
