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
from neuro.spectral import compute_log_power_frames
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
    """Declared test criteria reject the investigated 75-step replay; saved-plan reconstruction still agrees."""
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
        thresholds=ScientificThresholds(
            calibration_identity="test-fixture calibration", max_growth_ratio=3.0, max_nmse=1.25
        ),
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
    """A stable fixture passes declared synthetic criteria and retains their provenance."""
    model = _create_synthetic_model(gain=0.5, horizon=75, dt=0.02)
    trajectories = [_synthetic_trajectory(n_steps=200, seed=101)]

    report = evaluate_control_horizon(
        model,
        trajectories,
        horizon=75,
        dt=0.02,
        conditioning="actual",
        thresholds=ScientificThresholds(
            calibration_identity="test-fixture calibration", max_growth_ratio=3.0, max_nmse=1.25
        ),
    )

    assert report.eligibility.status == EligibilityStatus.PASS
    assert report.eligibility.passed is True
    criteria = report.to_dict()["metadata"]["scientific_thresholds"]
    assert criteria["calibration_identity"] == "test-fixture calibration"
    assert criteria["max_growth_ratio"] == 3.0
    assert report.n_evaluated_windows > 0
    assert report.max_growth_ratio <= 3.0
    growth = report.terminal_metrics["amplitude_growth"]
    assert growth is not None
    assert growth <= 3.0
    term_nmse = report.terminal_metrics["nmse"]
    assert term_nmse is not None
    assert term_nmse <= 1.25


def test_unstable_fixture_fails_eligibility() -> None:
    """An unstable fixture fails the explicitly identified synthetic growth criteria."""
    model = _create_synthetic_model(gain=1.1, horizon=75, dt=0.02)
    trajectories = [_synthetic_trajectory(n_steps=200, seed=202)]

    report = evaluate_control_horizon(
        model,
        trajectories,
        horizon=75,
        dt=0.02,
        conditioning="actual",
        thresholds=ScientificThresholds(calibration_identity="test-fixture calibration", max_growth_ratio=3.0),
    )

    assert report.eligibility.status == EligibilityStatus.FAIL
    assert not report.eligibility.passed
    assert any("growth ratio" in reason for reason in report.eligibility.reasons)


def test_insufficient_horizon_coverage_yields_insufficient_evidence() -> None:
    """Even with declared calibration, incomplete future coverage never qualifies a candidate."""
    model = _create_synthetic_model(gain=0.5, horizon=75, dt=0.02)
    # Trajectory of 50 steps is shorter than horizon of 75
    short_trajectories = [_synthetic_trajectory(n_steps=50, seed=303)]

    report = evaluate_control_horizon(
        model,
        short_trajectories,
        horizon=75,
        dt=0.02,
        conditioning="actual",
        thresholds=ScientificThresholds(calibration_identity="test-fixture calibration", min_windows=1),
    )

    assert report.n_evaluated_windows == 0
    assert report.n_excluded_windows > 0
    assert report.eligibility.status == EligibilityStatus.INSUFFICIENT_EVIDENCE
    assert not report.eligibility.passed
    assert any("coverage" in reason.lower() for reason in report.eligibility.reasons)


def test_near_zero_reference_energy_yields_rejected_or_insufficient_evidence() -> None:
    """A calibrated evaluation rejects near-zero energy and marks its scalar growth metric unavailable."""
    model = _create_synthetic_model(gain=0.5, horizon=75, dt=0.02)
    flat_zero_trajectories = [_synthetic_trajectory(n_steps=200, flat_zero=True)]

    report = evaluate_control_horizon(
        model,
        flat_zero_trajectories,
        horizon=75,
        dt=0.02,
        conditioning="actual",
        thresholds=ScientificThresholds(calibration_identity="test-fixture calibration", min_reference_energy=1e-8),
    )

    assert report.eligibility.status == EligibilityStatus.INSUFFICIENT_EVIDENCE
    assert not report.eligibility.passed
    assert any("reference energy" in reason.lower() for reason in report.eligibility.reasons)
    term_growth = report.terminal_metrics["growth_ratio"]
    assert term_growth is None


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
    """The comparison workflow rejects unstable candidates against declared synthetic calibration."""
    unstable_model = _create_synthetic_model(gain=1.1, horizon=75, dt=0.02)
    trajectories = [_synthetic_trajectory(n_steps=200, seed=505)]

    eligibility = check_candidate_eligibility(
        unstable_model,
        trajectories,
        horizon=75,
        dt=0.02,
        thresholds=ScientificThresholds(calibration_identity="test-fixture calibration", max_growth_ratio=3.0),
    )

    assert eligibility.status == EligibilityStatus.FAIL
    assert not eligibility.passed


def test_comparison_validate_grid_rejects_ineligible_arm(tmp_path: Path) -> None:
    """The arm gate rejects calibrated instability and the grid rejects missing calibration."""
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

    eligibility = check_arm_eligibility(
        cell, trajectories, thresholds=ScientificThresholds(calibration_identity="test-fixture calibration")
    )
    assert not eligibility.passed
    assert eligibility.status == EligibilityStatus.FAIL

    # When no trajectories provided: yields INSUFFICIENT_EVIDENCE, never PASS
    no_calib = check_arm_eligibility(cell, None)
    assert not no_calib.passed
    assert no_calib.status == EligibilityStatus.INSUFFICIENT_EVIDENCE

    # validate_grid with trajectories raises ValueError
    with pytest.raises(ValueError, match="failed Control Horizon eligibility"):
        validate_grid([cell], trajectories)


def test_missing_calibration_cannot_qualify_a_stable_predictor() -> None:
    """Neither omitted criteria nor uncalibrated defaults qualify a stable Predictor."""
    model = _create_synthetic_model(horizon=4)
    data = [_synthetic_trajectory(n_steps=20)]
    for thresholds in (None, ScientificThresholds()):
        report = evaluate_control_horizon(model, data, 4, thresholds=thresholds)
        assert report.eligibility.status == EligibilityStatus.INSUFFICIENT_EVIDENCE
        assert any("calibration" in reason.lower() for reason in report.eligibility.reasons)


def test_missing_spectral_geometry_cannot_satisfy_a_spectral_limit() -> None:
    """A missing spectral measurement cannot satisfy an explicitly required limit."""
    model = _create_synthetic_model(horizon=4)
    report = evaluate_control_horizon(
        model,
        [_synthetic_trajectory(n_steps=20)],
        4,
        thresholds=ScientificThresholds(max_spectral_error=0.0, calibration_identity="test-fixture calibration"),
    )
    assert report.eligibility.status == EligibilityStatus.INSUFFICIENT_EVIDENCE
    assert any("spectral" in reason.lower() for reason in report.eligibility.reasons)


def test_zero_input_diagnostics_do_not_score_unmatched_recordings() -> None:
    """Zero-input growth remains visible without claiming actual-input accuracy."""
    model = _create_synthetic_model(horizon=4)
    report = evaluate_control_horizon(model, [_synthetic_trajectory(n_steps=20)], 4, conditioning="zero")
    assert np.isnan(report.rmse).all()
    assert np.isnan(report.nmse).all()
    assert report.one_step_metrics["rmse"] is None
    assert report.terminal_metrics["nmse"] is None
    assert np.isfinite(report.amplitude_growth).all()
    assert not report.eligibility.passed
    assert report.metadata.growth_reference == "measurement_history"


@pytest.mark.parametrize("horizon", [3, 7])
def test_spectral_errors_use_causal_frames_at_each_lookahead(horizon: int) -> None:
    """First and terminal spectral errors use history and their exact lookahead endpoints."""
    model = _create_synthetic_model(gain=0.5, horizon=horizon)
    u, y = _synthetic_trajectory(n_steps=30)
    geometry = StftGeometry(n_segment=4, n_hop=2, kernel_width=3)
    anchor = 12
    report = evaluate_control_horizon(model, [(u, y)], horizon, geometry=geometry, start=anchor, stride=30)
    assert report.spectral_error is not None
    assert report.spectral_error.shape == (horizon,)
    support = geometry.sample_support_steps(50.0)
    history = y[anchor - support + 2 : anchor + 1]
    predictions = y[anchor] * 0.5 ** np.arange(1, horizon + 1)[:, None]
    predicted = np.concatenate([history, predictions])
    recorded = np.concatenate([history, y[anchor + 1 : anchor + horizon + 1]])
    expected = np.array(
        [
            np.mean(
                (
                    compute_log_power_frames(predicted[i : i + support], geometry, fs=50.0)
                    - compute_log_power_frames(recorded[i : i + support], geometry, fs=50.0)
                )
                ** 2
            )
            for i in range(horizon)
        ]
    )
    np.testing.assert_allclose(report.spectral_error, expected, rtol=1e-5, atol=1e-7)
    assert report.one_step_metrics["spectral_error"] == pytest.approx(expected[0])
    assert report.terminal_metrics["spectral_error"] == pytest.approx(expected[-1])


def test_comparison_loads_reference_geometry_for_spectral_qualification(tmp_path: Path) -> None:
    """The comparison gate applies the spectral limit using the saved reference geometry."""
    model = _create_synthetic_model(horizon=4)
    artifact = tmp_path / "model"
    model.save(artifact)
    reference = tmp_path / "reference.npz"
    geometry = StftGeometry(n_segment=4, n_hop=2)
    np.savez(
        reference,
        Pref_frames=np.zeros((2, geometry.n_values(50.0))),
        fs=50.0,
        n_segment=4,
        n_hop=2,
        band_hz=np.array([-1.0, -1.0]),
        n_bin_pool=1,
        kernel="boxcar",
        kernel_width=1,
    )
    config = {
        "controller": {
            "dt": 0.02,
            "problem": {
                "artifact": str(artifact),
                "reference": str(reference),
                "horizon": 4,
            },
        }
    }
    data = [_synthetic_trajectory(n_steps=30)]
    missing = check_arm_eligibility(config, data)
    assert missing.status == EligibilityStatus.INSUFFICIENT_EVIDENCE
    assert any("calibration" in reason.lower() for reason in missing.reasons)
    rejected = check_arm_eligibility(
        config,
        data,
        thresholds=ScientificThresholds(
            max_spectral_error=0.0,
            calibration_identity="test-fixture calibration",
        ),
    )
    assert rejected.status == EligibilityStatus.FAIL
    assert any("Spectral error" in reason for reason in rejected.reasons)


def test_planned_reconstruction_withholds_actual_input_accuracy(tmp_path: Path) -> None:
    """Saved-plan reconstruction remains separate from actual-input accuracy and eligibility."""
    horizon = 4
    model = _create_synthetic_model(horizon=horizon)
    u, y = _synthetic_trajectory(n_steps=20)
    times = np.arange(len(y), dtype=np.float64) * model.dt
    forecasts = np.asarray(y[:, None, :] * 0.5 ** np.arange(horizon + 1)[None, :, None], dtype=np.float64)
    run = Run(
        tmp_path,
        {},
        {
            "controller.t": times,
            "controller.u": u,
            "controller.planned_u": np.zeros((len(y), horizon, model.m)),
            "controller.predicted_y": forecasts,
            "estimator.t": times,
            "estimator.x_hat": y,
        },
    )
    report = evaluate_control_horizon(model, run, horizon, conditioning="planned")
    assert report.reconstruction is not None
    assert report.reconstruction["is_reconstructed"]
    assert np.isnan(report.rmse).all()
    assert np.isnan(report.nmse).all()
    assert report.spectral_error is None
    assert report.terminal_metrics["rmse"] is None
    assert report.eligibility.status == EligibilityStatus.INSUFFICIENT_EVIDENCE
