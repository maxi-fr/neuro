from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from neuro.ood import (
    OODIndex,
    binned_prediction_error,
    evaluate_prediction_errors,
    extract_mpc_rollout_queries,
    extract_run_clinical_metrics,
    extract_state_action_pairs,
    format_ood_report,
    horizon_statistics,
    print_ood_report,
)

if TYPE_CHECKING:
    from neuro.predictor.inference import InferencePredictor

_SEED = 42


def test_ood_index_fit_and_query() -> None:
    """OODIndex computes K-NN distances, global distances, and calibrated percentiles."""
    rng = np.random.default_rng(_SEED)
    ref = rng.standard_normal((100, 10))
    cal = rng.standard_normal((50, 10))

    index = OODIndex.fit(ref, cal, k=5)
    assert index.reference_count == 100
    assert index.calibration_count == 50
    assert index.dimension == 10
    assert index.coefficient_of_variation() > 0.0

    # Query in-distribution vs clear outlier
    id_query = rng.standard_normal((5, 10))
    outlier = np.full((1, 10), 20.0)

    d_id = index.compute_distances(id_query)
    d_out = index.compute_distances(outlier)
    assert d_out[0] > np.max(d_id)

    q_id = index.percentiles(d_id)
    q_out = index.percentiles(d_out)
    assert np.all((q_id >= 0.0) & (q_id <= 1.0))
    assert q_out[0] == 1.0

    # Global standardized distance
    g_id = index.global_distances(id_query)
    g_out = index.global_distances(outlier)
    assert g_out[0] > np.max(g_id)


def test_extract_state_action_pairs_shapes() -> None:
    """extract_state_action_pairs produces matching sample counts and correct feature widths."""
    rng = np.random.default_rng(_SEED + 1)
    trajs = [
        (rng.standard_normal((30, 3)), rng.standard_normal((30, 8))),
        (rng.standard_normal((40, 3)), rng.standard_normal((40, 8))),
    ]

    # Without history: z = [y_t, u_t] -> dimension 8 + 3 = 11
    z_no_hist, y_next_no_hist = extract_state_action_pairs(trajs, n_u=0)
    # Each trajectory of length T yields T - 1 one-step transitions
    expected_samples = (30 - 1) + (40 - 1)
    assert z_no_hist.shape == (expected_samples, 11)
    assert y_next_no_hist.shape == (expected_samples, 8)

    # With history: n_u = 4 -> z = [y_t, u_{t-3:t+1}] -> dimension 8 + 4 * 3 = 20
    z_hist, y_next_hist = extract_state_action_pairs(trajs, n_u=4)
    expected_hist_samples = (30 - 4) + (40 - 4)
    assert z_hist.shape == (expected_hist_samples, 20)
    assert y_next_hist.shape == (expected_hist_samples, 8)


def test_horizon_statistics() -> None:
    """horizon_statistics groups rollout metrics by horizon index and computes summaries."""
    n_decisions = 20
    horizon = 15
    percentiles = np.linspace(0.1, 0.9, n_decisions * horizon).reshape(n_decisions, horizon)
    distances = percentiles * 2.0

    stats = horizon_statistics(percentiles, distances)
    assert stats["mean_percentile"].shape == (horizon,)
    assert stats["median_percentile"].shape == (horizon,)
    assert stats["p95_rate"].shape == (horizon,)
    assert stats["p99_rate"].shape == (horizon,)
    assert stats["mean_distance"].shape == (horizon,)
    assert stats["q_max"].shape == (n_decisions,)
    assert stats["q_mean"].shape == (n_decisions,)


def test_binned_prediction_error() -> None:
    """binned_prediction_error partitions errors into monotonically ordered percentile bins."""
    percentiles = np.array([0.1, 0.2, 0.6, 0.8, 0.92, 0.96, 0.995])
    errors = np.array([1.0, 1.2, 2.0, 3.0, 4.0, 5.0, 6.0])

    binned = binned_prediction_error(percentiles, errors)
    assert len(binned) == 6
    # First bin [0, 0.50] should have 2 points
    assert binned[0]["count"] == 2
    np.testing.assert_allclose(binned[0]["mean_error"], 1.1)


def test_extract_mpc_rollout_queries(tmp_path: Path) -> None:
    """extract_mpc_rollout_queries extracts valid decisions and constructs rollout queries."""
    rng = np.random.default_rng(_SEED + 2)
    n_decisions = 10
    horizon = 5
    n_outputs = 4
    n_controls = 2

    pred_y = rng.standard_normal((n_decisions, horizon + 1, n_outputs))
    plan_u = rng.standard_normal((n_decisions, horizon, n_controls))
    warmup = np.array([True, True, False, False, False, False, False, False, False, False])

    mock_log = tmp_path / "mock_log.npz"
    np.savez(
        mock_log,
        allow_pickle=True,
        **{
            "controller.predicted_y": pred_y,
            "controller.planned_u": plan_u,
            "controller.warmup": warmup,
        },
    )

    queries, planned_u, valid_decisions = extract_mpc_rollout_queries(mock_log, n_u=0)
    assert len(valid_decisions) == 8
    assert queries.shape == (8, horizon, n_outputs + n_controls)
    assert planned_u.shape == (8, horizon, n_controls)

    # Test with control history
    queries_hist, _, _ = extract_mpc_rollout_queries(mock_log, n_u=3)
    assert queries_hist.shape == (8, horizon, n_outputs + 3 * n_controls)


def test_format_ood_report() -> None:
    """format_ood_report produces structured text containing all evaluation criteria."""
    rng = np.random.default_rng(_SEED + 3)
    ref = rng.standard_normal((30, 5))
    cal = rng.standard_normal((20, 5))
    index = OODIndex.fit(ref, cal, k=3)

    mock_defs = {
        "cal_distances": np.array([1.0, 1.2, 1.5]),
        "mpc_distances_flat": np.array([0.9, 1.1, 1.3]),
        "cv_score": 0.08,
        "h_stats": {
            "mean_percentile": np.array([0.5, 0.4, 0.3]),
            "median_percentile": np.array([0.5, 0.4, 0.3]),
        },
        "drift_delta": -0.2,
        "p95_rate": 1.5,
        "p99_rate": 0.2,
        "mean_q_max": 0.6,
        "mean_q_mean": 0.3,
        "bins_summary": [
            {
                "bin_label": "0--50%",
                "count": 10,
                "mean_error": 1.0,
                "std_error": 0.1,
                "median_error": 0.95,
            },
            {
                "bin_label": "99--100%",
                "count": 2,
                "mean_error": 1.5,
                "std_error": 0.2,
                "median_error": 1.45,
            },
        ],
        "corr": 0.25,
        "horizon": 5,
        "dim": 5,
        "active_k": 3,
        "ood_index": index,
        "mpc_percentiles": np.array([[0.6, 0.2, 0.1], [0.7, 0.3, 0.1]]),
        "e_realized": np.array([1.2]),
        "clinical_metrics": {
            "seizure_burden": 0.059,
            "n_seizing_final": 7.0,
            "delivered_charge": 6.865,
            "mean_amplitude": 0.095,
            "kirchhoff_max": 1.1e-16,
            "ez_recruited": 3.0,
            "pz_recruited": 1.0,
            "healthy_recruited": 3.0,
            "solve_time_p95_s": 6.19,
            "realtime_factor": 103.2,
            "solve_success_rate": 1.0,
        },
    }

    report = format_ood_report(mock_defs)
    assert "MPC OOD SUPPORT EVALUATION REPORT" in report
    assert "CLOSED-LOOP CLINICAL EFFICACY & CONTROL PERFORMANCE" in report
    assert "Seizure Burden:          5.90%" in report
    assert "Total Delivered Charge:  6.865 mA*s" in report
    assert "REALIZED TRAJECTORY (k=0) VS. PLANNED ROLLOUTS (k >= 1)" in report
    assert "CRITERION 1: OOD Distance Support Comparison" in report
    assert "CRITERION 6: Distance Concentration Diagnostic" in report
    assert "CRITERION 2: Lookahead Horizon Compounding Drift" in report
    assert "CRITERION 3: Support Violation Frequency" in report
    assert "CRITERION 4: One-Step Dynamics Prediction Error" in report
    assert "CRITERION 5: Multi-Step Rollout Error" in report
    assert "EXECUTIVE SUMMARY" in report
    assert "Clinical Efficacy: Seizure Burden = 5.90%" in report
    assert "[PASS]" in report

    # Verify print_ood_report executes without error
    print_ood_report(mock_defs)


def test_extract_run_clinical_metrics() -> None:
    """extract_run_clinical_metrics reads clinical and solver metrics from run artifacts."""
    run_dir = Path("artifacts/canonical_predictors_comparison/1_observable_mlp/runs/fast_linear_kw5_s7000")
    if not run_dir.exists():
        return

    metrics = extract_run_clinical_metrics(run_dir)
    assert "seizure_burden" in metrics
    assert "delivered_charge" in metrics
    assert "mean_amplitude" in metrics
    assert "kirchhoff_max" in metrics
    assert metrics["seizure_burden"] > 0.0
    assert metrics["delivered_charge"] > 0.0
    assert metrics["kirchhoff_max"] < 1e-10


class _MockPredictor:
    """Mock predictor for testing multi-step rollout evaluation."""

    def __init__(self, p: int, n_y: int = 2, n_u: int = 3) -> None:
        self.p = p
        self.n_y = n_y
        self.n_u = n_u

    def free_run(self, _y_hist: np.ndarray, _u_hist: np.ndarray, u_fut: np.ndarray) -> np.ndarray:
        n_batches, horizon, _ = u_fut.shape
        return np.zeros((n_batches, horizon, self.p), dtype=np.float64)


def test_evaluate_prediction_errors_with_history() -> None:
    """evaluate_prediction_errors aligns control history windows without negative index slicing."""
    rng = np.random.default_rng(_SEED + 4)
    trajs = [
        (rng.standard_normal((40, 3)), rng.standard_normal((40, 6))),
        (rng.standard_normal((50, 3)), rng.standard_normal((50, 6))),
    ]
    ref_z, _ = extract_state_action_pairs(trajs[:1], n_u=5)
    cal_z, _ = extract_state_action_pairs(trajs[1:], n_u=5)
    index = OODIndex.fit(ref_z, cal_z, k=3)

    model = _MockPredictor(p=6, n_y=2, n_u=3)
    q1, e1, q_max, e_max = evaluate_prediction_errors(
        cast("InferencePredictor", model), index, trajs[1:], horizon=10, n_u=5, stride=2
    )
    assert len(q1) == len(e1)
    assert len(q1) > 0
    assert len(q_max) == len(e_max)
    assert len(q_max) > 0
