from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from neuro.ood import (
    CandidateState,
    OODIndex,
    binned_prediction_error,
    evaluate_prediction_errors,
    extract_mpc_rollout_queries,
    extract_run_clinical_metrics,
    extract_state_action_pairs,
    format_ood_report,
    horizon_statistics,
    print_ood_report,
    select_ood_candidate_states,
    simulate_ras_branch,
)
from neuro.predictor.data import load_trajectory

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
    z_no_hist, y_next_no_hist = extract_state_action_pairs(trajs, n_y=1, n_u=0)
    expected_samples = (30 - 1) + (40 - 1)
    assert z_no_hist.shape == (expected_samples, 11)
    assert y_next_no_hist.shape == (expected_samples, 8)

    # With control history only: n_u = 4 -> z = [y_t, u_{t-3:t+1}] -> dimension 8 + 4 * 3 = 20
    z_hist, y_next_hist = extract_state_action_pairs(trajs, n_y=1, n_u=4)
    expected_hist_samples = (30 - 4) + (40 - 4)
    assert z_hist.shape == (expected_hist_samples, 20)
    assert y_next_hist.shape == (expected_hist_samples, 8)

    # With both measurement and control history: n_y = 5, n_u = 4 -> dimension 5*8 + 4*3 = 52
    z_full, y_next_full = extract_state_action_pairs(trajs, n_y=5, n_u=4)
    expected_full_samples = (30 - 5) + (40 - 5)
    assert z_full.shape == (expected_full_samples, 52)
    assert y_next_full.shape == (expected_full_samples, 8)


def test_extract_mpc_rollout_queries(tmp_path: Path) -> None:
    """extract_mpc_rollout_queries extracts valid decisions and constructs rollout queries."""
    rng = np.random.default_rng(_SEED + 2)
    n_decisions = 10
    horizon = 5
    n_outputs = 4
    n_controls = 2

    pred_y = rng.standard_normal((n_decisions, horizon + 1, n_outputs))
    plan_u = rng.standard_normal((n_decisions, horizon, n_controls))
    applied_u = rng.standard_normal((n_decisions, n_controls))
    warmup = np.array([True, True, False, False, False, False, False, False, False, False])

    mock_log = tmp_path / "mock_log.npz"
    np.savez(
        mock_log,
        allow_pickle=True,
        **{
            "controller.predicted_y": pred_y,
            "controller.planned_u": plan_u,
            "controller.u": applied_u,
            "controller.warmup": warmup,
        },
    )

    queries, planned_u, valid_decisions = extract_mpc_rollout_queries(mock_log, n_y=1, n_u=0)
    assert len(valid_decisions) == 8
    assert queries.shape == (8, horizon, n_outputs + n_controls)
    assert planned_u.shape == (8, horizon, n_controls)

    # Test with predictor history: n_y = 3, n_u = 3
    # Priming requires max(2, 2) = 2. Decisions 0 and 1 in warmup, 2..9 valid (8 valid).
    queries_hist, _, valid_hist = extract_mpc_rollout_queries(mock_log, n_y=3, n_u=3)
    assert len(valid_hist) == 8
    assert queries_hist.shape == (8, horizon, 3 * n_outputs + 3 * n_controls)


def test_deterministic_predictor_history_parity(tmp_path: Path) -> None:
    """Deterministic fixture with distinct values at every sample and electrode verifies query alignment."""
    n_decisions = 10
    horizon = 4
    n_outputs = 3
    n_controls = 2
    n_y = 3
    n_u = 4

    pred_y = np.zeros((n_decisions, horizon + 1, n_outputs), dtype=np.float64)
    for d in range(n_decisions):
        pred_y[d, 0, :] = [1000 * d + c for c in range(n_outputs)]
        for k in range(1, horizon + 1):
            pred_y[d, k, :] = [10000 + 1000 * d + 100 * k + c for c in range(n_outputs)]

    plan_u = np.zeros((n_decisions, horizon, n_controls), dtype=np.float64)
    for d in range(n_decisions):
        for k in range(horizon):
            plan_u[d, k, :] = [50000 + 500 * d + 50 * k + m for m in range(n_controls)]

    applied_u = np.zeros((n_decisions, n_controls), dtype=np.float64)
    for d in range(n_decisions):
        applied_u[d, :] = [20000 + 200 * d + m for m in range(n_controls)]

    warmup = np.zeros(n_decisions, dtype=bool)
    warmup[0] = True  # d=0 in warmup

    mock_log = tmp_path / "deterministic_log.npz"
    np.savez(
        mock_log,
        allow_pickle=True,
        **{
            "controller.predicted_y": pred_y,
            "controller.planned_u": plan_u,
            "controller.u": applied_u,
            "controller.warmup": warmup,
        },
    )

    queries, _, valid_decisions = extract_mpc_rollout_queries(mock_log, n_y=n_y, n_u=n_u)

    # Required priming is max(3-1, 4-1) = 3. Valid decisions: [3, 4, 5, 6, 7, 8, 9] (7 decisions).
    np.testing.assert_array_equal(valid_decisions, np.array([3, 4, 5, 6, 7, 8, 9]))
    assert queries.shape == (7, horizon, n_y * n_outputs + n_u * n_controls)

    # Knot 0 (k=0) at decision d=5 (idx 2 in valid_decisions):
    expected_y_k0 = np.concatenate([pred_y[3, 0], pred_y[4, 0], pred_y[5, 0]])
    expected_u_k0 = np.concatenate([applied_u[2], applied_u[3], applied_u[4], plan_u[5, 0]])
    expected_z_k0 = np.concatenate([expected_y_k0, expected_u_k0])
    np.testing.assert_array_equal(queries[2, 0], expected_z_k0)

    # Knot 1 (k=1) at decision d=5:
    expected_y_k1 = np.concatenate([pred_y[4, 0], pred_y[5, 0], pred_y[5, 1]])
    expected_u_k1 = np.concatenate([applied_u[3], applied_u[4], plan_u[5, 0], plan_u[5, 1]])
    expected_z_k1 = np.concatenate([expected_y_k1, expected_u_k1])
    np.testing.assert_array_equal(queries[2, 1], expected_z_k1)

    # Knot 2 (k=2) at decision d=5:
    expected_y_k2 = np.concatenate([pred_y[5, 0], pred_y[5, 1], pred_y[5, 2]])
    expected_u_k2 = np.concatenate([applied_u[4], plan_u[5, 0], plan_u[5, 1], plan_u[5, 2]])
    expected_z_k2 = np.concatenate([expected_y_k2, expected_u_k2])
    np.testing.assert_array_equal(queries[2, 2], expected_z_k2)

    # Knot 3 (k=3) at decision d=5:
    expected_y_k3 = np.concatenate([pred_y[5, 1], pred_y[5, 2], pred_y[5, 3]])
    expected_u_k3 = np.concatenate([plan_u[5, 0], plan_u[5, 1], plan_u[5, 2], plan_u[5, 3]])
    expected_z_k3 = np.concatenate([expected_y_k3, expected_u_k3])
    np.testing.assert_array_equal(queries[2, 3], expected_z_k3)


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
    """evaluate_prediction_errors aligns measurement and control history windows causally."""
    rng = np.random.default_rng(_SEED + 4)
    trajs = [
        (rng.standard_normal((40, 3)), rng.standard_normal((40, 6))),
        (rng.standard_normal((50, 3)), rng.standard_normal((50, 6))),
    ]
    ref_z, _ = extract_state_action_pairs(trajs[:1], n_y=2, n_u=3)
    cal_z, _ = extract_state_action_pairs(trajs[1:], n_y=2, n_u=3)
    index = OODIndex.fit(ref_z, cal_z, k=3)

    model = _MockPredictor(p=6, n_y=2, n_u=3)
    q1, e1, q_max, e_max = evaluate_prediction_errors(
        cast("InferencePredictor", model), index, trajs[1:], horizon=10, n_y=2, n_u=3, stride=2
    )
    assert len(q1) == len(e1)
    assert len(q1) > 0
    assert len(q_max) == len(e_max)
    assert len(q_max) > 0


def test_disjoint_trajectories_and_reference_standardization() -> None:
    """OODIndex standardizes features strictly from reference statistics without calibration leakage."""
    rng = np.random.default_rng(_SEED + 5)
    # Reference set with mean ~ 100, std ~ 10
    ref_raw = rng.standard_normal((50, 4)) * 10.0 + 100.0
    # Calibration set with mean ~ 200, std ~ 20
    cal_raw = rng.standard_normal((30, 4)) * 20.0 + 200.0

    index = OODIndex.fit(ref_raw, cal_raw, k=5)

    # Standardizer mean and sigma must match ref_raw, not cal_raw or combined
    np.testing.assert_allclose(index.mu, np.mean(ref_raw, axis=0), rtol=1e-5)
    np.testing.assert_allclose(index.sigma, np.std(ref_raw, axis=0), rtol=1e-5)

    # Standardizing ref_raw must yield zero mean
    std_ref = index.standardize(ref_raw)
    np.testing.assert_allclose(np.mean(std_ref, axis=0), 0.0, atol=1e-7)

    # Standardizing cal_raw must NOT yield zero mean since cal is shifted
    std_cal = index.standardize(cal_raw)
    assert np.all(np.abs(np.mean(std_cal, axis=0)) > 5.0)


def test_select_ood_candidate_states(tmp_path: Path) -> None:
    """select_ood_candidate_states selects peak OOD decisions and enforces minimum intervals."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("dynamics:\n  seed: 7000\ncontroller:\n  dt: 0.06\n", encoding="utf-8")

    n_decisions = 20
    t_ctrl = np.arange(n_decisions, dtype=np.float64) * 0.06
    warmup = np.zeros(n_decisions, dtype=bool)
    warmup[:2] = True  # First 2 warmup

    pred_y = np.zeros((n_decisions, 5, 4), dtype=np.float64)
    plan_u = np.zeros((n_decisions, 4, 2), dtype=np.float64)

    mock_log = tmp_path / "log.npz"
    np.savez(
        mock_log,
        allow_pickle=True,
        **{
            "controller.t": t_ctrl,
            "controller.u": np.zeros((n_decisions, 2)),
            "controller.predicted_y": pred_y,
            "controller.planned_u": plan_u,
            "controller.warmup": warmup,
        },
    )

    # 18 valid decisions
    scores = np.full(18, 0.4, dtype=np.float64)
    # Burst 1: decisions 3 and 4 (valid[3]=5 at t=0.30s, valid[4]=6 at t=0.36s)
    scores[3] = 0.96
    scores[4] = 0.98  # higher peak
    # Burst 2: decision 12 (valid[12]=14 at t=0.84s)
    scores[12] = 0.97

    candidates = select_ood_candidate_states(
        tmp_path, ood_scores=scores, threshold=0.95, min_interval_s=0.3, max_candidates=5
    )

    assert len(candidates) == 2
    # Chronologically sorted
    assert candidates[0].decision_idx == 6
    assert np.isclose(candidates[0].t_star, 0.36)
    assert np.isclose(candidates[0].ood_score, 0.98)
    assert candidates[0].seed == 7000

    assert candidates[1].decision_idx == 14
    assert np.isclose(candidates[1].t_star, 0.84)
    assert np.isclose(candidates[1].ood_score, 0.97)


def test_simulate_ras_branch(tmp_path: Path) -> None:
    """simulate_ras_branch produces compatible trajectory archives with history buffer."""
    run_dir = Path("artifacts/canonical_predictors_comparison/1_observable_mlp/runs/fast_linear_kw5_s7000")
    if not run_dir.exists():
        return

    candidate = CandidateState(
        run_dir=run_dir,
        seed=7000,
        decision_idx=10,
        t_star=0.60,
        ood_score=0.98,
    )

    branch_files = simulate_ras_branch(
        candidate,
        ras_seeds=[101, 102],
        duration_s=0.2,
        history_s=0.1,
        output_dir=tmp_path / "branches",
        amp=2.0,
    )

    assert len(branch_files) == 2
    for bf in branch_files:
        assert bf.exists()
        with np.load(bf) as data:
            assert "sensor_0.t" in data
            assert "sensor_0.y_mea" in data
            assert "controller.t" in data
            assert "controller.u" in data
            assert "metadata_candidate_t_star" in data
            assert np.isclose(float(data["metadata_candidate_t_star"]), 0.60)
            assert np.isclose(float(data["metadata_relative_t_star"]), 0.10)

        # Verify load_trajectory loads cleanly with valid shapes
        u_data, y_data = load_trajectory(str(bf), None, downsample=200, dt=0.0001)
        assert len(u_data) == len(y_data)
        assert u_data.ndim == 2
        assert y_data.ndim == 2
        assert u_data.shape[1] == 3
        assert y_data.shape[1] == 62
