from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from neuro.connectome import Connectome, load_connectome
from neuro.estimator import UKFEstimator
from neuro.jansen_rit import JansenRitDynamics, JansenRitParams


def _toy_connectome(n_nodes: int = 2) -> Connectome:
    """Build a small deterministic connectome for testing."""
    rng = np.random.default_rng(0)
    weights = rng.random((n_nodes, n_nodes))
    np.fill_diagonal(weights, 0.0)
    tract_lengths = rng.random((n_nodes, n_nodes)) * 10.0
    speed = 50.0
    labels = np.array([f"r{i}" for i in range(n_nodes)], dtype=np.str_)
    channels = np.array([f"c{i}" for i in range(n_nodes)], dtype=np.str_)
    return Connectome(
        weights=weights,
        tract_lengths=tract_lengths,
        centres=rng.random((n_nodes, 3)),
        region_labels=labels,
        hemispheres=np.zeros(n_nodes, dtype=np.bool_),
        speed=speed,
        delays=tract_lengths / speed,
        gain=np.eye(n_nodes),
        channel_labels=channels,
        region_index={label: idx for idx, label in enumerate(labels)},
        channel_index={label: idx for idx, label in enumerate(channels)},
    )


def test_estimator_shapes() -> None:
    """Verify that UKFEstimator initializes with correct filter dimensions and noise shapes."""
    conn = _toy_connectome(2)
    est = UKFEstimator(
        dt=1e-4,
        n_nodes=2,
        gain=conn.gain,
        gamma=conn.gamma,
        delays=conn.delays,
        initial_k=0.5,
        initial_w=conn.weights,
    )

    # State dimension: 6*N dynamic states + 1 (K) + N^2 (weights) = 12 + 1 + 4 = 17
    assert est.dim_x == 17
    assert est.ukf.x.shape == (17,)
    assert est.ukf.P.shape == (17, 17)
    assert est.ukf.Q.shape == (17, 17)
    assert est.ukf.R.shape == (2, 2)  # 2 channels selected by default

    # Verify process noise has non-zeros only at x5' positions (indices 4 and 10) and parameter positions
    q_diag = np.diag(est.ukf.Q)
    assert q_diag[4] == 1e-3
    assert q_diag[10] == 1e-3
    assert q_diag[12] == 1e-5  # K
    assert np.all(q_diag[13:] == 1e-5)  # w_ij
    # Other state variables should have 0 process noise
    assert q_diag[0] == 0.0
    assert q_diag[1] == 0.0


def test_parameter_constraints() -> None:
    """Verify that constraints (non-negativity of K/w, zero w_ii) are enforced after update."""
    conn = _toy_connectome(2)
    est = UKFEstimator(
        dt=1e-4,
        n_nodes=2,
        gain=conn.gain,
        gamma=conn.gamma,
        delays=conn.delays,
        initial_k=0.5,
        initial_w=conn.weights,
    )

    # Manually overwrite the UKF state vector with negative values and non-zero diagonal connection weights
    x_test = np.zeros(17)
    x_test[12] = -0.5  # K
    x_test[13] = -0.1  # w_00 (diagonal, negative)
    x_test[14] = 0.8  # w_01 (positive)
    x_test[15] = -0.2  # w_10 (negative)
    x_test[16] = 0.5  # w_11 (diagonal, positive)
    est.ukf.x = x_test

    # Dummy measurements and inputs
    y_mea = np.zeros(2)
    u = np.zeros(1)

    # Perform step (will predict, update, and clip)
    x_hat, _log = est.update(0.0, y_mea, u)

    # Check returned state matches augmented dimension (17)
    assert isinstance(x_hat, np.ndarray)
    assert x_hat.shape == (17,)

    # Check constraints are enforced
    assert est.ukf.x[12] >= 0.0  # K >= 0
    assert est.ukf.x[13] == 0.0  # w_00 = 0 (diagonal)
    assert est.ukf.x[14] >= 0.0  # w_01 >= 0
    assert est.ukf.x[15] >= 0.0  # w_10 >= 0
    assert est.ukf.x[16] == 0.0  # w_11 = 0 (diagonal)


def test_state_transition_matches_dynamics() -> None:
    """Verify that fx transition matches a deterministic step of JansenRitDynamics."""
    conn = _toy_connectome(2)
    params = JansenRitParams(A=3.25, sigma=0.0)  # deterministic
    dt = 1e-4

    # Run true dynamics step
    dyn = JansenRitDynamics(dt=dt, connectome=conn, K=0.5, params=params, seed=42)
    rng = np.random.default_rng(123)
    x_dyn = rng.normal(size=(6, 2))
    dyn.x = x_dyn.copy()
    out_dyn, _ = dyn.evaluate(0.0, 0.0)

    # Run estimator state transition
    est = UKFEstimator(
        dt=dt,
        n_nodes=2,
        gain=conn.gain,
        gamma=conn.gamma,
        delays=conn.delays,
        initial_k=0.5,
        initial_w=conn.weights,
        params=params,
    )

    x_aug = np.zeros(17)
    x_aug[:12] = x_dyn.flatten()
    x_aug[12] = 0.5  # K
    x_aug[13:] = conn.weights.flatten()

    # Call ukf.fx
    x_next_est = est.ukf.fx(x_aug, dt, 0.0, 0)

    # Check matching dynamic states
    np.testing.assert_allclose(x_next_est[:12], out_dyn, atol=1e-12)
    # Check parameters remain constant in prediction
    np.testing.assert_allclose(x_next_est[12:], x_aug[12:], atol=1e-12)


def test_measurement_matches_sensor() -> None:
    """Verify that hx maps dynamic states to the configured selected EEG channels."""
    conn = _toy_connectome(2)
    est = UKFEstimator(
        dt=1e-4,
        n_nodes=2,
        gain=conn.gain,
        selected_channels=[0],  # only measure channel 0
    )

    x_aug = np.zeros(17)
    # Set x2 and x3 on nodes: LFP = x2 - x3
    x_aug[2] = 2.0  # node 0, x2
    x_aug[3] = 3.0  # node 1, x2
    x_aug[4] = 0.5  # node 0, x3
    x_aug[5] = 1.0  # node 1, x3

    y_node = np.array([1.5, 2.0])  # [2.0 - 0.5, 3.0 - 1.0]
    expected_eeg = conn.gain @ y_node
    expected_sub = expected_eeg[[0]]

    y_est = est.ukf.hx(x_aug)
    np.testing.assert_allclose(y_est, expected_sub, atol=1e-12)


def test_estimator_from_config() -> None:
    """Smoke test for creating UKFEstimator from a configuration dictionary."""
    config = {
        "dt": 1e-4,
        "speed": 50.0,
        "selected_channels": ["F3", "P3"],
        "q_x5": 1e-4,
        "q_K": 1e-6,
        "q_w": 1e-6,
        "r_channel": 1e-3,
        "p_state": 1e-3,
        "p_K": 1e-3,
        "p_w": 1e-3,
        "initial_K": 0.5,
    }
    est = UKFEstimator.from_config(config)
    assert est.n_nodes == 76  # standard TVB connectome has 76 regions
    assert est.dim_z == 2  # selected_channels length
    assert est.ukf.x[6 * 76] == 0.5  # initial K value


def test_estimator_with_extensions() -> None:
    """Verify state dimensions, clipping, transition and measurement functions with extensions enabled."""
    conn = _toy_connectome(2)

    # Initial state dimension: 6*N (12) + 1 (K) + N^2 (4) = 17
    # + 2 EEG gains (n_nodes = 2) = 19
    est = UKFEstimator(
        dt=1e-4,
        n_nodes=2,
        gain=conn.gain,
        delays=conn.delays,
        estimate_eeg_gains=True,
    )

    assert est.dim_x == 19
    assert est.ukf.x.shape == (19,)
    assert est.ukf.P.shape == (19, 19)
    assert est.ukf.Q.shape == (19, 19)

    # Verify initial value of gains are 1.0 by default
    assert np.all(est.ukf.x[17:19] == 1.0)  # EEG gains

    # Manually set negative parameters to test clipping
    x_test = est.ukf.x.copy()
    x_test[12] = 0.5  # K
    x_test[17] = -0.1  # gain 0
    x_test[18] = 1.5  # gain 1
    est.ukf.x = x_test

    # Dummy update should clip negative values
    y_mea = np.zeros(2)
    u = 0.0
    _x_hat, log = est.update(0.0, y_mea, u)

    assert est.ukf.x[17] == 0.0  # gain 0 clipped
    assert est.ukf.x[18] >= 0.0
    assert log.g_est is not None

    # Test hx with estimated gains: y_eeg_i = g_i * y_node_i
    x_hx = np.zeros(19)
    x_hx[2] = 2.0  # node 0, x2
    x_hx[4] = 0.5  # node 0, x3 => LFP_0 = 1.5
    x_hx[3] = 3.0  # node 1, x2
    x_hx[5] = 1.0  # node 1, x3 => LFP_1 = 2.0
    x_hx[17] = 2.0  # gain 0
    x_hx[18] = 0.5  # gain 1

    # expected LFP: [1.5 * 2.0, 2.0 * 0.5] = [3.0, 1.0]
    expected_eeg = np.array([3.0, 1.0])
    y_est = est.ukf.hx(x_hx)
    np.testing.assert_allclose(y_est, expected_eeg, atol=1e-12)


def test_estimator_gamma_from_target_electrode() -> None:
    """Verify that steering matrix gamma is correctly constructed from target electrodes by name or index."""
    # Use n_nodes=4 to keep allocations small; the electrode-resolution logic is independent of network size.
    # Test case 1: Target electrodes picked by name
    config_names = {
        "dt": 1e-4,
        "speed": 50.0,
        "n_nodes": 4,
        "target_electrode": ["F3", "P3"],
    }
    est_names = UKFEstimator.from_config(config_names)
    assert est_names.n_elec == 2
    assert est_names.gamma_2d is not None
    assert est_names.gamma_2d.shape == (2, est_names.n_nodes)

    # Test case 2: Target electrodes picked by index
    config_indices = {
        "dt": 1e-4,
        "speed": 50.0,
        "n_nodes": 4,
        "target_electrode": [2, 5],
    }
    est_indices = UKFEstimator.from_config(config_indices)
    assert est_indices.n_elec == 2
    assert est_indices.gamma_2d is not None
    assert est_indices.gamma_2d.shape == (2, est_indices.n_nodes)

    # Verify that the calculated gammas are identical (electrode index matches label)
    conn = load_connectome()
    labels = [conn.channel_labels[2], conn.channel_labels[5]]

    config_matched_names = {
        "dt": 1e-4,
        "speed": 50.0,
        "n_nodes": 4,
        "target_electrode": labels,
    }
    est_matched_names = UKFEstimator.from_config(config_matched_names)
    assert est_indices.gamma_2d is not None
    assert est_matched_names.gamma_2d is not None
    np.testing.assert_allclose(est_indices.gamma_2d, est_matched_names.gamma_2d)

    # Test case 3: Same logic applies to JansenRitDynamics.from_config
    config_dynamics = {
        "dt": 1e-4,
        "speed": 50.0,
        "K": 0.5,
        "n_nodes": 4,
        "target_electrode": ["F3", "P3"],
    }
    dyn = JansenRitDynamics.from_config(config_dynamics)
    assert dyn.n_elec == 2
    assert dyn.gamma_2d is not None
    assert dyn.gamma_2d.shape == (2, dyn.x.shape[1])


def test_estimator_update_accepts_pre_subsetted_and_full() -> None:
    """Verify that update accepts both full-channel arrays and pre-subsetted EEG arrays."""
    conn = _toy_connectome(2)
    # Selected channels is index 0 only
    est = UKFEstimator(
        dt=1e-4,
        n_nodes=2,
        gain=conn.gain,
        selected_channels=[0],
    )
    # Full channel vector of shape (2,)
    y_full = np.array([1.2, 3.4])
    # Pre-subsetted vector of shape (1,)
    y_sub = np.array([1.2])

    # Try update with full channel vector
    x_hat_full, _ = est.update(0.0, y_full, 0.0)
    # Try update with pre-subsetted vector
    x_hat_sub, _ = est.update(0.0, y_sub, 0.0)

    assert isinstance(x_hat_full, np.ndarray)
    assert x_hat_full.shape == (17,)
    assert isinstance(x_hat_sub, np.ndarray)
    assert x_hat_sub.shape == (17,)

    # Test error raising for invalid shape
    with pytest.raises(ValueError, match="matches neither dim_z"):
        est.update(0.0, np.array([1.2, 3.4, 5.6]), 0.0)
