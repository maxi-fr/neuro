"""Stage 2 validation test suite for coupled whole-brain networks.

Covers:
- All-healthy control (background behavior, quiescent or low amplitude)
- EZ/PZ recruitment (oscillations spreading from EZ to other regions)
- Coupling correctness (network isolation tests)
- Delayed vs. instantaneous comparison (delay impacts propagation onset)
"""

from dataclasses import replace

import numpy as np
import pytest

from neuro.connectome import Connectome, load_connectome
from neuro.jansen_rit import JansenRitParams, output, simulate_network


@pytest.fixture(scope="module")
def connectome() -> Connectome:
    """Load the TVB connectome once for the module and scale its weights.

    Dividing by 1.40 calibrates the TVB human connectome density so that
    the all-healthy control stays quiescent, while the EZ/PZ network
    successfully recruits downstream nodes under noise.
    """
    conn = load_connectome()
    return replace(conn, weights=conn.weights / 1.40)


@pytest.fixture(scope="module")
def ez_pz_indices(connectome: Connectome) -> tuple[list[int], list[int]]:
    """Indices of EZ and PZ regions in the connectome."""
    ez_names = ("lHC", "lPHC", "lAMYG")
    pz_names = ("lTCI", "lTCV")
    ez_idxs = [connectome.region_index[name] for name in ez_names]
    pz_idxs = [connectome.region_index[name] for name in pz_names]
    return ez_idxs, pz_idxs


def _get_onset_time(y: np.ndarray, t: np.ndarray, threshold: float = 5.0) -> float:
    """Find the first time the absolute signal crosses a threshold."""
    idx = np.where(np.abs(y) > threshold)[0]
    if len(idx) == 0:
        return float("inf")
    return t[idx[0]]


def test_network_all_healthy_control(connectome: Connectome) -> None:
    """If all nodes are healthy (A = 3.25) and K = 0.75, the network stays quiescent."""
    # Deterministic control stays flat
    _, x_traj = simulate_network(
        params=JansenRitParams(A=3.25),
        connectome=connectome,
        K=0.75,
        duration=2.0,
        deterministic=True,
    )
    y = output(x_traj)
    # Steady window after transient
    y_steady = y[:, 1000:]
    ptps = np.ptp(y_steady, axis=1)
    assert np.all(ptps < 0.05)

    # Stochastic control stays in background range (< 5.0)
    _t, x_traj_noise = simulate_network(
        params=JansenRitParams(A=3.25),
        connectome=connectome,
        K=0.75,
        duration=2.0,
        deterministic=False,
        seed=42,
    )
    y_noise = output(x_traj_noise)
    y_noise_steady = y_noise[:, 1000:]
    ptps_noise = np.ptp(y_noise_steady, axis=1)
    assert np.all(ptps_noise < 5.0)
    assert np.all(ptps_noise > 0.5)


def test_network_recruitment(connectome: Connectome, ez_pz_indices: tuple[list[int], list[int]]) -> None:
    """With EZ/PZ gains set, oscillation starts in EZ and recruits other nodes."""
    ez_idxs, pz_idxs = ez_pz_indices
    n_nodes = len(connectome.region_labels)

    a_gains = np.full(n_nodes, 3.25)
    a_gains[ez_idxs] = 3.6
    a_gains[pz_idxs] = 3.4
    params = JansenRitParams(A=a_gains)

    t, x_traj = simulate_network(
        params=params,
        connectome=connectome,
        K=0.75,
        duration=4.0,
        deterministic=False,
        use_delays=True,
        seed=42,
    )
    y = output(x_traj)

    # EZ nodes should oscillate immediately
    y_steady = y[:, 1000:]
    ptps = np.ptp(y_steady, axis=1)
    assert np.all(ptps[ez_idxs] > 5.0)

    # At least some PZ nodes should get recruited
    assert np.any(ptps[pz_idxs] > 5.0)

    # Onsets: EZ nodes start first, then PZ, then others
    onset_times = np.array([_get_onset_time(y[i], t, threshold=5.0) for i in range(n_nodes)])

    # Check that EZ onsets are small and finite
    assert np.all(onset_times[ez_idxs] < 1.0)
    # Check that PZ onsets are finite (recruited)
    assert np.all(onset_times[pz_idxs] < float("inf"))
    # Check that PZ onsets are strictly after the start of EZ oscillations
    assert np.all(onset_times[pz_idxs] > np.min(onset_times[ez_idxs]))


def test_network_coupling_correctness(connectome: Connectome, ez_pz_indices: tuple[list[int], list[int]]) -> None:
    """Zeroing incoming weights to a downstream node prevents its recruitment,

    while isolated EZ nodes still oscillate.
    """
    ez_idxs, pz_idxs = ez_pz_indices
    n_nodes = len(connectome.region_labels)

    a_gains = np.full(n_nodes, 3.25)
    a_gains[ez_idxs] = 3.6
    a_gains[pz_idxs] = 3.4
    params = JansenRitParams(A=a_gains)

    # Create a connectome copy with modified weights
    custom_connectome = replace(connectome, weights=connectome.weights.copy())

    # 1. Isolate a PZ node (e.g., lTCI) by zeroing its incoming connections
    ltci_idx = connectome.region_index["lTCI"]
    custom_connectome.weights[ltci_idx, :] = 0.0

    # 2. Isolate one EZ node (e.g., lHC) by zeroing its incoming connections
    lhc_idx = connectome.region_index["lHC"]
    custom_connectome.weights[lhc_idx, :] = 0.0

    _t, x_traj = simulate_network(
        params=params,
        connectome=custom_connectome,
        K=0.75,
        duration=4.0,
        deterministic=False,
        seed=42,
    )
    y = output(x_traj)
    y_steady = y[:, 1000:]
    ptps = np.ptp(y_steady, axis=1)

    # Isolated PZ node (lTCI) must stop seizing (stays in background range < 5.0)
    assert ptps[ltci_idx] < 5.0

    # Isolated EZ node (lHC) must still oscillate on its own (since A = 3.6 is limit cycle)
    assert ptps[lhc_idx] > 5.0

    # Other non-isolated PZ nodes (e.g., lTCV) should still get recruited
    ltcv_idx = connectome.region_index["lTCV"]
    assert ptps[ltcv_idx] > 5.0


def test_network_delays_vs_instantaneous(connectome: Connectome, ez_pz_indices: tuple[list[int], list[int]]) -> None:
    """Delays change recruitment propagation timing but not the ability to seize."""
    ez_idxs, pz_idxs = ez_pz_indices
    n_nodes = len(connectome.region_labels)

    a_gains = np.full(n_nodes, 3.25)
    a_gains[ez_idxs] = 3.6
    a_gains[pz_idxs] = 3.4
    params = JansenRitParams(A=a_gains)

    t, x_traj_delayed = simulate_network(
        params=params,
        connectome=connectome,
        K=0.75,
        duration=4.0,
        deterministic=False,
        use_delays=True,
        seed=42,
    )
    y_delayed = output(x_traj_delayed)

    t, x_traj_instant = simulate_network(
        params=params,
        connectome=connectome,
        K=0.75,
        duration=4.0,
        deterministic=False,
        use_delays=False,
        seed=42,
    )
    y_instant = output(x_traj_instant)

    # Both networks should successfully seize/recruit
    ptps_delayed = np.ptp(y_delayed[:, 1000:], axis=1)
    ptps_instant = np.ptp(y_instant[:, 1000:], axis=1)
    assert np.all(ptps_delayed[ez_idxs] > 5.0)
    assert np.all(ptps_instant[ez_idxs] > 5.0)

    # Compare recruitment onset times for a PZ node (e.g. lTCI)
    ltci_idx = connectome.region_index["lTCI"]
    onset_delayed = _get_onset_time(y_delayed[ltci_idx], t)
    onset_instant = _get_onset_time(y_instant[ltci_idx], t)

    assert onset_delayed < float("inf")
    assert onset_instant < float("inf")
    # Delayed propagation should make the seizure reach lTCI later
    assert onset_delayed > onset_instant
