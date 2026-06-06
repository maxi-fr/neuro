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
from neuro.jansen_rit import JansenRitDynamics, JansenRitParams, output, simulate_network

_DT = 1e-3


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
    # Deterministic (sigma = 0) control stays flat
    _, x_traj = simulate_network(
        params=JansenRitParams(A=3.25, sigma=0.0),
        connectome=connectome,
        K=0.75,
        duration=2.0,
        dt=_DT,
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
        dt=_DT,
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
        dt=_DT,
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
        dt=_DT,
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


def _two_node_delayed_connectome() -> Connectome:
    """Minimal 2-node connectome: one-way 0 -> 1 coupling with a 5-step conduction delay."""
    weights = np.zeros((2, 2))
    weights[1, 0] = 0.6  # node 1 receives from node 0 only
    delays = np.zeros((2, 2))
    delays[1, 0] = 5.0  # ms -> round(5 / (dt * 1000)) = 5 steps at dt = 1e-3
    return Connectome(
        weights=weights,
        tract_lengths=np.zeros((2, 2)),
        centres=np.zeros((2, 3)),
        region_labels=np.array(["n0", "n1"]),
        hemispheres=np.array([False, False]),
        speed=1.0,
        delays=delays,
        gain=np.zeros((1, 2)),
        channel_labels=np.array(["c0"]),
        region_index={"n0": 0, "n1": 1},
        channel_index={"c0": 0},
    )


def test_step_index_recovered_from_accumulated_time() -> None:
    """``round(t / dt)`` recovers the step index under the orchestrator's ``t += dt`` drift.

    ``JansenRitDynamics.dynamics`` derives its circular-history index from ``t`` instead of
    a stored counter. ``Simulation.run`` advances time by accumulation (``t += dt``), so the
    ``t`` handed to the plant drifts from the ideal ``k * dt``; this guards that the drift
    never flips the recovered index over a long horizon.
    """
    t = 0.0
    for k in range(100_000):
        assert round(t / _DT) == k
        t += _DT


def test_history_coupling_index_robust_to_accumulated_time() -> None:
    """Driving the plant with accumulated ``t`` (as ``Simulation.run`` does) matches exact ``k * dt``.

    The delayed coupling indexes a circular buffer by ``k = round(t / dt)``; if accumulated
    floating-point time shifted that index (e.g. ``int()`` truncation or drift), the downstream
    node would read the wrong delay slot and the two trajectories would diverge.
    """
    connectome = _two_node_delayed_connectome()
    params = JansenRitParams(A=np.array([3.6, 3.25]), sigma=0.0)
    initial_state = np.zeros((6, 2))
    initial_state[0, 0] = 1.0  # kick node 0 off the (unstable) equilibrium so it limit-cycles
    n_steps = 3000

    def run(*, accumulate: bool) -> np.ndarray:
        dyn = JansenRitDynamics(
            dt=_DT, params=params, connectome=connectome, K=0.75, seed=0, initial_state=initial_state
        )
        xs = np.zeros((6, 2, n_steps + 1))
        xs[:, :, 0] = dyn.x
        t = 0.0
        for k in range(n_steps):
            dyn.evaluate(t if accumulate else k * _DT, 0.0)
            xs[:, :, k + 1] = dyn.x
            t += _DT
        return xs

    x_exact = run(accumulate=False)
    x_accum = run(accumulate=True)

    # Coupling must actually have driven the downstream node, so the comparison is non-vacuous.
    assert np.ptp(output(x_exact)[1]) > 1.0
    # Exact index recovery -> bit-identical trajectories regardless of how t is produced.
    assert np.array_equal(x_exact, x_accum)


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
        dt=_DT,
        seed=42,
    )
    y_delayed = output(x_traj_delayed)

    # Instantaneous coupling = a connectome whose conduction delays are all zero.
    conn_instant = replace(connectome, delays=np.zeros_like(connectome.delays))
    _, x_traj_instant = simulate_network(
        params=params,
        connectome=conn_instant,
        K=0.75,
        duration=4.0,
        dt=_DT,
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
