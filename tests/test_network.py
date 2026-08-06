from dataclasses import replace

import numpy as np
import pytest

from neuro.connectome import Connectome
from neuro.jansen_rit import JansenRitDynamics, JansenRitParams, lfp, simulate_network
from neuro.types import FloatArray

_DT = 1e-4
_K = 0.5357  # canonical global coupling used across these seizure-network tests


@pytest.fixture(scope="module")
def connectome() -> Connectome:
    """Load the TVB connectome once for the module."""
    return Connectome.from_config({})


def _run(
    conn: Connectome, *, a: float | FloatArray, duration: float, seed: int | None = None, deterministic: bool = False
) -> tuple[FloatArray, FloatArray]:
    """Build the plant on ``conn`` (at coupling ``_K``) and integrate it for ``duration`` s."""
    params = JansenRitParams(A=a, sigma=0.0) if deterministic else JansenRitParams(A=a)
    dyn = JansenRitDynamics(dt=_DT, params=params, conn=replace(conn, K=_K), seed=seed)
    return simulate_network(dyn=dyn, duration=duration)


@pytest.fixture(scope="module")
def ez_pz_indices(connectome: Connectome) -> tuple[list[int], list[int]]:
    """Indices of EZ and PZ regions in the connectome."""
    ez_names = ("lHC", "lPHC", "lAMYG")
    pz_names = ("lTCI", "lTCV")
    ez_idxs = [connectome.region_index[name] for name in ez_names]
    pz_idxs = [connectome.region_index[name] for name in pz_names]
    return ez_idxs, pz_idxs


def _get_onset_time(y: FloatArray, t: FloatArray, threshold: float = 5.0) -> float:
    """Find the first time the absolute signal crosses a threshold."""
    idx = np.where(np.abs(y) > threshold)[0]
    if len(idx) == 0:
        return float("inf")
    return t[idx[0]]


def test_network_all_healthy_control(connectome: Connectome) -> None:
    """If all nodes are healthy (A = 3.25) and K = 0.5357, the network stays quiescent."""
    # Deterministic (sigma = 0) control stays flat
    _, x_traj = _run(connectome, a=3.25, duration=2.0, deterministic=True)
    y = lfp(x_traj)
    # Steady window after transient
    y_steady = y[:, round(1.0 / _DT) :]
    ptps = np.ptp(y_steady, axis=1)
    assert np.all(ptps < 0.05)

    # Stochastic control stays in background range (< 5.0)
    _t, x_traj_noise = _run(connectome, a=3.25, duration=2.0, seed=69)
    y_noise = lfp(x_traj_noise)
    y_noise_steady = y_noise[:, round(1.0 / _DT) :]
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

    t, x_traj = _run(connectome, a=a_gains, duration=4.0, seed=69)
    y = lfp(x_traj)

    # EZ nodes should oscillate immediately
    y_steady = y[:, round(1.0 / _DT) :]
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

    # Create a connectome copy with modified weights
    custom_connectome = replace(connectome, weights=connectome.weights.copy())

    # 1. Isolate a PZ node (e.g., lTCI) by zeroing its incoming connections
    ltci_idx = connectome.region_index["lTCI"]
    custom_connectome.weights[ltci_idx, :] = 0.0

    # 2. Isolate one EZ node (e.g., lHC) by zeroing its incoming connections
    lhc_idx = connectome.region_index["lHC"]
    custom_connectome.weights[lhc_idx, :] = 0.0

    _t, x_traj = _run(custom_connectome, a=a_gains, duration=4.0, seed=69)
    y = lfp(x_traj)
    y_steady = y[:, round(1.0 / _DT) :]
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
    delays[1, 0] = 5.0  # ms -> round(5 / (dt * 1000)) = 50 steps at dt = 1e-4
    return Connectome(
        K=_K,
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
    n_steps = 30000  # 3 s at dt = 1e-4, enough for the downstream node to be driven

    def run(*, accumulate: bool) -> FloatArray:
        dyn = JansenRitDynamics(dt=_DT, params=params, conn=connectome, seed=0, initial_state=initial_state)
        xs = np.zeros((6, 2, n_steps + 1))
        xs[:, :, 0] = dyn.x
        t = 0.0
        for k in range(n_steps):
            dyn.evaluate(t if accumulate else k * _DT, np.array([0.0]))
            xs[:, :, k + 1] = dyn.x
            t += _DT
        return xs

    x_exact = run(accumulate=False)
    x_accum = run(accumulate=True)

    # Coupling must actually have driven the downstream node, so the comparison is non-vacuous.
    assert np.ptp(lfp(x_exact)[1]) > 1.0
    # Exact index recovery -> bit-identical trajectories regardless of how t is produced.
    assert np.array_equal(x_exact, x_accum)


def test_network_delays_vs_instantaneous(connectome: Connectome, ez_pz_indices: tuple[list[int], list[int]]) -> None:
    """Delays change recruitment propagation timing but not the ability to seize."""
    ez_idxs, pz_idxs = ez_pz_indices
    n_nodes = len(connectome.region_labels)

    a_gains = np.full(n_nodes, 3.25)
    a_gains[ez_idxs] = 3.6
    a_gains[pz_idxs] = 3.4

    t, x_traj_delayed = _run(connectome, a=a_gains, duration=4.0, seed=69)
    y_delayed = lfp(x_traj_delayed)

    # Instantaneous coupling = a connectome whose conduction delays are all zero.
    conn_instant = replace(connectome, delays=np.zeros_like(connectome.delays))
    _, x_traj_instant = _run(conn_instant, a=a_gains, duration=4.0, seed=69)
    y_instant = lfp(x_traj_instant)

    # Both networks should successfully seize/recruit
    ptps_delayed = np.ptp(y_delayed[:, round(1.0 / _DT) :], axis=1)
    ptps_instant = np.ptp(y_instant[:, round(1.0 / _DT) :], axis=1)
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
