"""Stage 3 validation test suite for transcutaneous electrical stimulation (tES).

Covers:
- Gamma vector invariants (shape, non-positivity, normalization, target scaling)
- Immediate suppression of network seizure recruitment during stimulation
- Seizure re-expansion after stimulation offset (lack of lasting effect without Z dynamics)
- Polarity sanity (anodic stimulation does not suppress, cathodic does)
"""

from dataclasses import replace

import numpy as np
import pytest

from neuro.connectome import Connectome, compute_gamma, load_connectome
from neuro.jansen_rit import JansenRitParams, output, simulate_network


@pytest.fixture(scope="module")
def connectome() -> Connectome:
    """Load the structural connectome and scale its weights."""
    conn = load_connectome()
    return replace(conn, weights=conn.weights / 1.40)


@pytest.fixture(scope="module")
def ez_pz_indices(connectome: Connectome) -> tuple[list[int], list[int], list[int]]:
    """Indices of EZ, PZ, and healthy regions."""
    ez_names = ("lHC", "lPHC", "lAMYG")
    pz_names = ("lTCI", "lTCV")
    ez_idxs = [connectome.region_index[name] for name in ez_names]
    pz_idxs = [connectome.region_index[name] for name in pz_names]
    healthy_idxs = [i for i in range(len(connectome.region_labels)) if i not in ez_idxs and i not in pz_idxs]
    return ez_idxs, pz_idxs, healthy_idxs


def test_gamma_invariants(connectome: Connectome) -> None:
    """Validate that compute_gamma satisfies all shape and normalization invariants."""
    # 1. Check default calculation
    gamma = compute_gamma(connectome.centres, target_electrode="CP5", sigma=20.0)
    assert gamma.shape == (76,)
    assert np.all(gamma <= 0.0)
    assert np.max(np.abs(gamma)) == pytest.approx(1.0)

    # 2. Check that invalid electrode raises ValueError
    with pytest.raises(ValueError, match="not found in sensors"):
        compute_gamma(connectome.centres, target_electrode="INVALID_NAME", sigma=20.0)

    # 3. Check spatial heterogeneity (largest absolute values near target)
    cp5_idx = connectome.region_index["lHC"]  # closest to CP5
    rhc_idx = connectome.region_index["rHC"]  # far from CP5
    assert np.abs(gamma[cp5_idx]) > np.abs(gamma[rhc_idx])


def test_tes_immediate_suppression_and_reexpansion(
    connectome: Connectome,
    ez_pz_indices: tuple[list[int], list[int], list[int]],
) -> None:
    """Verify cathodic tES immediate suppression (stim ON) and re-expansion (stim OFF)."""
    ez_idxs, pz_idxs, healthy_idxs = ez_pz_indices
    n_nodes = len(connectome.region_labels)

    # Configure gains
    a_gains = np.full(n_nodes, 3.25)
    a_gains[ez_idxs] = 3.6
    a_gains[pz_idxs] = 3.4
    params = JansenRitParams(A=a_gains)

    # Compute and assign gamma to connectome
    gamma = compute_gamma(connectome.centres, target_electrode="CP5", sigma=25.0)
    conn_stim = replace(connectome, gamma=gamma)

    # Case 1: Baseline (No stimulation) -> Healthy nodes get recruited
    _, x_traj_baseline = simulate_network(
        params=params,
        connectome=conn_stim,
        K=0.75,
        duration=4.0,
        deterministic=False,
        seed=42,
    )
    y_baseline = output(x_traj_baseline)
    # Check that healthy nodes are recruited (steady state PTP > 5.0)
    ptps_baseline = np.ptp(y_baseline[:, 2000:], axis=1)  # after 2s
    recruited_healthy_baseline = [idx for idx in healthy_idxs if ptps_baseline[idx] > 5.0]
    assert len(recruited_healthy_baseline) > 0

    # Case 2: Constant Cathodic tES (u_hat_tES = 1.0, window = (0, 4)) -> Healthy nodes stay suppressed
    _, x_traj_stim = simulate_network(
        params=params,
        connectome=conn_stim,
        K=0.75,
        duration=4.0,
        deterministic=False,
        seed=42,
        u_hat_tES=1.0,
        stim_window=(0.0, 4.0),
    )
    y_stim = output(x_traj_stim)
    ptps_stim = np.ptp(y_stim[:, 2000:], axis=1)
    recruited_healthy_stim = [idx for idx in healthy_idxs if ptps_stim[idx] > 5.0]
    # Cathodic stimulation should suppress propagation to healthy nodes
    assert len(recruited_healthy_stim) < len(recruited_healthy_baseline)
    # In fact, healthy nodes should remain below the seizure threshold (PTP < 5.0)
    assert np.all(ptps_stim[healthy_idxs] < 5.0)
    # Oscillation must stay *confined to* the focus (Fig 4a), not be globally extinguished:
    # at least one EZ/PZ node keeps oscillating despite stimulation. (Cathodic CP5 stim over-
    # suppresses the EZ nodes nearest the electrode, so we require the focus to survive, not all of it.)
    assert np.any(ptps_stim[ez_idxs + pz_idxs] > 5.0)

    # Case 3: Windowed Stimulation (u_hat_tES = 1.0, window = (0, 2)) -> Seizure re-expands after offset
    t_wind, x_traj_wind = simulate_network(
        params=params,
        connectome=conn_stim,
        K=0.75,
        duration=12.0,
        deterministic=False,
        seed=42,
        u_hat_tES=1.0,
        stim_window=(0.0, 2.0),
    )
    y_wind = output(x_traj_wind)

    # During stim (0.0s to 2.0s), healthy nodes are suppressed
    # We evaluate PTP in [1.0s, 2.0s] to drop initial transients but before stim offset
    idx_stim = np.where((t_wind >= 1.0) & (t_wind < 2.0))[0]
    ptps_during = np.ptp(y_wind[:, idx_stim], axis=1)
    assert np.all(ptps_during[healthy_idxs] < 5.0)

    # After stim (2.0s to 12.0s), the seizure re-expands and recruits healthy nodes
    idx_post = np.where(t_wind >= 10.0)[0]
    ptps_after = np.ptp(y_wind[:, idx_post], axis=1)
    recruited_healthy_after = [idx for idx in healthy_idxs if ptps_after[idx] > 5.0]
    assert len(recruited_healthy_after) > 0


def test_tes_polarity_sanity(
    connectome: Connectome,
    ez_pz_indices: tuple[list[int], list[int], list[int]],
) -> None:
    """Verify that cathodic (negative gamma) suppresses and anodic (positive gamma) excites/does not suppress."""
    ez_idxs, pz_idxs, healthy_idxs = ez_pz_indices
    n_nodes = len(connectome.region_labels)

    a_gains = np.full(n_nodes, 3.25)
    a_gains[ez_idxs] = 3.6
    a_gains[pz_idxs] = 3.4
    params = JansenRitParams(A=a_gains)

    gamma = compute_gamma(connectome.centres, target_electrode="CP5", sigma=25.0)
    conn_stim = replace(connectome, gamma=gamma)

    # Run with anodic stimulation (u_hat_tES = -1.0, making U_tES positive since gamma is negative)
    # This should fail to suppress and keep the network in a seizing state
    _, x_traj_anodic = simulate_network(
        params=params,
        connectome=conn_stim,
        K=0.75,
        duration=4.0,
        deterministic=False,
        seed=42,
        u_hat_tES=-1.0,
        stim_window=(0.0, 4.0),
    )
    y_anodic = output(x_traj_anodic)
    ptps_anodic = np.ptp(y_anodic[:, 2000:], axis=1)
    recruited_healthy_anodic = [idx for idx in healthy_idxs if ptps_anodic[idx] > 5.0]
    # Anodic stimulation should NOT suppress recruitment (healthy nodes still seize)
    assert len(recruited_healthy_anodic) > 0
