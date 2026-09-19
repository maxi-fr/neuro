from __future__ import annotations

import numpy as np
import pytest

from neuro.connectome import Connectome
from neuro.stimulation.analytical import compute_gamma
from neuro.stimulation.montage_optimization import (
    SelectedMontageStim,
    build_target_field,
    load_field_projection_matrix,
    refit_active_montage_qp,
    solve_l1_montage_qp,
    sweep_sparse_montages,
)


@pytest.fixture(scope="module")
def connectome() -> Connectome:
    """Load the connectome once for the test module."""
    return Connectome.from_config({})


def test_build_target_field(connectome: Connectome) -> None:
    """Target field must have -s0 at target regions and zero elsewhere."""
    targets = ["lHC", "lPHC", "lAMYG"]
    s_star = build_target_field(connectome, targets, s0=0.8)
    n_nodes = len(connectome.region_labels)
    assert s_star.shape == (n_nodes,)

    target_indices = [connectome.region_index[r] for r in targets]
    np.testing.assert_allclose(s_star[target_indices], -0.8)

    non_target_indices = [i for i in range(n_nodes) if i not in target_indices]
    np.testing.assert_allclose(s_star[non_target_indices], 0.0)


def test_solve_l1_montage_qp_satisfies_kcl_and_bounds() -> None:
    """L1 regularized QP must satisfy Kirchhoff and current bounds."""
    rng = np.random.default_rng(42)
    n_nodes, n_channels = 20, 8
    l_mat = rng.standard_normal((n_nodes, n_channels))
    s_star = np.zeros(n_nodes)
    s_star[:3] = -1.0
    i_max = 2.0

    u = solve_l1_montage_qp(l_mat, s_star, lam=1e-3, i_max=i_max)
    assert u.shape == (n_channels,)
    assert np.isclose(np.sum(u), 0.0, atol=1e-5)
    assert np.all(np.abs(u) <= i_max + 1e-5)


def test_lambda_sweep_increases_sparsity() -> None:
    """High regularization must produce sparser or zeroed currents than low regularization."""
    rng = np.random.default_rng(42)
    n_nodes, n_channels = 20, 8
    l_mat = rng.standard_normal((n_nodes, n_channels))
    s_star = np.zeros(n_nodes)
    s_star[:3] = -1.0

    u_dense = solve_l1_montage_qp(l_mat, s_star, lam=1e-4, i_max=2.0)
    u_sparse = solve_l1_montage_qp(l_mat, s_star, lam=1e2, i_max=2.0)

    n_active_dense = int(np.sum(np.abs(u_dense) > 1e-4))
    n_active_sparse = int(np.sum(np.abs(u_sparse) > 1e-4))
    assert n_active_sparse <= n_active_dense
    assert n_active_sparse == 0


def test_refit_active_montage_qp_debiases() -> None:
    """Refitting on active electrodes removes shrinkage and reduces targeting MSE."""
    rng = np.random.default_rng(42)
    n_nodes, n_channels = 20, 6
    l_mat = rng.standard_normal((n_nodes, n_channels))
    s_star = np.zeros(n_nodes)
    s_star[:2] = -1.5

    active_indices = np.array([0, 1, 2])
    u_refit = refit_active_montage_qp(l_mat, s_star, active_indices, i_max=2.0)

    assert u_refit.shape == (3,)
    assert np.isclose(np.sum(u_refit), 0.0, atol=1e-5)
    assert np.all(np.abs(u_refit) <= 2.0 + 1e-5)
    assert np.linalg.norm(u_refit) > 0.0


def test_sweep_sparse_montages_returns_budgeted_candidates(connectome: Connectome) -> None:
    """Sweep discovers active montages within 2-5 electrode budget satisfying KCL."""
    targets = ["lHC", "lPHC", "lAMYG"]
    s_star = build_target_field(connectome, targets, s0=0.8)

    channels = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]
    l_mat = compute_gamma(connectome.centres, channels).T

    candidates = sweep_sparse_montages(
        l_mat,
        s_star,
        channel_labels=np.asarray(channels, dtype=np.str_),
        min_electrodes=2,
        max_electrodes=5,
        i_max=2.0,
        n_lambdas=15,
    )

    assert len(candidates) > 0
    for cand in candidates:
        assert 2 <= cand.n_active <= 5
        assert np.isclose(np.sum(cand.u_refit), 0.0, atol=1e-5)
        assert np.all(np.abs(cand.u_refit) <= 2.0 + 1e-5)
        assert cand.mse_refit <= cand.mse_l1 + 1e-6


def test_load_field_projection_matrix_roast(connectome: Connectome) -> None:
    """Loading projection matrix loads real ROAST 3D FEM model when >=10 channels."""
    l_mat, labels, is_fallback, desc = load_field_projection_matrix(
        "data/roast_field_projection_3d.npz",
        connectome,
    )
    assert is_fallback is False
    assert l_mat.shape == (len(connectome.region_labels), 63)
    assert len(labels) == 63
    assert "roast 3d fem" in desc.lower()


def test_load_field_projection_matrix_fallback(connectome: Connectome) -> None:
    """Loading projection matrix falls back gracefully to analytical model when <10 channels."""
    l_mat, labels, is_fallback, desc = load_field_projection_matrix(
        "data/roast_field_projection_3d_3ch.npz",
        connectome,
    )
    assert is_fallback is True
    assert l_mat.shape == (len(connectome.region_labels), 62)
    assert len(labels) == 62
    assert "analytical" in desc.lower()


def test_selected_montage_stim_projects_correctly() -> None:
    """SelectedMontageStim must project electrode currents to regional drive."""
    n_nodes, n_active = 10, 3
    l_subset = np.ones((n_nodes, n_active))
    labels = np.array(["C3", "C4", "Cz"])
    stim = SelectedMontageStim(l_subset, labels)

    assert stim.n_controls == 3
    u = np.array([1.0, -0.5, -0.5])
    drive = stim.project(u)
    assert drive.shape == (n_nodes,)
    np.testing.assert_allclose(drive, 0.0)

    with pytest.raises(ValueError, match="control has 2 electrodes"):
        stim.project(np.array([1.0, -1.0]))
