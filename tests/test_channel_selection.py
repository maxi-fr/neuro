import numpy as np
import pytest

from neuro.connectome import Connectome
from neuro.jansen_rit import JansenRitParams
from utils.channel_selection import (
    compute_channel_gramians,
    leadfield_focus_scores,
    pca_loading_scores,
    qr_pivot_selection,
    reconstruction_error,
    select_channels,
)

_DT = 1e-4


def _two_node_connectome() -> Connectome:
    """Minimal 2-node, 2-channel connectome with a one-way coupling and identity gain."""
    weights = np.zeros((2, 2))
    weights[1, 0] = 0.6
    return Connectome(
        K=1.0,
        weights=weights,
        tract_lengths=np.zeros((2, 2)),
        centres=np.zeros((2, 3)),
        region_labels=np.array(["n0", "n1"]),
        hemispheres=np.array([False, False]),
        speed=1.0,
        delays=np.zeros((2, 2)),
        gain=np.eye(2),
        channel_labels=np.array(["c0", "c1"]),
        region_index={"n0": 0, "n1": 1},
        channel_index={"c0": 0, "c1": 1},
        gamma=np.zeros((1, 2)),
    )


def test_pca_loading_scores_ranks_dominant_channel() -> None:
    rng = np.random.default_rng(0)
    n_samples = 2000
    t = np.arange(n_samples) * 1e-3

    strong = 5.0 * np.sin(2 * np.pi * 3.0 * t)
    weak1 = 0.01 * rng.standard_normal(n_samples)
    weak2 = 0.01 * rng.standard_normal(n_samples)
    signals = np.stack([strong, weak1, weak2])

    scores, explained_variance_ratio = pca_loading_scores(signals, variance_threshold=0.95)
    assert scores.shape == (3,)
    assert explained_variance_ratio.shape == (3,)
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]


def test_pca_loading_scores_full_variance_threshold() -> None:
    rng = np.random.default_rng(1)
    signals = rng.standard_normal((4, 100))
    scores, explained_variance_ratio = pca_loading_scores(signals, variance_threshold=1.0)
    assert scores.shape == (4,)
    assert explained_variance_ratio.shape == (4,)
    assert np.all(np.isfinite(scores))
    assert np.all(np.isfinite(explained_variance_ratio))
    assert np.allclose(np.sum(explained_variance_ratio), 1.0)


def test_pca_loading_scores_invalid_shape() -> None:
    with pytest.raises(ValueError, match="2-D array"):
        pca_loading_scores(np.zeros(10))


def test_qr_pivot_selection_avoids_redundant_channels() -> None:
    n_samples = 2000
    t = np.arange(n_samples) * 1e-3

    loud = 5.0 * np.sin(2 * np.pi * 3.0 * t)
    duplicate = loud + 1e-6 * np.sin(2 * np.pi * 11.0 * t)  # near-copy of the loud channel
    independent = 4.0 * np.sin(2 * np.pi * 7.0 * t)  # loud but orthogonal
    weak = 0.01 * np.sin(2 * np.pi * 13.0 * t)
    signals = np.stack([loud, duplicate, independent, weak])

    selected = qr_pivot_selection(signals, n_select=2, variance_threshold=0.95)
    # First pivot is one of the loud near-duplicates; the second is the orthogonal
    # channel, not the redundant copy a pure loading-score ranking would pick.
    assert selected[0] in {0, 1}
    assert selected[1] == 2


def test_qr_pivot_selection_invalid_shape() -> None:
    with pytest.raises(ValueError, match="2-D array"):
        qr_pivot_selection(np.zeros(10), n_select=2)


def test_leadfield_focus_scores() -> None:
    gain = np.array([[1.0, -2.0, 3.0], [0.0, 4.0, 0.0]])
    scores = leadfield_focus_scores(gain, focus_regions=[0, 2])
    assert np.allclose(scores, [4.0, 0.0])


def test_reconstruction_error_recovers_linear_target() -> None:
    n_samples = 1000
    t = np.arange(n_samples) * 1e-3
    source = np.sin(2 * np.pi * 2.0 * t)
    linear_target = 2.0 * source - 3.0  # exact affine function of the source
    independent = np.sin(2 * np.pi * 9.0 * t + 1.0)
    signals = np.stack([source, linear_target, independent])
    train, test = signals[:, :600], signals[:, 600:]

    # A linearly dependent target is reconstructed almost exactly from the source.
    assert reconstruction_error(train, test, selected=[0], targets=[1]) < 1e-6
    # An orthogonal target cannot be reconstructed from the source.
    assert reconstruction_error(train, test, selected=[0], targets=[2]) > 0.5


def test_compute_channel_gramians_node_subset_shape() -> None:
    connectome = _two_node_connectome()
    params = JansenRitParams(A=3.25, sigma=0.0)
    x0 = np.zeros((6, 2))

    gramians = compute_channel_gramians(
        x0, params, connectome, K=0.54, dt=_DT, duration=0.05, epsilon=1e-5, decimate=10, node_subset=[1]
    )
    # Only node 1's six states are perturbed, so each Gramian is 6x6, not 12x12.
    assert gramians.shape == (2, 6, 6)


def test_compute_channel_gramians_shape_and_psd() -> None:
    connectome = _two_node_connectome()
    params = JansenRitParams(A=3.25, sigma=0.0)
    x0 = np.zeros((6, 2))

    gramians = compute_channel_gramians(
        x0,
        params,
        connectome,
        K=0.54,
        dt=_DT,
        duration=0.05,
        epsilon=1e-5,
        decimate=10,
    )

    n_dim = 6 * 2
    assert gramians.shape == (2, n_dim, n_dim)
    for gramian in gramians:
        # Symmetric (each is psi_i @ psi_i.T * dt) and positive semi-definite.
        assert np.allclose(gramian, gramian.T)
        eigvals = np.linalg.eigvalsh(gramian)
        assert np.all(eigvals >= -1e-12)


def test_select_channels_monotonic_and_valid() -> None:
    rng = np.random.default_rng(2)
    n_channels, n_dim = 5, 4

    gramians = np.empty((n_channels, n_dim, n_dim))
    for i in range(n_channels):
        a = rng.standard_normal((n_dim, n_dim))
        gramians[i] = a @ a.T  # PSD

    for criterion in ("min_eig", "det"):
        selected = select_channels(gramians, n_select=n_channels, criterion=criterion)
        assert len(selected) == n_channels
        assert set(selected) == set(range(n_channels))

        # Adding more channels can only grow the accumulated Gramian (Loewner order),
        # so the criterion score is monotonically non-decreasing.
        accumulated = np.zeros((n_dim, n_dim))
        scores = []
        for idx in selected:
            accumulated = accumulated + gramians[idx]
            if criterion == "det":
                _, score = np.linalg.slogdet(accumulated + 1e-9 * np.eye(n_dim))
            else:
                score = np.linalg.eigvalsh(accumulated + 1e-9 * np.eye(n_dim)).min()
            scores.append(score)

        assert np.all(np.diff(scores) >= -1e-9)


def test_select_channels_invalid_n_select() -> None:
    gramians = np.zeros((3, 2, 2))
    with pytest.raises(ValueError, match="n_select"):
        select_channels(gramians, n_select=0)
    with pytest.raises(ValueError, match="n_select"):
        select_channels(gramians, n_select=4)
