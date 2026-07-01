"""EEG channel-selection metrics for model reduction.

Implements the two ranking criteria from ``knowledge-base/Notes/model_reduction.md``:

* :func:`pca_loading_scores` -- per-channel PCA loading scores from the SVD of an EEG
  trajectory.
* :func:`compute_channel_gramians` / :func:`select_channels` -- per-channel empirical
  observability Gramians (via finite-difference state perturbations) and greedy
  subset selection by determinant or minimum eigenvalue.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.linalg import qr

from neuro.jansen_rit import JansenRitParams, lfp, simulate_network

if TYPE_CHECKING:
    from collections.abc import Sequence

    from neuro.connectome import Connectome
    from neuro.types import FloatArray


def pca_loading_scores(signals: FloatArray, variance_threshold: float = 0.95) -> tuple[FloatArray, FloatArray]:
    """Per-channel PCA loading score and explained variance ratios.

    Centers each channel, takes the SVD of the centered data, selects the smallest
    number ``K`` of principal components whose cumulative explained variance reaches
    ``variance_threshold``, and returns:
    1. ``Score(i) = sum_{j=1}^K sigma_j^2 V[i,j]^2``.
    2. Explained variance ratio for each of the principal components.

    Parameters
    ----------
    signals
        Input signals, shape (n_channels, n_samples).
    variance_threshold
        Minimum cumulative fraction of variance the selected components must explain.

    Returns
    -------
    scores : FloatArray
        Per-channel loading score, shape (n_channels,).
    explained_variance_ratio : FloatArray
        Fraction of total variance explained by each principal component, shape (n_components,).

    Raises
    ------
    ValueError
        If signals is not a 2-D array.
    """
    if signals.ndim != 2:  # noqa: PLR2004
        msg = f"Expected 2-D array of shape (n_channels, n_samples), got shape {signals.shape}"
        raise ValueError(msg)

    centered = signals - signals.mean(axis=1, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered.T, full_matrices=False)

    explained_variance = singular_values**2
    explained_variance_ratio = (explained_variance / explained_variance.sum()).astype(np.float64)
    cumulative = np.cumsum(explained_variance_ratio)
    n_components = min(int(np.searchsorted(cumulative, variance_threshold)) + 1, singular_values.size)

    weighted = explained_variance[:n_components, None] * vt[:n_components, :] ** 2
    return weighted.sum(axis=0).astype(np.float64), explained_variance_ratio


def qr_pivot_selection(signals: FloatArray, n_select: int, variance_threshold: float = 0.95) -> list[int]:
    """Select non-redundant channels by pivoted QR (Q-DEIM) on the dominant POD modes.

    The SVD of the centered data gives spatial modes over channels (the rows of
    ``Vt``). A column-pivoted QR of the energy-weighted top-``K`` modes greedily
    picks the channels that span the dominant subspace with the least redundancy:
    the first pivot is the highest-variance channel (matching the top of
    :func:`pca_loading_scores`), and each subsequent pivot is the channel most
    orthogonal to those already chosen. This avoids the near-duplicate neighbours a
    pure loading-score ranking selects.

    Parameters
    ----------
    signals
        Input signals, shape (n_channels, n_samples).
    n_select
        Number of channels to select.
    variance_threshold
        Cumulative variance the retained principal components must explain; sets the
        number of modes ``K`` that QR pivots over.

    Returns
    -------
    list[int]
        Selected channel indices, in pivot (selection) order.

    Raises
    ------
    ValueError
        If signals is not a 2-D array.
    """
    if signals.ndim != 2:  # noqa: PLR2004
        msg = f"Expected 2-D array of shape (n_channels, n_samples), got shape {signals.shape}"
        raise ValueError(msg)

    centered = signals - signals.mean(axis=1, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered.T, full_matrices=False)
    cumulative = np.cumsum(singular_values**2 / (singular_values**2).sum())
    n_components = min(int(np.searchsorted(cumulative, variance_threshold)) + 1, singular_values.size)

    weighted_modes = singular_values[:n_components, None] * vt[:n_components, :]
    _, _, pivots = qr(weighted_modes, pivoting=True)
    return pivots[:n_select].tolist()


def leadfield_focus_scores(gain: FloatArray, focus_regions: Sequence[int]) -> FloatArray:
    """Per-channel absolute leadfield sensitivity onto a set of focus regions.

    Sums ``|L[:, r]|`` over the focus regions ``r``; channels that physically pick up
    the seizure focus score highest. Ranking by this score is purely structural (no
    simulation), so it makes the channel selection aware of the known epileptogenic
    zone instead of of overall signal amplitude.

    Parameters
    ----------
    gain
        EEG forward operator ``L``, shape (n_channels, n_regions).
    focus_regions
        Indices of the regions of interest (e.g. the EZ/PZ nodes).

    Returns
    -------
    FloatArray
        Per-channel focus score, shape (n_channels,).
    """
    return np.abs(gain[:, list(focus_regions)]).sum(axis=1).astype(np.float64)


def compute_channel_gramians(  # noqa: PLR0913
    x0: FloatArray,
    params: JansenRitParams,
    connectome: Connectome,
    K: float,
    dt: float,
    duration: float,
    epsilon: float = 1e-6,
    decimate: int = 50,
    node_subset: Sequence[int] | None = None,
) -> FloatArray:
    """Per-channel empirical observability Gramian around the linearization point ``x0``.

    For each perturbed state dimension, perturbs ``x0`` by ``epsilon`` and re-simulates
    (noiseless, ``sigma = 0``) to build the EEG sensitivity
    ``Psi[j, i, t] = (eeg_pert[i, t] - eeg_base[i, t]) / epsilon``. Because
    ``Wo(S) = sum_t Psi_S(t)^T Psi_S(t) dt`` decomposes additively over the channels in
    ``S``, this returns one ``(D, D)`` Gramian per channel; the Gramian of any channel
    subset is the sum of its channels' matrices (see :func:`select_channels`).

    By default all ``6 * N`` state dimensions are perturbed (``D = 6N``). Passing
    ``node_subset`` restricts the perturbations to the ``6 * len(node_subset)`` states of
    those nodes, so the Gramian measures observability of only those nodes' hidden states
    (``D = 6 * len(node_subset)``). For a focal seizure this both targets the criterion at
    the epileptogenic zone and keeps ``D`` small enough for the subset Gramian to be full
    rank from only a few channels, which is what makes the ``min_eig`` criterion meaningful.

    Parameters
    ----------
    x0
        Linearization point, shape (6, N).
    params
        Model parameters; ``sigma`` is overridden to ``0.0`` for deterministic
        sensitivity simulations.
    connectome
        Structural connectome, including the EEG gain ``L`` of shape (n_channels, N).
    K
        Global coupling strength.
    dt
        Integration step in seconds.
    duration
        Duration of each sensitivity simulation, in seconds.
    epsilon
        Perturbation size added to each state dimension.
    decimate
        Keep every ``decimate``-th sample of the EEG sensitivity before integrating,
        to bound memory use.
    node_subset
        Indices of the nodes whose states are perturbed. ``None`` perturbs every node.

    Returns
    -------
    FloatArray
        Per-channel observability Gramians, shape (n_channels, D, D), with
        ``D = 6 * len(node_subset)`` (or ``6N`` when ``node_subset`` is ``None``).
    """
    n_state, n_nodes = x0.shape
    nodes = list(range(n_nodes)) if node_subset is None else list(node_subset)
    perturbed = [(state_idx, node_idx) for state_idx in range(n_state) for node_idx in nodes]
    n_dim = len(perturbed)
    # Overlay the connectome's network structure onto the local (noiseless) params once;
    # only the initial state varies between runs.
    network = JansenRitParams.from_config({"connectome": connectome, "dt": dt, "params": {"K": K}})
    sim_params = replace(
        params,
        sigma=0.0,
        eeg_gain=network.eeg_gain,
        w_weights=network.w_weights,
        delay_steps=network.delay_steps,
        gamma=network.gamma,
        K=network.K,
    )

    def _run_eeg(x_init: FloatArray) -> FloatArray:
        _, x_traj = simulate_network(
            params=sim_params,
            duration=duration,
            dt=dt,
            initial_state=x_init,
            seed=0,
        )
        eeg = connectome.gain @ lfp(x_traj)
        return eeg[:, ::decimate]

    eeg_base = _run_eeg(x0)
    n_channels, n_samples = eeg_base.shape

    psi = np.empty((n_dim, n_channels, n_samples), dtype=np.float64)
    for j, (state_idx, node_idx) in enumerate(perturbed):
        x_pert = x0.copy()
        x_pert[state_idx, node_idx] += epsilon
        psi[j] = (_run_eeg(x_pert) - eeg_base) / epsilon

    dt_decim = dt * decimate
    gramians = np.empty((n_channels, n_dim, n_dim), dtype=np.float64)
    for i in range(n_channels):
        psi_i = psi[:, i, :]
        gramians[i] = (psi_i @ psi_i.T) * dt_decim
    return gramians


def select_channels(
    channel_gramians: FloatArray,
    n_select: int,
    criterion: Literal["det", "min_eig"] = "min_eig",
    ridge: float = 1e-9,
) -> list[int]:
    """Greedily select channel indices maximizing the observability Gramian criterion.

    Starting from an empty subset, repeatedly adds the channel whose per-channel
    Gramian (see :func:`compute_channel_gramians`) most increases ``det`` (via
    log-determinant) or the minimum eigenvalue of the accumulated subset Gramian
    ``sum_{i in S} channel_gramians[i] + ridge * I``.

    Parameters
    ----------
    channel_gramians
        Per-channel observability Gramians, shape (n_channels, n_dim, n_dim).
    n_select
        Number of channels to select.
    criterion
        ``"det"`` maximizes the log-determinant; ``"min_eig"`` maximizes the smallest
        eigenvalue of the accumulated Gramian.
    ridge
        Diagonal regularization added to the accumulated Gramian for numerical
        stability when the selected subset is small.

    Returns
    -------
    list[int]
        Selected channel indices, in selection order.

    Raises
    ------
    ValueError
        If ``n_select`` is not between 1 and the number of channels.
    """
    n_channels, n_dim, _ = channel_gramians.shape
    if not 1 <= n_select <= n_channels:
        msg = f"n_select must be between 1 and {n_channels}, got {n_select}"
        raise ValueError(msg)

    def _score(matrix: FloatArray) -> float:
        if criterion == "det":
            sign, logdet = np.linalg.slogdet(matrix)
            return float(logdet) if sign > 0 else float("-inf")
        return float(np.linalg.eigvalsh(matrix).min())

    accumulated = ridge * np.eye(n_dim, dtype=np.float64)
    remaining = set(range(n_channels))
    selected: list[int] = []
    for _ in range(n_select):
        best_idx = max(remaining, key=lambda i: _score(accumulated + channel_gramians[i]))
        selected.append(best_idx)
        accumulated = accumulated + channel_gramians[best_idx]
        remaining.remove(best_idx)
    return selected


def reconstruction_error(
    train: FloatArray,
    test: FloatArray,
    selected: Sequence[int],
    targets: Sequence[int] | None = None,
) -> float:
    """Held-out relative error of linearly reconstructing channels from a subset.

    Standardizes every channel by its training mean and standard deviation (so the
    score is not dominated by high-amplitude channels), fits a least-squares map from
    the ``selected`` channels to the ``targets`` on ``train``, and reports the relative
    Frobenius error on ``test``. ``targets`` defaults to all channels. Lower is better;
    this is the objective check that lets the selection criteria be compared head-to-head.

    Parameters
    ----------
    train, test
        Disjoint time windows of the same channels, each shape (n_channels, n_samples).
    selected
        Indices of the channels used as the reconstruction inputs.
    targets
        Indices of the channels to reconstruct; defaults to all channels.

    Returns
    -------
    float
        Relative L2 reconstruction error on ``test``.
    """
    mean = train.mean(axis=1, keepdims=True)
    std = train.std(axis=1, keepdims=True)
    std[std == 0.0] = 1.0
    train_z = (train - mean) / std
    test_z = (test - mean) / std

    target_idx = list(range(train.shape[0])) if targets is None else list(targets)
    coeffs, *_ = np.linalg.lstsq(train_z[list(selected)].T, train_z[target_idx].T, rcond=None)
    predicted = (test_z[list(selected)].T @ coeffs).T
    return float(np.linalg.norm(predicted - test_z[target_idx]) / np.linalg.norm(test_z[target_idx]))
