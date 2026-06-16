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
import numpy.typing as npt

from neuro.jansen_rit import output, simulate_network

if TYPE_CHECKING:
    from neuro.connectome import Connectome
    from neuro.jansen_rit import JansenRitParams

FloatArray = npt.NDArray[np.float64]


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


def compute_channel_gramians(  # noqa: PLR0913
    x0: FloatArray,
    params: JansenRitParams,
    connectome: Connectome,
    K: float,  # noqa: N803
    dt: float,
    duration: float,
    epsilon: float = 1e-6,
    decimate: int = 50,
) -> FloatArray:
    """Per-channel empirical observability Gramian around the linearization point ``x0``.

    For each of the ``6 * N`` flattened state dimensions, perturbs ``x0`` by
    ``epsilon`` and re-simulates (noiseless, ``sigma = 0``) to build the EEG
    sensitivity ``Psi[j, i, t] = (eeg_pert[i, t] - eeg_base[i, t]) / epsilon``. Because
    ``Wo(S) = sum_t Psi_S(t)^T Psi_S(t) dt`` decomposes additively over the channels in
    ``S``, this returns one ``(6N, 6N)`` Gramian per channel; the Gramian of any
    channel subset is the sum of its channels' matrices (see :func:`select_channels`).

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

    Returns
    -------
    FloatArray
        Per-channel observability Gramians, shape (n_channels, 6N, 6N).
    """
    n_state, n_nodes = x0.shape
    n_dim = n_state * n_nodes
    det_params = replace(params, sigma=0.0)

    def _run_eeg(x_init: FloatArray) -> FloatArray:
        _, x_traj = simulate_network(
            params=det_params,
            duration=duration,
            dt=dt,
            connectome=connectome,
            K=K,
            initial_state=x_init,
            seed=0,
        )
        eeg = connectome.gain @ output(x_traj)
        return eeg[:, ::decimate]

    eeg_base = _run_eeg(x0)
    n_channels, n_samples = eeg_base.shape

    psi = np.empty((n_dim, n_channels, n_samples), dtype=np.float64)
    for j in range(n_dim):
        state_idx, node_idx = divmod(j, n_nodes)
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
