"""Collect seizure EEG data and rank EEG channels for model reduction.

Runs the whole-brain Jansen-Rit network with the EZ/PZ gain configuration from
``notebooks/interactive_simulation.py`` (a seizure that originates in
``lHC``/``lPHC``/``lAMYG`` and propagates to ``lTCI``/``lTCV``), saves the resulting EEG
to ``{output_dir}/seizure_data.npz``, and ranks the 62 EEG channels by two criteria from
``knowledge-base/Notes/model_reduction.md``:

* PCA loading scores of the steady-state seizure EEG.
* Empirical observability Gramian, evaluated via a greedy channel selection that
  maximizes the determinant or the minimum eigenvalue of the accumulated Gramian.

The Gramian computation runs ``6 * 76 + 1 = 457`` short deterministic simulations (one
baseline plus one per perturbed state dimension), so ``--gramian-duration`` should stay
small; the default of 1 second already takes several minutes.

Usage
-----
    uv run python scripts/run_channel_selection.py
    uv run python scripts/run_channel_selection.py --duration 5 --gramian-duration 0.3 --n-select 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import numpy.typing as npt

from neuro.connectome import load_connectome
from neuro.jansen_rit import JansenRitParams, output, simulate_network
from utils.channel_selection import (
    compute_channel_gramians,
    leadfield_focus_scores,
    pca_loading_scores,
    qr_pivot_selection,
    reconstruction_error,
    select_channels,
)
from utils.processing import steady_window

_DT = 1e-4
_K = 0.75
_SEED = 42
_EZ_REGIONS = ("lHC", "lPHC", "lAMYG")
_PZ_REGIONS = ("lTCI", "lTCV")
_A_HEALTHY = 3.25
_A_EZ = 3.6
_A_PZ = 3.4
_TRAIN_FRACTION = 0.7

OUTPUT_DIR = Path("simulations/channel_selection")


def _compare_subsets(
    subsets: dict[str, list[int]],
    eeg_steady: npt.NDArray[np.float64],
    focus_scores: npt.NDArray[np.float64],
    n_select: int,
    labels: npt.NDArray[np.str_],
) -> dict[str, dict[str, float]]:
    """Score each candidate subset by held-out EEG reconstruction error and focus coverage.

    ``recon_error`` is the relative error of reconstructing all 62 standardized channels
    from the subset on a held-out time window (lower is better). ``focus_coverage`` is the
    subset's summed leadfield-to-focus score divided by the best achievable for ``n_select``
    channels (1.0 = the most focus-proximal channels possible).
    """
    n_train = int(_TRAIN_FRACTION * eeg_steady.shape[1])
    train, test = eeg_steady[:, :n_train], eeg_steady[:, n_train:]
    focus_best = float(np.sort(focus_scores)[::-1][:n_select].sum())

    metrics = {
        name: {
            "recon_error": reconstruction_error(train, test, sel),
            "focus_coverage": float(focus_scores[sel].sum()) / focus_best,
        }
        for name, sel in subsets.items()
    }

    name_width = max(len(name) for name in subsets)
    print(f"\n{'criterion':<{name_width}}  recon_err  focus_cov  channels")
    for name, sel in subsets.items():
        m = metrics[name]
        chans = ", ".join(labels[i] for i in sel)
        print(f"{name:<{name_width}}  {m['recon_error']:>9.4f}  {m['focus_coverage']:>9.3f}  {chans}")
    return metrics


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the channel-selection runner."""
    parser = argparse.ArgumentParser(description="Collect seizure EEG data and rank channels for model reduction.")
    parser.add_argument("--duration", type=float, default=20.0, help="Seizure simulation duration in seconds.")
    parser.add_argument("--transient-ms", type=float, default=1000.0, help="Leading transient to discard.")
    parser.add_argument(
        "--gramian-duration", type=float, default=1.0, help="Duration of each Gramian sensitivity simulation."
    )
    parser.add_argument("--epsilon", type=float, default=1e-6, help="State perturbation size for the Gramian.")
    parser.add_argument("--n-select", type=int, default=8, help="Number of channels to select per criterion.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for outputs.")
    return parser.parse_args()


def main(  # noqa: PLR0913
    *,
    duration: float = 20.0,
    transient_ms: float = 1000.0,
    gramian_duration: float = 1.0,
    epsilon: float = 1e-6,
    n_select: int = 8,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Run the seizure simulation, save the EEG, and report channel rankings."""
    output_dir.mkdir(parents=True, exist_ok=True)

    connectome = load_connectome(speed=50.0)
    n_nodes = len(connectome.region_labels)

    a_gains = np.full(n_nodes, _A_HEALTHY)
    a_gains[[connectome.region_index[name] for name in _EZ_REGIONS]] = _A_EZ
    a_gains[[connectome.region_index[name] for name in _PZ_REGIONS]] = _A_PZ

    print(f"Running {duration} s seizure simulation (dt={_DT}, K={_K}, seed={_SEED})...")
    t, x_traj = simulate_network(
        params=JansenRitParams(A=a_gains),
        connectome=connectome,
        K=_K,
        duration=duration,
        dt=_DT,
        seed=_SEED,
    )
    eeg = connectome.gain @ output(x_traj)

    eeg_steady = steady_window(eeg, _DT * 1000.0, transient_ms)
    t_steady = steady_window(t, _DT * 1000.0, transient_ms)
    x0 = x_traj[:, :, -1]

    npz_path = output_dir / "seizure_data.npz"
    np.savez(npz_path, t=t_steady, eeg=eeg_steady, x0=x0, channel_labels=connectome.channel_labels)
    print(f"Wrote {npz_path}  (eeg shape={eeg_steady.shape})")

    # 1. PCA loading scores
    pca_scores, pca_explained_variance_ratio = pca_loading_scores(eeg_steady, variance_threshold=0.95)
    pca_ranking = np.argsort(pca_scores)[::-1]

    cum_var = np.cumsum(pca_explained_variance_ratio)
    n_comps = min(int(np.searchsorted(cum_var, 0.95)) + 1, pca_explained_variance_ratio.size)
    print(f"\nPCA: {n_comps} principal components explain >= 95% of variance.")
    print(f"Explained variance ratio of top {min(n_select, len(pca_explained_variance_ratio))} components:")
    print("  " + ", ".join(f"PC{i + 1}: {val:.4f}" for i, val in enumerate(pca_explained_variance_ratio[:n_select])))

    print(f"\nTop {n_select} channels by PCA loading score:")
    for idx in pca_ranking[:n_select]:
        print(f"  {connectome.channel_labels[idx]:>4s}  score={pca_scores[idx]:.4e}")

    # 2. Empirical observability Gramian
    print(f"\nComputing observability Gramians ({6 * n_nodes + 1} simulations of {gramian_duration} s)...")
    gramians = compute_channel_gramians(
        x0,
        JansenRitParams(A=a_gains),
        connectome,
        K=_K,
        dt=_DT,
        duration=gramian_duration,
        epsilon=epsilon,
    )

    det_selection = select_channels(gramians, n_select, criterion="det")
    min_eig_selection = select_channels(gramians, n_select, criterion="min_eig")

    print(f"\nTop {n_select} channels by Gramian determinant criterion (selection order):")
    print("  " + ", ".join(connectome.channel_labels[idx] for idx in det_selection))

    print(f"\nTop {n_select} channels by Gramian minimum-eigenvalue criterion (selection order):")
    print("  " + ", ".join(connectome.channel_labels[idx] for idx in min_eig_selection))

    # 3. Focus-restricted Gramian: observe only the EZ/PZ hidden states (objective-aware,
    #    and small enough that the subset Gramian is full rank so min_eig is meaningful).
    focus_nodes = [connectome.region_index[name] for name in (*_EZ_REGIONS, *_PZ_REGIONS)]
    print(f"\nComputing focus-restricted Gramians ({6 * len(focus_nodes) + 1} simulations of {gramian_duration} s)...")
    gramians_focus = compute_channel_gramians(
        x0,
        JansenRitParams(A=a_gains),
        connectome,
        K=_K,
        dt=_DT,
        duration=gramian_duration,
        epsilon=epsilon,
        node_subset=focus_nodes,
    )
    det_focus = select_channels(gramians_focus, n_select, criterion="det")
    min_eig_focus = select_channels(gramians_focus, n_select, criterion="min_eig")

    # 4. Redundancy-aware (QR / Q-DEIM) and structural focus selectors.
    qr_selection = qr_pivot_selection(eeg_steady, n_select, variance_threshold=0.95)
    focus_scores = leadfield_focus_scores(connectome.gain, focus_nodes)
    focus_selection = np.argsort(focus_scores)[::-1][:n_select].tolist()
    rng_baseline = np.random.default_rng(_SEED)
    random_selection = rng_baseline.choice(eeg_steady.shape[0], n_select, replace=False).tolist()

    # 5. Head-to-head comparison by held-out reconstruction error and focus coverage.
    subsets = {
        "PCA loading": pca_ranking[:n_select].tolist(),
        "QR / Q-DEIM": qr_selection,
        "Leadfield-focus": focus_selection,
        "Gramian det (full)": det_selection,
        "Gramian min_eig (full)": min_eig_selection,
        "Gramian det (focus)": det_focus,
        "Gramian min_eig (focus)": min_eig_focus,
        "Random baseline": random_selection,
    }
    metrics = _compare_subsets(subsets, eeg_steady, focus_scores, n_select, connectome.channel_labels)

    results_path = output_dir / "channel_ranking.npz"
    np.savez(
        results_path,
        pca_scores=pca_scores,
        pca_ranking=pca_ranking,
        pca_explained_variance_ratio=pca_explained_variance_ratio,
        det_selection=np.array(det_selection),
        min_eig_selection=np.array(min_eig_selection),
        det_focus=np.array(det_focus),
        min_eig_focus=np.array(min_eig_focus),
        qr_selection=np.array(qr_selection),
        focus_selection=np.array(focus_selection),
        focus_scores=focus_scores,
        recon_errors=np.array([metrics[name]["recon_error"] for name in subsets]),
        focus_coverage=np.array([metrics[name]["focus_coverage"] for name in subsets]),
        subset_names=np.array(list(subsets)),
        channel_labels=connectome.channel_labels,
    )
    print(f"\nWrote {results_path}")


if __name__ == "__main__":
    cli_args = parse_args()
    main(
        duration=cli_args.duration,
        transient_ms=cli_args.transient_ms,
        gramian_duration=cli_args.gramian_duration,
        epsilon=cli_args.epsilon,
        n_select=cli_args.n_select,
        output_dir=cli_args.output_dir,
    )
