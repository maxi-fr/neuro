from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from experiments.observable_predictability.archive import load_evaluation_bundle
from experiments.observable_predictability.ensemble import load_manifest, load_scalp_rollout, manifest_identities
from experiments.observable_predictability.features import condition_waveform_pair
from experiments.observable_predictability.scoring import SeizureRelevanceResult

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import style

if TYPE_CHECKING:
    from experiments.observable_predictability.archive import ObservableEvaluationBundle
    from neuro.types import FloatArray

LABELS = {"waveform": "Waveform", "spectral": "Spectral log-power", "connectivity": "Functional connectivity"}
COLORS = ("C0", "C1", "C2")
LOOKAHEAD_LABEL = r"Lookahead $h$ (s)"
ARTIFACTS = Path(__file__).resolve().parents[3] / "artifacts"


def plot_example_replays(replay_dir: Path, manifest: dict[str, Any], output: Path) -> None:
    """Plot the TP9 waveform across branches from one seizure plant snapshot."""
    channel = manifest["channel_labels"].index("TP9")
    fig, ax = plt.subplots(figsize=style.figsize(height_in=2.0), layout="constrained")
    identities = manifest_identities(manifest, split="eval", plant="seizure", branch="saturated", arm="zero")
    seed = identities[0].parent_seed
    replays = [identity for identity in identities if identity.parent_seed == seed][:8]
    for identity in replays:
        history, future = load_scalp_rollout(replay_dir, identity, manifest)
        _, waveform = condition_waveform_pair(
            future[channel, :, None], manifest["fs"], history_raw=history[channel, :, None]
        )
        times = (np.arange(len(waveform)) + 1) / 50
        ax.plot(times, waveform[:, 0], alpha=0.5)
    ax.set(xlabel=LOOKAHEAD_LABEL, ylabel=r"TP9 waveform $y_{r,\mathrm{TP9}}$")
    ax.grid()
    fig.savefig(output / "observable_replays.pdf")
    plt.close(fig)


def plot_penalty_scatter(bundle: ObservableEvaluationBundle, lag_points: FloatArray, output: Path) -> None:
    """Plot frame-level costs against seizure state with pooled Pearson correlations."""
    fig, axes = plt.subplots(1, 3, sharex=True, figsize=style.figsize(height_in=2.25), layout="constrained")
    cost_labels = (
        r"Cost $\mathcal J_{\mathrm{wave}}$",
        r"Cost $\mathcal J_{\mathrm{spec}}$",
        r"Cost $\mathcal J_{\mathrm{conn}}(q)$",
    )
    for ax, (rep, label), color, cost_label in zip(axes, LABELS.items(), COLORS, cost_labels, strict=True):
        points = lag_points if rep == "connectivity" else bundle.seizure_relevance[rep].finite_pairs()
        pearson = SeizureRelevanceResult.pearson(points)
        title = rf"$r={pearson:.3f}$" if np.isfinite(pearson) else "undefined $r$"
        ax.scatter(*points.T, color=color, s=1.8, alpha=0.18, linewidths=0, rasterized=True)
        ax.set(title=f"{label}\n{title}", xlabel=r"Seizure state $s(t_q)$", ylabel=cost_label)
        ax.grid()
        ax.yaxis.set_major_locator(MaxNLocator(4))
    fig.savefig(output / "observable_penalty_state.pdf", dpi=300)
    plt.close(fig)


def plot_predictability( # noqa: PLR0913, PLR0917
    bundle: ObservableEvaluationBundle,
    split_path: Path,
    lag_times: FloatArray,
    lag_curve: FloatArray,
    lag_window_samples: int,
    output: Path,
) -> None:
    """Plot potential predictability for EEG representations and waveform components."""
    with np.load(split_path) as data:
        times = data["times"]
        waveform, mean, residual = data["ceilings"][0, :3]

    fig, (left, right) = plt.subplots(
        1, 2, sharex=True, sharey=True, figsize=style.figsize(height_in=2.6), layout="constrained"
    )
    for (rep, label), color in zip(LABELS.items(), COLORS, strict=True):
        if rep == "connectivity":
            left.plot(lag_times, lag_curve, label=label, color=color)
        else:
            curve = bundle.potential_predictability[rep]
            left.plot(curve.times, curve.signal_fraction, label=label, color=color)
    window_s = lag_window_samples / bundle.metadata.waveform_geometry["fs"]
    first_post_branch = lag_times[np.searchsorted(lag_times, window_s)]
    left.axvline(first_post_branch, color="0.55", linewidth=0.6, linestyle="--")

    right.plot(times, waveform, label="Waveform", color=COLORS[0])
    right.plot(times, mean, label="Trailing weighted mean", color="C3")
    right.plot(times, residual, label="Waveform minus mean", color="C4")

    for ax in (left, right):
        ax.axhline(0, color="0.55", linewidth=0.6)
        ax.grid()
        ax.set_xlabel(LOOKAHEAD_LABEL)
        ax.set_xlim(None, 2.0)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1))
    left.set_ylabel(r"Potential predictability $\widehat P(h)$")
    fig.savefig(output / "observable_potential_predictability.pdf")
    plt.close(fig)


def main() -> None:
    """Build three native-size thesis PDFs from the expanded replays and lag-asymmetry scores."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=ARTIFACTS / "observable_predictability_expanded",
        help="Directory containing evaluation_scores.npz",
    )
    parser.add_argument("--replays", type=Path, help="Override the replay directory, defaulting to ARCHIVE/replays")
    parser.add_argument("--waveform-split", type=Path, help="Override the waveform_split.npz path beside the archive")
    parser.add_argument(
        "--lag-scores",
        type=Path,
        default=ARTIFACTS / "fc_time_domain" / "window_25" / "curves.npz",
        help="Lag scores generated by artifacts/fc_time_domain/compare.py --short-window on the same replays",
    )
    args = parser.parse_args()
    replay_dir = args.replays or args.archive / "replays"
    bundle = load_evaluation_bundle(args.archive)
    manifest = load_manifest(replay_dir)
    if manifest["identities"] != bundle.manifest["identities"]:
        parser.error("The replay stores and evaluation archive have different rollout identities.")

    output_dir = Path(__file__).resolve().parent

    with np.load(args.lag_scores) as data:
        lag_times = data["times"]
        lag_curve = data["lag_asymmetry"]
        lag_points = data["lag_mean_points"]
        lag_window_samples = data["window_samples"].item()
    np.testing.assert_array_equal(lag_points[:, 0], bundle.seizure_relevance["waveform"].finite_pairs()[:, 0])
    plot_example_replays(replay_dir, manifest, output_dir)
    plot_penalty_scatter(bundle, lag_points, output_dir)
    plot_predictability(
        bundle,
        args.waveform_split or args.archive / "waveform_split.npz",
        lag_times,
        lag_curve,
        lag_window_samples,
        output_dir,
    )


if __name__ == "__main__":
    main()
