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
from experiments.observable_predictability.plots import representation_shared_history_s

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import style

if TYPE_CHECKING:
    from experiments.observable_predictability.archive import ObservableEvaluationBundle

LABELS = {"waveform": "Waveform", "spectral": "Spectral log-power", "connectivity": "Functional connectivity"}
COLORS = ("#0072B2", "#D55E00", "#009E73")
ARTIFACTS = Path(__file__).resolve().parents[3] / "artifacts"


def plot_example_replays(replay_dir: Path, manifest: dict[str, Any], output: Path) -> None:
    """Illustrate replay divergence at TP9 from one fixed seizure plant snapshot."""
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
        ax.plot(times, waveform[:, 0], color=COLORS[0], alpha=0.5)
    ax.set(xlabel="Lookahead (s)", ylabel="TP9 EEG")
    ax.grid()
    fig.savefig(output / "observable_replays.pdf")
    plt.close(fig)


def plot_penalty_scatter(bundle: ObservableEvaluationBundle, output: Path) -> None:
    """Plot frame-level costs against seizure state with pooled Pearson coefficients."""
    fig, axes = plt.subplots(1, 3, sharex=True, figsize=style.figsize(height_in=2.25), layout="constrained")
    cost_labels = (
        r"Cost $\mathcal J_{\mathrm{wave}}$",
        r"Cost $\mathcal J_{\mathrm{spec}}$",
        r"Cost $\mathcal J_{\mathrm{conn}}$",
    )
    for ax, (rep, label), color, cost_label in zip(axes, LABELS.items(), COLORS, cost_labels, strict=True):
        result = bundle.seizure_relevance[rep]
        points = result.finite_pairs()
        pearson = result.pearson(points)
        title = rf"$r={pearson:.3f}$" if np.isfinite(pearson) else "undefined $r$"
        ax.scatter(*points.T, color=color, s=1.8, alpha=0.18, linewidths=0, rasterized=True)
        ax.set(title=f"{label}\n{title}", xlabel=r"Seizure state $s$", ylabel=cost_label)
        ax.grid()
        ax.yaxis.set_major_locator(MaxNLocator(4))
    fig.savefig(output / "observable_penalty_state.pdf", dpi=300)
    plt.close(fig)


def plot_predictability(bundle: ObservableEvaluationBundle, split_path: Path, output: Path) -> None:
    """Compare EEG representations and waveform components across all snapshots."""
    with np.load(split_path) as data:
        times = data["times"]
        waveform, mean, residual = data["ceilings"][0, :3]

    fig, (left, right) = plt.subplots(
        1, 2, sharex=True, sharey=True, figsize=style.figsize(height_in=2.6), layout="constrained"
    )
    for (rep, label), color in zip(LABELS.items(), COLORS, strict=True):
        curve = bundle.potential_predictability[rep]
        left.plot(curve.times, curve.signal_fraction, label=label, color=color)

    right.plot(times, waveform, label="Waveform", color=COLORS[0])
    right.plot(times, mean, label="Trailing weighted mean", color="#CC79A7")
    right.plot(times, residual, label="Waveform minus mean", color="#009E73")

    for ax in (left, right):
        ax.axhline(0, color="0.55", linewidth=0.6)
        ax.grid()
        ax.set_xlabel("Lookahead (s)")
        ax.set_xlim(None, 2.0)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1))
    left.set_ylabel("Signal / total variance")
    fig.savefig(output / "observable_potential_predictability.pdf")
    plt.close(fig)


def main() -> None:
    """Build three native-size thesis PDFs from the default expanded evaluation bundle."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=ARTIFACTS / "observable_predictability_expanded",
        help="Directory containing evaluation_scores.npz",
    )
    parser.add_argument("--replays", type=Path, help="Override the replay directory, defaulting to ARCHIVE/replays")
    parser.add_argument("--waveform-split", type=Path, help="Override the waveform_split.npz path beside the archive")
    args = parser.parse_args()
    replay_dir = args.replays or args.archive / "replays"
    bundle = load_evaluation_bundle(args.archive)
    manifest = load_manifest(replay_dir)
    if manifest["identities"] != bundle.manifest["identities"]:
        parser.error("The replay stores and evaluation archive have different rollout identities.")

    output_dir = Path(__file__).resolve().parent

    plot_example_replays(replay_dir, manifest, output_dir)
    plot_penalty_scatter(bundle, output_dir)
    plot_predictability(bundle, args.waveform_split or args.archive / "waveform_split.npz", output_dir)


if __name__ == "__main__":
    main()
