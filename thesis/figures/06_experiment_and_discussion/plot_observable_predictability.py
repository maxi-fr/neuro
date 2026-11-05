from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from observable_predictability.archive import load_evaluation_bundle
from observable_predictability.ensemble import load_manifest, load_scalp_rollout, manifest_identities
from observable_predictability.features import condition_waveform_pair
from observable_predictability.plots import representation_shared_history_s

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import style

if TYPE_CHECKING:
    from observable_predictability.archive import ObservableEvaluationBundle

LABELS = {"waveform": "Waveform", "spectral": "Spectral log-power", "connectivity": "Functional connectivity"}
COLORS = ("#0072B2", "#D55E00", "#009E73")


def plot_example_replays(replay_dir: Path, manifest: dict[str, Any], output: Path) -> None:
    """Illustrate replay divergence at TP9 from one fixed seizure Plant snapshot."""
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
        ax.plot(times, waveform[:, 0], color=COLORS[1], alpha=0.5)
    ax.set(xlabel="Lookahead (s)", ylabel="TP9 EEG (mV)")
    ax.grid()
    fig.savefig(output / "observable_replays.pdf")
    plt.close(fig)


def plot_penalty_scatter(bundle: ObservableEvaluationBundle, output: Path) -> None:
    """Plot every finite matched penalty and Seizure State pair with pooled Pearson coefficients."""
    fig, axes = plt.subplots(1, 3, sharex=True, figsize=style.figsize(height_in=2.05), layout="constrained")
    for ax, (rep, label), color in zip(axes, LABELS.items(), COLORS, strict=True):
        result = bundle.seizure_relevance[rep]
        points = result.finite_pairs()
        pearson = result.pearson(points)
        title = rf"$r={pearson:.3f}$" if np.isfinite(pearson) else "undefined $r$"
        ax.scatter(*points.T, color=color, s=1.8, alpha=0.18, linewidths=0, rasterized=True)
        ax.set(title=f"{label}\n{title}", xlabel="Seizure State", xlim=(0, 1))
        ax.yaxis.set_major_locator(MaxNLocator(4))
    axes[0].set_ylabel("Suppression penalty")
    fig.savefig(output / "observable_penalty_state.pdf", dpi=300)
    plt.close(fig)


def plot_predictability(bundle: ObservableEvaluationBundle, output: Path) -> None:
    """Compare full-state potential predictability on a shared lookahead axis."""
    fig, ax = plt.subplots(figsize=style.figsize(height_in=2.45), layout="constrained")
    for (rep, label), color in zip(LABELS.items(), COLORS, strict=True):
        curve = bundle.potential_predictability[rep]
        ax.plot(curve.times, curve.signal_fraction, label=label, color=color)
        history_s = representation_shared_history_s(bundle, rep)
        if history_s > 0:
            ax.axvline(history_s, color=color, linestyle="--", alpha=0.65)
    ax.axhline(0, color="0.55", linewidth=0.6)
    ax.set(xlabel="Lookahead (s)", ylabel="Signal / total variance")
    ax.legend(ncols=3, loc="lower center", bbox_to_anchor=(0.5, 1))
    fig.savefig(output / "observable_potential_predictability.pdf")
    plt.close(fig)


def plot_waveform_split(split_path: Path, output: Path) -> None:
    """Plot the Waveform split pooled and grouped by starting Seizure State."""
    with np.load(split_path) as data:
        times, ceilings, counts = (data[key] for key in ("times", "ceilings", "counts"))
        threshold = float(data["threshold"])
        support = float(data["support_s"])
    titles = ("All snapshots", rf"$s_0 < {threshold:g}$", rf"$s_0 \geq {threshold:g}$")
    labels = ("Waveform", "Trailing weighted mean", "Waveform minus mean")
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=style.figsize(height_in=2.25), layout="constrained")
    for ax, curves, title, count in zip(axes, ceilings, titles, counts, strict=True):
        for label, curve, color in zip(labels, curves, COLORS, strict=True):
            ax.plot(times, curve, label=label, color=color)
        ax.axhline(0, color="0.55", linewidth=0.6)
        ax.axvline(support, color="0.55", linewidth=0.6, linestyle="--")
        ax.set(title=rf"{title} ($n={count}$)", xlabel="Lookahead (s)")
    axes[0].set_ylabel("Signal / total variance")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncols=3)
    fig.savefig(output / "observable_waveform_split.pdf")
    plt.close(fig)


def main() -> None:
    """Build four native-size thesis PDFs from an evaluation archive and its replay stores."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("archive", type=Path, help="Directory containing evaluation_scores.npz")
    parser.add_argument("replays", type=Path, help="Directory containing the matching replay stores")
    parser.add_argument("--waveform-split", type=Path, help="Override the waveform_split.npz path beside the archive")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    bundle = load_evaluation_bundle(args.archive)
    manifest = load_manifest(args.replays)
    if manifest["identities"] != bundle.manifest["identities"]:
        parser.error("The replay stores and evaluation archive have different rollout identities.")
    args.output.mkdir(parents=True, exist_ok=True)
    plot_example_replays(args.replays, manifest, args.output)
    plot_penalty_scatter(bundle, args.output)
    plot_predictability(bundle, args.output)
    plot_waveform_split(args.waveform_split or args.archive / "waveform_split.npz", args.output)


if __name__ == "__main__":
    main()
