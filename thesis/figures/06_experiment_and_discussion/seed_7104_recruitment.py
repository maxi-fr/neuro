import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import style

FIGURES = Path(__file__).resolve().parent
BRANCHES = FIGURES.parents[2] / "artifacts/thesis/seed_7104_snapshot_branches/branches.csv"


def plot_seed_7104() -> None:
    """Show recruitment across 100 paired branches and under the original noise at four snapshots."""
    data = np.genfromtxt(BRANCHES, names=True, delimiter=",", dtype=None, encoding="utf-8")
    fig, axes = plt.subplots(2, 4, sharex=True, sharey=True, layout="constrained",
                             figsize=style.figsize(height_in=2.9))
    for column, time in enumerate((1.38, 1.56, 1.62, 1.68)):
        axes[0, column].set_title(f"Snapshot {time:.2f} s")
        for row, (plan, label, color) in enumerate((
            ("waveform_MLP", "Waveform MLP", "C1"), ("spectral_CNN", "Spectral CNN", "C0")
        )):
            ax = axes[row, column]
            subset = data[(data["snapshot_s"] == time) & (data["plan"] == plan)]
            alternative = subset[subset["draw_0_recorded"] > 0]["endpoint_regions"]
            recorded = subset[subset["draw_0_recorded"] == 0]["endpoint_regions"].item()
            assert len(alternative) == 100
            counts, frequencies = np.unique(alternative, return_counts=True)
            ax.bar(counts, frequencies, width=0.85, color=color)
            ax.plot(recorded, -2.6, "k^", markersize=4, clip_on=False)
            ax.axhline(0, color="0.3", linewidth=0.5)
            ax.set(xlim=(0, 37), ylim=(-4.5, 55), xticks=[0, 10, 20, 30], yticks=[0, 10, 20, 30, 40, 50])
            ax.set_axisbelow(True)
            ax.grid(axis="y")
            if column == 0:
                ax.set_ylabel(f"{label}\nBranches")
    fig.supxlabel("Seizing regions at the end of the plan", fontsize=style.FONT_LABEL)
    fig.savefig(FIGURES / "seed_7104_recruitment.pdf")
    plt.close(fig)


if __name__ == "__main__":
    plot_seed_7104()
