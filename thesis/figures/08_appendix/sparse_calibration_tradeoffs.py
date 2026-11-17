from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import style

FIGURES = Path(__file__).resolve().parent
RESULTS = FIGURES.parents[2] / "artifacts" / "sweep_2026_11" / "sparse_ch05"
METHODS = {
    "l1": (r"Exact $L_1$", "#264653"),
    "bonmin": ("Bonmin", "#d75c37"),
    "candidate": ("Grouped candidates", "#2a8f77"),
    "homotopy": ("Homotopy", "#74539a"),
}
PANELS = (
    ("delivered_charge", "Delivered charge (mAs)"),
    ("intentional_off_s", "Off time (s)"),
)


def save_sparse_calibration_tradeoffs() -> None:
    """Plot screening and confirmation results at the appendix text width."""
    fig, axes = plt.subplots(1, 2, figsize=style.figsize(height_in=2.8), sharey=True, layout="constrained")
    for table, marker, filled in (("screen", "o", False), ("five_seeds", "D", True)):
        with (RESULTS / f"{table}.csv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                _, color = METHODS[row["arm"].split("_", 1)[0]]
                for ax, (metric, _) in zip(axes, PANELS, strict=True):
                    ax.scatter(
                        float(row[metric]),
                        100 * float(row["seizure_burden"]),
                        marker=marker,
                        s=19 if filled else 12,
                        linewidths=0.65,
                        edgecolors=color,
                        facecolors=color if filled else "none",
                        zorder=3 if filled else 2,
                    )

    for ax, (_, label) in zip(axes, PANELS, strict=True):
        ax.axhline(6, color="0.35", linestyle="--")
        ax.set_xlabel(label)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.spines[["right", "top"]].set_visible(False)
    axes[0].set_ylabel(r"Mean seizure burden (\%)")
    handles = [
        Line2D([], [], marker="o", color=color, linestyle="none", markersize=4, label=label)
        for label, color in METHODS.values()
    ]
    handles.extend(
        [
            Line2D(
                [],
                [],
                marker="o",
                markerfacecolor="none",
                color="0.35",
                linestyle="none",
                markersize=4,
                label="Two-seed screen",
            ),
            Line2D([], [], marker="D", color="0.35", linestyle="none", markersize=4, label="Five-seed confirmation"),
        ]
    )
    fig.legend(handles=handles, loc="outside upper center", ncols=3, frameon=False)
    fig.savefig(FIGURES / "sparse_calibration_tradeoffs.pdf")
    plt.close(fig)


if __name__ == "__main__":
    save_sparse_calibration_tradeoffs()
