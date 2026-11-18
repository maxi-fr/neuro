from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import style

FIGURES = Path(__file__).resolve().parent
RESULTS = FIGURES.parents[2] / "artifacts" / "sparse_stimulation_102"
METHODS = {
    "l1": (r"Exact $L_1$", "#264653"),
    "bonmin": ("Bonmin", "#d75c37"),
    "candidate": ("Grouped candidates", "#2a8f77"),
    "homotopy": ("Homotopy", "#74539a"),
    "combined": (r"Bonmin + $L_1$", "#d29a2c"),
}
PANELS = (
    ("delivered_charge", "Delivered charge (mAs)"),
    ("intentional_off_s", "Off time (s)"),
)


def save_sparse_calibration_tradeoffs() -> None:
    """Plot the 1.02 s calibration screens and five-seed confirmations."""
    fig, axes = plt.subplots(1, 2, figsize=style.figsize(height_in=2.8), sharey=True, layout="constrained")
    sources = {
        "appendix_continuous": {"appendix_l1_osqp"},
        "l1_weight_screen": {"l1_040", "l1_050", "l1_065"},
        "l1_weight_confirm": {"l1_065"},
        "l1_7004": {"l1_058", "l1_060", "l1_062"},
        "l1_expanded": {"l1_058", "l1_060", "l1_062"},
        "activation_screen": {"bonmin_180", "candidate_070", "candidate_180", "candidate_240", "homotopy_180"},
        "activation_240": {"bonmin_240", "homotopy_240"},
        "activation_confirm": {"bonmin_240", "candidate_240", "homotopy_240"},
        "activation_expanded": {"bonmin_300", "candidate_300", "homotopy_300"},
        "combined_screen": {"combined_001", "combined_004", "combined_008"},
        "combined_confirm": {"combined_001"},
        "combined_008_confirm": {"combined_008"},
    }
    groups = defaultdict(dict)
    for batch, arms in sources.items():
        with (RESULTS / batch / "sparse_rows.csv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                if row["arm"] in arms and not row["error"]:
                    groups[row["arm"]][int(row["seed"])] = row

    for arm, seeds in groups.items():
        for subset, marker, filled in (({7000, 7001}, "o", False), ({7000, 7001, 7002, 7004, 7005}, "D", True)):
            if not subset.issubset(seeds):
                continue
            rows = [seeds[seed] for seed in sorted(subset)]
            family = "l1" if arm.startswith(("l1_", "appendix_l1_")) else arm.split("_", 1)[0]
            _, color = METHODS[family]
            burden = 100 * sum(float(row["seizure_burden"]) for row in rows) / len(rows)
            for ax, (metric, _) in zip(axes, PANELS, strict=True):
                value = sum(float(row[metric]) for row in rows) / len(rows)
                ax.scatter(
                    value,
                    burden,
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
