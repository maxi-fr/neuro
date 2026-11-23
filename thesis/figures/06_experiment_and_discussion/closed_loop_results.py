from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import style

FIGURES = Path(__file__).resolve().parent
CAMPAIGN = FIGURES.parents[2] / "artifacts" / "thesis" / "sweep_2026_11"
EXTENSION = CAMPAIGN / "stimulation_cost_weight_extension"
CHARGE_TARGET = 30


def save_effort_sweeps() -> None:
    """Plot the extended development-seed sweeps and their first mean charge target crossings."""
    fig, panels = plt.subplots(
        2, 2, figsize=style.figsize(height_in=3.8),
        sharex="col", sharey="row", layout="constrained",
    )
    metrics = (
        ("delivered_charge", "Delivered charge (mAs)", 1, CHARGE_TARGET),
        ("seizure_burden", r"Seizure burden (\%)", 100, 6),
    )
    for axes, (_, ylabel, _, target) in zip(panels, metrics, strict=True):
        axes[0].set_ylabel(ylabel)
        for ax in axes:
            ax.axhspan(0, target, color="0.92", zorder=0)
            ax.axhline(target, color="0.55", linewidth=0.6, linestyle="--")
            ax.set_axisbelow(True)
            ax.grid(axis="y")

    sweeps = (
        (0, "DMDc", "C0", (
            CAMPAIGN / "observable_dmd/calibration/hop3/rows.csv",
            EXTENSION / "spectral_dmdc/rows.csv",
            EXTENSION / "spectral_dmdc_dense/rows.csv",
        )),
        (0, "CNN", "C1", (
            CAMPAIGN / "observable_effort_calibration/cnn/rows.csv",
            EXTENSION / "spectral_cnn/rows.csv",
        )),
        (1, "MLP", "C0", (
            CAMPAIGN / "waveform_matched_geometry/mlp/final_effort/rows.csv",
            EXTENSION / "waveform_mlp/rows.csv",
        )),
    )
    for column, label, color, sources in sweeps:
        rows = []
        for source in sources:
            with source.open(newline="", encoding="utf-8") as stream:
                rows.extend(row for row in csv.DictReader(stream) if "wu_" in row["arm"])
        weights = np.array([float(row["arm"].split("wu_")[1].replace("_", ".")) for row in rows])
        unique_weights = np.unique(weights)
        charge_ax, burden_ax = panels[:, column]
        for ax, (metric, _, scale, _) in zip((charge_ax, burden_ax), metrics, strict=True):
            values = np.array([float(row[metric]) * scale for row in rows])
            means = [values[weights == weight].mean() for weight in unique_weights]
            ax.scatter(weights, values, color=color, alpha=0.45, s=9, linewidths=0)
            ax.plot(unique_weights, means, color=color, label=label)
            if metric == "delivered_charge":
                below = next(index for index, mean in enumerate(means) if mean <= CHARGE_TARGET)
                bracket = slice(below - 1, below + 1)
                crossing = np.exp(np.interp(
                    CHARGE_TARGET, means[bracket][::-1], np.log(unique_weights[bracket][::-1]),
                ))

        guide = ConnectionPatch(
            (crossing, 1), (crossing, 0),
            coordsA=charge_ax.get_xaxis_transform(), coordsB=burden_ax.get_xaxis_transform(),
            color=color, linewidth=0.7, linestyle=":", alpha=0.8, clip_on=False,
        )
        guide.set_in_layout(False)
        fig.add_artist(guide)
        charge_ax.scatter(crossing, CHARGE_TARGET, color=color, s=13, zorder=4)
        offset, alignment = (-4, "right") if label == "CNN" else (4, "left")
        charge_ax.annotate(
            f"{crossing:.2g}", (crossing, 1), xycoords=charge_ax.get_xaxis_transform(),
            xytext=(offset, -5), textcoords="offset points", ha=alignment, va="top",
            color=color, fontsize=style.FONT_ANNOTATION,
        )

    panels[0, 0].set_title("Spectral predictors")
    panels[0, 1].set_title("Waveform MLP")
    panels[0, 0].legend(loc="upper right")
    panels[0, 0].set(ylim=(0, 50), yticks=np.arange(0, 51, 10))
    panels[1, 0].set(ylim=(0, 39), yticks=np.arange(0, 37, 6))
    for ax in panels[1]:
        ax.set(xscale="log", xlabel=r"Stimulation cost weight $w_{L_2}$")

    fig.savefig(FIGURES / "closed_loop_effort_sweeps.pdf")
    plt.close(fig)


if __name__ == "__main__":
    save_effort_sweeps()
