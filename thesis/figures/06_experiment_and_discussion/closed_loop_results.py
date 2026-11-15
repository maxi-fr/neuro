from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import style

FIGURES = Path(__file__).resolve().parent
CAMPAIGN = FIGURES.parents[2] / "artifacts" / "sweep_2026_11"


def save_effort_sweeps() -> None:
    """Plot charge and seizure burden for every seed and the mean of each MLP sweep.""" # we dont look at just the MLP sweeps. rather spectral DMD and CNN and waveform MLP
    fig, panels = plt.subplots(2, 2, figsize=style.figsize(height_in=3.8), layout="constrained")
    for column, (label, source) in enumerate((
        ("Spectral MLP", CAMPAIGN / "observable_effort_calibration/mlp/rows.csv"),
        ("Waveform MLP", CAMPAIGN / "waveform_matched_geometry/mlp/final_effort/rows.csv"),
    )):
        with source.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        weights = np.array([float(row["arm"].removeprefix("wu_").replace("_", ".")) for row in rows])
        unique_weights = np.unique(weights)
        panels[0, column].set_title(label)
        for ax, (metric, ylabel, scale) in zip(panels[:, column], (
            ("delivered_charge", "Delivered charge (mAs)", 1),
            ("seizure_burden", r"Seizure burden (\%)", 100),
        ), strict=True):
            values = np.array([float(row[metric]) * scale for row in rows])
            ax.scatter(weights, values, color="C0", alpha=0.45, s=9)
            ax.plot(unique_weights, [values[weights == weight].mean() for weight in unique_weights], color="C0")
            ax.set(xscale="log", ylabel=ylabel)
            ax.grid()
        panels[1, column].set_xlabel(r"Quadratic current weight $w_u$")
    fig.savefig(FIGURES / "closed_loop_effort_sweeps.pdf")
    plt.close(fig)


if __name__ == "__main__":
    save_effort_sweeps()
