from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import style

FIGURES = Path(__file__).resolve().parent


def save_reserved_current_traces() -> None:
    """Compare the summed absolute electrode currents for four penalties on seed 7104."""
    traces = np.genfromtxt(FIGURES / "reserved_current_traces_7104.csv", delimiter=",", names=True)
    times = traces["time_s"]
    first_replan = times[np.flatnonzero(traces["control_started"])[0]]
    arms = (
        ("Quadratic", "quadratic_mA"),
        (r"$L_1$", "l1_mA"),
        ("Activation", "activation_mA"),
        (r"$L_1$ + activation", "combined_mA"),
    )
    fig, axes = plt.subplots(
        4, 1, figsize=style.figsize(height_in=3.65), sharex=True, sharey=True, layout="constrained"
    )
    for ax, (label, column) in zip(axes, arms, strict=True):
        ax.step(times, traces[column], where="post", color="C0")
        ax.axvspan(0, first_replan, color="0.92")
        ax.set_title(label, loc="left")
        ax.set(ylim=(-0.2, 4.25), yticks=[0, 2, 4])
        ax.grid(axis="y")
    axes[-1].set(xlim=(0, 12), xticks=np.arange(0, 13, 2), xlabel="Time (s)")
    fig.supylabel("Summed absolute current (mA)", fontsize=style.FONT_LABEL)
    fig.savefig(FIGURES / "reserved_current_traces_7104.pdf")
    plt.close(fig)


if __name__ == "__main__":
    save_reserved_current_traces()
