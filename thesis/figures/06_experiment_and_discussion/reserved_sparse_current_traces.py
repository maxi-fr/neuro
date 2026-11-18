from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import style

EXPERIMENT = Path(__file__).resolve().parents[3] / "artifacts" / "sparse_stimulation_reserved"
SEED = 7104


def save_reserved_current_traces() -> None:
    """Plot the three applied-current traces on reserved seed 7104."""
    runs = EXPERIMENT / "reserved" / "runs"
    arms = (
        (r"Quadratic $L_2$", "quadratic_osqp"),
        (r"Exact $L_1$", "l1_osqp"),
        (r"$L_1$ with grouped activation", "combined_candidates"),
    )
    fig, axes = plt.subplots(
        3, 1, figsize=style.figsize(height_in=2.85), sharex=True, sharey=True, layout="constrained"
    )
    for ax, (label, arm) in zip(axes, arms, strict=True):
        with np.load(runs / f"{arm}_s{SEED}" / "log.npz", allow_pickle=False) as trace:
            times = trace["controller.t"]
            currents = np.abs(trace["controller.u"]).sum(axis=1)
            first_replan = times[np.flatnonzero(trace["controller.solve_attempted"])[0]]
        ax.step(times, currents, where="post", color="C0")
        ax.axvspan(0, first_replan, color="0.92")
        ax.set_title(label, loc="left")
        ax.set(ylim=(-0.2, 4.25), yticks=[0, 2, 4])
        ax.grid(axis="y")
    axes[-1].set(xlim=(0, 12), xticks=np.arange(0, 13, 2), xlabel="Time (s)")
    fig.supylabel("Summed absolute current (mA)", fontsize=style.FONT_LABEL)
    fig.savefig(EXPERIMENT / "reserved_current_traces_7104.pdf")
    plt.close(fig)


if __name__ == "__main__":
    save_reserved_current_traces()
