from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import style

FIGURES = Path(__file__).resolve().parent
CAMPAIGN = FIGURES.parents[2] / "artifacts" / "thesis" / "sweep_2026_11"
RESERVED = CAMPAIGN / "sparse_ch05" / "batches" / "activation_reserved" / "runs"


def save_sparse_current_traces() -> None:
    """Plot summed absolute applied currents on reserved seed 7104."""
    traces = (
        (r"Quadratic $L_2$", CAMPAIGN / "observable_benchmark/runs/obs_dmdc_s7104/log.npz"),
        (r"$L_1$", RESERVED / "l1_pure_234_s7104/trace.npz"),
        (r"$L_1$ with binary activation", RESERVED / "bonmin_hybrid_l1_234_a01_s7104/trace.npz"),
    )
    fig, axes = plt.subplots(
        3, 1, figsize=style.figsize(height_in=2.85), sharex=True, sharey=True,
        layout="constrained",
    )
    for ax, (label, path) in zip(axes, traces, strict=True):
        with np.load(path) as trace:
            times = trace["controller.t"]
            currents = np.abs(trace["controller.u"]).sum(axis=1)
        ax.step(times, currents, where="post", color="C0")
        ax.axvspan(0, 1.38, color="0.92")
        ax.set_title(label, loc="left")
        ax.set(ylim=(-0.2, 4.25), yticks=[0, 2, 4])
        ax.grid(axis="y")
    axes[-1].set(xlim=(0, 12), xticks=np.arange(0, 13, 2), xlabel="Time (s)")
    fig.supylabel("Summed absolute current (mA)", fontsize=style.FONT_LABEL)
    fig.savefig(FIGURES / "sparse_current_traces.pdf")
    plt.close(fig)


if __name__ == "__main__":
    save_sparse_current_traces()
