"""Generate excitability_regimes.pdf: the Jansen-Rit excitability regimes."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from neuro.connectome import Connectome
from neuro.jansen_rit import JansenRitDynamics, JansenRitParams, lfp, simulate_network
from neuro.types import FloatArray  # noqa: TC001 -- runtime import keeps the script runnable standalone

DT = 1e-4
DURATION = 8.0
TRANSIENT_S = 4.0
SEED = 7
K = 0.60
SIGMA = 280.0
SPEED_MM_PER_MS = 50.0
PSD_MAX_HZ = 60.0
A_BACKGROUND, A_PZ, A_EZ = 3.25, 3.4, 3.6
EZ_NODES = (40, 47, 62)
PZ_NODES = (69, 72)
PROBE_NODE = 69
OUT = Path(__file__).with_suffix(".pdf")


def _isolated_connectome() -> Connectome:
    """One-node connectome carrying no coupling and no delay."""
    return Connectome(
        K=K,
        weights=np.zeros((1, 1)),
        tract_lengths=np.zeros((1, 1)),
        centres=np.zeros((1, 3)),
        region_labels=np.array(["n0"]),
        hemispheres=np.zeros(1, dtype=bool),
        speed=SPEED_MM_PER_MS,
        delays=np.zeros((1, 1)),
        region_index={"n0": 0},
    )


def _isolated(a_gain: float) -> tuple[FloatArray, FloatArray]:
    """Post-transient time vector and lfp of one uncoupled node at gain ``a_gain``."""
    dyn = JansenRitDynamics(
        dt=DT, params=JansenRitParams(A=a_gain, sigma=SIGMA), conn=_isolated_connectome(), seed=SEED
    )
    t, x = simulate_network(dyn=dyn, duration=DURATION)
    keep = t >= TRANSIENT_S
    return t[keep] - TRANSIENT_S, lfp(x)[0, keep]


def _in_network(node: int) -> tuple[FloatArray, FloatArray]:
    """Post-transient time vector and lfp of ``node`` in the full connectome network."""
    conn = Connectome.from_config({"speed": SPEED_MM_PER_MS, "K": K})
    gains = np.full(len(conn.region_labels), A_BACKGROUND)
    gains[list(EZ_NODES)] = A_EZ
    gains[list(PZ_NODES)] = A_PZ
    dyn = JansenRitDynamics(dt=DT, params=JansenRitParams(A=gains, sigma=SIGMA), conn=conn, seed=SEED)
    t, x = simulate_network(dyn=dyn, duration=DURATION)
    keep = t >= TRANSIENT_S
    return t[keep] - TRANSIENT_S, lfp(x)[node, keep]


def main() -> None:
    """Write the four-row time-series and power-spectrum figure."""
    panels = [
        (_isolated(A_BACKGROUND), "(a) Background, $A_i = 3.25$ mV, uncoupled"),
        (_isolated(A_PZ), "(b) Propagation zone, $A_i = 3.4$ mV, uncoupled"),
        (_in_network(PROBE_NODE), "(c) Propagation zone, $A_i = 3.4$ mV, in the network"),
        (_isolated(A_EZ), "(d) Epileptogenic zone, $A_i = 3.6$ mV, uncoupled"),
    ]

    fig, axes = plt.subplots(len(panels), 2, figsize=(7.0, 7.0), constrained_layout=True)
    for row, ((t, y), label) in enumerate(panels):
        axes[row, 0].plot(t, y, lw=0.7, color="C0")
        axes[row, 0].set_ylabel("$y_\\mathrm{LFP}$ / mV")
        axes[row, 0].set_title(label, fontsize=8, loc="left")
        axes[row, 0].set_xlim(0.0, 4.0)

        f, pxx = welch(y, fs=1.0 / DT, nperseg=2**14)
        band = f <= PSD_MAX_HZ
        axes[row, 1].semilogy(f[band], pxx[band], lw=0.8, color="C3")
        axes[row, 1].set_ylabel("PSD / mV$^2$ Hz$^{-1}$")
        axes[row, 1].set_xlim(0.0, PSD_MAX_HZ)

        for ax in axes[row]:
            ax.grid(visible=True, lw=0.4, alpha=0.4)
            ax.set_axisbelow(True)

    axes[-1, 0].set_xlabel("Time / s")
    axes[-1, 1].set_xlabel("Frequency / Hz")
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
