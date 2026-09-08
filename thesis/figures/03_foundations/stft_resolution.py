"""Generate stft_resolution.pdf: what segment length buys and what averaging buys."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import decimate
from scipy.signal.windows import hann

from neuro.connectome import Connectome
from neuro.jansen_rit import JansenRitDynamics, JansenRitParams, lfp, simulate_network
from neuro.types import FloatArray  # noqa: TC001 -- runtime import keeps the script runnable standalone

DT = 1e-4
DURATION = 44.0
TRANSIENT_S = 4.0
SEED = 7
SIGMA = 280.0
A_EZ = 3.6
DECIMATION = (20, 10)
FS = 1.0 / (DT * DECIMATION[0] * DECIMATION[1])
SEGMENT_LENGTHS = (25, 50, 200)
KERNEL_WIDTHS = (1, 4, 16)
KERNEL_SEGMENT = 200
KERNEL_HOP = 100
BAND_MAX_HZ = 12.0
PANEL_A_KERNEL = 8
OUT = Path(__file__).with_suffix(".pdf")


def _ez_trace() -> FloatArray:
    """Decimated local field potential of one uncoupled epileptogenic-zone node."""
    conn = Connectome(
        K=0.0,
        weights=np.zeros((1, 1)),
        tract_lengths=np.zeros((1, 1)),
        centres=np.zeros((1, 3)),
        region_labels=np.array(["n0"]),
        hemispheres=np.zeros(1, dtype=bool),
        speed=50.0,
        delays=np.zeros((1, 1)),
        region_index={"n0": 0},
    )
    dyn = JansenRitDynamics(dt=DT, params=JansenRitParams(A=A_EZ, sigma=SIGMA), conn=conn, seed=SEED)
    t, x = simulate_network(dyn=dyn, duration=DURATION)
    y = lfp(x)[0, t >= TRANSIENT_S]
    return decimate(decimate(y, DECIMATION[0], ftype="fir"), DECIMATION[1], ftype="fir")


def _periodograms(y: FloatArray, n_segment: int, n_hop: int) -> tuple[FloatArray, FloatArray]:
    """One-sided density-scaled Hann periodograms of every segment, with the frequency axis."""
    n_frames = (y.size - n_segment) // n_hop + 1
    taper = hann(n_segment, sym=False)
    segments = np.stack([y[q * n_hop : q * n_hop + n_segment] for q in range(n_frames)])
    power = np.abs(np.fft.rfft(segments * taper[None, :], axis=1)) ** 2
    fold = np.full(n_segment // 2 + 1, 2.0)
    fold[0] = 1.0
    if n_segment % 2 == 0:
        fold[-1] = 1.0
    power = power * fold[None, :] / (FS * np.sum(taper**2))
    return np.fft.rfftfreq(n_segment, d=1.0 / FS), power


def main() -> None:
    """Write the two-panel resolution and variance figure."""
    y = _ez_trace()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)

    ax = axes[0]
    for idx, n_segment in enumerate(SEGMENT_LENGTHS):
        freqs, power = _periodograms(y, n_segment, n_segment // 2)
        label = rf"$L_\mathrm{{s}} = {n_segment}$, $\Delta f = {FS / n_segment:.2f}$ Hz"
        ax.semilogy(
            freqs,
            power[:PANEL_A_KERNEL].mean(axis=0),
            lw=0.9,
            marker="o",
            ms=2.5,
            color=f"C{idx}",
            label=label,
        )
    ax.set_title("(a) Three segment lengths, matched averaging", fontsize=8, loc="left")

    ax = axes[1]
    freqs, power = _periodograms(y, KERNEL_SEGMENT, KERNEL_HOP)
    end = max(KERNEL_WIDTHS)
    for idx, width in enumerate(KERNEL_WIDTHS):
        ax.semilogy(
            freqs, power[end - width : end].mean(axis=0), lw=0.9, color=f"C{idx}", label=rf"$L_\mathrm{{k}} = {width}$"
        )
    ax.set_title("(b) One segment length, three kernel widths", fontsize=8, loc="left")

    for ax in axes:
        ax.set_xlim(0.0, BAND_MAX_HZ)
        ax.set_ylim(1e-3, 2e1)
        ax.set_xlabel("Frequency / Hz")
        ax.set_ylabel("PSD / mV$^2$ Hz$^{-1}$")
        ax.grid(visible=True, lw=0.4, alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(fontsize=7, loc="lower left")

    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
