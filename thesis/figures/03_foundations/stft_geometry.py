"""Generate stft_geometry.pdf: the segmentation grid a Short-Time Fourier Transform imposes."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import chirp
from scipy.signal.windows import hann

FS = 200.0
DURATION = 3.0
F_START, F_END = 4.0, 45.0
N_SEGMENT = 64
N_HOP = 32
N_DRAWN = 4
OUT = Path(__file__).with_suffix(".pdf")


def _periodograms(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hann periodograms of every segment, with the frequency axis and the frame times."""
    n_frames = (y.size - N_SEGMENT) // N_HOP + 1
    taper = hann(N_SEGMENT, sym=False)
    segments = np.stack([y[q * N_HOP : q * N_HOP + N_SEGMENT] for q in range(n_frames)])
    spectrum = np.fft.rfft(segments * taper[None, :], axis=1)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(N_SEGMENT, d=1.0 / FS)
    frame_times = (np.arange(n_frames) * N_HOP + N_SEGMENT) / FS
    return freqs, frame_times, power


def main() -> None:
    """Write the two-panel segmentation-and-lattice figure."""
    t = np.arange(int(DURATION * FS)) / FS
    y = chirp(t, f0=F_START, t1=DURATION, f1=F_END, method="linear")
    freqs, frame_times, power = _periodograms(y)

    fig, axes = plt.subplots(2, 1, figsize=(6.6, 4.6), height_ratios=[1.0, 1.25], constrained_layout=True)

    ax = axes[0]
    ax.plot(t, y, lw=0.7, color="C0", zorder=3)
    taper = hann(N_SEGMENT, sym=False)
    seg_t = np.arange(N_SEGMENT) / FS
    for q in range(N_DRAWN):
        base = -1.7 - 0.62 * q
        offset = q * N_HOP / FS
        label = r"Hann taper $g[n]$" if q == 0 else None
        ax.fill_between(seg_t + offset, base, base + 0.5 * taper, color=f"C{q + 1}", alpha=0.35, lw=0)
        ax.plot(seg_t + offset, base + 0.5 * taper, lw=0.9, color=f"C{q + 1}", label=label)
        ax.text(offset + N_SEGMENT / FS + 0.02, base + 0.12, f"$q = {q}$", ha="left", va="center", fontsize=7)
    ax.annotate(
        "",
        xy=(0.0, 1.55),
        xytext=(N_SEGMENT / FS, 1.55),
        arrowprops={"arrowstyle": "<->", "lw": 0.8, "color": "0.25"},
    )
    ax.text(0.5 * N_SEGMENT / FS, 1.68, r"$L_\mathrm{s}$", ha="center", va="bottom", fontsize=8)
    ax.annotate(
        "",
        xy=(0.0, -4.15),
        xytext=(N_HOP / FS, -4.15),
        arrowprops={"arrowstyle": "<->", "lw": 0.8, "color": "0.25"},
    )
    ax.text(0.5 * N_HOP / FS, -4.42, r"$H_\mathrm{s}$", ha="center", va="top", fontsize=8)
    ax.set_ylim(-5.3, 2.4)
    ax.set_yticks([-1.0, 0.0, 1.0])
    ax.set_ylabel("Amplitude / 1")
    ax.set_title("(a) Segments on the hop grid", fontsize=8, loc="left")
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[1]
    edges_t = np.concatenate([[frame_times[0] - N_HOP / FS], frame_times])
    d_f = freqs[1] - freqs[0]
    edges_f = np.concatenate([freqs - 0.5 * d_f, [freqs[-1] + 0.5 * d_f]])
    mesh = ax.pcolormesh(edges_t, edges_f, 10.0 * np.log10(power.T + 1e-12), cmap="viridis", shading="flat")
    fig.colorbar(mesh, ax=ax, label="Power / dB", pad=0.02)
    ax.vlines(frame_times, 0.0, FS / 2, colors="w", lw=0.35, alpha=0.5)
    ax.annotate(
        "",
        xy=(0.0, 112.0),
        xytext=(N_SEGMENT / FS, 112.0),
        arrowprops={"arrowstyle": "<->", "lw": 0.9, "color": "0.25"},
    )
    ax.text(N_SEGMENT / FS + 0.03, 112.0, "samples spectrum $q = 0$ waits for", ha="left", va="center", fontsize=7)
    ax.set_ylim(0.0, 128.0)
    ax.set_yticks([0.0, 25.0, 50.0, 75.0, 100.0])
    ax.set_ylabel("Frequency / Hz")
    ax.set_xlabel("Time / s")
    ax.set_title("(b) The spectra the grid produces", fontsize=8, loc="left")

    for ax in axes:
        ax.grid(visible=True, lw=0.4, alpha=0.4)
        ax.set_axisbelow(True)
        ax.set_xlim(0.0, DURATION)

    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
