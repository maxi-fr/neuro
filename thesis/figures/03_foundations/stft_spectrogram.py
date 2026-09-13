"""Generate stft_spectrogram.pdf: the periodograms of a propagation-zone node under recruitment."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import decimate
from scipy.signal.windows import hann

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import style

from neuro.connectome import Connectome
from neuro.jansen_rit import JansenRitDynamics, JansenRitParams, lfp, resting_state, simulate_network
from neuro.types import FloatArray  # noqa: TC001 -- runtime import keeps the script runnable standalone

DT = 1e-4
DURATION = 20.0
SEED = 7
K = 0.60
SIGMA = 280.0
SPEED_MM_PER_MS = 50.0
A_BACKGROUND, A_PZ, A_EZ = 3.25, 3.4, 3.6
EZ_NODES = (40, 47, 62)
PZ_NODES = (69, 72)
PROBE_NODE = 69
DECIMATION = (20, 10)
FS = 1.0 / (DT * DECIMATION[0] * DECIMATION[1])
N_SEGMENT = 50
N_HOP = 25
LOG_FLOOR = 1e-8
OUT = Path(__file__).with_suffix(".pdf")


def _pz_trace() -> tuple[FloatArray, FloatArray]:
    """Decimated time vector and local field potential of a propagation-zone node in the network."""
    conn = Connectome.from_config({"speed": SPEED_MM_PER_MS, "K": K})
    gains = np.full(len(conn.region_labels), A_BACKGROUND)
    gains[list(EZ_NODES)] = A_EZ
    gains[list(PZ_NODES)] = A_PZ
    dyn = JansenRitDynamics(
        dt=DT,
        params=JansenRitParams(A=gains, sigma=SIGMA),
        conn=conn,
        seed=SEED,
        initial_state=resting_state(conn, DT),
    )
    _, x = simulate_network(dyn=dyn, duration=DURATION)
    y = decimate(decimate(lfp(x)[PROBE_NODE], DECIMATION[0], ftype="fir"), DECIMATION[1], ftype="fir")
    return np.arange(y.size) / FS, y


def _periodograms(y: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Frequency axis, segment end times and periodograms of ``y`` in decibels, on the segmentation grid."""
    n_frames = (y.size - N_SEGMENT) // N_HOP + 1
    taper = hann(N_SEGMENT, sym=False)
    segments = np.stack([y[q * N_HOP : q * N_HOP + N_SEGMENT] for q in range(n_frames)])
    power = np.abs(np.fft.rfft(segments * taper[None, :], axis=1)) ** 2
    fold = np.full(N_SEGMENT // 2 + 1, 2.0)
    fold[0] = 1.0
    if N_SEGMENT % 2 == 0:
        fold[-1] = 1.0
    power = power * fold[None, :] / (FS * np.sum(taper**2))
    freqs = np.fft.rfftfreq(N_SEGMENT, d=1.0 / FS)
    frame_times = (np.arange(n_frames) * N_HOP + N_SEGMENT) / FS
    return freqs, frame_times, 10.0 * np.log10(power + LOG_FLOOR)


def main() -> None:
    """Write the trace-and-spectra figure."""
    t, y = _pz_trace()
    freqs, frame_times, power_db = _periodograms(y)

    fig, axes = plt.subplots(2, 1, figsize=style.figsize(height_in=4.0), height_ratios=[1.0, 1.3], constrained_layout=True)

    ax = axes[0]
    ax.plot(t, y, lw=0.6, color="C0")
    ax.set_ylabel(r"$y_{\mathrm{LFP}}$ / mV")
    ax.set_title("(a) Propagation-zone node recruited by the epileptogenic zone", loc="left")

    ax = axes[1]
    d_f = freqs[1] - freqs[0]
    edges_t = np.concatenate([[frame_times[0] - N_HOP / FS], frame_times])
    edges_f = np.concatenate([freqs - 0.5 * d_f, [freqs[-1] + 0.5 * d_f]])
    mesh = ax.pcolormesh(edges_t, edges_f, power_db.T, cmap="viridis", shading="flat")
    fig.colorbar(mesh, ax=ax, label=r"$\hat{\Phi}_q$ / dB", pad=0.02)
    ax.set_ylabel("Frequency / Hz")
    ax.set_xlabel("Time / s")
    ax.set_title("(b) Spectrogram", loc="left")

    for ax in axes:
        ax.set_xlim(0.0, DURATION)
        ax.grid(visible=True, lw=0.4, alpha=0.4)
        ax.set_axisbelow(True)

    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
