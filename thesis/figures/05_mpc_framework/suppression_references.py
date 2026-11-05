"""Compare reference-based suppression penalties on saved unstimulated EEG."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import style

from neuro.config import StftGeometry
from neuro.filtering import antialias_filter
from neuro.plant.eeg import build_eeg_leadfield
from neuro.spectral import compute_log_power_frames
from neuro.types import FloatArray

OUT = Path(__file__).resolve().with_suffix(".pdf")
DATA = OUT.parents[3] / "data"
FS = 50
GEOMETRY = StftGeometry(n_segment=50, n_hop=25)
QUANTILE = 0.90


def load_channel(path: Path, channel: int) -> FloatArray:
    """Return one channel filtered and sampled at 50 Hz from the 10 kHz recording."""
    with np.load(path) as run:
        samples = run["sensor_0.y_mea"][:, channel].copy()
    return antialias_filter(samples[:, None], fs=10_000, downsample=200)[::200, 0]


def log_power(samples: FloatArray) -> FloatArray:
    """Return one log-power spectrum per frame."""
    return compute_log_power_frames(samples[:, None], GEOMETRY, fs=FS)[:, 0, :]


def main() -> None:
    """Plot healthy reference statistics and compare held-out healthy and seizure runs."""
    _, channels = build_eeg_leadfield()
    channel = list(channels).index("CP5")
    healthy_dir = DATA / "healthy_reference_100hz"
    reference = [load_channel(healthy_dir / f"sim_{i:03}.npz", channel)[2 * FS :] for i in range(9)]
    mean = np.concatenate(reference).mean()
    reference_frames = np.concatenate([log_power(y) for y in reference])
    lower, q25, q75, envelope = np.quantile(reference_frames, [0.10, 0.25, 0.75, QUANTILE], axis=0)
    frequencies = np.arange(13)
    healthy = load_channel(healthy_dir / "sim_009.npz", channel)[2 * FS : 20 * FS]
    seizure = load_channel(DATA / "uncontrolled_2" / "log.npz", channel)[2 * FS : 20 * FS]

    fig, (wave, spectral) = plt.subplots(1, 2, figsize=style.figsize(height_in=2.55), layout="constrained")
    wave.axhline(mean, color="0.25", linestyle="--", label="Healthy mean")
    spectral.fill_between(frequencies, lower[:13], envelope[:13], color="0.85", label="Reference 10th--90th pct.")
    spectral.fill_between(frequencies, q25[:13], q75[:13], color="0.65", label="Reference 25th--75th pct.")
    for samples, label, color in [(healthy, "Healthy", "#0072B2"), (seizure, "Seizure", "#D55E00")]:
        frames = log_power(samples)
        wave.plot(10 + np.arange(4 * FS) / FS, samples[8 * FS : 12 * FS], label=label, color=color)
        spectral.plot(frequencies, np.median(frames, axis=0)[:13], label=f"{label} median", color=color)
        print(label, "RMS", np.sqrt(np.mean((samples - mean) ** 2)))
        print(label, "exceedance 3 Hz", np.mean(frames[:, 3] > envelope[3]), "frames", len(frames))

    wave.set(title="(a) CP5 waveform", xlabel="Time (s)", ylabel="EEG amplitude")
    spectral.set(title="(b) CP5 spectral reference", xlabel="Frequency (Hz)", ylabel="Log power")
    for ax in (wave, spectral):
        ax.grid()
        ax.legend()
    fig.savefig(OUT)


if __name__ == "__main__":
    main()
