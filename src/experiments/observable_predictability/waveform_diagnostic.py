from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from matplotlib import pyplot as plt

from experiments.observable_predictability.ensemble import (
    eeg_history_path,
    eeg_path,
    load_manifest,
    manifest_identities,
    region_history_path,
)
from experiments.observable_predictability.features import baseline_stft_geometry, condition_waveform_pair
from experiments.observable_predictability.metrics import SEIZURE_BAND_HZ
from experiments.observable_predictability.scoring import compute_potential_predictability, compute_replay_moments
from neuro.plant.seizure_calibration import SEIZURE_PTP_MV, SPREAD_WINDOW_S
from neuro.spectral import LOG_FLOOR, compute_forecast_log_power_frames, compute_segment_window

if TYPE_CHECKING:
    from experiments.observable_predictability.scoring import PotentialPredictability, ReplayMoments
    from neuro.config import StftGeometry
    from neuro.types import FloatArray


def frame_coefficients(history: FloatArray, future: FloatArray, geometry: StftGeometry) -> np.ndarray:
    """Compute full complex coefficients on the causal Frame grid."""
    prefix = geometry.n_segment - geometry.n_hop
    stitched = np.concatenate((history[-prefix:], future))
    segments = np.lib.stride_tricks.sliding_window_view(stitched, geometry.n_segment, axis=0)[:: geometry.n_hop]
    window = compute_segment_window(
        geometry.n_segment, window=geometry.window, asymmetric_window=geometry.asymmetric_window
    )
    return np.fft.rfft(segments * window, axis=-1)


def snapshot_phase_agreement(coefficients: np.ndarray, n_replays: int) -> FloatArray:
    """Average the pairwise phase cosines over bins and channels, weighted by replay power."""
    power = abs(coefficients) ** 2
    unit = np.divide(coefficients, abs(coefficients), out=np.zeros_like(coefficients), where=power > 0)
    grouped_unit = unit.reshape(-1, n_replays, *unit.shape[1:])
    grouped_power = power.reshape(-1, n_replays, *power.shape[1:])
    active = np.count_nonzero(grouped_power, axis=1)
    pairs = active * (active - 1)
    phase_sum = abs(grouped_unit.sum(axis=1)) ** 2 - active
    pair_cosine = np.divide(phase_sum, pairs, out=np.zeros_like(phase_sum), where=pairs > 0)
    weights = np.where(pairs > 0, grouped_power.mean(axis=1), 0.0)
    return (weights * pair_cosine).sum(axis=(-1, -2)) / weights.sum(axis=(-1, -2))


def main(directory: Path, destination: Path) -> None:
    """Compare waveform, power and phase on matched evaluation replays with recorded histories."""
    manifest = load_manifest(directory)
    geometry = baseline_stft_geometry()
    fs = 50.0
    lo, hi = geometry.bin_range(fs)
    window = compute_segment_window(
        geometry.n_segment, window=geometry.window, asymmetric_window=geometry.asymmetric_window
    )
    scale = np.sqrt(2.0 / (fs * np.sum(window**2)))
    frequencies = np.fft.rfftfreq(geometry.n_segment, 1 / fs)[lo:hi]
    in_band = (frequencies >= SEIZURE_BAND_HZ[0]) & (frequencies <= SEIZURE_BAND_HZ[1])
    moments = {}
    phase_agreement = []
    snapshot_states = []
    spread_samples = round(SPREAD_WINDOW_S * manifest["region_fs"])
    for branch in manifest["branches"]:
        if branch["plant"] != "seizure":
            continue
        name = branch["name"]
        identities = manifest_identities(manifest, split="eval", plant="seizure", branch=name, arm="zero")
        n_replays = manifest["splits"]["eval"]["n_replicates"]
        seeds = manifest["splits"]["eval"]["seeds"]["seizure"]
        np.testing.assert_equal(
            [(identity.parent_seed, identity.replicate) for identity in identities],
            [(seed, replay) for seed in seeds for replay in range(n_replays)],
        )
        snapshot_states.append(
            snapshot_seizure_state(region_history_path(directory, name), spread_samples)[np.argsort(seeds)]
        )
        history_store = np.load(eeg_history_path(directory, name), mmap_mode="r")
        future_store = np.load(eeg_path(directory, name, "zero"), mmap_mode="r")
        waveforms, coefficients, means = [], [], []
        for row, continuation in enumerate(future_store):
            history, waveform = condition_waveform_pair(
                continuation.T, manifest["fs"], history_raw=history_store[row // n_replays].T
            )
            # Conditioning returns a history when one is supplied.
            spectrum = frame_coefficients(cast("FloatArray", history), waveform, geometry)
            coeff = spectrum[..., lo:hi] * scale
            if row == 0:
                np.testing.assert_allclose(
                    np.log(abs(coeff) ** 2 + LOG_FLOOR),
                    compute_forecast_log_power_frames(cast("FloatArray", history), waveform, geometry, fs=fs),
                    atol=1e-8,
                )
            waveforms.append(waveform)
            coefficients.append(coeff)
            means.append(spectrum[..., 0].real / window.sum())
        waveform = np.stack(waveforms)
        coeff = np.stack(coefficients)
        power = abs(coeff) ** 2
        coeff_band = coeff[..., in_band]
        power_band = power[..., in_band]
        frames = (np.arange(coeff.shape[1]) + 1) * geometry.n_hop / fs
        wave_times = (np.arange(waveform.shape[1]) + 1) / fs
        frame_means = np.stack(means)
        targets = {
            "waveform": (waveform, wave_times),
            "log_power_full": (np.log(power + LOG_FLOOR), frames),
            "waveform_centered": (waveform[:, geometry.n_hop - 1 :: geometry.n_hop] - frame_means, frames),
            "frame_mean": (frame_means, frames),
            "complex_band": (np.stack((coeff_band.real, coeff_band.imag), axis=-1), frames),
            "magnitude_band": (abs(coeff_band), frames),
            "linear_power_band": (power_band, frames),
            "log_power_band": (np.log(power_band + LOG_FLOOR), frames),
        }
        for label, (values, times) in targets.items():
            moments.setdefault(label, []).append(compute_replay_moments(values, times, identities))

        phase_agreement.extend(snapshot_phase_agreement(coeff_band, n_replays))
        print(f"Analyzed {name}", flush=True)  # noqa: T201 -- report progress during replay analysis

    results = {label: compute_potential_predictability(blocks) for label, blocks in moments.items()}
    q10, median, q90 = np.quantile(phase_agreement, [0.1, 0.5, 0.9], axis=0)
    complex_noise, magnitude_noise = (
        np.concatenate([block.variances for block in moments[label]]).mean(axis=0).sum(axis=-1)
        for label in ("complex_band", "magnitude_band")
    )
    phase_noise_fraction = 1 - magnitude_noise / complex_noise
    save_results(
        results, frames, (q10, median, q90), phase_noise_fraction, geometry.n_segment / fs, destination=destination
    )
    save_waveform_split(moments, snapshot_states, geometry.n_hop, geometry.n_segment / fs, destination)


def snapshot_seizure_state(history_path: Path, spread_samples: int) -> FloatArray:
    """Measure the fraction of recruited regions using LFP ending at the snapshot."""
    history = np.load(history_path, mmap_mode="r")
    return (np.ptp(history[..., -spread_samples:], axis=-1) > SEIZURE_PTP_MV).mean(axis=-1)


def save_waveform_split(
    moments: dict[str, list[ReplayMoments]],
    snapshot_states: list[FloatArray],
    hop_samples: int,
    support_s: float,
    destination: Path,
) -> None:
    """Save waveform component ceilings pooled and split by Seizure State at the snapshot."""
    threshold = 0.25
    state = np.concatenate(snapshot_states)
    groups = (np.ones_like(state, dtype=bool), state < threshold, state >= threshold)
    times = moments["frame_mean"][0].times
    ceilings = []
    for group in groups:
        curves = []
        for target in ("waveform", "frame_mean", "waveform_centered"):
            blocks = moments[target]
            means = np.concatenate([block.means for block in blocks])[group]
            variances = np.concatenate([block.variances for block in blocks])[group]
            selected = replace(blocks[0], means=means, variances=variances)
            score = compute_potential_predictability([selected]).signal_fraction
            curves.append(score[hop_samples - 1 :: hop_samples] if target == "waveform" else score)
        ceilings.append(curves)
    np.savez_compressed(
        destination / "waveform_split.npz",
        times=times,
        ceilings=ceilings,
        counts=[np.count_nonzero(group) for group in groups],
        state_ranges=[(state[group].min(), state[group].max()) for group in groups],
        threshold=threshold,
        support_s=support_s,
    )


def save_results(  # noqa: PLR0913 -- separate curve data, support marker, and output directory
    results: dict[str, PotentialPredictability],
    frames: FloatArray,
    quantiles: tuple[FloatArray, FloatArray, FloatArray],
    phase_noise_fraction: FloatArray,
    support_s: float,
    *,
    destination: Path,
) -> None:
    """Write the sampled curves and plot with the shared-history boundary."""
    q10, median, q90 = quantiles
    columns = [frames, *quantiles, phase_noise_fraction]
    for result in results.values():
        indices = np.searchsorted(result.times.round(8), frames.round(8))
        np.testing.assert_allclose(result.times[indices], frames)
        columns.append(result.signal_fraction[indices])
    np.savetxt(
        destination / "curves.csv",
        np.column_stack(columns),
        delimiter=",",
        header="lookahead_s,phase_p10,phase_median,phase_p90,phase_noise_fraction," + ",".join(results),
        comments="",
    )
    fig, (phase_ax, waveform_ax, band_ax) = plt.subplots(3, 1, figsize=(8, 8), sharex=True, layout="constrained")
    phase_ax.fill_between(frames, q10, q90, alpha=0.2, label="10th to 90th snapshot percentiles")
    phase_ax.plot(frames, median, label="Median snapshot")
    phase_ax.set_ylabel("Pairwise phase agreement")
    groups = (
        (
            waveform_ax,
            {
                "waveform": "50 Hz waveform",
                "frame_mean": "Trailing weighted mean",
                "waveform_centered": "Waveform minus trailing mean",
                "log_power_full": "Full-band log-power",
            },
        ),
        (
            band_ax,
            {
                "complex_band": "Complex coefficients",
                "magnitude_band": "Magnitude",
                "log_power_band": "Log-power",
            },
        ),
    )
    for ax, labels in groups:
        for target, label in labels.items():
            result = results[target]
            ax.plot(result.times, result.signal_fraction, label=label)
        ax.set_ylabel("Signal / total variance")
    band_ax.set(xlabel="Lookahead from snapshot (s)", title="Matched 4 to 12 Hz coefficients")
    waveform_ax.set_title("Broadband waveform and log-power")
    for ax in (phase_ax, waveform_ax, band_ax):
        ax.axhline(0, color="0.65", linewidth=0.7)
        ax.axvline(support_s, color="0.55", linestyle="--", linewidth=0.8)
        ax.legend()
    fig.savefig(destination / "phase_and_power.png", dpi=180)


if __name__ == "__main__":
    source, output = sys.argv[1:]
    main(Path(source), Path(output))
