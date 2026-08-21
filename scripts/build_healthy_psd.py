from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import yaml
from simulate.config import deep_merge, load_config
from simulate.experiment import ExperimentManager

from neuro.predictor.data import load_trajectory
from neuro.provenance import data_plant_fingerprint, plant_fingerprint
from neuro.spectral import compute_periodograms, windowed_mean_square
from neuro.validation import validate_simulation_config


def _batch_configs(config: dict) -> list[dict]:
    """Expand the ``experiments:`` batch form: the first entry is full, the rest are overrides."""
    raw = config["experiments"]
    merged = [raw[0]]
    for override in raw[1:]:
        merged.append(deep_merge(merged[-1], override))
    return merged


def run_healthy_simulation(config_path: Path, output_dir: Path, workers: int) -> None:
    """Run the healthy reference simulations described by ``config_path`` into ``output_dir``."""
    configs = _batch_configs(load_config(config_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / config_path.name)

    for merged in configs:
        validate_simulation_config(merged)
    ExperimentManager(output_dir).run_batch(configs, max_num_processes=workers, use_mmap=True)


def build_healthy_psd(  # noqa: PLR0913
    config_path: Path,
    data_dir: Path,
    output_path: Path,
    *,
    quantile: float = 0.90,
    window_s: float = 1.0,
    hop_s: float = 0.5,
    workers: int = 1,
) -> Path:
    """Pool healthy periodograms into a per-``(channel, bin)`` quantile envelope and save it to npz.

    The same npz also carries ``Pref_ms``, the per-channel mean-square envelope the ``eeg_ms``
    observable hinges against, measured on the same segment grid.

    The decimation is read from ``config_path`` rather than passed in, so the envelope is always
    measured at the rate the closed loop runs the cost at.
    """
    first_cfg = _batch_configs(yaml.safe_load(config_path.read_text(encoding="utf-8")))[0]
    dt_plant = float(first_cfg["dynamics"]["dt"])
    downsample = int(first_cfg["estimator"]["downsample"])
    fs = 1.0 / (dt_plant * downsample)
    window = round(window_s * fs)
    hop = round(hop_s * fs)

    data_files = sorted(f for f in data_dir.glob("*.npz") if f.name != output_path.name)
    if not data_files:
        print(f"No trajectory files in {data_dir}. Running simulations from {config_path}...")
        run_healthy_simulation(config_path, data_dir, workers=workers)
        data_files = sorted(f for f in data_dir.glob("*.npz") if f.name != output_path.name)
    if not data_files:
        msg = f"No simulation data files found or generated in {data_dir}"
        raise RuntimeError(msg)

    all_windows = []
    all_ms_windows = []
    for f in data_files:
        _, y = load_trajectory(str(f), n_steps=None, downsample=downsample, dt=dt_plant)
        p_windows = compute_periodograms(y, fs=fs, window=window, hop=hop)
        if len(p_windows) > 0:
            all_windows.append(p_windows)
            all_ms_windows.append(windowed_mean_square(y, window=window, hop=hop))
    if not all_windows:
        msg = "No valid windows extracted from trajectory files."
        raise RuntimeError(msg)

    stacked = np.concatenate(all_windows, axis=0)  # (total_windows, n_channels, n_bins)
    reference = np.quantile(stacked, quantile, axis=0)  # (n_channels, n_bins)
    # The eeg_ms observable's reference; see MsEnvelope for why it is measured rather than derived.
    reference_ms = np.quantile(np.concatenate(all_ms_windows, axis=0), quantile, axis=0)  # (n_channels,)

    fp = data_plant_fingerprint(data_dir) or plant_fingerprint(first_cfg)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        Pref=reference,
        Pref_ms=reference_ms,
        freqs=np.fft.rfftfreq(window, 1.0 / fs),
        fs=fs,
        L=window,
        R=hop,
        quantile=quantile,
        n_windows=len(stacked),
        plant_fingerprint=str(fp) if fp is not None else "",
    )
    print(f"Saved healthy PSD reference to {output_path}:")
    print(f"  Shape: {reference.shape} (channels={reference.shape[0]}, bins={reference.shape[1]})")
    print(f"  Pooled windows: {len(stacked)}")
    print(f"  Quantile: {quantile}")
    print(f"  Geometry: L={window} ({window_s} s), R={hop} ({hop_s} s), fs={fs} Hz")
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build healthy EEG reference envelope from simulations.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/simulation/healthy_reference.yaml"),
        help="Path to healthy reference simulation YAML config.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/healthy_reference"),
        help="Directory to store or load simulation trajectories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/healthy_psd.npz"),
        help="Path to output reference envelope NPZ file.",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.90,
        help="Quantile for healthy reference envelope (default: 0.90).",
    )
    parser.add_argument(
        "--window-s",
        type=float,
        default=1.0,
        help="Periodogram window length in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--hop-s",
        type=float,
        default=0.5,
        help="Periodogram hop size in seconds (default: 0.5).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of workers for simulation batch (default: 4).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for building healthy PSD reference."""
    args = parse_args()
    build_healthy_psd(
        config_path=args.config,
        data_dir=args.data_dir,
        output_path=args.output,
        quantile=args.quantile,
        window_s=args.window_s,
        hop_s=args.hop_s,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
