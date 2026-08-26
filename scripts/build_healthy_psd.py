from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import yaml
from simulate.config import deep_merge, load_config
from simulate.experiment import ExperimentManager

from neuro.config import StftGeometry
from neuro.predictor.data import load_trajectory
from neuro.provenance import data_plant_fingerprint, plant_fingerprint
from neuro.spectral import compute_log_power_frames, compute_periodograms, windowed_mean_square
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
    geometry: StftGeometry | None = None,
    workers: int = 1,
) -> Path:
    """Pool healthy periodograms and log-power Frames into quantile envelopes and save to npz.

    The same npz carries ``Pref`` for the waveform spectral hinge, ``Pref_ms`` for the
    ``eeg_ms`` Observable, and ``Pref_frames`` for the Observable hinge.
    """
    first_cfg = _batch_configs(yaml.safe_load(config_path.read_text(encoding="utf-8")))[0]
    dt_plant = float(first_cfg["dynamics"]["dt"])
    downsample = int(first_cfg["estimator"]["downsample"])
    fs = 1.0 / (dt_plant * downsample)
    if geometry is None:
        window = round(window_s * fs)
        hop = round(hop_s * fs)
        geometry = StftGeometry(n_segment=window, n_hop=hop)
    else:
        window = geometry.n_segment
        hop = geometry.n_hop

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
    all_frames = []
    for f in data_files:
        _, y = load_trajectory(str(f), n_steps=None, downsample=downsample, dt=dt_plant)
        p_windows = compute_periodograms(y, fs=fs, window=window, hop=hop)
        frames = compute_log_power_frames(y, geometry, fs=fs)
        if len(p_windows) > 0:
            all_windows.append(p_windows)
            all_ms_windows.append(windowed_mean_square(y, window=window, hop=hop))
        if len(frames) > 0:
            all_frames.append(frames)
    if not all_windows or not all_frames:
        msg = "No valid windows extracted from trajectory files."
        raise RuntimeError(msg)

    stacked = np.concatenate(all_windows, axis=0)  # (total_windows, n_channels, n_bins)
    reference = np.quantile(stacked, quantile, axis=0)  # (n_channels, n_bins)
    reference_ms = np.quantile(np.concatenate(all_ms_windows, axis=0), quantile, axis=0)  # (n_channels,)
    stacked_frames = np.concatenate(all_frames, axis=0)  # (total_frames, n_channels, n_values)
    reference_frames = np.quantile(stacked_frames, quantile, axis=0)  # (n_channels, n_values)

    fp = data_plant_fingerprint(data_dir) or plant_fingerprint(first_cfg)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        Pref=reference,
        Pref_ms=reference_ms,
        Pref_frames=reference_frames,
        freqs=np.fft.rfftfreq(window, 1.0 / fs),
        fs=fs,
        L=window,
        R=hop,
        quantile=quantile,
        n_windows=len(stacked),
        n_frames=len(stacked_frames),
        n_segment=geometry.n_segment,
        n_hop=geometry.n_hop,
        band_hz=np.asarray(geometry.band_hz if geometry.band_hz is not None else [-1.0, -1.0]),
        n_bin_pool=geometry.n_bin_pool,
        kernel=geometry.kernel,
        kernel_width=geometry.kernel_width,
        plant_fingerprint=str(fp) if fp is not None else "",
    )
    print(f"Saved healthy PSD reference to {output_path}:")
    print(f"  Shape: {reference.shape} (channels={reference.shape[0]}, bins={reference.shape[1]})")
    print(
        f"  Observable shape: {reference_frames.shape} (channels={reference_frames.shape[0]}, values={reference_frames.shape[1]})"
    )
    print(f"  Pooled windows: {len(stacked)}, frames: {len(stacked_frames)}")
    print(f"  Quantile: {quantile}")
    print(f"  Geometry: L={window} ({window / fs:.3f} s), R={hop} ({hop / fs:.3f} s), fs={fs} Hz")
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
        "--band-hz",
        type=float,
        nargs=2,
        default=None,
        metavar=("LO", "HI"),
        help="STFT frequency band in Hz (e.g. 3.0 12.0).",
    )
    parser.add_argument(
        "--bin-pool",
        type=int,
        default=1,
        help="STFT bin pooling factor (default: 1).",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        choices=["boxcar", "triangular", "hann"],
        default="boxcar",
        help="Frame Kernel type (default: boxcar).",
    )
    parser.add_argument(
        "--kernel-width",
        type=int,
        default=1,
        help="Frame Kernel width in frames (default: 1).",
    )
    parser.add_argument(
        "--segment-steps",
        type=int,
        default=None,
        help="STFT segment length in samples (overrides --window-s).",
    )
    parser.add_argument(
        "--hop-steps",
        type=int,
        default=None,
        help="STFT hop size in samples (overrides --hop-s).",
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
    geom = None
    if (
        args.segment_steps is not None
        or args.hop_steps is not None
        or args.band_hz is not None
        or args.bin_pool > 1
        or args.kernel != "boxcar"
        or args.kernel_width > 1
    ):
        first_cfg = _batch_configs(yaml.safe_load(args.config.read_text(encoding="utf-8")))[0]
        dt_plant = float(first_cfg["dynamics"]["dt"])
        downsample = int(first_cfg["estimator"]["downsample"])
        fs = 1.0 / (dt_plant * downsample)
        n_seg = args.segment_steps if args.segment_steps is not None else round(args.window_s * fs)
        n_hp = args.hop_steps if args.hop_steps is not None else round(args.hop_s * fs)
        band = tuple(args.band_hz) if args.band_hz is not None else None
        geom = StftGeometry(
            n_segment=n_seg,
            n_hop=n_hp,
            band_hz=band,
            n_bin_pool=args.bin_pool,
            kernel=args.kernel,
            kernel_width=args.kernel_width,
        )

    build_healthy_psd(
        config_path=args.config,
        data_dir=args.data_dir,
        output_path=args.output,
        quantile=args.quantile,
        window_s=args.window_s,
        hop_s=args.hop_s,
        geometry=geom,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
