"""Analyze and plot simulation results from saved output.

Loads a simulation's logs.npz and config.yaml from a specified directory,
computes metrics (eeg mean, std, network synchronization R, control max),
and plots/saves EEG traces and power spectral density.

Usage
-----
    uv run python scripts/plot_simulation.py --dir simulations/jansen_rit_baseline
    uv run python scripts/plot_simulation.py --dir simulations/jansen_rit_baseline --transient-ms 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")  # non-interactive backend so the script works headless

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from simulate.config import load_config

from neuro.connectome import load_connectome
from utils.plotting import plot_psd, plot_signals
from utils.processing import steady_window, synchronization
from utils.save_plots import ThesisPlotSaver

if TYPE_CHECKING:
    from neuro.types import FloatArray

N_CHANNELS_TO_PLOT = 8
PSD_NPERSEG = 8192  # High resolution spectral segment length for EEG bands


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the simulation plotting utility."""
    parser = argparse.ArgumentParser(description="Analyze and plot simulation results.")
    parser.add_argument(
        "--dir",
        type=Path,
        required=True,
        help="Path to the simulation output directory containing logs.npz and config.yaml.",
    )
    parser.add_argument(
        "--transient-ms",
        type=float,
        default=0.0,
        help="Leading transient discarded before metrics and plots (milliseconds).",
    )
    return parser.parse_args()


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Load simulation outputs, compute metrics, and save figures."""
    args = parse_args()
    dir_path: Path = args.dir

    config_path = dir_path / "config.yaml"
    if not config_path.exists():
        msg = f"Configuration file not found: {config_path}"
        raise FileNotFoundError(msg)

    npz_path = dir_path / "logs.npz"
    if not npz_path.exists():
        msg = f"Simulation log file not found: {npz_path}"
        raise FileNotFoundError(msg)

    config = load_config(config_path)
    dt_s = float(config["dynamics"]["dt"])
    dt_ms = dt_s * 1000.0  # plotting/PSD utilities expect milliseconds

    activity = None
    with np.load(npz_path) as data:
        eeg: FloatArray = data["y_mea"].T  # (n_sensors, n_samples)
        u: FloatArray = data["u"]

        # Calculate network synchronization from logs if Jansen-Rit dynamics are used
        r_sync = None
        is_jansen_rit = "JansenRit" in config["dynamics"]["class_path"]
        if is_jansen_rit and "x" in data:
            try:
                x_flat = data["x"].squeeze()
                if x_flat.ndim == 1:
                    x_flat = x_flat[np.newaxis, :]
                n_samples_logged = x_flat.shape[0]
                x_grid = x_flat.reshape((n_samples_logged, 6, -1))
                activity = (x_grid[:, 1, :] - x_grid[:, 2, :]).T  # (n_nodes, n_samples)

                # Discard transient for synchronization metric if requested
                activity_steady = activity
                if args.transient_ms > 0.0:
                    activity_steady = steady_window(activity, dt_ms, args.transient_ms)

                r_sync = synchronization(activity_steady)
            except Exception as e:  # noqa: BLE001
                print(f"Could not calculate network synchronization: {e}")

    # Drop transient window if requested
    if args.transient_ms > 0.0:
        print(f"Discarding leading {args.transient_ms} ms transient for metrics and plotting...")
        eeg = steady_window(eeg, dt_ms, args.transient_ms)
        n_drop = round(args.transient_ms / dt_ms)
        u = u[n_drop:]
        if activity is not None:
            activity = steady_window(activity, dt_ms, args.transient_ms)

    print(f"Loaded simulation logs from {npz_path}")
    print(
        f"  EEG shape={eeg.shape}  mean={eeg.mean():+.4e}  std={eeg.std():.4e}  finite={bool(np.isfinite(eeg).all())}",
    )
    if r_sync is not None:
        print(f"  network synchronization R = {r_sync:.3f}")
    print(f"  control |u| max={np.abs(u).max():.3e}")

    metadata = {
        "config": str(config_path),
        "output_dir": str(dir_path.parent),
        "name": dir_path.name,
        "transient_ms": args.transient_ms,
    }
    if r_sync is not None:
        metadata["synchronization_R"] = float(r_sync)

    saver = ThesisPlotSaver(base_dir=str(dir_path.parent))
    n_sensors = eeg.shape[0]
    n_shown = min(N_CHANNELS_TO_PLOT, n_sensors)
    channels = list(range(n_shown))

    # 1. Plot signals (traces)
    fig_signals, _ = plot_signals(
        eeg,
        dt_ms=dt_ms,
        channels_to_plot=channels,
        channel_names=[f"EEG {i}" for i in range(n_sensors)],
        title=f"EEG output (first {n_shown} of {n_sensors} channels)",
        color="#ff7f0e",
    )

    # 2. Plot Fourier power spectrum
    nperseg = min(PSD_NPERSEG, eeg.shape[1])
    fig_freq, _ = plot_psd(
        eeg,
        dt_ms=dt_ms,
        channels_to_plot=channels,
        channel_names=[f"EEG {i}" for i in range(n_sensors)],
        max_freq=50.0,
        nperseg=nperseg,
    )

    # 3. Plot select nodes (one EZ, one PZ, and one healthy) if regional activity is available
    fig_nodes = None
    if activity is not None:
        try:
            connectome = load_connectome()
            ez_node = "lHC"
            pz_node = "lTCI"
            healthy_node = "rHC"
            ez_idx = connectome.region_index[ez_node]
            pz_idx = connectome.region_index[pz_node]
            healthy_idx = connectome.region_index[healthy_node]

            node_signals = activity[[ez_idx, pz_idx, healthy_idx], :]
            node_names = [f"{ez_node} (EZ)", f"{pz_node} (PZ)", f"{healthy_node} (Healthy)"]
            node_colors = ["#d62728", "#ff7f0e", "#1f77b4"]

            fig_nodes, _ = plot_signals(
                node_signals,
                dt_ms=dt_ms,
                channel_names=node_names,
                channels_to_plot=[0, 1, 2],
                stacked=True,
                title="Representative Node Outputs",
                color=node_colors,
            )
        except Exception as e:  # noqa: BLE001
            print(f"Could not plot select nodes: {e}")

    # Save both figures and time-series data in a single subdirectory
    figs = {
        "signals": fig_signals,
        "spectrum": fig_freq,
    }
    if fig_nodes is not None:
        figs["nodes"] = fig_nodes

    saver.save(
        figs,
        name=dir_path.name,
        metadata=metadata,
        data={"eeg": eeg, "u": u},
        overwrite=True,
    )
    plt.close(fig_signals)
    plt.close(fig_freq)
    if fig_nodes is not None:
        plt.close(fig_nodes)

    print(f"Saved EEG plots and data folder to {dir_path}")


if __name__ == "__main__":
    main()
