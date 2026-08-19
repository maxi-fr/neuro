from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from simulate.config import load_config

from neuro.connectome import Connectome
from utils.plotting import plot_psd, plot_signals
from utils.processing import steady_window, synchronization
from utils.save_plots import ThesisPlotSaver

if TYPE_CHECKING:
    from neuro.types import FloatArray

N_CHANNELS_TO_PLOT = 8
PSD_NPERSEG = 8192


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the simulation plotting utility."""
    parser = argparse.ArgumentParser(description="Analyze and plot simulation results.")
    parser.add_argument(
        "--dir",
        type=Path,
        required=True,
        help="Path to the simulation output directory.",
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

    config_files = list(dir_path.glob("*.yaml")) + list(dir_path.glob("*.yml"))
    if not config_files:
        msg = f"No YAML configuration file found in {dir_path}"
        raise FileNotFoundError(msg)
    config_path = config_files[0]

    npz_path = dir_path / "log.npz" if (dir_path / "log.npz").exists() else dir_path / "logs.npz"
    if not npz_path.exists():
        msg = f"Simulation log not found (checked log.npz and logs.npz in {dir_path})"
        raise FileNotFoundError(msg)

    config = load_config(config_path)
    dt_s = float(config["dynamics"]["dt"])
    dt_ms = dt_s * 1000.0

    regional_lfp = None
    with np.load(npz_path) as data:
        eeg: FloatArray = np.asarray(data["sensor_0.y_mea"]).T
        u_raw = data.get("controller.u", None)
        u: FloatArray | None = np.asarray(u_raw) if u_raw is not None else None

        r_sync = None
        if "JansenRit" in config["dynamics"]["class_path"] and "dynamics.x" in data:
            try:
                x_flat = data["dynamics.x"].squeeze()
                if x_flat.ndim == 1:
                    x_flat = x_flat[np.newaxis, :]
                x_grid = x_flat.reshape((x_flat.shape[0], 6, -1))
                regional_lfp = (x_grid[:, 1, :] - x_grid[:, 2, :]).T

                lfp_steady = (
                    steady_window(regional_lfp, dt_ms, args.transient_ms) if args.transient_ms > 0.0 else regional_lfp
                )
                r_sync = synchronization(lfp_steady)
            except Exception as e:  # noqa: BLE001
                print(f"Could not calculate network synchronization: {e}")

    if args.transient_ms > 0.0:
        print(f"Discarding leading {args.transient_ms} ms transient for metrics and plotting...")
        eeg = steady_window(eeg, dt_ms, args.transient_ms)
        n_drop = round(args.transient_ms / dt_ms)
        if u is not None:
            u = u[n_drop:]
        if regional_lfp is not None:
            regional_lfp = steady_window(regional_lfp, dt_ms, args.transient_ms)

    print(f"Loaded simulation logs from {npz_path}")
    print(
        f"  EEG shape={eeg.shape}  mean={eeg.mean():+.4e}  std={eeg.std():.4e}  finite={bool(np.isfinite(eeg).all())}",
    )
    if r_sync is not None:
        print(f"  network synchronization R = {r_sync:.3f}")
    if u is not None:
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

    fig_signals, _ = plot_signals(
        eeg,
        dt_ms=dt_ms,
        channels_to_plot=channels,
        channel_names=[f"EEG {i}" for i in range(n_sensors)],
        title=f"EEG output (first {n_shown} of {n_sensors} channels)",
        color="#ff7f0e",
    )

    nperseg = min(PSD_NPERSEG, eeg.shape[1])
    fig_freq, _ = plot_psd(
        eeg,
        dt_ms=dt_ms,
        channels_to_plot=channels,
        channel_names=[f"EEG {i}" for i in range(n_sensors)],
        max_freq=50.0,
        nperseg=nperseg,
    )

    figs: dict[str, plt.Figure] = {
        "signals": fig_signals,
        "spectrum": fig_freq,
    }

    if regional_lfp is not None:
        try:
            connectome = Connectome.from_config(config["dynamics"].get("connectome", {}))
            ez_idx = connectome.region_index["lHC"]
            pz_idx = connectome.region_index["lTCI"]
            healthy_idx = connectome.region_index["rHC"]

            node_signals = regional_lfp[[ez_idx, pz_idx, healthy_idx], :]
            node_names = ["lHC (EZ)", "lTCI (PZ)", "rHC (Healthy)"]
            node_colors = ["#d62728", "#ff7f0e", "#1f77b4"]

            fig_nodes, _ = plot_signals(
                node_signals,
                dt_ms=dt_ms,
                channel_names=node_names,
                channels_to_plot=[0, 1, 2],
                stacked=True,
                title="Representative Regional LFPs",
                color=node_colors,
            )
            figs["nodes"] = fig_nodes
        except Exception as e:  # noqa: BLE001
            print(f"Could not plot select nodes: {e}")

    data_to_save: dict[str, FloatArray] = {"eeg": eeg}
    if u is not None:
        data_to_save["u"] = u

    saver.save(
        figs,
        name=dir_path.name,
        metadata=metadata,
        data=data_to_save,
        overwrite=True,
    )
    for fig in figs.values():
        plt.close(fig)

    print(f"Saved EEG plots and data folder to {dir_path}")


if __name__ == "__main__":
    main()
