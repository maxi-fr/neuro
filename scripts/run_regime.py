"""Run an open-loop Jansen-Rit regime and save EEG and spectrum plots.

Reads a regime config (the ``dynamics`` block of an orchestrator-style YAML such
as ``scripts/configs/jansen_rit_healthy.yaml``), drives the plant with no control
via :func:`neuro.sweep.run_open_loop`, and renders:

* a stacked EEG-trace figure, and
* a Welch power spectrum of the EEG.

It also reports the network synchronization and dominant EEG frequency so the
regime (healthy alpha / epileptiform / parkinsonian beta) is quantified.

Usage
-----
    uv run python scripts/run_regime.py --config scripts/configs/jansen_rit_healthy.yaml
    uv run python scripts/run_regime.py --config scripts/configs/jansen_rit_pd.yaml --plot-path artifacts/pd.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from simulate.config import load_config

from neuro.sweep import run_open_loop
from utils.plotting import plot_fourier, plot_signals
from utils.processing import dominant_frequency, steady_window, synchronization
from utils.save_plots import ThesisPlotSaver

DEFAULT_CONFIG = Path("scripts/configs/jansen_rit_healthy.yaml")
DEFAULT_PLOT_PATH = Path("artifacts/jansen_rit_healthy.png")
DEFAULT_TRANSIENT_MS = 1000.0  # discarded before measuring/plotting (removes ramp-up)
N_EEG_TRACES = 8
PSD_NPERSEG = 8192  # Welch segment: ~1.2 Hz resolution at dt=0.1 ms (fs=10 kHz)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the regime runner."""
    parser = argparse.ArgumentParser(description="Run an open-loop Jansen-Rit regime and plot it.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Regime YAML config.")
    parser.add_argument("--duration-ms", type=float, default=None, help="Override the config's t_end (ms).")
    parser.add_argument(
        "--transient-ms",
        type=float,
        default=DEFAULT_TRANSIENT_MS,
        help="Leading transient discarded before metrics/plots (ms).",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=DEFAULT_PLOT_PATH,
        help="Path prefix (stem) for the saved plots and data folder.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting (just print metrics).")
    return parser.parse_args()


def main() -> None:
    """Load a regime config, run it open-loop, and save the diagnostic plots."""
    args = parse_args()
    config = load_config(args.config)
    dyn = config["dynamics"]
    dt_ms = float(dyn["dt"])
    duration_ms = float(args.duration_ms if args.duration_ms is not None else config["t_end"])

    print(f"Running regime {args.config.name} for {duration_ms} ms at dt={dt_ms} ms...")
    activity, eeg = run_open_loop(
        model_params=dyn.get("model_params"),
        duration_ms=duration_ms,
        dt=dt_ms,
        nsig=dyn.get("nsig", 0.0),
        seed=dyn.get("seed", 0),
        connectome=dyn.get("connectome", "connectivity_76.zip"),
        coupling_strength=dyn.get("coupling_strength", 1.0),
    )

    # Drop the ramp-up transient so metrics/spectra reflect the steady regime.
    activity = steady_window(activity, dt_ms, args.transient_ms)
    eeg = steady_window(eeg, dt_ms, args.transient_ms)

    r_sync = synchronization(activity)
    f_dom = dominant_frequency(eeg, dt_ms)
    print(f"  activity={activity.shape}  eeg={eeg.shape}")
    print(f"  network synchronization R = {r_sync:.3f}")
    print(f"  dominant EEG frequency    = {f_dom:.2f} Hz")

    if args.no_plot:
        return

    saver = ThesisPlotSaver(base_dir=str(args.plot_path.parent))
    metadata = {
        "config_file": str(args.config),
        "transient_ms": args.transient_ms,
        "synchronization_R": float(r_sync),
        "dominant_frequency_Hz": float(f_dom),
    }

    # 1. Plot signals (traces)
    fig_signals, _ = plot_signals(
        eeg,
        dt_ms=dt_ms,
        channels_to_plot=list(range(min(N_EEG_TRACES, eeg.shape[0]))),
        channel_names=[f"EEG {i}" for i in range(eeg.shape[0])],
        title=f"EEG output — {args.config.stem}",
        color="#ff7f0e",
    )

    # 2. Plot Fourier power spectrum
    nperseg = min(PSD_NPERSEG, eeg.shape[1])
    fig_freq, _ = plot_fourier(
        eeg,
        dt_ms=dt_ms,
        mode="power",
        channels_to_plot=list(range(min(N_EEG_TRACES, eeg.shape[0]))),
        channel_names=[f"EEG {i}" for i in range(eeg.shape[0])],
        max_freq=50.0,
        nperseg=nperseg,
    )
    if fig_freq.axes:
        fig_freq.axes[0].set_title(f"EEG Power Spectrum — {args.config.stem}")

    # Save both figures and time-series data in a single subdirectory
    saver.save(
        {"signals": fig_signals, "spectrum": fig_freq},
        name=args.plot_path.stem,
        metadata=metadata,
        data={"eeg": eeg},
        overwrite=True,
    )
    plt.close(fig_signals)
    plt.close(fig_freq)

    print(f"Saved regime plots and data folder to {args.plot_path.parent / args.plot_path.stem}")


if __name__ == "__main__":
    main()
