"""Run an open-loop Jansen-Rit regime and save EEG / activity / spectrum plots.

Reads a regime config (the ``dynamics`` block of an orchestrator-style YAML such
as ``scripts/configs/jansen_rit_healthy.yaml``), drives the plant with no control
via :func:`neuro.sweep.run_open_loop`, and renders:

* a 2x2 dashboard (node activity + EEG, each with its Welch power spectrum), and
* a stacked EEG-trace figure.

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
from utils.plotting import plot_dashboard, plot_signals
from utils.processing import dominant_frequency, steady_window, synchronization

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
        help="PNG path for the dashboard; the EEG-trace figure is saved alongside as *_eeg.png.",
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

    nperseg = min(PSD_NPERSEG, eeg.shape[1])
    fig = plot_dashboard(
        activity,
        eeg,
        dt_ms=dt_ms,
        nperseg=nperseg,
        title=f"{args.config.stem}  (R={r_sync:.2f}, f_dom={f_dom:.1f} Hz)",
    )
    args.plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot_path, dpi=150)
    plt.close(fig)

    eeg_path = args.plot_path.with_name(f"{args.plot_path.stem}_eeg{args.plot_path.suffix}")
    fig_eeg, _ = plot_signals(
        eeg,
        dt_ms=dt_ms,
        channels_to_plot=list(range(min(N_EEG_TRACES, eeg.shape[0]))),
        channel_names=[f"EEG {i}" for i in range(eeg.shape[0])],
        title=f"EEG output — {args.config.stem}",
        color="#ff7f0e",
    )
    fig_eeg.savefig(eeg_path, dpi=150)
    plt.close(fig_eeg)
    print(f"Saved {args.plot_path} and {eeg_path}")


if __name__ == "__main__":
    main()
