"""Run a whole-brain simulation through the ``simulate`` orchestrator.

Loads a YAML config describing all six components (dynamics, output, reference,
sensor, estimator, controller), runs :class:`~simulate.simulation.Simulation` to
``t_end``, and lets the framework's :class:`~simulate.logger.Logger` write every
signal to ``{output_dir}/{name}/logs.npz``. Finally renders the logged EEG output.

The shipped default config (``scripts/configs/jansen_rit_baseline.yaml``) is an
open-loop Jansen-Rit baseline -- the ``ZeroController`` injects no control, so the
plant evolves freely. This is the orchestrated counterpart to the manual loop in
``run_baseline.py``.

Usage
-----
    uv run python scripts/run_simulation.py
    uv run python scripts/run_simulation.py --duration-ms 2000
    uv run python scripts/run_simulation.py --config scripts/configs/my_run.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")  # non-interactive backend so the script works headless

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from simulate.config import load_config
from simulate.simulation import Simulation

from utils.plotting import plot_psd, plot_signals
from utils.processing import steady_window, synchronization
from utils.save_plots import ThesisPlotSaver

FloatArray = npt.NDArray[np.float64]


DEFAULT_CONFIG = Path("scripts/configs/jansen_rit_baseline.yaml")
N_CHANNELS_TO_PLOT = 8
OUTPUT_DIR = Path("simulations/")
PSD_NPERSEG = 8192  # High resolution spectral segment length for EEG bands


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the orchestrated runner."""
    parser = argparse.ArgumentParser(description="Run a simulation via the simulate orchestrator.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="YAML config for the Simulation.",
    )
    parser.add_argument(
        "--duration-ms",
        type=float,
        default=None,
        help="Override the config's t_end (milliseconds).",
    )
    parser.add_argument(
        "--transient-ms",
        type=float,
        default=0.0,
        help="Leading transient discarded before metrics and plots (milliseconds).",
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Filename prefix for the merged .npz.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting entirely.",
    )
    return parser.parse_args()


def main() -> None:  # noqa: PLR0915
    """Load the config, run the orchestrated simulation, and report results."""
    args = parse_args()

    config = load_config(args.config)
    if args.duration_ms is not None:
        config["t_end"] = args.duration_ms
    dt_ms = float(config["dynamics"]["dt"])

    sim = Simulation.from_config(config)

    print(
        f"Running {config['dynamics']['class_path']} for {config['t_end']} ms "
        f"at dt={dt_ms} ms -> {round(float(config['t_end']) / dt_ms) + 1} steps...",
    )

    name = args.name or args.config.stem
    run_dir = OUTPUT_DIR / name

    sim.run(run_dir, prefix="logs")
    sim.export_results(run_dir, prefix="logs")

    npz_path = run_dir / "logs.npz"
    with np.load(npz_path) as data:
        eeg: FloatArray = data["universal_y"].T  # (n_sensors, n_samples)
        u: FloatArray = data["universal_u"]

        # Calculate network synchronization from logs if Jansen-Rit dynamics are used
        r_sync = None
        is_jansen_rit = "JansenRit" in config["dynamics"]["class_path"]
        if is_jansen_rit and "universal_x" in data:
            try:
                x_flat = data["universal_x"].squeeze()
                if x_flat.ndim == 1:
                    x_flat = x_flat[np.newaxis, :]
                n_samples_logged = x_flat.shape[0]
                x_grid = x_flat.reshape((n_samples_logged, 6, -1))
                activity = (x_grid[:, 1, :] - x_grid[:, 2, :]).T  # (n_nodes, n_samples)

                # Discard transient for synchronization metric if requested
                if args.transient_ms > 0.0:
                    activity = steady_window(activity, dt_ms, args.transient_ms)

                r_sync = synchronization(activity)
            except Exception as e:  # noqa: BLE001
                print(f"Could not calculate network synchronization: {e}")

    # Drop transient window if requested
    if args.transient_ms > 0.0:
        print(f"Discarding leading {args.transient_ms} ms transient for metrics and plotting...")
        eeg = steady_window(eeg, dt_ms, args.transient_ms)
        n_drop = round(args.transient_ms / dt_ms)
        u = u[n_drop:]

    print(f"Wrote {npz_path}")
    print(
        f"  EEG shape={eeg.shape}  mean={eeg.mean():+.4e}  std={eeg.std():.4e}  finite={bool(np.isfinite(eeg).all())}",
    )
    if r_sync is not None:
        print(f"  network synchronization R = {r_sync:.3f}")
    print(f"  control |u| max={np.abs(u).max():.3e} (open loop -> expected 0)")

    if not args.no_plot:
        metadata = {
            "config": str(args.config),
            "output_dir": str(OUTPUT_DIR),
            "name": name,
            "transient_ms": args.transient_ms,
        }
        if r_sync is not None:
            metadata["synchronization_R"] = float(r_sync)

        saver = ThesisPlotSaver(base_dir=str(OUTPUT_DIR))
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

        # Save both figures and time-series data in a single subdirectory
        saver.save(
            {"signals": fig_signals, "spectrum": fig_freq},
            name=name,
            metadata=metadata,
            data={"eeg": eeg, "u": u},
            overwrite=True,
        )
        plt.close(fig_signals)
        plt.close(fig_freq)

        print(f"Saved EEG plots and data folder to {run_dir}")


if __name__ == "__main__":
    main()
