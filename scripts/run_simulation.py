"""Run a whole-brain simulation through the ``simulate`` orchestrator.

Loads a YAML config describing all six components (dynamics, output, reference,
sensor, estimator, controller), runs :class:`~simulate.simulation.Simulation` to
``t_end``, and lets the framework's :class:`~simulate.logger.Logger` write every
signal to ``{output_dir}/{prefix}.npz``. Finally renders the logged EEG output.

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

from neuro.plotting import plot_fourier, plot_signals

FloatArray = npt.NDArray[np.float64]

DEFAULT_CONFIG = Path("scripts/configs/jansen_rit_baseline.yaml")
DEFAULT_OUTPUT_DIR = Path("artifacts/sim")
DEFAULT_PREFIX = "jansen_rit_baseline"
DEFAULT_PLOT_PATH = Path("artifacts/jansen_rit_baseline.png")
N_CHANNELS_TO_PLOT = 8


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the orchestrated runner."""
    parser = argparse.ArgumentParser(description="Run a simulation via the simulate orchestrator.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML config for the Simulation (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--duration-ms",
        type=float,
        default=None,
        help="Override the config's t_end (milliseconds).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for the logged .npz output (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=DEFAULT_PREFIX,
        help=f"Filename prefix for the merged .npz (default: {DEFAULT_PREFIX}).",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=DEFAULT_PLOT_PATH,
        help=f"PNG output path for the diagnostic EEG plot (default: {DEFAULT_PLOT_PATH}).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting entirely.",
    )
    return parser.parse_args()


def plot_eeg(eeg: FloatArray, dt_ms: float, path: Path) -> None:
    """Render EEG channel traces and a power spectrum from the logged output."""
    n_shown = min(N_CHANNELS_TO_PLOT, eeg.shape[0])
    channels = list(range(n_shown))

    fig, (ax_trace, ax_freq) = plt.subplots(2, 1, figsize=(10, 8), layout="constrained")
    plot_signals(
        eeg,
        dt_ms=dt_ms,
        channels_to_plot=channels,
        channel_names=[f"EEG {i}" for i in range(eeg.shape[0])],
        title=f"EEG output (first {n_shown} of {eeg.shape[0]} channels)",
        color="#ff7f0e",
        ax=ax_trace,
    )
    plot_fourier(
        eeg,
        dt_ms=dt_ms,
        mode="power",
        channels_to_plot=channels,
        channel_names=[f"EEG {i}" for i in range(eeg.shape[0])],
        max_freq=50.0,
        ax=ax_freq,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
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

    sim.run(args.output_dir, prefix=args.prefix)
    sim.export_results(args.output_dir, prefix=args.prefix)

    npz_path = args.output_dir / f"{args.prefix}.npz"
    with np.load(npz_path) as data:
        eeg: FloatArray = data["universal_y"].T  # (n_sensors, n_samples)
        u: FloatArray = data["universal_u"]

    print(f"Wrote {npz_path}")
    print(
        f"  EEG shape={eeg.shape}  mean={eeg.mean():+.4e}  std={eeg.std():.4e}  finite={bool(np.isfinite(eeg).all())}",
    )
    print(f"  control |u| max={np.abs(u).max():.3e} (open loop -> expected 0)")

    if not args.no_plot:
        plot_eeg(eeg, dt_ms, args.plot_path)
        print(f"Saved EEG plot to {args.plot_path}")


if __name__ == "__main__":
    main()
