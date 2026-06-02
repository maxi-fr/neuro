"""Run an uncontrolled baseline simulation of the FHN whole-brain plant.

Instantiates :class:`NativeFHNDynamics` + :class:`FHNOutput`, advances them
step-by-step for ~1 s of simulated time using the ``evaluate`` API, and reports
summary statistics on the EEG projection. Also renders diagnostic EEG plots so the
simulation can be inspected visually. Useful as a smoke test for the plant and
as a worked example of the step-by-step loop.

Usage
-----
    uv run python scripts/run_baseline.py
    uv run python scripts/run_baseline.py --duration-ms 2000 --save out.npz
    uv run python scripts/run_baseline.py --plot-path artifacts/baseline.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import matplotlib as mpl

mpl.use("Agg")  # non-interactive backend so the script works headless

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from neuro.plant import FHNOutput, NativeFHNDynamics
from utils.plotting import plot_heatmap, plot_signals
from utils.save_plots import ThesisPlotSaver

FloatArray = npt.NDArray[np.float64]

DEFAULT_PLOT_PATH = Path("artifacts/baseline_eeg.png")
N_CHANNELS_TO_TRACE = 8


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the baseline runner."""
    parser = argparse.ArgumentParser(description="Run uncontrolled FHN plant baseline.")
    parser.add_argument(
        "--duration-ms",
        type=float,
        default=1000.0,
        help="Total simulated time in milliseconds (default: 1000).",
    )
    parser.add_argument(
        "--step-ms",
        type=float,
        default=0.1,
        help="Integration step (dt) in milliseconds (default: 0.1).",
    )
    parser.add_argument(
        "--sigma-ou",
        type=float,
        default=0.05,
        help="OU noise amplitude for the FHN dynamics (default: 0.05).",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to an .npz file in which to save activity and EEG.",
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


def summarise(name: str, array: FloatArray) -> None:
    """Print shape and basic statistics for an output array."""
    print(
        f"  {name:<8s} shape={array.shape}  "
        f"mean={array.mean():+.4f}  std={array.std():.4f}  "
        f"min={array.min():+.4f}  max={array.max():+.4f}  "
        f"finite={bool(np.isfinite(array).all())}",
    )


def main() -> None:
    """Run the baseline simulation and report aggregated statistics."""
    args = parse_args()
    n_steps = round(args.duration_ms / args.step_ms)

    dynamics = NativeFHNDynamics(dt=args.step_ms, sigma_ou=args.sigma_ou)
    output = FHNOutput(dt=args.step_ms, n_nodes=dynamics.n_nodes)
    print(
        f"NativeFHNDynamics: n_nodes={dynamics.n_nodes}  dt={dynamics.dt} ms  "
        f"sigma_ou={dynamics.sigma_ou}  leadfield={output.leadfield.shape}",
    )
    print(f"Running {n_steps} steps of {args.step_ms} ms ({args.duration_ms} ms total)...")

    eeg_history: list[FloatArray] = []
    t = 0.0
    u = np.zeros((1, 1))
    for _ in range(n_steps):
        x_raw, _ = dynamics.evaluate(t, u)
        eeg_raw, _ = output.evaluate(t, x_raw, u)

        eeg = cast("FloatArray", eeg_raw)

        eeg_history.append(eeg)
        t += args.step_ms

    eeg_full: FloatArray = np.column_stack(eeg_history)

    print("Aggregated outputs:")
    summarise("eeg", eeg_full)

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.save, eeg=eeg_full, leadfield=output.leadfield)
        print(f"Saved outputs to {args.save}")

    if not args.no_plot:
        metadata = {
            "duration_ms": args.duration_ms,
            "step_ms": args.step_ms,
            "sigma_ou": args.sigma_ou,
        }
        saver = ThesisPlotSaver(base_dir=str(args.plot_path.parent))

        # 1. Plot signals (traces)
        n_sensors = eeg_full.shape[0]
        n_shown = min(N_CHANNELS_TO_TRACE, n_sensors)
        fig_signals, _ = plot_signals(
            eeg_full,
            dt_ms=dynamics.dt,
            channels_to_plot=list(range(n_shown)),
            channel_names=[f"ch{i}" for i in range(n_sensors)],
            title=f"EEG channel traces (first {n_shown} of {n_sensors})",
        )

        # 2. Plot heatmap
        fig_heat, _ = plot_heatmap(
            eeg_full,
            dt_ms=dynamics.dt,
            title=f"EEG heatmap (all {n_sensors} channels)",
        )

        # Save both figures and time-series data in a single subdirectory
        saver.save(
            {"signals": fig_signals, "heatmap": fig_heat},
            name=args.plot_path.stem,
            metadata=metadata,
            data={"eeg": eeg_full},
            overwrite=True,
        )
        plt.close(fig_signals)
        plt.close(fig_heat)

        print(f"Saved EEG plots and data folder to {args.plot_path.parent / args.plot_path.stem}")


if __name__ == "__main__":
    main()
