"""Run an uncontrolled baseline simulation of the FHN whole-brain plant.

Instantiates :class:`FHNPlant`, advances it step-by-step for ~1 s of simulated
time using the closed-loop-style API, and reports summary statistics on the
node activity and EEG projection. Also renders diagnostic EEG plots so the
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

import matplotlib as mpl

mpl.use("Agg")  # non-interactive backend so the script works headless

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from neuro.plant import FHNPlant

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
        default=10.0,
        help="Simulation chunk per step() call in milliseconds (default: 10).",
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


def plot_eeg(eeg: FloatArray, dt_ms: float, path: Path) -> None:
    """Render diagnostic EEG plots: stacked channel traces and a heatmap.

    Parameters
    ----------
    eeg
        EEG array of shape ``(n_sensors, n_samples)``.
    dt_ms
        Sampling period in milliseconds, used to build the time axis.
    path
        Destination PNG path. Parent directories are created as needed.
    """
    n_sensors, n_samples = eeg.shape
    t_s = np.arange(n_samples) * dt_ms / 1000.0  # seconds

    fig, (ax_trace, ax_heat) = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        gridspec_kw={"height_ratios": [2, 1]},
    )

    # Stacked traces: first N_CHANNELS_TO_TRACE channels, vertically offset.
    n_shown = min(N_CHANNELS_TO_TRACE, n_sensors)
    offset = 4.0 * float(eeg.std())
    for i in range(n_shown):
        ax_trace.plot(t_s, eeg[i] + i * offset, linewidth=0.7)
    ax_trace.set_title(f"EEG channel traces (first {n_shown} of {n_sensors})")
    ax_trace.set_xlabel("time (s)")
    ax_trace.set_ylabel("channel (offset)")
    ax_trace.set_yticks([i * offset for i in range(n_shown)])
    ax_trace.set_yticklabels([f"ch{i}" for i in range(n_shown)])
    ax_trace.set_xlim(t_s[0], t_s[-1])

    # Heatmap of all sensors over time.
    vlim = float(np.max(np.abs(eeg)))
    image = ax_heat.imshow(
        eeg,
        aspect="auto",
        origin="lower",
        extent=(float(t_s[0]), float(t_s[-1]), 0.0, float(n_sensors)),
        cmap="RdBu_r",
        vmin=-vlim,
        vmax=vlim,
    )
    ax_heat.set_title(f"EEG heatmap (all {n_sensors} channels)")
    ax_heat.set_xlabel("time (s)")
    ax_heat.set_ylabel("channel")
    fig.colorbar(image, ax=ax_heat, label="amplitude")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run the baseline simulation and report aggregated statistics."""
    args = parse_args()
    n_steps = round(args.duration_ms / args.step_ms)

    plant = FHNPlant(sigma_ou=args.sigma_ou)
    print(
        f"FHNPlant: n_nodes={plant.n_nodes}  dt={plant.dt} ms  "
        f"sigma_ou={plant.sigma_ou}  leadfield={plant.leadfield.shape}",
    )
    print(f"Running {n_steps} steps of {args.step_ms} ms ({args.duration_ms} ms total)...")

    activity_chunks: list[FloatArray] = []
    eeg_chunks: list[FloatArray] = []
    for _ in range(n_steps):
        activity, eeg = plant.step(args.step_ms)
        activity_chunks.append(activity)
        eeg_chunks.append(eeg)

    activity_full: FloatArray = np.concatenate(activity_chunks, axis=1)
    eeg_full: FloatArray = np.concatenate(eeg_chunks, axis=1)

    print("Aggregated outputs:")
    summarise("activity", activity_full)
    summarise("eeg", eeg_full)

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.save, activity=activity_full, eeg=eeg_full, leadfield=plant.leadfield)
        print(f"Saved outputs to {args.save}")

    if not args.no_plot:
        plot_eeg(eeg_full, plant.dt, args.plot_path)
        print(f"Saved EEG plot to {args.plot_path}")


if __name__ == "__main__":
    main()
