"""Network 2-D state-space map of the Jansen-Rit plant (Chouzouris Fig. 3 analog).

Sweeps the background input mu and the global coupling strength sigma over a grid,
simulating the full 76-node network at each operating point and classifying it by
synchronization, dominant frequency, and oscillation amplitude. Grid and run
lengths come from ``scripts/configs/state_map.yaml``.

WARNING: one full network simulation runs per grid cell -- this is the expensive
script. Keep the grid coarse (see the config) unless you need detail.

Usage
-----
    uv run python scripts/run_state_map.py
    uv run python scripts/run_state_map.py --config scripts/configs/state_map.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from simulate.config import load_config

from neuro.sweep import network_state_map
from utils.plotting import plot_state_map
from utils.save_plots import ThesisPlotSaver

DEFAULT_CONFIG = Path("scripts/configs/state_map.yaml")
DEFAULT_OUTPUT_DIR = Path("artifacts")

# Standard Jansen-Rit parameters held fixed; mu is overwritten per grid column.
BASE_PARAMS: dict[str, list[float]] = {
    "A": [3.25],
    "B": [22.0],
    "a": [0.1],
    "b": [0.05],
    "v0": [5.52],
    "nu_max": [0.0025],
    "r": [0.56],
    "J": [135.0],
    "a_1": [1.0],
    "a_2": [0.8],
    "a_3": [0.25],
    "a_4": [0.25],
    "mu": [0.22],
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the state-map runner."""
    parser = argparse.ArgumentParser(description="Network 2-D (mu x sigma) Jansen-Rit state-space map.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="State-map grid YAML.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for the PNGs.")
    return parser.parse_args()


def _linspace(block: dict[str, float]) -> np.ndarray:
    return np.linspace(float(block["start"]), float(block["stop"]), int(block["num"]))


def main() -> None:
    """Run the (mu x sigma) sweep and save synchronization / frequency maps."""
    args = parse_args()
    cfg = load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mu_values = _linspace(cfg["mu"])
    sigma_values = _linspace(cfg["sigma"])
    print(f"State map: {len(mu_values)} mu x {len(sigma_values)} sigma = {mu_values.size * sigma_values.size} cells...")

    result = network_state_map(
        mu_values.tolist(),
        sigma_values.tolist(),
        base_params=BASE_PARAMS,
        duration_ms=float(cfg["duration_ms"]),
        transient_ms=float(cfg["transient_ms"]),
        dt=float(cfg["dt"]),
        nsig=float(cfg.get("nsig", 0.0)),
    )

    panels = [
        ("sync", "Synchronization R", "viridis"),
        ("freq", "Dominant frequency (Hz)", "magma"),
        ("amplitude", "Median amplitude (mV)", "cividis"),
    ]
    saver = ThesisPlotSaver(base_dir=str(args.output_dir))

    for key, label, cmap in panels:
        fig, _ = plot_state_map(
            result["mu"],
            result["sigma"],
            result[key],
            metric_label=label,
            title=f"Network state map — {label}",
            cmap=cmap,
        )
        metadata = {
            "metric": key,
            "metric_label": label,
            "config_file": str(args.config),
        }
        plot_name = f"state_map_{key}"
        saver.save(fig, name=plot_name, metadata=metadata, overwrite=True)
        plt.close(fig)
        print(f"  saved state map plot folder to {args.output_dir / plot_name}")


if __name__ == "__main__":
    main()
