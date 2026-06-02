"""Single-node Jansen-Rit bifurcation diagrams (vs background input mu and gain A).

Decouples the network (``coupling_strength = 0``) and sweeps a model parameter
deterministically, recording the steady-state activity envelope (min/max) and the
dominant oscillation frequency at each value -- the classic Grimbert-Faugeras /
Chouzouris Fig. 1 diagram. Ranges come from ``scripts/configs/bifurcation.yaml``.

Usage
-----
    uv run python scripts/run_bifurcation.py
    uv run python scripts/run_bifurcation.py --config scripts/configs/bifurcation.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from simulate.config import load_config

from neuro.sweep import sweep_1d
from utils.plotting import plot_bifurcation_1d

DEFAULT_CONFIG = Path("scripts/configs/bifurcation.yaml")
DEFAULT_OUTPUT_DIR = Path("artifacts")

# Standard Jansen-Rit parameters held fixed; the swept parameter overrides its entry.
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
    """Parse command-line arguments for the bifurcation runner."""
    parser = argparse.ArgumentParser(description="Single-node Jansen-Rit bifurcation diagrams.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Bifurcation sweep YAML.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for the PNGs.")
    return parser.parse_args()


def _linspace(block: dict[str, float]) -> np.ndarray:
    return np.linspace(float(block["start"]), float(block["stop"]), int(block["num"]))


def _run_one(param: str, label: str, values: np.ndarray, cfg: dict, output_dir: Path) -> None:
    print(f"Sweeping {param} over {len(values)} values [{values[0]:.3g}, {values[-1]:.3g}]...")
    result = sweep_1d(
        param,
        values.tolist(),
        base_params=BASE_PARAMS,
        duration_ms=float(cfg["duration_ms"]),
        transient_ms=float(cfg["transient_ms"]),
        dt=float(cfg["dt"]),
    )
    fig, _ = plot_bifurcation_1d(
        result["values"],
        result["min"],
        result["max"],
        result["freq"],
        param_label=label,
        title=f"Single-node Jansen-Rit bifurcation vs {label}",
    )
    path = output_dir / f"bifurcation_{param}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def main() -> None:
    """Run the mu and A bifurcation sweeps and save the diagrams."""
    args = parse_args()
    cfg = load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _run_one("mu", r"background input $\mu$", _linspace(cfg["mu"]), cfg, args.output_dir)
    _run_one("A", "excitatory gain $A$ (mV)", _linspace(cfg["A"]), cfg, args.output_dir)


if __name__ == "__main__":
    main()
