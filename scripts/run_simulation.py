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

import yaml
from simulate.config import load_config
from simulate.simulation import Simulation

DEFAULT_CONFIG = Path("scripts/configs/jansen_rit_baseline.yaml")
OUTPUT_DIR = Path("simulations/")


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
        "--name",
        type=str,
        help="Filename prefix for the merged .npz.",
    )
    return parser.parse_args()


def main() -> None:
    """Load the config, run the orchestrated simulation, and export results."""
    args = parse_args()

    config = load_config(args.config)
    if args.duration_ms is not None:
        config["t_end"] = args.duration_ms / 1000.0  # CLI is ms; config t_end is in seconds
    dt_s = float(config["dynamics"]["dt"])
    dt_ms = dt_s * 1000.0

    sim = Simulation.from_config(config)

    t_end_s = float(config["t_end"])
    print(
        f"Running {config['dynamics']['class_path']} for {t_end_s} s "
        f"at dt={dt_ms} ms -> {round(t_end_s / dt_s) + 1} steps...",
    )

    name = args.name or args.config.stem
    run_dir = OUTPUT_DIR / name

    sim.run(run_dir, prefix="logs")
    sim.export_results(run_dir, prefix="logs")

    # Save the config dictionary as YAML in the simulation directory
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False)

    npz_path = run_dir / "logs.npz"
    print(f"Wrote {npz_path}")
    print(f"Saved configuration to {config_path}")


if __name__ == "__main__":
    main()
