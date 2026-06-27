"""Generate a multi-trial ``experiments:`` YAML from a single base Simulation config.

Reads a base config (one Simulation: dynamics/reference/sensors/estimator/controller) and writes
a new YAML whose ``experiments:`` list expands it into ``--n-trials`` independent trials. Trial 0
is the full base; each later entry is a minimal override that bumps ``dynamics.seed`` (and, when
the controller is a :class:`~neuro.control.WaveformController`, ``controller.input_seed``) so that
``scripts/run_simulation.py`` deep-merges them into distinct runs executed by the ExperimentManager.

Usage
-----
    uv run python scripts/generate_experiment.py configs/simulation/jansen_rit_seizure_excited.yaml \
        --n-trials 20 --output configs/simulation/plant_excited_experiment.yaml
    uv run python scripts/run_simulation.py configs/simulation/plant_excited_experiment.yaml
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml
from simulate.config import load_config

_INPUT_SEED_OFFSET = 100_000


def build_experiments(  # noqa: PLR0913
    base: dict[str, Any],
    n_trials: int,
    seed_base: int,
    input_seed_offset: int,
    initial_state_file: Path | None = None,
    initial_state_index_base: int = 0,
) -> dict[str, Any]:
    """Expand a base config into an ``experiments`` list with a per-trial seed sweep.

    The tES-input RNG seed is offset from the plant-noise seed so trials can share a noise
    realisation while drawing distinct inputs; it is only emitted when the controller is a
    :class:`~neuro.control.WaveformController`.
    """
    controller = base.get("controller", {})
    is_waveform = str(controller.get("class_path", "")).endswith("WaveformController")

    first = copy.deepcopy(base)
    first.setdefault("dynamics", {})["seed"] = seed_base
    if initial_state_file is not None:
        first["dynamics"]["initial_state_file"] = str(initial_state_file)
        first["dynamics"]["initial_state_index"] = initial_state_index_base
    if is_waveform:
        first.setdefault("controller", {})["input_seed"] = seed_base + input_seed_offset

    experiments: list[dict[str, Any]] = [first]
    for idx in range(1, n_trials):
        override: dict[str, Any] = {"dynamics": {"seed": seed_base + idx}}
        if initial_state_file is not None:
            override["dynamics"]["initial_state_index"] = initial_state_index_base + idx
        if is_waveform:
            override["controller"] = {"input_seed": seed_base + idx + input_seed_offset}
        experiments.append(override)
    return {"experiments": experiments}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the experiment generator."""
    parser = argparse.ArgumentParser(description="Expand a base Simulation config into an experiments YAML.")
    parser.add_argument("base_config", type=Path, help="Base single-Simulation YAML config.")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of trials to generate.")
    parser.add_argument("--seed-base", type=int, default=69, help="Plant seed for trial 0; incremented per trial.")
    parser.add_argument(
        "--input-seed-offset",
        type=int,
        default=_INPUT_SEED_OFFSET,
        help="Offset added to the plant seed for the WaveformController input RNG.",
    )
    parser.add_argument(
        "--initial-state-file",
        type=Path,
        default=None,
        help="Optional path to .npz file containing burn-in states.",
    )
    parser.add_argument(
        "--initial-state-index-base",
        type=int,
        default=0,
        help="Index into the initial_state_file array for trial 0; incremented per trial.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Path for the generated experiments YAML.")
    return parser.parse_args()


def main() -> None:
    """Generate and write the experiments YAML."""
    args = parse_args()
    base = load_config(args.base_config)
    experiments = build_experiments(
        base,
        args.n_trials,
        args.seed_base,
        args.input_seed_offset,
        initial_state_file=args.initial_state_file,
        initial_state_index_base=args.initial_state_index_base,
    )
    with args.output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(experiments, handle, sort_keys=False, default_flow_style=False)
    print(f"Wrote {args.output} ({args.n_trials} trials from {args.base_config})")


if __name__ == "__main__":
    main()
