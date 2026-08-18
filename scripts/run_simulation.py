import argparse
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from simulate.config import deep_merge, load_config
from simulate.experiment import ExperimentManager
from simulate.simulation import Simulation

from neuro.validation import validate_simulation_config


def _redundant_keys(data: dict[str, np.ndarray]) -> set[str]:
    """Name the logged channels that carry no information.

    A noiseless sensor's 'truth' and 'noise' -- a duplicate of 'y_mea' and
    zeros, together most of an excitation trajectory's bytes.
    """
    drop: set[str] = set()
    for key, value in data.items():
        measured = data.get(key.replace(".truth", ".y_mea"))
        is_dead_noise = key.endswith(".noise") and not value.any()
        is_dup_truth = key.endswith(".truth") and measured is not None and np.array_equal(value, measured)
        if is_dead_noise or is_dup_truth:
            drop.add(key)
    return drop


def main() -> None:
    """Execute the main entry point for the simulation CLI."""
    parser = argparse.ArgumentParser(description="Modular Python Framework for Control System Simulation")
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the YAML configuration file.",
    )

    parser.add_argument(
        "--mmap",
        action="store_true",
        help="Enable memory-mapped logging to bound RAM usage for long runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save simulation results (default: simulation_<current_datetime>).",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Enable zlib compression for output files (default: False/uncompressed).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Processes to run an `experiments:` batch across (default: 1).",
    )

    args = parser.parse_args()

    config_path = Path(args.config_file)
    if not config_path.exists():
        sys.exit(1)

    config = load_config(config_path)

    if args.output_dir is None:
        local_now = datetime.now(UTC).astimezone()
        output_dir = Path(f"artifacts/simulation_{local_now.strftime('%Y-%m-%d_%H-%M-%S')}")
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output_dir / config_path.name)

    if "experiments" in config:
        manager = ExperimentManager(output_dir)

        raw_configs = config["experiments"]
        if not raw_configs:
            sys.exit(0)

        configs = [raw_configs[0]]
        for override in raw_configs[1:]:
            configs.append(deep_merge(configs[-1], override))
        for merged in configs:
            validate_simulation_config(merged)

        manager.run_batch(configs, max_num_processes=args.workers, use_mmap=args.mmap, compress=args.compress)
    else:
        validate_simulation_config(config)
        sim = Simulation.from_config(config)
        sim.run(output_dir, prefix="log", use_mmap=args.mmap)
        sim.export_results(output_dir, prefix="log", compress=args.compress)

    for npz_path in output_dir.rglob("*.npz"):
        data = dict(np.load(npz_path))
        drop = _redundant_keys(data)
        if drop:
            for key in drop:
                del data[key]
            save = np.savez_compressed if args.compress else np.savez
            save(npz_path, **data)


if __name__ == "__main__":
    main()
