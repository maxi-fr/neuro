import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from simulate.config import deep_merge, load_config
from simulate.experiment import ExperimentManager
from simulate.simulation import Simulation


def main() -> None:
    """Execute the main entry point for the simulation CLI."""
    parser = argparse.ArgumentParser(description="Modular Python Framework for Control System Simulation")
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10_000,
        help="Steps per output chunk file (default: 10000). Use 0 to disable chunking.",
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

    args = parser.parse_args()

    config_path = Path(args.config_file)
    if not config_path.exists():
        sys.exit(1)

    config = load_config(config_path)

    chunk_size = args.chunk_size if args.chunk_size > 0 else None

    if args.output_dir is None:
        local_now = datetime.now(UTC).astimezone()
        output_dir_str = f"results/simulation_{local_now.strftime('%Y-%m-%d_%H-%M-%S')}"
    else:
        output_dir_str = args.output_dir

    if "experiments" in config:
        manager = ExperimentManager(output_dir=output_dir_str)

        raw_configs = config["experiments"]
        if not raw_configs:
            sys.exit(0)

        configs = [raw_configs[0]]
        for override in raw_configs[1:]:
            configs.append(deep_merge(configs[-1], override))

        manager.run_batch(configs, chunk_size=chunk_size, compress=args.compress)
    else:
        output_dir = Path(output_dir_str)
        sim = Simulation.from_config(config)
        sim.run(output_dir=output_dir, prefix="log", chunk_size=chunk_size, compress=args.compress)
        sim.export_results(output_dir, prefix="log", compress=args.compress)


if __name__ == "__main__":
    main()
