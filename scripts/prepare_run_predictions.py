import argparse
from pathlib import Path

from simulate.config import load_config

from neuro.predictor.replay import prepare_replay
from neuro.run_view import Run, discover_runs


def main() -> None:
    """Prepare actual-input predictions offline for each selected run and Predictor config."""
    parser = argparse.ArgumentParser(
        description="Replay Predictors on saved decision measurements; never rerun the Plant."
    )
    parser.add_argument("directory", type=Path, help="Run or collection directory.")
    parser.add_argument(
        "--config",
        type=Path,
        action="append",
        help="Candidate simulation config; repeat for multiple Predictors. Defaults to each run's own Predictor.",
    )
    args = parser.parse_args()
    directories = discover_runs(args.directory)
    if not directories:
        parser.error("No runs with config.yaml and log.npz found.")
    for directory in directories:
        run = Run.load(directory)
        configs = [(path.stem, load_config(path)) for path in args.config] if args.config else [("own", run.config)]
        for name, config in configs:
            destination = directory / "replays" / f"{name}.npz"
            print(f"Preparing {destination}", flush=True)
            prepare_replay(run, config, destination)


if __name__ == "__main__":
    main()
