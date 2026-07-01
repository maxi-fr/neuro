"""Script to train a JAX-based Neural Network to predict EEG signals (Multi-step predictor).

This script implements an MLP using Equinox and Optax to predict a horizon of N steps
of EEG data from past EEG and past/future stimulation inputs. It supports training on
multiple trajectories.
"""

import argparse
import datetime
import shutil
from pathlib import Path
from typing import Any

import yaml

from neuro.nn_training import train_and_save_predictor


def main() -> None:
    """Execute the main script."""
    parser = argparse.ArgumentParser(description="Run JAX NN Predictor on multiple trajectories.")
    parser.add_argument(
        "--config", type=str, default="configs/nn_predictor/nn_predictor_config.yaml", help="Path to config YAML."
    )
    parser.add_argument("--data-path", type=str, help="Override config data path.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return

    with config_path.open() as f:
        config: dict[str, Any] = yaml.safe_load(f)

    sim_cfg = config.get("simulation", {})
    data_path_str = args.data_path or sim_cfg.get("data_path")
    if not data_path_str:
        print("data_path not specified in config or arguments.")
        return

    data_path = Path(data_path_str)
    if not data_path.is_dir():
        print(f"data_path is not a valid directory: {data_path}")
        return

    data_files = sorted([str(p) for p in data_path.glob("*.npz")])
    if not data_files:
        print(f"No .npz data files found in: {data_path}")
        return

    artifact_dir = config.get("artifact")
    if artifact_dir is None:
        local_now = datetime.datetime.now(datetime.UTC).astimezone()
        artifact_dir = f"artifacts/nn_predictor_{local_now.strftime('%Y-%m-%d_%H-%M-%S')}"
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, artifact_dir / config_path.name)

    train_and_save_predictor(config, data_files, artifact_dir)

    print(f"Saved NN predictor artifact -> {artifact_dir / 'model.eqx'}")
    print(f"Plot saved to {artifact_dir / 'comparison.png'}")


if __name__ == "__main__":
    main()
