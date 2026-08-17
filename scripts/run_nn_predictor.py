import argparse
import shutil
from pathlib import Path

from neuro.config import load_config, resolve_artifact_dir, resolve_data_files
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
    try:
        config = load_config(config_path)
        data_files = resolve_data_files(config, args.data_path)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        return

    artifact_dir = resolve_artifact_dir(config.artifact, "nn_predictor")
    shutil.copy2(config_path, artifact_dir / config_path.name)

    train_and_save_predictor(config, data_files, artifact_dir)

    print(f"Saved NN predictor artifact -> {artifact_dir / 'model.npz'}")
    print(f"Plot saved to {artifact_dir / 'comparison.png'}")


if __name__ == "__main__":
    main()
