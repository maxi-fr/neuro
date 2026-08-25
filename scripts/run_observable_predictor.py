import argparse
import shutil
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt

from neuro.config import load_config, resolve_artifact_dir, resolve_data_files
from neuro.predictor.observable_train import ObservableTrainingResult
from neuro.predictor.train import train


def plot_training_curves(result: ObservableTrainingResult, plot_path: Path) -> None:
    """Plot the train/validation MSE in standardized log-Observable space."""
    plt.figure(figsize=(8, 5))
    plt.plot(result.train_losses, label="Train", linewidth=2.0)
    plt.plot(result.val_losses, label="Validation", linewidth=2.0)
    plt.xlabel("Epoch")
    plt.ylabel("MSE (standardized log observable)")
    plt.title("Observable-space predictor training")
    plt.legend()
    plt.grid(visible=True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()


def main() -> None:
    """Train the observable-space predictor named by a config carrying an ``observable`` block."""
    parser = argparse.ArgumentParser(description="Train the observable-space (frame-grid) predictor.")
    parser.add_argument("--config", type=str, default="configs/nn_predictor/observable_stft.yaml")
    parser.add_argument("--data-path", type=str, help="Override config data path.")
    args = parser.parse_args()

    config_path = Path(args.config)
    try:
        config = load_config(config_path)
        data_files = resolve_data_files(config, args.data_path)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        return

    if config.observable is None:
        print(f"{config_path} has no 'observable' block; use scripts/run_nn_predictor.py instead.")
        return

    artifact_dir = resolve_artifact_dir(config.artifact, "observable_predictor")
    shutil.copy2(config_path, artifact_dir / config_path.name)

    result = cast("ObservableTrainingResult", train(config, data_files))
    result.save(artifact_dir)
    plot_training_curves(result, artifact_dir / "loss_curve.png")

    print(
        f"Held-out log-observable MSE: {result.val_log_mse:.4f}, "
        f"du sensitivity: {result.du_sensitivity:.4f}, "
        f"non-overlapping training targets: {result.n_independent_samples}"
    )
    print(f"Saved observable predictor checkpoint -> {artifact_dir / 'model.npz'}")


if __name__ == "__main__":
    main()
