import argparse
import shutil
from pathlib import Path

from neuro.config import load_config, resolve_artifact_dir, resolve_data_files
from neuro.predictor.evaluation import free_run_stats
from neuro.predictor.plotting import plot_rollout_comparison, plot_training_curves
from neuro.predictor.train import TrainingResult, train


def main() -> None:
    """Execute the main script."""
    parser = argparse.ArgumentParser(description="Run torch NN Predictor on multiple trajectories.")
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

    result = train(config, data_files)
    result.save(artifact_dir)
    plot_training_curves(result, artifact_dir / "loss_curve.png")
    plot_rollout_comparison(result, artifact_dir / "comparison.png")

    scores = free_run_stats(result.free_run, result.log_energy)
    summary = ", ".join(f"{k}: {v:.4f}" for k, v in scores.items() if isinstance(v, float))
    # A closed-form fit has no epoch loop, so du sensitivity is only measured on the gradient arm.
    if isinstance(result, TrainingResult):
        summary += f", du sensitivity: {result.du_sensitivity:.4f}"
    print(summary)
    print(f"Saved NN predictor checkpoint -> {artifact_dir / 'model.npz'}")
    print(f"Plot saved to {artifact_dir / 'comparison.png'}")


if __name__ == "__main__":
    main()
