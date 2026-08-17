import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from neuro.config import load_config, resolve_artifact_dir, resolve_data_files
from neuro.predictor.train import TrainingResult, train
from utils.plotting import plot_multistep_predictions

MAX_PLOT_ANCHORS = 200
MAX_PLOT_CHANNELS = 4


def plot_training_curves(result: TrainingResult, plot_path: Path) -> None:
    """Plot training and validation loss curves with per-loss components."""
    plt.figure(figsize=(8, 5))
    plt.plot(result.train_losses, label="Train Total", linewidth=2.0)
    plt.plot(result.val_losses, label="Val Total", linewidth=2.0)

    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, key in enumerate(result.train_components):
        color = prop_cycle[(i + 2) % len(prop_cycle)]
        plt.plot(
            result.train_components[key],
            label=f"Train {key}",
            linewidth=1.0,
            linestyle="--",
            alpha=0.7,
            color=color,
        )
        if key in result.val_components:
            plt.plot(
                result.val_components[key],
                label=f"Val {key}",
                linewidth=1.0,
                linestyle=":",
                alpha=0.7,
                color=color,
            )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(visible=True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()


def plot_rollout_comparison(result: TrainingResult, plot_path: Path) -> None:
    """Overlay free-run rollout fans on the first held-out trajectory."""
    art = result.artifact
    u, y = result.val_trajs[0]
    priming = art.priming_steps
    n_anchors = min(MAX_PLOT_ANCHORS, len(y) - priming - art.horizon)

    # The rollout primed on history up to t - 1 predicts y[t : t + horizon], so its anchor is t - 1.
    y_pred = np.stack(
        [
            art.rollout(art.prime(y[t - priming : t], u[t - priming : t]), u[t : t + art.horizon])
            for t in range(priming, priming + n_anchors)
        ]
    )

    fig, _ = plot_multistep_predictions(
        y_true=y[priming - 1 : priming - 1 + n_anchors],
        y_pred=y_pred,
        dt=art.dt,
        channels=list(range(min(MAX_PLOT_CHANNELS, art.n_channels))),
        stride=art.horizon,
        title=f"EEG {art.horizon}-Step Free-Run Rollout",
    )
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)


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

    print(
        f"Rollout NMSE: {result.rollout.pooled:.4f}, log-energy error: {result.log_energy.pooled:.4f}, "
        f"du sensitivity: {result.du_sensitivity:.4f}"
    )
    print(f"Saved NN predictor artifact -> {artifact_dir / 'model.npz'}")
    print(f"Plot saved to {artifact_dir / 'comparison.png'}")


if __name__ == "__main__":
    main()
