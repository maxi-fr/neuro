from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from neuro.predictor.inference import ObservableMLPModel, WaveformMLPModel
from neuro.predictor.train import TrainingResult
from utils.plotting import plot_multistep_predictions

if TYPE_CHECKING:
    from neuro.predictor.inference import InferencePredictor
    from neuro.predictor.ridge import RidgeTrainingResult

MAX_PLOT_ANCHORS = 200
MAX_PLOT_CHANNELS = 4


def plot_training_curves(result: TrainingResult | RidgeTrainingResult, plot_path: Path | str) -> None:
    """Plot training and validation loss curves with per-loss components if available.

    Parameters
    ----------
    result : TrainingResult | RidgeTrainingResult
        Training result produced by :func:`~neuro.predictor.train.train`.
    plot_path : Path or str
        Path where the loss curve PNG is saved.
    """
    if not isinstance(result, TrainingResult) or not result.train_losses:
        return

    plot_path = Path(plot_path)
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
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=300)
    plt.close()


def plot_rollout_comparison(
    result: TrainingResult | RidgeTrainingResult,
    plot_path: Path | str,
    *,
    max_anchors: int = MAX_PLOT_ANCHORS,
    max_channels: int = MAX_PLOT_CHANNELS,
) -> None:
    """Overlay free-run rollout fans on the first held-out trajectory.

    Parameters
    ----------
    result : TrainingResult | RidgeTrainingResult
        Training result produced by :func:`~neuro.predictor.train.train`.
    plot_path : Path or str
        Path where the comparison PNG is saved.
    max_anchors : int, optional
        Maximum number of fan anchors to plot. Defaults to 200.
    max_channels : int, optional
        Maximum number of channels to display. Defaults to 4.
    """
    if not hasattr(result, "val_trajs") or not result.val_trajs or not hasattr(result, "predictor"):
        return

    model = result.predictor
    meta, arrays = model.to_checkpoint()
    inference: InferencePredictor
    if "geometry" in meta:
        inference = ObservableMLPModel.from_checkpoint(meta, arrays)
        is_observable = True
    else:
        inference = WaveformMLPModel.from_checkpoint(meta, arrays)
        is_observable = False

    u, y = result.val_trajs[0]
    priming = inference.priming_steps
    n_anchors = min(max_anchors, len(y) - priming - model.horizon)
    if n_anchors <= 0:
        return

    # The rollout primed on history up to t - 1 predicts y[t : t + horizon], so its anchor is t - 1.
    anchors = range(priming, priming + n_anchors)
    y_pred = np.asarray(
        inference.free_run(
            np.stack([y[t - priming : t] for t in anchors]),
            np.stack([u[t - priming : t] for t in anchors]),
            np.stack([u[t : t + model.horizon] for t in anchors]),
        )
    )

    prefix = "Observable" if is_observable else "EEG"
    n_out = model.n_outputs if is_observable else model.n_channels
    fig, _ = plot_multistep_predictions(
        y_true=y[priming - 1 : priming - 1 + n_anchors],
        y_pred=y_pred,
        dt=model.dt,
        channels=list(range(min(max_channels, n_out))),
        stride=model.horizon,
        title=f"{prefix} {model.horizon}-Step Free-Run Rollout",
    )
    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
