"""Script to train a JAX-based Neural Network to predict EEG signals (Multi-step predictor).

This script implements an MLP using Equinox and Optax to predict a horizon of N steps
of EEG data from past EEG and past/future stimulation inputs. It supports training on
multiple trajectories.
"""

import argparse
import datetime
import json
import shutil
from pathlib import Path
from typing import Any

import jax
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.preprocessing import RobustScaler, StandardScaler

from neuro.jansen_rit_jax import enable_x64
from neuro.nn_training import (
    create_model,
    evaluate_model,
    plot_training_curves,
    prepare_datasets,
    reshape_flat_to_channels,
    reshape_to_trajectory,
    scale_dataset,
    train_model,
)
from neuro.prediction import AutoregressivePredictor, MLPArtifact
from utils.plotting import plot_multistep_predictions


def main() -> None:  # noqa: PLR0915
    """Execute the main script."""
    enable_x64()
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
    downsample = sim_cfg.get("downsample", 1)
    n_steps_cfg = sim_cfg.get("n_steps", 2000)
    dt_base = sim_cfg.get("dt", 1e-4)
    dt_real = dt_base * downsample

    model_cfg = config.get("model", {})
    n_y = model_cfg.get("n_y", 5)
    n_u = model_cfg.get("n_u", 5)
    horizon = model_cfg.get("horizon", 5)
    hidden_size = int(model_cfg.get("hidden_size", 128))
    depth = model_cfg.get("depth", 2)
    activation = model_cfg.get("activation", "relu")

    train_cfg = config.get("training", {})
    epochs = train_cfg.get("epochs", 100)
    batch_size = train_cfg.get("batch_size", 128)
    learning_rate = float(train_cfg.get("learning_rate", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    train_split = float(train_cfg.get("train_split", 0.8))
    curriculum_decay_fraction = float(train_cfg.get("curriculum_decay_fraction", 0.8))
    seed = int(train_cfg.get("seed", 69))
    w_psd = float(train_cfg.get("w_psd", 0.0))
    w_fc = float(train_cfg.get("w_fc", 0.0))
    patience = int(train_cfg.get("patience", 50))
    scaler_type = train_cfg.get("scaler", "standard")
    global_scaling = train_cfg.get("global_scaling", False)

    artifact_dir = config.get("artifact")
    if artifact_dir is None:
        local_now = datetime.datetime.now(datetime.UTC).astimezone()
        artifact_dir = f"artifacts/nn_predictor_{local_now.strftime('%Y-%m-%d_%H-%M-%S')}"
    artifact_dir: Path = Path(artifact_dir)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, artifact_dir / config_path.name)
    artifact = artifact_dir / "model.eqx"

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

    X_full, Y_full, C_y = prepare_datasets(data_files, n_steps_cfg, downsample, n_y, n_u, horizon)
    C_u = (X_full.shape[1] - n_y * C_y) // (n_u + horizon)

    split_idx = int(train_split * len(X_full))
    X_train, X_val = X_full[:split_idx], X_full[split_idx:]
    Y_train, Y_val = Y_full[:split_idx], Y_full[split_idx:]

    y_past_train = reshape_flat_to_channels(X_train[:, : n_y * C_y], C_y)
    u_past_train = reshape_flat_to_channels(X_train[:, n_y * C_y : n_y * C_y + n_u * C_u], C_u)

    if scaler_type == "robust":
        scaler_y = RobustScaler()
        scaler_u = RobustScaler()
    else:
        scaler_y = StandardScaler()
        scaler_u = StandardScaler()

    if global_scaling:
        scaler_y.fit(y_past_train.reshape(-1, 1))
        scaler_u.fit(u_past_train.reshape(-1, 1))
    else:
        scaler_y.fit(y_past_train)
        scaler_u.fit(u_past_train)

    X_train_s, Y_train_s = scale_dataset(
        X_train, Y_train, scaler_y, scaler_u, n_y, n_u, C_y, C_u, global_scaling=global_scaling
    )
    X_val_s, Y_val_s = scale_dataset(
        X_val, Y_val, scaler_y, scaler_u, n_y, n_u, C_y, C_u, global_scaling=global_scaling
    )

    key = jax.random.PRNGKey(seed)
    in_size = n_y * C_y + n_u * C_u
    out_size = C_y

    model = create_model(in_size, out_size, hidden_size, depth, key, n_y, n_u, horizon, C_y, C_u, activation=activation)

    model, train_losses, val_losses = train_model(
        model,
        X_train_s,
        Y_train_s,
        X_val_s,
        Y_val_s,
        epochs,
        batch_size,
        learning_rate,
        weight_decay,
        curriculum_decay_fraction,
        C_y,
        w_psd=w_psd,
        w_fc=w_fc,
        patience=patience,
    )

    loss_plot_path = artifact_dir / "loss_curve.png"
    plot_training_curves(train_losses, val_losses, str(loss_plot_path))

    if not isinstance(model, AutoregressivePredictor):
        msg = f"expected train_model to return an AutoregressivePredictor, got {type(model)}"
        raise TypeError(msg)
    u_mean = getattr(scaler_u, "mean_", getattr(scaler_u, "center_", None))
    u_scale = getattr(scaler_u, "scale_", None)
    y_mean = getattr(scaler_y, "mean_", getattr(scaler_y, "center_", None))
    y_scale = getattr(scaler_y, "scale_", None)
    if u_mean is None or u_scale is None or y_mean is None or y_scale is None:
        msg = "Scalers must be fitted before saving the artifact."
        raise ValueError(msg)
    MLPArtifact(
        model=model,
        dt=float(dt_real),
        downsample=int(downsample),
        u_mean=np.asarray(u_mean, dtype=np.float64),
        u_scale=np.asarray(u_scale, dtype=np.float64),
        y_mean=np.asarray(y_mean, dtype=np.float64),
        y_scale=np.asarray(y_scale, dtype=np.float64),
    ).save(artifact)
    print(f"Saved NN predictor artifact -> {artifact}")

    Y_pred_abs_flat, _mse = evaluate_model(model, scaler_y, X_val_s, Y_val, C_y, global_scaling=global_scaling)

    stats_path = artifact_dir / "training_stats.json"
    stats = {
        "train_loss": train_losses,
        "val_loss": val_losses,
    }
    stats_path.write_text(json.dumps(stats, indent=2))

    plot_path = artifact_dir / "comparison.png"
    Y_pred_unflat = np.asarray(reshape_to_trajectory(Y_pred_abs_flat, horizon, C_y))

    n_plot_samples = min(200, len(Y_val))
    split_idx * dt_real
    fig, _ = plot_multistep_predictions(
        y_true=Y_val[:n_plot_samples],
        y_pred=Y_pred_unflat[:n_plot_samples],
        dt=dt_real,
        channels=list(range(min(4, C_y))),
        stride=horizon,
        title=f"EEG {horizon}-Step Ahead Prediction",
    )
    fig.savefig(str(plot_path), dpi=300)
    plt.close(fig)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
