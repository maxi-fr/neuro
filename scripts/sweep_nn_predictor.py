"""Script to run hyperparameter search for the JAX-based NN predictor using Optuna."""

import argparse
import copy
import datetime
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import matplotlib.pyplot as plt
import numpy as np
import optuna
import yaml
from sklearn.preprocessing import RobustScaler, StandardScaler

from neuro.nn_training import (
    create_model,
    evaluate_model,
    fit_latent_projection,
    plot_training_curves,
    prepare_datasets,
    reshape_flat_to_channels,
    reshape_to_trajectory,
    scale_dataset,
    train_model,
)
from neuro.prediction import AutoregressivePredictor, MLPArtifact
from utils.plotting import plot_multistep_predictions

if TYPE_CHECKING:
    from neuro.types import FloatArray


def suggest_params(trial: optuna.Trial, sweep_cfg: dict[str, Any], group: str, config: dict[str, Any]) -> None:
    """Suggest parameters for a given group (e.g. 'model' or 'training') and update the config."""
    if group not in sweep_cfg or not isinstance(sweep_cfg[group], dict):
        return

    for param_name, param_def in sweep_cfg[group].items():
        if not isinstance(param_def, dict) or "type" not in param_def:
            continue

        ptype = param_def["type"]
        if ptype == "categorical":
            config[group][param_name] = trial.suggest_categorical(param_name, param_def["choices"])
        elif ptype == "int":
            config[group][param_name] = trial.suggest_int(
                param_name, param_def["low"], param_def["high"], log=param_def.get("log", False)
            )
        elif ptype == "float":
            config[group][param_name] = trial.suggest_float(
                param_name, param_def["low"], param_def["high"], log=param_def.get("log", False)
            )
        elif ptype == "loguniform":
            config[group][param_name] = trial.suggest_float(param_name, param_def["low"], param_def["high"], log=True)


def objective(  # noqa: PLR0915
    trial: optuna.Trial, base_config: dict[str, Any], data_files: list[str], base_artifact_dir: Path
) -> float:
    """Optuna objective function for training and evaluating the NN predictor."""
    # Create a deep copy of the config so we don't overwrite the base
    config = copy.deepcopy(base_config)

    sweep_cfg = config.get("sweep", {})
    suggest_params(trial, sweep_cfg, "model", config)
    suggest_params(trial, sweep_cfg, "training", config)

    print(f"\n{'=' * 40}")
    print(f"Starting Trial {trial.number}")
    print(f"{'=' * 40}")
    print("Suggested hyperparameters:")
    for k, v in trial.params.items():
        print(f"  {k}: {v}")
    print("-" * 40)

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
    depth = int(model_cfg.get("depth", 2))
    projection_cfg = model_cfg.get("projection", {})
    enable_projection = bool(projection_cfg.get("enable", False))
    latent_dim = projection_cfg.get("latent_dim")
    explained_variance = projection_cfg.get("explained_variance")

    train_cfg = config.get("training", {})
    epochs = int(train_cfg.get("epochs", 100))
    batch_size = int(train_cfg.get("batch_size", 128))
    learning_rate = float(train_cfg.get("learning_rate", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    train_split = float(train_cfg.get("train_split", 0.8))
    curriculum_decay_fraction = float(train_cfg.get("curriculum_decay_fraction", 0.8))
    seed = int(train_cfg.get("seed", 69))
    w_psd = float(train_cfg.get("w_psd", 0.0))
    w_fc = float(train_cfg.get("w_fc", 0.0))
    patience = int(train_cfg.get("patience", 50))
    global_scaling = train_cfg["global_scaling"]

    # We save artifacts for EVERY trial
    trial_dir = base_artifact_dir / f"trial_{trial.number}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Save the specific config used for this trial
    trial_config_path = trial_dir / "trial_config.yaml"
    with trial_config_path.open("w") as f:
        yaml.dump(config, f)

    artifact = trial_dir / "model.eqx"

    projection: tuple[FloatArray, FloatArray] | None = None
    if enable_projection:
        if w_psd > 0 or w_fc > 0:
            msg = "Latent projection is incompatible with the per-channel PSD/FC losses (set w_psd=w_fc=0)."
            raise ValueError(msg)
        projection = fit_latent_projection(data_files, n_steps_cfg, downsample, latent_dim, explained_variance)

    X_full, Y_full, C_y = prepare_datasets(data_files, n_steps_cfg, downsample, n_y, n_u, horizon, projection)
    C_u = (X_full.shape[1] - n_y * C_y) // (n_u + horizon)

    split_idx = int(train_split * len(X_full))
    X_train, X_val = X_full[:split_idx], X_full[split_idx:]
    Y_train, Y_val = Y_full[:split_idx], Y_full[split_idx:]

    y_past_train = reshape_flat_to_channels(X_train[:, : n_y * C_y], C_y)
    u_past_train = reshape_flat_to_channels(X_train[:, n_y * C_y : n_y * C_y + n_u * C_u], C_u)

    scaler_type = train_cfg.get("scaler", "standard")
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

    X_train_s, Y_train_s = scale_dataset(X_train, Y_train, scaler_y, scaler_u, n_y, n_u, C_y, C_u, global_scaling)
    X_val_s, Y_val_s = scale_dataset(X_val, Y_val, scaler_y, scaler_u, n_y, n_u, C_y, C_u, global_scaling)

    key = jax.random.PRNGKey(
        seed + trial.number
    )  # Vary seed slightly across trials to avoid correlation if params are identical
    in_size = n_y * C_y + n_u * C_u
    out_size = C_y

    model = create_model(in_size, out_size, hidden_size, depth, key, n_y, n_u, horizon, C_y, C_u)

    try:
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
    except ValueError as e:
        if "NaN" in str(e):
            print(f"Trial {trial.number} pruned: {e}")
            raise optuna.TrialPruned from e
        raise

    loss_plot_path = trial_dir / "loss_curve.png"
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
        latent_basis=projection[0] if projection is not None else None,
        latent_mean=projection[1] if projection is not None else None,
    ).save(artifact)

    Y_pred_abs_flat, mse = evaluate_model(model, scaler_y, X_val_s, Y_val, C_y, global_scaling)

    print(f"\nTrial {trial.number} completed with MSE: {mse}")

    stats_path = trial_dir / "training_stats.json"
    stats = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "mse": float(mse),
    }
    stats_path.write_text(json.dumps(stats, indent=2))

    plot_path = trial_dir / "comparison.png"
    Y_pred_unflat = np.asarray(reshape_to_trajectory(Y_pred_abs_flat, horizon, C_y))

    n_plot_samples = min(200, len(Y_val))
    y_true_anchors = X_val[:n_plot_samples, (n_y - 1) * C_y : n_y * C_y]
    fig, _ = plot_multistep_predictions(
        y_true=y_true_anchors,
        y_pred=Y_pred_unflat[:n_plot_samples],
        dt=dt_real,
        channels=list(range(min(4, C_y))),
        stride=horizon,
        title=f"EEG {horizon}-Step Ahead Prediction",
    )
    fig.savefig(str(plot_path), dpi=300)
    plt.close(fig)

    return mse


def main() -> None:
    """Execute the hyperparameter sweep script."""
    parser = argparse.ArgumentParser(description="Run Optuna Sweep for JAX NN Predictor.")
    parser.add_argument(
        "--config", type=str, default="configs/nn_predictor/sweep_config.yaml", help="Path to sweep config YAML."
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

    sweep_cfg = config.get("sweep", {})
    n_trials = sweep_cfg.get("n_trials", 20)

    artifact_dir = sweep_cfg.get("artifact")
    if artifact_dir is None:
        local_now = datetime.datetime.now(datetime.UTC).astimezone()
        artifact_dir = f"artifacts/sweep_nn_predictor_{local_now.strftime('%Y-%m-%d_%H-%M-%S')}"
    base_artifact_dir: Path = Path(artifact_dir)
    base_artifact_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(config_path, base_artifact_dir / config_path.name)

    study_name = "nn_predictor_sweep"
    db_path = base_artifact_dir / f"{study_name}.db"
    storage_url = f"sqlite:///{db_path.resolve()}"

    study = optuna.create_study(study_name=study_name, storage=storage_url, direction="minimize", load_if_exists=True)
    study.optimize(lambda trial: objective(trial, config, data_files, base_artifact_dir), n_trials=n_trials)

    print("\n================== SWEEP COMPLETED ==================")
    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value (MSE): {best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    print(f"\nArtifacts saved in: {base_artifact_dir.resolve()}")


if __name__ == "__main__":
    main()
