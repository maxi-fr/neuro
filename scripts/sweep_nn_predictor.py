"""Script to run hyperparameter search for the JAX-based NN predictor using Optuna."""

import argparse
import copy
import datetime
import shutil
from pathlib import Path
from typing import Any

import optuna
import yaml

from neuro.nn_training import train_and_save_predictor


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


def objective(
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

    # We save artifacts for EVERY trial
    trial_dir = base_artifact_dir / f"trial_{trial.number}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Save the specific config used for this trial
    with (trial_dir / "trial_config.yaml").open("w") as f:
        yaml.dump(config, f)

    # Vary the seed across trials to avoid correlation if params are identical.
    try:
        mse = train_and_save_predictor(config, data_files, trial_dir, seed_offset=trial.number)
    except ValueError as e:
        if "NaN" in str(e):
            print(f"Trial {trial.number} pruned: {e}")
            raise optuna.TrialPruned from e
        raise

    print(f"\nTrial {trial.number} completed with MSE: {mse}")
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
