"""Script to run hyperparameter search for the JAX-based NN predictor using Optuna."""

import argparse
import shutil
from pathlib import Path

import optuna
import yaml

from neuro.config import (
    ModelConfig,
    NNPredictorConfig,
    TrainingConfig,
    load_config,
    resolve_artifact_dir,
    resolve_data_files,
)
from neuro.nn_training import train_and_save_predictor


def objective(
    trial: optuna.Trial, base_config: NNPredictorConfig, data_files: list[str], base_artifact_dir: Path
) -> float:
    """Optuna objective: apply the trial's suggested params, then train and evaluate the predictor."""
    sweep = base_config.sweep
    if sweep is None:
        msg = "objective requires a config with a 'sweep' section."
        raise RuntimeError(msg)

    model_overrides = {name: spec.suggest(trial, name) for name, spec in sweep.model.items()}
    training_overrides = {name: spec.suggest(trial, name) for name, spec in sweep.training.items()}
    # Re-validate the merged sections so an override that is not a real hyperparameter raises.
    config = base_config.model_copy(
        update={
            "model": ModelConfig.model_validate({**base_config.model.model_dump(), **model_overrides}),
            "training": TrainingConfig.model_validate({**base_config.training.model_dump(), **training_overrides}),
        }
    )

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

    # Save the resolved config used for this trial (without the sweep search space).
    trial_config = config.model_dump(exclude={"sweep"})
    with (trial_dir / "trial_config.yaml").open("w") as f:
        yaml.dump(trial_config, f)

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
    try:
        config = load_config(config_path)
        data_files = resolve_data_files(config, args.data_path)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        return

    if config.sweep is None:
        print("No 'sweep' section found in config.")
        return

    base_artifact_dir = resolve_artifact_dir(config.sweep.artifact, "sweep_nn_predictor")
    shutil.copy2(config_path, base_artifact_dir / config_path.name)

    study_name = "nn_predictor_sweep"
    db_path = base_artifact_dir / f"{study_name}.db"
    storage_url = f"sqlite:///{db_path.resolve()}"

    study = optuna.create_study(study_name=study_name, storage=storage_url, direction="minimize", load_if_exists=True)
    study.optimize(
        lambda trial: objective(trial, config, data_files, base_artifact_dir), n_trials=config.sweep.n_trials
    )

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
