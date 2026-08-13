import argparse
import shutil
from pathlib import Path

import optuna
import yaml

from neuro.closed_loop_eval import evaluate_closed_loop_suppression
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
    trial: optuna.Trial,
    base_config: NNPredictorConfig,
    data_files: list[str],
    base_artifact_dir: Path,
) -> float:
    """Optuna objective: apply the trial's suggested params, train predictor, and compute metric."""
    sweep = base_config.sweep
    if sweep is None:
        msg = "objective requires a config with a 'sweep' section."
        raise RuntimeError(msg)

    model_overrides = {name: spec.suggest(trial, name) for name, spec in sweep.model.items()}
    training_overrides = {name: spec.suggest(trial, name) for name, spec in sweep.training.items()}
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

    trial_dir = base_artifact_dir / f"trial_{trial.number}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    trial_config = config.model_dump(exclude={"sweep"})
    with (trial_dir / "trial_config.yaml").open("w") as f:
        yaml.dump(trial_config, f)

    try:
        nmse_rollout = train_and_save_predictor(config, data_files, trial_dir, seed_offset=trial.number)
    except ValueError as e:
        if "NaN" in str(e):
            print(f"Trial {trial.number} pruned: {e}")
            raise optuna.TrialPruned from e
        raise

    trial.set_user_attr("nmse_rollout", float(nmse_rollout))

    if sweep.closed_loop is not None:
        print(f"\nEvaluating closed-loop seizure suppression on trial {trial.number}...")
        score, summary = evaluate_closed_loop_suppression(trial_dir, sweep.closed_loop)
        for k, v in summary.items():
            trial.set_user_attr(k, v)
        print(
            f"\nTrial {trial.number} completed with score: {score:.4f} "
            f"({int(summary['suppressed_seeds'])}/{int(summary['total_seeds'])} seeds suppressed, "
            f"mean amplitude: {summary['mean_amplitude']:.2%}, rollout NMSE: {nmse_rollout:.4f})"
        )
        return score

    print(f"\nTrial {trial.number} completed with rollout NMSE: {nmse_rollout}")
    return nmse_rollout


def main() -> None:
    """Execute the hyperparameter sweep script."""
    parser = argparse.ArgumentParser(description="Run Optuna Sweep for JAX NN Predictor.")
    parser.add_argument("--config", type=str, required=True, help="Path to sweep config YAML.")
    parser.add_argument("--data-path", type=str, help="Override config data path.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    data_files = resolve_data_files(config, args.data_path)

    if config.sweep is None:
        msg = f"No 'sweep' section found in config: {config_path}"
        raise ValueError(msg)

    base_artifact_dir = resolve_artifact_dir(config.sweep.artifact, "sweep_nn_predictor")
    shutil.copy2(config_path, base_artifact_dir / config_path.name)

    study_name = "nn_predictor_sweep"
    db_path = base_artifact_dir / f"{study_name}.db"
    storage_url = f"sqlite:///{db_path.resolve()}"

    study = optuna.create_study(study_name=study_name, storage=storage_url, direction="minimize", load_if_exists=True)
    study.optimize(
        lambda trial: objective(trial, config, data_files, base_artifact_dir),
        n_trials=config.sweep.n_trials,
    )

    print("\n================== SWEEP COMPLETED ==================")
    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value (Score): {best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    print(f"\nArtifacts saved in: {base_artifact_dir.resolve()}")


if __name__ == "__main__":
    main()
