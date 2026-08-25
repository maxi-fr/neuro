import argparse
import shutil
from pathlib import Path

from neuro.config import load_config, resolve_artifact_dir, resolve_data_files
from neuro.predictor.sweep import OptunaSweep


def main() -> None:
    """Execute the Optuna hyperparameter sweep for the NN predictors."""
    parser = argparse.ArgumentParser(description="Run Optuna Sweep for the NN Predictors.")
    parser.add_argument("--config", type=str, required=True, help="Path to sweep config YAML.")
    parser.add_argument("--data-path", type=str, help="Override config data path.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    data_files = resolve_data_files(config, args.data_path)

    if config.sweep is None:
        msg = f"No 'sweep' section found in config: {config_path}"
        raise ValueError(msg)

    artifact_dir = resolve_artifact_dir(config.sweep.artifact, "sweep_nn_predictor")
    shutil.copy2(config_path, artifact_dir / config_path.name)

    study = OptunaSweep(config, data_files, artifact_dir).run()

    print("\n================== SWEEP COMPLETED ==================")
    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value (Score): {best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    print(f"\nArtifacts saved in: {artifact_dir.resolve()}")


if __name__ == "__main__":
    main()
