from __future__ import annotations

import argparse
import csv
from pathlib import Path

import optuna

from neuro.config import load_esn_config, resolve_artifact_dir, resolve_data_files
from neuro.predictor.sweep import GridSweep

optuna.logging.set_verbosity(optuna.logging.WARNING)


def main() -> None:
    """Execute the ESN sweep and write one CSV row per outer-grid cell."""
    parser = argparse.ArgumentParser(
        description="Sweep ESN hyperparameters over the reservoir-size x ridge-lambda grid."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/nn_predictor/esn.yaml"),
        help="Path to base ESN predictor config file.",
    )
    parser.add_argument("--data-path", type=str, default=None, help="Optional data directory override.")
    parser.add_argument("--out-csv", type=Path, default=Path("esn_sweep_results.csv"), help="Output CSV path.")
    args = parser.parse_args()

    cfg = load_esn_config(args.config)
    data_files = resolve_data_files(cfg, args.data_path)
    if cfg.sweep is None:
        msg = f"No 'sweep' section found in config: {args.config}"
        raise ValueError(msg)

    artifact_dir = resolve_artifact_dir(cfg.sweep.artifact, "sweep_esn")
    results = GridSweep(cfg, data_files, artifact_dir).run()

    param_names = sorted({name for result in results for name in result.params})
    candidate_names = sorted({name for result in results for name in result.candidates})
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["N", "ridge_lambda", "value", *param_names, *candidate_names])
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "N": result.reservoir_size,
                    "ridge_lambda": result.ridge_lambda,
                    "value": f"{result.value:.6f}",
                    **{name: result.params[name] for name in param_names},
                    **{name: f"{result.candidates[name]:.6f}" for name in candidate_names},
                }
            )

    print(f"Sweep complete: {len(results)} cells; results written to {args.out_csv.resolve()}")


if __name__ == "__main__":
    main()
