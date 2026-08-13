from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from neuro.artifacts import evaluate_rollouts
from neuro.config import (
    load_esn_config,
    resolve_artifact_dir,
    resolve_data_files,
)
from neuro.esn import (
    ESNArtifact,
    generate_reservoir,
    harvest_normal_equations,
    solve_ridge,
)
from neuro.esn_training import prepare_training_data


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for ESN training."""
    parser = argparse.ArgumentParser(description="Train an Echo State Network (ESN) predictor.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/nn_predictor/esn_8s.yaml"),
        help="Path to ESN predictor config file.",
    )
    parser.add_argument("--data-path", type=str, default=None, help="Optional data directory override.")
    return parser.parse_args()


def main() -> None:
    """Train ESN model, save artifact and training stats."""
    args = parse_args()
    cfg = load_esn_config(args.config)

    data_files = resolve_data_files(cfg, args.data_path)

    print("Loading trajectories and fitting feature pipelines...", flush=True)
    data = prepare_training_data(cfg, data_files)
    print(
        f"Loaded {len(data.train_trajs)} training trajectories and {len(data.val_trajs)} validation trajectories.",
        flush=True,
    )

    print(
        f"Generating reservoir (N={cfg.model.reservoir_size}, rho={cfg.model.spectral_radius}, density={cfg.model.density})...",
        flush=True,
    )
    w_res, w_in = generate_reservoir(
        reservoir_size=cfg.model.reservoir_size,
        spectral_radius=cfg.model.spectral_radius,
        density=cfg.model.density,
        input_scaling=cfg.model.input_scaling,
        in_dim=data.in_dim,
        seed=cfg.training.seed,
    )

    print("Harvesting states over training trajectories...", flush=True)
    t0_harvest = time.perf_counter()
    G, P = harvest_normal_equations(
        trajectories=data.train_trajs,
        y_pipeline=data.y_pipeline,
        u_pipeline=data.u_pipeline,
        w_res=w_res,
        w_in=w_in,
        leak_rate=cfg.model.leak_rate,
        washout=cfg.model.washout,
        noise_sigma=cfg.model.noise_sigma,
        seed=cfg.training.seed,
    )
    harvest_seconds = time.perf_counter() - t0_harvest
    print(f"Harvest complete in {harvest_seconds:.2f} s", flush=True)

    print(f"Solving ridge regression (lambda={cfg.model.ridge_lambda})...", flush=True)
    t0_fit = time.perf_counter()
    w_out = solve_ridge(G, P, cfg.model.ridge_lambda)
    fit_seconds = time.perf_counter() - t0_fit
    print(f"Fit complete in {fit_seconds:.2f} s", flush=True)

    art = ESNArtifact(
        w_in=w_in,
        w_out=w_out,
        w_res=w_res,
        dt=cfg.simulation.dt * cfg.simulation.downsample,
        downsample=cfg.simulation.downsample,
        horizon=cfg.model.horizon,
        reservoir_size=cfg.model.reservoir_size,
        leak_rate=cfg.model.leak_rate,
        spectral_radius=cfg.model.spectral_radius,
        washout=cfg.model.washout,
        input_scaling=cfg.model.input_scaling,
        density=cfg.model.density,
        noise_sigma=cfg.model.noise_sigma,
        ridge_lambda=cfg.model.ridge_lambda,
        seed=cfg.training.seed,
        y_pipeline=data.y_pipeline,
        u_pipeline=data.u_pipeline,
    )

    print("Evaluating validation rollout NMSE...", flush=True)
    rollout = evaluate_rollouts(art, data.val_trajs, cfg.model.horizon)
    print(f"Validation rollout NMSE at horizon {cfg.model.horizon}: {rollout.pooled:.4f}", flush=True)

    artifact_dir = resolve_artifact_dir(cfg.artifact, "esn")
    artifact_base = artifact_dir / "model"
    art.save(artifact_base)

    stats = {
        "harvest_seconds": harvest_seconds,
        "fit_seconds": fit_seconds,
        "nmse_rollout": rollout.pooled,
        "nmse_rollout_per_step": rollout.per_step.tolist(),
    }
    (artifact_dir / "training_stats.json").write_text(json.dumps(stats, indent=2))
    with (artifact_dir / "config.yaml").open("w") as f:
        yaml.dump(cfg.model_dump(), f)

    print(f"Artifact successfully saved to {artifact_dir}", flush=True)


if __name__ == "__main__":
    main()
