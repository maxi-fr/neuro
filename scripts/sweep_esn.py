from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import optuna

from neuro.config import (
    FloatParam,
    ParamSpec,
    load_esn_config,
    resolve_data_files,
)
from neuro.esn import (
    ESNArtifact,
    generate_reservoir,
    harvest_normal_equations,
    solve_ridge,
)
from neuro.esn_predictor_casadi import ESNSymbolicModel
from neuro.esn_training import prepare_training_data
from neuro.predictor.esn_module import ESNModule
from neuro.predictor.evaluation import evaluate_rollouts
from neuro.provenance import training_provenance

optuna.logging.set_verbosity(optuna.logging.WARNING)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for ESN hyperparameter sweep."""
    parser = argparse.ArgumentParser(description="Sweep ESN hyperparameters over reservoir sizes N.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/nn_predictor/esn.yaml"),
        help="Path to base ESN predictor config file.",
    )
    parser.add_argument("--data-path", type=str, default=None, help="Optional data directory override.")
    parser.add_argument("--out-csv", type=Path, default=Path("esn_sweep_results.csv"), help="Output CSV path.")
    parser.add_argument(
        "--reservoir-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Reservoir sizes to sweep (overrides config).",
    )
    parser.add_argument("--n-trials", type=int, default=None, help="Number of Optuna trials per N (overrides config).")
    parser.add_argument("--append", action="store_true", help="Append results to out-csv if file exists.")
    return parser.parse_args()


def default_search_space() -> dict[str, ParamSpec]:
    """Default Optuna search space if not defined in config: the four dimensions of plan sec. 9.2."""
    return {
        "spectral_radius": FloatParam(type="float", low=0.1, high=1.5),
        "leak_rate": FloatParam(type="float", low=0.01, high=1.0),
        "density": FloatParam(type="float", low=0.01, high=0.5),
        "noise_sigma": FloatParam(type="float", low=0.0, high=0.5),
    }


def main() -> None:  # noqa: PLR0915
    """Run outer grid over N and inner Optuna sweep per N."""
    args = parse_args()
    cfg = load_esn_config(args.config)

    data_files = resolve_data_files(cfg, args.data_path)

    print("Loading trajectories and fitting standardizers...")
    data = prepare_training_data(cfg, data_files)
    provenance = training_provenance(data_files, cfg.simulation.cutoff_hz)
    print(f"Loaded {len(data.train_trajs)} train trajectories and {len(data.val_trajs)} val trajectories.")

    reservoir_sizes = (
        args.reservoir_sizes
        if args.reservoir_sizes is not None
        else (cfg.sweep.reservoir_sizes if cfg.sweep and cfg.sweep.reservoir_sizes else [100, 250, 500, 1000])
    )
    lambdas = cfg.sweep.lambdas if cfg.sweep and cfg.sweep.lambdas else [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    n_trials = args.n_trials if args.n_trials is not None else (cfg.sweep.n_trials if cfg.sweep else 50)
    search_space = cfg.sweep.model if (cfg.sweep and cfg.sweep.model) else default_search_space()

    out_csv = args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    csv_fields = [
        "N",
        "best_val_nmse",
        "best_ridge_lambda",
        "spectral_radius",
        "leak_rate",
        "density",
        "input_scaling",
        "noise_sigma",
        "priming_steps",
        "f_step_n_nodes",
        "harvest_seconds",
        "fit_seconds",
    ]

    file_exists = out_csv.exists() and out_csv.stat().st_size > 0
    mode = "a" if (args.append and file_exists) else "w"

    with out_csv.open(mode, newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=csv_fields)
        if not (args.append and file_exists):
            writer.writeheader()

        for N in reservoir_sizes:
            print(f"\n--- Starting Optuna study for N = {N} ({n_trials} trials) ---")
            study = optuna.create_study(direction="minimize")

            def objective(trial: optuna.Trial, res_size: int = N) -> float:
                params = {k: spec.suggest(trial, k) for k, spec in search_space.items()}

                spec_rad = float(params.get("spectral_radius", cfg.model.spectral_radius))
                leak = float(params.get("leak_rate", cfg.model.leak_rate))
                dens = float(params.get("density", cfg.model.density))
                inp_scale = float(params.get("input_scaling", cfg.model.input_scaling))
                noise = float(params.get("noise_sigma", cfg.model.noise_sigma))
                prime_steps = int(params.get("priming_steps", cfg.model.priming_steps))

                try:
                    w_res, w_in = generate_reservoir(
                        reservoir_size=res_size,
                        spectral_radius=spec_rad,
                        density=dens,
                        input_scaling=inp_scale,
                        in_dim=data.in_dim,
                        seed=cfg.training.seed,
                    )
                except ValueError:
                    return float("inf")

                t0_h = time.perf_counter()
                G, P = harvest_normal_equations(
                    trajectories=data.train_trajs,
                    y_std=data.y_std,
                    u_std=data.u_std,
                    w_res=w_res,
                    w_in=w_in,
                    leak_rate=leak,
                    priming_steps=prime_steps,
                    noise_sigma=noise,
                    seed=cfg.training.seed,
                )
                harvest_time = time.perf_counter() - t0_h

                trial_best_nmse = float("inf")
                trial_best_lam = lambdas[0]
                trial_best_fit_time = 0.0

                for lam in lambdas:
                    t0_f = time.perf_counter()
                    w_out = solve_ridge(G, P, lam)
                    fit_time = time.perf_counter() - t0_f

                    model = ESNModule(
                        w_res=w_res,
                        w_in=w_in,
                        w_out=w_out,
                        leak_rate=leak,
                        priming_steps=prime_steps,
                        horizon=cfg.model.horizon,
                        dt=cfg.simulation.dt * cfg.simulation.downsample,
                        y_std=data.y_std,
                        u_std=data.u_std,
                    )
                    val_nmse = evaluate_rollouts(model, data.val_trajs, cfg.model.horizon).pooled
                    if val_nmse < trial_best_nmse:
                        trial_best_nmse = val_nmse
                        trial_best_lam = lam
                        trial_best_fit_time = fit_time

                trial.set_user_attr("ridge_lambda", trial_best_lam)
                trial.set_user_attr("harvest_time", harvest_time)
                trial.set_user_attr("fit_time", trial_best_fit_time)
                return trial_best_nmse

            study.optimize(objective, n_trials=n_trials)
            best_t = study.best_trial
            best_nmse = best_t.value
            best_lam = best_t.user_attrs.get("ridge_lambda", lambdas[0])
            h_time = best_t.user_attrs.get("harvest_time", 0.0)
            f_time = best_t.user_attrs.get("fit_time", 0.0)
            b_params = best_t.params

            spec_rad = float(b_params.get("spectral_radius", cfg.model.spectral_radius))
            leak = float(b_params.get("leak_rate", cfg.model.leak_rate))
            dens = float(b_params.get("density", cfg.model.density))
            inp_scale = float(b_params.get("input_scaling", cfg.model.input_scaling))
            noise = float(b_params.get("noise_sigma", cfg.model.noise_sigma))
            prime_steps = int(b_params.get("priming_steps", cfg.model.priming_steps))

            w_res, w_in = generate_reservoir(N, spec_rad, dens, inp_scale, data.in_dim, cfg.training.seed)
            G, P = harvest_normal_equations(
                data.train_trajs,
                data.y_std,
                data.u_std,
                w_res,
                w_in,
                leak,
                prime_steps,
                noise,
                cfg.training.seed,
            )
            w_out = solve_ridge(G, P, best_lam)

            winning_art = ESNArtifact(
                w_in=w_in,
                w_out=w_out,
                w_res=w_res,
                dt=cfg.simulation.dt * cfg.simulation.downsample,
                downsample=cfg.simulation.downsample,
                horizon=cfg.model.horizon,
                reservoir_size=N,
                leak_rate=leak,
                spectral_radius=spec_rad,
                priming_steps=prime_steps,
                input_scaling=inp_scale,
                density=dens,
                noise_sigma=noise,
                ridge_lambda=best_lam,
                seed=cfg.training.seed,
                y_std=data.y_std,
                u_std=data.u_std,
                provenance=provenance,
            )
            sym_model = ESNSymbolicModel(winning_art)
            n_nodes = sym_model.f_step.n_nodes()

            row = {
                "N": N,
                "best_val_nmse": f"{best_nmse:.6f}",
                "best_ridge_lambda": f"{best_lam:.1e}",
                "spectral_radius": f"{spec_rad:.4f}",
                "leak_rate": f"{leak:.4f}",
                "density": f"{dens:.4f}",
                "input_scaling": f"{inp_scale:.4f}",
                "noise_sigma": f"{noise:.4f}",
                "priming_steps": prime_steps,
                "f_step_n_nodes": n_nodes,
                "harvest_seconds": f"{h_time:.2f}",
                "fit_seconds": f"{f_time:.2f}",
            }
            writer.writerow(row)
            f_csv.flush()
            print(f"N={N}: Best NMSE={best_nmse:.4f}, f_step nodes={n_nodes}, harvest={h_time:.2f}s, fit={f_time:.2f}s")

    print(f"\nSweep complete. Results written to {out_csv}")


if __name__ == "__main__":
    main()
