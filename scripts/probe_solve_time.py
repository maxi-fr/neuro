# ruff: noqa: C901, PLR0912, PLR0915
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from neuro.artifacts import build_symbolic_model, load_any_artifact
from neuro.control.nlp import MPCNlp
from neuro.control.solvers import (
    IpoptMPCSolver,
    MPCSolver,
    SqpFallbackMPCSolver,
    SqpMPCSolver,
)
from neuro.predictor.data import load_trajectory


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for MPC solve time probe."""
    parser = argparse.ArgumentParser(
        description="Benchmark MPC NLP build time and solve times for a predictor artifact."
    )
    parser.add_argument("--artifact", type=Path, required=True, help="Artifact basename path.")
    parser.add_argument("--data", type=Path, required=True, help="Directory of held-out .npz trajectories.")
    parser.add_argument(
        "--horizon", type=int, default=None, help="MPC horizon steps (defaults to artifact native horizon)."
    )
    parser.add_argument(
        "--shooting-depth",
        type=int,
        default=None,
        help="Shooting depth D (defaults to horizon, i.e. single shooting).",
    )
    parser.add_argument("--n-solves", type=int, default=50, help="Number of solves to benchmark.")
    parser.add_argument("--u-max", type=float, default=5.0, help="Per-electrode control bound.")
    parser.add_argument("--w-y", type=float, default=1.0, help="Weight on predicted EEG power.")
    parser.add_argument("--w-u", type=float, default=0.0, help="Weight on control effort.")
    parser.add_argument(
        "--solver",
        type=str,
        choices=["ipopt", "sqp", "sqp_fallback"],
        default="sqp_fallback",
        help="Solver mode: sqp_fallback, sqp, or ipopt.",
    )
    parser.add_argument(
        "--sqp-qpsol",
        type=str,
        choices=["qpoases", "osqp", "qrqp"],
        default="qpoases",
        help="QP subsolver for SQP (qpoases, osqp, qrqp).",
    )
    parser.add_argument(
        "--sqp-hessian",
        type=str,
        choices=["limited-memory", "exact"],
        default="limited-memory",
        help="Hessian approximation for SQP.",
    )
    parser.add_argument("--sqp-max-iter", type=int, default=15, help="SQP max iterations before fallback.")
    parser.add_argument("--max-iter", type=int, default=100, help="IPOPT max iterations per solve.")
    return parser.parse_args()


def main() -> None:
    """Benchmark NLP build and solve times on primed state samples."""
    args = parse_args()
    art = load_any_artifact(args.artifact)
    horizon = args.horizon if args.horizon is not None else art.horizon
    shooting_depth = args.shooting_depth if args.shooting_depth is not None else horizon

    model = build_symbolic_model(art)

    f_step_nodes = model.f_step.n_nodes()
    u_max_arr = np.full(model.n_controls, args.u_max, dtype=np.float64)

    shooting_label = "single shooting" if shooting_depth >= horizon else f"multiple shooting (D={shooting_depth})"
    print(
        f"Building MPC NLP (horizon={horizon}, {shooting_label}, solver={args.solver}, qpsol={args.sqp_qpsol})...",
        flush=True,
    )
    t0_build = time.perf_counter()
    mpc_nlp = MPCNlp.build(
        model,
        horizon=horizon,
        shooting_depth=shooting_depth,
        n_controls=model.n_controls,
        u_max=u_max_arr,
        w_y=args.w_y,
        w_y_terminal=None,
        w_u=args.w_u,
        w_u_l1=0.0,
    )
    solver_obj: MPCSolver
    if args.solver == "sqp_fallback":
        solver_obj = SqpFallbackMPCSolver.build(
            mpc_nlp,
            max_iter=args.max_iter,
            sqp_qpsol=args.sqp_qpsol,
            sqp_hessian=args.sqp_hessian,
            sqp_max_iter=args.sqp_max_iter,
        )
    elif args.solver == "sqp":
        solver_obj = SqpMPCSolver.build(
            mpc_nlp,
            qpsol=args.sqp_qpsol,
            hessian_approximation=args.sqp_hessian,
            max_iter=args.sqp_max_iter,
        )
    else:
        solver_obj = IpoptMPCSolver.build(mpc_nlp, max_iter=args.max_iter)
    build_seconds = time.perf_counter() - t0_build

    files = sorted(str(p) for p in args.data.glob("*.npz"))
    if not files:
        msg = f"No .npz trajectories found in {args.data}"
        raise SystemExit(msg)

    print(f"Sampling initial states x0 from {len(files)} held-out trajectories...", flush=True)
    x0_samples = []
    for f in files:
        u_raw, y_raw = load_trajectory(f, None, art.downsample, art.dt / art.downsample)
        for t0 in range(art.priming_steps, len(y_raw) - horizon, 50):
            y_hist = y_raw[t0 - art.priming_steps : t0]
            u_hist = u_raw[t0 - art.priming_steps : t0]
            x0_samples.append(art.prime(y_hist, u_hist))
            if len(x0_samples) >= args.n_solves:
                break
        if len(x0_samples) >= args.n_solves:
            break

    if not x0_samples:
        msg = "Could not sample any initial states from test data."
        raise SystemExit(msg)

    solve_times = []
    iter_counts = []
    num_capped = 0
    num_fallback = 0

    m, h = model.n_controls, horizon
    u_guess = np.zeros((h, m), dtype=np.float64)
    w0 = u_guess.reshape(-1)

    print(f"Running {len(x0_samples)} MPC solves ({args.solver})...", flush=True)
    for x0 in x0_samples:
        t0_solve = time.perf_counter()
        res = solver_obj.solve(x0, w0)
        solve_time = (time.perf_counter() - t0_solve) * 1000.0
        solve_times.append(solve_time)
        iter_counts.append(res.n_iter)
        if res.capped:
            num_capped += 1
        if res.fallback:
            num_fallback += 1

    solve_times_arr = np.array(solve_times)

    iter_counts_arr = np.array(iter_counts)

    med_time = float(np.median(solve_times_arr))
    p90_time = float(np.percentile(solve_times_arr, 90))
    med_iters = float(np.median(iter_counts_arr))
    p90_iters = float(np.percentile(iter_counts_arr, 90))

    model_name = art.model_type
    print("\n================ BENCHMARK RESULTS ================", flush=True)
    print(f"Artifact:          {args.artifact} ({model_name})", flush=True)
    print(f"Solver mode:       {args.solver} (QP: {args.sqp_qpsol})", flush=True)
    print(f"f_step graph size: {f_step_nodes} nodes", flush=True)
    print(f"NLP build time:    {build_seconds:.4f} s", flush=True)
    print(f"Solves evaluated:  {len(x0_samples)}", flush=True)
    print(f"Solve wall time:   median {med_time:.2f} ms | p90 {p90_time:.2f} ms", flush=True)
    print(f"Iterations:        median {med_iters:.0f} | p90 {p90_iters:.0f}", flush=True)
    if args.solver == "sqp_fallback":
        print(f"Fallback triggered:{num_fallback} / {len(x0_samples)} solves", flush=True)
    print(f"Max iter capped:   {num_capped} / {len(x0_samples)} solves", flush=True)
    print("===================================================", flush=True)


if __name__ == "__main__":
    main()
