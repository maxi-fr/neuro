# ruff: noqa: C901, PLR0912, PLR0915
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import TYPE_CHECKING, cast

import casadi as ca
import numpy as np

from neuro.checkpoint import build_symbolic_model, load_any
from neuro.control.nlp import MPCNlp
from neuro.control.solvers import (
    IpoptMPCSolver,
    MPCSolver,
    SqpFallbackMPCSolver,
    SqpMPCSolver,
)
from neuro.observable import load_log_reference
from neuro.observable_casadi import ObservableSymbolicModel
from neuro.predictor.data import load_trajectory
from neuro.predictor.esn_module import ESNModule
from neuro.predictor.module import AutoregressiveMLP
from neuro.predictor.observable_module import StepwiseObservableMLP
from neuro.spectral import PsdEnvelope

if TYPE_CHECKING:
    from neuro.esn_predictor_casadi import ESNSymbolicModel
    from neuro.nn_predictor_casadi import NNSymbolicModel
    from neuro.types import FloatArray


def _load_predictor(path: Path) -> AutoregressiveMLP | ESNModule | StepwiseObservableMLP:
    """Load the torch Predictor whose checkpoint ``path`` names, for State Absorption."""
    ckpt = load_any(path)
    if ckpt.model_type == "mlp":
        return AutoregressiveMLP.load(path)
    if ckpt.model_type == "esn":
        return ESNModule.load(path)
    return StepwiseObservableMLP.load(path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for MPC solve time probe."""
    parser = argparse.ArgumentParser(
        description="Benchmark MPC NLP build time and solve times for a predictor checkpoint."
    )
    parser.add_argument("--artifact", type=Path, required=True, help="Checkpoint basename path.")
    parser.add_argument("--data", type=Path, required=True, help="Directory of held-out .npz trajectories.")
    parser.add_argument(
        "--horizon", type=int, default=None, help="MPC horizon steps (defaults to checkpoint native horizon)."
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
    parser.add_argument("--w-psd", type=float, default=0.0, help="Weight on the spectral / observable hinge.")
    parser.add_argument(
        "--psd-ref",
        type=Path,
        default=None,
        help="Healthy envelope npz; required for an observable checkpoint and whenever --w-psd > 0.",
    )
    return parser.parse_args()


def time_objective_and_jacobian(mpc_nlp: MPCNlp, x0: FloatArray, n_reps: int) -> tuple[float, float]:
    """Median milliseconds for one objective evaluation and one objective-Jacobian evaluation.

    One evaluation is not a solve, so the solve loop below is timed as well.
    """
    x_sym, p_sym, f_sym = mpc_nlp.nlp["x"], mpc_nlp.nlp["p"], mpc_nlp.nlp["f"]
    objective = ca.Function("f_obj", [x_sym, p_sym], [f_sym])
    jacobian = ca.Function("f_jac", [x_sym, p_sym], [ca.jacobian(f_sym, x_sym)])
    w0 = np.zeros(x_sym.numel())

    times = []
    for fn in (objective, jacobian):
        fn(w0, x0)  # warm the graph, so the first call's allocation is not timed
        samples = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            fn(w0, x0)
            samples.append((time.perf_counter() - t0) * 1000.0)
        times.append(float(np.median(samples)))
    return times[0], times[1]


def main() -> None:
    """Benchmark NLP build and solve times on primed state samples."""
    args = parse_args()
    ckpt = load_any(args.artifact)
    predictor = _load_predictor(args.artifact)
    horizon = args.horizon if args.horizon is not None else ckpt.horizon
    shooting_depth = args.shooting_depth if args.shooting_depth is not None else horizon

    model = build_symbolic_model(ckpt)
    observable = isinstance(model, ObservableSymbolicModel)
    if observable:
        if args.psd_ref is None:
            msg = "an observable checkpoint costs a hinge against --psd-ref; pass one."
            raise SystemExit(msg)
        shooting_depth = horizon

    graph_nodes = (
        model.f_forecast.n_nodes() if observable else cast("NNSymbolicModel | ESNSymbolicModel", model).f_step.n_nodes()
    )
    log_reference = load_log_reference(args.psd_ref, model.geometry, model.fs) if observable and args.psd_ref else None
    psd_envelope = PsdEnvelope.load(args.psd_ref) if not observable and args.psd_ref is not None else None
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
        w_y=0.0 if observable else args.w_y,
        w_y_terminal=None,
        w_u=args.w_u,
        w_u_l1=0.0,
        w_psd=args.w_psd,
        psd_envelope=psd_envelope,
        log_reference=log_reference,
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
        u_raw, y_raw = load_trajectory(f, None, ckpt.downsample, ckpt.dt / ckpt.downsample)
        for t0 in range(predictor.priming_steps, len(y_raw) - horizon, 50):
            y_hist = y_raw[t0 - predictor.priming_steps : t0]
            u_hist = u_raw[t0 - predictor.priming_steps : t0]
            x0_samples.append(predictor.prime(y_hist, u_hist))
            if len(x0_samples) >= args.n_solves:
                break
        if len(x0_samples) >= args.n_solves:
            break

    if not x0_samples:
        msg = "Could not sample any initial states from test data."
        raise SystemExit(msg)

    obj_ms, jac_ms = time_objective_and_jacobian(mpc_nlp, x0_samples[0], n_reps=20)

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

    model_name = ckpt.model_type
    print("\n================ BENCHMARK RESULTS ================", flush=True)
    print(f"Checkpoint:       {args.artifact} ({model_name})", flush=True)
    print(f"Solver mode:       {args.solver} (QP: {args.sqp_qpsol})", flush=True)
    print(f"Prediction graph:  {graph_nodes} nodes ({'f_forecast' if observable else 'f_step'})", flush=True)
    print(f"Objective / Jacobian: {obj_ms:.2f} ms | {jac_ms:.2f} ms (median of 20)", flush=True)
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
