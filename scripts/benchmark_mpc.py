"""Benchmark the MPC optimal-control problem (OCP) solve speed.

Loads a trained NN-predictor artifact, builds an :class:`~neuro.control.MPCController`,
warms up its history window with realistic-magnitude EEG, then times ``update()`` over a
handful of warm-started solves. Reports problem size, solver build time, the cold (first)
solve, and steady-state warm-solve timings -- both the end-to-end ``update()`` wall time
(what the closed loop pays per step) and IPOPT's own ``t_wall_total`` -- plus iteration
counts and success rate.

This is the baseline harness for comparing OCP formulations: rerun it after a change
(state condensing, ``expand``, ReLU smoothing, SQP/QP backend, shorter horizon) and read
the same rows side by side. EEG inputs are synthetic but deterministic (seeded, scaled to
the model's own statistics), so timings are comparable across runs and formulations.

Example
-------
    uv run python scripts/benchmark_mpc.py \
        --artifact artifacts/sweep_nn_predictor_2026-06-28_13-33-04/trial_49/model \
        --horizons 3 5 10 20
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax

jax.config.update("jax_enable_x64", True)  # noqa: FBT003 -- float64 parity with the predictor

import numpy as np  # noqa: E402

from neuro.control import MPCController  # noqa: E402
from neuro.nn_predictor_casadi import NNSymbolicModel  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Callable

    from neuro.types import FloatArray


@dataclass(frozen=True)
class BenchResult:
    """Timing summary for one horizon (milliseconds unless noted)."""

    horizon: int
    n_vars: int
    n_eq: int
    build_s: float
    cold_update_ms: float
    warm_update_ms: list[float]
    warm_ipopt_ms: list[float]
    iters: list[int]
    n_success: int


def _eeg_sampler(model: NNSymbolicModel, rng: np.random.Generator) -> Callable[[], FloatArray]:
    """Return a callable drawing one realistic-magnitude EEG vector from the model's scalers."""
    n_ch = model.n_channels
    std = model.artifact.y_pipeline.standardizer
    if std is None:
        msg = "y-pipeline must carry a standardizer"
        raise ValueError(msg)
    mean = np.broadcast_to(std.center, (n_ch,))
    scale = np.broadcast_to(std.scale, (n_ch,))
    return lambda: mean + scale * rng.standard_normal(n_ch)


def benchmark_horizon(  # noqa: PLR0913
    model: NNSymbolicModel,
    horizon: int,
    *,
    dt: float,
    u_max: float,
    w_y: float,
    w_u: float,
    n_solves: int,
    seed: int,
    max_iter: int,
    max_cpu_time: float | None,
    expand: bool,
) -> BenchResult:
    """Build an MPC at ``horizon`` and time its cold solve and ``n_solves`` warm solves."""
    rng = np.random.default_rng(seed)
    sample = _eeg_sampler(model, rng)

    t0 = time.perf_counter()
    ctrl = MPCController(
        dt=dt,
        model=model,
        u_max=u_max,
        horizon=horizon,
        w_y=w_y,
        w_u=w_u,
        max_iter=max_iter,
        max_cpu_time=max_cpu_time,
        expand=expand,
    )
    build_s = time.perf_counter() - t0

    n_y = model.artifact.n_y
    k = 0
    # Fill the window: the first n_y-1 updates are warm-up (emit zero, no solve).
    for _ in range(n_y - 1):
        ctrl.update(k * dt, np.array([0.0]), sample())
        k += 1

    # The next update triggers the first (cold-started) solve.
    t0 = time.perf_counter()
    ctrl.update(k * dt, np.array([0.0]), sample())
    cold_update_ms = (time.perf_counter() - t0) * 1e3
    k += 1

    # A couple of untimed warm solves to reach steady warm-start behaviour.
    for _ in range(2):
        ctrl.update(k * dt, np.array([0.0]), sample())
        k += 1

    warm_update_ms: list[float] = []
    warm_ipopt_ms: list[float] = []
    iters: list[int] = []
    n_success = 0
    for _ in range(n_solves):
        t0 = time.perf_counter()
        _u, log = ctrl.update(k * dt, np.array([0.0]), sample())
        warm_update_ms.append((time.perf_counter() - t0) * 1e3)
        stats = ctrl._solver.stats()  # noqa: SLF001 -- introspection for benchmarking
        warm_ipopt_ms.append(float(stats.get("t_wall_total", float("nan")) * 1e3))
        iters.append(int(stats.get("iter_count", -1)))
        n_success += int(log.success)
        k += 1

    # Output-condensed NLP: lift only the per-step EEG output, not the full state.
    return BenchResult(
        horizon=horizon,
        n_vars=horizon * (model.n_controls + model.n_channels),
        n_eq=horizon * model.n_channels,
        build_s=build_s,
        cold_update_ms=cold_update_ms,
        warm_update_ms=warm_update_ms,
        warm_ipopt_ms=warm_ipopt_ms,
        iters=iters,
        n_success=n_success,
    )


def _fmt_row(r: BenchResult, n_solves: int) -> str:
    """Format one result as an aligned table row."""
    med_u, min_u = statistics.median(r.warm_update_ms), min(r.warm_update_ms)
    med_ip = statistics.median(r.warm_ipopt_ms)
    med_it = int(statistics.median(r.iters))
    return (
        f"{r.horizon:>3} {r.n_vars:>8} {r.n_eq:>8} {r.build_s:>8.1f} {r.cold_update_ms:>9.0f} "
        f"{med_u:>9.0f} {min_u:>9.0f} {med_ip:>10.0f} {med_it:>7} {r.n_success:>3}/{n_solves}"
    )


def main() -> None:
    """Parse arguments, run the horizon sweep, and print a comparison table."""
    parser = argparse.ArgumentParser(description="Benchmark MPCController OCP solve speed.")
    parser.add_argument("--artifact", required=True, help="NN-predictor artifact basename (.eqx/.json/.scalers.npz).")
    parser.add_argument("--horizons", type=int, nargs="+", default=[3, 5, 10, 20], help="Horizons to sweep.")
    parser.add_argument(
        "--dt", type=float, default=0.01, help="Controller step (s); should match the model's native dt."
    )
    parser.add_argument("--u-max", type=float, default=3.0, help="Per-electrode amplitude bound.")
    parser.add_argument("--w-y", type=float, default=1.0, help="EEG-power cost weight.")
    parser.add_argument("--w-u", type=float, default=0.01, help="Control-effort cost weight.")
    parser.add_argument("--n-solves", type=int, default=5, help="Timed warm solves per horizon.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the synthetic EEG inputs.")
    parser.add_argument("--max-iter", type=int, default=100, help="IPOPT iteration cap per solve.")
    parser.add_argument(
        "--max-cpu-time", type=float, default=None, help="Per-solve IPOPT wall-time budget (s); unset = unbounded."
    )
    parser.add_argument(
        "--expand",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Expand the NLP from MX to SX (slower here for the MLP-heavy graph; off by default).",
    )
    args = parser.parse_args()

    model = NNSymbolicModel.from_artifact(args.artifact)
    a = model.artifact
    print(f"Model: {args.artifact}", flush=True)
    print(
        f"  state_dim={model.state_shape[0]}  n_channels={model.n_channels}  n_controls={model.n_controls}  "
        f"n_y={a.n_y}  n_u={a.n_u}  native_horizon={a.horizon}  dt={a.dt:.3f}",
        flush=True,
    )
    print(
        f"  cost: w_y={args.w_y} w_u={args.w_u}  |u|<={args.u_max}  (warm solves timed: {args.n_solves})\n", flush=True
    )
    print("  H   n_vars     n_eq  build_s  cold_ms  warm_med  warm_min   ipopt_ms  iters     ok", flush=True)
    print("  " + "-" * 84, flush=True)

    for horizon in args.horizons:
        print(f"  building H={horizon} ...", end="\r", flush=True)
        r = benchmark_horizon(
            model,
            horizon,
            dt=args.dt,
            u_max=args.u_max,
            w_y=args.w_y,
            w_u=args.w_u,
            n_solves=args.n_solves,
            seed=args.seed,
            max_iter=args.max_iter,
            max_cpu_time=args.max_cpu_time,
            expand=args.expand,
        )
        print("  " + _fmt_row(r, args.n_solves), flush=True)


if __name__ == "__main__":
    main()
