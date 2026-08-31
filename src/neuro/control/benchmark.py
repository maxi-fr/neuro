from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
from trajopt.benchmarks import ClosedLoopComparison, SolverComparison, compare_solvers, compare_solvers_closed_loop
from trajopt.problem import MPCState
from trajopt.solvers.altro import ALTRO
from trajopt.solvers.boxqp import BoxQP
from trajopt.solvers.ilqr import ILQR
from trajopt.solvers.options import SolverOptions
from trajopt.transcription.ipopt import Ipopt
from trajopt.transcription.osqp import OSQP
from trajopt.transcription.single_shooting import SingleShooting

from neuro.control.mpc import (
    build_bipolar_waveform_problem,
    build_observable_problem,
    build_waveform_problem,
)
from neuro.predictor.inference import InferencePredictor

if TYPE_CHECKING:
    from pathlib import Path

    from trajopt.transcription.result import Solver

_BIPOLAR_CONTROLS = 2


def get_benchmark_solver(
    name: str,
    *,
    options: dict[str, Any] | SolverOptions | None = None,
) -> Solver:
    """Instantiate a configured solver by name for neurostimulation benchmarks.

    Parameters
    ----------
    name
        One of `'single_shooting'`, `'ipopt'`, `'altro'`, `'boxqp'`, `'ilqr'`, `'osqp'`.
    options
        Optional solver options dictionary or SolverOptions dataclass.
    """
    name_lower = name.lower()
    opts_dict: dict[str, Any] = options if isinstance(options, dict) else {}
    if name_lower in ("single_shooting", "ss"):
        ipopt_opts = {"print_level": 0, "hessian_approximation": "limited-memory", **opts_dict}
        return SingleShooting(solver=Ipopt(options=ipopt_opts))
    if name_lower == "ipopt":
        ipopt_opts = {"print_level": 0, "hessian_approximation": "limited-memory", **opts_dict}
        return Ipopt(options=ipopt_opts)
    if name_lower == "altro":
        solver_options = options if isinstance(options, SolverOptions) else SolverOptions(**opts_dict)
        return ALTRO(options=solver_options)
    if name_lower in ("boxqp", "box_ilqr"):
        solver_options = options if isinstance(options, SolverOptions) else SolverOptions(**opts_dict)
        return BoxQP(options=solver_options)
    if name_lower == "ilqr":
        solver_options = options if isinstance(options, SolverOptions) else SolverOptions(**opts_dict)
        return ILQR(options=solver_options)
    if name_lower == "osqp":
        osqp_opts = {"eps_abs": 1e-6, "eps_rel": 1e-6, "max_iter": 4000, **opts_dict}
        return OSQP(options=osqp_opts)
    msg = f"Unknown solver name '{name}'. Expected one of: single_shooting, ipopt, altro, boxqp, ilqr, osqp."
    raise ValueError(msg)


def format_open_loop_table(comparison: SolverComparison) -> str:
    """Format an open-loop SolverComparison into a markdown table."""
    lines = [
        f"### Open-Loop Solver Comparison: {comparison.model} (n_repeats={comparison.n_repeats})",
        "",
        "| Solver | Success | Iterations | Cost | Constraint Violation | First Call (s) | Median Time (ms) | Min Time (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for row in comparison.rows:
        succ_str = "Yes" if row.success else "No"
        med_ms = row.timing.median_time_s * 1000.0
        min_ms = row.timing.min_time_s * 1000.0
        first_s = row.timing.first_call_time_s
        lines.append(
            f"| {row.solver} | {succ_str} | {row.iterations} | {row.cost:.5f} | {row.constraint_violation:.2e} | "
            f"{first_s:.3f} | {med_ms:.2f} | {min_ms:.2f} |"
        )
    return "\n".join(lines)


def format_closed_loop_table(comparison: ClosedLoopComparison) -> str:
    """Format a closed-loop ClosedLoopComparison into a markdown table."""
    lines = [
        f"### Closed-Loop Receding Horizon MPC Comparison: {comparison.model} (num_steps={comparison.num_steps})",
        "",
        "| Solver | Mean Latency (ms) | Median (ms) | P95 (ms) | P99 (ms) | Frequency (Hz) | Warmstart Speedup |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for row in comparison.rows:
        st = row.stats
        lines.append(
            f"| {row.solver} | {st.mean_latency_s * 1000.0:.2f} | {st.median_latency_s * 1000.0:.2f} | "
            f"{st.p95_latency_s * 1000.0:.2f} | {st.p99_latency_s * 1000.0:.2f} | "
            f"{st.sustained_frequency_hz:.1f} | {st.warmstart_speedup:.2f}x |"
        )
    return "\n".join(lines)


def run_waveform_benchmark(  # noqa: PLR0913 -- benchmark configuration knobs
    artifact: str | Path,
    *,
    horizon: int = 10,
    u_max: float = 0.5,
    w_y: float = 1.0,
    w_u: float = 0.1,
    kirchhoff: bool = True,
    n_repeats: int = 5,
    num_steps: int = 10,
    include_bipolar_boxqp: bool = True,
) -> tuple[SolverComparison, ClosedLoopComparison]:
    """Run open-loop and closed-loop benchmarks on a Waveform Optimal Control Problem.

    Parameters
    ----------
    artifact
        Path to the Waveform Predictor checkpoint.
    horizon
        Control Horizon in steps.
    u_max
        Per-electrode current limit.
    w_y
        Tracking cost weight.
    w_u
        Quadratic control effort weight.
    kirchhoff
        Whether to enforce Kirchhoff Current Law equality.
    n_repeats
        Repeats for open-loop timing.
    num_steps
        Steps for closed-loop MPC simulation.
    include_bipolar_boxqp
        Whether to include the reduced bipolar Box-iLQR formulation when model has m=2.
    """
    problem = build_waveform_problem(
        artifact,
        horizon=horizon,
        u_max=u_max,
        w_y=w_y,
        w_u=w_u,
        kirchhoff=kirchhoff,
    )
    model = problem.model
    if not isinstance(model, InferencePredictor):
        msg = "problem.model must implement InferencePredictor"
        raise TypeError(msg)
    x0 = jnp.zeros(model.n)
    state = MPCState.initial(problem, x0=x0, dt=model.dt)

    solvers: dict[str, Solver] = {
        "SingleShooting(Ipopt)": get_benchmark_solver("single_shooting"),
        "ALTRO": get_benchmark_solver("altro"),
        "OSQP": get_benchmark_solver("osqp"),
    }

    if not kirchhoff:
        solvers["BoxQP"] = get_benchmark_solver("boxqp")

    open_comp = compare_solvers(problem, state, solvers, n_repeats=n_repeats)
    closed_comp = compare_solvers_closed_loop(problem, state, solvers, num_steps=num_steps)

    if include_bipolar_boxqp and model.m == _BIPOLAR_CONTROLS and kirchhoff:
        prob_bipolar = build_bipolar_waveform_problem(
            artifact,
            horizon=horizon,
            u_max=u_max,
            w_y=w_y,
            w_u=w_u,
        )
        bipolar_model = prob_bipolar.model
        if not isinstance(bipolar_model, InferencePredictor):
            msg = "prob_bipolar.model must implement InferencePredictor"
            raise TypeError(msg)
        state_bipolar = MPCState.initial(prob_bipolar, x0=x0, dt=bipolar_model.dt)
        bipolar_solvers: dict[str, Solver] = {"Bipolar-BoxQP": get_benchmark_solver("boxqp")}
        bipolar_open = compare_solvers(prob_bipolar, state_bipolar, bipolar_solvers, n_repeats=n_repeats)
        bipolar_closed = compare_solvers_closed_loop(prob_bipolar, state_bipolar, bipolar_solvers, num_steps=num_steps)

        open_comp = SolverComparison(
            model=open_comp.model,
            n_repeats=n_repeats,
            rows=(*open_comp.rows, *bipolar_open.rows),
        )
        closed_comp = ClosedLoopComparison(
            model=closed_comp.model,
            num_steps=num_steps,
            rows=(*closed_comp.rows, *bipolar_closed.rows),
        )

    return open_comp, closed_comp


def run_observable_benchmark(  # noqa: PLR0913 -- benchmark configuration knobs
    artifact: str | Path,
    envelope_ref: str | Path,
    *,
    horizon: int = 4,
    u_max: float = 0.5,
    w_u: float = 1.0,
    w_hinge: float = 2.0,
    kirchhoff: bool = True,
    n_repeats: int = 5,
    num_steps: int = 10,
) -> tuple[SolverComparison, ClosedLoopComparison]:
    """Run open-loop and closed-loop benchmarks on an Observable Optimal Control Problem.

    Parameters
    ----------
    artifact
        Path to the Observable Predictor checkpoint.
    envelope_ref
        Path to the healthy Observable envelope npz.
    horizon
        Control Horizon in Frames.
    u_max
        Per-electrode current limit.
    w_u
        Quadratic control effort weight.
    w_hinge
        Hinge cost weight against healthy envelope.
    kirchhoff
        Whether to enforce Kirchhoff Current Law equality.
    n_repeats
        Repeats for open-loop timing.
    num_steps
        Steps for closed-loop MPC simulation.
    """
    problem = build_observable_problem(
        artifact,
        horizon=horizon,
        u_max=u_max,
        w_u=w_u,
        w_hinge=w_hinge,
        envelope_ref=envelope_ref,
        kirchhoff=kirchhoff,
    )
    model = problem.model
    if not isinstance(model, InferencePredictor):
        msg = "problem.model must implement InferencePredictor"
        raise TypeError(msg)
    n_outputs = getattr(model, "n_outputs", 1)
    n_y = getattr(model, "n_y", 1)
    rng = np.random.default_rng(42)
    x0_np = np.asarray(model.initial_state(), dtype=np.float64)
    x0_np[: n_y * n_outputs] = rng.uniform(-1.0, 1.0, n_y * n_outputs)
    state = MPCState.initial(problem, x0=jnp.asarray(x0_np), dt=model.dt)

    solvers: dict[str, Solver] = {
        "SingleShooting(Ipopt)": get_benchmark_solver("single_shooting"),
        "ALTRO": get_benchmark_solver("altro"),
        "OSQP": get_benchmark_solver("osqp"),
    }

    open_comp = compare_solvers(problem, state, solvers, n_repeats=n_repeats)
    closed_comp = compare_solvers_closed_loop(problem, state, solvers, num_steps=num_steps)

    return open_comp, closed_comp
