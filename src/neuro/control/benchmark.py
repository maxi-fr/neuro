from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from trajopt.benchmarks import (
    ClosedLoopComparison,
    SolverComparison,
    compare_solvers,
    compare_solvers_closed_loop,
)
from trajopt.problem import BoundaryConditions, Problem
from trajopt.program import WarmStart
from trajopt.solvers.altro import ALTRO
from trajopt.solvers.boxqp import BoxQP
from trajopt.solvers.ilqr import ILQR
from trajopt.solvers.options import SolverOptions
from trajopt.transcription.ipopt import Ipopt
from trajopt.transcription.osqp import OSQP
from trajopt.transcription.single_shooting import SingleShooting

from neuro.control.mpc import (
    build_observable_problem,
    build_waveform_problem,
    canonicalize_duals,
)
from neuro.predictor.inference import InferencePredictor

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from trajopt.transcription.result import Solver

    from neuro.spectral import HealthyReference

logger = logging.getLogger(__name__)

# An infeasible transcription can grind rather than fail, so every Ipopt backend is capped: a row
# that hits the cap is reported as a non-converged solver instead of hanging the benchmark.
_IPOPT_DEFAULTS = {"print_level": 0, "hessian_approximation": "limited-memory", "max_iter": 300}

_LABELS = {
    "single_shooting": "SingleShooting(Ipopt)",
    "ipopt": "Ipopt(MultipleShooting)",
    "altro": "ALTRO",
    "boxqp": "BoxQP",
    "ilqr": "ILQR",
    "osqp": "OSQP",
}


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
        return SingleShooting(solver=Ipopt(options={**_IPOPT_DEFAULTS, **opts_dict}))
    if name_lower == "ipopt":
        return Ipopt(options={**_IPOPT_DEFAULTS, **opts_dict})
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


def initial_conditions(problem: Problem, x0: jax.Array) -> tuple[BoundaryConditions, WarmStart]:
    """Boundary conditions and a cold warm start for `problem` measured at ``x0`` of shape ``(n,)``.

    No reference window is set: every neurostimulation objective bakes its target in, so there is
    nothing for the driver to retarget at run time.
    """
    return (
        BoundaryConditions(x0=x0, t0=jnp.zeros((), dtype=jnp.float64)),
        WarmStart.cold(problem, x0),
    )


def describe_problem(problem: Problem) -> str:
    """Label a Problem by its grid and both transcriptions' decision-variable counts.

    The count is the header rather than a footnote because it is what separates the two Ipopt rows:
    single shooting eliminates the states, so its primal is ``(N - 1) * m`` and does not grow with
    the Predictor's trailing window, while the direct transcription carries ``N * n`` state
    variables and the matching defect rows on top.
    """
    N, n, m = int(problem.N), int(problem.model.n), int(problem.model.m)
    return f"{type(problem.model).__name__} (N={N}, n={n}, m={m}; SS {(N - 1) * m} vars, MS {N * n + (N - 1) * m} vars)"


def _compare_per_solver(
    solvers: Mapping[str, Solver],
    compare: Callable[[str, Solver], Sequence[Any]],
) -> list[Any]:
    """Compare each solver on its own, dropping any that raises rather than losing the whole table.

    trajopt lets a solver's exception propagate out of its `compare_solvers`, which is right for a
    library but wrong here: a formulation like multiple-shooting Ipopt that cannot take this
    transcription would take every other row down with it. A raising solver is logged at ERROR --
    loud enough that a genuine bug in the setup is not mistaken for a solver the formulation
    defeats -- and left out of the table.
    """
    rows: list[Any] = []
    for label, solver in solvers.items():
        try:
            rows.extend(compare(label, solver))
        except Exception:
            logger.exception("benchmark dropped '%s'", label)
    return rows


def compare_open_loop(
    problem: Problem,
    bc: BoundaryConditions,
    ws: WarmStart,
    solvers: Mapping[str, Solver],
    *,
    n_repeats: int,
) -> SolverComparison:
    """Open-loop comparison over `solvers`, dropping any that raises instead of losing the table."""
    rows = _compare_per_solver(
        solvers,
        lambda label, solver: compare_solvers(problem, bc, ws, {label: solver}, n_repeats=n_repeats).rows,
    )
    return SolverComparison(model=describe_problem(problem), n_repeats=n_repeats, rows=tuple(rows))


def compare_closed_loop(
    problem: Problem,
    bc: BoundaryConditions,
    ws: WarmStart,
    solvers: Mapping[str, Solver],
    *,
    num_steps: int,
) -> ClosedLoopComparison:
    """Closed-loop comparison over `solvers`, dropping any that raises, as `compare_open_loop` does."""
    rows = _compare_per_solver(
        solvers,
        lambda label, solver: compare_solvers_closed_loop(problem, bc, ws, {label: solver}, num_steps=num_steps).rows,
    )
    return ClosedLoopComparison(model=describe_problem(problem), num_steps=num_steps, rows=tuple(rows))


# A cap on the benchmark's own runtime, not a control deadline. On the deployment-scale
# transcriptions the interior-point rows do not converge, so an uncapped call runs to `max_iter`,
# which at 49k variables is hours per solve and would stall the suite. A row that hits the cap
# reports the iterations it reached, which is the number that decides whether it was ever close.
_IPOPT_MAX_CPU_SECONDS = 120.0


def benchmark_solvers(*names: str) -> dict[str, Solver]:
    """Build the labelled solver set the benchmarks compare, capping the Ipopt-backed rows' runtime."""
    return {
        _LABELS[name]: canonicalize_duals(
            get_benchmark_solver(
                name,
                options={"max_cpu_time": _IPOPT_MAX_CPU_SECONDS}
                if "ipopt" in name or name == "single_shooting"
                else None,
            )
        )
        for name in names
    }


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
    reference: HealthyReference | None = None,
    kirchhoff: bool = True,
    reduce_kirchhoff: bool = False,
    n_repeats: int = 5,
    num_steps: int = 10,
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
    reference
        :class:`~neuro.spectral.HealthyReference` container carrying empirical channel means.
    kirchhoff
        Whether to enforce Kirchhoff Current Law as a hard equality constraint.
    reduce_kirchhoff
        Satisfy Kirchhoff by null-space reduction instead, exactly rather than to a tolerance,
        the per-electrode limit becoming a coupled polytope on the reduced controls.
    n_repeats
        Repeats for open-loop timing.
    num_steps
        Steps for closed-loop MPC simulation.
    """
    problem = build_waveform_problem(
        artifact,
        horizon=horizon,
        u_max=u_max,
        w_y=w_y,
        w_u=w_u,
        reference=reference,
        kirchhoff=kirchhoff,
        reduce_kirchhoff=reduce_kirchhoff,
    )
    model = problem.model
    if not isinstance(model, InferencePredictor):
        msg = "problem.model must implement InferencePredictor"
        raise TypeError(msg)
    x0 = jnp.zeros(model.n)
    bc, ws = initial_conditions(problem, x0)

    names = ["single_shooting", "ipopt", "altro", "osqp"]
    if not kirchhoff and not reduce_kirchhoff:
        # The DDP backends clamp elementwise, so they have a seam for the box bounds and for
        # nothing else: they join only when neither Kirchhoff formulation is in the problem.
        names.extend(("boxqp", "ilqr"))
    solvers = benchmark_solvers(*names)

    open_comp = compare_open_loop(problem, bc, ws, solvers, n_repeats=n_repeats)
    closed_comp = compare_closed_loop(problem, bc, ws, solvers, num_steps=num_steps)

    return open_comp, closed_comp


def run_observable_benchmark(  # noqa: PLR0913 -- benchmark configuration knobs
    artifact: str | Path,
    reference: HealthyReference | None = None,
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
    reference
        :class:`~neuro.spectral.HealthyReference` container carrying healthy Observable envelope.
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
        reference=reference,
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
    bc, ws = initial_conditions(problem, jnp.asarray(x0_np))

    solvers = benchmark_solvers("single_shooting", "ipopt", "altro", "osqp")

    open_comp = compare_open_loop(problem, bc, ws, solvers, n_repeats=n_repeats)
    closed_comp = compare_closed_loop(problem, bc, ws, solvers, num_steps=num_steps)

    return open_comp, closed_comp
