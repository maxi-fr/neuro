from __future__ import annotations

import dataclasses
import importlib
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from trajopt.solvers.altro import ALTRO
from trajopt.solvers.boxqp import BoxQP
from trajopt.solvers.options import SolverOptions
from trajopt.transcription.ipopt import Ipopt
from trajopt.transcription.result import constraint_row_count
from trajopt.transcription.single_shooting import SingleShooting

from neuro.control.costs import has_whole_horizon_cost

if TYPE_CHECKING:
    from trajopt.problem import (
        BoundaryConditions,
        Problem,
    )
    from trajopt.program import Program, WarmStart
    from trajopt.transcription.result import Solver, SolverResult


_NO_DUALS = np.zeros(0, dtype=np.float64)

IPOPT_DEFAULTS: dict[str, Any] = {
    "print_level": 0,
    "hessian_approximation": "limited-memory",
    "tol": 1e-3,
    "acceptable_tol": 1e-2,
    "acceptable_iter": 5,
    "max_iter": 300,
}


def _default_solver(problem: Problem) -> Solver:
    """Select IPOPT single shooting, using homotopy for binary controls or Horizon constraints."""
    if problem.binary_control_indices or problem.constraints.horizon_constraints:
        from neuro.control.homotopy import SingleShootingBinaryIpopt  # noqa: PLC0415 -- homotopy uses solver defaults

        return SingleShootingBinaryIpopt(Ipopt(options=IPOPT_DEFAULTS))
    return SingleShooting(solver=Ipopt(options=IPOPT_DEFAULTS))


def ensure_solver_supports_objective(problem: Problem, solver: Solver) -> None:
    """Validate the solver against whole-horizon Costs and coupled control constraints."""
    if has_whole_horizon_cost(problem.obj.stage_cost) and not isinstance(solver, (SingleShooting, Ipopt)):
        msg = (
            f"{type(solver).__name__} expands costs per knot and cannot score the whole-horizon "
            "hinge cost; use a transcription solver (e.g. SingleShooting(Ipopt(...)))."
        )
        raise ValueError(msg)
    if isinstance(solver, BoxQP) and sum(problem.constraints.p) > 0:
        msg = (
            f"BoxQP only supports uncoupled box bounds, but problem has non-box constraints "
            f"(p={problem.constraints.p}); use ALTRO or Ipopt for Kirchhoff's law."
        )
        raise ValueError(msg)


def _instantiate_solver(class_path: str, cfg: dict[str, Any]) -> Solver:
    """Import and instantiate a Solver from its class_path and config, defaulting Ipopt options."""
    if not isinstance(class_path, str) or "." not in class_path:
        msg = f"solver 'class_path' must be a dot-separated import path, got {class_path!r}"
        raise ValueError(msg)

    module_name, class_name = class_path.rsplit(".", 1)
    target_cls = getattr(importlib.import_module(module_name), class_name)

    if "solver" in cfg and isinstance(cfg["solver"], dict):
        cfg["solver"] = _build_solver(cfg["solver"])

    if issubclass(target_cls, Ipopt):
        cfg["options"] = {**IPOPT_DEFAULTS, **cfg.get("options", {})}
    elif issubclass(target_cls, SingleShooting) and "solver" not in cfg:
        cfg["solver"] = Ipopt(options=dict(IPOPT_DEFAULTS))
    elif "options" in cfg and isinstance(cfg["options"], dict) and issubclass(target_cls, (ALTRO, BoxQP)):
        cfg["options"] = SolverOptions(**cfg["options"])

    res = target_cls(**cfg) if cfg else target_cls()
    if not isinstance(res, (ALTRO, BoxQP, SingleShooting, Ipopt)) and not hasattr(res, "solve"):
        msg = f"Target class {class_path} does not implement Solver interface"
        raise TypeError(msg)
    return res


def _build_solver(spec: dict[str, Any] | Solver | None, problem: Problem | None = None) -> Solver:
    """Instantiate a Solver from a ``{class_path, ...}`` dict or return the benchmark default.

    Parameters
    ----------
    spec
        Solver instance, config dict with ``'class_path'``, or ``None``.
    problem
        Optional Problem instance used to select the benchmark-winning default when ``spec`` is None.
    """
    if spec is None:
        if problem is None:
            msg = "Cannot select default solver without a Problem instance"
            raise ValueError(msg)
        return _default_solver(problem)

    if not isinstance(spec, dict):
        if not hasattr(spec, "solve"):
            msg = f"Invalid solver object of type {type(spec).__name__}: must be a dict with 'class_path' or implement .solve()"
            raise TypeError(msg)
        if problem is not None:
            ensure_solver_supports_objective(problem, spec)
        return spec

    cfg = spec.copy()
    if "class_path" not in cfg:
        msg = f"solver config must contain 'class_path', got keys: {list(cfg.keys())}"
        raise ValueError(msg)

    solver_instance = _instantiate_solver(cfg.pop("class_path"), cfg)
    if problem is not None:
        ensure_solver_supports_objective(problem, solver_instance)
    return solver_instance


@dataclasses.dataclass(frozen=True)
class CanonicalDuals:
    """Solver adapter dropping duals a backend reports in a layout the warm start cannot shift.

    ``SingleShooting`` eliminates the states, so its primal is the control trajectory alone and
    Ipopt hands back bound duals of length ``(N - 1) * m`` and constraint duals over the user rows
    only. ``WarmStart`` is defined over the full Primal Vector and the canonical row order, and
    trajopt folds the backend's duals in without checking, so the next ``MPC.shift`` fails on the
    bound duals' shape and would mis-gather the constraint duals. Replacing a mismatched vector
    with the empty one puts the backend in the case the driver already handles: it returned no
    duals, so the warm start keeps its own.

    Only the backends that actually report a shifted layout are wrapped, because the adapter hides
    the backend's own type: trajopt classifies a solver by ``isinstance`` -- for the benchmark's
    ``linearizing`` column and for `ensure_solver_supports_objective` -- and a wrapper it cannot
    see through would misreport every solver it covers.
    """

    solver: Solver

    def solve(self, program: Program, bc: BoundaryConditions, ws: WarmStart) -> SolverResult:
        """Solve through the wrapped backend, blanking duals whose length is not the canonical one."""
        res = self.solver.solve(program, bc, ws)
        problem = program.problem
        N, n, m = int(problem.N), int(problem.model.n), int(problem.model.m)
        lam_ok = len(res.lam) in (0, constraint_row_count(problem))
        mu_ok = len(res.mu) in (0, N * n + (N - 1) * m)
        if lam_ok and mu_ok:
            return res
        # ``_replace`` carries every other field of the backend's own NamedTuple through untouched,
        # including ones the SolverResult Protocol does not name -- ALTRO's ``al`` among them, which
        # the driver reads back off the result to restore its multipliers.
        # ``SolverResult`` is a Protocol, so ``_replace`` is not on it; every backend satisfying
        # it is a NamedTuple, which is what makes the copy possible.
        return cast("Any", res)._replace(
            lam=res.lam if lam_ok else _NO_DUALS,
            mu=res.mu if mu_ok else _NO_DUALS,
        )


def canonicalize_duals(solver: Solver) -> Solver:
    """Wrap `solver` in :class:`CanonicalDuals` when its backend reports duals in a shifted layout.

    Only the Ipopt-backed transcriptions do. Every other backend is returned as it came, so trajopt
    can still classify it by ``isinstance`` -- which is what decides the benchmark's ``linearizing``
    column and what `ensure_solver_supports_objective` reads.
    """
    return CanonicalDuals(solver) if isinstance(solver, (SingleShooting, Ipopt)) else solver
