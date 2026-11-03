from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from trajopt.solvers.altro import ALTRO
from trajopt.solvers.boxqp import BoxQP
from trajopt.solvers.options import SolverOptions
from trajopt.transcription.ipopt import Ipopt
from trajopt.transcription.single_shooting import SingleShooting

from neuro.control.costs import has_whole_horizon_cost

if TYPE_CHECKING:
    from trajopt.problem import Problem
    from trajopt.transcription.result import Solver


IPOPT_DEFAULTS: dict[str, Any] = {
    "print_level": 0,
    "hessian_approximation": "limited-memory",
    "tol": 1e-3,
    "acceptable_tol": 1e-2,
    "acceptable_iter": 5,
    "max_iter": 300,
}


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
    """Instantiate a configured Solver or select stock IPOPT single shooting.

    Parameters
    ----------
    spec
        Solver instance, config dict with ``'class_path'``, or ``None``.
    problem
        Optional Problem used to validate the configured Solver against its Objective.
    """
    if spec is None:
        return SingleShooting(solver=Ipopt(options=IPOPT_DEFAULTS))

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
