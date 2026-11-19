from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

import equinox as eqx
import numpy as np
import trajopt.transcription.single_shooting as ss
from trajopt.costs.l1 import L1ControlCost
from trajopt.costs.sum import cost_terms
from trajopt.solvers.altro import ALTRO
from trajopt.solvers.boxqp import BoxQP
from trajopt.solvers.options import SolverOptions
from trajopt.transcription.ipopt import Ipopt
from trajopt.transcription.single_shooting import SingleShooting

from neuro.control.condensed import CondensedIPOPT, CondensedOSQP
from neuro.control.costs import AffineObservableHinge, CondensedL1ControlCost, has_whole_horizon_cost
from neuro.control.multiple_shooting import MultipleShooting
from neuro.control.narx import NarxShooting

if TYPE_CHECKING:
    from trajopt.problem import Problem
    from trajopt.trajectory import Trajectory
    from trajopt.transcription.result import Solver

_MAX_ADJOINT_STATE_DIM: int = 1000

# High-dimensional state vectors (e.g. Observable CNN with n=11,931) make backward adjoint
# multiplier recovery evaluate an 11,931 x 11,931 state Jacobian 17 times in JAX (1.14 GB/knot,
# ~165s/solve, triggering system out-of-memory). Since duals are unused, guard against this.
if hasattr(ss, "_recover_canonical_multipliers"):
    _orig_recover = ss._recover_canonical_multipliers  # noqa: SLF001 -- patching internal multiplier recovery for large states

    def _safe_recover(problem: Problem, traj: Trajectory, info: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """Recover canonical Multipliers with cached derivatives below the state-dimension limit."""
        if int(problem.model.n) > _MAX_ADJOINT_STATE_DIM:
            return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
        return _orig_recover(problem, traj, info)

    # Compile the pure JAX kernels while keeping the upstream NumPy assembly and dimension guard.
    vars(ss).update(
        _state_stationarity_rhs=eqx.filter_jit(ss._state_stationarity_rhs),  # noqa: SLF001 -- cache lifted Objective and constraint differentiation
        _backward_adjoint_recursion=eqx.filter_jit(ss._backward_adjoint_recursion),  # noqa: SLF001 -- cache Dynamics differentiation
        _recover_canonical_multipliers=_safe_recover,
    )


IPOPT_DEFAULTS: dict[str, Any] = {
    "print_level": 0,
    "hessian_approximation": "limited-memory",
    "tol": 1e-3,
    "acceptable_tol": 1e-2,
    "acceptable_iter": 5,
    "max_iter": 300,
}


def ensure_solver_supports_objective(problem: Problem, solver: Solver) -> None:
    """Validate solver compatibility with physical and epigraph constraints and Costs."""
    stage_terms = cost_terms(problem.obj.stage_cost)
    if any(isinstance(term, CondensedL1ControlCost) for term in stage_terms) and not isinstance(
        solver, (CondensedOSQP, CondensedIPOPT)
    ):
        msg = "affine_hinge with exact L1 requires CondensedOSQP or CondensedIPOPT"
        raise ValueError(msg)
    if any(isinstance(term, AffineObservableHinge) for term in stage_terms) and not isinstance(
        solver, (CondensedOSQP, CondensedIPOPT, Ipopt, SingleShooting, NarxShooting, MultipleShooting)
    ):
        msg = "affine_hinge requires CondensedOSQP, CondensedIPOPT, or IPOPT"
        raise ValueError(msg)
    if has_whole_horizon_cost(problem.obj.stage_cost) and (
        isinstance(solver, CondensedIPOPT)
        or not isinstance(solver, (SingleShooting, Ipopt, NarxShooting, MultipleShooting))
    ):
        msg = (
            f"{type(solver).__name__} expands costs per knot and cannot score the whole-horizon "
            "hinge cost; use a transcription solver (e.g. SingleShooting(Ipopt(...)))."
        )
        raise ValueError(msg)
    if isinstance(solver, BoxQP) and (
        sum(problem.constraints.p) > 0 or any(isinstance(term, L1ControlCost) for term in stage_terms)
    ):
        msg = (
            "BoxQP only supports uncoupled box bounds. Use ALTRO or Ipopt for "
            "non-box constraints, including exact L1 epigraph constraints."
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

    if issubclass(target_cls, Ipopt) and not issubclass(target_cls, CondensedIPOPT):
        cfg["options"] = {**IPOPT_DEFAULTS, **cfg.get("options", {})}
    elif issubclass(target_cls, (SingleShooting, NarxShooting, MultipleShooting)) and "solver" not in cfg:
        cfg["solver"] = Ipopt(options=dict(IPOPT_DEFAULTS))
    elif "options" in cfg and isinstance(cfg["options"], dict) and issubclass(target_cls, (ALTRO, BoxQP)):
        cfg["options"] = SolverOptions(**cfg["options"])

    res = target_cls(**cfg) if cfg else target_cls()
    if not isinstance(res, (ALTRO, BoxQP, SingleShooting, Ipopt, NarxShooting, MultipleShooting)) and not hasattr(
        res, "solve"
    ):
        msg = f"Target class {class_path} does not implement Solver interface"
        raise TypeError(msg)
    return res


def _build_solver(spec: dict[str, Any] | Solver | None, problem: Problem | None = None) -> Solver:
    """Instantiate and validate a configured Solver or stock IPOPT single shooting.

    Parameters
    ----------
    spec
        Solver instance, config dict with ``'class_path'``, or ``None``.
    problem
        Optional Problem used to validate the configured Solver against its Objective.
    """
    if spec is None:
        spec = SingleShooting(solver=Ipopt(options=IPOPT_DEFAULTS))

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
