from __future__ import annotations

import dataclasses
import importlib
from typing import TYPE_CHECKING, Any, Self

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from simulate.controller import Controller
from trajopt.cones import ZeroCone
from trajopt.constraints.bounds import ControlBound
from trajopt.constraints.constraint_list import ConstraintList
from trajopt.constraints.linear import LinearConstraint
from trajopt.costs.objective import Objective
from trajopt.costs.quadratic import DiagonalCost
from trajopt.dynamics.base import DiscreteDynamics
from trajopt.problem import MPCState, Problem
from trajopt.solvers.altro import ALTRO
from trajopt.solvers.boxqp import BoxQP
from trajopt.solvers.options import SolverOptions
from trajopt.transcription.ipopt import Ipopt
from trajopt.transcription.single_shooting import SingleShooting

from neuro.control.costs import (
    ExcludeInitialKnotState,
    KirchhoffPenaltyCost,
    L1ControlCost,
    ObservableHingeCost,
    SpectralHingeCost,
    StateOutputs,
    SumCost,
    has_whole_horizon_cost,
)
from neuro.predictor.inference import InferencePredictor, ObservableMLPModel, WaveformMLPModel
from neuro.spectral import ObservableEnvelope, PsdEnvelope

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import ArrayLike
    from trajopt.costs.base import CostFunction
    from trajopt.transcription.result import Solver

    from neuro.types import FloatArray


@dataclasses.dataclass(frozen=True)
class TrajOptMPCLog:
    """Per-step diagnostics: the applied control, optimal cost, solver success, and warm-up flag."""

    u: FloatArray
    cost: float
    success: bool
    warmup: bool


def _build_problem(spec: dict[str, Any] | Problem) -> tuple[Problem, MPCState | None]:
    """Instantiate a Problem from an instance or a ``{class_path, ...}`` dict, plus an optional MPCState."""
    if isinstance(spec, Problem):
        return spec, None
    cfg = spec.copy()
    class_path: str = cfg.pop("class_path")
    module_name, func_name = class_path.rsplit(".", 1)
    target = getattr(importlib.import_module(module_name), func_name)
    res = target(**cfg) if cfg else target()
    if isinstance(res, tuple):
        state = res[1] if len(res) > 1 and isinstance(res[1], MPCState) else None
        return res[0], state
    return res, None


def kirchhoff_constraint(n: int, m: int) -> LinearConstraint:
    """Kirchhoff current-law equality: the per-electrode currents sum to zero at each step.

    The incumbent NLP's ``_sum_to_zero`` equality, expressed with
    ``trajopt.constraints.linear``: ``A @ u - b = 0`` with ``A = ones(1, m)`` and ``b = 0``
    on the control block of ``z = [x; u]`` at every non-terminal knot.
    """
    return LinearConstraint(
        n=n,
        m=m,
        A=jnp.ones((1, m)),
        b=jnp.zeros(1),
        sense=ZeroCone(),
        inds=range(n, n + m),
    )


def _combine_costs(costs: list[CostFunction]) -> CostFunction:
    """Return the single stage cost, wrapping several sub-costs in a :class:`SumCost`."""
    return SumCost(costs) if len(costs) > 1 else costs[0]


def _spectral_envelope(psd_ref: str | Path | None, w_psd: float) -> PsdEnvelope | None:
    """Load the healthy PSD envelope when ``w_psd`` enables the spectral hinge, else ``None``."""
    if w_psd <= 0:
        return None
    if psd_ref is None:
        msg = "psd_ref must be provided when w_psd > 0"
        raise ValueError(msg)
    return PsdEnvelope.load(psd_ref)


def _observable_envelope(envelope_ref: str | Path | None, w_hinge: float) -> ObservableEnvelope | None:
    """Load the healthy Observable envelope when ``w_hinge`` enables the hinge, else ``None``."""
    if w_hinge <= 0:
        return None
    if envelope_ref is None:
        msg = "envelope_ref must be provided when w_hinge > 0"
        raise ValueError(msg)
    return ObservableEnvelope.load(envelope_ref)


def _assemble_problem(
    model: DiscreteDynamics,
    objective: Objective,
    *,
    N: int,
    u_max: ArrayLike,
    kirchhoff: bool,
) -> Problem:
    """Add the control box bounds (and optional Kirchhoff equality) and build the ``Problem``."""
    n, m = model.n, model.m
    u_max_arr = np.broadcast_to(np.atleast_1d(np.asarray(u_max, dtype=np.float64)), (m,))
    constraints = ConstraintList(n=n, m=m, N=N)
    constraints.add_constraint(ControlBound(n=n, m=m, u_min=-u_max_arr, u_max=u_max_arr), range(N - 1))
    if kirchhoff:
        constraints.add_constraint(kirchhoff_constraint(n, m), range(N - 1))
    return Problem(model=model, obj=objective, constraints=constraints, N=N)


def _default_solver(problem: Problem) -> Solver:
    """Select the default solver: BoxQP for bipolar models, SingleShooting(Ipopt) for general problems."""
    if isinstance(problem.model, BipolarReducedModel):
        return BoxQP()
    ipopt = Ipopt(options={"print_level": 0, "hessian_approximation": "limited-memory"})
    return SingleShooting(solver=ipopt)


def ensure_solver_supports_objective(problem: Problem, solver: Solver) -> None:
    """Raise when a solver does not support the problem's objective or constraints."""
    if has_whole_horizon_cost(problem.obj.stage_cost) and not isinstance(solver, (SingleShooting, Ipopt)):
        msg = (
            f"{type(solver).__name__} expands costs per knot and cannot score the whole-horizon "
            "hinge cost; use a transcription solver (e.g. SingleShooting(Ipopt(...)))."
        )
        raise ValueError(msg)
    if isinstance(solver, BoxQP) and sum(problem.constraints.p) > 0:
        msg = (
            f"BoxQP only supports uncoupled box bounds, but problem has non-box constraints "
            f"(p={problem.constraints.p}); use ALTRO or BipolarReducedModel for Kirchhoff equality constraints."
        )
        raise ValueError(msg)


def _instantiate_solver(class_path: str, cfg: dict[str, Any]) -> Solver:
    """Import and instantiate a solver from its class_path and config."""
    if not isinstance(class_path, str) or "." not in class_path:
        msg = f"solver 'class_path' must be a dot-separated import path, got {class_path!r}"
        raise ValueError(msg)

    module_name, class_name = class_path.rsplit(".", 1)
    target_cls = getattr(importlib.import_module(module_name), class_name)

    if "solver" in cfg and isinstance(cfg["solver"], dict):
        cfg["solver"] = _build_solver(cfg["solver"])

    if "options" in cfg and isinstance(cfg["options"], dict) and issubclass(target_cls, (ALTRO, BoxQP)):
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


def _validate_waveform_envelope(envelope: PsdEnvelope, model: WaveformMLPModel) -> None:
    """Ensure the healthy spectral envelope matches the predictor's channels and sampling rate."""
    if envelope.power.shape[0] != model.n_channels:
        msg = f"envelope channel count ({envelope.power.shape[0]}) does not match model channel count ({model.n_channels})."
        raise ValueError(msg)
    model_fs = 1.0 / model.dt
    if not np.isclose(envelope.fs, model_fs, rtol=1e-9):
        msg = f"envelope sampling rate ({envelope.fs:g} Hz) does not match model sampling rate ({model_fs:g} Hz)."
        raise ValueError(msg)


def _validate_observable_envelope(envelope: ObservableEnvelope, model: ObservableMLPModel) -> None:
    """Ensure the healthy Observable envelope matches the predictor's channels, rate, and geometry."""
    if envelope.power.shape[0] != model.n_channels:
        msg = f"envelope channel count ({envelope.power.shape[0]}) does not match model channel count ({model.n_channels})."
        raise ValueError(msg)
    envelope_frame_rate = model.geometry.frame_rate(envelope.fs)
    model_frame_rate = 1.0 / model.dt
    if not np.isclose(envelope_frame_rate, model_frame_rate, rtol=1e-9):
        msg = (
            f"envelope sampling rate ({envelope.fs:g} Hz) is a Frame rate of {envelope_frame_rate:g} Hz "
            f"at hop {model.geometry.n_hop}, but the model steps at {model_frame_rate:g} Hz."
        )
        raise ValueError(msg)
    if envelope.geometry != model.geometry:
        differing = ", ".join(
            f"{field} ({getattr(envelope.geometry, field)!r} vs {getattr(model.geometry, field)!r})"
            for field in type(model.geometry).model_fields
            if getattr(envelope.geometry, field) != getattr(model.geometry, field)
        )
        msg = f"envelope geometry does not match model geometry: {differing}."
        raise ValueError(msg)


_BIPOLAR_CONTROLS = 2


class BipolarReducedModel(DiscreteDynamics, InferencePredictor):
    """DiscreteDynamics adapter mapping scalar control v in [-u_max, u_max] to bipolar u = [v, -v]^T.

    Enables exact Box-iLQR on 2-electrode montages by reducing the control dimension m from 2 to 1,
    rendering the Kirchhoff constraint identically satisfied without altering the box bounds.
    """

    base_model: InferencePredictor
    n_y: int = eqx.field(static=True)
    n_u: int = eqx.field(static=True)
    n_channels: int = eqx.field(static=True)
    n_controls: int = eqx.field(static=True)
    n_outputs: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)

    def __init__(self, base_model: InferencePredictor) -> None:
        """Wrap a base InferencePredictor model with m=2."""
        if base_model.m != _BIPOLAR_CONTROLS:
            msg = f"BipolarReducedModel requires base_model with m=2, got m={base_model.m}"
            raise ValueError(msg)
        super().__init__(n=base_model.n, m=1, ne=base_model.ne)
        self.base_model = base_model
        self.n_y = int(base_model.n_y)
        self.n_u = int(base_model.n_u)
        self.n_channels = int(base_model.n_channels)
        self.n_controls = 1
        self.n_outputs = int(base_model.n_outputs)
        self.dt = float(base_model.dt)

    def discrete_dynamics(
        self,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array,
        dt: float | jax.Array,
    ) -> jax.Array:
        """Advance one step with u_full = [u[0], -u[0]]."""
        del t, dt
        u_val = u[0] if u.ndim > 0 else u
        u_full = jnp.array([u_val, -u_val])
        return self.base_model.discrete_dynamics(x, u_full, 0.0, self.dt)

    def absorb(self, state: FloatArray, y: FloatArray, u: FloatArray) -> FloatArray:
        """Absorb with u_full = [u[0], -u[0]]."""
        u_arr = np.asarray(u, dtype=np.float64)
        u_val = u_arr[0] if u_arr.ndim > 0 else float(u_arr)
        u_full = np.array([u_val, -u_val], dtype=np.float64)
        return self.base_model.absorb(state, y, u_full)

    def is_ready(self, state: FloatArray) -> bool:
        """Report readiness from base model."""
        return self.base_model.is_ready(state)

    def initial_state(self) -> FloatArray:
        """Return base model's initial state."""
        return self.base_model.initial_state()

    def free_run(
        self,
        y_hists: FloatArray,
        u_hists: FloatArray,
        u_futures: FloatArray,
    ) -> jax.Array:
        """Free-run with bipolar control expansion."""
        u_f_arr = np.asarray(u_futures, dtype=np.float64)
        u_f_full = np.stack([u_f_arr[..., 0], -u_f_arr[..., 0]], axis=-1)
        u_h_arr = np.asarray(u_hists, dtype=np.float64)
        u_h_full = np.stack([u_h_arr[..., 0], -u_h_arr[..., 0]], axis=-1) if u_h_arr.shape[-1] == 1 else u_h_arr
        return self.base_model.free_run(y_hists, u_h_full, u_f_full)

    def to_checkpoint(self) -> tuple[dict[str, Any], dict[str, FloatArray]]:
        """Return base model checkpoint."""
        return self.base_model.to_checkpoint()

    @classmethod
    def from_checkpoint(cls, meta: dict[str, Any], arrays: dict[str, FloatArray]) -> Self:
        """Rebuild base model then wrap."""
        if "geometry" in meta:
            base: InferencePredictor = ObservableMLPModel.from_checkpoint(meta, arrays)
        else:
            base = WaveformMLPModel.from_checkpoint(meta, arrays)
        return cls(base)


def build_waveform_problem(  # noqa: PLR0913 -- checkpoint plus the ten MPC cost/bound knobs
    artifact: str | Path,
    *,
    horizon: int,
    u_max: ArrayLike,
    w_y: float = 1.0,
    w_u: float = 0.0,
    w_y_terminal: float | None = None,
    w_u_l1: float = 0.0,
    w_psd: float = 0.0,
    psd_ref: str | Path | None = None,
    kirchhoff: bool = False,
    w_kirchhoff: float = 0.0,
) -> Problem:
    """Assemble the waveform MPC problem: model adapter, objective, box and Kirchhoff bounds.

    The objective minimizes tracking deviation from zero, quadratic and L1 control effort, and
    one-sided log-power PSD hinges against ``psd_ref``. The constraints are the control box
    bounds ``-u_max <= u <= u_max``, plus the Kirchhoff sum-to-zero equality when ``kirchhoff``
    is set.

    Parameters
    ----------
    artifact
        Suffix-less stem of the numpy-readable MLP checkpoint.
    horizon
        Control Horizon counted in model steps; the trajopt horizon is ``horizon + 1`` knot
        points.
    u_max
        Per-electrode amplitude bound: a scalar shared by every electrode or a
        length-``n_controls`` vector.
    w_y
        Weight on state tracking error in the stage cost.
    w_u
        Weight on control effort (quadratic) in the stage cost.
    w_y_terminal
        Weight on the terminal knot state tracking error. When ``None`` (default), inherits
        ``w_y``.
    w_u_l1
        Weight on the L1 norm of the control effort (a sparse-stimulation penalty); ``0``
        disables it (default).
    w_psd
        Weight on the spectral hinge cost: the mean squared amount by which predicted log-power
        exceeds ``psd_ref``'s healthy envelope. ``0`` (default) disables it.
    psd_ref
        Path to the healthy reference envelope npz written by ``scripts/build_healthy_psd.py``.
        Required when ``w_psd > 0``.
    kirchhoff
        Add the Kirchhoff sum-to-zero equality on the controls. Off by default; the incumbent
        applies it unconditionally, so full parity sets it.
    w_kirchhoff
        Weight on the quadratic penalty formulation of Kirchhoff's Current Law:
        ``(w_kirchhoff / horizon) * (sum(u))^2``. ``0`` (default) disables it.
    """
    model = WaveformMLPModel.load(artifact)
    n, m = model.n, model.m
    N = horizon + 1

    z_last = slice((model.n_y - 1) * model.n_channels, model.n_y * model.n_channels)
    Q = jnp.zeros(n).at[z_last].set(2.0 * w_y * model.y_scale**2 / horizon)
    xf = jnp.zeros(n).at[z_last].set(-model.y_center / model.y_scale)
    stage = DiagonalCost.tracking(Q, jnp.full(m, 2.0 * w_u / horizon), xf, jnp.zeros(m))
    costs: list[CostFunction] = [ExcludeInitialKnotState(stage)]
    if w_u_l1 > 0:
        costs.append(L1ControlCost(n=n, m=m, w_l1=w_u_l1, horizon=horizon))
    if w_kirchhoff > 0:
        costs.append(KirchhoffPenaltyCost(n=n, m=m, w_k=w_kirchhoff, horizon=horizon))
    envelope = _spectral_envelope(psd_ref, w_psd)
    if envelope is not None:
        _validate_waveform_envelope(envelope, model)
        outputs = StateOutputs(
            n=n,
            m=m,
            n_y=model.n_y,
            n_outputs=model.n_channels,
            center=model.y_center,
            scale=model.y_scale,
        )
        costs.append(SpectralHingeCost(outputs, envelope, w_psd=w_psd, horizon=horizon))
    stage_cost: CostFunction = _combine_costs(costs)
    # The terminal knot carries the horizon's final output, weighted by ``w_y_terminal`` when
    # given else ``w_y`` -- the incumbent's last-step stage cost. Always explicit, because the
    # composite's derived terminal (``SumCost.as_terminal``) would otherwise carry the
    # control-only L1 and whole-horizon hinge into a knot that has no control.
    w_y_final = w_y_terminal if w_y_terminal is not None else w_y
    Q_f = jnp.zeros(n).at[z_last].set(2.0 * w_y_final * model.y_scale**2 / horizon)
    terminal = DiagonalCost.terminal_tracking(Q_f, xf, m)
    objective = Objective(stage_cost=stage_cost, terminal_cost=terminal, N=N)

    return _assemble_problem(model, objective, N=N, u_max=u_max, kirchhoff=kirchhoff)


def build_bipolar_waveform_problem(  # noqa: PLR0913 -- checkpoint plus cost knobs
    artifact: str | Path,
    *,
    horizon: int,
    u_max: float | ArrayLike,
    w_y: float = 1.0,
    w_u: float = 0.0,
    w_y_terminal: float | None = None,
    w_u_l1: float = 0.0,
) -> Problem:
    """Assemble the 1D reduced bipolar waveform MPC problem for Box-iLQR.

    The scalar control ``v`` in ``[-u_max, u_max]`` is mapped to bipolar currents ``u = [v, -v]^T``,
    satisfying Kirchhoff's Current Law exactly without requiring coupled constraints.
    """
    base_model = WaveformMLPModel.load(artifact)
    model = BipolarReducedModel(base_model)
    n, m = model.n, model.m  # m = 1
    N = horizon + 1

    z_last = slice((base_model.n_y - 1) * base_model.n_channels, base_model.n_y * base_model.n_channels)
    Q = jnp.zeros(n).at[z_last].set(2.0 * w_y * base_model.y_scale**2 / horizon)
    xf = jnp.zeros(n).at[z_last].set(-base_model.y_center / base_model.y_scale)
    # Quadratic control on [v, -v] is v^2 + (-v)^2 = 2 * v^2, so diagonal weight is 2 * (2 * w_u / horizon)
    R_diag = jnp.full(m, 4.0 * w_u / horizon)
    stage = DiagonalCost.tracking(Q, R_diag, xf, jnp.zeros(m))
    costs: list[CostFunction] = [ExcludeInitialKnotState(stage)]
    if w_u_l1 > 0:
        # L1 norm on [v, -v] is |v| + |-v| = 2 * |v|, so weight is 2 * w_u_l1
        costs.append(L1ControlCost(n=n, m=m, w_l1=2.0 * w_u_l1, horizon=horizon))
    stage_cost: CostFunction = _combine_costs(costs)

    w_y_final = w_y_terminal if w_y_terminal is not None else w_y
    Q_f = jnp.zeros(n).at[z_last].set(2.0 * w_y_final * base_model.y_scale**2 / horizon)
    terminal = DiagonalCost.terminal_tracking(Q_f, xf, m)
    objective = Objective(stage_cost=stage_cost, terminal_cost=terminal, N=N)

    u_max_val = float(np.atleast_1d(np.asarray(u_max))[0])
    constraints = ConstraintList(n=n, m=m, N=N)
    constraints.add_constraint(ControlBound(n=n, m=m, u_min=[-u_max_val], u_max=[u_max_val]), range(N - 1))
    return Problem(model=model, obj=objective, constraints=constraints, N=N)


def build_observable_problem(  # noqa: PLR0913 -- checkpoint plus the MPC cost/bound knobs
    artifact: str | Path,
    *,
    horizon: int,
    u_max: ArrayLike,
    w_u: float = 0.0,
    w_u_l1: float = 0.0,
    w_hinge: float = 0.0,
    envelope_ref: str | Path | None = None,
    kirchhoff: bool = False,
    w_kirchhoff: float = 0.0,
) -> Problem:
    """Assemble the observable MPC problem: model adapter, objective, box and Kirchhoff bounds.

    The model steps one Frame per call on the hop grid. The objective minimizes the one-sided
    log-power hinge against the healthy Observable envelope, plus quadratic and L1 control effort
    penalties. The constraints are the control box bounds ``-u_max <= u <= u_max``, plus the
    Kirchhoff sum-to-zero equality when ``kirchhoff`` is set.

    Parameters
    ----------
    artifact
        Suffix-less stem of the numpy-readable Observable MLP checkpoint.
    horizon
        Control Horizon counted in Frames; the trajopt horizon is ``horizon + 1`` knot points.
    u_max
        Per-electrode amplitude bound: a scalar shared by every electrode or a
        length-``n_controls`` vector.
    w_u
        Weight on control effort (quadratic) in the cost.
    w_u_l1
        Weight on the L1 norm of the control effort (a sparse-stimulation penalty); ``0``
        disables it (default).
    w_hinge
        Weight on the hinge cost: the mean squared amount by which the predicted log-power
        Frames exceed ``envelope_ref``'s healthy envelope. ``0`` (default) disables it.
    envelope_ref
        Path to the healthy reference Observable envelope npz written by
        ``scripts/build_healthy_psd.py``. Required when ``w_hinge > 0``; its stored geometry
        drives the cost.
    kirchhoff
        Add the Kirchhoff sum-to-zero equality on the controls.
    w_kirchhoff
        Weight on the quadratic penalty formulation of Kirchhoff's Current Law:
        ``(w_kirchhoff / horizon) * (sum(u))^2``. ``0`` (default) disables it.
    """
    model = ObservableMLPModel.load(artifact)
    n, m = model.n, model.m
    N = horizon + 1

    stage = DiagonalCost.tracking(jnp.zeros(n), jnp.full(m, 2.0 * w_u / horizon), jnp.zeros(n), jnp.zeros(m))
    costs: list[CostFunction] = [ExcludeInitialKnotState(stage)]
    if w_u_l1 > 0:
        costs.append(L1ControlCost(n=n, m=m, w_l1=w_u_l1, horizon=horizon))
    if w_kirchhoff > 0:
        costs.append(KirchhoffPenaltyCost(n=n, m=m, w_k=w_kirchhoff, horizon=horizon))
    envelope = _observable_envelope(envelope_ref, w_hinge)
    # The stage trajectory carries every Frame of the Control Horizon but the last, which lives
    # only in the terminal knot; the terminal Cost scores it so no predicted Frame goes unpriced.
    terminal: CostFunction = DiagonalCost.terminal_tracking(jnp.zeros(n), jnp.zeros(n), m)
    if envelope is not None:
        _validate_observable_envelope(envelope, model)
        outputs = StateOutputs(
            n=n,
            m=m,
            n_y=model.n_y,
            n_outputs=model.n_outputs,
            center=model.y_center,
            scale=model.y_scale,
        )
        costs.append(ObservableHingeCost(outputs, envelope, w_hinge=w_hinge, horizon=horizon))
        terminal = ObservableHingeCost(outputs, envelope, w_hinge=w_hinge, horizon=horizon, terminal=True)
    stage_cost: CostFunction = _combine_costs(costs)
    objective = Objective(stage_cost=stage_cost, terminal_cost=terminal, N=N)

    return _assemble_problem(model, objective, N=N, u_max=u_max, kirchhoff=kirchhoff)


class TrajOptMPCController(Controller[TrajOptMPCLog]):
    """Receding-horizon MPC for the waveform predictor, built directly on trajopt primitives.

    Owns the true absorbed predictor state and ``u_last`` as private instance attributes, and
    reproduces the incumbent MPC ``update`` loop shape -- ``absorb``, then
    ``with_measurement``, ``problem.solve``, extract the first control, ``shift`` -- without
    instantiating ``TrajOptMPC``. ``TrajOptMPC.update`` is the reference for how ``Problem``,
    ``MPCState`` and the solver compose, not a dependency.
    """

    def __init__(
        self,
        dt: float,
        problem: Problem,
        solver: Solver | dict[str, Any] | None = None,
        initial_state: MPCState | None = None,
    ) -> None:
        """Initialize the controller and its persistent MPC state.

        Parameters
        ----------
        dt
            Controller update step in seconds; should equal the predictor's native dt.
        problem
            The trajopt optimal-control problem: model adapter + objective + constraint list.
        solver
            Solver backend instance or ``{class_path, ...}`` config dict. When omitted, the
            benchmark-winning default is selected: ``BoxQP`` for bipolar models,
            ``SingleShooting(Ipopt)`` for non-separable whole-horizon costs, and ``ALTRO``
            for general problems.
        initial_state
            Initial MPC warm-start state; defaults to one built from the unprimed model state,
            with the NaN padding of the unprimed EEG window replaced by zeros so the seed is
            finite for every solver (the multiple-shooting transcription starts from the full
            state trajectory, not just the controls).
        """
        super().__init__(dt)
        self.problem = problem
        model = problem.model
        if not isinstance(model, InferencePredictor):
            msg = f"problem.model ({type(model).__name__}) does not implement the InferencePredictor priming seam"
            raise TypeError(msg)
        self.model = model
        self.solver = _build_solver(solver, problem)

        unprimed = jnp.nan_to_num(jnp.asarray(self.model.initial_state()), nan=0.0)
        self.state = initial_state if initial_state is not None else MPCState.initial(problem, x0=unprimed, dt=dt)
        self._state = np.asarray(self.model.initial_state(), dtype=np.float64)
        self._u_last = np.zeros(problem.model.m, dtype=np.float64)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate from a config dict, dispatching the Problem and Solver factories.

        Follows the standard ``{class_path, ...}`` pattern: ``problem`` is either a Problem instance
        or a ``{class_path, ...}`` dict naming a Problem-building factory (e.g.
        ``neuro.control.mpc.build_waveform_problem``); ``solver`` is an optional
        ``{class_path, ...}`` dict naming a solver backend (e.g. ``trajopt.solvers.altro.ALTRO``);
        ``initial_state`` is optional. When ``solver`` is omitted, the benchmark-winning default
        is selected automatically.
        """
        problem, initial_state = _build_problem(config["problem"])
        solver = _build_solver(config.get("solver"), problem)
        return cls(
            dt=float(config["dt"]),
            problem=problem,
            solver=solver,
            initial_state=initial_state or config.get("initial_state"),
        )

    def update(
        self,
        t: float,
        ref: FloatArray,  # noqa: ARG002 -- the goal is baked into the objective
        x_hat: FloatArray,
    ) -> tuple[FloatArray, TrajOptMPCLog]:
        """Ingest the current EEG measurement, solve the receding-horizon problem, emit the first control."""
        self._state = np.asarray(self.model.absorb(self._state, np.asarray(x_hat).reshape(-1), self._u_last))

        if not self.model.is_ready(self._state):
            u_zero = np.zeros(self.model.m, dtype=np.float64)
            self._u_last = u_zero
            return u_zero, TrajOptMPCLog(u=u_zero, cost=0.0, success=True, warmup=True)

        state = self.state.with_measurement(jnp.asarray(self._state), t=t)
        solved = self.problem.solve(state, solver=self.solver)
        u_cmd = np.asarray(solved.controls[0], dtype=np.float64)
        self.state = solved.shift(self.dt)
        self._u_last = u_cmd
        return u_cmd, TrajOptMPCLog(
            u=u_cmd.copy(),
            cost=float(self.problem.cost(solved)),
            success=solved.status == "converged",
            warmup=False,
        )
