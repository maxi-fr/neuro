from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from trajopt.cones import NegativeOrthant, ZeroCone
from trajopt.constraints.bounds import ControlBound
from trajopt.constraints.constraint_list import ConstraintList
from trajopt.constraints.horizon import LinearHorizonConstraint
from trajopt.constraints.linear import LinearConstraint
from trajopt.costs.objective import Objective
from trajopt.costs.output import OutputCost
from trajopt.costs.pseudo_huber import PseudoHuberControlCost
from trajopt.costs.quadratic import DiagonalCost
from trajopt.problem import (
    Problem,
    _EpigraphModel,
    add_l1_epigraph,
)

from neuro.control.costs import (
    ExcludeInitialKnotState,
    ObservableFrameHingeCost,
    ObservableHingeCost,
    SumCost,
)
from neuro.control.model_wrappers import (
    AugmentedActivationModel,
    EpigraphInferenceModel,
)
from neuro.predictor.inference import (
    InferencePredictor,
    ObservableCNNModel,
    ObservableMLPModel,
    WaveformCNNModel,
    WaveformMLPModel,
)
from neuro.spectral import HealthyReference, ObservableEnvelope

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike
    from trajopt.constraints.constraint_list import BuiltConstraintList
    from trajopt.costs.base import CostFunction
    from trajopt.dynamics.base import DiscreteDynamics, IntegratorCallable
    from trajopt.dynamics.integrators import Integrator

    from neuro.types import FloatArray


def _build_problem(spec: dict[str, Any] | Problem) -> Problem:
    """Instantiate a Problem from an instance or a ``{class_path, ...}`` dict naming a factory."""
    if isinstance(spec, Problem):
        return spec
    cfg = spec.copy()
    if "reference" in cfg and isinstance(cfg["reference"], (str, Path)):
        cfg["reference"] = HealthyReference.load(cfg["reference"])
    class_path: str = cfg.pop("class_path")
    module_name, func_name = class_path.rsplit(".", 1)
    target = getattr(importlib.import_module(module_name), func_name)
    return target(**cfg) if cfg else target()


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


def _observable_envelope(
    reference: HealthyReference | None,
    w_hinge: float,
) -> ObservableEnvelope | None:
    """Extract the healthy Observable envelope when ``w_hinge`` enables the hinge, else ``None``."""
    if w_hinge <= 0:
        return None
    if reference is None:
        msg = "reference must be provided when w_hinge > 0"
        raise ValueError(msg)
    if reference.observable is None:
        msg = "reference contains no Observable envelope"
        raise ValueError(msg)
    return reference.observable


def _amplitude_coupling_constraint(n: int, m_elec: int, u_max_arr: FloatArray) -> LinearConstraint:
    """Express |u_i| <= u_max_i * delta as [I; -I] u - u_max delta <= 0."""
    m = m_elec + 1
    A = np.zeros((2 * m_elec, m), dtype=np.float64)
    u_max_vec = np.asarray(u_max_arr, dtype=np.float64)
    for i in range(m_elec):
        A[2 * i, i] = 1.0
        A[2 * i, -1] = -u_max_vec[i]
        A[2 * i + 1, i] = -1.0
        A[2 * i + 1, -1] = -u_max_vec[i]
    b = np.zeros(2 * m_elec, dtype=np.float64)
    return LinearConstraint(
        n=n,
        m=m,
        A=jnp.asarray(A),
        b=jnp.asarray(b),
        sense=NegativeOrthant(),
        inds=range(n, n + m),
    )


def _augmented_kirchhoff_constraint(n: int, m_elec: int) -> LinearConstraint:
    """Express sum(u_i) = 0 for augmented controls [u; delta]."""
    m = m_elec + 1
    A = np.zeros((1, m), dtype=np.float64)
    A[0, :m_elec] = 1.0
    b = np.zeros(1, dtype=np.float64)
    return LinearConstraint(
        n=n,
        m=m,
        A=jnp.asarray(A),
        b=jnp.asarray(b),
        sense=ZeroCone(),
        inds=range(n, n + m),
    )


class NeuroProblem(Problem):
    """Optimal control Problem carrying neurostimulation weights, bounds, and Observable reference."""

    horizon: int = eqx.field(static=True)
    w_active: float | None = eqx.field(static=True, default=None)
    w_hinge: float = eqx.field(static=True, default=0.0)
    w_u: float = eqx.field(static=True, default=0.0)
    w_u_l1: float = eqx.field(static=True, default=0.0)
    u_max: jax.Array = eqx.field(default=None)  # type: ignore[assignment]
    kirchhoff: bool = eqx.field(static=True, default=False)
    envelope: ObservableEnvelope | None = eqx.field(static=True, default=None)
    max_active_intervals: int | None = eqx.field(static=True, default=None)
    continuous_problem: Problem | None = eqx.field(default=None)

    def __init__(  # noqa: PLR0913, PLR0917 -- carries problem definition plus domain evaluation metadata
        self,
        model: DiscreteDynamics,
        obj: Objective,
        constraints: BuiltConstraintList | ConstraintList | None = None,
        N: int | None = None,
        dt: float | jax.Array = 0.05,
        integrator: Integrator | IntegratorCallable | None = None,
        binary_control_indices: Sequence[int] = (),
        *,
        horizon: int,
        u_max: ArrayLike,
        w_active: float | None = None,
        w_hinge: float = 0.0,
        w_u: float = 0.0,
        w_u_l1: float = 0.0,
        kirchhoff: bool = False,
        envelope: ObservableEnvelope | None = None,
        max_active_intervals: int | None = None,
        continuous_problem: Problem | None = None,
    ) -> None:
        """Construct a NeuroProblem carrying domain weights, envelope, and bounds."""
        super().__init__(
            model=model,
            obj=obj,
            constraints=constraints,
            N=N,
            dt=dt,
            integrator=integrator,
            binary_control_indices=binary_control_indices,
        )
        self.horizon = int(horizon)
        self.u_max = jnp.asarray(u_max, dtype=jnp.float64)
        self.w_active = w_active
        self.w_hinge = float(w_hinge)
        self.w_u = float(w_u)
        self.w_u_l1 = float(w_u_l1)
        self.kirchhoff = bool(kirchhoff)
        self.envelope = envelope
        self.max_active_intervals = max_active_intervals
        self.continuous_problem = continuous_problem


def _assemble_problem(  # noqa: PLR0913 -- the horizon's grid joins the five it already took
    model: DiscreteDynamics,
    objective: Objective,
    *,
    N: int,
    dt: float,
    u_max: ArrayLike,
    kirchhoff: bool,
    binary_control_indices: Sequence[int] = (),
    max_active_intervals: int | None = None,
    w_active: float | None = None,
    w_hinge: float = 0.0,
    w_u: float = 0.0,
    w_u_l1: float = 0.0,
    envelope: ObservableEnvelope | None = None,
    continuous_problem: Problem | None = None,
) -> NeuroProblem:
    """Build boxed Control Currents with optional activation and Kirchhoff constraints."""
    n, m = model.n, model.m
    horizon = N - 1
    if binary_control_indices:
        m_elec = m - len(binary_control_indices)
        u_max_arr = np.broadcast_to(np.atleast_1d(np.asarray(u_max, dtype=np.float64)), (m_elec,)).copy()
        constraints = ConstraintList(n=n, m=m, N=N)
        u_min = np.concatenate([-u_max_arr, [0.0] * len(binary_control_indices)])
        u_max_bound = np.concatenate([u_max_arr, [1.0] * len(binary_control_indices)])
        constraints.add_constraint(ControlBound(n=n, m=m, u_min=u_min, u_max=u_max_bound), range(N - 1))
        constraints.add_constraint(_amplitude_coupling_constraint(n, m_elec, u_max_arr), range(N - 1))
        if kirchhoff:
            constraints.add_constraint(_augmented_kirchhoff_constraint(n, m_elec), range(N - 1))
        if max_active_intervals is not None:
            inds = [k * (n + m) + n + m - 1 for k in range(N - 1)]
            A_cap = jnp.ones((1, len(inds)))
            b_cap = jnp.array([float(max_active_intervals)])
            constraints.add_horizon_constraint(
                LinearHorizonConstraint(A=A_cap, b=b_cap, sense=NegativeOrthant(), inds=inds)
            )
        return NeuroProblem(
            model=model,
            obj=objective,
            constraints=constraints,
            N=N,
            dt=dt,
            binary_control_indices=binary_control_indices,
            horizon=horizon,
            u_max=u_max_arr,
            w_active=w_active,
            w_hinge=w_hinge,
            w_u=w_u,
            w_u_l1=w_u_l1,
            kirchhoff=kirchhoff,
            envelope=envelope,
            max_active_intervals=max_active_intervals,
            continuous_problem=continuous_problem,
        )

    u_max_arr = np.broadcast_to(np.atleast_1d(np.asarray(u_max, dtype=np.float64)), (m,))
    constraints = ConstraintList(n=n, m=m, N=N)
    constraints.add_constraint(ControlBound(n=n, m=m, u_min=-u_max_arr, u_max=u_max_arr), range(N - 1))
    if kirchhoff:
        constraints.add_constraint(kirchhoff_constraint(n, m), range(N - 1))
    return NeuroProblem(
        model=model,
        obj=objective,
        constraints=constraints,
        N=N,
        dt=dt,
        horizon=horizon,
        u_max=u_max_arr,
        w_active=w_active,
        w_hinge=w_hinge,
        w_u=w_u,
        w_u_l1=w_u_l1,
        kirchhoff=kirchhoff,
        envelope=envelope,
        max_active_intervals=max_active_intervals,
    )


def _validate_waveform_envelope(envelope: ObservableEnvelope, model: WaveformMLPModel | WaveformCNNModel) -> None:
    """Ensure the healthy Observable envelope matches the waveform predictor's channels and sample rate."""
    if envelope.power.shape[0] != model.n_channels:
        msg = f"envelope channel count ({envelope.power.shape[0]}) does not match model channel count ({model.n_channels})."
        raise ValueError(msg)
    model_fs = 1.0 / model.dt
    if not np.isclose(envelope.fs, model_fs, rtol=1e-9):
        msg = f"envelope sampling rate ({envelope.fs:g} Hz) does not match model sampling rate ({model_fs:g} Hz)."
        raise ValueError(msg)


def _validate_observable_envelope(envelope: ObservableEnvelope, model: ObservableMLPModel | ObservableCNNModel) -> None:
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


def _resolve_waveform_target(
    ref: HealthyReference | None,
    base: WaveformMLPModel | WaveformCNNModel,
    *,
    tracking_active: bool,
    w_hinge: float,
) -> FloatArray | None:
    """Resolve and validate the physical tracking target from a HealthyReference."""
    if tracking_active and ref is None:
        msg = "reference must be provided when w_y > 0"
        raise ValueError(msg)
    if w_hinge > 0 and ref is None:
        msg = "reference must be provided when w_hinge > 0"
        raise ValueError(msg)
    if not tracking_active or ref is None:
        return None
    if ref.eeg_mean is not None and len(ref.eeg_mean) == base.n_channels:
        y_ref = ref.eeg_mean
    elif ref.lfp_mean is not None and len(ref.lfp_mean) == base.n_channels:
        y_ref = ref.lfp_mean
    else:
        msg = f"reference mean vector does not match model channel count ({base.n_channels})"
        raise ValueError(msg)
    return np.asarray(y_ref, dtype=np.float64)


def _load_waveform_runtime(artifact: str | Path) -> WaveformMLPModel | WaveformCNNModel:
    """Load and validate a waveform MLP or CNN runtime checkpoint."""
    base = InferencePredictor.load(artifact)
    if not isinstance(base, (WaveformMLPModel, WaveformCNNModel)):
        msg = f"waveform checkpoint required, got {type(base).__name__}"
        raise TypeError(msg)
    return base


def _waveform_terminal_cost(
    base_terminal: CostFunction,
    frame_hinge: ObservableFrameHingeCost | None,
    w_hinge_terminal: float | None = None,
) -> CostFunction:
    """Combine output tracking and optional terminal Frame hinge into the terminal Cost."""
    if frame_hinge is None:
        return base_terminal
    return _combine_costs([base_terminal, frame_hinge.as_terminal(w_hinge=w_hinge_terminal)])


def _wrap_epigraph_if_requested(
    problem: NeuroProblem,
    w_u_l1: float,
    l1_mode: str,
) -> NeuroProblem:
    """Wrap problem with exact L1 epigraph reformulation scaled by Control Horizon when requested."""
    if l1_mode not in ("smooth", "epigraph"):
        msg = f"l1_mode must be 'smooth' or 'epigraph', got {l1_mode!r}"
        raise ValueError(msg)
    if w_u_l1 > 0 and l1_mode == "epigraph":
        epi_problem = add_l1_epigraph(problem, weight=w_u_l1 / problem.horizon)
        epi_model = EpigraphInferenceModel(cast("_EpigraphModel", epi_problem.model))
        return NeuroProblem(
            model=epi_model,
            obj=epi_problem.obj,
            constraints=epi_problem.constraints,
            N=epi_problem.N,
            dt=epi_problem.dt,
            binary_control_indices=epi_problem.binary_control_indices,
            horizon=problem.horizon,
            u_max=problem.u_max,
            w_active=problem.w_active,
            w_hinge=problem.w_hinge,
            w_u=problem.w_u,
            w_u_l1=w_u_l1,
            kirchhoff=problem.kirchhoff,
            envelope=problem.envelope,
            max_active_intervals=problem.max_active_intervals,
            continuous_problem=problem.continuous_problem,
        )
    return problem


def build_waveform_problem(  # noqa: PLR0913 -- checkpoint plus the eleven MPC cost/bound knobs
    artifact: str | Path,
    *,
    horizon: int,
    u_max: ArrayLike,
    w_y: float = 1.0,
    w_u: float = 0.0,
    w_y_terminal: float | None = None,
    w_u_l1: float = 0.0,
    w_hinge: float = 0.0,
    w_hinge_terminal: float | None = None,
    reference: HealthyReference | None = None,
    kirchhoff: bool = False,
    l1_mode: str = "smooth",
) -> NeuroProblem:
    """Assemble physical-current Waveform MPC with tracking, spectral and effort Costs.

    The objective minimizes tracking deviation from the healthy reference operating point,
    quadratic and L1 control effort, and one-sided log-power Frame hinges against ``reference``.
    The constraints are the control box bounds ``-u_max <= u <= u_max``, plus the Kirchhoff
    sum-to-zero equality when ``kirchhoff`` is set.

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
        Weight on Raw EEG output tracking error in the stage Cost.
    w_u
        Weight on Control Current effort (quadratic) in the stage Cost.
    w_y_terminal
        Weight on the terminal knot Raw EEG output tracking error. When ``None`` (default), inherits
        ``w_y``.
    w_u_l1
        Weight on the L1 norm of the Control Current effort (a sparse-stimulation penalty); ``0``
        disables it (default).
    w_hinge
        Weight on the spectral hinge Cost: the mean squared amount by which predicted log-power Frames
        exceeds ``reference``'s healthy envelope. ``0`` (default) disables it.
    w_hinge_terminal
        Weight on the terminal knot spectral hinge Cost. When ``None`` (default), inherits ``w_hinge``.
    reference
        :class:`~neuro.spectral.HealthyReference` container carrying empirical channel
        means and/or Observable envelope. Required when ``w_y > 0`` or ``w_hinge > 0``.
    kirchhoff
        Add the Kirchhoff sum-to-zero equality on the controls. Off by default; the incumbent
        applies it unconditionally, so full parity sets it.
    l1_mode
        Sparsity reformulation mode: ``"smooth"`` (default pseudo-Huber surrogate) or
        ``"epigraph"`` (exact L1 epigraph reformulation).
    """
    base = _load_waveform_runtime(artifact)
    w_hinge_eff = max(w_hinge, w_hinge if w_hinge_terminal is None else w_hinge_terminal)
    envelope = _observable_envelope(reference, w_hinge_eff)
    if envelope is not None:
        _validate_waveform_envelope(envelope, base)
        support = envelope.geometry.sample_support_steps(envelope.fs)
        if base.n_history < support:
            base = base.with_history(support)
    model: DiscreteDynamics = base
    n, m = model.n, model.m
    N = horizon + 1

    w_y_final = w_y_terminal if w_y_terminal is not None else w_y
    target_val = _resolve_waveform_target(
        reference,
        base,
        tracking_active=(w_y > 0 or w_y_final > 0),
        w_hinge=w_hinge_eff,
    )

    target = jnp.zeros(model.p) if target_val is None else jnp.asarray(target_val)
    Q = jnp.full(model.p, 2.0 * w_y / horizon)
    R = jnp.full(m, 2.0 * w_u / horizon)
    stage = DiagonalCost.tracking(Q, R, target, jnp.zeros(m))
    output_stage = OutputCost(model, stage)
    costs: list[CostFunction] = [ExcludeInitialKnotState(output_stage)]
    if w_u_l1 > 0 and l1_mode == "smooth":
        costs.append(PseudoHuberControlCost(n=n, m=m, weight=w_u_l1 / horizon, delta=1e-3))
    frame_hinge: ObservableFrameHingeCost | None = None
    if envelope is not None:
        frame_hinge = ObservableFrameHingeCost(model, envelope, w_hinge=w_hinge, horizon=horizon)
        costs.append(frame_hinge)
    stage_cost: CostFunction = _combine_costs(costs)
    if w_y_terminal is not None and w_y_terminal != w_y:
        Q_f = jnp.full(model.p, 2.0 * w_y_final / horizon)
        base_terminal: CostFunction = OutputCost(model, DiagonalCost.terminal_tracking(Q_f, target, m=m))
    else:
        base_terminal = output_stage.as_terminal()
    terminal_cost = _waveform_terminal_cost(base_terminal, frame_hinge, w_hinge_terminal=w_hinge_terminal)
    objective = Objective(stage_cost=stage_cost, terminal_cost=terminal_cost, N=N)

    problem = _assemble_problem(
        model,
        objective,
        N=N,
        dt=base.dt,
        u_max=u_max,
        kirchhoff=kirchhoff,
        w_u=w_u,
        w_u_l1=w_u_l1,
        w_hinge=w_hinge,
        envelope=envelope,
    )
    return _wrap_epigraph_if_requested(problem, w_u_l1=w_u_l1, l1_mode=l1_mode)


def _validate_integer_observable_problem_args(
    horizon: int,
    *,
    w_active: float | None,
    max_active_intervals: int | None,
) -> bool:
    """Validate activation weights and caps and report whether binary controls are needed."""
    if max_active_intervals is not None and (
        isinstance(max_active_intervals, bool)
        or not isinstance(max_active_intervals, int)
        or not 0 <= max_active_intervals <= horizon
    ):
        msg = "max_active_intervals must be an integer between zero and horizon"
        raise ValueError(msg)
    if w_active is not None and (not np.isfinite(w_active) or w_active < 0):
        msg = "w_active must be finite and nonnegative"
        raise ValueError(msg)
    if max_active_intervals is not None and w_active is not None:
        msg = "max_active_intervals and w_active are mutually exclusive"
        raise ValueError(msg)
    return w_active is not None or max_active_intervals is not None


def build_observable_problem(  # noqa: PLR0913 -- checkpoint plus the MPC cost/bound knobs
    artifact: str | Path | ObservableMLPModel | ObservableCNNModel,
    *,
    horizon: int,
    u_max: ArrayLike,
    w_u: float = 0.0,
    w_u_l1: float = 0.0,
    w_hinge: float = 0.0,
    w_hinge_terminal: float | None = None,
    reference: HealthyReference | None = None,
    kirchhoff: bool = False,
    w_active: float | None = None,
    max_active_intervals: int | None = None,
    l1_mode: str = "smooth",
) -> NeuroProblem:
    """Assemble Observable MPC with optional activation controls and the Control Budget.

    The model steps one Frame per call on the hop grid. The objective minimizes the one-sided
    log-power hinge against the healthy Observable envelope, plus quadratic and L1 control effort
    penalties. The constraints are the control box bounds ``-u_max <= u <= u_max``, plus the
    Kirchhoff sum-to-zero equality when ``kirchhoff`` is set.

    Parameters
    ----------
    artifact
        Observable Predictor checkpoint path or loaded model instance.
    horizon
        Control Horizon counted in Frames; the trajopt horizon is ``horizon + 1`` knot points.
    u_max
        Per-electrode amplitude bound: a scalar shared by every electrode or a
        length-``n_controls`` vector.
    w_u
        Weight on Control Current effort (quadratic) in the Cost.
    w_u_l1
        Weight on the L1 norm of the Control Current effort (a sparse-stimulation penalty); ``0``
        disables it (default).
    w_hinge
        Weight on the hinge Cost: the mean squared amount by which the predicted log-power
        Frames exceed ``reference``'s healthy envelope. ``0`` (default) disables it.
    reference
        :class:`~neuro.spectral.HealthyReference` container carrying the healthy Observable envelope.
        Required when ``w_hinge > 0``.
    kirchhoff
        Add the Kirchhoff sum-to-zero equality on the controls.
    w_active
        Weight on enabled Control Horizon intervals (activation penalty). Mutually exclusive with
        ``max_active_intervals``.
    max_active_intervals
        Upper bound on the number of enabled intervals per Control Horizon (activation cap).
        Mutually exclusive with ``w_active``.
    l1_mode
        Sparsity reformulation mode: ``"smooth"`` (default pseudo-Huber surrogate) or
        ``"epigraph"`` (exact L1 epigraph reformulation).
    """
    base = (
        artifact
        if isinstance(artifact, (ObservableMLPModel, ObservableCNNModel))
        else InferencePredictor.load(artifact)
    )
    if not isinstance(base, (ObservableMLPModel, ObservableCNNModel)):
        msg = f"Observable checkpoint required, got {type(base).__name__}"
        raise TypeError(msg)
    integer_mode = _validate_integer_observable_problem_args(
        horizon,
        w_active=w_active,
        max_active_intervals=max_active_intervals,
    )

    model: DiscreteDynamics = AugmentedActivationModel(base) if integer_mode else base
    n, m = model.n, model.m
    N = horizon + 1

    if w_hinge > 0 and reference is None:
        msg = "reference must be provided when w_hinge > 0"
        raise ValueError(msg)

    R = jnp.zeros(m)
    r = jnp.zeros(m)
    m_phys = base.m
    R = R.at[:m_phys].set(2.0 * w_u / horizon)
    if w_active is not None:
        r = r.at[-1].set(w_active / horizon)
    stage = DiagonalCost(Q=jnp.zeros(n), R=R, q=jnp.zeros(n), r=r)
    costs: list[CostFunction] = [ExcludeInitialKnotState(stage)]
    if w_u_l1 > 0 and l1_mode == "smooth":
        costs.append(PseudoHuberControlCost(n=n, m=m, weight=w_u_l1 / horizon, delta=1e-3))
    w_hinge_term = w_hinge if w_hinge_terminal is None else float(w_hinge_terminal)
    w_hinge_eff = max(w_hinge, w_hinge_term)
    envelope = _observable_envelope(reference, w_hinge_eff)
    # The stage trajectory carries every Frame of the Control Horizon but the last, which lives
    # only in the terminal knot; the terminal Cost scores it so no predicted Frame goes unpriced.
    terminal: CostFunction = DiagonalCost.terminal_tracking(jnp.zeros(n), jnp.zeros(n), m)
    if envelope is not None:
        _validate_observable_envelope(envelope, base)
        hinge = ObservableHingeCost(envelope, w_hinge=w_hinge, horizon=horizon)
        output_hinge = OutputCost(model, hinge)
        costs.append(ExcludeInitialKnotState(output_hinge))
        term_hinge = ObservableHingeCost(envelope, w_hinge=w_hinge_term, horizon=horizon)
        terminal = OutputCost(model, term_hinge.as_terminal())
    stage_cost: CostFunction = _combine_costs(costs)
    objective = Objective(stage_cost=stage_cost, terminal_cost=terminal, N=N)

    u_max_arr = np.broadcast_to(np.atleast_1d(np.asarray(u_max, dtype=np.float64)), (base.m,)).copy()
    cont_prob = None
    if integer_mode:
        cont_prob = build_observable_problem(
            artifact=base,
            horizon=horizon,
            u_max=u_max,
            w_u=w_u,
            w_u_l1=w_u_l1,
            w_hinge=w_hinge,
            reference=reference,
            kirchhoff=kirchhoff,
            max_active_intervals=None,
            w_active=None,
            l1_mode=l1_mode,
        )

    problem = _assemble_problem(
        model,
        objective,
        N=N,
        dt=base.dt,
        u_max=u_max_arr,
        kirchhoff=kirchhoff,
        binary_control_indices=(m - 1,) if integer_mode else (),
        max_active_intervals=max_active_intervals,
        w_active=w_active,
        w_hinge=w_hinge,
        w_u=w_u,
        w_u_l1=w_u_l1,
        envelope=envelope,
        continuous_problem=cont_prob,
    )
    return _wrap_epigraph_if_requested(problem, w_u_l1=w_u_l1, l1_mode=l1_mode)
