"""Model-agnostic discrete-dynamics interface for the CasADi MPC and SysID solvers.

Defines the :class:`SymbolicModel` protocol -- the abstract ``x_next = step(x, u)``,
``y = output(x)`` contract the optimizers are written against -- and
:class:`JRSymbolicModel`, the Jansen-Rit adapter that fulfils it by reusing the
symbolic functions in :mod:`neuro.jansen_rit_casadi`.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Self

import casadi as ca
import numpy as np
import yaml

from neuro.jansen_rit import JansenRitParams
from neuro.jansen_rit_casadi import (
    JRSymbolicParams,
    eeg,
    heun_step,
    project_control,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def load_config(filepath: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file into a raw dictionary."""
    with Path(filepath).open() as f:
        config = yaml.safe_load(f)
        if not isinstance(config, dict):
            msg = f"Config file {filepath} must be a dictionary"
            raise TypeError(msg)
        return config


def shoot(  # noqa: PLR0913
    opti: ca.Opti,
    f_step: ca.Function,
    f_out: ca.Function,
    history: Sequence[ca.MX],
    controls: Sequence[ca.MX],
    state_shape: tuple[int, int],
    free: Sequence[ca.MX] = (),
    stride: int = 1,
    step0: int = 0,
) -> tuple[list[ca.MX], list[tuple[ca.MX, int]]]:
    """Roll a compiled model forward over ``len(controls)`` steps with PCMS sparsification.

    ``history`` is the newest-first initial state bundle (length ``history_depth + 1``);
    ``controls[k]`` is the control at relative step ``k``. ``free`` are free-parameter
    actuals grafted onto the trailing inputs of ``f_step``/``f_out``. Every ``stride``
    absolute steps (the absolute step is ``step0 + k``) a continuity state variable is
    introduced and tied to the predicted state, so ``stride == 1`` is full multiple
    shooting and ``stride >= len(controls)`` is single shooting. ``step0`` keeps the
    sparsification aligned to absolute step indices when the caller has already consumed
    some steps. Returns the per-step outputs ``Y_pred`` and the ``(var, abs_step)``
    continuity pairs.
    """
    hist = list(history)
    y_pred: list[ca.MX] = []
    ms_vars: list[tuple[ca.MX, int]] = []
    n = len(controls)
    for k in range(n):
        x_next = f_step(*hist, controls[k], *free)
        abs_step = step0 + k + 1
        if abs_step % stride == 0 and (k + 1) < n:
            var = opti.variable(*state_shape)
            opti.subject_to(var == x_next)
            ms_vars.append((var, abs_step))
            x_next = var
        hist.insert(0, x_next)
        hist.pop()
        y_pred.append(f_out(x_next, *free))
    return y_pred, ms_vars


class SymbolicModel(Protocol):
    """Abstract discrete-time model: ``x_next = step(history, u)``, ``y = output(x)``.

    The optimizers (MPC / SysID) drive a model purely through this interface, so the
    JR-specific dynamics stay isolated behind it.

    The "state" is the per-step live state of shape ``state_shape``. A delayed model
    carries its bounded past as a newest-first ``history`` list of length
    ``history_depth + 1``; a memoryless model has ``history_depth == 0`` and
    ``step([x], u)`` reduces to the pure ``F(x, u)`` map.

    ``f_step``/``f_out`` are the reusable compiled single-step and output
    ``ca.Function``s. ``free_syms`` maps each free-parameter name to its ``MX.sym``
    placeholder; those placeholders are the trailing inputs of ``f_step``/``f_out`` (the
    mapping is empty for a numeric model with no free parameters).
    """

    state_shape: tuple[int, int]
    history_depth: int
    n_elec: int
    n_channels: int
    free_syms: dict[str, ca.MX]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate a symbolic model from a config dict."""
        ...

    def step(self, history: Sequence[ca.SX | ca.MX], u: ca.SX | ca.MX) -> ca.SX | ca.MX:
        """Advance one step: newest-first ``history`` bundle and control ``u`` -> next live state."""
        ...

    def output(self, x: ca.SX | ca.MX) -> ca.SX | ca.MX:
        """Map a live state to the measured output ``y``."""
        ...

    @property
    def f_step(self) -> ca.Function:
        """Reusable compiled single-step function ``(xh0..xhd, u, *free) -> x_next``."""
        ...

    @property
    def f_out(self) -> ca.Function:
        """Reusable compiled output function ``(x, *free) -> y``."""
        ...


def _jr_symbolic_params(base: JansenRitParams) -> JRSymbolicParams:
    """Build numeric ``JRSymbolicParams`` from a ``JansenRitParams``.

    ``A``/``mean_input`` are broadcast to ``(1, n_nodes)`` and ``gamma`` is reshaped to
    2-D; every other field passes straight through. Shared by the numeric (MPC) path and
    the SysID path, which then swaps free parameters for symbolic placeholders.
    """
    n_nodes = base.w_weights.shape[0]
    gamma = np.asarray(base.gamma, dtype=np.float64)
    if gamma.ndim == 1:
        gamma = gamma.reshape(1, -1)
    return JRSymbolicParams(
        A=np.broadcast_to(np.asarray(base.A), (1, n_nodes)),
        B=base.B,
        a=base.a,
        b=base.b,
        C1=base.C1,
        C2=base.C2,
        C3=base.C3,
        C4=base.C4,
        e0=base.e0,
        v0=base.v0,
        r=base.r,
        mean_input=np.broadcast_to(np.asarray(base.mean_input), (1, n_nodes)),
        sigma=base.sigma,
        eeg_gain=base.eeg_gain,
        gamma=gamma,
        K=base.K,
        w_weights=base.w_weights,
        delay_steps=base.delay_steps,
    )


def _free_param_shape(p: str, n_nodes: int, n_elec: int, n_channels: int) -> tuple[int, int]:
    """Natural ``(rows, cols)`` of a free Jansen-Rit parameter's symbolic placeholder."""
    if p in ("A", "mean_input"):
        return (1, n_nodes)
    if p == "eeg_gain":
        return (n_channels, n_nodes)
    if p == "gamma":
        return (n_elec, n_nodes)
    if p == "w_weights":
        return (n_nodes, n_nodes)
    return (1, 1)


class JRSymbolicModel:
    """Jansen-Rit adapter fulfilling :class:`SymbolicModel`.

    Thin wrapper over a :class:`JRSymbolicParams` (numeric for MPC, symbolic for the
    SysID decision variables) plus the integration step ``dt``; delegates to the
    functions in :mod:`neuro.jansen_rit_casadi`. The electrode-to-node projection is a
    JR-internal detail handled inside :meth:`step`.
    """

    def __init__(self, params: JRSymbolicParams, dt: float, *, jit: bool = False) -> None:
        """Build the adapter from its symbolic parameters and timestep."""
        self.params = params
        self.dt = dt
        self.jit = jit
        self.free_syms: dict[str, ca.MX] = {}

        n_nodes = params.w_weights.shape[1]
        self.state_shape: tuple[int, int] = (6, n_nodes)
        self.history_depth = int(np.max(params.delay_steps))
        self.n_channels = params.eeg_gain.shape[0]

        gamma = params.gamma
        if isinstance(gamma, (ca.SX, ca.MX)):
            # Symbolic gamma (a SysID decision variable) flows straight into
            # project_control; it is already shaped (n_elec, n_nodes).
            self.gamma = gamma
        else:
            gamma = np.asarray(gamma, dtype=np.float64)
            if gamma.ndim == 1:
                gamma = gamma.reshape(1, -1)
            self.gamma = gamma
        self.n_elec = self.gamma.shape[0]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Build the model from a global config dict (numeric parameters)."""
        return cls(_jr_symbolic_params(JansenRitParams.from_config(config)), float(config["dt"]))

    @classmethod
    def with_free_params(cls, base: JansenRitParams, dt: float, free_params: list[str], *, jit: bool = False) -> Self:
        """Build a model whose ``free_params`` are symbolic ``ca.Function`` inputs.

        Non-free parameters are baked in numerically from ``base``; each free parameter
        becomes a fresh ``MX.sym`` of its natural shape, stored on ``free_syms`` in
        ``free_params`` order. ``f_step``/``f_out`` then take those placeholders as
        trailing inputs, and a solver grafts its decision variables onto them per call.
        """
        params = _jr_symbolic_params(base)
        n_nodes = base.w_weights.shape[0]
        n_elec = params.gamma.shape[0]
        n_channels = base.eeg_gain.shape[0]

        free_syms = {p: ca.MX.sym(p, *_free_param_shape(p, n_nodes, n_elec, n_channels)) for p in free_params}
        for p, sym in free_syms.items():
            setattr(params, p, sym)

        model = cls(params, dt, jit=jit)
        model.free_syms = free_syms
        return model

    @property
    def extra_inputs(self) -> tuple[ca.MX, ...]:
        """Free-parameter placeholders, in ``free_syms`` order, used as trailing Function inputs."""
        return tuple(self.free_syms.values())

    def step(self, history: Sequence[ca.SX | ca.MX], u: ca.SX | ca.MX) -> ca.SX | ca.MX:
        """Advance the delayed network one step; ``u`` is the raw (electrode-space) control."""
        u_proj = project_control(u, self.gamma)
        return heun_step(history, u_proj, self.params, self.dt)

    def output(self, x: ca.SX | ca.MX) -> ca.SX | ca.MX:
        """Measure the EEG output from a live state."""
        return eeg(x, self.params.eeg_gain)

    @property
    def _compile_opts(self) -> dict[str, Any]:
        """CasADi ``Function`` options; ``jit`` compiles with the system C compiler at ``-O3``."""
        return {"jit": True, "compiler": "shell", "jit_options": {"flags": ["-O3"]}} if self.jit else {}

    @cached_property
    def f_step(self) -> ca.Function:
        """Reusable compiled single-step function ``(xh0..xhd, u, *free) -> x_next`` (cached).

        Calling it in a horizon loop inserts one call node per step instead of inlining the
        whole ``O(N^2)`` step graph at every step, keeping the build and per-solve derivative
        graphs small on whole-brain networks. This is why callers must never enable Opti's
        ``expand`` (it re-inlines the call nodes and reintroduces the construction blow-up).
        The trailing ``free`` inputs are the :attr:`free_syms` placeholders, so a solver can
        graft decision variables onto them at each call site.
        """
        xh_syms = [ca.MX.sym(f"xh{i}", *self.state_shape) for i in range(self.history_depth + 1)]
        u_sym = ca.MX.sym("u_step", self.n_elec, 1)
        return ca.Function(
            "F_step", [*xh_syms, u_sym, *self.extra_inputs], [self.step(xh_syms, u_sym)], self._compile_opts
        )

    @cached_property
    def f_out(self) -> ca.Function:
        """Reusable compiled output function ``(x, *free) -> y`` (cached)."""
        x_out = ca.MX.sym("x_out", *self.state_shape)
        return ca.Function("F_out", [x_out, *self.extra_inputs], [self.output(x_out)], self._compile_opts)
