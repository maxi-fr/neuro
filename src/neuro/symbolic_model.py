"""Model-agnostic discrete-dynamics interface for the CasADi MPC and SysID solvers.

Defines the :class:`SymbolicModel` protocol -- the abstract ``x_next = step(x, u)``,
``y = output(x)`` contract the optimizers are written against -- and
:class:`JRSymbolicModel`, the Jansen-Rit adapter that fulfils it by reusing the
symbolic functions in :mod:`neuro.jansen_rit_casadi`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Self

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

    import casadi as ca


def load_config(filepath: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file into a raw dictionary."""
    with Path(filepath).open() as f:
        config = yaml.safe_load(f)
        if not isinstance(config, dict):
            msg = f"Config file {filepath} must be a dictionary"
            raise TypeError(msg)
        return config


class SymbolicModel(Protocol):
    """Abstract discrete-time model: ``x_next = step(history, u)``, ``y = output(x)``.

    The optimizers (MPC / SysID) drive a model purely through this interface, so the
    JR-specific dynamics stay isolated behind it.

    The "state" is the per-step live state of shape ``state_shape``. A delayed model
    carries its bounded past as a newest-first ``history`` list of length
    ``history_depth + 1``; a memoryless model has ``history_depth == 0`` and
    ``step([x], u)`` reduces to the pure ``F(x, u)`` map.
    """

    state_shape: tuple[int, int]
    history_depth: int
    n_elec: int
    n_channels: int

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate a symbolic model from a config dict."""
        ...

    def step(self, history: Sequence[ca.SX | ca.MX], u: ca.SX | ca.MX) -> ca.SX | ca.MX:
        """Advance one step: newest-first ``history`` bundle and control ``u`` -> next live state."""
        raise NotImplementedError

    def output(self, x: ca.SX | ca.MX) -> ca.SX | ca.MX:
        """Map a live state to the measured output ``y``."""
        ...


class JRSymbolicModel:
    """Jansen-Rit adapter fulfilling :class:`SymbolicModel`.

    Thin wrapper over a :class:`JRSymbolicParams` (numeric for MPC, symbolic for the
    SysID decision variables) plus the integration step ``dt``; delegates to the
    functions in :mod:`neuro.jansen_rit_casadi`. The electrode-to-node projection is a
    JR-internal detail handled inside :meth:`step`.
    """

    def __init__(self, params: JRSymbolicParams, dt: float) -> None:
        """Build the adapter from its symbolic parameters and timestep."""
        self.params = params
        self.dt = dt

        n_nodes = params.w_weights.shape[1]
        self.state_shape: tuple[int, int] = (6, n_nodes)
        self.history_depth = int(np.max(params.delay_steps))
        self.n_channels = params.eeg_gain.shape[0]

        gamma = np.asarray(params.gamma, dtype=np.float64)
        if gamma.ndim == 1:
            gamma = gamma.reshape(1, -1)
        self.gamma = gamma
        self.n_elec = gamma.shape[0]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Build the model from a global config dict (numeric parameters)."""
        base = JansenRitParams.from_config(config)
        n_nodes = base.w_weights.shape[0]
        params = JRSymbolicParams(
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
            gamma=base.gamma,
            K=base.K,
            w_weights=base.w_weights,
            delay_steps=base.delay_steps,
        )
        return cls(params, float(config["dt"]))

    def step(self, history: Sequence[ca.SX | ca.MX], u: ca.SX | ca.MX) -> ca.SX | ca.MX:
        """Advance the delayed network one step; ``u`` is the raw (electrode-space) control."""
        u_proj = project_control(u, self.gamma)
        return heun_step(history, u_proj, self.params, self.dt)

    def output(self, x: ca.SX | ca.MX) -> ca.SX | ca.MX:
        """Measure the EEG output from a live state."""
        return eeg(x, self.params.eeg_gain)
