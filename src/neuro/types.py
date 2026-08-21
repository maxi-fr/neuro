from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Sequence

    import casadi as ca

    from neuro.config import ObservableGeometry

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]
StrArray = npt.NDArray[np.str_]


class SymbolicModel(Protocol):
    """Protocol shared by CasADi symbolic models (NNSymbolicModel, ESNSymbolicModel)."""

    @property
    def state_shape(self) -> tuple[int, int]:
        """State dimension shape."""
        ...

    @property
    def n_controls(self) -> int:
        """Control input count."""
        ...

    @property
    def n_channels(self) -> int:
        """Output channel count."""
        ...

    @property
    def native_horizon(self) -> int:
        """Native prediction horizon of the underlying artifact."""
        ...

    @property
    def is_linear(self) -> bool:
        """Whether the model represents a linear dynamic system."""
        ...

    @property
    def f_step(self) -> ca.Function:
        """Symbolic state step function."""
        ...

    @property
    def f_out(self) -> ca.Function:
        """Symbolic output function."""
        ...

    def step(self, history: Sequence[ca.SX | ca.MX], u: ca.SX | ca.MX) -> ca.SX | ca.MX:
        """Advance symbolic state by one step under control input u."""
        ...

    def output(self, x: ca.SX | ca.MX) -> ca.SX | ca.MX:
        """Decode symbolic state to raw EEG output."""
        ...

    def initial_state(self) -> FloatArray:
        """Return the unprimed or zero initial state vector."""
        ...

    def absorb(self, state: FloatArray, y: FloatArray, u: FloatArray) -> FloatArray:
        """Absorb a new measurement y and previously applied control u into state."""
        ...

    def is_ready(self, state: FloatArray) -> bool:
        """Return True if state has absorbed sufficient history to begin control."""
        ...


class ObservableModel(Protocol):
    """Protocol for models that forecast an Observable over the Control Horizon in one shot.

    A sibling of :class:`SymbolicModel`, not a variant of it: the stepping members are absent, so a
    model lacking them fails at build time instead of degrading the autoregressive path.
    """

    @property
    def state_shape(self) -> tuple[int, int]:
        """History state dimension shape."""
        ...

    @property
    def n_controls(self) -> int:
        """Control input count."""
        ...

    @property
    def n_channels(self) -> int:
        """Output channel count."""
        ...

    @property
    def n_values(self) -> int:
        """Scored values a Frame carries per channel."""
        ...

    @property
    def fs(self) -> float:
        """Sampling frequency the Frame grid is resolved at."""
        ...

    @property
    def native_horizon(self) -> int:
        """Control Horizon in samples the underlying artifact was fit at."""
        ...

    @property
    def geometry(self) -> ObservableGeometry:
        """The resolved Observable geometry the model forecasts on."""
        ...

    def n_frames(self, horizon: int) -> int:
        """Count the Frames the forecast emits over ``horizon`` samples."""
        ...

    def forecast(self, x0: ca.SX | ca.MX, u_seq: ca.SX | ca.MX) -> ca.SX | ca.MX:
        """Forecast the log-Observable: ``(state, (horizon, m) controls) -> (channels * values, frames)``."""
        ...

    @property
    def f_forecast(self) -> ca.Function:
        """Symbolic forecast function at the native horizon."""
        ...

    def initial_state(self) -> FloatArray:
        """Return the unprimed or zero initial state vector."""
        ...

    def absorb(self, state: FloatArray, y: FloatArray, u: FloatArray) -> FloatArray:
        """Absorb a new measurement y and previously applied control u into state."""
        ...

    def is_ready(self, state: FloatArray) -> bool:
        """Return True if state has absorbed sufficient history to begin control."""
        ...
