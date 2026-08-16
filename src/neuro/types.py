from collections.abc import Sequence
from typing import Protocol

import casadi as ca
import numpy as np
import numpy.typing as npt

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
