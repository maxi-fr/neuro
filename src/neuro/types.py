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
    def f_step(self) -> ca.Function:
        """Symbolic state step function."""
        ...

    @property
    def f_out(self) -> ca.Function:
        """Symbolic output function."""
        ...
