"""Controller components for the simulate framework.

Currently only :class:`ZeroController`, a no-op controller that always emits a
zero control vector. It is the controller to use for *open-loop* runs through the
:class:`~simulate.simulation.Simulation` orchestrator (which always requires a
controller), and a placeholder until a real neurostimulation control law lands.
"""

from __future__ import annotations

from typing import Any, Self, cast

import numpy as np
from pydantic import BaseModel, ConfigDict
from simulate.controller import Controller

from neuro.plant import FloatArray  # noqa: TC001  (runtime import: pydantic resolves the log field)


class ZeroControllerLog(BaseModel):
    """Pydantic model for ZeroController logging."""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ZeroController(Controller[ZeroControllerLog]):
    """Controller that ignores its inputs and always outputs a zero vector.

    Emitting a fixed ``(n_u, 1)`` control sidesteps the dimensional mismatch the
    stock :class:`~simulate.controller.PIDController` hits on a vector-output
    plant: the orchestrator seeds the loop with a scalar ``y_k = 0.0``, so the
    first estimated state is a scalar while every later one is the full EEG
    measurement vector -- no single PID gain shape fits both steps.
    """

    def __init__(self, dt: float, n_u: int = 1) -> None:
        """Initialize the zero controller for an ``n_u``-dimensional control."""
        super().__init__(dt)
        self.n_u = n_u

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary."""
        return cls(dt=float(config["dt"]), n_u=int(config.get("n_u", 1)))

    def update(
        self,
        t: float,  # noqa: ARG002
        ref: float | np.ndarray,  # noqa: ARG002
        x_hat: float | np.ndarray,  # noqa: ARG002
    ) -> tuple[float | np.ndarray, ZeroControllerLog]:
        """Return a zero control vector regardless of reference or state."""
        u_vec = np.zeros((self.n_u, 1), dtype=np.float64)
        return cast("FloatArray", self.from_col_vec(u_vec)), ZeroControllerLog()
