from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Self

import numpy as np
from pydantic import Field
from simulate.controller import Controller

from neuro.config import StrictConfig

if TYPE_CHECKING:
    from neuro.types import FloatArray


class _ZeroControllerConfig(StrictConfig):
    """Config schema for :class:`ZeroController`."""

    dt: float = Field(gt=0)
    n_u: int = Field(default=1, ge=1)


@dataclasses.dataclass(frozen=True)
class ZeroControllerLog:
    """Log carrying the zero control vector."""

    u: FloatArray


class ZeroController(Controller[ZeroControllerLog]):
    """Controller that ignores its inputs and always outputs a zero ``(n_u,)`` vector."""

    def __init__(self, dt: float, n_u: int = 1) -> None:
        """Initialize the zero controller for an ``n_u``-dimensional control."""
        super().__init__(dt)
        self.n_u = n_u

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary."""
        cfg = _ZeroControllerConfig.model_validate(config)
        return cls(dt=cfg.dt, n_u=cfg.n_u)

    def update(
        self,
        t: float,  # noqa: ARG002
        ref: FloatArray,  # noqa: ARG002
        x_hat: FloatArray,  # noqa: ARG002
    ) -> tuple[FloatArray, ZeroControllerLog]:
        """Return a zero control vector regardless of reference or state."""
        u = np.zeros(self.n_u, dtype=np.float64)
        return u, ZeroControllerLog(u=u)
