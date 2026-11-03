from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from neuro.types import FloatArray


def execution_steps(dt: float, replan_dt: float, horizon: int, control_start_s: float | None) -> tuple[int, int]:
    """Validate the native timing grid and return prefix length and common release index."""
    if not math.isfinite(dt) or dt <= 0 or not math.isfinite(replan_dt) or replan_dt <= 0:
        msg = "native dt and replan_dt must be positive and finite"
        raise ValueError(msg)
    ratio = replan_dt / dt
    steps = round(ratio)
    if steps < 1 or not math.isclose(ratio, steps, rel_tol=1e-9):
        msg = "replan_dt must be a positive integer multiple of native dt"
        raise ValueError(msg)
    if horizon < steps:
        msg = "Control Horizon must cover the execution prefix"
        raise ValueError(msg)
    start = 0.0 if control_start_s is None else control_start_s
    if not math.isfinite(start) or start < 0:
        msg = "control_start_s must be nonnegative and finite"
        raise ValueError(msg)
    release = round(start / dt)
    if not math.isclose(start / dt, release, abs_tol=1e-9) or release % steps:
        msg = "control_start_s must lie on the replanning grid anchored at time zero"
        raise ValueError(msg)
    return steps, release


def validate_control_blocks(block: int, horizon: int, prefix: int) -> None:
    """Require uniform blocks of one native step or one complete execution prefix."""
    if isinstance(block, (bool, np.bool_)) or not isinstance(block, (int, np.integer)):
        msg = "control_block_steps must be a positive integer"
        raise TypeError(msg)
    if block < 1 or block not in (1, prefix) or horizon % block:
        msg = "control_block_steps must be one or the execution prefix and divide the native Control Horizon"
        raise ValueError(msg)


class CommandPlayback:
    """Native-grid solve boundaries and a copied physical-current execution prefix."""

    def __init__(
        self,
        dt: float,
        replan_dt: float,
        horizon: int,
        control_start_s: float | None,
        priming_steps: int,
    ) -> None:
        """Fix the timing grid and reject an explicit release shorter than Predictor Priming."""
        self.steps, self.release = execution_steps(dt, replan_dt, horizon, control_start_s)
        if control_start_s is not None and self.release < priming_steps - 1:
            msg = "control_start_s is too early for Predictor Priming"
            raise ValueError(msg)
        self.dt = dt
        self.explicit_start = control_start_s is not None
        self.plan_tick: int | None = None
        self.prefix: FloatArray | None = None

    def phase(self, t: float, *, ready: bool) -> tuple[int, bool, bool]:
        """Return native index, Warm-up status and solve flag without resetting the grid to readiness."""
        tick = round(t / self.dt)
        if not math.isclose(t / self.dt, tick, abs_tol=1e-7):
            msg = "controller update time must lie on the native grid"
            raise ValueError(msg)
        if self.explicit_start and tick >= self.release and not ready:
            msg = "control_start_s is inadequate for the absorbed Estimator/Predictor history"
            raise ValueError(msg)
        due = tick >= self.release and tick % self.steps == 0 and ready
        warmup = self.plan_tick is None and not due
        return tick, warmup, due

    def store(self, tick: int, plan: FloatArray, *, success: bool) -> None:
        """Copy a successful execution prefix or invalidate it after a failed solve."""
        self.plan_tick = tick
        self.prefix = plan[: self.steps].copy() if success else None

    def emit(self, tick: int, n_controls: int) -> tuple[FloatArray, int, bool]:
        """Emit a queued physical current or zero fallback, retaining its native plan index."""
        index = -1 if self.plan_tick is None else tick - self.plan_tick
        fallback = self.plan_tick is not None and self.prefix is None
        current = np.zeros(n_controls) if self.prefix is None else self.prefix[index].copy()
        return current, index, fallback
