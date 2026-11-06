from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from neuro.types import BoolArray, FloatArray


def current_hold_durations(times: FloatArray, end_s: float) -> FloatArray:
    """Measure each logged current hold through the next timestamp or the run end."""
    boundaries = np.minimum(np.append(times, end_s), end_s)
    return np.maximum(0.0, np.diff(boundaries))


def delivered_charge(currents: FloatArray, durations: FloatArray) -> float:
    """Integrate absolute applied current over its hold durations."""
    return float(np.sum(np.abs(currents) * durations[:, None]))


def kirchhoff_residual(currents: FloatArray) -> float:
    """Return the largest absolute sum of electrode currents at a logged tick."""
    return float(np.max(np.abs(np.sum(currents, axis=1))))


def solve_attempt_mask(warmup: FloatArray | BoolArray, attempted: FloatArray | BoolArray | None = None) -> BoolArray:
    """Select actual attempts, using the Warm-up Period mask for older logs."""
    if attempted is not None:
        return attempted.reshape(-1).astype(bool)
    return ~warmup.reshape(-1).astype(bool)


def solve_time_metrics(times: FloatArray) -> dict[str, float]:
    """Summarize runtimes of actual attempts, with NaN for an empty sample."""
    if len(times) == 0:
        return dict.fromkeys(
            ("solve_time_mean_s", "solve_time_p95_s", "solve_time_median_s", "solve_time_max_s"), float("nan")
        )
    return {
        "solve_time_mean_s": float(np.mean(times)),
        "solve_time_p95_s": float(np.quantile(times, 0.95)),
        "solve_time_median_s": float(np.median(times)),
        "solve_time_max_s": float(np.max(times)),
    }
