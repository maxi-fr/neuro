from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from simulate.simulation import Simulation

from neuro.comparison import lfp_logging, spread_metrics
from neuro.connectome import Connectome
from neuro.seizure import SEIZURE_PTP_MV

if TYPE_CHECKING:
    from neuro.types import FloatArray


SEEDS = (7000, 7001, 7002, 7004, 7005)
CURRENT_ZERO_MA = 1e-6
ACTIVE_DECISION_THRESHOLD = 0.5
MAX_SEIZING_REGIONS_EXCLUSIVE = 5
REQUIRED_SUPPRESSED_SEEDS = 4


@dataclass(frozen=True)
class SolveFailureError(Exception):
    """Status of the first unsuccessful controller solve."""

    status: str
    time_s: float
    solve_time_s: float


def controller_arms(base: dict[str, Any], *, cap: int, weight: float) -> dict[str, dict[str, Any]]:
    """Build paired continuous, capped, and penalized CasADi controller configurations."""
    arms = {}
    for name, setting in (
        ("continuous", {}),
        ("capped", {"max_active_intervals": cap}),
        ("penalized", {"w_active": weight}),
    ):
        config = copy.deepcopy(base)
        config["t_end"] = 12.0
        config["controller"]["class_path"] = "neuro.control.casadi.CasADiMPCController"
        problem = config["controller"]["problem"]
        problem["class_path"] = "neuro.control.casadi.build_casadi_observable_problem"
        problem.pop("max_active_intervals", None)
        problem.pop("w_active", None)
        problem.update(setting)
        arms[name] = config
    return arms


def delivery_metrics(
    times: FloatArray,
    currents: FloatArray,
    active: FloatArray | None,
    *,
    end_s: float,
    zero_ma: float = CURRENT_ZERO_MA,
) -> dict[str, float | int | None]:
    """Reduce applied zero-order-held currents and first-step binary decisions."""
    t = np.asarray(times, dtype=np.float64).reshape(-1)
    u = np.asarray(currents, dtype=np.float64)
    durations = np.diff(np.append(t, end_s)) if len(t) else np.empty(0)
    delivered = np.any(np.abs(u) > zero_ma, axis=1) if len(t) else np.empty(0, dtype=bool)
    return {
        "enabled_steps": None
        if active is None
        else int(np.sum(np.asarray(active, dtype=np.float64) > ACTIVE_DECISION_THRESHOLD)),
        "applied_nonzero_steps": int(np.sum(delivered)),
        "delivered_current_duty_s": float(np.sum(durations[delivered])),
        "delivered_current_duty_cycle": float(np.sum(durations[delivered]) / end_s),
        "delivered_charge_mc": float(np.sum(np.abs(u) * durations[:, None])),
        "squared_current_ma2_s": float(np.sum(u**2 * durations[:, None])),
    }


def score_seed(config: dict[str, Any], *, threshold: float = SEIZURE_PTP_MV) -> dict[str, Any]:
    """Run one Plant seed, stopping at the first failed solve and retaining its status."""
    config = lfp_logging(config)
    sim = Simulation.from_config(config)
    controller = sim.controller
    original_update = controller.update

    def stop_on_failure(t: float, ref: FloatArray, x_hat: FloatArray) -> tuple[FloatArray, Any]:
        """Prevent the Plant from advancing after a failed optimization."""
        current, log = original_update(t, ref, x_hat)
        if not log.warmup and not log.success:
            raise SolveFailureError(log.status, t, log.solve_time)
        return current, log

    controller.update = stop_on_failure  # ty:ignore[invalid-assignment] -- wrap this run's controller update
    failure: SolveFailureError | None = None
    started = time.perf_counter()
    try:
        sim.run()
    except SolveFailureError as exc:
        failure = exc
    elapsed = time.perf_counter() - started
    if sim.logger is None:
        msg = "Simulation logger is missing after run."
        raise RuntimeError(msg)
    logger = sim.logger
    control_t, controls = logger.signal("controller", "u")
    available = {field for component, field in logger.signals() if component == "controller"}
    active = None
    if "planned_active" in available:
        plans = logger.signal("controller", "planned_active")[1]
        if plans.ndim > 1:
            active = plans[:, 0]
            if not np.isfinite(active).any():
                active = None
    end_s = float(failure.time_s if failure is not None else config["t_end"])
    solved = logger.signal("controller", "warmup")[1].reshape(-1) == 0
    times = logger.signal("controller", "solve_time")[1].reshape(-1)[solved]
    if failure is not None:
        times = np.append(times, failure.solve_time_s)
    result: dict[str, Any] = {
        "seed": config["dynamics"]["seed"],
        "completed": failure is None,
        "solver_failure_status": None if failure is None else failure.status,
        "stopped_at_s": end_s,
        "wall_time_s": elapsed,
        "solver_construction_s": getattr(controller, "construction_time_s", None),
        "solve_count": len(times),
        "solver_failures": int(failure is not None),
        "solve_time_mean_s": float(np.mean(times)) if len(times) else None,
        "solve_time_p95_s": float(np.quantile(times, 0.95)) if len(times) else None,
        "solve_time_max_s": float(np.max(times)) if len(times) else None,
        **delivery_metrics(control_t, controls, active, end_s=end_s),
        "n_seizing_final": None,
    }
    if failure is None and end_s >= 1.0:
        connectome = Connectome.from_config(config["dynamics"]["connectome"])
        result["n_seizing_final"] = int(
            spread_metrics(logger.signal("dynamics", "lfp")[1], sim.dt, connectome, threshold=threshold)[
                "n_seizing_final"
            ]
        )
    return result


def target_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Report completed seeds below five seizing regions per arm."""
    summary = {}
    for arm in {row["arm"] for row in rows}:
        arm_rows = [row for row in rows if row["arm"] == arm]
        suppressed = sum(
            row["completed"] and row["n_seizing_final"] < MAX_SEIZING_REGIONS_EXCLUSIVE for row in arm_rows
        )
        summary[arm] = {
            "suppressed_seeds": suppressed,
            "target_met": suppressed >= REQUIRED_SUPPRESSED_SEEDS,
            "completed_seeds": sum(row["completed"] for row in arm_rows),
        }
    return summary
