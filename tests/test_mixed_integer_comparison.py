import dataclasses

import numpy as np
import pytest
from simulate.logger import create_logger

from neuro.control.casadi import CasADiMPCLog
from neuro.mixed_integer_comparison import controller_arms, delivery_metrics, score_seed, target_summary


def test_delivery_uses_applied_currents_and_first_binary_decisions() -> None:
    """Count enabled intervals separately from current that reached the Plant."""
    metrics = delivery_metrics(
        np.array([0.0, 0.1, 0.2]),
        np.array([[0.0, 0.0], [0.2, -0.2], [1e-7, -1e-7]]),
        np.array([1.0, 1.0, 0.0]),
        end_s=0.3,
    )
    assert metrics["enabled_steps"] == 2
    assert metrics["applied_nonzero_steps"] == 1
    assert metrics["delivered_current_duty_s"] == pytest.approx(0.1)
    assert metrics["delivered_charge_mc"] == pytest.approx(0.04000002)
    assert metrics["squared_current_ma2_s"] == pytest.approx(0.008000000000002)


def test_three_arms_keep_shared_config_and_target_requires_four_completed_seeds() -> None:
    """Vary only the mode while retaining the five-seed suppression rule."""
    base = {
        "controller": {"class_path": "old", "problem": {"class_path": "old", "u_max": 2.0}},
        "dynamics": {"seed": 1},
    }
    arms = controller_arms(base, cap=3, weight=0.5)
    assert set(arms) == {"continuous", "capped", "penalized"}
    assert arms["capped"]["controller"]["problem"]["max_active_intervals"] == 3
    assert arms["penalized"]["controller"]["problem"]["w_active"] == 0.5
    assert "max_active_intervals" not in arms["continuous"]["controller"]["problem"]
    assert all(arm["t_end"] == 12.0 for arm in arms.values())
    rows = [{"arm": "capped", "completed": True, "n_seizing_final": count} for count in (2, 3, 4, 4)] + [
        {"arm": "capped", "completed": False, "n_seizing_final": None}
    ]
    assert target_summary(rows)["capped"] == {"suppressed_seeds": 4, "target_met": True, "completed_seeds": 4}


def test_solver_status_logger_accepts_status_longer_than_warmup() -> None:
    """Retain Bonmin's status after the warmup row initializes logger buffers."""
    logger = create_logger({"controller": 2})
    warmup = CasADiMPCLog(
        u=np.zeros(2),
        cost=0.0,
        success=True,
        warmup=True,
        status="warmup",
        solve_time=0.0,
        predicted_y=np.zeros((3, 1)),
        planned_u=np.zeros((2, 2)),
    )
    logger.log(0.0, {"controller": warmup})
    logger.log(0.1, {"controller": dataclasses.replace(warmup, status="SUCCESS", warmup=False)})
    assert logger.signal("controller", "status")[1].tolist() == ["warmup", "SUCCESS"]


def test_failed_solve_stops_plant_before_advancing_and_preserves_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the failed solver result while preventing its current from reaching the Plant."""

    class Controller:
        construction_time_s = 0.2

        def update(self, _t: float, _ref: np.ndarray, _x_hat: np.ndarray) -> tuple[np.ndarray, CasADiMPCLog]:
            """Return a failed solve with a current the Plant must never receive."""
            current = np.ones(2)
            return current, CasADiMPCLog(
                u=current,
                cost=0.0,
                success=False,
                warmup=False,
                status="INFEASIBLE",
                solve_time=0.3,
                predicted_y=np.zeros((3, 1)),
                planned_u=np.zeros((2, 2)),
            )

    class FakeSimulation:
        def __init__(self) -> None:
            """Initialize a single warmup log and an untouched Plant."""
            self.controller = Controller()
            self.logger = create_logger({"controller": 1})
            self.advanced = False

        def run(self) -> None:
            """Stop before advancing when the wrapped update reports failure."""
            warmup = CasADiMPCLog(
                u=np.zeros(2),
                cost=0.0,
                success=True,
                warmup=True,
                status="warmup",
                solve_time=0.0,
                predicted_y=np.zeros((3, 1)),
                planned_u=np.zeros((2, 2)),
            )
            self.logger.log(0.0, {"controller": warmup})
            self.controller.update(0.1, np.zeros(1), np.zeros(1))
            self.advanced = True

    sim = FakeSimulation()
    monkeypatch.setattr("neuro.mixed_integer_comparison.Simulation.from_config", lambda _config: sim)
    result = score_seed({"t_end": 12.0, "dynamics": {"seed": 7000}})
    assert result["solver_failure_status"] == "INFEASIBLE"
    assert result["solver_failures"] == 1
    assert result["solve_time_mean_s"] == 0.3
    assert result["completed"] is False
    assert sim.advanced is False
