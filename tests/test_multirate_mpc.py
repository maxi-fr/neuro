from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from simulate.logger import RamLogger
from test_mpc import _build_checkpoint, _ref
from trajopt.trajectory import Trajectory
from trajopt.transcription.ipopt import IpoptResult

from experiments.benchmark import benchmark_controller
from experiments.comparison import cost_metrics, solver_metrics
from experiments.ood import (
    _extract_control_metrics_from_log,
    _extract_solver_metrics_from_log,
    extract_mpc_rollout_queries,
)
from neuro.control.casadi import CasADiMPCController, CasADiMPCLog, build_casadi_waveform_problem
from neuro.control.mpc import TrajOptMPCController, TrajOptMPCLog
from neuro.control.problems import build_waveform_problem
from neuro.runs.record import Run
from neuro.runs.spectra import compare_spectra
from neuro.runs.view import ViewSettings

if TYPE_CHECKING:
    from pathlib import Path

    from trajopt.problem import BoundaryConditions
    from trajopt.program import Program, WarmStart

    from neuro.types import FloatArray


@pytest.mark.parametrize("block", [1, 6])
@pytest.mark.parametrize("backend", ["trajopt", "casadi"])
def test_native_absorption_and_distinct_prefix_playback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, block: int
) -> None:
    artifact = _build_checkpoint(tmp_path, n_y=6, n_u=6, horizon=12, depth=0)
    controller = (
        TrajOptMPCController(
            dt=0.01,
            problem=build_waveform_problem(artifact, horizon=12, control_block_steps=block, w_y=0.0, u_max=2.0),
            replan_dt=0.06,
            control_start_s=0.06,
        )
        if backend == "trajopt"
        else CasADiMPCController(
            dt=0.01,
            problem=build_casadi_waveform_problem(artifact, horizon=12, control_block_steps=block, w_y=0.0, u_max=2.0),
            replan_dt=0.06,
            control_start_s=0.06,
        )
    )
    measured, guesses, plans = [], [], []

    def plan() -> FloatArray:
        current = np.repeat((np.arange(12) + 1.0)[:, None], 2, axis=1) * 0.01
        return np.asarray(np.repeat(current[::block], block, axis=0) + 0.2 * len(measured), dtype=np.float64)

    if isinstance(controller, TrajOptMPCController):

        def solve(program: Program, bc: BoundaryConditions, ws: WarmStart) -> IpoptResult:
            measured.append(np.asarray(bc.x0).copy())
            guesses.append(np.asarray(ws.unpack(program.problem)[1]).copy())
            controls = jnp.asarray(plan())
            plans.append(np.asarray(controls).copy())
            updated = ws.with_primal(program.problem, U=controls)
            states, controls = updated.unpack(program.problem)
            return IpoptResult(
                Trajectory(states, controls, bc.t0 + jnp.arange(13) * 0.01, program.problem.dt),
                len(measured) != 2,
                0 if len(measured) != 2 else -1,
                "test",
                1.0,
                updated.Z,
                {},
                0.0,
            )

        monkeypatch.setattr(type(controller.mpc.program), "solve", solve)
    else:

        def solve_state(x0: FloatArray, initial_u: FloatArray, initial_active: object = None) -> CasADiMPCLog:
            measured.append(x0.copy())
            guesses.append(initial_u.copy())
            controls = plan()
            plans.append(controls.copy())
            return CasADiMPCLog(
                u=controls[0] if len(measured) != 2 else np.zeros(2),
                cost=1.0,
                success=len(measured) != 2,
                warmup=False,
                status="test",
                solve_time=0.01,
                predicted_y=np.zeros((13, 2)),
                planned_u=controls,
            )

        monkeypatch.setattr(controller, "solve_state", solve_state)

    state = controller.model.initial_state()
    previous, logs, absorbed = np.zeros(2), [], []
    for k in range(19):
        measurement = np.full(2, k * 0.1)
        state = controller.model.absorb(state, measurement, previous)
        command, log = controller.update(k * 0.01, np.zeros(1), measurement)
        logs.append(log)
        absorbed.append(state.copy())
        if k < 6:
            assert log.warmup
            assert log.plan_index == -1
            np.testing.assert_array_equal(command, 0.0)
        elif k < 12:
            np.testing.assert_allclose(command, plans[0][k - 6])
            assert log.plan_index == k - 6
        elif k < 18:
            assert log.fallback
            np.testing.assert_array_equal(command, 0.0)
        if not log.solve_attempted:
            assert not log.success
            assert np.isnan(log.solve_time)
            assert np.isnan(log.predicted_y).all()
            assert np.isnan(log.planned_u).all()
        assert log.controller_time >= 0.0
        previous = command.copy()
    assert [k for k, log in enumerate(logs) if log.solve_attempted] == [6, 12, 18]
    np.testing.assert_allclose(guesses[1][:-6], plans[0][6:])
    np.testing.assert_allclose(guesses[1][-6:], np.repeat(plans[0][-1:], 6, axis=0))
    np.testing.assert_allclose(measured, np.asarray(absorbed)[[6, 12, 18]])
    assert logs[12].planned_u[0, 0] != 0.0


@pytest.mark.parametrize(
    ("replan_dt", "start", "horizon"),
    [(0.025, None, 12), (0.0, None, 12), (0.06, 0.07, 12), (0.06, 0.0, 12), (0.06, None, 5)],
)
@pytest.mark.parametrize("backend", ["trajopt", "casadi"])
def test_invalid_execution_timing_fails_at_construction(
    tmp_path: Path, backend: str, replan_dt: float, start: float | None, horizon: int
) -> None:
    artifact = _build_checkpoint(tmp_path, n_y=4, horizon=12, depth=0)
    if backend == "trajopt":
        with pytest.raises(ValueError, match=r"replan_dt|Horizon|control_start_s"):
            TrajOptMPCController(
                dt=0.01,
                problem=build_waveform_problem(artifact, horizon=horizon, w_y=0.0, u_max=2.0),
                replan_dt=replan_dt,
                control_start_s=start,
            )
    else:
        with pytest.raises(ValueError, match=r"replan_dt|Horizon|control_start_s"):
            CasADiMPCController(
                dt=0.01,
                problem=build_casadi_waveform_problem(artifact, horizon=horizon, w_y=0.0, u_max=2.0),
                replan_dt=replan_dt,
                control_start_s=start,
            )


def test_attempt_metrics_include_failure_and_exclude_playback(tmp_path: Path) -> None:
    logger = RamLogger(5)
    for k in range(5):
        attempted = k in (1, 4)
        logger.log(
            k * 0.01,
            {
                "controller": TrajOptMPCLog(
                    u=np.zeros(2),
                    cost=1.0 if attempted else np.nan,
                    success=k == 1,
                    warmup=k == 0,
                    solve_time=(0.02 if k == 1 else 0.08) if attempted else np.nan,
                    predicted_y=np.full((2, 2), np.nan),
                    planned_u=np.full((1, 2), np.nan),
                    solve_attempted=attempted,
                    cost_spectral=3.0 if attempted else np.nan,
                    cost_quadratic_effort=2.0 if attempted else np.nan,
                    cost_sparse_effort=0.0 if attempted else np.nan,
                    controller_time=0.001,
                )
            },
        )
    result = solver_metrics(logger)
    assert result["solve_count"] == 2
    assert result["solve_failures"] == 1
    assert result["solve_success_rate"] == 0.5
    assert result["solve_time_mean_s"] == pytest.approx(0.05)
    assert result["solve_time_p95_s"] == pytest.approx(np.quantile([0.02, 0.08], 0.95))
    assert result["controller_time_s"] == pytest.approx(0.005)
    assert cost_metrics(logger)["cost_spectral_mean"] == 3.0
    arrays = {f"controller.{name}": logger.signal("controller", name)[1] for _, name in logger.signals()}
    ood_metrics = {}
    _extract_solver_metrics_from_log(arrays, ood_metrics)
    assert ood_metrics["solve_success_rate"] == result["solve_success_rate"]
    assert ood_metrics["solve_time_p95_s"] == result["solve_time_p95_s"]


def test_plan_filtering_preserves_native_observation_targets(tmp_path: Path) -> None:
    times = np.arange(13, dtype=np.float64) * 0.01
    observed = np.sin(np.arange(13, dtype=np.float64))[:, None]
    predictions = np.full((13, 7, 1), np.nan)
    predictions[0] = observed[:7]
    predictions[6] = observed[6:13]
    run = Run(
        tmp_path,
        {"controller": {"dt": 0.01}},
        {
            "controller.t": times,
            "controller.solve_attempted": np.array([k in (0, 6) for k in range(13)], dtype=np.float64),
            "controller.predicted_y": predictions,
            "estimator.t": times,
            "estimator.x_hat": observed,
        },
    )
    series = run.planned_predictions()
    np.testing.assert_allclose(series.t, [0.0, 0.06])
    np.testing.assert_allclose(series.observation_times, times)
    result = compare_spectra(series, ViewSettings(decision=0.06, spectral_steps=6))
    assert result is not None
    np.testing.assert_allclose(result.predicted, result.observed)


def test_ood_attempt_queries_keep_native_measurement_and_current_history(tmp_path: Path) -> None:
    path = tmp_path / "log.npz"
    predictions = np.full((13, 3, 1), np.nan)
    predictions[6, :, 0] = [6, 60, 600]
    predictions[12, :, 0] = [12, 120, 1200]
    plans = np.full((13, 2, 1), np.nan)
    plans[[6, 12], :, 0] = [[100, 101], [200, 201]]
    np.savez(
        path,
        allow_pickle=False,
        **{
            "controller.t": np.arange(13) * 0.01,
            "controller.predicted_y": predictions,
            "controller.planned_u": plans,
            "controller.u": np.arange(13, dtype=np.float64)[:, None] + 20.0,
            "controller.solve_attempted": np.array([k in (6, 12) for k in range(13)]),
            "estimator.t": np.arange(13) * 0.01,
            "estimator.x_hat": np.arange(13, dtype=np.float64)[:, None],
        },
    )
    queries, _, indices = extract_mpc_rollout_queries(path, n_y=3, n_u=3)
    np.testing.assert_array_equal(indices, [6, 12])
    np.testing.assert_allclose(queries[0, 0], [4, 5, 6, 24, 25, 100])
    np.testing.assert_allclose(queries[0, 1], [5, 6, 60, 25, 100, 101])


def test_charge_uses_irregular_native_intervals_and_clips_endpoint() -> None:
    metrics = {}
    _extract_control_metrics_from_log(
        {
            "controller.t": np.array([0.0, 0.01, 0.04, 0.1, 0.11]),
            "controller.u": np.array([[1.0], [2.0], [3.0], [999.0], [999.0]]),
        },
        {"t_end": 0.1},
        metrics,
    )
    assert metrics["delivered_charge"] == pytest.approx(0.01 + 0.06 + 0.18)


@pytest.mark.parametrize("block", [1, 6])
@pytest.mark.parametrize("backend", ["trajopt", "casadi"])
def test_common_release_and_controller_execution_benchmark(tmp_path: Path, backend: str, block: int) -> None:
    first_solves = []
    for history in (2, 6):
        artifact = _build_checkpoint(tmp_path, n_y=history, n_u=2, horizon=12, depth=0)
        if backend == "trajopt":
            controller = TrajOptMPCController.from_config(
                {
                    "dt": 0.01,
                    "replan_dt": 0.06,
                    "control_start_s": 0.06,
                    "problem": build_waveform_problem(
                        artifact, horizon=12, control_block_steps=block, w_y=0.0, u_max=2.0
                    ),
                }
            )
            identity = controller.mpc.program
        else:
            controller = CasADiMPCController.from_config(
                {
                    "dt": 0.01,
                    "replan_dt": 0.06,
                    "control_start_s": 0.06,
                    "problem": build_casadi_waveform_problem(
                        artifact, horizon=12, control_block_steps=block, w_y=0.0, u_max=2.0
                    ),
                }
            )
            identity = controller._solver  # noqa: SLF001 -- compiled solver identity is the lifetime contract
        for k in range(13):
            current, log = controller.evaluate(k * 0.01, np.zeros(1), np.ones(2))
            held, held_log = controller.evaluate((k + 0.5) * 0.01, np.zeros(1), np.full(2, 999.0))
            np.testing.assert_allclose(held, current)
            assert held_log is log
            if k < 6:
                np.testing.assert_array_equal(current, 0.0)
                assert not log.solve_attempted
            if k == 6:
                first_solves.append(k)
                assert log.solve_attempted
        if isinstance(controller, TrajOptMPCController):
            assert controller.mpc.program is identity
        else:
            assert controller._solver is identity  # noqa: SLF001 -- no playback-time solver reconstruction
    assert first_solves == [6, 6]

    artifact = _build_checkpoint(tmp_path, n_y=1, n_u=1, horizon=12, depth=0)
    controller = TrajOptMPCController(
        dt=0.01,
        replan_dt=0.06,
        problem=build_waveform_problem(artifact, horizon=12, w_y=0.0, u_max=2.0),
    )
    result = benchmark_controller(controller, np.ones((13, 2)))
    assert result["solve_count"] == 3
    assert result["solves_per_simulated_second"] == pytest.approx(3 / 0.13)
    assert result["controller_time_s"] >= result["first_attempt_controller_time_s"]
    assert result["warm_solve_time_mean_s"] >= 0.0


@pytest.mark.parametrize("backend", ["trajopt", "casadi"])
def test_playback_uses_its_own_prefix_without_numerical_reevaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    artifact = _build_checkpoint(tmp_path, n_y=1, n_u=1, horizon=12, depth=0)
    if backend == "trajopt":
        controller = TrajOptMPCController(
            dt=0.01,
            replan_dt=0.06,
            problem=build_waveform_problem(artifact, horizon=12, u_max=2.0, reference=_ref(2)),
        )
    else:
        controller = CasADiMPCController(
            dt=0.01,
            replan_dt=0.06,
            problem=build_casadi_waveform_problem(artifact, horizon=12, u_max=2.0, reference=_ref(2)),
        )
    _, solved = controller.update(0.0, np.zeros(1), np.ones(2))
    assert solved.success
    expected = solved.planned_u.copy()
    solved.planned_u.fill(999.0)

    def unexpected(*_args: object, **_kwargs: object) -> None:
        pytest.fail("playback must not solve, roll out, or reevaluate Cost")

    monkeypatch.setattr(controller, "_plan", unexpected)
    if isinstance(controller, TrajOptMPCController):
        monkeypatch.setattr("neuro.control.mpc.planned_outputs", unexpected)
        monkeypatch.setattr("neuro.control.mpc.decompose_cost", unexpected)
    else:
        monkeypatch.setattr(controller, "_predicted_y_fn", unexpected)
        monkeypatch.setattr(controller, "_costs_fn", unexpected)
    for k in range(1, 6):
        command, log = controller.update(k * 0.01, np.zeros(1), np.full(2, 100.0 * k))
        np.testing.assert_allclose(command, expected[k])
        assert not log.solve_attempted


@pytest.mark.parametrize("backend", ["trajopt", "casadi"])
def test_default_readiness_keeps_zero_anchored_phase(tmp_path: Path, backend: str) -> None:
    artifact = _build_checkpoint(tmp_path, n_y=4, n_u=2, horizon=12, depth=0)
    if backend == "trajopt":
        controller = TrajOptMPCController(
            dt=0.01,
            replan_dt=0.06,
            problem=build_waveform_problem(artifact, horizon=12, w_y=0.0, u_max=2.0),
        )
    else:
        controller = CasADiMPCController(
            dt=0.01,
            replan_dt=0.06,
            problem=build_casadi_waveform_problem(artifact, horizon=12, w_y=0.0, u_max=2.0),
        )
    attempts = []
    for k in range(8):
        _, log = controller.update(k * 0.01, np.zeros(1), np.ones(2))
        if log.solve_attempted:
            attempts.append(k)
    assert attempts == [6]
