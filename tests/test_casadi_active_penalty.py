from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from test_casadi_mpc import _build_checkpoint, _observable_fixture, _ref
from test_observable_cnn import _model
from test_waveform_cnn import _cnn

from neuro.control.casadi import (
    CasADiMPCController,
    build_casadi_observable_problem,
    build_casadi_waveform_problem,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("observable", [False, True])
@pytest.mark.parametrize("cnn", [False, True])
def test_active_penalty_cost_and_physical_plan(tmp_path: Path, *, observable: bool, cnn: bool) -> None:
    """The weighted binary count augments existing Costs on four Predictor representations."""
    if observable:
        if cnn:
            artifact = tmp_path / "observable_cnn"
            _model(n_values=1, n_controls=2).save(artifact)
        else:
            artifact, _ = _observable_fixture(tmp_path)
        problem = build_casadi_observable_problem(
            artifact, horizon=2, u_max=[0.4, 0.6], w_u=0.1, w_u_l1=0.05, w_active=0.7
        )
    else:
        if cnn:
            artifact = tmp_path / "waveform_cnn"
            _cnn(depth=1).save(artifact)
        else:
            artifact = _build_checkpoint(tmp_path, depth=0, horizon=2)
        problem = build_casadi_waveform_problem(artifact, horizon=2, u_max=0.5, w_y=0.0, w_u=0.1, w_active=0.7)
    controller = CasADiMPCController(problem.model.dt, problem)
    controls = np.zeros((2, 2))
    active = np.array([1.0, 0.0])
    state = np.zeros(problem.model.n)
    costs = controller.decompose_cost(state, controls, active)
    assert costs["cost_active"] == pytest.approx(0.7)
    assert costs["active_count"] == 1
    assert costs["cost_total"] == pytest.approx(
        costs["cost_tracking"]
        + costs["cost_quadratic_effort"]
        + costs["cost_sparse_effort"]
        + costs["cost_spectral"]
        + costs["cost_active"]
    )
    measurement = np.zeros(problem.model.n_outputs)
    for step in range(problem.model.n_history):
        current, log = controller.update(step * problem.model.dt, np.zeros(1), measurement)
    assert log.success
    assert log.planned_active is not None
    np.testing.assert_allclose(log.planned_active, np.round(log.planned_active), atol=1e-6)
    assert np.all(np.abs(log.planned_u) <= problem.u_max * log.planned_active[:, None] + 1e-6)
    np.testing.assert_allclose(log.planned_u.sum(axis=1), 0.0, atol=1e-6)
    np.testing.assert_allclose(current, log.planned_u[0])
    assert log.active_count == pytest.approx(float(np.sum(log.planned_active)))
    assert log.active_count is not None
    assert log.cost_active == pytest.approx(0.7 * log.active_count)
    np.testing.assert_allclose(controller._active_guess, np.r_[log.planned_active[1:], log.planned_active[-1]])  # noqa: SLF001 -- shifted warm-start contract


@pytest.mark.parametrize("observable", [False, True])
def test_active_penalty_configuration_rejects_cap_and_invalid_weights(tmp_path: Path, *, observable: bool) -> None:
    """Only one integer control mode can be configured with a valid weight."""
    artifact = _observable_fixture(tmp_path)[0] if observable else _build_checkpoint(tmp_path, depth=0, horizon=2)
    for weight in (-1.0, float("inf"), float("nan")):
        if observable:
            with pytest.raises(ValueError, match="w_active"):
                build_casadi_observable_problem(artifact, horizon=2, u_max=0.5, w_active=weight)
        else:
            with pytest.raises(ValueError, match="w_active"):
                build_casadi_waveform_problem(artifact, horizon=2, u_max=0.5, w_y=0.0, w_active=weight)
    if observable:
        with pytest.raises(ValueError, match="mutually exclusive"):
            build_casadi_observable_problem(artifact, horizon=2, u_max=0.5, w_active=0.2, max_active_intervals=1)
    else:
        with pytest.raises(ValueError, match="mutually exclusive"):
            build_casadi_waveform_problem(artifact, horizon=2, u_max=0.5, w_y=0.0, w_active=0.2, max_active_intervals=1)


def test_active_penalty_failed_solve_applies_zero_current(tmp_path: Path) -> None:
    """A failed penalty-mode solve retains the zero-current safety behavior."""
    artifact = _build_checkpoint(tmp_path, depth=0, horizon=2)
    problem = build_casadi_waveform_problem(artifact, horizon=2, u_max=0.5, w_y=0.0, w_active=0.2)
    controller = CasADiMPCController(problem.model.dt, problem)
    measurement = np.zeros(problem.model.n_outputs)
    for step in range(problem.model.n_history):
        controller.update(step * problem.model.dt, np.zeros(1), measurement)

    class FailedSolver:
        """Raise when invoked."""

        def __call__(self, **kwargs: object) -> None:  # noqa: ARG002 -- simulate a failed solver call
            """Simulate an unavailable solver result."""
            msg = "solver failed"
            raise RuntimeError(msg)

    controller._solver = FailedSolver()  # ty: ignore[invalid-assignment]  # noqa: SLF001 -- exercise failure behavior
    current, log = controller.update(problem.model.n_history * problem.model.dt, np.zeros(1), measurement)
    np.testing.assert_array_equal(current, np.zeros(problem.model.n_controls))
    assert not log.success
    assert log.status == "solver failed"
    assert log.planned_active is not None
