from __future__ import annotations

from typing import TYPE_CHECKING

import casadi as ca
import jax.numpy as jnp
import numpy as np
import pytest
from test_casadi_mpc import _build_checkpoint, _observable_fixture, _ref
from test_observable_cnn import _model
from test_waveform_cnn import _cnn

from neuro.control.casadi import CasADiMPCController, _step_casadi, build_casadi_waveform_problem
from neuro.predictor.inference import ObservableCNNModel, WaveformCNNModel
from neuro.spectral import HealthyReference, ObservableEnvelope

if TYPE_CHECKING:
    from pathlib import Path


def test_waveform_cnn_fixed_sequence_parity(tmp_path: Path) -> None:
    """CasADi matches the waveform CNN's physical one-step and horizon outputs."""
    artifact = tmp_path / "cnn"
    _cnn().save(artifact)
    model = WaveformCNNModel.load(artifact)
    rng = np.random.default_rng(401)
    state = rng.normal(size=model.n)
    controls = rng.normal(size=(3, model.n_controls))
    x_sym = ca.SX.sym("x", model.n)
    u_sym = ca.SX.sym("u", model.n_controls)
    step = ca.Function("cnn_step", [x_sym, u_sym], list(_step_casadi(x_sym, u_sym, model)))
    for control in controls:
        expected = np.asarray(model.discrete_dynamics(jnp.asarray(state), jnp.asarray(control), 0.0, model.dt))
        actual, output = step(state, control)
        np.testing.assert_allclose(np.asarray(actual).reshape(-1), expected, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(
            np.asarray(output).reshape(-1), np.asarray(model.output(jnp.asarray(expected))), atol=1e-9
        )
        state = expected


@pytest.mark.parametrize("cap", [0, 1])
@pytest.mark.parametrize("cnn", [False, True])
def test_configured_waveform_cap_produces_physical_plan(tmp_path: Path, cap: int, *, cnn: bool) -> None:
    """Bonmin caps shared activation decisions and emits bounded balanced currents."""
    artifact = tmp_path / "predictor"
    if cnn:
        _cnn(depth=1).save(artifact)
        model = WaveformCNNModel.load(artifact)
    else:
        artifact = _build_checkpoint(tmp_path, depth=0, horizon=2)
        model = None
    controller = CasADiMPCController.from_config(
        {
            "dt": 0.01,
            "problem": {
                "class_path": "neuro.control.casadi.build_casadi_waveform_problem",
                "artifact": str(artifact),
                "horizon": 2,
                "u_max": 0.5,
                "w_y": 1.0,
                "w_u": 0.1,
                "reference": _ref(3 if cnn else 2),
                "max_active_intervals": cap,
            },
        }
    )
    measurement = np.zeros(controller.model.n_outputs)
    for step in range(controller.model.n_history):
        current, log = controller.update(step * controller.dt, np.zeros(1), measurement)
    assert log.success
    assert log.planned_active is not None
    np.testing.assert_allclose(log.planned_active, np.round(log.planned_active), atol=1e-6)
    assert log.planned_active.sum() <= cap + 1e-6
    assert np.all(np.abs(log.planned_u) <= 0.5 * log.planned_active[:, None] + 1e-6)
    np.testing.assert_allclose(log.planned_u.sum(axis=1), 0.0, atol=1e-6)
    np.testing.assert_allclose(current, log.planned_u[0])
    if cap == 0:
        np.testing.assert_allclose(log.planned_u, 0.0, atol=1e-6)
    np.testing.assert_allclose(controller._active_guess, np.r_[log.planned_active[1:], log.planned_active[-1]])  # noqa: SLF001 -- shifted warm-start contract
    assert model is None or controller.model.n_outputs == model.n_outputs


def test_failed_integer_solve_issues_no_current(tmp_path: Path) -> None:
    """A failed capped solve reports failure and applies zero Control Current."""
    artifact = _build_checkpoint(tmp_path, depth=0, horizon=2)
    problem = build_casadi_waveform_problem(artifact, horizon=2, u_max=0.5, reference=_ref(2), max_active_intervals=1)
    controller = CasADiMPCController(0.01, problem)
    measurement = np.zeros(controller.model.n_outputs)
    for step in range(controller.model.n_history):
        controller.update(step * controller.dt, np.zeros(1), measurement)

    class FailedSolver:
        """Return an unsuccessful Bonmin result after a feasible prior solve."""

        def __call__(self, **kwargs: object) -> dict[str, ca.DM]:
            """Return a finite but unusable candidate."""
            return {"x": ca.DM(np.zeros(np.asarray(kwargs["x0"]).size))}

        def stats(self) -> dict[str, object]:
            """Expose a failed solver status."""
            return {"success": False, "return_status": "infeasible"}

    controller._solver = FailedSolver()  # ty: ignore[invalid-assignment]  # noqa: SLF001 -- exercise failure contract
    current, log = controller.update(controller.model.n_history * controller.dt, np.zeros(1), measurement)
    assert not log.success
    assert log.status == "infeasible"
    assert log.planned_active is not None
    np.testing.assert_array_equal(current, np.zeros(controller.n_controls))


@pytest.mark.parametrize("cap", [0, 1])
@pytest.mark.parametrize("cnn", [False, True])
def test_configured_observable_cap_produces_physical_frame_plan(tmp_path: Path, cap: int, *, cnn: bool) -> None:
    """Bonmin caps Observable MLP and CNN steps on their native Frame grid."""
    if cnn:
        artifact = tmp_path / "observable_cnn"
        _model(n_values=1, n_controls=2).save(artifact)
        model = ObservableCNNModel.load(artifact)
        reference = HealthyReference(
            observable=ObservableEnvelope(
                power=np.full((model.n_channels, model.n_values), -0.5),
                fs=model.geometry.n_hop / model.dt,
                geometry=model.geometry,
            )
        )
    else:
        artifact, reference = _observable_fixture(tmp_path)
    controller = CasADiMPCController.from_config(
        {
            "dt": 0.04 if cnn else 0.1,
            "problem": {
                "class_path": "neuro.control.casadi.build_casadi_observable_problem",
                "artifact": str(artifact),
                "horizon": 2,
                "u_max": [0.4, 0.6],
                "w_u": 0.1,
                "w_u_l1": 0.05,
                "w_hinge": 0.5,
                "reference": reference,
                "max_active_intervals": cap,
            },
        }
    )
    measurement = np.zeros((controller.model.n_channels, controller.model.n_outputs // controller.model.n_channels))
    for step in range(controller.model.n_history):
        current, log = controller.update(step * controller.dt, np.zeros(1), measurement)
    assert log.success
    assert log.planned_active is not None
    assert log.predicted_y.shape == (3, controller.model.n_outputs)
    np.testing.assert_allclose(log.planned_active, np.round(log.planned_active), atol=1e-6)
    assert log.planned_active.sum() <= cap + 1e-6
    assert np.all(np.abs(log.planned_u) <= np.array([0.4, 0.6]) * log.planned_active[:, None] + 1e-6)
    np.testing.assert_allclose(log.planned_u.sum(axis=1), 0.0, atol=1e-6)
    np.testing.assert_allclose(current, log.planned_u[0])
    np.testing.assert_allclose(log.cost, log.cost_spectral + log.cost_quadratic_effort + log.cost_sparse_effort)
    np.testing.assert_allclose(controller._active_guess, np.r_[log.planned_active[1:], log.planned_active[-1]])  # noqa: SLF001 -- shifted warm-start contract
    if cap == 0:
        np.testing.assert_allclose(log.planned_u, 0.0, atol=1e-6)

    class FailedSolver:
        """Expose a failed Bonmin solve after the successful Observable plan."""

        def __call__(self, **kwargs: object) -> dict[str, ca.DM]:
            """Return a candidate that must not be applied."""
            return {"x": ca.DM(np.zeros(np.asarray(kwargs["x0"]).size))}

        def stats(self) -> dict[str, object]:
            """Report the failed solver status."""
            return {"success": False, "return_status": "infeasible"}

    controller._solver = FailedSolver()  # ty: ignore[invalid-assignment]  # noqa: SLF001 -- exercise failure contract
    failed_current, failed_log = controller.update(controller.model.n_history * controller.dt, np.zeros(1), measurement)
    assert not failed_log.success
    assert failed_log.status == "infeasible"
    assert failed_log.planned_active is not None
    np.testing.assert_array_equal(failed_current, np.zeros(controller.n_controls))
