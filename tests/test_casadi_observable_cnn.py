from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from test_observable_cnn import _model

from neuro.control.casadi import CasADiMPCController, build_casadi_observable_problem
from neuro.control.costs import ObservableHingeCost
from neuro.predictor.inference import ObservableCNNModel
from neuro.spectral import HealthyReference, ObservableEnvelope

if TYPE_CHECKING:
    from pathlib import Path

    from neuro.types import Activation


@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("activation", ["relu", "tanh", "softplus"])
@pytest.mark.parametrize("n_values", [1, 4])
def test_cnn_fixed_sequence_frame_and_cost_parity(
    tmp_path: Path, activation: Activation, *, residual: bool, n_values: int
) -> None:
    """CasADi matches each structured CNN Frame and every active Cost contribution."""
    module = _model(n_values=n_values, residual=residual, activation=activation)
    artifact = tmp_path / "observable_cnn"
    module.save(artifact)
    model = ObservableCNNModel.load(artifact)
    reference = HealthyReference(
        observable=ObservableEnvelope(
            power=np.full((model.n_channels, model.n_values), -0.5),
            fs=model.geometry.n_hop / model.dt,
            geometry=model.geometry,
        )
    )
    problem = build_casadi_observable_problem(
        artifact, horizon=3, u_max=0.6, w_u=0.3, w_u_l1=0.2, w_hinge=1.4, reference=reference
    )
    controller = CasADiMPCController(model.dt, problem)
    rng = np.random.default_rng(52)
    state = rng.normal(size=model.n)
    controls = rng.uniform(-0.3, 0.3, size=(3, model.n_controls))
    states = [state]
    for control in controls:
        states.append(np.asarray(model.discrete_dynamics(jnp.asarray(states[-1]), jnp.asarray(control), 0.0, model.dt)))
    expected = np.stack([np.asarray(model.output(jnp.asarray(x))) for x in states])
    actual = np.asarray(controller._predicted_y_fn(state, controls.reshape(-1))).T  # noqa: SLF001 -- parity seam
    assert actual.reshape(4, model.n_channels, model.n_values).shape == (4, model.n_channels, model.n_values)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)

    total, tracking, quadratic, sparse, hinge = (
        float(value)
        for value in controller._costs_fn(state, controls.reshape(-1))  # noqa: SLF001 -- parity seam
    )
    assert reference.observable is not None
    hinge_cost = ObservableHingeCost(reference.observable, w_hinge=1.4, horizon=3)
    expected_hinge = sum(float(hinge_cost.evaluate(jnp.asarray(frame))) for frame in expected[1:])
    expected_sparse = float(np.sum((0.2 / 3) * (np.hypot(controls, 1e-3) - 1e-3)))
    expected_quadratic = 0.3 / 3 * np.sum(controls**2)
    np.testing.assert_allclose(
        [tracking, quadratic, sparse, hinge, total],
        [
            0.0,
            expected_quadratic,
            expected_sparse,
            expected_hinge,
            expected_quadratic + expected_sparse + expected_hinge,
        ],
        rtol=1e-8,
        atol=1e-8,
    )


def test_cnn_configured_controller_emits_feasible_plan(tmp_path: Path) -> None:
    """The configured CNN controller advances on Frame steps and emits a feasible physical plan."""
    module = _model(n_controls=2)
    artifact = tmp_path / "observable_cnn"
    module.save(artifact)
    model = ObservableCNNModel.load(artifact)
    reference = HealthyReference(
        observable=ObservableEnvelope(
            power=np.full((model.n_channels, model.n_values), -0.5),
            fs=model.geometry.n_hop / model.dt,
            geometry=model.geometry,
        )
    )
    controller = CasADiMPCController.from_config(
        {
            "dt": model.dt,
            "problem": {
                "class_path": "neuro.control.casadi.build_casadi_observable_problem",
                "artifact": str(artifact),
                "horizon": 3,
                "u_max": 0.6,
                "w_u": 0.1,
                "w_u_l1": 0.05,
                "w_hinge": 0.5,
                "reference": reference,
                "kirchhoff": True,
            },
        }
    )
    measurement = np.zeros((model.n_channels, model.n_values))
    for step in range(model.n_history - 1):
        _, log = controller.update(step * model.dt, np.zeros(1), measurement)
        assert log.warmup
    current, log = controller.update((model.n_history - 1) * model.dt, np.zeros(1), measurement)
    assert log.success
    assert not log.warmup
    assert log.predicted_y.shape == (4, model.n_outputs)
    assert log.planned_u.shape == (3, model.n_controls)
    np.testing.assert_allclose(current, log.planned_u[0])
    np.testing.assert_allclose(log.planned_u.sum(axis=1), 0.0, atol=1e-6)
    assert np.all(np.abs(log.planned_u) <= 0.6 + 1e-6)
    np.testing.assert_allclose(log.cost, log.cost_spectral + log.cost_quadratic_effort + log.cost_sparse_effort)
