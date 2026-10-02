from __future__ import annotations

from typing import TYPE_CHECKING

import casadi as ca
import jax.numpy as jnp
import numpy as np
import pytest
from test_casadi_mpc import _write_reference
from test_waveform_cnn import _cnn

from neuro.config import StftGeometry
from neuro.control.casadi import (
    CasADiMPCController,
    _output_casadi,
    _step_casadi,
    build_casadi_waveform_problem,
    decompose_casadi_cost,
)
from neuro.control.mpc import build_waveform_problem, decompose_cost
from neuro.predictor.inference import WaveformCNNModel
from neuro.spectral import HealthyReference, ObservableEnvelope

if TYPE_CHECKING:
    from pathlib import Path

    from neuro.types import Activation


def _reference(geom: StftGeometry, n_channels: int, fs: float) -> HealthyReference:
    """Construct a healthy reference with an Observable envelope and EEG mean."""
    return HealthyReference(
        observable=ObservableEnvelope(
            power=np.full((n_channels, geom.n_values(fs)), -0.5),
            fs=fs,
            geometry=geom,
        ),
        eeg_mean=np.zeros(n_channels),
    )


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("activation", ["relu", "tanh", "softplus"])
@pytest.mark.parametrize("residual", [False, True])
def test_waveform_cnn_one_step_and_horizon_rollout_parity(
    tmp_path: Path, activation: Activation, *, residual: bool, depth: int
) -> None:
    """CasADi agrees with the JAX predictor on one-step and Control Horizon outputs."""
    module = _cnn(depth=depth, activation=activation, residual=residual)
    artifact = tmp_path / f"cnn_{depth}_{activation}_{residual}"
    module.save(artifact)
    model = WaveformCNNModel.load(artifact)

    horizon = 4
    rng = np.random.default_rng(101)
    state = rng.standard_normal(model.n)
    controls = rng.uniform(-0.4, 0.4, size=(horizon, model.n_controls))

    # One-step parity check
    x_sym = ca.SX.sym("x", model.n)
    u_sym = ca.SX.sym("u", model.n_controls)
    step_fn = ca.Function("step", [x_sym, u_sym], list(_step_casadi(x_sym, u_sym, model)))
    out_fn = ca.Function("out", [x_sym], [_output_casadi(x_sym, model)])

    expected_next_x = np.asarray(model.discrete_dynamics(jnp.asarray(state), jnp.asarray(controls[0]), 0.0, model.dt))
    expected_next_y = np.asarray(model.output(jnp.asarray(expected_next_x)))
    actual_next_x, actual_next_y = step_fn(state, controls[0])

    np.testing.assert_allclose(np.asarray(actual_next_x).reshape(-1), expected_next_x, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.asarray(actual_next_y).reshape(-1), expected_next_y, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.asarray(out_fn(actual_next_x)).reshape(-1), expected_next_y, rtol=1e-9, atol=1e-9)

    # Control Horizon rollout parity check
    states_jax = [state]
    curr_state = state
    for k in range(horizon):
        curr_state = np.asarray(
            model.discrete_dynamics(jnp.asarray(curr_state), jnp.asarray(controls[k]), 0.0, model.dt)
        )
        states_jax.append(curr_state)
    expected_rollout_y = np.array([np.asarray(model.output(jnp.asarray(s))) for s in states_jax])

    x0_sym = ca.SX.sym("x0", model.n)
    u_seq_sym = ca.SX.sym("u_seq", horizon * model.n_controls)
    x_curr = x0_sym
    y_preds = [_output_casadi(x_curr, model)]
    for k in range(horizon):
        u_k = u_seq_sym[k * model.n_controls : (k + 1) * model.n_controls]
        x_curr, y_next = _step_casadi(x_curr, u_k, model)
        y_preds.append(y_next)

    rollout_fn = ca.Function("rollout", [x0_sym, u_seq_sym], [ca.horzcat(*y_preds)])
    actual_rollout_y = np.asarray(rollout_fn(state, controls.reshape(-1))).T

    np.testing.assert_allclose(actual_rollout_y, expected_rollout_y, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize(
    ("w_y", "w_y_terminal", "w_u", "w_u_l1", "w_hinge"),
    [
        (1.0, 2.5, 0.3, 0.2, 1.5),
        (1.0, 1.0, 0.0, 0.0, 0.0),
        (0.5, 2.0, 0.1, 0.0, 0.0),
        (0.0, 0.0, 0.2, 0.1, 2.0),
    ],
)
def test_waveform_cnn_cost_parity_with_trajopt(
    tmp_path: Path,
    w_y: float,
    w_y_terminal: float,
    w_u: float,
    w_u_l1: float,
    w_hinge: float,
) -> None:
    """Every enabled Cost contribution matches the reference trajopt implementation."""
    geom = StftGeometry(n_segment=15, n_hop=5, kernel="hann", kernel_width=2)
    fs = 100.0
    dt = 1.0 / fs
    horizon = 5

    module = _cnn(depth=1)
    artifact = tmp_path / "cnn_cost"
    module.save(artifact)
    ref = _reference(geom, n_channels=module.n_channels, fs=fs)

    prob_trajopt = build_waveform_problem(
        artifact,
        horizon=horizon,
        u_max=0.8,
        w_y=w_y,
        w_y_terminal=w_y_terminal,
        w_u=w_u,
        w_u_l1=w_u_l1,
        w_hinge=w_hinge,
        reference=ref,
        kirchhoff=True,
    )
    prob_casadi = build_casadi_waveform_problem(
        artifact,
        horizon=horizon,
        u_max=0.8,
        w_y=w_y,
        w_y_terminal=w_y_terminal,
        w_u=w_u,
        w_u_l1=w_u_l1,
        w_hinge=w_hinge,
        reference=ref,
        kirchhoff=True,
    )

    model = prob_trajopt.model
    rng = np.random.default_rng(202)
    x0 = rng.standard_normal(model.n)
    u_seq = rng.uniform(-0.4, 0.4, (horizon, model.m))

    states = [x0]
    curr_x = x0
    for k in range(horizon):
        curr_x = np.asarray(model.discrete_dynamics(jnp.asarray(curr_x), jnp.asarray(u_seq[k]), 0.0, dt))
        states.append(curr_x)
    states_jax = jnp.asarray(np.array(states))

    decomp_trajopt = decompose_cost(prob_trajopt, states_jax, jnp.asarray(u_seq), dt=dt)
    decomp_casadi = decompose_casadi_cost(prob_casadi, x0, u_seq)

    np.testing.assert_allclose(
        decomp_casadi["cost_tracking"], float(decomp_trajopt["cost_tracking"]), rtol=1e-8, atol=1e-8
    )
    np.testing.assert_allclose(
        decomp_casadi["cost_quadratic_effort"], float(decomp_trajopt["cost_quadratic_effort"]), rtol=1e-8, atol=1e-8
    )
    np.testing.assert_allclose(
        decomp_casadi["cost_sparse_effort"], float(decomp_trajopt["cost_sparse_effort"]), rtol=1e-8, atol=1e-8
    )
    np.testing.assert_allclose(
        decomp_casadi["cost_spectral"], float(decomp_trajopt["cost_spectral"]), rtol=1e-8, atol=1e-8
    )
    np.testing.assert_allclose(decomp_casadi["cost_total"], float(decomp_trajopt["cost_total"]), rtol=1e-8, atol=1e-8)


def test_waveform_cnn_configured_controller_emits_feasible_plan(tmp_path: Path) -> None:
    """A configured waveform CNN controller produces bounded, balanced plans and shifts initial guess."""
    geom = StftGeometry(n_segment=15, n_hop=5, kernel="hann", kernel_width=2)
    fs = 100.0
    dt = 1.0 / fs
    horizon = 4
    u_max = 0.6

    module = _cnn(depth=1)
    artifact = tmp_path / "cnn_controller"
    module.save(artifact)
    ref_path = _write_reference(tmp_path, geom, n_channels=module.n_channels, fs=fs)

    config = {
        "dt": dt,
        "problem": {
            "class_path": "neuro.control.casadi.build_casadi_waveform_problem",
            "artifact": str(artifact),
            "horizon": horizon,
            "u_max": u_max,
            "w_y": 1.0,
            "w_y_terminal": 2.0,
            "w_u": 0.1,
            "w_u_l1": 0.05,
            "w_hinge": 0.5,
            "reference": str(ref_path),
            "kirchhoff": True,
        },
    }
    controller = CasADiMPCController.from_config(config)

    rng = np.random.default_rng(303)
    n_history = controller.model.n_history

    # Warmup steps emit zero current and warmup flags
    for step in range(n_history - 1):
        meas = rng.standard_normal(controller.n_channels)
        u, log = controller.update(step * dt, ref=np.zeros(controller.n_channels), x_hat=meas)
        assert log.warmup
        assert log.success
        np.testing.assert_array_equal(u, np.zeros(controller.n_controls))
        assert np.all(np.isnan(log.planned_u))
        assert np.all(np.isnan(log.predicted_y))

    # First ready step solves and emits feasible plan
    meas = rng.standard_normal(controller.n_channels)
    u_first, log_first = controller.update((n_history - 1) * dt, ref=np.zeros(controller.n_channels), x_hat=meas)
    assert not log_first.warmup
    assert log_first.success
    assert log_first.predicted_y.shape == (horizon + 1, controller.model.n_outputs)
    assert log_first.planned_u.shape == (horizon, controller.n_controls)
    np.testing.assert_allclose(u_first, log_first.planned_u[0])
    np.testing.assert_allclose(log_first.planned_u.sum(axis=1), 0.0, atol=1e-6)
    assert np.all(np.abs(log_first.planned_u) <= u_max + 1e-6)

    expected_cost = (
        log_first.cost_tracking
        + log_first.cost_quadratic_effort
        + log_first.cost_sparse_effort
        + log_first.cost_spectral
    )
    np.testing.assert_allclose(log_first.cost, expected_cost, rtol=1e-5)

    # Verify successful solve shifted initial guess
    expected_shifted_first = np.vstack([log_first.planned_u[1:], log_first.planned_u[-1:]])
    np.testing.assert_allclose(controller._u_guess, expected_shifted_first)  # noqa: SLF001 -- initial guess shift check

    # Subsequent step verifies second solve succeeds and shifts its plan
    meas = rng.standard_normal(controller.n_channels)
    _, log_second = controller.update(n_history * dt, ref=np.zeros(controller.n_channels), x_hat=meas)
    assert log_second.success
    expected_shifted_second = np.vstack([log_second.planned_u[1:], log_second.planned_u[-1:]])
    np.testing.assert_allclose(controller._u_guess, expected_shifted_second)  # noqa: SLF001 -- second solve shift check


def test_waveform_cnn_failed_solve_issues_zero_current(tmp_path: Path) -> None:
    """A failed waveform CNN solve reports status, flags failure, and issues zero current."""
    module = _cnn(depth=1)
    artifact = tmp_path / "cnn_fail"
    module.save(artifact)
    model = WaveformCNNModel.load(artifact)

    problem = build_casadi_waveform_problem(
        model,
        horizon=3,
        u_max=0.5,
        w_y=1.0,
        w_u=0.1,
        reference=HealthyReference(eeg_mean=np.zeros(model.n_channels)),
        kirchhoff=True,
        solver_options={"ipopt.max_iter": 0},
    )
    controller = CasADiMPCController(dt=model.dt, problem=problem)

    rng = np.random.default_rng(404)
    for step in range(model.n_history - 1):
        controller.update(step * model.dt, ref=np.zeros(model.n_channels), x_hat=rng.standard_normal(model.n_channels))

    u, log = controller.update(
        (model.n_history - 1) * model.dt,
        ref=np.zeros(model.n_channels),
        x_hat=rng.standard_normal(model.n_channels),
    )
    assert not log.warmup
    assert not log.success
    assert log.status == "Maximum_Iterations_Exceeded"
    np.testing.assert_array_equal(u, np.zeros(controller.n_controls))
