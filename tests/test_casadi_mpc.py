from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import casadi as ca
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from trajopt.transcription.ipopt import Ipopt

from neuro.control.casadi import (
    CasADiMPCController,
    CasADiMPCLog,
    CasADiWaveformProblem,
    _output_casadi,
    _step_casadi,
    build_casadi_waveform_problem,
)
from neuro.control.mpc import TrajOptMPCController, build_waveform_problem
from neuro.predictor.inference import WaveformMLPModel
from neuro.predictor.module import AutoregressiveMLP
from neuro.spectral import HealthyReference
from neuro.transforms import Standardizer
from neuro.validation import validate_simulation_config

if TYPE_CHECKING:
    from pathlib import Path

    from neuro.types import Activation, FloatArray

_SEED = 7

_WAVEFORM_PARITY_CONTROLS = np.array(
    [
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [-0.5000000097735464, 0.5000000097735464],
        [-0.2197725016350544, 0.2197725016350544],
        [-0.5000000099748614, 0.5000000099748614],
        [-0.5000000099693410, 0.5000000099693410],
        [-0.5000000099527547, 0.5000000099527547],
    ]
)
_WAVEFORM_PARITY_COSTS = [
    0.0,
    0.0,
    0.0,
    1.4970273984042963,
    1.6780944598038890,
    1.8600552355429352,
    1.7721805744284902,
    2.4054735588528806,
]


def _ref(n: int = 2) -> HealthyReference:
    return HealthyReference(eeg_mean=np.zeros(n))


def _random_layers(rng: np.random.Generator, sizes: list[int]) -> tuple[tuple[FloatArray, FloatArray], ...]:
    return tuple(
        (rng.uniform(-1.0, 1.0, (out, inp)) / np.sqrt(inp), rng.uniform(-1.0, 1.0, out) / np.sqrt(inp))
        for inp, out in itertools.pairwise(sizes)
    )


def _build_checkpoint(
    tmp_path: Path,
    *,
    n_y: int = 4,
    n_u: int = 3,
    horizon: int = 3,
    n_channels: int = 2,
    n_controls: int = 2,
    depth: int = 2,
    activation: Activation = "relu",
    residual: bool = True,
) -> Path:
    rng = np.random.default_rng(_SEED)
    in_size = n_y * n_channels + n_u * n_controls
    scalers = {
        "u_mean": rng.uniform(-1.0, 1.0, n_controls),
        "u_scale": rng.uniform(0.5, 2.0, n_controls),
        "y_mean": rng.uniform(-1.0, 1.0, n_channels),
        "y_scale": rng.uniform(0.5, 2.0, n_channels),
    }
    model = AutoregressiveMLP(
        n_y=n_y,
        n_u=n_u,
        horizon=horizon,
        n_channels=n_channels,
        n_controls=n_controls,
        n_outputs=n_channels,
        hidden_size=5,
        depth=depth,
        activation=activation,
        residual=residual,
        dt=0.01,
        y_std=Standardizer(center=scalers["y_mean"], scale=scalers["y_scale"]),
        u_std=Standardizer(center=scalers["u_mean"], scale=scalers["u_scale"]),
    )
    linears = [m for m in model.layers if isinstance(m, torch.nn.Linear)]
    sizes = [in_size, *([5] * depth), n_channels]
    with torch.no_grad():
        for lin, (w, b) in zip(linears, _random_layers(rng, sizes), strict=True):
            lin.weight.copy_(torch.as_tensor(w, dtype=torch.float32))
            lin.bias.copy_(torch.as_tensor(b, dtype=torch.float32))
    path = tmp_path / f"art_{activation}_{depth}_{residual}"
    model.save(path)
    return path


def test_configuration_constructs_continuous_controller(tmp_path: Path) -> None:
    artifact = _build_checkpoint(tmp_path, depth=1)
    ref_path = tmp_path / "ref.npz"
    np.savez(ref_path, eeg_mean=np.zeros(2))

    config = {
        "controller": {
            "class_path": "neuro.control.casadi.CasADiMPCController",
            "dt": 0.01,
            "problem": {
                "class_path": "neuro.control.casadi.build_casadi_waveform_problem",
                "artifact": str(artifact),
                "horizon": 3,
                "u_max": 0.6,
                "w_y": 1.5,
                "w_u": 0.05,
                "reference": str(ref_path),
                "kirchhoff": True,
            },
        }
    }

    controller = CasADiMPCController.from_config(config["controller"])
    assert isinstance(controller, CasADiMPCController)
    assert controller.horizon == 3
    assert controller.n_controls == 2
    assert controller.problem.w_y == 1.5
    assert controller.problem.w_u == 0.05
    assert controller.problem.kirchhoff
    np.testing.assert_allclose(controller.problem.u_max, [0.6, 0.6])

    sim_config = {
        "dynamics": {"class_path": "neuro.jansen_rit.JansenRitDynamics", "dt": 0.01},
        "estimator": {"class_path": "neuro.filtering.PassThroughEstimator", "dt": 0.01},
        "sensors": {"class_path": "simulate.sensor.GaussianSensor", "dt": 0.01},
        "controller": config["controller"],
    }
    validate_simulation_config(sim_config)


def test_state_absorption_and_native_controller_step_outputs(tmp_path: Path) -> None:
    n_y = 3
    artifact = _build_checkpoint(tmp_path, n_y=n_y, horizon=4)
    problem = build_casadi_waveform_problem(
        artifact, horizon=4, u_max=0.5, w_y=1.0, w_u=0.1, reference=_ref(2), kirchhoff=True
    )
    controller = CasADiMPCController(dt=0.01, problem=problem)

    rng = np.random.default_rng(_SEED + 1)
    for step in range(n_y - 1):
        u, log = controller.update(step * 0.01, ref=np.zeros(1), x_hat=rng.standard_normal(2))
        assert log.warmup
        assert log.success
        assert log.status == "warmup"
        np.testing.assert_array_equal(u, np.zeros(2))
        assert np.isnan(log.predicted_y).all()
        assert np.isnan(log.planned_u).all()

    u, log = controller.update((n_y - 1) * 0.01, ref=np.zeros(1), x_hat=rng.standard_normal(2))
    assert not log.warmup
    assert log.success
    assert log.status in ("Solve_Succeeded", "Solved_To_Acceptable_Level")
    assert log.solve_time > 0.0
    assert log.predicted_y.shape == (5, 2)
    assert log.planned_u.shape == (4, 2)
    np.testing.assert_allclose(u, log.planned_u[0])
    assert np.all(np.abs(log.planned_u) <= 0.5 + 1e-6)
    np.testing.assert_allclose(log.planned_u.sum(axis=1), 0.0, atol=1e-6)
    assert log.cost > 0.0
    assert log.cost_tracking >= 0.0
    assert log.cost_quadratic_effort >= 0.0
    np.testing.assert_allclose(log.cost, log.cost_tracking + log.cost_quadratic_effort, rtol=1e-5)


def test_successful_solve_shifts_initial_guess(tmp_path: Path) -> None:
    artifact = _build_checkpoint(tmp_path, depth=0, n_y=2, horizon=4)
    problem = build_casadi_waveform_problem(
        artifact, horizon=4, u_max=0.5, w_y=1.0, w_u=0.1, reference=_ref(2), kirchhoff=True
    )
    controller = CasADiMPCController(dt=0.01, problem=problem)

    controller.update(0.0, ref=np.zeros(1), x_hat=np.array([0.1, -0.1]))
    _, log = controller.update(0.01, ref=np.zeros(1), x_hat=np.array([0.2, -0.2]))
    assert log.success

    expected_shifted_guess = np.vstack([log.planned_u[1:], log.planned_u[-1:]])
    np.testing.assert_allclose(controller._u_guess, expected_shifted_guess)  # noqa: SLF001 -- internal initial guess state


def test_failed_solve_exposes_status_and_issues_zero_current(tmp_path: Path) -> None:
    artifact = _build_checkpoint(tmp_path, n_y=2, horizon=3)
    problem = build_casadi_waveform_problem(
        artifact,
        horizon=3,
        u_max=0.5,
        w_y=1.0,
        w_u=0.1,
        reference=_ref(2),
        kirchhoff=True,
        solver_options={"ipopt.max_iter": 0},
    )
    controller = CasADiMPCController(dt=0.01, problem=problem)

    controller.update(0.0, ref=np.zeros(1), x_hat=np.array([0.1, -0.1]))
    u, log = controller.update(0.01, ref=np.zeros(1), x_hat=np.array([0.2, -0.2]))

    assert not log.success
    assert log.status == "Maximum_Iterations_Exceeded"
    np.testing.assert_array_equal(u, np.zeros(2))


@pytest.mark.parametrize("activation", ["relu", "tanh", "softplus"])
@pytest.mark.parametrize("residual", [False, True])
def test_casadi_and_predictor_one_step_agreement(tmp_path: Path, activation: Activation, *, residual: bool) -> None:
    artifact = _build_checkpoint(tmp_path, depth=2, activation=activation, residual=residual)
    model = WaveformMLPModel.load(artifact)

    rng = np.random.default_rng(_SEED + 2)
    state = rng.standard_normal(model.n)
    u = rng.standard_normal(model.n_controls)

    next_state_jax = np.asarray(model.discrete_dynamics(jnp.asarray(state), jnp.asarray(u), 0.0, 0.01))
    next_output_jax = np.asarray(model.output(jnp.asarray(next_state_jax)))

    x_sym = ca.SX.sym("x", model.n)
    u_sym = ca.SX.sym("u", model.n_controls)
    step_fn = ca.Function("step", [x_sym, u_sym], list(_step_casadi(x_sym, u_sym, model)))
    out_fn = ca.Function("out", [x_sym], [_output_casadi(x_sym, model)])

    res_x, res_y = step_fn(state, u)
    next_state_ca = np.asarray(res_x).reshape(-1)
    next_output_ca = np.asarray(res_y).reshape(-1)
    eval_output_ca = np.asarray(out_fn(next_state_ca)).reshape(-1)

    np.testing.assert_allclose(next_state_ca, next_state_jax, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(next_output_ca, next_output_jax, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(eval_output_ca, next_output_jax, rtol=1e-10, atol=1e-10)


def test_casadi_and_predictor_control_horizon_rollout_agreement(tmp_path: Path) -> None:
    horizon = 5
    artifact = _build_checkpoint(tmp_path, depth=2, activation="softplus", residual=True, horizon=horizon)
    model = WaveformMLPModel.load(artifact)

    rng = np.random.default_rng(_SEED + 3)
    state = rng.standard_normal(model.n)
    controls = rng.standard_normal((horizon, model.n_controls))

    states_jax = [state]
    curr_state = state
    for k in range(horizon):
        curr_state = np.asarray(model.discrete_dynamics(jnp.asarray(curr_state), jnp.asarray(controls[k]), 0.0, 0.01))
        states_jax.append(curr_state)
    outputs_jax = np.array([np.asarray(model.output(jnp.asarray(s))) for s in states_jax])

    x0_sym = ca.SX.sym("x0", model.n)
    u_sym = ca.SX.sym("u", horizon * model.n_controls)
    x_curr = x0_sym
    y_preds = [_output_casadi(x_curr, model)]
    for k in range(horizon):
        u_k = u_sym[k * model.n_controls : (k + 1) * model.n_controls]
        x_curr, y_next = _step_casadi(x_curr, u_k, model)
        y_preds.append(y_next)

    rollout_fn = ca.Function("rollout", [x0_sym, u_sym], [ca.horzcat(*y_preds)])
    outputs_ca = np.asarray(rollout_fn(state, controls.reshape(-1))).T

    np.testing.assert_allclose(outputs_ca, outputs_jax, rtol=1e-10, atol=1e-10)


def test_casadi_and_trajopt_ipopt_parity_on_continuous_problem(tmp_path: Path) -> None:
    artifact = _build_checkpoint(tmp_path, depth=0, n_y=3, horizon=3)
    ref = _ref(2)

    prob_trajopt = build_waveform_problem(
        artifact, horizon=3, u_max=0.5, w_y=1.0, w_u=0.1, kirchhoff=True, reference=ref
    )
    solver_trajopt = Ipopt(options={"print_level": 0, "hessian_approximation": "limited-memory"})
    ctrl_trajopt = TrajOptMPCController(dt=0.01, problem=prob_trajopt, solver=solver_trajopt)

    prob_casadi = build_casadi_waveform_problem(
        artifact, horizon=3, u_max=0.5, w_y=1.0, w_u=0.1, kirchhoff=True, reference=ref
    )
    ctrl_casadi = CasADiMPCController(dt=0.01, problem=prob_casadi)

    measurements = [
        np.array([0.1, -0.1]),
        np.array([-0.2, 0.2]),
        np.array([0.4, -0.3]),
        np.array([-0.5, 0.5]),
    ]

    for k, meas in enumerate(measurements):
        u_t, log_t = ctrl_trajopt.update(k * 0.01, ref=np.zeros(1), x_hat=meas)
        u_c, log_c = ctrl_casadi.update(k * 0.01, ref=np.zeros(1), x_hat=meas)

        assert log_t.warmup == log_c.warmup
        if not log_t.warmup:
            np.testing.assert_allclose(u_c, u_t, atol=1e-3)
            np.testing.assert_allclose(log_c.cost, log_t.cost, rtol=1e-3, atol=1e-3)
            np.testing.assert_allclose(log_c.planned_u.sum(axis=1), 0.0, atol=1e-6)
            assert np.all(np.abs(log_c.planned_u) <= 0.5 + 1e-6)


def test_reproduces_mpc_controller_golden_parity(tmp_path: Path) -> None:
    artifact = _build_checkpoint(tmp_path, depth=0, horizon=3, residual=False)
    prob_casadi = build_casadi_waveform_problem(
        artifact,
        horizon=3,
        u_max=0.5,
        w_y=1.0,
        w_u=0.0,
        kirchhoff=True,
        reference=_ref(2),
        solver_options={"ipopt.tol": 1e-6, "ipopt.acceptable_tol": 1e-6},
    )
    controller = CasADiMPCController(dt=0.01, problem=prob_casadi)

    rng = np.random.default_rng(_SEED + 5)
    controls = []
    costs = []
    for k in range(8):
        u, log = controller.update(k * controller.dt, ref=np.array([0.0]), x_hat=rng.standard_normal(2))
        controls.append(np.atleast_1d(np.asarray(u, dtype=np.float64)))
        costs.append(log.cost)

    np.testing.assert_allclose(controls, _WAVEFORM_PARITY_CONTROLS, atol=1e-4)
    np.testing.assert_allclose(costs, _WAVEFORM_PARITY_COSTS, atol=1e-4)
