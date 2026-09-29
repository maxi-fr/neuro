from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import casadi as ca
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from trajopt.transcription.ipopt import Ipopt

from neuro.config import StftGeometry
from neuro.control.casadi import (
    CasADiMPCController,
    CasADiMPCLog,
    CasADiWaveformProblem,
    _compute_casadi_observable_frames,
    _output_casadi,
    _step_casadi,
    build_casadi_waveform_problem,
    decompose_casadi_cost,
)
from neuro.control.costs import jax_compute_observable_frames
from neuro.control.mpc import TrajOptMPCController, build_waveform_problem, decompose_cost
from neuro.predictor.inference import WaveformMLPModel
from neuro.predictor.module import AutoregressiveMLP
from neuro.spectral import HealthyReference, ObservableEnvelope
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
    dt: float = 0.01,
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
        dt=dt,
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


def _write_reference(tmp_path: Path, geom: StftGeometry, n_channels: int = 2, fs: float = 100.0) -> Path:
    n_values = geom.n_values(fs)
    ref_path = tmp_path / "healthy_ref.npz"
    np.savez_compressed(
        ref_path,
        eeg_mean=np.zeros(n_channels),
        Pref_frames=np.full((n_channels, n_values), -2.0),
        fs=fs,
        n_segment=geom.n_segment,
        n_hop=geom.n_hop,
        band_hz=np.asarray(geom.band_hz if geom.band_hz is not None else [-1.0, -1.0]),
        n_bin_pool=geom.n_bin_pool,
        kernel=geom.kernel,
        kernel_width=geom.kernel_width,
    )
    return ref_path


def test_configuration_constructs_controller_with_all_costs(tmp_path: Path) -> None:
    """Configuration constructs the continuous CasADi controller with all Cost options."""
    geom = StftGeometry(n_segment=20, n_hop=5, kernel="hann", kernel_width=2)
    fs = 100.0
    dt = 1.0 / fs
    artifact = _build_checkpoint(tmp_path, depth=1, horizon=25, dt=dt)
    ref_path = _write_reference(tmp_path, geom, n_channels=2, fs=fs)

    config = {
        "controller": {
            "class_path": "neuro.control.casadi.CasADiMPCController",
            "dt": dt,
            "problem": {
                "class_path": "neuro.control.casadi.build_casadi_waveform_problem",
                "artifact": str(artifact),
                "horizon": 25,
                "u_max": 0.5,
                "w_y": 1.2,
                "w_y_terminal": 2.5,
                "w_u": 0.1,
                "w_u_l1": 0.3,
                "w_hinge": 1.5,
                "reference": str(ref_path),
                "kirchhoff": True,
            },
        }
    }

    controller = CasADiMPCController.from_config(config["controller"])
    assert isinstance(controller, CasADiMPCController)
    assert controller.horizon == 25
    assert controller.problem.w_y == 1.2
    assert controller.problem.w_y_terminal == 2.5
    assert controller.problem.w_u == 0.1
    assert controller.problem.w_u_l1 == 0.3
    assert controller.problem.w_hinge == 1.5
    assert controller.problem.envelope is not None
    support = geom.sample_support_steps(fs)
    assert controller.model.n_history >= support

    sim_config = {
        "dynamics": {"class_path": "neuro.jansen_rit.JansenRitDynamics", "dt": dt},
        "estimator": {"class_path": "neuro.filtering.PassThroughEstimator", "dt": dt},
        "sensors": {"class_path": "simulate.sensor.GaussianSensor", "dt": dt},
        "controller": config["controller"],
    }
    validate_simulation_config(sim_config)


@pytest.mark.parametrize(
    ("w_y", "w_y_terminal", "w_u", "w_u_l1", "w_hinge"),
    [
        (1.5, 3.5, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.8, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.2, 0.0),
        (0.0, 0.0, 0.0, 0.0, 2.5),
        (1.2, 2.8, 0.4, 0.7, 1.8),
    ],
)
def test_casadi_and_trajopt_agree_on_every_cost_contribution(
    tmp_path: Path,
    w_y: float,
    w_y_terminal: float,
    w_u: float,
    w_u_l1: float,
    w_hinge: float,
) -> None:
    """CasADi and trajopt decompose_cost agree on every Cost contribution and total Cost."""
    geom = StftGeometry(n_segment=15, n_hop=5, kernel="hann", kernel_width=2)
    fs = 100.0
    dt = 1.0 / fs
    horizon = 6
    artifact = _build_checkpoint(tmp_path, depth=1, horizon=horizon, dt=dt)
    ref_path = _write_reference(tmp_path, geom, n_channels=2, fs=fs)
    ref = HealthyReference.load(ref_path)

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
    rng = np.random.default_rng(_SEED + 10)
    x0 = rng.standard_normal(model.n)
    u_seq = rng.uniform(-0.5, 0.5, (horizon, model.m))

    states = [x0]
    curr_x = x0
    for k in range(horizon):
        curr_x = np.asarray(model.discrete_dynamics(jnp.asarray(curr_x), jnp.asarray(u_seq[k]), 0.0, dt))
        states.append(curr_x)
    states_jax = jnp.asarray(np.array(states))

    decomp_trajopt = decompose_cost(prob_trajopt, states_jax, jnp.asarray(u_seq), dt=dt)
    decomp_casadi = decompose_casadi_cost(prob_casadi, x0, u_seq)

    np.testing.assert_allclose(decomp_casadi["cost_tracking"], decomp_trajopt["cost_tracking"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        decomp_casadi["cost_quadratic_effort"], decomp_trajopt["cost_quadratic_effort"], rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        decomp_casadi["cost_sparse_effort"], decomp_trajopt["cost_sparse_effort"], rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(decomp_casadi["cost_spectral"], decomp_trajopt["cost_spectral"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(decomp_casadi["cost_total"], decomp_trajopt["cost_total"], rtol=1e-10, atol=1e-10)


def test_spectral_cost_preserves_frame_geometry_and_history(tmp_path: Path) -> None:
    """Spectral Observable Frame computation in CasADi preserves Frame geometry, pooling, and history."""
    fs = 100.0
    geometries = [
        StftGeometry(n_segment=20, n_hop=5, kernel_width=1, n_bin_pool=1),
        StftGeometry(n_segment=20, n_hop=5, kernel="hann", kernel_width=3, n_bin_pool=2),
        StftGeometry(n_segment=16, n_hop=8, kernel="triangular", kernel_width=2, n_bin_pool=1),
    ]

    for geom in geometries:
        rng = np.random.default_rng(_SEED + 20)
        support = geom.sample_support_steps(fs)
        horizon = 10
        total_samples = support - 1 + horizon + 1
        y = rng.standard_normal((total_samples, 2))

        jax_frames = np.asarray(jax_compute_observable_frames(jnp.asarray(y), geom, fs=fs))
        y_sx_list = [ca.SX(y[i : i + 1, :].T) for i in range(total_samples)]
        ca_frames = _compute_casadi_observable_frames(y_sx_list, geom, fs=fs)
        assert len(ca_frames) == len(jax_frames)
        for i, f_sx in enumerate(ca_frames):
            f_val = np.asarray(ca.Function(f"f_{i}", [], [f_sx])()["o0"])
            np.testing.assert_allclose(f_val, jax_frames[i], rtol=1e-10, atol=1e-12)


def test_configured_controller_with_all_costs_returns_feasible_currents_and_decomposed_log(tmp_path: Path) -> None:
    """Configured controller with all Costs returns bounded, balanced currents and decomposed log."""
    geom = StftGeometry(n_segment=15, n_hop=5, kernel="hann", kernel_width=2)
    fs = 100.0
    dt = 1.0 / fs
    horizon = 5
    artifact = _build_checkpoint(tmp_path, depth=0, horizon=horizon, dt=dt)
    ref_path = _write_reference(tmp_path, geom, n_channels=2, fs=fs)
    ref = HealthyReference.load(ref_path)

    u_max = 0.6
    prob_casadi = build_casadi_waveform_problem(
        artifact,
        horizon=horizon,
        u_max=u_max,
        w_y=1.0,
        w_y_terminal=2.5,
        w_u=0.1,
        w_u_l1=0.2,
        w_hinge=1.5,
        reference=ref,
        kirchhoff=True,
    )
    controller = CasADiMPCController(dt=dt, problem=prob_casadi)

    rng = np.random.default_rng(_SEED + 30)
    for k in range(10):
        meas = rng.standard_normal(2)
        u, log = controller.update(k * dt, ref=np.zeros(2), x_hat=meas)
        if not log.warmup:
            assert log.success
            assert np.all(np.abs(u) <= u_max + 1e-6)
            assert np.sum(u) == pytest.approx(0.0, abs=1e-6)
            assert np.all(np.abs(log.planned_u) <= u_max + 1e-6)
            np.testing.assert_allclose(np.sum(log.planned_u, axis=1), 0.0, atol=1e-6)

            assert log.cost_tracking >= 0.0
            assert log.cost_quadratic_effort >= 0.0
            assert log.cost_sparse_effort >= 0.0
            assert log.cost_spectral >= 0.0
            expected_tot = log.cost_tracking + log.cost_quadratic_effort + log.cost_sparse_effort + log.cost_spectral
            assert log.cost == pytest.approx(expected_tot, rel=1e-5)
