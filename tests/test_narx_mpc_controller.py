from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any, Literal

import casadi as ca
import numpy as np
import pytest
from simulate.simulation import Simulation

from neuro.control.narx_mpc import NarxMPCController, NarxMPCNlp
from neuro.control.nonlinear_mpc import MPCController, MPCControllerLog
from neuro.control.solvers import (
    IpoptMPCSolver,
    MPCSolveResult,
    SqpFallbackMPCSolver,
    SqpMPCSolver,
)
from neuro.esn import ESNArtifact, generate_reservoir
from neuro.esn_predictor_casadi import ESNSymbolicModel
from neuro.nn_predictor_casadi import NNSymbolicModel
from neuro.predictor.artifact import MLPArtifact
from neuro.spectral import PsdEnvelope, compute_periodograms, hinge_penalty
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from pathlib import Path

    from neuro.types import FloatArray

_SEED = 7


def _random_layers(rng: np.random.Generator, sizes: list[int]) -> tuple[tuple[FloatArray, FloatArray], ...]:
    """Random ``(weight (out, in), bias (out,))`` pairs."""
    return tuple(
        (rng.uniform(-1.0, 1.0, (out, inp)) / np.sqrt(inp), rng.uniform(-1.0, 1.0, out) / np.sqrt(inp))
        for inp, out in itertools.pairwise(sizes)
    )


def _build_artifact(
    tmp_path: Path,
    *,
    n_y: int = 4,
    n_u: int = 3,
    horizon: int = 3,
    n_channels: int = 2,
    n_controls: int = 2,
    is_linear: bool = False,
) -> Path:
    """Save a synthetic MLP artifact and return its stem path."""
    rng = np.random.default_rng(_SEED)
    in_size = n_y * n_channels + n_u * n_controls
    sizes = [in_size, n_channels] if is_linear else [in_size, 5, 5, n_channels]
    scalers = {
        "u_mean": rng.uniform(-1.0, 1.0, n_controls),
        "u_scale": rng.uniform(0.5, 2.0, n_controls),
        "y_mean": rng.uniform(-1.0, 1.0, n_channels),
        "y_scale": rng.uniform(0.5, 2.0, n_channels),
    }
    artifact = tmp_path / f"art_{'lin' if is_linear else 'mlp'}_{horizon}_{n_channels}"
    MLPArtifact(
        layers=_random_layers(rng, sizes),
        activation="relu",
        n_y=n_y,
        n_u=n_u,
        horizon=horizon,
        n_channels=n_channels,
        n_controls=n_controls,
        dt=0.01,
        downsample=1,
        y_std=Standardizer(center=scalers["y_mean"], scale=scalers["y_scale"]),
        u_std=Standardizer(center=scalers["u_mean"], scale=scalers["u_scale"]),
    ).save(artifact)
    return artifact


def _build_tiny_esn_artifact(
    tmp_path: Path,
    *,
    reservoir_size: int = 30,
    priming_steps: int = 5,
    horizon: int = 3,
    n_channels: int = 2,
    n_controls: int = 2,
) -> Path:
    """Save a synthetic ESN artifact for testing."""
    rng = np.random.default_rng(_SEED)
    in_dim = n_channels + n_controls + 1
    w_res, w_in = generate_reservoir(
        reservoir_size=reservoir_size,
        spectral_radius=0.9,
        density=0.2,
        input_scaling=0.1,
        in_dim=in_dim,
        seed=_SEED,
    )
    w_out = rng.uniform(-0.1, 0.1, size=(n_channels, reservoir_size + 1))
    y_std = Standardizer(center=rng.uniform(-1.0, 1.0, n_channels), scale=rng.uniform(0.5, 2.0, n_channels))
    u_std = Standardizer(center=rng.uniform(-1.0, 1.0, n_controls), scale=rng.uniform(0.5, 2.0, n_controls))
    artifact = tmp_path / "esn_art"
    ESNArtifact(
        w_in=w_in,
        w_out=w_out,
        w_res=w_res,
        dt=0.01,
        downsample=1,
        horizon=horizon,
        reservoir_size=reservoir_size,
        leak_rate=0.1,
        spectral_radius=0.9,
        priming_steps=priming_steps,
        input_scaling=0.1,
        density=0.2,
        noise_sigma=0.0,
        ridge_lambda=1e-3,
        seed=_SEED,
        y_std=y_std,
        u_std=u_std,
    ).save(artifact)
    return artifact


def _drive(
    controller: NarxMPCController | MPCController, n_steps: int, n_channels: int
) -> list[tuple[FloatArray, MPCControllerLog]]:
    """Feed ``n_steps`` random EEG measurements through ``update``."""
    rng = np.random.default_rng(_SEED + 1)
    out = []
    for k in range(n_steps):
        u, log = controller.update(k * controller.dt, ref=np.array([0.0]), x_hat=rng.standard_normal(n_channels))
        out.append((np.atleast_1d(np.asarray(u, dtype=np.float64)), log))
    return out


def test_warmup_emits_zero_until_window_filled(tmp_path: Path) -> None:
    """While the EEG window is zero-padded, the MPC emits zeros."""
    n_y = 4
    model = NNSymbolicModel.from_artifact(_build_artifact(tmp_path, n_y=n_y))
    controller = NarxMPCController(dt=0.01, model=model, u_max=0.5, horizon=3)

    results = _drive(controller, n_steps=n_y, n_channels=model.n_channels)
    for u, log in results[: n_y - 1]:
        assert log.warmup
        np.testing.assert_array_equal(u, np.zeros(model.n_controls))

    assert not results[-1][1].warmup


def test_update_respects_bounds(tmp_path: Path) -> None:
    """Past warm-up, update returns a finite bound-respecting control."""
    u_max = 0.5
    model = NNSymbolicModel.from_artifact(_build_artifact(tmp_path, n_y=4))
    controller = NarxMPCController(dt=0.01, model=model, u_max=u_max, horizon=3, w_y=1.0, w_u=0.0)

    u, _ = _drive(controller, n_steps=6, n_channels=model.n_channels)[-1]
    assert u.shape == (model.n_controls,)
    assert np.isfinite(u).all()
    assert np.all(np.abs(u) <= u_max + 1e-6)


def test_pure_effort_cost_yields_zero_control(tmp_path: Path) -> None:
    """With w_y=0 the cost minimizer is u=0."""
    model = NNSymbolicModel.from_artifact(_build_artifact(tmp_path, n_y=4))
    controller = NarxMPCController(dt=0.01, model=model, u_max=1.0, horizon=3, w_y=0.0, w_u=1.0)

    u, log = _drive(controller, n_steps=6, n_channels=model.n_channels)[-1]
    assert log.success
    np.testing.assert_allclose(u, np.zeros(model.n_controls), atol=1e-4)


def test_control_obeys_kirchhoff_current_law(tmp_path: Path) -> None:
    """Emitted controls sum to zero across electrodes."""
    model = NNSymbolicModel.from_artifact(_build_artifact(tmp_path, n_y=4))
    controller = NarxMPCController(dt=0.01, model=model, u_max=5.0, horizon=3, w_y=1.0, w_u=0.0)

    u, log = _drive(controller, n_steps=6, n_channels=model.n_channels)[-1]
    assert not log.warmup
    assert np.linalg.norm(u, ord=1) > 1e-3
    assert abs(float(np.sum(u))) < 1e-6


def test_l1_penalty_drives_control_toward_zero(tmp_path: Path) -> None:
    """Large L1 control-effort penalty drives applied controls toward zero."""
    model = NNSymbolicModel.from_artifact(_build_artifact(tmp_path, n_y=4))
    base: dict[str, Any] = {"dt": 0.01, "model": model, "u_max": 5.0, "horizon": 3, "w_y": 1.0, "w_u": 0.0}

    u_l2 = _drive(NarxMPCController(w_u_l1=0.0, **base), n_steps=6, n_channels=model.n_channels)[-1][0]
    u_l1 = _drive(NarxMPCController(w_u_l1=1000.0, **base), n_steps=6, n_channels=model.n_channels)[-1][0]

    assert np.linalg.norm(u_l2, ord=1) > 1e-3
    np.testing.assert_allclose(u_l1, np.zeros(model.n_controls), atol=1e-4)


def test_per_electrode_bounds_rejected_when_mismatched(tmp_path: Path) -> None:
    """Mismatched u_max length raises ValueError."""
    model = NNSymbolicModel.from_artifact(_build_artifact(tmp_path, n_controls=2))
    with pytest.raises(ValueError, match="u_max has 3 entries but n_controls is 2"):
        NarxMPCController(dt=0.01, model=model, u_max=[1.0, 2.0, 3.0], horizon=3)


def test_from_config_loads_artifact_and_defaults_horizon(tmp_path: Path) -> None:
    """from_config loads the artifact and defaults horizon."""
    artifact = _build_artifact(tmp_path, horizon=5)
    controller = NarxMPCController.from_config({"dt": 0.01, "artifact": str(artifact), "u_max": 3.0, "w_u_l1": 0.25})
    assert controller.dt == 0.01
    assert controller.horizon == 5
    assert controller.n_controls == 2
    assert controller.w_u_l1 == 0.25


def test_lifted_variable_count_is_output_sized(tmp_path: Path) -> None:
    """The decision vector has H*(n_controls + n_channels) variables without L1."""
    horizon, n_channels, n_controls = 5, 2, 2
    model = NNSymbolicModel.from_artifact(
        _build_artifact(tmp_path, n_y=4, n_u=3, n_channels=n_channels, n_controls=n_controls)
    )
    controller = NarxMPCController(dt=0.01, model=model, u_max=0.5, horizon=horizon)
    assert controller._mpc_nlp.lbx.size == horizon * (n_controls + n_channels)  # noqa: SLF001

    controller_l1 = NarxMPCController(dt=0.01, model=model, u_max=0.5, horizon=horizon, w_u_l1=1.0)
    assert controller_l1._mpc_nlp.lbx.size == horizon * (2 * n_controls + n_channels)  # noqa: SLF001


def test_matches_single_shooting_mpc_on_linear_model(tmp_path: Path) -> None:
    """On a linear model, NarxMPCController matches single shooting MPCController."""
    artifact = _build_artifact(tmp_path, is_linear=True, n_y=4, n_u=3, horizon=4, n_channels=2, n_controls=2)
    model = NNSymbolicModel.from_artifact(artifact)
    narx = NarxMPCController(dt=0.01, model=model, u_max=0.3, horizon=4, w_y=1.0, w_u=0.05, solver="ipopt")
    single = MPCController(
        dt=0.01,
        model=model,
        u_max=0.3,
        horizon=4,
        shooting_depth=4,
        w_y=1.0,
        w_u=0.05,
        solver="ipopt",
    )

    us_narx = [u for u, _ in _drive(narx, n_steps=8, n_channels=2)]
    us_single = [u for u, _ in _drive(single, n_steps=8, n_channels=2)]
    np.testing.assert_allclose(np.array(us_narx), np.array(us_single), rtol=1e-5, atol=1e-5)


def test_terminal_eeg_cost_weighting(tmp_path: Path) -> None:
    """Terminal cost weight w_y_terminal is applied to the final step."""
    model = NNSymbolicModel.from_artifact(_build_artifact(tmp_path, n_y=4))
    nlp_uniform = NarxMPCNlp.build(
        model,
        horizon=4,
        n_controls=model.n_controls,
        u_max=np.array([1.0, 1.0]),
        w_y=1.0,
        w_y_terminal=None,
    )
    nlp_terminal = NarxMPCNlp.build(
        model,
        horizon=4,
        n_controls=model.n_controls,
        u_max=np.array([1.0, 1.0]),
        w_y=1.0,
        w_y_terminal=10.0,
    )
    assert nlp_uniform.nlp["f"] is not nlp_terminal.nlp["f"]


def test_closed_loop_simulation_runs(tmp_path: Path) -> None:
    """NarxMPCController runs inside a closed-loop Simulation orchestrator."""
    n_channels, u_max = 3, 3.0
    artifact = _build_artifact(tmp_path, n_y=4, n_u=3, horizon=3, n_channels=n_channels, n_controls=2)

    config = {
        "t_end": 0.05,
        "dynamics": {
            "class_path": "neuro.jansen_rit.JansenRitDynamics",
            "dt": 1e-4,
            "seed": 69,
            "connectome": {"speed": 50.0, "K": 0.5357},
            "stimulation": {"model": "analytical", "electrodes": ["CP5", "T7"], "spread": 20.0},
            "params": {"A": 3.25},
        },
        "reference": {"class_path": "simulate.reference.StepReference", "dt": 1e-4, "step_value": 0.0},
        "sensors": {
            "class_path": "simulate.sensor.GaussianSensor",
            "dt": 1e-4,
            "std_dev": 0.0,
            "measurement": {
                "class_path": "neuro.eeg.EEGMeasurement",
                "selected_channels": [0, 1, 2],
            },
        },
        "estimator": {"class_path": "simulate.estimator.IdentityEstimator", "dt": 1e-4},
        "controller": {
            "class_path": "neuro.control.narx_mpc.NarxMPCController",
            "dt": 0.01,
            "artifact": str(artifact),
            "horizon": 3,
            "u_max": u_max,
            "w_y": 1.0,
            "w_u": 0.01,
        },
    }

    sim = Simulation.from_config(config)
    sim.run()

    assert sim.logger is not None
    us = sim.logger.signal("controller", "u")
    assert us.shape[1] == 2
    assert np.all(np.abs(us) <= u_max + 1e-6)


def test_update_returns_shared_log(tmp_path: Path) -> None:
    """update returns MPCControllerLog with valid diagnostics."""
    model = NNSymbolicModel.from_artifact(_build_artifact(tmp_path, n_y=4))
    controller = NarxMPCController(dt=0.01, model=model, u_max=1.0, horizon=3)
    _, log = _drive(controller, n_steps=6, n_channels=model.n_channels)[-1]
    assert isinstance(log, MPCControllerLog)
    assert not log.warmup
    assert isinstance(log.n_iter, int)
    assert isinstance(log.capped, bool)
    assert isinstance(log.fallback, bool)


def test_non_mlp_model_rejected(tmp_path: Path) -> None:
    """Non-MLP models and artifacts are rejected by NarxMPCController."""
    art_path = _build_tiny_esn_artifact(tmp_path, priming_steps=3)
    esn_model = ESNSymbolicModel.from_artifact(art_path)

    with pytest.raises(TypeError, match="requires an NNSymbolicModel"):
        NarxMPCController(dt=0.01, model=esn_model, u_max=1.0, horizon=3)  # ty: ignore[invalid-argument-type]

    with pytest.raises(TypeError, match="requires an MLPArtifact"):
        NarxMPCController.from_config(
            {
                "dt": 0.01,
                "artifact": str(art_path),
                "u_max": 1.0,
                "horizon": 3,
            }
        )


@pytest.mark.parametrize(
    ("horizon", "length", "hop", "n_channels"),
    [(6, 4, 2, 2), (75, 50, 25, 8)],
    ids=["toy", "production-geometry"],
)
def test_spectral_cost_matches_numpy_periodogram(
    tmp_path: Path, horizon: int, length: int, hop: int, n_channels: int
) -> None:
    """NarxMPCNlp spectral cost matches numpy periodogram reference."""
    n_controls, fs = 2, 50.0
    length // 2 + 1

    artifact = _build_artifact(tmp_path, n_y=2, n_u=2, horizon=horizon, n_channels=n_channels, n_controls=n_controls)
    model = NNSymbolicModel.from_artifact(artifact)

    rng = np.random.default_rng(_SEED + 10)
    x0 = rng.standard_normal(model.state_shape[0])
    u_fixed = np.zeros((horizon, n_controls))

    x = x0
    y_steps = []
    for step in range(horizon):
        x = np.asarray(model.f_step(x, u_fixed[step])).reshape(-1)
        y_steps.append(np.asarray(model.f_out(x)).reshape(-1))
    y_traj = np.array(y_steps)

    windows = compute_periodograms(y_traj, fs=fs, window=length, hop=hop)
    envelope = PsdEnvelope(power=np.median(windows, axis=0), fs=fs, window=length, hop=hop)

    expected_cost = hinge_penalty(windows[..., 1:], envelope.power[:, 1:])

    narx_nlp = NarxMPCNlp.build(
        model,
        horizon=horizon,
        n_controls=n_controls,
        u_max=np.array([5.0, 5.0]),
        w_y=0.0,
        w_y_terminal=None,
        w_u=0.0,
        w_u_l1=0.0,
        w_psd=1.0,
        psd_envelope=envelope,
    )
    solver = IpoptMPCSolver.build(narx_nlp, max_iter=0, ipopt_options={"max_iter": 0})

    # Exact initial guess for y matching the zero-control rollout
    y_guess = list(y_traj)
    w0 = np.concatenate([u_fixed.reshape(-1), *[model.artifact.encode(yg) for yg in y_guess]])

    sol = solver.function(x0=w0, lbx=narx_nlp.lbx, ubx=narx_nlp.ubx, lbg=narx_nlp.lbg, ubg=narx_nlp.ubg, p=x0)
    np.testing.assert_allclose(float(sol["f"]), expected_cost, rtol=1e-6, atol=1e-8)


def test_spectral_cost_is_zero_when_under_envelope(tmp_path: Path) -> None:
    """When predicted spectrum is under envelope, NarxMPCNlp cost is 0."""
    n_y, n_u, n_channels, n_controls = 2, 2, 2, 2
    horizon, length, hop = 6, 4, 2
    n_bins = length // 2 + 1

    artifact = _build_artifact(
        tmp_path, n_y=n_y, n_u=n_u, horizon=horizon, n_channels=n_channels, n_controls=n_controls
    )
    model = NNSymbolicModel.from_artifact(artifact)

    rng = np.random.default_rng(_SEED + 11)
    envelope = PsdEnvelope(power=np.full((n_channels, n_bins), 1e8), fs=50.0, window=length, hop=hop)
    x0 = rng.standard_normal(model.state_shape[0])

    narx_nlp = NarxMPCNlp.build(
        model,
        horizon=horizon,
        n_controls=n_controls,
        u_max=np.array([5.0, 5.0]),
        w_y=0.0,
        w_y_terminal=None,
        w_u=0.0,
        w_u_l1=0.0,
        w_psd=100.0,
        psd_envelope=envelope,
    )
    solver = IpoptMPCSolver.build(narx_nlp, max_iter=0, ipopt_options={"max_iter": 0})

    sol = solver.function(
        x0=np.zeros(horizon * (n_controls + n_channels)),
        lbx=narx_nlp.lbx,
        ubx=narx_nlp.ubx,
        lbg=narx_nlp.lbg,
        ubg=narx_nlp.ubg,
        p=x0,
    )
    assert float(sol["f"]) == 0.0


def test_spectral_narx_mpc_controller_from_config(tmp_path: Path) -> None:
    """NarxMPCController.from_config loads psd_ref npz and configures spectral cost."""
    artifact = _build_artifact(tmp_path, horizon=6, n_channels=2, n_controls=2)
    psd_npz_path = tmp_path / "psd.npz"
    np.savez(
        psd_npz_path,
        Pref=np.ones((2, 3)),
        freqs=np.array([0.0, 12.5, 25.0]),
        fs=50.0,
        L=4,
        R=2,
        quantile=0.9,
        n_windows=10,
        plant_fingerprint="test",
    )

    controller = NarxMPCController.from_config(
        {
            "dt": 0.02,
            "artifact": str(artifact),
            "u_max": 2.0,
            "horizon": 6,
            "w_y": 0.0,
            "w_u": 1.0,
            "w_psd": 500.0,
            "psd_ref": str(psd_npz_path),
            "psd_window_s": 0.08,
            "psd_hop_s": 0.04,
        }
    )

    assert controller.w_psd == 500.0
    assert controller.psd_envelope is not None
    assert controller.psd_envelope.window == 4
    assert controller.psd_envelope.hop == 2
    assert controller.psd_envelope.power.shape == (2, 3)


@pytest.mark.parametrize("qpsol", ["qpoases", "qrqp", "osqp"])
def test_narx_mpc_controller_sqp_standalone(tmp_path: Path, qpsol: Literal["qpoases", "qrqp", "osqp"]) -> None:
    """NarxMPCController operates with standalone SQP solver."""
    model = NNSymbolicModel.from_artifact(_build_artifact(tmp_path, n_y=4))
    u_max = 2.0
    controller = NarxMPCController(
        dt=0.01,
        model=model,
        u_max=u_max,
        horizon=3,
        w_y=1.0,
        w_u=0.1,
        solver="sqp",
        sqp_qpsol=qpsol,
        sqp_max_iter=15,
    )
    assert controller.solver == "sqp"
    assert controller.sqp_qpsol == qpsol
    results = _drive(controller, n_steps=6, n_channels=model.n_channels)
    u_last, log_last = results[-1]
    assert not log_last.warmup
    assert not log_last.fallback
    assert u_last.shape == (model.n_controls,)
    assert np.isfinite(u_last).all()
    assert np.all(np.abs(u_last) <= u_max + 1e-6)


def test_narx_mpc_controller_sqp_fallback_on_failure(tmp_path: Path) -> None:
    """When SQP fails to converge, IPOPT fallback is invoked and produces valid control."""
    model = NNSymbolicModel.from_artifact(_build_artifact(tmp_path, n_y=4))
    u_max = 2.0
    controller = NarxMPCController(
        dt=0.01,
        model=model,
        u_max=u_max,
        horizon=3,
        w_y=1.0,
        w_u=0.0,
        solver="sqp_fallback",
        sqp_max_iter=1,
        max_iter=100,
    )
    assert controller.solver == "sqp_fallback"
    results = _drive(controller, n_steps=6, n_channels=model.n_channels)
    u_cold, log_cold = results[3]
    assert not log_cold.warmup
    assert log_cold.fallback
    assert log_cold.success
    assert np.isfinite(u_cold).all()
    assert np.all(np.abs(u_cold) <= u_max + 1e-6)


def test_narx_mpc_controller_sqp_fallback_on_exception(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When SQP raises an unexpected exception, IPOPT fallback is invoked smoothly."""
    model = NNSymbolicModel.from_artifact(_build_artifact(tmp_path, n_y=4))
    u_max = 2.0
    controller = NarxMPCController(
        dt=0.01,
        model=model,
        u_max=u_max,
        horizon=3,
        w_y=1.0,
        w_u=0.0,
        solver="sqp_fallback",
    )

    def _broken_sqp(*_args: object, **_kwargs: object) -> MPCSolveResult:
        msg = "Simulated QP crash"
        raise RuntimeError(msg)

    assert isinstance(controller._solver_obj, SqpFallbackMPCSolver)  # noqa: SLF001
    monkeypatch.setattr(controller._solver_obj._sqp_solver, "solve", _broken_sqp)  # noqa: SLF001

    results = _drive(controller, n_steps=5, n_channels=model.n_channels)
    u_active, log_active = results[3]
    assert not log_active.warmup
    assert log_active.fallback
    assert log_active.success
    assert np.isfinite(u_active).all()
    assert np.all(np.abs(u_active) <= u_max + 1e-6)


def test_narx_mpc_controller_invalid_solver_raises(tmp_path: Path) -> None:
    """Invalid solver mode raises a ValueError."""
    model = NNSymbolicModel.from_artifact(_build_artifact(tmp_path, n_y=4))
    with pytest.raises(ValueError, match="solver must be 'ipopt', 'sqp', or 'sqp_fallback'"):
        NarxMPCController(dt=0.01, model=model, u_max=1.0, solver="bad_solver")  # ty: ignore[invalid-argument-type]


def test_narx_mpc_nlp_and_solver_builders(tmp_path: Path) -> None:
    """NarxMPCNlp.build builds NarxMPCNlp and solver builders instantiate solver wrappers."""
    model = NNSymbolicModel.from_artifact(_build_artifact(tmp_path, n_y=4))
    u_max = np.full(model.n_controls, 2.0)

    narx_nlp = NarxMPCNlp.build(
        model,
        horizon=3,
        n_controls=model.n_controls,
        u_max=u_max,
        w_y=1.0,
        w_u=0.0,
        w_u_l1=0.0,
    )
    assert isinstance(narx_nlp, NarxMPCNlp)
    assert "x" in narx_nlp.nlp
    assert "f" in narx_nlp.nlp
    assert "g" in narx_nlp.nlp
    assert "p" in narx_nlp.nlp

    s_ipopt = IpoptMPCSolver.build(narx_nlp)
    assert isinstance(s_ipopt, IpoptMPCSolver)
    assert isinstance(s_ipopt.function, ca.Function)

    s_sqp = SqpMPCSolver.build(narx_nlp)
    assert isinstance(s_sqp, SqpMPCSolver)
    assert isinstance(s_sqp.function, ca.Function)

    s_fallback = SqpFallbackMPCSolver.build(narx_nlp)
    assert isinstance(s_fallback, SqpFallbackMPCSolver)
    assert isinstance(s_fallback.sqp_function, ca.Function)
    assert isinstance(s_fallback.ipopt_function, ca.Function)


@pytest.mark.parametrize(
    ("shooting_depth", "expected_n_vars", "expected_n_defects"),
    [
        (1, 50 * 2 + 50 * 2, 50 * 2),  # Full lifting: 50 ctrl + 50 output = 200 vars, 100 defects
        (10, 50 * 2 + 4 * (4 * 2), 4 * (4 * 2)),  # S=4 nodes: 100 ctrl + 32 node = 132 vars, 32 defects
        (25, 50 * 2 + 1 * (4 * 2), 1 * (4 * 2)),  # S=1 node: 100 ctrl + 8 node = 108 vars, 8 defects
        (50, 50 * 2, 0),  # Single shooting: 100 ctrl = 100 vars, 0 defects
    ],
)
def test_partially_condensed_narx_nlp_dimensions(
    tmp_path: Path, shooting_depth: int, expected_n_vars: int, expected_n_defects: int
) -> None:
    """Variable and constraint counts match expected dimensions for partial condensing."""
    horizon, n_channels, n_controls, n_y = 50, 2, 2, 4
    model = NNSymbolicModel.from_artifact(
        _build_artifact(tmp_path, n_y=n_y, n_u=3, horizon=horizon, n_channels=n_channels, n_controls=n_controls)
    )
    u_max = np.full(n_controls, 1.0)
    nlp = NarxMPCNlp.build(
        model,
        horizon=horizon,
        shooting_depth=shooting_depth,
        n_controls=n_controls,
        u_max=u_max,
        w_y=1.0,
        w_u=0.01,
    )

    assert nlp.lbx.size == expected_n_vars
    assert nlp.ubx.size == expected_n_vars
    assert nlp.nlp["x"].numel() == expected_n_vars
    assert nlp.nlp["g"].numel() == expected_n_defects + horizon
    assert np.all(nlp.lbx[: horizon * n_controls] == -1.0)
    assert np.all(nlp.ubx[: horizon * n_controls] == 1.0)
    if expected_n_vars > horizon * n_controls:
        assert np.all(nlp.lbx[horizon * n_controls :] == -np.inf)
        assert np.all(nlp.ubx[horizon * n_controls :] == np.inf)

    # With L1 epigraph
    nlp_l1 = NarxMPCNlp.build(
        model,
        horizon=horizon,
        shooting_depth=shooting_depth,
        n_controls=n_controls,
        u_max=u_max,
        w_y=1.0,
        w_u=0.0,
        w_u_l1=1.0,
    )
    assert nlp_l1.lbx.size == expected_n_vars + horizon * n_controls
    assert nlp_l1.nlp["g"].numel() == expected_n_defects + horizon + 2 * horizon * n_controls


@pytest.mark.parametrize("shooting_depth", [1, 10, 25, 50])
def test_shooting_depth_solution_equivalence_on_linear_model(tmp_path: Path, shooting_depth: int) -> None:
    """Optimal solution u0* is mathematically identical across all shooting depths on linear model."""
    horizon = 50
    artifact = _build_artifact(tmp_path, is_linear=True, n_y=4, n_u=3, horizon=horizon, n_channels=2, n_controls=2)
    model = NNSymbolicModel.from_artifact(artifact)

    single_controller = MPCController(
        dt=0.01,
        model=model,
        u_max=0.5,
        horizon=horizon,
        shooting_depth=horizon,
        w_y=1.0,
        w_u=0.05,
        solver="ipopt",
    )
    narx_controller = NarxMPCController(
        dt=0.01,
        model=model,
        u_max=0.5,
        horizon=horizon,
        shooting_depth=shooting_depth,
        w_y=1.0,
        w_u=0.05,
        solver="ipopt",
    )

    us_single = [u for u, _ in _drive(single_controller, n_steps=6, n_channels=2)]
    us_narx = [u for u, _ in _drive(narx_controller, n_steps=6, n_channels=2)]

    np.testing.assert_allclose(np.array(us_narx), np.array(us_single), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shooting_depth", [1, 2, 3, 6])
def test_partially_condensed_narx_mpc_controller_sqp(tmp_path: Path, shooting_depth: int) -> None:
    """Partially condensed NarxMPCController operates with SQP and fallback solvers."""
    model = NNSymbolicModel.from_artifact(
        _build_artifact(tmp_path, is_linear=True, n_y=4, horizon=6, n_channels=2, n_controls=2)
    )
    u_max = 2.0
    controller = NarxMPCController(
        dt=0.01,
        model=model,
        u_max=u_max,
        horizon=6,
        shooting_depth=shooting_depth,
        w_y=1.0,
        w_u=0.1,
        solver="sqp_fallback",
        sqp_max_iter=10,
    )
    results = _drive(controller, n_steps=6, n_channels=model.n_channels)
    u_last, log_last = results[-1]
    assert not log_last.warmup
    assert log_last.success
    assert u_last.shape == (model.n_controls,)
    assert np.isfinite(u_last).all()
    assert np.all(np.abs(u_last) <= u_max + 1e-6)
    assert abs(float(np.sum(u_last))) < 1e-5


def test_narx_mpc_controller_from_config_shooting_depth(tmp_path: Path) -> None:
    """from_config properly parses shooting_depth."""
    artifact = _build_artifact(tmp_path, horizon=50)
    controller = NarxMPCController.from_config(
        {"dt": 0.01, "artifact": str(artifact), "u_max": 3.0, "shooting_depth": 25}
    )
    assert controller.shooting_depth == 25
    assert controller.horizon == 50
