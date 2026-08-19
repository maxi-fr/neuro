from __future__ import annotations

import itertools
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import casadi as ca
import numpy as np
import pytest
import scipy.signal as sps
from scipy.signal.windows import hann
from simulate.simulation import Simulation

from neuro.control import (
    IpoptMPCSolver,
    MPCController,
    MPCControllerLog,
    MPCNlp,
    MPCSolveResult,
    SqpFallbackMPCSolver,
    SqpMPCSolver,
)
from neuro.esn import ESNArtifact, generate_reservoir
from neuro.esn_predictor_casadi import ESNSymbolicModel
from neuro.nn_predictor_casadi import NNSymbolicModel
from neuro.predictor.artifact import MLPArtifact
from neuro.predictor.data import load_trajectory
from neuro.spectral import PsdEnvelope, compute_periodograms, hinge_penalty
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from neuro.types import FloatArray

_SEED = 7


def _random_layers(rng: np.random.Generator, sizes: list[int]) -> tuple[tuple[FloatArray, FloatArray], ...]:
    """Random ``(weight (out, in), bias (out,))`` pairs, drawn uniformly from ``+-1/sqrt(fan_in)``."""
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
) -> Path:
    """Save a tiny synthetic MLP artifact and return its basename path."""
    rng = np.random.default_rng(_SEED)
    in_size = n_y * n_channels + n_u * n_controls
    scalers = {
        "u_mean": rng.uniform(-1.0, 1.0, n_controls),
        "u_scale": rng.uniform(0.5, 2.0, n_controls),
        "y_mean": rng.uniform(-1.0, 1.0, n_channels),
        "y_scale": rng.uniform(0.5, 2.0, n_channels),
    }
    artifact = tmp_path / "art"
    MLPArtifact(
        layers=_random_layers(rng, [in_size, 5, 5, n_channels]),
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


def _drive(controller: MPCController, n_steps: int, n_channels: int) -> list[tuple[FloatArray, MPCControllerLog]]:
    """Feed ``n_steps`` random EEG measurements through ``update`` and collect the outputs."""
    rng = np.random.default_rng(_SEED + 1)
    out = []
    for k in range(n_steps):
        u, log = controller.update(k * controller.dt, ref=np.array([0.0]), x_hat=rng.standard_normal(n_channels))
        out.append((np.atleast_1d(np.asarray(u, dtype=np.float64)), log))
    return out


def test_output_condensation_matches_full_state_rollout(tmp_path: Path) -> None:
    """The output-condensed rollout reproduces the full-state f_step/f_out trajectory exactly.

    The MPC lifts only the per-step EEG output and rebuilds the predictor's history windows
    from the parameter state plus earlier lifted outputs (see ``_build_solver``). That must
    equal rolling the full shift-register state with ``f_step`` and reading ``f_out`` --
    otherwise the condensed NLP would not be the same optimization problem.
    """
    n_y, n_u, n_channels, n_controls = 4, 3, 2, 2
    artifact = _build_artifact(tmp_path, n_y=n_y, n_u=n_u, n_channels=n_channels, n_controls=n_controls)
    model = NNSymbolicModel.from_artifact(artifact)
    art = MLPArtifact.load(artifact)
    rng = np.random.default_rng(_SEED + 2)
    horizon = 5
    x0 = rng.standard_normal(model.state_shape[0])
    controls = rng.standard_normal((horizon, n_controls))

    x = x0
    y_full = []
    for u in controls:
        x = np.asarray(model.f_step(x, u)).reshape(-1)
        y_full.append(np.asarray(model.f_out(x)).reshape(-1))

    y_sym = ca.MX.sym("y_win", n_y * n_channels)
    u_sym = ca.MX.sym("u_win", n_u * n_controls)
    predict = ca.Function("predict", [y_sym, u_sym], [model.predict_output(y_sym, u_sym)])

    split = n_y * n_channels
    y_win = [x0[i * n_channels : (i + 1) * n_channels] for i in range(n_y)]
    u_win = [x0[split + i * n_controls : split + (i + 1) * n_controls] for i in range(n_u)]
    y_cond = []
    for u in controls:
        u_win = [*u_win[1:], u]
        y_pred = np.asarray(predict(np.concatenate(y_win), np.concatenate(u_win))).reshape(-1)
        y_cond.append(art.decode(y_pred))
        y_win = [*y_win[1:], y_pred]

    np.testing.assert_allclose(np.array(y_cond), np.array(y_full), rtol=1e-9, atol=1e-9)


def test_warmup_emits_zero_until_window_filled(tmp_path: Path) -> None:
    """While the EEG window is still zero-padded, the MPC holds off and emits zeros."""
    n_y = 4
    model = NNSymbolicModel.from_artifact(_build_artifact(tmp_path, n_y=n_y))
    controller = MPCController(dt=0.01, model=model, u_max=0.5, horizon=3)

    results = _drive(controller, n_steps=n_y, n_channels=model.n_channels)
    for u, log in results[: n_y - 1]:
        assert log.warmup
        np.testing.assert_array_equal(u, np.zeros(model.n_controls))

    assert not results[-1][1].warmup


def test_update_respects_bounds(tmp_path: Path) -> None:
    """Past warm-up, update returns a finite (n_controls,) control within the box bounds.

    Does not assert solver convergence: with a random-weight ReLU net the suppression
    landscape is non-smooth, so IPOPT may not certify KKT -- but the applied control is
    still finite and bound-respecting (here pushed toward the bound to cut EEG power).
    """
    u_max = 0.5
    model = NNSymbolicModel.from_artifact(_build_artifact(tmp_path, n_y=4))
    controller = MPCController(dt=0.01, model=model, u_max=u_max, horizon=3, w_y=1.0, w_u=0.0)

    u, _ = _drive(controller, n_steps=6, n_channels=model.n_channels)[-1]
    assert u.shape == (model.n_controls,)
    assert np.isfinite(u).all()
    assert np.all(np.abs(u) <= u_max + 1e-6)


def test_control_obeys_kirchhoff_current_law(tmp_path: Path) -> None:
    """The emitted control sums to zero across electrodes (Kirchhoff's current law).

    With ``w_y=1, w_u=0`` the MPC actively stimulates to cut predicted EEG power, yet the
    montage's per-electrode currents must still balance so no net current is injected.
    """
    model = NNSymbolicModel.from_artifact(_build_artifact(tmp_path, n_y=4))
    controller = MPCController(dt=0.01, model=model, u_max=5.0, horizon=3, w_y=1.0, w_u=0.0)

    u, log = _drive(controller, n_steps=6, n_channels=model.n_channels)[-1]
    assert not log.warmup
    assert np.linalg.norm(u, ord=1) > 1e-3
    assert abs(float(np.sum(u))) < 1e-6


def test_pure_effort_cost_yields_zero_control(tmp_path: Path) -> None:
    """With w_y=0 the cost is sum||u||^2, whose unconstrained minimizer is u=0."""
    model = NNSymbolicModel.from_artifact(_build_artifact(tmp_path, n_y=4))
    controller = MPCController(dt=0.01, model=model, u_max=1.0, horizon=3, w_y=0.0, w_u=1.0)

    u, log = _drive(controller, n_steps=6, n_channels=model.n_channels)[-1]
    assert log.success
    np.testing.assert_allclose(u, np.zeros(model.n_controls), atol=1e-4)


def test_l1_penalty_drives_control_toward_zero(tmp_path: Path) -> None:
    """A large L1 control-effort penalty suppresses stimulation, driving the applied control to 0.

    With ``w_y>0`` and ``w_u=0`` the baseline MPC stimulates to cut predicted EEG power; adding a
    dominant L1 effort penalty (epigraph-reformulated into the NLP) makes stimulating uneconomical,
    so the control collapses to (near-)zero.
    """
    model = NNSymbolicModel.from_artifact(_build_artifact(tmp_path, n_y=4))
    base: dict[str, Any] = {"dt": 0.01, "model": model, "u_max": 5.0, "horizon": 3, "w_y": 1.0, "w_u": 0.0}

    u_l2 = _drive(MPCController(w_u_l1=0.0, **base), n_steps=6, n_channels=model.n_channels)[-1][0]
    u_l1 = _drive(MPCController(w_u_l1=1000.0, **base), n_steps=6, n_channels=model.n_channels)[-1][0]

    assert np.linalg.norm(u_l2, ord=1) > 1e-3
    np.testing.assert_allclose(u_l1, np.zeros(model.n_controls), atol=1e-4)


def test_per_electrode_bounds_rejected_when_mismatched(tmp_path: Path) -> None:
    """A u_max length that is neither 1 nor n_controls is rejected."""
    model = NNSymbolicModel.from_artifact(_build_artifact(tmp_path, n_controls=2))
    with pytest.raises(ValueError, match="u_max has 3 entries but n_controls is 2"):
        MPCController(dt=0.01, model=model, u_max=[1.0, 2.0, 3.0], horizon=3)


def test_from_config_loads_artifact_and_defaults_horizon(tmp_path: Path) -> None:
    """from_config loads the artifact path and defaults the horizon to the model's."""
    artifact = _build_artifact(tmp_path, horizon=5)
    controller = MPCController.from_config({"dt": 0.01, "artifact": str(artifact), "u_max": 3.0, "w_u_l1": 0.25})
    assert controller.dt == 0.01
    assert controller.horizon == 5
    assert controller.n_controls == 2
    assert controller.w_u_l1 == 0.25


def test_closed_loop_simulation_runs(tmp_path: Path) -> None:
    """The MPC closes the loop through the orchestrator and keeps controls within bounds."""
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
            "class_path": "neuro.control.MPCController",
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


def _build_tiny_esn_artifact(
    tmp_path: Path,
    *,
    reservoir_size: int = 30,
    priming_steps: int = 5,
    horizon: int = 3,
    n_channels: int = 2,
    n_controls: int = 2,
) -> Path:
    """Save a tiny synthetic ESN artifact for testing."""
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


def test_mpc_controller_with_esn_model(tmp_path: Path) -> None:
    """MPCController runs end-to-end with an ESNSymbolicModel across warmup and active steps."""
    priming_steps = 5
    art_path = _build_tiny_esn_artifact(tmp_path, priming_steps=priming_steps)
    model = ESNSymbolicModel.from_artifact(art_path)
    u_max = 0.5
    controller = MPCController(dt=0.01, model=model, u_max=u_max, horizon=3, w_y=1.0, w_u=0.0)

    results = _drive(controller, n_steps=priming_steps + 3, n_channels=model.n_channels)
    for u, log in results[: priming_steps - 1]:
        assert log.warmup
        np.testing.assert_array_equal(u, np.zeros(model.n_controls))

    for u, log in results[priming_steps - 1 :]:
        assert not log.warmup
        assert u.shape == (model.n_controls,)
        assert np.isfinite(u).all()
        assert np.all(np.abs(u) <= u_max + 1e-6)


def test_mpc_from_config_loads_esn_artifact(tmp_path: Path) -> None:
    """MPCController.from_config routes through build_symbolic_model for ESN artifacts."""
    art_path = _build_tiny_esn_artifact(tmp_path, priming_steps=3)
    controller = MPCController.from_config(
        {
            "dt": 0.01,
            "artifact": str(art_path),
            "u_max": 0.5,
            "horizon": 3,
        }
    )
    assert isinstance(controller.model, ESNSymbolicModel)
    assert controller.horizon == 3
    assert controller.n_controls == 2


def test_symbolic_model_priming_seam(tmp_path: Path) -> None:
    """initial_state, absorb, and is_ready match the artifact prime contract for NN and ESN."""
    rng = np.random.default_rng(_SEED + 5)

    # NN model priming
    n_y, n_u, n_ch, n_ctrl = 4, 3, 2, 2
    nn_art_path = _build_artifact(tmp_path, n_y=n_y, n_u=n_u, n_channels=n_ch, n_controls=n_ctrl)
    nn_art = MLPArtifact.load(nn_art_path)
    nn_model = NNSymbolicModel(nn_art)

    assert nn_model.native_horizon == nn_art.horizon
    state = nn_model.initial_state()
    assert not nn_model.is_ready(state)

    # Nonzero controls: with a zeroed u buffer the two paths agree whatever their u alignment.
    y_seq = rng.standard_normal((n_y, n_ch))
    u_seq = rng.standard_normal((n_y, n_ctrl))
    for t in range(n_y - 1):
        state = nn_model.absorb(state, y_seq[t], u_seq[t])
        assert not nn_model.is_ready(state)

    state = nn_model.absorb(state, y_seq[n_y - 1], u_seq[n_y - 1])
    assert nn_model.is_ready(state)
    expected_nn = nn_art.prime(y_seq, u_seq)
    np.testing.assert_allclose(state, expected_nn, atol=1e-12)

    # ESN model priming
    priming_steps, res_size = 6, 20
    esn_art_path = _build_tiny_esn_artifact(
        tmp_path, reservoir_size=res_size, priming_steps=priming_steps, n_channels=n_ch
    )
    esn_art = ESNArtifact.load(esn_art_path)
    esn_model = ESNSymbolicModel(esn_art)

    assert esn_model.native_horizon == esn_art.horizon
    esn_state = esn_model.initial_state()
    assert not esn_model.is_ready(esn_state)

    y_esn = rng.standard_normal((priming_steps, n_ch))
    u_esn = rng.standard_normal((priming_steps, n_ctrl))
    for t in range(priming_steps - 1):
        esn_state = esn_model.absorb(esn_state, y_esn[t], u_esn[t])
        assert not esn_model.is_ready(esn_state)

    esn_state = esn_model.absorb(esn_state, y_esn[priming_steps - 1], u_esn[priming_steps - 1])
    assert esn_model.is_ready(esn_state)
    expected_esn = esn_art.prime(y_esn, u_esn)
    np.testing.assert_allclose(esn_state, expected_esn, atol=1e-12)


@pytest.mark.parametrize(
    ("horizon", "length", "hop", "n_channels"),
    [(6, 4, 2, 2), (75, 50, 25, 8)],
    ids=["toy", "production-geometry"],
)
def test_spectral_cost_matches_numpy_periodogram(
    tmp_path: Path, horizon: int, length: int, hop: int, n_channels: int
) -> None:
    """CasADi spectral cost matches the numpy/scipy periodogram reference the envelope is built with."""
    n_controls, fs = 2, 50.0
    n_bins = length // 2 + 1

    artifact = _build_artifact(tmp_path, n_y=2, n_u=2, horizon=horizon, n_channels=n_channels, n_controls=n_controls)
    model = NNSymbolicModel.from_artifact(artifact)

    rng = np.random.default_rng(_SEED + 10)
    envelope = PsdEnvelope(power=np.abs(rng.normal(size=(n_channels, n_bins))) + 0.1, fs=fs, window=length, hop=hop)
    x0 = rng.standard_normal(model.state_shape[0])
    u_fixed = np.zeros((horizon, n_controls))

    x = x0
    y_steps = []
    for step in range(horizon):
        x = np.asarray(model.f_step(x, u_fixed[step])).reshape(-1)
        y_steps.append(np.asarray(model.f_out(x)).reshape(-1))
    y_traj = np.array(y_steps)  # (horizon, n_channels)

    expected_cost = hinge_penalty(compute_periodograms(y_traj, fs=fs, window=length, hop=hop), envelope.power)

    mpc_nlp = MPCNlp.build(
        model,
        horizon=horizon,
        shooting_depth=horizon,
        n_controls=n_controls,
        u_max=np.array([5.0, 5.0]),
        w_y=0.0,
        w_y_terminal=None,
        w_u=0.0,
        w_u_l1=0.0,
        w_psd=1.0,
        psd_envelope=envelope,
    )
    solver = IpoptMPCSolver.build(mpc_nlp, max_iter=0, ipopt_options={"max_iter": 0})

    sol = solver.function(
        x0=u_fixed.reshape(-1), lbx=mpc_nlp.lbx, ubx=mpc_nlp.ubx, lbg=mpc_nlp.lbg, ubg=mpc_nlp.ubg, p=x0
    )
    np.testing.assert_allclose(float(sol["f"]), expected_cost, rtol=1e-6, atol=1e-8)


def test_spectral_cost_is_zero_when_under_envelope(tmp_path: Path) -> None:
    """When the predicted spectrum is everywhere under the reference envelope, cost is exactly 0."""
    n_y, n_u, n_channels, n_controls = 2, 2, 2, 2
    horizon, L, R = 6, 4, 2
    n_bins = L // 2 + 1

    artifact = _build_artifact(
        tmp_path, n_y=n_y, n_u=n_u, horizon=horizon, n_channels=n_channels, n_controls=n_controls
    )
    model = NNSymbolicModel.from_artifact(artifact)

    rng = np.random.default_rng(_SEED + 11)
    envelope = PsdEnvelope(power=np.full((n_channels, n_bins), 1e8), fs=50.0, window=L, hop=R)
    x0 = rng.standard_normal(model.state_shape[0])

    mpc_nlp = MPCNlp.build(
        model,
        horizon=horizon,
        shooting_depth=horizon,
        n_controls=n_controls,
        u_max=np.array([5.0, 5.0]),
        w_y=0.0,
        w_y_terminal=None,
        w_u=0.0,
        w_u_l1=0.0,
        w_psd=100.0,
        psd_envelope=envelope,
    )
    solver = IpoptMPCSolver.build(mpc_nlp, max_iter=0, ipopt_options={"max_iter": 0})

    sol = solver.function(
        x0=np.zeros(horizon * n_controls),
        lbx=mpc_nlp.lbx,
        ubx=mpc_nlp.ubx,
        lbg=mpc_nlp.lbg,
        ubg=mpc_nlp.ubg,
        p=x0,
    )
    assert float(sol["f"]) == 0.0


def test_spectral_mpc_controller_from_config(tmp_path: Path) -> None:
    """MPCController.from_config loads psd_ref npz and configures spectral cost."""
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

    controller = MPCController.from_config(
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


def test_healthy_plant_scores_near_zero_on_envelope() -> None:
    """The healthy plant scores ~0 on its own p90 reference envelope."""
    psd_path = Path("data/healthy_psd.npz")
    if not psd_path.exists():
        pytest.skip("data/healthy_psd.npz not generated yet")

    envelope = PsdEnvelope.load(psd_path)

    data_files = sorted(Path("data/healthy_reference").glob("sim_*.npz"))
    if not data_files:
        pytest.skip("data/healthy_reference trajectories not present")

    _, y = load_trajectory(str(data_files[0]), n_steps=None, downsample=200, dt=1e-4)
    power = compute_periodograms(y, fs=envelope.fs, window=envelope.window, hop=envelope.hop)
    assert hinge_penalty(power, envelope.power) < 0.05


@pytest.mark.parametrize("qpsol", ["qpoases", "qrqp", "osqp"])
def test_mpc_controller_sqp_standalone(tmp_path: Path, qpsol: Literal["qpoases", "qrqp", "osqp"]) -> None:
    """MPCController operates with standalone SQP solver using various QP subsolvers."""
    model = NNSymbolicModel.from_artifact(_build_artifact(tmp_path, n_y=4))
    u_max = 2.0
    controller = MPCController(
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


def test_mpc_controller_sqp_fallback_on_failure(tmp_path: Path) -> None:
    """When SQP fails to converge, IPOPT fallback is invoked and produces valid control."""
    model = NNSymbolicModel.from_artifact(_build_artifact(tmp_path, n_y=4))
    u_max = 2.0
    controller = MPCController(
        dt=0.01,
        model=model,
        u_max=u_max,
        horizon=3,
        w_y=1.0,
        w_u=0.0,
        solver="sqp_fallback",
        sqp_max_iter=1,  # Strict 1-iteration cap forces SQP fallback on cold initial solve
        max_iter=100,
    )
    assert controller.solver == "sqp_fallback"
    results = _drive(controller, n_steps=6, n_channels=model.n_channels)
    # The first active step (index 3) is cold and fails 1-iter SQP, triggering IPOPT fallback
    u_cold, log_cold = results[3]
    assert not log_cold.warmup
    assert log_cold.fallback
    assert log_cold.success
    assert np.isfinite(u_cold).all()
    assert np.all(np.abs(u_cold) <= u_max + 1e-6)

    # Subsequent warm-started steps also produce valid bounded control
    for u, log in results[3:]:
        assert log.success
        assert np.isfinite(u).all()
        assert np.all(np.abs(u) <= u_max + 1e-6)


def test_mpc_controller_sqp_fallback_on_exception(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When SQP raises an unexpected exception, IPOPT fallback is invoked smoothly."""
    model = NNSymbolicModel.from_artifact(_build_artifact(tmp_path, n_y=4))
    u_max = 2.0
    controller = MPCController(
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


def test_mpc_controller_from_config_sqp_fallback(tmp_path: Path) -> None:
    """from_config properly passes SQP and fallback configuration options."""
    artifact = _build_artifact(tmp_path, horizon=5)
    controller = MPCController.from_config(
        {
            "dt": 0.01,
            "artifact": str(artifact),
            "u_max": 3.0,
            "solver": "sqp_fallback",
            "sqp_qpsol": "osqp",
            "sqp_hessian": "limited-memory",
            "sqp_max_iter": 20,
            "sqp_lbfgs_memory": 5,
        }
    )
    assert controller.solver == "sqp_fallback"
    assert controller.sqp_qpsol == "osqp"
    assert controller.sqp_hessian == "limited-memory"
    assert controller.sqp_max_iter == 20
    assert controller.sqp_lbfgs_memory == 5


def test_mpc_controller_invalid_solver_raises(tmp_path: Path) -> None:
    """Invalid solver mode raises a ValueError."""
    model = NNSymbolicModel.from_artifact(_build_artifact(tmp_path, n_y=4))
    with pytest.raises(ValueError, match="solver must be 'ipopt', 'sqp', or 'sqp_fallback'"):
        MPCController(dt=0.01, model=model, u_max=1.0, solver="bad_solver")  # ty: ignore[invalid-argument-type]


def test_mpc_nlp_and_solver_builders(tmp_path: Path) -> None:
    """MPCNlp.build builds MPCNlp and solver builders instantiate correct solver wrappers."""
    model = NNSymbolicModel.from_artifact(_build_artifact(tmp_path, n_y=4))
    u_max = np.full(model.n_controls, 2.0)

    mpc_nlp = MPCNlp.build(
        model,
        horizon=3,
        shooting_depth=3,
        n_controls=model.n_controls,
        u_max=u_max,
        w_y=1.0,
        w_u=0.0,
        w_u_l1=0.0,
    )
    assert isinstance(mpc_nlp, MPCNlp)
    assert "x" in mpc_nlp.nlp
    assert "f" in mpc_nlp.nlp
    assert "g" in mpc_nlp.nlp
    assert "p" in mpc_nlp.nlp

    s_ipopt = IpoptMPCSolver.build(mpc_nlp)
    assert isinstance(s_ipopt, IpoptMPCSolver)
    assert isinstance(s_ipopt.function, ca.Function)

    s_sqp = SqpMPCSolver.build(mpc_nlp)
    assert isinstance(s_sqp, SqpMPCSolver)
    assert isinstance(s_sqp.function, ca.Function)

    s_fallback = SqpFallbackMPCSolver.build(mpc_nlp)
    assert isinstance(s_fallback, SqpFallbackMPCSolver)
    assert isinstance(s_fallback.sqp_function, ca.Function)
    assert isinstance(s_fallback.ipopt_function, ca.Function)
