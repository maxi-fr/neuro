from __future__ import annotations

from typing import TYPE_CHECKING, Any

import casadi as ca
import equinox as eqx
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)  # noqa: FBT003

from simulate.simulation import Simulation  # noqa: E402

from neuro.control import MPCController, MPCControllerLog  # noqa: E402
from neuro.nn_predictor_casadi import NNSymbolicModel  # noqa: E402
from neuro.prediction import AutoregressivePredictor, MLPArtifact  # noqa: E402
from neuro.transforms import PCAProjection, Pipeline, Standardizer  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

    from neuro.types import FloatArray

_SEED = 7


def _standardizer_pipeline(center: FloatArray, scale: FloatArray) -> Pipeline:
    """A single-step standardizer pipeline from raw ``center``/``scale`` arrays."""
    return Pipeline((Standardizer(center=np.asarray(center, np.float64), scale=np.asarray(scale, np.float64)),))


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
    mlp = eqx.nn.MLP(
        in_size=in_size,
        out_size=n_channels,
        width_size=5,
        depth=2,
        activation=jax.nn.relu,
        key=jax.random.PRNGKey(0),
    )
    wrapped = AutoregressivePredictor(
        model=mlp, n_y=n_y, n_u=n_u, horizon=horizon, n_channels=n_channels, n_controls=n_controls, activation="relu"
    )
    scalers = {
        "u_mean": rng.uniform(-1.0, 1.0, n_controls),
        "u_scale": rng.uniform(0.5, 2.0, n_controls),
        "y_mean": rng.uniform(-1.0, 1.0, n_channels),
        "y_scale": rng.uniform(0.5, 2.0, n_channels),
    }
    artifact = tmp_path / "art"
    MLPArtifact(
        model=wrapped,
        dt=0.01,
        downsample=1,
        y_pipeline=_standardizer_pipeline(scalers["y_mean"], scalers["y_scale"]),
        u_pipeline=_standardizer_pipeline(scalers["u_mean"], scalers["u_scale"]),
    ).save(artifact)
    return artifact


def _build_projection_artifact(
    tmp_path: Path,
    *,
    n_y: int = 4,
    n_u: int = 3,
    horizon: int = 3,
    k: int = 2,
    n_eeg: int = 6,
    n_controls: int = 2,
) -> Path:
    """Save a tiny artifact whose predictor runs in a ``k``-dim PCA latent space."""
    rng = np.random.default_rng(_SEED)
    q, _ = np.linalg.qr(rng.standard_normal((n_eeg, n_eeg)))
    basis = np.ascontiguousarray(q[:, :k].T)
    mean = rng.standard_normal(n_eeg)
    mlp = eqx.nn.MLP(
        in_size=n_y * k + n_u * n_controls,
        out_size=k,
        width_size=5,
        depth=2,
        activation=jax.nn.relu,
        key=jax.random.PRNGKey(0),
    )
    wrapped = AutoregressivePredictor(
        model=mlp, n_y=n_y, n_u=n_u, horizon=horizon, n_channels=k, n_controls=n_controls, activation="relu"
    )
    y_pipeline = Pipeline(
        (
            Standardizer(center=rng.uniform(-1.0, 1.0, n_eeg), scale=rng.uniform(0.5, 2.0, n_eeg)),
            PCAProjection(basis=basis, mean=mean),
        )
    )
    u_pipeline = _standardizer_pipeline(rng.uniform(-1.0, 1.0, n_controls), rng.uniform(0.5, 2.0, n_controls))
    artifact = tmp_path / "art_proj"
    MLPArtifact(model=wrapped, dt=0.01, downsample=1, y_pipeline=y_pipeline, u_pipeline=u_pipeline).save(artifact)
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


def test_projection_shrinks_shooting_state_and_respects_bounds(tmp_path: Path) -> None:
    """With a PCA-projection artifact the MPC's shooting state is latent-sized and the loop runs.

    The controller is fed raw ``n_eeg``-channel measurements and encodes them to the ``k``
    latent components internally, so the NLP state is ``n_y*k + n_u*n_controls`` rather than
    ``n_y*n_eeg + ...`` -- the whole point of the projection for solve time.
    """
    n_y, n_u, k, n_eeg, n_controls = 4, 3, 2, 6, 2
    u_max = 0.5
    model = NNSymbolicModel.from_artifact(
        _build_projection_artifact(tmp_path, n_y=n_y, n_u=n_u, k=k, n_eeg=n_eeg, n_controls=n_controls)
    )
    controller = MPCController(dt=0.01, model=model, u_max=u_max, horizon=3, w_y=1.0, w_u=0.0)

    assert model.n_channels == k
    assert model.n_eeg_channels == n_eeg
    assert model.state_shape[0] == n_y * k + n_u * n_controls

    u, log = _drive(controller, n_steps=n_y + 2, n_channels=n_eeg)[-1]
    assert not log.warmup
    assert u.shape == (n_controls,)
    assert np.isfinite(u).all()
    assert np.all(np.abs(u) <= u_max + 1e-6)
