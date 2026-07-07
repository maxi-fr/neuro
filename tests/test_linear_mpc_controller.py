"""Tests for :class:`neuro.control.LinearMPCController`.

The linear MPC embeds a *linear* (0-hidden-layer) CasADi NN predictor and solves the
receding-horizon suppression problem as a convex QP -- either the stacked ``"sparse"``
formulation (OSQP) or the condensed ``"dense"`` one (qpOASES). With a *synthetic*
(random-weight) depth-0 artifact these tests verify the machinery: both formulations solve
the same QP, agree with the nonlinear IPOPT MPC on the (linear) model, respect the box
bounds, and the loop closes through the ``simulate`` orchestrator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import numpy as np
import pytest

# float64 parity with the NN predictor; enable before any array is created.
jax.config.update("jax_enable_x64", True)  # noqa: FBT003

from simulate.simulation import Simulation  # noqa: E402

from neuro.control import (  # noqa: E402
    LinearMPCController,
    MPCController,
)
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
    depth: int = 0,
    n_y: int = 4,
    n_u: int = 3,
    horizon: int = 4,
    n_channels: int = 2,
    n_controls: int = 2,
) -> Path:
    """Save a tiny synthetic MLP artifact and return its basename path (``depth=0`` is linear)."""
    rng = np.random.default_rng(_SEED)
    in_size = n_y * n_channels + n_u * n_controls
    mlp = eqx.nn.MLP(
        in_size=in_size,
        out_size=n_channels,
        width_size=5,
        depth=depth,
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
    horizon: int = 4,
    k: int = 2,
    n_eeg: int = 6,
    n_controls: int = 2,
) -> Path:
    """Save a tiny linear (depth-0) artifact whose predictor runs in a ``k``-dim PCA latent space."""
    rng = np.random.default_rng(_SEED)
    q, _ = np.linalg.qr(rng.standard_normal((n_eeg, n_eeg)))
    basis = np.ascontiguousarray(q[:, :k].T)
    mean = rng.standard_normal(n_eeg)
    mlp = eqx.nn.MLP(
        in_size=n_y * k + n_u * n_controls,
        out_size=k,
        width_size=5,
        depth=0,
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


def _drive(
    controller: LinearMPCController | MPCController, n_steps: int, n_channels: int
) -> list[tuple[FloatArray, Any]]:
    """Feed ``n_steps`` deterministic EEG measurements through ``update`` and collect the outputs."""
    rng = np.random.default_rng(_SEED + 1)
    out = []
    for k in range(n_steps):
        u, log = controller.update(k * controller.dt, ref=np.array([0.0]), x_hat=rng.standard_normal(n_channels))
        out.append((np.atleast_1d(np.asarray(u, dtype=np.float64)), log))
    return out


def test_sparse_dense_ipopt_equivalence(tmp_path: Path) -> None:
    """sparse (OSQP), dense (qpOASES), and the IPOPT MPC solve the same QP on a linear model.

    Driving exactly ``n_y`` steps lets the window fill with identical measurements and all-zero
    controls, so every controller's first real solve sees the *same* ``x0``. With ``w_u>0`` the
    QP is strictly convex (unique optimum), so the applied control must agree across the two QP
    formulations (tightly) and the nonlinear IPOPT MPC on the same affine model (loosely).
    """
    n_y, n_channels = 4, 2
    kw: dict[str, Any] = {"dt": 0.01, "u_max": 5.0, "horizon": 4, "w_y": 1.0, "w_u": 0.1}
    model = NNSymbolicModel.from_artifact(_build_artifact(tmp_path, n_y=n_y, n_channels=n_channels))

    u_sparse = _drive(LinearMPCController(model=model, formulation="sparse", **kw), n_y, n_channels)[-1][0]
    u_dense = _drive(LinearMPCController(model=model, formulation="dense", **kw), n_y, n_channels)[-1][0]
    u_ipopt = _drive(MPCController(model=model, **kw), n_y, n_channels)[-1][0]

    np.testing.assert_allclose(u_sparse, u_dense, atol=1e-5)
    np.testing.assert_allclose(u_sparse, u_ipopt, atol=1e-3)


@pytest.mark.parametrize("formulation", ["sparse", "dense"])
def test_update_respects_bounds(tmp_path: Path, formulation: str) -> None:
    """Past warm-up, update returns a finite ``(n_controls,)`` control within the box bounds."""
    u_max = 0.5
    model = NNSymbolicModel.from_artifact(_build_artifact(tmp_path, n_y=4))
    controller = LinearMPCController(
        dt=0.01, model=model, u_max=u_max, horizon=4, w_y=1.0, w_u=0.0, formulation=formulation
    )

    u, log = _drive(controller, n_steps=6, n_channels=model.n_channels)[-1]
    assert not log.warmup
    assert u.shape == (model.n_controls,)
    assert np.isfinite(u).all()
    assert np.all(np.abs(u) <= u_max + 1e-6)


@pytest.mark.parametrize("formulation", ["sparse", "dense"])
def test_pure_effort_cost_yields_zero_control(tmp_path: Path, formulation: str) -> None:
    """With w_y=0 the cost is sum||u||^2, whose unique constrained minimizer is u=0."""
    model = NNSymbolicModel.from_artifact(_build_artifact(tmp_path, n_y=4))
    controller = LinearMPCController(
        dt=0.01, model=model, u_max=1.0, horizon=4, w_y=0.0, w_u=1.0, formulation=formulation
    )

    u, log = _drive(controller, n_steps=6, n_channels=model.n_channels)[-1]
    assert log.success
    np.testing.assert_allclose(u, np.zeros(model.n_controls), atol=1e-4)


@pytest.mark.parametrize("formulation", ["sparse", "dense"])
def test_warmup_emits_zero_until_window_filled(tmp_path: Path, formulation: str) -> None:
    """While the EEG window is still zero-padded, the MPC holds off and emits zeros."""
    n_y = 4
    model = NNSymbolicModel.from_artifact(_build_artifact(tmp_path, n_y=n_y))
    controller = LinearMPCController(dt=0.01, model=model, u_max=0.5, horizon=4, formulation=formulation)

    results = _drive(controller, n_steps=n_y, n_channels=model.n_channels)
    for u, log in results[: n_y - 1]:
        assert log.warmup
        np.testing.assert_array_equal(u, np.zeros(model.n_controls))
    assert not results[-1][1].warmup


def test_from_config_loads_artifact_and_honours_formulation(tmp_path: Path) -> None:
    """from_config loads the artifact, defaults the horizon, and honours the formulation."""
    artifact = _build_artifact(tmp_path, horizon=5)
    controller = LinearMPCController.from_config(
        {"dt": 0.01, "artifact": str(artifact), "u_max": 3.0, "formulation": "dense"}
    )
    assert controller.dt == 0.01
    assert controller.horizon == 5  # defaulted from the artifact
    assert controller.n_controls == 2
    assert controller.formulation == "dense"


def test_nonlinear_model_rejected(tmp_path: Path) -> None:
    """A non-linear (depth>0) predictor is rejected -- the QP would silently linearize it."""
    model = NNSymbolicModel.from_artifact(_build_artifact(tmp_path, depth=2))
    with pytest.raises(ValueError, match="requires a linear predictor"):
        LinearMPCController(dt=0.01, model=model, u_max=1.0, horizon=4)


def test_invalid_formulation_rejected(tmp_path: Path) -> None:
    """An unknown formulation name is rejected up front."""
    model = NNSymbolicModel.from_artifact(_build_artifact(tmp_path))
    with pytest.raises(ValueError, match="formulation must be"):
        LinearMPCController(dt=0.01, model=model, u_max=1.0, horizon=4, formulation="banded")


@pytest.mark.parametrize("formulation", ["sparse", "dense"])
def test_projection_runs_and_respects_bounds(tmp_path: Path, formulation: str) -> None:
    """With a PCA-projection artifact the linear MPC runs in latent space and stays in bounds."""
    n_y, n_u, k, n_eeg, n_controls = 4, 3, 2, 6, 2
    u_max = 0.5
    model = NNSymbolicModel.from_artifact(
        _build_projection_artifact(tmp_path, n_y=n_y, n_u=n_u, k=k, n_eeg=n_eeg, n_controls=n_controls)
    )
    controller = LinearMPCController(
        dt=0.01, model=model, u_max=u_max, horizon=4, w_y=1.0, w_u=0.0, formulation=formulation
    )

    assert model.n_channels == k
    assert model.state_shape[0] == n_y * k + n_u * n_controls  # latent-sized, not n_y*n_eeg

    # Feed raw EEG measurements (dim n_eeg); the controller encodes them internally.
    u, log = _drive(controller, n_steps=n_y + 2, n_channels=n_eeg)[-1]
    assert not log.warmup
    assert u.shape == (n_controls,)
    assert np.isfinite(u).all()
    assert np.all(np.abs(u) <= u_max + 1e-6)


@pytest.mark.parametrize("formulation", ["sparse", "dense"])
def test_closed_loop_simulation_runs(tmp_path: Path, formulation: str) -> None:
    """The linear MPC closes the loop through the orchestrator and keeps controls within bounds."""
    n_channels, u_max = 3, 3.0
    artifact = _build_artifact(tmp_path, n_y=4, n_u=3, horizon=4, n_channels=n_channels, n_controls=2)

    config = {
        "t_end": 0.05,
        "dynamics": {
            "class_path": "neuro.jansen_rit.JansenRitDynamics",
            "dt": 1e-4,
            "seed": 69,
            "connectome": {
                "speed": 50.0,
                "target_electrode": ["CP5", "T7"],
                "gamma_spread": 20.0,
                "K": 0.5357,
            },
            "params": {"A": 3.25},
        },
        "reference": {"class_path": "simulate.reference.StepReference", "dt": 1e-4, "step_value": 0.0},
        "sensors": {
            "class_path": "simulate.sensor.GaussianSensor",
            "dt": 1e-4,
            "std_dev": 0.0,
            "measurement": {
                "class_path": "neuro.measurement.EEGMeasurement",
                "speed": 50.0,
                "selected_channels": [0, 1, 2],
            },
        },
        "estimator": {"class_path": "simulate.estimator.IdentityEstimator", "dt": 1e-4},
        "controller": {
            "class_path": "neuro.control.LinearMPCController",
            "dt": 0.01,
            "artifact": str(artifact),
            "horizon": 4,
            "u_max": u_max,
            "w_y": 1.0,
            "w_u": 0.01,
            "formulation": formulation,
        },
    }

    sim = Simulation.from_config(config)
    sim.run()

    us = np.stack([np.atleast_1d(np.asarray(entry["u"], dtype=np.float64)) for entry in sim.logger.core_logs])
    assert us.shape[1] == 2  # n_controls
    assert np.all(np.abs(us) <= u_max + 1e-6)
