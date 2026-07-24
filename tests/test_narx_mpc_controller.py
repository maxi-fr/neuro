"""Tests for :class:`neuro.control.NarxMPCController` and :class:`LinearNarxMPCController`.

These MPCs pose the *output-lifted* (minimal-realization NARX) suppression problem: they lift
only the per-step model-space output ``y_k`` rather than the full shift-register window-state
that :class:`~neuro.control.MPCController` / :class:`~neuro.control.LinearMPCController` carry.
With a *synthetic* (random-weight) artifact these tests verify the machinery -- the solver
runs, box bounds are respected, the history windows roll, and the loop closes through the
``simulate`` orchestrator -- and, crucially, that on a linear model the output-lifted forms
agree with the shipped full-state controllers (they are the same optimization problem).
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
    LinearNarxMPCController,
    MPCController,
    MPCControllerLog,
    NarxMPCController,
)
from neuro.nn_predictor_casadi import NNSymbolicModel  # noqa: E402
from neuro.prediction import AutoregressivePredictor, MLPArtifact  # noqa: E402
from neuro.transforms import PCAProjection, Pipeline, Standardizer  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

    from simulate.controller import Controller

    from neuro.types import FloatArray

_SEED = 7


def _standardizer_pipeline(center: FloatArray, scale: FloatArray) -> Pipeline:
    """A single-step standardizer pipeline from raw ``center``/``scale`` arrays."""
    return Pipeline((Standardizer(center=np.asarray(center, np.float64), scale=np.asarray(scale, np.float64)),))


def _build_artifact(
    tmp_path: Path,
    *,
    depth: int = 2,
    n_y: int = 4,
    n_u: int = 3,
    horizon: int = 3,
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
    depth: int = 2,
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
        depth=depth,
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


def _drive(controller: Controller[Any], n_steps: int, n_channels: int) -> list[tuple[FloatArray, Any]]:
    """Feed ``n_steps`` random EEG measurements through ``update`` and collect the outputs."""
    rng = np.random.default_rng(_SEED + 1)
    out = []
    for k in range(n_steps):
        u, log = controller.update(k * controller.dt, ref=np.array([0.0]), x_hat=rng.standard_normal(n_channels))
        out.append((np.atleast_1d(np.asarray(u, dtype=np.float64)), log))
    return out


def test_warmup_emits_zero_until_window_filled(tmp_path: Path) -> None:
    """While the EEG window is still zero-padded, the MPC holds off and emits zeros."""
    n_y = 4
    model = NNSymbolicModel.from_artifact(_build_artifact(tmp_path, n_y=n_y))
    controller = NarxMPCController(dt=0.01, model=model, u_max=0.5, horizon=3)

    results = _drive(controller, n_steps=n_y, n_channels=model.n_channels)
    for u, log in results[: n_y - 1]:
        assert log.warmup
        np.testing.assert_array_equal(u, np.zeros(model.n_controls))
    assert not results[-1][1].warmup


def test_update_respects_bounds(tmp_path: Path) -> None:
    """Past warm-up, update returns a finite (n_controls,) control within the box bounds.

    Does not assert solver convergence: with a random-weight ReLU net the suppression landscape
    is non-smooth, so IPOPT may not certify KKT -- but the applied control is still finite and
    bound-respecting.
    """
    u_max = 0.5
    model = NNSymbolicModel.from_artifact(_build_artifact(tmp_path, n_y=4))
    controller = NarxMPCController(dt=0.01, model=model, u_max=u_max, horizon=3, w_y=1.0, w_u=0.0)

    u, _ = _drive(controller, n_steps=6, n_channels=model.n_channels)[-1]
    assert u.shape == (model.n_controls,)
    assert np.isfinite(u).all()
    assert np.all(np.abs(u) <= u_max + 1e-6)


def test_pure_effort_cost_yields_zero_control(tmp_path: Path) -> None:
    """With w_y=0 the cost is sum||u||^2, whose unconstrained minimizer is u=0."""
    model = NNSymbolicModel.from_artifact(_build_artifact(tmp_path, n_y=4))
    controller = NarxMPCController(dt=0.01, model=model, u_max=1.0, horizon=3, w_y=0.0, w_u=1.0)

    u, log = _drive(controller, n_steps=6, n_channels=model.n_channels)[-1]
    assert log.success
    np.testing.assert_allclose(u, np.zeros(model.n_controls), atol=1e-4)


def test_control_obeys_kirchhoff_current_law(tmp_path: Path) -> None:
    """The emitted control sums to zero across electrodes (Kirchhoff's current law).

    While stimulating (``w_y=1, w_u=0``) the output-lifted NLP still balances the montage's
    per-electrode currents so no net current is injected.
    """
    model = NNSymbolicModel.from_artifact(_build_artifact(tmp_path, n_y=4))
    controller = NarxMPCController(dt=0.01, model=model, u_max=5.0, horizon=3, w_y=1.0, w_u=0.0)

    u, _ = _drive(controller, n_steps=6, n_channels=model.n_channels)[-1]
    assert np.linalg.norm(u, ord=1) > 1e-3  # it actually stimulates
    assert abs(float(np.sum(u))) < 1e-6  # ... yet the currents sum to zero


def test_linear_control_obeys_kirchhoff_current_law(tmp_path: Path) -> None:
    """The output-lifted QP's emitted control sums to zero across electrodes (Kirchhoff)."""
    model = NNSymbolicModel.from_artifact(_build_artifact(tmp_path, depth=0, n_y=4, horizon=4))
    controller = LinearNarxMPCController(dt=0.01, model=model, u_max=5.0, horizon=4, w_y=1.0, w_u=0.0)

    u, log = _drive(controller, n_steps=6, n_channels=model.n_channels)[-1]
    assert log.success
    assert np.linalg.norm(u, ord=1) > 1e-3  # it actually stimulates
    assert abs(float(np.sum(u))) < 1e-8  # ... yet the currents sum to zero (exact for a QP)


def test_l1_penalty_drives_control_toward_zero(tmp_path: Path) -> None:
    """A large L1 control-effort penalty suppresses stimulation in the output-lifted NLP: u -> 0."""
    model = NNSymbolicModel.from_artifact(_build_artifact(tmp_path, n_y=4))
    base: dict[str, Any] = {"dt": 0.01, "model": model, "u_max": 5.0, "horizon": 3, "w_y": 1.0, "w_u": 0.0}

    u_l2 = _drive(NarxMPCController(w_u_l1=0.0, **base), n_steps=6, n_channels=model.n_channels)[-1][0]
    u_l1 = _drive(NarxMPCController(w_u_l1=1000.0, **base), n_steps=6, n_channels=model.n_channels)[-1][0]

    assert np.linalg.norm(u_l2, ord=1) > 1e-3  # baseline stimulates
    np.testing.assert_allclose(u_l1, np.zeros(model.n_controls), atol=1e-4)


def test_per_electrode_bounds_rejected_when_mismatched(tmp_path: Path) -> None:
    """A u_max length that is neither 1 nor n_controls is rejected."""
    model = NNSymbolicModel.from_artifact(_build_artifact(tmp_path, n_controls=2))
    with pytest.raises(ValueError, match="u_max has 3 entries but n_controls is 2"):
        NarxMPCController(dt=0.01, model=model, u_max=[1.0, 2.0, 3.0], horizon=3)


def test_from_config_loads_artifact_and_defaults_horizon(tmp_path: Path) -> None:
    """from_config loads the artifact path and defaults the horizon to the model's."""
    artifact = _build_artifact(tmp_path, horizon=5)
    controller = NarxMPCController.from_config({"dt": 0.01, "artifact": str(artifact), "u_max": 3.0})
    assert controller.dt == 0.01
    assert controller.horizon == 5  # defaulted from the artifact
    assert controller.n_controls == 2


def test_lifted_variable_count_is_output_sized(tmp_path: Path) -> None:
    """The decision vector is ``H*(n_controls + n_channels)`` -- outputs lifted, not the full state."""
    n_y, n_u, n_channels, n_controls, horizon = 4, 3, 2, 2, 5
    model = NNSymbolicModel.from_artifact(
        _build_artifact(tmp_path, n_y=n_y, n_u=n_u, n_channels=n_channels, n_controls=n_controls)
    )
    controller = NarxMPCController(dt=0.01, model=model, u_max=0.5, horizon=horizon)
    assert controller._lbx.size == horizon * (n_controls + n_channels)  # noqa: SLF001


def test_projection_lifts_latent_outputs_and_respects_bounds(tmp_path: Path) -> None:
    """With a PCA-projection artifact the lifted outputs are latent-sized (``H*k``) and the loop runs."""
    n_y, n_u, k, n_eeg, n_controls, horizon = 4, 3, 2, 6, 2, 3
    u_max = 0.5
    model = NNSymbolicModel.from_artifact(
        _build_projection_artifact(tmp_path, n_y=n_y, n_u=n_u, k=k, n_eeg=n_eeg, n_controls=n_controls)
    )
    controller = NarxMPCController(dt=0.01, model=model, u_max=u_max, horizon=horizon, w_y=1.0, w_u=0.0)

    assert model.n_channels == k
    assert model.n_eeg_channels == n_eeg
    # Lifted outputs live in the k-dim latent space, not the n_eeg raw space.
    assert controller._lbx.size == horizon * (n_controls + k)  # noqa: SLF001

    u, log = _drive(controller, n_steps=n_y + 2, n_channels=n_eeg)[-1]
    assert not log.warmup
    assert u.shape == (n_controls,)
    assert np.isfinite(u).all()
    assert np.all(np.abs(u) <= u_max + 1e-6)


def test_matches_full_state_mpc_on_linear_model(tmp_path: Path) -> None:
    """On a linear (depth-0) model the output-lifted NLP is the same convex problem as MPCController.

    Driven with the same measurement stream and a strictly-convex cost (w_u > 0), the two
    controllers must emit the same control at every step -- validating the output-lifted
    formulation against the shipped full-state multiple-shooting one.
    """
    artifact = _build_artifact(tmp_path, depth=0, n_y=4, n_u=3, horizon=4, n_channels=2, n_controls=2)
    lifted = NarxMPCController(dt=0.01, model=NNSymbolicModel.from_artifact(artifact), u_max=0.3, w_y=1.0, w_u=0.05)
    full = MPCController(dt=0.01, model=NNSymbolicModel.from_artifact(artifact), u_max=0.3, w_y=1.0, w_u=0.05)

    us_lifted = [u for u, _ in _drive(lifted, n_steps=8, n_channels=2)]
    us_full = [u for u, _ in _drive(full, n_steps=8, n_channels=2)]
    np.testing.assert_allclose(np.array(us_lifted), np.array(us_full), rtol=1e-5, atol=1e-5)


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
            "class_path": "neuro.control.NarxMPCController",
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
    us = np.stack([np.atleast_1d(np.asarray(entry["u"], dtype=np.float64)) for entry in sim.logger.core_logs])
    assert us.shape[1] == 2  # n_controls
    assert np.all(np.abs(us) <= u_max + 1e-6)


def test_update_returns_shared_log(tmp_path: Path) -> None:
    """update returns the shared MPCControllerLog past warm-up."""
    model = NNSymbolicModel.from_artifact(_build_artifact(tmp_path, n_y=4))
    controller = NarxMPCController(dt=0.01, model=model, u_max=1.0, horizon=3)
    _, log = _drive(controller, n_steps=6, n_channels=model.n_channels)[-1]
    assert isinstance(log, MPCControllerLog)


def test_linear_requires_depth_zero(tmp_path: Path) -> None:
    """A nonlinear (depth>0) predictor is rejected with a pointer to NarxMPCController."""
    model = NNSymbolicModel.from_artifact(_build_artifact(tmp_path, depth=2))
    with pytest.raises(ValueError, match="requires a linear predictor"):
        LinearNarxMPCController(dt=0.01, model=model, u_max=1.0, horizon=3)


def test_linear_pure_effort_yields_zero_control(tmp_path: Path) -> None:
    """With w_y=0 the QP minimizes sum||u||^2 -> u=0."""
    model = NNSymbolicModel.from_artifact(_build_artifact(tmp_path, depth=0, n_y=4))
    controller = LinearNarxMPCController(dt=0.01, model=model, u_max=1.0, horizon=3, w_y=0.0, w_u=1.0)

    u, log = _drive(controller, n_steps=6, n_channels=model.n_channels)[-1]
    assert log.success
    np.testing.assert_allclose(u, np.zeros(model.n_controls), atol=1e-4)


def test_linear_l1_penalty_yields_sparse_control(tmp_path: Path) -> None:
    """The L1 control-effort penalty soft-thresholds the output-lifted QP's controls to exact zeros.

    A 3-electrode montage is used so that per-channel sparsity is compatible with the Kirchhoff
    sum-to-zero constraint: with only 2 electrodes ``sum(u)=0`` forces ``[a, -a]``, which permits
    all-or-nothing sparsity but never a single zero.
    """
    n_y, n_ch, n_controls = 4, 2, 3
    model = NNSymbolicModel.from_artifact(
        _build_artifact(tmp_path, depth=0, n_y=n_y, horizon=4, n_channels=n_ch, n_controls=n_controls)
    )
    base: dict[str, Any] = {"dt": 0.01, "model": model, "u_max": 5.0, "horizon": 4, "w_y": 1.0, "w_u": 0.0}

    u_l2 = _drive(LinearNarxMPCController(w_u_l1=0.0, **base), n_y, n_ch)[-1][0]
    u_l1 = _drive(LinearNarxMPCController(w_u_l1=1.0, **base), n_y, n_ch)[-1][0]
    u_big = _drive(LinearNarxMPCController(w_u_l1=1000.0, **base), n_y, n_ch)[-1][0]

    assert int(np.count_nonzero(np.abs(u_l2) < 1e-6)) == 0  # pure L2 leaves every control nonzero
    assert int(np.count_nonzero(np.abs(u_l1) < 1e-6)) >= 1  # L1 zeroes out at least one control
    np.testing.assert_allclose(u_big, np.zeros(n_controls), atol=1e-6)  # large L1 -> all zero


@pytest.mark.parametrize("w_u_l1", [0.0, 0.5])
def test_linear_matches_sparse_linear_mpc(tmp_path: Path, w_u_l1: float) -> None:
    """The output-lifted QP agrees with LinearMPCController's full-state sparse QP on a linear model.

    The optional L1 control-effort penalty is epigraph-reformulated identically in both, so the
    equivalence holds with or without it.
    """
    artifact = _build_artifact(tmp_path, depth=0, n_y=4, n_u=3, horizon=4, n_channels=2, n_controls=2)
    lifted = LinearNarxMPCController(
        dt=0.01, model=NNSymbolicModel.from_artifact(artifact), u_max=0.3, w_y=1.0, w_u=0.05, w_u_l1=w_u_l1
    )
    full = LinearMPCController(
        dt=0.01,
        model=NNSymbolicModel.from_artifact(artifact),
        u_max=0.3,
        w_y=1.0,
        w_u=0.05,
        w_u_l1=w_u_l1,
        formulation="sparse",
    )

    us_lifted = [u for u, _ in _drive(lifted, n_steps=8, n_channels=2)]
    us_full = [u for u, _ in _drive(full, n_steps=8, n_channels=2)]
    np.testing.assert_allclose(np.array(us_lifted), np.array(us_full), rtol=1e-5, atol=1e-5)


def test_linear_matches_nonlinear_narx_on_linear_model(tmp_path: Path) -> None:
    """OSQP output-lifted QP and IPOPT output-lifted NLP coincide on a linear model."""
    artifact = _build_artifact(tmp_path, depth=0, n_y=4, n_u=3, horizon=4, n_channels=2, n_controls=2)
    qp = LinearNarxMPCController(dt=0.01, model=NNSymbolicModel.from_artifact(artifact), u_max=0.3, w_y=1.0, w_u=0.05)
    nlp = NarxMPCController(dt=0.01, model=NNSymbolicModel.from_artifact(artifact), u_max=0.3, w_y=1.0, w_u=0.05)

    us_qp = [u for u, _ in _drive(qp, n_steps=8, n_channels=2)]
    us_nlp = [u for u, _ in _drive(nlp, n_steps=8, n_channels=2)]
    np.testing.assert_allclose(np.array(us_qp), np.array(us_nlp), rtol=1e-5, atol=1e-5)


def test_linear_from_config_loads_artifact(tmp_path: Path) -> None:
    """from_config loads a linear artifact and defaults the horizon to the model's."""
    artifact = _build_artifact(tmp_path, depth=0, horizon=5)
    controller = LinearNarxMPCController.from_config(
        {"dt": 0.01, "artifact": str(artifact), "u_max": 3.0, "w_u_l1": 0.25}
    )
    assert controller.horizon == 5
    assert controller.n_controls == 2
    assert controller.w_u_l1 == 0.25


def test_linear_projection_respects_bounds(tmp_path: Path) -> None:
    """A linear projection artifact runs, lifts latent outputs, and respects the box bounds."""
    n_y, n_u, k, n_eeg, n_controls, horizon = 4, 3, 2, 6, 2, 3
    u_max = 0.5
    model = NNSymbolicModel.from_artifact(
        _build_projection_artifact(tmp_path, depth=0, n_y=n_y, n_u=n_u, k=k, n_eeg=n_eeg, n_controls=n_controls)
    )
    controller = LinearNarxMPCController(dt=0.01, model=model, u_max=u_max, horizon=horizon, w_y=1.0, w_u=0.01)
    assert controller._lbx.size == horizon * (n_controls + k)  # noqa: SLF001

    u, log = _drive(controller, n_steps=n_y + 2, n_channels=n_eeg)[-1]
    assert not log.warmup
    assert np.all(np.abs(u) <= u_max + 1e-6)
