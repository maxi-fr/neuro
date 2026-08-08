from __future__ import annotations

from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import numpy as np
import pytest

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


@pytest.mark.parametrize(("w_u_l1", "w_du"), [(0.0, 0.0), (0.5, 0.0), (0.0, 0.7), (0.5, 0.7)])
def test_sparse_dense_ipopt_equivalence(tmp_path: Path, w_u_l1: float, w_du: float) -> None:
    """sparse (OSQP), dense (qpOASES), and the IPOPT MPC solve the same QP on a linear model.

    Driving exactly ``n_y`` steps lets the window fill with identical measurements and all-zero
    controls, so every controller's first real solve sees the *same* ``x0``. With ``w_u>0`` the
    QP is strictly convex (unique optimum), so the applied control must agree across the two QP
    formulations (tightly) and the nonlinear IPOPT MPC on the same affine model (loosely). The
    optional L1 penalty ``w_u_l1`` (epigraph-reformulated) and the slew-rate penalty ``w_du``
    (which reads ``u_{-1}`` off the shared window-state parameter) must be applied identically by
    all three, so equivalence holds for every combination of them.
    """
    n_y, n_channels = 4, 2
    kw: dict[str, Any] = {
        "dt": 0.01,
        "u_max": 5.0,
        "horizon": 4,
        "w_y": 1.0,
        "w_u": 0.1,
        "w_u_l1": w_u_l1,
        "w_du": w_du,
    }
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
def test_control_obeys_kirchhoff_current_law(tmp_path: Path, formulation: str) -> None:
    """The emitted control sums to zero across electrodes (Kirchhoff's current law).

    Both QP formulations carry the sum-to-zero equality, so even while stimulating
    (``w_y=1, w_u=0``) the montage's currents balance exactly.
    """
    model = NNSymbolicModel.from_artifact(_build_artifact(tmp_path, n_y=4))
    controller = LinearMPCController(
        dt=0.01, model=model, u_max=5.0, horizon=4, w_y=1.0, w_u=0.0, formulation=formulation
    )

    u, log = _drive(controller, n_steps=6, n_channels=model.n_channels)[-1]
    assert log.success
    assert np.linalg.norm(u, ord=1) > 1e-3
    assert abs(float(np.sum(u))) < 1e-8


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
def test_l1_penalty_yields_sparse_control(tmp_path: Path, formulation: str) -> None:
    """The L1 control-effort penalty produces *exact* zeros (sparsity) that the L2 term never does.

    The active-set/OSQP QP soft-thresholds the controls: a moderate ``w_u_l1`` snaps at least one
    component to exactly zero (unlike the pure-quadratic ``w_u``, which only shrinks), and a large
    weight drives the whole control to zero. A 3-electrode montage is used so that per-channel
    sparsity is compatible with the Kirchhoff sum-to-zero constraint: with only 2 electrodes
    ``sum(u)=0`` forces ``[a, -a]``, allowing all-or-nothing sparsity but never a single zero.
    """
    n_y, n_ch, n_controls = 4, 2, 3
    model = NNSymbolicModel.from_artifact(_build_artifact(tmp_path, n_y=n_y, n_channels=n_ch, n_controls=n_controls))
    base: dict[str, Any] = {"dt": 0.01, "u_max": 5.0, "horizon": 4, "w_y": 1.0, "w_u": 0.0, "formulation": formulation}

    u_l2 = _drive(LinearMPCController(model=model, w_u_l1=0.0, **base), n_y, n_ch)[-1][0]
    u_l1 = _drive(LinearMPCController(model=model, w_u_l1=1.0, **base), n_y, n_ch)[-1][0]
    u_big = _drive(LinearMPCController(model=model, w_u_l1=1000.0, **base), n_y, n_ch)[-1][0]

    n_zero_l2 = int(np.count_nonzero(np.abs(u_l2) < 1e-6))
    n_zero_l1 = int(np.count_nonzero(np.abs(u_l1) < 1e-6))
    assert n_zero_l2 == 0
    assert n_zero_l1 > n_zero_l2
    np.testing.assert_allclose(u_big, np.zeros(n_controls), atol=1e-6)


@pytest.mark.parametrize("formulation", ["sparse", "dense"])
def test_rate_penalty_suppresses_chattering(tmp_path: Path, formulation: str) -> None:
    """``w_du`` penalises step-to-step control change, so the applied sequence stops chattering.

    Measured as the total variation of the *applied* controls across solves: it must fall
    monotonically as ``w_du`` rises, and a large weight must pin each control to its predecessor.
    The across-solve pinning is the part that matters -- it only holds because the penalty reads
    ``u_{-1}`` from the window-state parameter rather than starting each horizon afresh.
    """
    n_y, n_ch, n_controls = 4, 2, 3
    model = NNSymbolicModel.from_artifact(_build_artifact(tmp_path, n_y=n_y, n_channels=n_ch, n_controls=n_controls))
    base: dict[str, Any] = {"dt": 0.01, "u_max": 5.0, "horizon": 4, "w_y": 1.0, "w_u": 0.0, "formulation": formulation}
    n_steps = n_y + 12

    def applied(w_du: float) -> FloatArray:
        """Controls applied after the window has filled, shape ``(n_solves, n_controls)``."""
        driven = _drive(LinearMPCController(model=model, w_du=w_du, **base), n_steps, n_ch)
        return np.array([u for u, log in driven if not log.warmup])

    tv = {w: np.abs(np.diff(applied(w), axis=0)).sum() for w in (0.0, 1.0, 1000.0)}
    assert tv[0.0] > tv[1.0] > tv[1000.0]
    assert tv[1000.0] < 1e-3 * tv[0.0]


@pytest.mark.parametrize("formulation", ["sparse", "dense"])
def test_l1_zero_weight_leaves_solver_structure_unchanged(tmp_path: Path, formulation: str) -> None:
    """``w_u_l1=0`` adds no slacks/inequalities; a positive weight enlarges the decision vector."""
    model = NNSymbolicModel.from_artifact(_build_artifact(tmp_path, n_y=4))
    off = LinearMPCController(dt=0.01, model=model, u_max=1.0, horizon=4, formulation=formulation)
    on = LinearMPCController(dt=0.01, model=model, u_max=1.0, horizon=4, w_u_l1=1.0, formulation=formulation)

    assert off._has_g  # both formulations carry the sum-to-zero equality even with L1 off  # noqa: SLF001
    assert on._has_g  # a positive L1 weight additionally introduces the epigraph inequalities  # noqa: SLF001
    assert on._lbx.size > off._lbx.size  # the slack variables were appended  # noqa: SLF001


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
        {"dt": 0.01, "artifact": str(artifact), "u_max": 3.0, "formulation": "dense", "w_u_l1": 0.25}
    )
    assert controller.dt == 0.01
    assert controller.horizon == 5
    assert controller.n_controls == 2
    assert controller.formulation == "dense"
    assert controller.w_u_l1 == 0.25


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
    assert model.state_shape[0] == n_y * k + n_u * n_controls

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

    assert sim.logger is not None
    us = np.stack([np.atleast_1d(np.asarray(entry["u"], dtype=np.float64)) for entry in sim.logger.core_logs])
    assert us.shape[1] == 2
    assert np.all(np.abs(us) <= u_max + 1e-6)
