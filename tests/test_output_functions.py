from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from trajopt.costs.output import OutputCost
from trajopt.costs.quadratic import DiagonalCost
from trajopt.mpc import MPC
from trajopt.solvers.boxqp import BoxQP
from trajopt.transcription.ipopt import Ipopt
from trajopt.transcription.single_shooting import SingleShooting

from neuro.config import StftGeometry
from neuro.connectome import Connectome
from neuro.control.costs import (
    ExcludeInitialKnotState,
    ObservableFrameHingeCost,
    ObservableHingeCost,
    SpectralHingeCost,
    jax_compute_log_power_frames,
    jax_compute_observable_frames,
)
from neuro.control.mpc import (
    NullspaceReducedModel,
    build_observable_problem,
    build_waveform_problem,
)
from neuro.jansen_rit import JansenRitParams
from neuro.predictor.inference import ObservableMLPModel, WaveformMLPModel
from neuro.predictor.jansen_rit import JansenRitModel, build_jansen_rit_problem
from neuro.spectral import HealthyReference, ObservableEnvelope, PsdEnvelope

if TYPE_CHECKING:
    from pathlib import Path

_SEED = 42
_DT = 1e-4


def _build_synthetic_checkpoint(
    tmp_path: Path,
    *,
    is_observable: bool = False,
    n_y: int = 4,
    n_u: int = 3,
    n_channels: int = 2,
    n_controls: int = 2,
    n_values: int = 5,
) -> Path:
    rng = np.random.default_rng(_SEED)
    n_outputs = n_channels * n_values if is_observable else n_channels
    y_center = rng.uniform(-1.0, 1.0, n_outputs)
    y_scale = rng.uniform(0.5, 2.0, n_outputs)
    u_center = np.zeros(n_controls)
    u_scale = np.ones(n_controls)
    in_dim = n_y * n_outputs + n_u * n_controls
    w1 = rng.standard_normal((8, in_dim)) * 0.1
    b1 = np.zeros(8)
    w2 = rng.standard_normal((n_outputs, 8)) * 0.1
    b2 = np.zeros(n_outputs)

    y_center_j = jnp.asarray(y_center)
    y_scale_j = jnp.asarray(y_scale)
    u_center_j = jnp.asarray(u_center)
    u_scale_j = jnp.asarray(u_scale)
    weights = (jnp.asarray(w1), jnp.asarray(w2))
    biases = (jnp.asarray(b1), jnp.asarray(b2))

    if is_observable:
        geom = StftGeometry(n_segment=20, n_hop=5, band_hz=(4.0, 24.0), n_bin_pool=2)
        model = ObservableMLPModel(
            n_y=n_y,
            n_u=n_u,
            horizon=4,
            n_channels=n_channels,
            n_controls=n_controls,
            n_outputs=n_outputs,
            hidden_size=8,
            depth=1,
            activation="relu",
            residual=False,
            dt=0.05,
            downsample=1,
            y_center=y_center_j,
            y_scale=y_scale_j,
            u_center=u_center_j,
            u_scale=u_scale_j,
            weights=weights,
            biases=biases,
            geometry=geom,
        )
    else:
        model = WaveformMLPModel(
            n_y=n_y,
            n_u=n_u,
            horizon=4,
            n_channels=n_channels,
            n_controls=n_controls,
            n_outputs=n_outputs,
            hidden_size=8,
            depth=1,
            activation="relu",
            residual=False,
            dt=0.01,
            downsample=1,
            y_center=y_center_j,
            y_scale=y_scale_j,
            u_center=u_center_j,
            u_scale=u_scale_j,
            weights=weights,
            biases=biases,
        )
    stem = tmp_path / ("observable_model" if is_observable else "waveform_model")
    model.save(stem)
    return stem


def _toy_jansen_rit_model(n_nodes: int = 3, n_channels: int = 2) -> JansenRitModel:
    rng = np.random.default_rng(_SEED)
    w = rng.uniform(0.0, 1.0, (n_nodes, n_nodes))
    np.fill_diagonal(w, 0.0)
    d = np.zeros((n_nodes, n_nodes), dtype=np.int64)
    conn = Connectome(
        K=0.6,
        weights=w,
        tract_lengths=d.astype(np.float64),
        centres=rng.uniform(-50.0, 50.0, (n_nodes, 3)),
        region_labels=np.array([f"reg_{i}" for i in range(n_nodes)], dtype=np.str_),
        hemispheres=np.zeros(n_nodes, dtype=bool),
        speed=50.0,
        delays=d.astype(np.float64),
        region_index={f"reg_{i}": i for i in range(n_nodes)},
    )
    leadfield = rng.standard_normal((n_channels, n_nodes))
    params = JansenRitParams(sigma=0.0)
    return JansenRitModel.from_plant_components(params=params, conn=conn, leadfield=leadfield, dt=_DT)


def test_waveform_mlp_model_output(tmp_path: Path) -> None:
    stem = _build_synthetic_checkpoint(tmp_path, is_observable=False, n_channels=3, n_y=4)
    model = WaveformMLPModel.load(stem)
    assert model.p == 3

    rng = np.random.default_rng(_SEED)
    x = rng.standard_normal(model.n)
    y = model.output(jnp.asarray(x))
    assert y.shape == (3,)

    # Canonical decoding: newest block unstandardized
    newest = x[(model.n_y - 1) * model.n_channels : model.n_y * model.n_channels]
    expected_y = newest * np.asarray(model.y_scale) + np.asarray(model.y_center)
    np.testing.assert_allclose(np.asarray(y), expected_y, rtol=1e-10, atol=1e-12)


def test_observable_mlp_model_output(tmp_path: Path) -> None:
    stem = _build_synthetic_checkpoint(tmp_path, is_observable=True, n_channels=2, n_values=5, n_y=3)
    model = ObservableMLPModel.load(stem)
    assert model.p == 10

    rng = np.random.default_rng(_SEED)
    x = rng.standard_normal(model.n)
    y = model.output(jnp.asarray(x))
    assert y.shape == (10,)

    newest = x[(model.n_y - 1) * model.n_outputs : model.n_y * model.n_outputs]
    expected_y = newest * np.asarray(model.y_scale) + np.asarray(model.y_center)
    np.testing.assert_allclose(np.asarray(y), expected_y, rtol=1e-10, atol=1e-12)


def test_jansen_rit_model_output() -> None:
    n_nodes = 3
    n_channels = 2
    model = _toy_jansen_rit_model(n_nodes=n_nodes, n_channels=n_channels)
    assert model.p == n_channels

    rng = np.random.default_rng(_SEED)
    x_ode = rng.standard_normal((6, n_nodes))
    hist = model.seed_history(jnp.asarray(x_ode))
    x = model.pack_state(jnp.asarray(x_ode), hist, 0.0)

    y = model.output(x)
    assert y.shape == (n_channels,)

    lfp = x_ode[1] - x_ode[2]
    expected_y = np.asarray(model.eeg_gain) @ lfp
    np.testing.assert_allclose(np.asarray(y), expected_y, rtol=1e-10, atol=1e-12)


def test_nullspace_reduced_model_output(tmp_path: Path) -> None:
    stem = _build_synthetic_checkpoint(tmp_path, is_observable=False, n_channels=3, n_controls=3)
    base_model = WaveformMLPModel.load(stem)
    reduced = NullspaceReducedModel(base_model)
    assert reduced.p == base_model.p

    rng = np.random.default_rng(_SEED)
    x = jnp.asarray(rng.standard_normal(base_model.n))
    v = jnp.asarray(rng.standard_normal(reduced.m))

    y_red = reduced.output(x, v)
    y_base = base_model.output(x)
    np.testing.assert_allclose(np.asarray(y_red), np.asarray(y_base), rtol=1e-10, atol=1e-12)


def _numerical_output_state_jacobian(
    model: WaveformMLPModel | JansenRitModel,
    x0: jax.Array,
    eps: float = 1e-6,
) -> np.ndarray:
    """Evaluate numerical output-state Jacobian ``(p, n)`` via central finite differences."""
    assert model.p is not None
    jac_fd = np.zeros((model.p, model.n))
    for j in range(model.n):
        dx = np.zeros(model.n)
        dx[j] = eps
        yp = np.asarray(model.output(x0 + jnp.asarray(dx)))
        ym = np.asarray(model.output(x0 - jnp.asarray(dx)))
        jac_fd[:, j] = (yp - ym) / (2.0 * eps)
    return jac_fd


def test_output_state_jacobian_finite_differences(tmp_path: Path) -> None:
    stem = _build_synthetic_checkpoint(tmp_path, is_observable=False, n_channels=2)
    model = WaveformMLPModel.load(stem)

    rng = np.random.default_rng(_SEED)
    x0 = jnp.asarray(rng.standard_normal(model.n))
    assert model.p is not None
    jac_ad = np.asarray(model.output_state_jacobian(x0))
    assert jac_ad.shape == (model.p, model.n)

    jac_fd = _numerical_output_state_jacobian(model, x0)
    np.testing.assert_allclose(jac_ad, jac_fd, rtol=1e-6, atol=1e-7)


def test_jansen_rit_output_state_jacobian_finite_differences() -> None:
    model = _toy_jansen_rit_model(n_nodes=3, n_channels=2)
    rng = np.random.default_rng(_SEED)
    x0 = jnp.asarray(rng.standard_normal(model.n))
    assert model.p is not None
    jac_ad = np.asarray(model.output_state_jacobian(x0))
    assert jac_ad.shape == (model.p, model.n)

    jac_fd = _numerical_output_state_jacobian(model, x0)
    np.testing.assert_allclose(jac_ad, jac_fd, rtol=1e-6, atol=1e-7)


def test_output_cost_quadratic_tracking_equivalence(tmp_path: Path) -> None:
    stem = _build_synthetic_checkpoint(tmp_path, is_observable=False, n_channels=2, n_y=4)
    model = WaveformMLPModel.load(stem)
    horizon = 5
    w_y = 3.0
    w_u = 0.5
    y_target = np.array([1.2, -0.4])

    # Old way (state space tracking with padded Q)
    z_last = slice((model.n_y - 1) * model.n_channels, model.n_y * model.n_channels)
    Q_old = jnp.zeros(model.n).at[z_last].set(2.0 * w_y * model.y_scale**2 / horizon)
    target_state = (y_target - np.asarray(model.y_center)) / np.asarray(model.y_scale)
    xf_old = jnp.zeros(model.n).at[z_last].set(target_state)
    R = jnp.full(model.m, 2.0 * w_u / horizon)
    cost_old = DiagonalCost.tracking(Q_old, R, xf_old, jnp.zeros(model.m))

    # New way (OutputCost wrapping DiagonalCost on output space)
    Q_new = jnp.full(model.p, 2.0 * w_y / horizon)
    cost_output = DiagonalCost.tracking(Q_new, R, jnp.asarray(y_target), jnp.zeros(model.m))
    cost_new = OutputCost(model, cost_output)

    rng = np.random.default_rng(_SEED)
    for _ in range(5):
        x = jnp.asarray(rng.standard_normal(model.n))
        u = jnp.asarray(rng.standard_normal(model.m))
        val_old = float(cost_old.evaluate(x, u))
        val_new = float(cost_new.evaluate(x, u))
        np.testing.assert_allclose(val_new, val_old, rtol=1e-10, atol=1e-12)


def test_jansen_rit_output_cost_tracking_equivalence() -> None:
    n_nodes = 3
    n_channels = 2
    model = _toy_jansen_rit_model(n_nodes=n_nodes, n_channels=n_channels)
    horizon = 5
    w_y = 2.0
    w_u = 0.1
    y_target = np.array([0.7, -1.1])

    # Manual incumbent evaluation
    def manual_cost(x: jax.Array, u: jax.Array) -> float:
        x_ode = x[: 6 * n_nodes].reshape(6, n_nodes)
        lfp = x_ode[1] - x_ode[2]
        y = np.asarray(model.eeg_gain) @ lfp
        track = (w_y / horizon) * np.sum((y - y_target) ** 2)
        effort = (w_u / horizon) * np.sum(np.asarray(u) ** 2)
        return float(track + effort)

    # New OutputCost formulation
    Q = jnp.full(model.p, 2.0 * w_y / horizon)
    R = jnp.full(model.m, 2.0 * w_u / horizon)
    stage_tracking = DiagonalCost.tracking(Q, R, jnp.asarray(y_target), jnp.zeros(model.m))
    cost_new = OutputCost(model, stage_tracking)

    rng = np.random.default_rng(_SEED)
    for _ in range(5):
        x = jnp.asarray(rng.standard_normal(model.n))
        u = jnp.asarray(rng.standard_normal(model.m))
        val_manual = manual_cost(x, u)
        val_new = float(cost_new.evaluate(x, u))
        np.testing.assert_allclose(val_new, val_manual, rtol=1e-10, atol=1e-12)


def test_spectral_hinge_cost_output_equivalence(tmp_path: Path) -> None:
    """Verify SpectralHingeCost produces identical values when decoding through model.output."""
    n_channels = 2
    horizon = 20
    window = 10
    hop = 5
    stem = _build_synthetic_checkpoint(tmp_path, is_observable=False, n_channels=n_channels, n_y=4)
    model = WaveformMLPModel.load(stem)

    envelope = PsdEnvelope(
        power=np.ones((n_channels, window // 2 + 1)),
        fs=100.0,
        window=window,
        hop=hop,
    )
    cost = SpectralHingeCost(model, envelope, w_psd=2.5, horizon=horizon)

    rng = np.random.default_rng(_SEED + 1)
    X = jnp.asarray(rng.standard_normal((horizon, model.n)))
    U = jnp.zeros((horizon, model.m))
    t = jnp.zeros(horizon)

    val_cost = cost.stage_costs(X, U, t)

    newest = np.asarray(X)[..., (model.n_y - 1) * model.n_channels : model.n_y * model.n_channels]
    y_incumbent = newest * np.asarray(model.y_scale) + np.asarray(model.y_center)
    log_power = jax_compute_log_power_frames(jnp.asarray(y_incumbent), fs=100.0, window=window, hop=hop)
    log_excess = log_power - jnp.log(jnp.asarray(envelope.power)[None, :, 1:])
    expected_stage_cost = float(2.5 * jnp.mean(jnp.maximum(0.0, log_excess) ** 2))

    np.testing.assert_allclose(float(val_cost[0]), expected_stage_cost, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(np.asarray(val_cost[1:]), np.zeros(horizon - 1))


def test_observable_frame_hinge_cost_output_equivalence(tmp_path: Path) -> None:
    """Verify ObservableFrameHingeCost produces identical values when decoding through model.output."""
    n_channels = 2
    fs = 100.0
    geom = StftGeometry(n_segment=20, n_hop=5)
    horizon = 25
    stem = _build_synthetic_checkpoint(tmp_path, is_observable=False, n_channels=n_channels, n_y=4)
    model = WaveformMLPModel.load(stem)

    envelope = ObservableEnvelope(
        power=np.zeros((n_channels, geom.n_values(fs))),
        fs=fs,
        geometry=geom,
    )
    cost = ObservableFrameHingeCost(model, envelope, w_hinge=3.0, horizon=horizon)

    rng = np.random.default_rng(_SEED + 2)
    X = jnp.asarray(rng.standard_normal((horizon, model.n)))
    U = jnp.zeros((horizon, model.m))
    t = jnp.zeros(horizon)

    val_cost = cost.stage_costs(X, U, t)

    newest = np.asarray(X)[..., (model.n_y - 1) * model.n_channels : model.n_y * model.n_channels]
    y_incumbent = newest * np.asarray(model.y_scale) + np.asarray(model.y_center)
    frames = jax_compute_observable_frames(jnp.asarray(y_incumbent), geom, fs=fs)
    expected_stage_cost = float(3.0 * jnp.mean(jnp.maximum(0.0, frames - jnp.asarray(envelope.power)[None]) ** 2))

    np.testing.assert_allclose(float(val_cost[0]), expected_stage_cost, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(np.asarray(val_cost[1:]), np.zeros(horizon - 1))


def test_build_jansen_rit_problem_assembles_and_solves() -> None:
    """Verify build_jansen_rit_problem assembles and solves to optimality."""
    model = _toy_jansen_rit_model(n_nodes=3, n_channels=2)
    ref = HealthyReference(lfp_mean=np.zeros(3), eeg_mean=np.zeros(2))
    problem = build_jansen_rit_problem(model, horizon=4, u_max=1.0, w_y=1.0, w_u=0.1, reference=ref)
    assert problem.N == 5

    x0 = jnp.zeros(model.n)
    solver = SingleShooting(solver=Ipopt(options={"print_level": 0, "max_iter": 50}))
    mpc = MPC(problem, solver, x0=x0)
    res = mpc.solve()
    assert res.success
    assert np.all(np.isfinite(mpc.controls))
