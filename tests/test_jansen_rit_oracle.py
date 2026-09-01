from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from neuro.connectome import Connectome
from neuro.jansen_rit import JansenRitDynamics, JansenRitParams
from neuro.predictor.jansen_rit import JansenRitModel, enable_x64, lfp_jax, sigmoid_jax
from neuro.predictor.oracle import FullStateSensor, JansenRitOracleEstimator

enable_x64()

_DT = 1e-4
_SEED = 42
_D3 = np.array([[0, 2, 1], [1, 0, 2], [2, 1, 0]], dtype=np.int64)


def _toy_plant(n_nodes: int = 3, *, delay_ms: float = 0.1) -> tuple[JansenRitParams, Connectome]:
    """A small deterministic network whose conduction delays survive coarsening to the Predictor grid."""
    rng = np.random.default_rng(7)
    w = rng.uniform(0.0, 1.0, (n_nodes, n_nodes))
    np.fill_diagonal(w, 0.0)
    d = _D3[:n_nodes, :n_nodes].copy()
    params = JansenRitParams(A=np.array([3.25, 3.4, 3.6])[:n_nodes], sigma=0.0)
    conn = Connectome(
        K=0.6,
        weights=w,
        tract_lengths=(d * 5.0).astype(np.float64),
        centres=rng.uniform(-50.0, 50.0, (n_nodes, 3)),
        region_labels=np.array([f"reg_{i}" for i in range(n_nodes)], dtype=np.str_),
        hemispheres=np.zeros(n_nodes, dtype=bool),
        speed=50.0,
        delays=(d * delay_ms).astype(np.float64),
        region_index={f"reg_{i}": i for i in range(n_nodes)},
    )
    return params, conn


def _run_oracle(
    dyn: JansenRitDynamics,
    est: JansenRitOracleEstimator,
    n_plant_steps: int,
    u: np.ndarray,
) -> list[np.ndarray]:
    """Drive the Plant for ``n_plant_steps`` under a held ``u``, collecting one handover per Estimator step."""
    sensor = FullStateSensor(dt=dyn.dt)
    handovers = []
    for step in range(n_plant_steps):
        t = step * dyn.dt
        y_mea, _ = sensor.evaluate(t, dyn.x, u)
        z, _ = est.evaluate(t, y_mea, u)
        handovers.append(np.asarray(z).copy())
        dyn.evaluate(t, u)
    return handovers


def test_oracle_handover_reproduces_plant_step_exactly() -> None:
    """At Predictor dt == Plant dt the handover makes one knot reproduce the Plant bit for bit."""
    params, conn = _toy_plant(3)
    dyn = JansenRitDynamics(dt=_DT, params=params, conn=conn, seed=_SEED)
    model = JansenRitModel.from_plant(dyn)
    est = JansenRitOracleEstimator(dt=_DT, model=model)
    u = np.zeros(dyn.n_controls)

    for step in range(8):
        t = step * _DT
        z, _ = est.evaluate(t, dyn.x.reshape(-1), u)
        z_next = model.discrete_dynamics(jnp.asarray(z), jnp.zeros(model.m), t, model.knot_dt)
        dyn.evaluate(t, u)
        x_pred, _, k = model.unpack_state(z_next)
        np.testing.assert_allclose(np.asarray(x_pred), dyn.x, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(float(k), float(step + 1))


def test_oracle_history_is_written_on_the_predictor_grid() -> None:
    """The handover buffer holds ``S(y)`` sampled every Predictor step, not every Plant step."""
    stride = 10
    dt_pred = stride * _DT
    params, conn = _toy_plant(3, delay_ms=2.0)
    dyn = JansenRitDynamics(dt=_DT, params=params, conn=conn, seed=_SEED)
    model = JansenRitModel.from_plant_components(params, conn=conn, dt=dt_pred)
    assert model.max_history_len > 1
    est = JansenRitOracleEstimator(dt=dt_pred, model=model)

    states = []
    u = np.zeros(dyn.n_controls)
    for step in range(stride * (model.max_history_len + 2)):
        if step % stride == 0:
            states.append(dyn.x.copy())
        dyn.evaluate(step * _DT, u)
    dyn2 = JansenRitDynamics(dt=_DT, params=params, conn=conn, seed=_SEED)
    handovers = _run_oracle(dyn2, est, stride * (model.max_history_len + 2), u)

    z_last = handovers[-1]
    _, hist, k = model.unpack_state(jnp.asarray(z_last))
    j_last = len(states) - 1
    np.testing.assert_allclose(float(k), float(j_last))
    for j, x_j in enumerate(
        states[max(0, j_last - model.max_history_len + 1) : j_last + 1],
        start=max(0, j_last - model.max_history_len + 1),
    ):
        s_expected = np.asarray(sigmoid_jax(lfp_jax(jnp.asarray(x_j)), model.e0, model.v0, model.r))
        np.testing.assert_allclose(np.asarray(hist)[j % model.max_history_len], s_expected, rtol=1e-12, atol=1e-14)


def test_oracle_rollout_tracks_plant_at_coarse_predictor_step() -> None:
    """A substepped Rollout from the handover tracks the deterministic Plant within integration error."""
    stride = 10
    dt_pred = stride * _DT
    substeps = 2
    params, conn = _toy_plant(3, delay_ms=2.0)
    dyn = JansenRitDynamics(dt=_DT, params=params, conn=conn, seed=_SEED)
    model = JansenRitModel.from_plant_components(params, conn=conn, dt=dt_pred, substeps=substeps)
    est = JansenRitOracleEstimator(dt=dt_pred, model=model)

    settle = stride * 50
    u = np.zeros(dyn.n_controls)
    _run_oracle(dyn, est, settle, u)

    z, _ = est.evaluate(settle * _DT, dyn.x.reshape(-1), u)
    n_knots = 20
    y_pred = []
    z_k = jnp.asarray(z)
    for knot in range(n_knots):
        z_k = model.discrete_dynamics(z_k, jnp.zeros(model.m), knot * model.knot_dt, model.knot_dt)
        x_ode, _, _ = model.unpack_state(z_k)
        y_pred.append(float(lfp_jax(x_ode)[0]))

    y_true = []
    for step in range(n_knots * stride * substeps):
        dyn.evaluate((settle + step) * _DT, u)
        if (step + 1) % (stride * substeps) == 0:
            y_true.append(float(dyn.x[1, 0] - dyn.x[2, 0]))

    err = np.asarray(y_pred) - np.asarray(y_true)
    nrmse = float(np.sqrt(np.mean(err**2)) / np.std(y_true))
    assert nrmse < 0.05


def test_absorb_rejects_a_measurement_of_the_wrong_size() -> None:
    """State Absorption raises instead of silently leaving the Predictor state untouched."""
    params, conn = _toy_plant(3)
    model = JansenRitModel.from_plant_components(params, conn=conn, dt=_DT)
    state = model.initial_state()
    with pytest.raises(ValueError, match="cannot absorb"):
        model.absorb(state, np.zeros(5), np.zeros(model.m))


def test_absorb_round_trips_the_oracle_handover() -> None:
    """The packed handover is absorbed verbatim, so the Predictor starts the solve on the Plant's state."""
    params, conn = _toy_plant(3, delay_ms=2.0)
    dyn = JansenRitDynamics(dt=_DT, params=params, conn=conn, seed=_SEED)
    model = JansenRitModel.from_plant_components(params, conn=conn, dt=1e-3)
    est = JansenRitOracleEstimator(dt=1e-3, model=model)
    handovers = _run_oracle(dyn, est, 200, np.zeros(dyn.n_controls))

    absorbed = model.absorb(model.initial_state(), handovers[-1], np.zeros(model.m))
    np.testing.assert_allclose(absorbed, handovers[-1], rtol=0, atol=0)
