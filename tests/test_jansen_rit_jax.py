"""Verify the JAX Jansen-Rit implementation against the numba/NumPy reference."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from neuro.jansen_rit import (
    JansenRitParams,
    _jr_rhs_jit,
    heun_step,
    sigmoid_jit,
    simulate_network,
)
from neuro.jansen_rit import (
    project_control as np_project_control,
)
from neuro.jansen_rit_jax import (
    JRParamsJax,
    coupling_from_history_jax,
    eeg_jax,
    from_jansen_rit_params,
    heun_step_det_jax,
    lfp_jax,
    project_control_jax,
    rhs_jax,
    rollout_jax,
    sigmoid_jax,
    simulate_jax,
)

# float64 parity is mandatory; enable before any array is created (no arrays exist at
# import time in jansen_rit_jax, so flipping the flag here is in time for every test).
jax.config.update("jax_enable_x64", True)  # noqa: FBT003

_DT = 1e-4
_SEED = 42

# Mixed delays so the circular-buffer gather is genuinely exercised.
_D3 = np.array([[0, 2, 1], [1, 0, 2], [2, 1, 0]], dtype=np.int64)


def _toy_params(n_nodes: int = 3, *, sigma: float = 0.0) -> JansenRitParams:
    """A small deterministic network with mixed delays for fast, TVB-free tests."""
    rng = np.random.default_rng(7)
    w = rng.uniform(0.0, 1.0, (n_nodes, n_nodes))
    np.fill_diagonal(w, 0.0)
    d = _D3[:n_nodes, :n_nodes].copy()
    gain = rng.uniform(-1.0, 1.0, (2, n_nodes))
    gamma = rng.uniform(-1.0, 1.0, (1, n_nodes))
    a_gain = np.array([3.25, 3.4, 3.6])[:n_nodes]
    return JansenRitParams(
        A=a_gain,
        sigma=sigma,
        w_weights=w,
        delay_steps=d,
        eeg_gain=gain,
        gamma=gamma,
        K=0.6,
    )


def test_x64_enabled() -> None:
    # The dtype is the real guard: without x64 every other tolerance silently fails.
    assert jnp.zeros(1).dtype == jnp.float64


def test_sigmoid_matches_numba() -> None:
    base = JansenRitParams()
    for val in np.linspace(-20.0, 20.0, 11):
        got = float(sigmoid_jax(val, base.e0, base.v0, base.r))
        want = sigmoid_jit(val, base.e0, base.v0, base.r)
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14)


def test_rhs_single_node_matches_numba() -> None:
    base = JansenRitParams()
    p = from_jansen_rit_params(base, 1)
    params_tuple = base.to_numba_tuple(1)
    rng = np.random.default_rng(_SEED)

    for _ in range(5):
        x = rng.standard_normal(6)
        dx_numba = _jr_rhs_jit(x, params_tuple, 0.0, 0.0)
        dx_jax = np.asarray(rhs_jax(jnp.asarray(x).reshape(6, 1), p, 0.0, 0.0)).flatten()
        np.testing.assert_allclose(dx_jax, dx_numba, rtol=1e-10, atol=1e-12)


def test_rhs_multi_node_matches_numba() -> None:
    n_nodes = 3
    base = _toy_params(n_nodes)
    p = from_jansen_rit_params(base, n_nodes)
    params_tuple = base.to_numba_tuple(n_nodes)
    rng = np.random.default_rng(_SEED + 2)

    for _ in range(5):
        x_2d = rng.standard_normal((6, n_nodes))
        c_val = rng.uniform(-1.0, 1.0, size=n_nodes)
        u_val = rng.uniform(-0.5, 0.5, size=n_nodes)

        dx_numba = _jr_rhs_jit(x_2d, params_tuple, c_val, u_val)
        dx_jax = np.asarray(rhs_jax(jnp.asarray(x_2d), p, jnp.asarray(c_val), jnp.asarray(u_val)))
        np.testing.assert_allclose(dx_jax, dx_numba, rtol=1e-10, atol=1e-12)


def test_coupling_with_delays_matches_reference() -> None:
    n_nodes = 3
    base = _toy_params(n_nodes)
    p = from_jansen_rit_params(base, n_nodes)
    w = np.asarray(base.w_weights)
    k_val = base.K
    length = int(_D3.max()) + 1

    rng = np.random.default_rng(_SEED + 6)
    history = rng.standard_normal((length, n_nodes))
    k = 5

    got = np.asarray(coupling_from_history_jax(jnp.asarray(history), jnp.asarray(k), p))

    want = np.zeros(n_nodes)
    for i in range(n_nodes):
        for j in range(n_nodes):
            want[i] += k_val * w[i, j] * history[(k - _D3[i, j]) % length, j]
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)


def test_heun_det_step_matches_numba() -> None:
    base = _toy_params(1, sigma=0.0)
    p = from_jansen_rit_params(base, 1)
    rng = np.random.default_rng(_SEED + 3)

    for _ in range(5):
        x = rng.standard_normal(6)
        u_val = float(rng.uniform(-0.5, 0.5))
        c_val = float(rng.uniform(-1.0, 1.0))

        x_numba = heun_step(x, u_val, base, _DT, xi=0.0, coupling=c_val)
        x_jax = np.asarray(heun_step_det_jax(jnp.asarray(x).reshape(6, 1), u_val, c_val, p, _DT)).flatten()
        np.testing.assert_allclose(x_jax, x_numba, rtol=1e-10, atol=1e-12)


def test_rollout_matches_simulate_network() -> None:
    n_steps = 20
    base = _toy_params(3, sigma=0.0)
    p = from_jansen_rit_params(base, 3)

    x0 = jnp.zeros((6, 3))
    controls = jnp.zeros((n_steps, 3))
    x_traj, _ = rollout_jax(x0, controls, p, _DT)

    _, x_ref = simulate_network(params=base, duration=n_steps * _DT, dt=_DT, seed=_SEED)
    np.testing.assert_allclose(np.asarray(x_traj), x_ref[:, :, 1:], rtol=1e-10, atol=1e-12)


def test_rollout_with_tes_matches_simulate_network() -> None:
    n_steps = 20
    base = _toy_params(3, sigma=0.0)
    p = from_jansen_rit_params(base, 3)
    gamma = np.asarray(base.gamma)  # (1, 3)
    u_amp = 50.0
    window = (5 * _DT, 12 * _DT)

    # Build the node-projected control schedule the same way simulate_network does.
    t_grid = np.arange(n_steps) * _DT
    mask = (t_grid >= window[0]) & (t_grid < window[1])
    u_node_row = np_project_control(np.array([u_amp]), gamma, 1)
    controls = np.zeros((n_steps, 3))
    controls[mask] = u_node_row

    x_traj, _ = rollout_jax(jnp.zeros((6, 3)), jnp.asarray(controls), p, _DT)

    _, x_ref = simulate_network(
        params=base, duration=n_steps * _DT, dt=_DT, seed=_SEED, u_hat_tES=u_amp, stim_window=window
    )
    np.testing.assert_allclose(np.asarray(x_traj), x_ref[:, :, 1:], rtol=1e-10, atol=1e-12)


def test_simulate_jax_matches_simulate_network() -> None:
    n_steps = 15
    base = _toy_params(3, sigma=0.0)
    p = from_jansen_rit_params(base, 3)

    t, x_traj = simulate_jax(p, duration=n_steps * _DT, dt=_DT)
    assert x_traj.shape == (6, 3, n_steps + 1)
    np.testing.assert_allclose(np.asarray(x_traj[:, :, 0]), np.zeros((6, 3)), atol=1e-12)
    np.testing.assert_allclose(np.asarray(t), np.arange(n_steps + 1) * _DT, rtol=1e-12)

    _, x_ref = simulate_network(params=base, duration=n_steps * _DT, dt=_DT, seed=_SEED)
    np.testing.assert_allclose(np.asarray(x_traj), x_ref, rtol=1e-10, atol=1e-12)


def test_lfp_eeg_project_control_match() -> None:
    n_nodes = 3
    rng = np.random.default_rng(_SEED + 4)
    x = rng.standard_normal((6, n_nodes))
    gain = rng.uniform(-1.0, 1.0, (2, n_nodes))

    np.testing.assert_allclose(np.asarray(lfp_jax(jnp.asarray(x))), x[1] - x[2], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(
        np.asarray(eeg_jax(jnp.asarray(x), jnp.asarray(gain))), gain @ (x[1] - x[2]), rtol=1e-10, atol=1e-12
    )

    n_elec = 2
    gamma = rng.uniform(0.0, 1.0, (n_elec, n_nodes))
    u = rng.uniform(-1.0, 1.0, n_elec)
    np.testing.assert_allclose(
        np.asarray(project_control_jax(jnp.asarray(u), jnp.asarray(gamma))),
        np_project_control(u, gamma, n_elec),
        rtol=1e-10,
        atol=1e-12,
    )


def test_grad_smoke() -> None:
    n_steps = 20
    base = _toy_params(3, sigma=0.0)
    p = from_jansen_rit_params(base, 3)
    x0 = jnp.zeros((6, 3))
    controls = jnp.zeros((n_steps, 3))

    def loss(pp: JRParamsJax) -> jax.Array:
        _, y = rollout_jax(x0, controls, pp, _DT)
        return jnp.sum(y**2)

    g = jax.grad(loss)(p)
    assert g.A.shape == (3,)
    assert np.all(np.isfinite(np.asarray(g.A)))
    assert g.w_weights.shape == (3, 3)
    assert np.all(np.isfinite(np.asarray(g.w_weights)))
    assert np.isfinite(float(g.K))
    assert g.mean_input.shape == (3,)
    assert np.all(np.isfinite(np.asarray(g.mean_input)))

    # Central finite-difference cross-check on A[0].
    eps = 1e-4
    p_plus = replace(p, A=p.A.at[0].add(eps))
    p_minus = replace(p, A=p.A.at[0].add(-eps))
    fd = (float(loss(p_plus)) - float(loss(p_minus))) / (2.0 * eps)
    np.testing.assert_allclose(float(g.A[0]), fd, rtol=1e-4, atol=1e-6)


def test_jit_runs() -> None:
    n_steps = 10
    base = _toy_params(3, sigma=0.0)
    p = from_jansen_rit_params(base, 3)
    x0 = jnp.zeros((6, 3))
    controls = jnp.zeros((n_steps, 3))

    x_ref, _ = rollout_jax(x0, controls, p, _DT)
    x_jit, _ = jax.jit(rollout_jax)(x0, controls, p, _DT)
    np.testing.assert_allclose(np.asarray(x_jit), np.asarray(x_ref), rtol=1e-10, atol=1e-12)


def test_stochastic_runs_and_differs_from_deterministic() -> None:
    n_steps = 1000
    base = _toy_params(1, sigma=200.0)
    p = from_jansen_rit_params(base, 1)

    _, x_stoch = simulate_jax(p, duration=n_steps * _DT, dt=_DT, key=jax.random.PRNGKey(0))
    _, x_det = simulate_jax(p, duration=n_steps * _DT, dt=_DT, key=None)

    assert np.all(np.isfinite(np.asarray(x_stoch)))
    assert not np.allclose(np.asarray(x_stoch), np.asarray(x_det))
