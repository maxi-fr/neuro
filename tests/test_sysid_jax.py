"""Tests for the pure-JAX reduced-network system identification (neuro.sysid_jax).

The deterministic Jansen-Rit limit cycle has a multimodal loss landscape, so the
recovery test asserts what the method actually delivers: gradient *refinement* drives
down the windowed N-step MSE and the fitted model tracks held-out N-step-ahead
predictions closely, even where per-node parameters stay degenerate.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import optax

import neuro.sysid_jax as sj
from neuro.jansen_rit import JansenRitParams
from neuro.jansen_rit_jax import JRParamsJax, from_jansen_rit_params, rollout_jax

# float64 parity is mandatory; enable before any array is created.
jax.config.update("jax_enable_x64", True)  # noqa: FBT003

_DT = 1e-3  # coarser than the plant's 1e-4 so a short window resolves the ~10 Hz rhythm


def _toy(a_gain: list[float], *, mean_input: float = 160.0, k: float = 1.0, seed: int = 0) -> JRParamsJax:
    """A small oscillatory reduced model with delays and a random coarse-like gain."""
    n = len(a_gain)
    rng = np.random.default_rng(seed)
    w = rng.uniform(0.0, 0.4, (n, n))
    w = np.triu(w, 1)
    w = w + w.T
    d = (np.abs(np.subtract.outer(np.arange(n), np.arange(n)))).astype(np.int64)  # 0..n-1 step delays
    gain = rng.uniform(-1.0, 1.0, (n, n))
    base = JansenRitParams(
        A=np.asarray(a_gain, dtype=np.float64),
        mean_input=mean_input,
        sigma=0.0,
        w_weights=w,
        delay_steps=d,
        eeg_gain=gain,
        gamma=np.zeros((1, n)),
        K=k,
    )
    return from_jansen_rit_params(base, n)


def _random_controls(n_steps: int, n_nodes: int, *, amp: float = 40.0, seed: int = 1) -> np.ndarray:
    """A persistently-exciting (random) node-level control schedule, shape ``(n_steps, n_nodes)``."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-amp, amp, (n_steps, n_nodes))


def _synth(p: JRParamsJax, controls: np.ndarray) -> np.ndarray:
    """Deterministic EEG the model would produce under ``controls`` (the 'plant')."""
    n_steps = controls.shape[0]
    return np.asarray(sj.model_eeg(p, jnp.zeros((6, p.n_nodes)), jnp.asarray(controls), _DT, n_steps))


# --------------------------------------------------------------------------------------
# Plumbing
# --------------------------------------------------------------------------------------
def test_x64_enabled() -> None:
    assert jnp.zeros(1).dtype == jnp.float64


def test_loading_weighted_delays() -> None:
    rng = np.random.default_rng(3)
    comp = rng.uniform(-1.0, 1.0, (5, 76))
    full = rng.uniform(0.0, 30.0, (76, 76))
    full = (full + full.T) / 2.0
    np.fill_diagonal(full, 0.0)

    ds = sj.loading_weighted_delays(comp, full, _DT)
    assert ds.shape == (5, 5)
    assert ds.dtype == np.int64
    assert np.array_equal(ds, ds.T)
    assert np.all(np.diag(ds) == 0)
    assert np.all(ds >= 0)


def test_reduce_via_pca() -> None:
    rng = np.random.default_rng(4)
    node_output = rng.standard_normal((800, 76))
    gain = rng.uniform(-1.0, 1.0, (62, 76))
    full = rng.uniform(0.0, 30.0, (76, 76))
    full = (full + full.T) / 2.0
    np.fill_diagonal(full, 0.0)

    red = sj.reduce_via_pca(node_output, gain, full, _DT, n_components=8)
    assert red.gain.shape == (62, 8)
    assert red.components.shape == (8, 76)
    assert red.delay_steps.shape == (8, 8)
    assert np.array_equal(red.delay_steps, red.delay_steps.T)
    assert 0.0 < red.explained_variance <= 1.0


def test_w_reparam_symmetric_nonneg_roundtrip() -> None:
    p = _toy([3.4, 3.5, 3.6])
    bounds = sj.make_bounds(2.0)
    theta = sj.pack_theta(p, ["w_weights"], bounds)
    w = np.asarray(sj.build_params(theta, p, ["w_weights"], bounds).w_weights)
    assert np.allclose(w, w.T)
    assert np.all(w >= 0.0)
    assert np.allclose(np.diag(w), 0.0)
    # The reparam inverts the base weights (which lie within (0, w_max)).
    np.testing.assert_allclose(w, np.asarray(p.w_weights), rtol=1e-5, atol=1e-6)


def test_a_reparam_roundtrip() -> None:
    p = _toy([3.3, 3.5, 3.7])
    bounds = sj.make_bounds(2.0)
    theta = sj.pack_theta(p, ["A"], bounds)
    a = sj.build_params(theta, p, ["A"], bounds).A
    np.testing.assert_allclose(np.asarray(a), np.asarray(p.A), rtol=1e-6, atol=1e-7)


def test_rollout_tbptt_matches_rollout_jax() -> None:
    # stop_gradient at window boundaries must not change the forward values.
    p = _toy([3.4, 3.6, 3.8])
    controls = jnp.zeros((200, p.n_nodes))
    x_full, _ = rollout_jax(jnp.zeros((6, p.n_nodes)), controls, p, _DT)
    x_tbptt = sj.rollout_tbptt(jnp.zeros((6, p.n_nodes)), controls, p, _DT, window=37)
    np.testing.assert_allclose(np.asarray(x_tbptt), np.asarray(x_full), rtol=1e-10, atol=1e-12)


# --------------------------------------------------------------------------------------
# Windowed N-step MSE loss
# --------------------------------------------------------------------------------------
def test_window_starts_and_recording_windows() -> None:
    n_steps, horizon, stride = 1000, 150, 150
    starts = sj._window_starts(n_steps, horizon, stride)  # noqa: SLF001
    assert starts[0] == 0
    assert np.all(starts[1:] - starts[:-1] == stride)
    assert starts[-1] + horizon <= n_steps

    p = _toy([3.4, 3.5, 3.6])
    controls = _random_controls(n_steps, p.n_nodes)
    y = _synth(p, controls)
    burn_in = 0
    x0 = jnp.zeros((6, p.n_nodes))
    rec = sj._build_recording_windows(y, controls, x0, burn_in, horizon, stride)  # noqa: SLF001
    n_windows = starts.shape[0]
    assert rec.targets.shape == (n_windows, y.shape[0], horizon)
    # First window's target must line up with the raw array at t0=0.
    np.testing.assert_allclose(np.asarray(rec.targets[0]), y[:, :horizon])


def test_loss_self_consistency() -> None:
    """No resets -- a model's loss against its own continuously-rolled-out data is ~0."""
    p = _toy([3.4, 3.5, 3.6])
    n_steps, burn_in, horizon, stride = 900, 300, 150, 150
    controls = _random_controls(n_steps, p.n_nodes)
    y = _synth(p, controls)
    x0 = jnp.zeros((6, p.n_nodes))
    rec = sj._build_recording_windows(y, controls, x0, burn_in, horizon, stride)  # noqa: SLF001
    pred_loss = sj.make_pred_loss([rec], _DT, n_steps, burn_in)
    assert float(pred_loss(p)) < 1e-5


def test_pred_loss_gradient_finite() -> None:
    p = _toy([3.4, 3.5, 3.6])
    n_steps, burn_in, horizon = 900, 300, 150
    controls = _random_controls(n_steps, p.n_nodes)
    y = _synth(p, controls)
    x0 = jnp.zeros((6, p.n_nodes))
    rec = sj._build_recording_windows(y, controls, x0, burn_in, horizon, horizon)  # noqa: SLF001
    pred_loss = sj.make_pred_loss([rec], _DT, n_steps, burn_in)

    bounds = sj.make_bounds(2.0)
    theta = sj.pack_theta(p, ["A", "w_weights"], bounds)

    def loss(th: dict) -> jax.Array:
        return pred_loss(sj.build_params(th, p, ["A", "w_weights"], bounds))

    g = jax.grad(loss)(theta)
    assert np.all(np.isfinite(np.asarray(g["A"])))
    assert np.all(np.isfinite(np.asarray(g["w_weights"])))


# --------------------------------------------------------------------------------------
# Refinement
# --------------------------------------------------------------------------------------
def test_n_step_recovery() -> None:
    p_true = _toy([3.3, 3.5, 3.7, 3.9], mean_input=160.0, k=1.0, seed=2)
    n_steps, burn_in, horizon = 1800, 500, 200
    controls = _random_controls(n_steps, p_true.n_nodes, seed=7)
    y = _synth(p_true, controls)

    res = sj.identify(
        replace(p_true, A=jnp.full((p_true.n_nodes,), 3.25)),
        ["A", "w_weights"],
        y,
        _DT,
        pred_cfg=sj.PredConfig(horizon=horizon, burn_in=burn_in, stride=horizon),
        window=300,
        controls=controls,
        w_max=2.0,
        refine_cfg=sj.RefineConfig(
            steps=300,
            lr=optax.exponential_decay(init_value=0.08, transition_steps=20, decay_rate=0.7),
            clip=1.0,
        ),
    )
    # Loss drops sharply.
    assert res.history[-1] < 0.1 * res.history[0]

    # Held-out check: a fresh control draw, same true params vs the fitted params --
    # the N-step rollout should track the true trajectory closely (low NRMSE).
    held_out_controls = _random_controls(n_steps, p_true.n_nodes, seed=99)
    y_true_held = _synth(p_true, held_out_controls)
    y_fit_held = _synth(res.params, held_out_controls)
    rmse = np.sqrt(np.mean((y_fit_held[:, burn_in:] - y_true_held[:, burn_in:]) ** 2, axis=1))
    std = np.std(y_true_held[:, burn_in:], axis=1) + 1e-12
    nrmse = np.mean(rmse / std)
    assert nrmse < 0.3  # TODO: tune once run against real data
