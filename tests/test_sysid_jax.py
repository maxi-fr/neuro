"""Tests for the pure-JAX reduced-network system identification (neuro.sysid_jax).

The deterministic Jansen-Rit limit cycle has a multimodal loss landscape, so these
tests assert what the method actually delivers (mirroring Pille et al. 2025): grid
*exploration* localises the regime, and gradient *refinement* drives down the loss and
reproduces the EEG *statistics* (functional recovery), even where per-node parameters
stay degenerate.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from scipy.signal import welch

import neuro.sysid_jax as sj
from neuro.jansen_rit import JansenRitParams
from neuro.jansen_rit_jax import JRParamsJax, eeg_jax, from_jansen_rit_params, rollout_jax

# float64 parity is mandatory; enable before any array is created.
jax.config.update("jax_enable_x64", True)  # noqa: FBT003

_DT = 1e-3  # coarser than the plant's 1e-4 so a short window resolves the ~10 Hz rhythm
_N_STEPS = 1500
_WINDOW = 300
_STAT = sj.StatConfig(nperseg=512, band=(4.0, 40.0))


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


def _synth(p: JRParamsJax) -> np.ndarray:
    """Deterministic EEG the model would produce (the self-consistency 'plant')."""
    x_traj = sj.rollout_tbptt(jnp.zeros((6, p.n_nodes)), jnp.zeros((_N_STEPS, p.n_nodes)), p, _DT, _WINDOW)
    return np.asarray(eeg_jax(x_traj, p.eeg_gain))


def _log_psd(sig: np.ndarray) -> np.ndarray:
    freqs, pxx = welch(sig, fs=1.0 / _DT, nperseg=512, axis=1)
    mask = (freqs >= 4.0) & (freqs <= 40.0)
    return np.log(pxx[:, mask] + 1e-30)


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
    theta = sj.pack_theta(p, ["w_weights"], w_max=2.0)
    w = np.asarray(sj.build_params(theta, p, ["w_weights"], w_max=2.0).w_weights)
    assert np.allclose(w, w.T)
    assert np.all(w >= 0.0)
    assert np.allclose(np.diag(w), 0.0)
    # The reparam inverts the base weights (which lie within (0, w_max)).
    np.testing.assert_allclose(w, np.asarray(p.w_weights), rtol=1e-5, atol=1e-6)


def test_a_reparam_roundtrip() -> None:
    p = _toy([3.3, 3.5, 3.7])
    theta = sj.pack_theta(p, ["A"], w_max=2.0)
    a = sj.build_params(theta, p, ["A"], w_max=2.0).A
    np.testing.assert_allclose(np.asarray(a), np.asarray(p.A), rtol=1e-6, atol=1e-7)


def test_rollout_tbptt_matches_rollout_jax() -> None:
    # stop_gradient at window boundaries must not change the forward values.
    p = _toy([3.4, 3.6, 3.8])
    controls = jnp.zeros((200, p.n_nodes))
    x_full, _ = rollout_jax(jnp.zeros((6, p.n_nodes)), controls, p, _DT)
    x_tbptt = sj.rollout_tbptt(jnp.zeros((6, p.n_nodes)), controls, p, _DT, window=37)
    np.testing.assert_allclose(np.asarray(x_tbptt), np.asarray(x_full), rtol=1e-10, atol=1e-12)


def test_loss_gradient_finite() -> None:
    p = _toy([3.4, 3.5, 3.6])
    y = _synth(p)
    targets = sj.build_targets(y, _DT, _STAT)
    stat_loss = sj.make_stat_loss(
        targets,
        sj.LossWeights(spec=1.0, spatial=0.5),
        jnp.zeros((6, p.n_nodes)),
        jnp.zeros((_N_STEPS, p.n_nodes)),
        _DT,
        _WINDOW,
    )
    theta = sj.pack_theta(p, ["A", "w_weights"], w_max=2.0)

    def loss(th: dict) -> jax.Array:
        return stat_loss(sj.build_params(th, p, ["A", "w_weights"], w_max=2.0))

    g = jax.grad(loss)(theta)
    assert np.all(np.isfinite(np.asarray(g["A"])))
    assert np.all(np.isfinite(np.asarray(g["w_weights"])))


# --------------------------------------------------------------------------------------
# Exploration and refinement
# --------------------------------------------------------------------------------------
def test_explore_globals_finds_true_k() -> None:
    p = _toy([3.5, 3.5, 3.5], k=1.2, seed=1)
    y = _synth(p)
    targets = sj.build_targets(y, _DT, _STAT)
    stat_loss = sj.make_stat_loss(
        targets,
        sj.LossWeights(spec=1.0, spatial=0.5),
        jnp.zeros((6, p.n_nodes)),
        jnp.zeros((_N_STEPS, p.n_nodes)),
        _DT,
        _WINDOW,
    )
    overrides, losses = sj.explore_globals(p, stat_loss, k_grid=[0.0, 0.4, 0.8, 1.2, 1.6, 2.0])
    assert overrides["K"] == 1.2  # the true coupling sits at the grid minimum
    assert np.all(np.isfinite(losses))


def test_explore_globals_finds_true_a() -> None:
    p = _toy([4.5, 4.5, 4.5], k=1.0, seed=1)
    y = _synth(p)
    targets = sj.build_targets(y, _DT, _STAT)
    stat_loss = sj.make_stat_loss(
        targets,
        sj.LossWeights(spec=1.0, spatial=0.5),
        jnp.zeros((6, p.n_nodes)),
        jnp.zeros((_N_STEPS, p.n_nodes)),
        _DT,
        _WINDOW,
    )
    overrides, losses = sj.explore_globals(p, stat_loss, a_grid=[2.0, 3.25, 4.5, 6.0])
    assert overrides["A"] == 4.5  # the true global gain sits at the grid minimum
    assert np.all(np.isfinite(losses))


def test_functional_recovery() -> None:
    p_true = _toy([3.3, 3.5, 3.7, 3.9], mean_input=160.0, k=1.0, seed=2)
    y = _synth(p_true)
    res = sj.identify(
        replace(p_true, A=jnp.full((p_true.n_nodes,), 3.25)),
        ["A", "w_weights"],
        y,
        _DT,
        window=_WINDOW,
        w_max=2.0,
        weights=sj.LossWeights(spec=1.0, spatial=0.5),
        stat_cfg=_STAT,
        refine_cfg=sj.RefineConfig(steps=200, lr=2e-2, clip=1.0),
    )
    # Loss drops sharply and the fitted model reproduces the spectral statistics.
    assert res.history[-1] < 0.5 * res.history[0]
    y_fit = _synth(res.params)
    corr = np.corrcoef(_log_psd(y).ravel(), _log_psd(y_fit).ravel())[0, 1]
    assert corr > 0.9


def test_spec_shape_mode_is_scale_invariant() -> None:
    # The "shape" spectral term must ignore a constant per-channel rescale of the data:
    # the spectral-only loss of a fixed model is unchanged when the data is scaled.
    p = _toy([3.4, 3.6, 3.8])
    y = _synth(p)
    cfg = sj.StatConfig(nperseg=512, band=(4.0, 40.0), spec_mode="shape")
    weights = sj.LossWeights(spec=1.0, spatial=0.0, time=0.0)
    x0 = jnp.zeros((6, p.n_nodes))
    controls = jnp.zeros((_N_STEPS, p.n_nodes))

    loss1 = sj.make_stat_loss(sj.build_targets(y, _DT, cfg), weights, x0, controls, _DT, _WINDOW)(p)
    loss2 = sj.make_stat_loss(sj.build_targets(5.0 * y, _DT, cfg), weights, x0, controls, _DT, _WINDOW)(p)
    np.testing.assert_allclose(float(loss1), float(loss2), rtol=1e-9, atol=1e-10)
