from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
from _predictor_reference import esn_absorb, esn_prime, esn_rollout, esn_step, mlp_forward, mlp_prime, mlp_rollout

from neuro.checkpoint import ESNCheckpoint, load_any, load_esn, load_mlp
from neuro.esn import generate_reservoir, harvest_normal_equations
from neuro.esn_predictor_casadi import ESNSymbolicModel
from neuro.predictor.module import AutoregressiveMLP
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from pathlib import Path

_SEED = 42


def _build_tiny_esn_checkpoint(
    tmp_path: Path,
    *,
    reservoir_size: int = 50,
    spectral_radius: float = 0.9,
    leak_rate: float = 0.1,
    density: float = 0.1,
    input_scaling: float = 0.1,
    priming_steps: int = 10,
    horizon: int = 50,
    n_channels: int = 2,
    n_controls: int = 2,
) -> Path:
    """Save a tiny synthetic ESN checkpoint for testing."""
    rng = np.random.default_rng(_SEED)
    in_dim = n_channels + n_controls + 1

    w_res, w_in = generate_reservoir(
        reservoir_size=reservoir_size,
        spectral_radius=spectral_radius,
        density=density,
        input_scaling=input_scaling,
        in_dim=in_dim,
        seed=_SEED,
    )

    w_out = rng.uniform(-0.1, 0.1, size=(n_channels, reservoir_size + 1))

    y_std = Standardizer(center=rng.uniform(-1.0, 1.0, n_channels), scale=rng.uniform(0.5, 2.0, n_channels))
    u_std = Standardizer(center=rng.uniform(-1.0, 1.0, n_controls), scale=rng.uniform(0.5, 2.0, n_controls))

    checkpoint_path = tmp_path / "esn_tiny"
    ESNCheckpoint(
        w_in=w_in,
        w_out=w_out,
        w_res=w_res,
        dt=0.02,
        downsample=200,
        horizon=horizon,
        reservoir_size=reservoir_size,
        leak_rate=leak_rate,
        spectral_radius=spectral_radius,
        priming_steps=priming_steps,
        input_scaling=input_scaling,
        density=density,
        noise_sigma=0.0,
        ridge_lambda=1e-3,
        seed=_SEED,
        y_std=y_std,
        u_std=u_std,
    ).save(checkpoint_path)
    return checkpoint_path


def _build_tiny_mlp_checkpoint(
    tmp_path: Path,
    *,
    n_y: int = 4,
    n_u: int = 3,
    horizon: int = 4,
    n_channels: int = 2,
    n_controls: int = 2,
) -> Path:
    """Save a tiny synthetic MLP checkpoint for testing."""
    rng = np.random.default_rng(_SEED)
    model = AutoregressiveMLP(
        n_y=n_y,
        n_u=n_u,
        horizon=horizon,
        n_channels=n_channels,
        n_controls=n_controls,
        hidden_size=5,
        depth=1,
        activation="relu",
        dt=0.02,
        y_std=Standardizer(center=rng.uniform(-1.0, 1.0, n_channels), scale=rng.uniform(0.5, 2.0, n_channels)),
        u_std=Standardizer(center=rng.uniform(-1.0, 1.0, n_controls), scale=rng.uniform(0.5, 2.0, n_controls)),
    )
    checkpoint = tmp_path / "mlp_tiny"
    model.save(checkpoint)
    return checkpoint


def test_casadi_step_matches_the_float64_reference(tmp_path: Path) -> None:
    """``ESNSymbolicModel.f_step`` equals the checkpoint's float64 free-running step to 1e-10."""
    ckpt_path = _build_tiny_esn_checkpoint(tmp_path)
    ckpt = load_esn(ckpt_path)
    model = ESNSymbolicModel.from_checkpoint(ckpt_path)

    rng = np.random.default_rng(_SEED)
    h = rng.standard_normal(ckpt.reservoir_size)
    u = rng.standard_normal(ckpt.n_controls)

    h_next_ca = np.asarray(model.f_step(h.reshape(-1, 1), u.reshape(-1, 1))).reshape(-1)
    h_next_np = esn_step(ckpt, h, ckpt.u_std.transform(u))

    np.testing.assert_allclose(h_next_ca, h_next_np, atol=1e-10)


def test_casadi_rollout_matches_the_float64_reference(tmp_path: Path) -> None:
    """Chaining f_step/f_out over 50 steps equals the checkpoint's float64 rollout to 1e-10."""
    ckpt_path = _build_tiny_esn_checkpoint(tmp_path, horizon=50)
    ckpt = load_esn(ckpt_path)
    model = ESNSymbolicModel.from_checkpoint(ckpt_path)

    rng = np.random.default_rng(_SEED + 1)
    h_init = rng.standard_normal(ckpt.reservoir_size)
    u_future = rng.standard_normal((50, ckpt.n_controls))

    y_preds_np = esn_rollout(ckpt, h_init, u_future)

    h_curr = h_init.reshape(-1, 1)
    y_preds_ca_list = []
    for t in range(len(u_future)):
        y_t = np.asarray(model.f_out(h_curr)).reshape(-1)
        y_preds_ca_list.append(y_t)
        h_curr = model.f_step(h_curr, u_future[t].reshape(-1, 1))

    y_preds_ca = np.array(y_preds_ca_list)
    np.testing.assert_allclose(y_preds_ca, y_preds_np, atol=1e-10)


def test_checkpoint_round_trip_preserves_weights(tmp_path: Path) -> None:
    """Save/load reproduces W_res (including sparsity pattern), W_in, W_out, standardizers, and metadata exactly."""
    ckpt_path = _build_tiny_esn_checkpoint(tmp_path)
    orig = load_esn(ckpt_path)

    save_path = tmp_path / "roundtrip"
    orig.save(save_path)
    assert (tmp_path / "roundtrip.npz").exists()

    loaded = load_esn(save_path)

    np.testing.assert_array_equal(orig.w_in, loaded.w_in)
    np.testing.assert_array_equal(orig.w_out, loaded.w_out)
    np.testing.assert_array_equal(orig.w_res.data, loaded.w_res.data)
    np.testing.assert_array_equal(orig.w_res.indices, loaded.w_res.indices)
    np.testing.assert_array_equal(orig.w_res.indptr, loaded.w_res.indptr)
    assert orig.w_res.shape == loaded.w_res.shape
    assert orig.leak_rate == loaded.leak_rate
    assert orig.priming_steps == loaded.priming_steps
    assert orig.ridge_lambda == loaded.ridge_lambda
    np.testing.assert_allclose(orig.y_std.center, loaded.y_std.center)
    np.testing.assert_allclose(orig.y_std.scale, loaded.y_std.scale)
    np.testing.assert_allclose(orig.u_std.center, loaded.u_std.center)
    np.testing.assert_allclose(orig.u_std.scale, loaded.u_std.scale)


def test_harvest_pairs_pre_update_state_with_target(tmp_path: Path) -> None:
    """Harvested rows hold h_t *before* it absorbs z[t], so the readout is one-step-ahead."""
    ckpt_path = _build_tiny_esn_checkpoint(tmp_path, priming_steps=5)
    ckpt = load_esn(ckpt_path)

    rng = np.random.default_rng(_SEED + 3)
    t_len = 40
    y_raw = rng.standard_normal((t_len, ckpt.n_channels))
    u_raw = rng.standard_normal((t_len, ckpt.n_controls))

    G, P = harvest_normal_equations(
        trajectories=[(u_raw, y_raw)],
        y_std=ckpt.y_std,
        u_std=ckpt.u_std,
        w_res=ckpt.w_res,
        w_in=ckpt.w_in,
        leak_rate=ckpt.leak_rate,
        priming_steps=ckpt.priming_steps,
        noise_sigma=0.0,
        seed=_SEED,
    )

    z = ckpt.y_std.transform(y_raw)
    v = ckpt.u_std.transform(u_raw)
    h_aug = np.ones((t_len - ckpt.priming_steps, ckpt.reservoir_size + 1))
    h = np.zeros(ckpt.reservoir_size)
    for t in range(t_len):
        if t >= ckpt.priming_steps:
            h_aug[t - ckpt.priming_steps, : ckpt.reservoir_size] = h
        h = esn_absorb(ckpt, h, z[t], v[t])

    np.testing.assert_allclose(G, h_aug.T @ h_aug, atol=1e-10)
    np.testing.assert_allclose(P, h_aug.T @ z[ckpt.priming_steps :], atol=1e-10)


def test_spectral_radius_is_scaled_to_rho() -> None:
    """Spectral radius of W_res equals target rho to 1e-8."""
    target_rho = 0.85
    w_res, _ = generate_reservoir(
        reservoir_size=100,
        spectral_radius=target_rho,
        density=0.1,
        input_scaling=0.1,
        in_dim=10,
        seed=_SEED,
    )
    vals = np.linalg.eigvals(w_res.toarray())
    abs_max = float(np.max(np.abs(vals)))
    np.testing.assert_allclose(abs_max, target_rho, atol=1e-8)


def test_priming_forgets_initial_state(tmp_path: Path) -> None:
    """Two different h0 primed on identical history converge after priming steps (echo state property)."""
    ckpt_path = _build_tiny_esn_checkpoint(tmp_path, spectral_radius=0.5, leak_rate=0.3, priming_steps=200)
    ckpt = load_esn(ckpt_path)

    rng = np.random.default_rng(_SEED)
    y_hist = rng.standard_normal((ckpt.priming_steps, ckpt.n_channels))
    u_hist = rng.standard_normal((ckpt.priming_steps, ckpt.n_controls))

    h_from_zero = esn_prime(ckpt, y_hist, u_hist)

    # The same history driven from a random h0 instead of prime's h0 = 0.
    z = ckpt.y_std.transform(y_hist)
    v = ckpt.u_std.transform(u_hist)
    h = rng.standard_normal(ckpt.reservoir_size)
    for t in range(ckpt.priming_steps):
        h = esn_absorb(ckpt, h, z[t], v[t])

    assert float(np.linalg.norm(h - h_from_zero)) < 1e-5


def test_rollout_stays_bounded_over_horizon(tmp_path: Path) -> None:
    """Free-running 50 steps produces finite output bounded by a multiple of training-data scale."""
    ckpt_path = _build_tiny_esn_checkpoint(tmp_path, horizon=50)
    ckpt = load_esn(ckpt_path)

    rng = np.random.default_rng(_SEED)
    h_init = np.zeros(ckpt.reservoir_size)
    u_future = rng.standard_normal((50, ckpt.n_controls))

    preds = esn_rollout(ckpt, h_init, u_future)
    assert np.isfinite(preds).all()
    assert float(np.abs(preds).max()) < 100.0


def test_mlp_prime_rollout_matches_hand_rolled_windows(tmp_path: Path) -> None:
    """The module's ``prime``/``rollout`` match a hand-indexed model-space AR loop bit-for-bit."""
    ckpt_path = _build_tiny_mlp_checkpoint(tmp_path)
    model = AutoregressiveMLP.load(ckpt_path)
    ckpt = load_mlp(ckpt_path)

    rng = np.random.default_rng(_SEED + 2)
    t_len = 100
    y_raw = rng.standard_normal((t_len, ckpt.n_channels))
    u_raw = rng.standard_normal((t_len, ckpt.n_controls))

    n_y, n_u = ckpt.n_y, ckpt.n_u
    max_steps = 50
    t0 = max(n_y, n_u)

    z = ckpt.y_std.transform(y_raw)
    w = ckpt.u_std.transform(u_raw)
    y_hist = list(z[t0 - n_y : t0])
    for t in range(max_steps):
        y_win = np.array(y_hist[-n_y:])
        u_win = w[t0 + t - n_u : t0 + t]
        y_hist.append(mlp_forward(np.concatenate([y_win.reshape(-1), u_win.reshape(-1)]), ckpt.layers, ckpt.activation))
    y_pred_manual = ckpt.y_std.inverse_transform(np.array(y_hist[n_y:]))

    # prime / rollout seam on the float32 module, versus the float64 hand-rolled loop.
    state = model.prime(y_raw[t0 - model.priming_steps : t0], u_raw[t0 - model.priming_steps : t0])
    y_pred_new = model.rollout(state, u_raw[t0 : t0 + max_steps])

    np.testing.assert_allclose(y_pred_new, y_pred_manual, atol=1e-5)


def test_load_any_dispatches_by_model_type(tmp_path: Path) -> None:
    """``load_any`` switches on ``model_type`` and reads ESN or MLP checkpoints correctly."""
    esn_path = _build_tiny_esn_checkpoint(tmp_path)
    mlp_path = _build_tiny_mlp_checkpoint(tmp_path)

    loaded_esn = load_any(esn_path)
    assert isinstance(loaded_esn, ESNCheckpoint)

    loaded_mlp = load_any(mlp_path)
    assert loaded_mlp.model_type == "mlp"

    bad_path = tmp_path / "bad_model"
    np.savez(bad_path.with_suffix(".npz"), meta=np.array(json.dumps({"model_type": "unknown"})))
    with pytest.raises(ValueError, match="unsupported model_type 'unknown'"):
        load_any(bad_path)
