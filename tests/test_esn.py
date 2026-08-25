from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from neuro.artifacts import load_any_artifact
from neuro.esn import (
    ESNArtifact,
    generate_reservoir,
    harvest_normal_equations,
)
from neuro.esn_predictor_casadi import ESNSymbolicModel
from neuro.predictor.artifact import MLPArtifact
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from pathlib import Path

_SEED = 42


def _build_tiny_esn_artifact(
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
    """Save a tiny synthetic ESN artifact for testing."""
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

    art = ESNArtifact(
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
    )

    artifact_path = tmp_path / "esn_tiny"
    art.save(artifact_path)
    return artifact_path


def _build_tiny_mlp_artifact(
    tmp_path: Path,
    *,
    n_y: int = 4,
    n_u: int = 3,
    horizon: int = 4,
    n_channels: int = 2,
    n_controls: int = 2,
) -> Path:
    """Save a tiny synthetic MLP artifact for testing."""
    rng = np.random.default_rng(_SEED)
    in_size = n_y * n_channels + n_u * n_controls
    layers = (
        (rng.standard_normal((5, in_size)) / np.sqrt(in_size), rng.standard_normal(5)),
        (rng.standard_normal((n_channels, 5)) / np.sqrt(5), rng.standard_normal(n_channels)),
    )
    scalers = {
        "u_mean": rng.uniform(-1.0, 1.0, n_controls),
        "u_scale": rng.uniform(0.5, 2.0, n_controls),
        "y_mean": rng.uniform(-1.0, 1.0, n_channels),
        "y_scale": rng.uniform(0.5, 2.0, n_channels),
    }
    artifact = tmp_path / "mlp_tiny"
    MLPArtifact(
        layers=layers,
        activation="relu",
        n_y=n_y,
        n_u=n_u,
        horizon=horizon,
        n_channels=n_channels,
        n_controls=n_controls,
        dt=0.02,
        downsample=200,
        y_std=Standardizer(center=scalers["y_mean"], scale=scalers["y_scale"]),
        u_std=Standardizer(center=scalers["u_mean"], scale=scalers["u_scale"]),
    ).save(artifact)
    return artifact


def test_casadi_step_matches_numpy_step(tmp_path: Path) -> None:
    """ESNSymbolicModel.f_step equals ESNPredictor's free-running numpy step to 1e-10."""
    art_path = _build_tiny_esn_artifact(tmp_path)
    art = ESNArtifact.load(art_path)
    model = ESNSymbolicModel.from_checkpoint(art_path)

    rng = np.random.default_rng(_SEED)
    h = rng.standard_normal(art.reservoir_size)
    u = rng.standard_normal(art.n_controls)

    h_next_ca = np.asarray(model.f_step(h.reshape(-1, 1), u.reshape(-1, 1))).reshape(-1)
    h_next_np = art.predictor.step(h, art.u_std.transform(u))

    np.testing.assert_allclose(h_next_ca, h_next_np, atol=1e-10)


def test_casadi_rollout_matches_numpy_rollout(tmp_path: Path) -> None:
    """Chaining f_step/f_out over 50 steps equals ESNArtifact.rollout to 1e-10."""
    art_path = _build_tiny_esn_artifact(tmp_path, horizon=50)
    art = ESNArtifact.load(art_path)
    model = ESNSymbolicModel.from_checkpoint(art_path)

    rng = np.random.default_rng(_SEED + 1)
    h_init = rng.standard_normal(art.reservoir_size)
    u_future = rng.standard_normal((50, art.n_controls))

    y_preds_np = art.rollout(h_init, u_future)

    h_curr = h_init.reshape(-1, 1)
    y_preds_ca_list = []
    for t in range(len(u_future)):
        y_t = np.asarray(model.f_out(h_curr)).reshape(-1)
        y_preds_ca_list.append(y_t)
        h_curr = model.f_step(h_curr, u_future[t].reshape(-1, 1))

    y_preds_ca = np.array(y_preds_ca_list)
    np.testing.assert_allclose(y_preds_ca, y_preds_np, atol=1e-10)


def test_artifact_round_trip_preserves_weights(tmp_path: Path) -> None:
    """Save/load reproduces W_res (including sparsity pattern), W_in, W_out, standardizers, and metadata exactly."""
    art_path = _build_tiny_esn_artifact(tmp_path)
    art_orig = ESNArtifact.load(art_path)

    art_save_path = tmp_path / "roundtrip"
    art_orig.save(art_save_path)
    assert (tmp_path / "roundtrip.npz").exists()

    art_loaded = ESNArtifact.load(art_save_path)

    np.testing.assert_array_equal(art_orig.w_in, art_loaded.w_in)
    np.testing.assert_array_equal(art_orig.w_out, art_loaded.w_out)
    np.testing.assert_array_equal(art_orig.w_res.data, art_loaded.w_res.data)
    np.testing.assert_array_equal(art_orig.w_res.indices, art_loaded.w_res.indices)
    np.testing.assert_array_equal(art_orig.w_res.indptr, art_loaded.w_res.indptr)
    assert art_orig.w_res.shape == art_loaded.w_res.shape
    assert art_orig.meta == art_loaded.meta
    np.testing.assert_allclose(art_orig.y_std.center, art_loaded.y_std.center)
    np.testing.assert_allclose(art_orig.y_std.scale, art_loaded.y_std.scale)
    np.testing.assert_allclose(art_orig.u_std.center, art_loaded.u_std.center)
    np.testing.assert_allclose(art_orig.u_std.scale, art_loaded.u_std.scale)


def test_harvest_pairs_pre_update_state_with_target(tmp_path: Path) -> None:
    """Harvested rows hold h_t *before* it absorbs z[t], so the readout is one-step-ahead."""
    art_path = _build_tiny_esn_artifact(tmp_path, priming_steps=5)
    art = ESNArtifact.load(art_path)

    rng = np.random.default_rng(_SEED + 3)
    t_len = 40
    y_raw = rng.standard_normal((t_len, art.n_channels))
    u_raw = rng.standard_normal((t_len, art.n_controls))

    G, P = harvest_normal_equations(
        trajectories=[(u_raw, y_raw)],
        y_std=art.y_std,
        u_std=art.u_std,
        w_res=art.w_res,
        w_in=art.w_in,
        leak_rate=art.leak_rate,
        priming_steps=art.priming_steps,
        noise_sigma=0.0,
        seed=_SEED,
    )

    z = art.encode(y_raw)
    v = art.u_std.transform(u_raw)
    h_aug = np.ones((t_len - art.priming_steps, art.reservoir_size + 1))
    h = np.zeros(art.reservoir_size)
    for t in range(t_len):
        if t >= art.priming_steps:
            h_aug[t - art.priming_steps, : art.reservoir_size] = h
        h = art.predictor.absorb(h, z[t], v[t])

    np.testing.assert_allclose(G, h_aug.T @ h_aug, atol=1e-10)
    np.testing.assert_allclose(P, h_aug.T @ z[art.priming_steps :], atol=1e-10)


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
    art_path = _build_tiny_esn_artifact(tmp_path, spectral_radius=0.5, leak_rate=0.3, priming_steps=200)
    art = ESNArtifact.load(art_path)

    rng = np.random.default_rng(_SEED)
    y_hist = rng.standard_normal((art.priming_steps, art.n_channels))
    u_hist = rng.standard_normal((art.priming_steps, art.n_controls))

    h_from_zero = art.prime(y_hist, u_hist)

    # The same history driven from a random h0 instead of prime's h0 = 0.
    z = art.encode(y_hist)
    v = art.u_std.transform(u_hist)
    h = rng.standard_normal(art.reservoir_size)
    for t in range(art.priming_steps):
        h = art.predictor.absorb(h, z[t], v[t])

    assert float(np.linalg.norm(h - h_from_zero)) < 1e-5


def test_rollout_stays_bounded_over_horizon(tmp_path: Path) -> None:
    """Free-running 50 steps produces finite output bounded by a multiple of training-data scale."""
    art_path = _build_tiny_esn_artifact(tmp_path, horizon=50)
    art = ESNArtifact.load(art_path)

    rng = np.random.default_rng(_SEED)
    h_init = np.zeros(art.reservoir_size)
    u_future = rng.standard_normal((50, art.n_controls))

    preds = art.rollout(h_init, u_future)
    assert np.isfinite(preds).all()
    assert float(np.abs(preds).max()) < 100.0


def test_mlp_prime_rollout_matches_hand_rolled_windows(tmp_path: Path) -> None:
    """MLPArtifact.prime/rollout match a hand-indexed model-space AR loop bit-for-bit."""
    art_path = _build_tiny_mlp_artifact(tmp_path)
    art = MLPArtifact.load(art_path)

    rng = np.random.default_rng(_SEED + 2)
    t_len = 100
    y_raw = rng.standard_normal((t_len, art.n_channels))
    u_raw = rng.standard_normal((t_len, art.n_controls))

    n_y, n_u = art.n_y, art.n_u
    max_steps = 50
    t0 = max(n_y, n_u)

    z = art.encode(y_raw)
    w = art.u_std.transform(u_raw)
    y_hist = list(z[t0 - n_y : t0])
    for t in range(max_steps):
        y_win = np.array(y_hist[-n_y:])
        u_win = w[t0 + t - n_u : t0 + t]
        y_hist.append(art.forward_1step(y_win.reshape(-1), u_win.reshape(-1)))
    y_pred_manual = art.decode(np.array(y_hist[n_y:]))

    # prime / rollout seam
    state = art.prime(y_raw[t0 - art.priming_steps : t0], u_raw[t0 - art.priming_steps : t0])
    y_pred_new = art.rollout(state, u_raw[t0 : t0 + max_steps])

    np.testing.assert_allclose(y_pred_new, y_pred_manual, atol=1e-12)


def test_load_any_artifact_dispatches_by_model_type(tmp_path: Path) -> None:
    """load_any_artifact switches on meta['model_type'] and loads ESN or MLP correctly."""
    esn_path = _build_tiny_esn_artifact(tmp_path)
    mlp_path = _build_tiny_mlp_artifact(tmp_path)

    loaded_esn = load_any_artifact(esn_path)
    assert isinstance(loaded_esn, ESNArtifact)

    loaded_mlp = load_any_artifact(mlp_path)
    assert isinstance(loaded_mlp, MLPArtifact)

    bad_path = tmp_path / "bad_model"
    np.savez(bad_path.with_suffix(".npz"), meta=np.array(json.dumps({"model_type": "unknown"})))
    with pytest.raises(ValueError, match="unsupported model_type 'unknown'"):
        load_any_artifact(bad_path)
