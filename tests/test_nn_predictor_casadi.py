"""Verify the CasADi NN-predictor port against the JAX/Equinox reference."""

from __future__ import annotations

from typing import TYPE_CHECKING

import casadi as ca
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

# float64 parity is mandatory; enable before any array is created (Linear layers read
# this flag at construction time via default_floating_dtype()).
jax.config.update("jax_enable_x64", True)  # noqa: FBT003

from neuro.nn_predictor_casadi import NNSymbolicModel, _extract_mlp_layers, _mlp_forward_ca  # noqa: E402
from neuro.prediction import AutoregressivePredictor, MLPArtifact, NNPredictor, PredictionWindow  # noqa: E402

_SEED = 42

_CASES = [
    pytest.param(2, 2, 3, 5, 2, 2, 2, id="normal"),
    pytest.param(2, 2, 3, 5, 0, 2, 2, id="depth0"),
    pytest.param(1, 1, 3, 5, 2, 2, 2, id="window1"),
]


def _np2(result: object) -> np.ndarray:
    return np.array(ca.DM(result))


def _build_artifact(
    tmp_path: Path,
    n_y: int,
    n_u: int,
    horizon: int,
    hidden_size: int,
    depth: int,
    n_channels: int,
    n_controls: int,
) -> tuple[Path, eqx.nn.MLP, dict[str, np.ndarray]]:
    """Build and save a tiny artifact via ``MLPArtifact.save``, returning its path, MLP, and scalers."""
    rng = np.random.default_rng(_SEED)
    in_size = n_y * n_channels + n_u * n_controls
    out_size = n_channels
    mlp = eqx.nn.MLP(
        in_size=in_size,
        out_size=out_size,
        width_size=hidden_size,
        depth=depth,
        activation=jax.nn.relu,
        key=jax.random.PRNGKey(0),
    )
    wrapped = AutoregressivePredictor(model=mlp, n_y=n_y, n_u=n_u, horizon=horizon, C_y=n_channels, C_u=n_controls)

    scalers = {
        "u_mean": rng.uniform(-1.0, 1.0, n_controls),
        "u_scale": rng.uniform(0.5, 2.0, n_controls),
        "y_mean": rng.uniform(-1.0, 1.0, n_channels),
        "y_scale": rng.uniform(0.5, 2.0, n_channels),
    }

    artifact = tmp_path / "art"
    MLPArtifact(model=wrapped, dt=0.01, downsample=1, **scalers).save(artifact)

    return artifact, mlp, scalers


@pytest.mark.parametrize(("n_y", "n_u", "horizon", "hidden_size", "depth", "n_channels", "n_controls"), _CASES)
def test_mlp_forward_matches_jax(
    tmp_path: Path,
    n_y: int,
    n_u: int,
    horizon: int,
    hidden_size: int,
    depth: int,
    n_channels: int,
    n_controls: int,
) -> None:
    """The CasADi MLP forward pass matches the Equinox MLP bit-for-bit."""
    artifact, mlp, _ = _build_artifact(tmp_path, n_y, n_u, horizon, hidden_size, depth, n_channels, n_controls)
    mlp_artifact = MLPArtifact.load(artifact)

    in_size = n_y * n_channels + n_u * n_controls
    x_sym = ca.SX.sym("x", in_size, 1)
    layers = _extract_mlp_layers(mlp_artifact.model.model)
    fn = ca.Function("f", [x_sym], [_mlp_forward_ca(x_sym, layers)])

    rng = np.random.default_rng(_SEED + 1)
    for _ in range(5):
        x = rng.standard_normal(in_size)
        got = _np2(fn(x)).flatten()
        want = np.asarray(mlp(jnp.asarray(x)))
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize(("n_y", "n_u", "horizon", "hidden_size", "depth", "n_channels", "n_controls"), _CASES)
def test_single_step_matches_manual_scan_iteration(
    tmp_path: Path,
    n_y: int,
    n_u: int,
    horizon: int,
    hidden_size: int,
    depth: int,
    n_channels: int,
    n_controls: int,
) -> None:
    """f_step/f_out match one hand-written iteration of the JAX scan body."""
    artifact, mlp, scalers = _build_artifact(tmp_path, n_y, n_u, horizon, hidden_size, depth, n_channels, n_controls)
    model = NNSymbolicModel.from_artifact(artifact)

    rng = np.random.default_rng(_SEED + 2)
    y_w_raw = rng.standard_normal((n_y, n_channels))
    u_w_raw = rng.standard_normal((n_u, n_controls))
    u_curr = rng.standard_normal(n_controls)

    x0 = np.concatenate([y_w_raw.flatten(), u_w_raw.flatten()])
    x_next = _np2(model.f_step(x0, u_curr)).flatten()
    y_next_ca = _np2(model.f_out(x_next)).flatten()

    new_u_w = np.concatenate([u_w_raw[1:], u_curr[None, :]], axis=0)
    y_w_scaled = (y_w_raw - scalers["y_mean"]) / scalers["y_scale"]
    new_u_w_scaled = (new_u_w - scalers["u_mean"]) / scalers["u_scale"]
    mlp_in = np.concatenate([y_w_scaled.flatten(), new_u_w_scaled.flatten()])
    y_next_scaled = np.asarray(mlp(jnp.asarray(mlp_in)))
    y_next_raw_want = y_next_scaled * scalers["y_scale"] + scalers["y_mean"]
    new_y_w_want = np.concatenate([y_w_raw[1:], y_next_raw_want[None, :]], axis=0)
    x_next_want = np.concatenate([new_y_w_want.flatten(), new_u_w.flatten()])

    np.testing.assert_allclose(x_next, x_next_want, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(y_next_ca, y_next_raw_want, rtol=1e-10, atol=1e-12)


def test_output_slices_most_recent_row_not_oldest(tmp_path: Path) -> None:
    """f_out must return the LAST row of the y-window, not the first.

    Uses distinguishable per-row values so a transposed/reversed slice would be caught,
    unlike random data where it could accidentally still pass.
    """
    n_y, n_u, n_channels, n_controls = 3, 2, 2, 2
    artifact, _mlp, _scalers = _build_artifact(tmp_path, n_y, n_u, 3, 5, 2, n_channels, n_controls)
    model = NNSymbolicModel.from_artifact(artifact)

    y_w = np.arange(n_y * n_channels, dtype=np.float64).reshape(n_y, n_channels)
    u_w = np.zeros((n_u, n_controls))
    x = np.concatenate([y_w.flatten(), u_w.flatten()])

    got = _np2(model.f_out(x)).flatten()
    np.testing.assert_array_equal(got, y_w[-1])


@pytest.mark.parametrize(("n_y", "n_u", "horizon", "hidden_size", "depth", "n_channels", "n_controls"), _CASES)
def test_multistep_rollout_matches_nn_predictor(
    tmp_path: Path,
    n_y: int,
    n_u: int,
    horizon: int,
    hidden_size: int,
    depth: int,
    n_channels: int,
    n_controls: int,
) -> None:
    """Chaining f_step/f_out over several steps matches NNPredictor.predict()."""
    artifact, _mlp, _scalers = _build_artifact(tmp_path, n_y, n_u, horizon, hidden_size, depth, n_channels, n_controls)
    model = NNSymbolicModel.from_artifact(artifact)
    jax_predictor = NNPredictor.load(artifact)

    rng = np.random.default_rng(_SEED + 3)
    ctx = max(n_y, n_u) + 2
    n_steps = 2 * horizon  # spans >1 chunk, exercising NNPredictor.predict's chunking loop
    dt = jax_predictor.dt

    y_ctx = rng.standard_normal((n_channels, ctx))
    u_ctx = rng.standard_normal((ctx, n_controls))
    u_future = rng.standard_normal((n_steps, n_controls))

    window = PredictionWindow(y_ctx=y_ctx, u_ctx=u_ctx, u_future=u_future, dt_data=dt)
    y_pred_jax = jax_predictor.predict(window, horizon_s=n_steps * dt)

    y_w_raw = y_ctx.T[-n_y:]
    u_w_raw = u_ctx[-n_u:]
    x = np.concatenate([y_w_raw.flatten(), u_w_raw.flatten()])
    y_pred_ca = []
    for k in range(n_steps):
        x = _np2(model.f_step(x, u_future[k])).flatten()
        y_pred_ca.append(_np2(model.f_out(x)).flatten())
    y_pred_ca_arr = np.stack(y_pred_ca, axis=1)

    np.testing.assert_allclose(y_pred_ca_arr, y_pred_jax, rtol=1e-10, atol=1e-12)


def test_mlp_artifact_round_trip_preserves_exact_weights_and_meta(tmp_path: Path) -> None:
    """MLPArtifact.save/.load round-trips weights/biases and metadata bit-exactly."""
    n_y, n_u, horizon, hidden_size, depth, n_channels, n_controls = 2, 2, 3, 5, 2, 2, 2
    artifact, mlp, scalers = _build_artifact(tmp_path, n_y, n_u, horizon, hidden_size, depth, n_channels, n_controls)

    mlp_artifact = MLPArtifact.load(artifact)
    got_layers = _extract_mlp_layers(mlp_artifact.model.model)
    want_layers = _extract_mlp_layers(mlp)

    assert len(got_layers) == len(want_layers)
    for (got_w, got_b), (want_w, want_b) in zip(got_layers, want_layers, strict=True):
        np.testing.assert_array_equal(got_w, want_w)
        np.testing.assert_array_equal(got_b, want_b)

    assert mlp_artifact.n_y == n_y
    assert mlp_artifact.n_u == n_u
    assert mlp_artifact.horizon == horizon
    assert mlp_artifact.n_channels == n_channels
    assert mlp_artifact.n_controls == n_controls
    assert mlp_artifact.dt == pytest.approx(0.01)
    assert mlp_artifact.downsample == 1
    np.testing.assert_array_equal(mlp_artifact.u_mean, scalers["u_mean"])
    np.testing.assert_array_equal(mlp_artifact.u_scale, scalers["u_scale"])
    np.testing.assert_array_equal(mlp_artifact.y_mean, scalers["y_mean"])
    np.testing.assert_array_equal(mlp_artifact.y_scale, scalers["y_scale"])
