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

    from neuro.types import FloatArray

# float64 parity is mandatory; enable before any array is created (Linear layers read
# this flag at construction time via default_floating_dtype()).
jax.config.update("jax_enable_x64", True)  # noqa: FBT003

from neuro.nn_predictor_casadi import NNSymbolicModel, _extract_mlp_layers, _mlp_forward_ca  # noqa: E402
from neuro.prediction import (  # noqa: E402
    AutoregressivePredictor,
    MLPArtifact,
    get_activation,
)
from neuro.transforms import PCAProjection, Pipeline, Standardizer  # noqa: E402


def _standardizer_pipeline(center: FloatArray, scale: FloatArray) -> Pipeline:
    """A single-step standardizer pipeline from raw ``center``/``scale`` arrays."""
    return Pipeline((Standardizer(center=np.asarray(center, np.float64), scale=np.asarray(scale, np.float64)),))


_SEED = 42

_CASES = [
    pytest.param(2, 2, 3, 5, 2, 2, 2, "relu", id="normal"),
    pytest.param(2, 2, 3, 5, 2, 2, 2, "tanh", id="tanh"),
    pytest.param(2, 2, 3, 5, 2, 2, 2, "softplus", id="softplus"),
    pytest.param(2, 2, 3, 5, 0, 2, 2, "relu", id="depth0"),
    pytest.param(1, 1, 3, 5, 2, 2, 2, "relu", id="window1"),
]


def _np2(result: object) -> FloatArray:
    return np.array(ca.DM(result))


def _jax_horizon_rollout(art: MLPArtifact, y_ctx: FloatArray, u_ctx: FloatArray, u_future: FloatArray) -> FloatArray:
    """Reference EEG rollout over one ``horizon`` via the JAX ``AutoregressivePredictor``.

    Mirrors what the CasADi ``f_step``/``f_out`` chain must reproduce: encode the raw EEG
    context into model space, run the model's native ``horizon``-step scan on standardized
    controls, then decode back to raw EEG. Returns shape ``(n_eeg_channels, horizon)``.
    """
    n_y, n_u, horizon = art.n_y, art.n_u, art.horizon
    y_past = art.encode(y_ctx.T[-n_y:])
    u_past_s = art.u_pipeline.transform(u_ctx[-n_u:])
    u_fut_s = art.u_pipeline.transform(u_future[:horizon])
    x_in = np.concatenate([y_past.flatten(), u_past_s.flatten(), u_fut_s.flatten()])
    y_model = np.asarray(art.model(jnp.asarray(x_in))).reshape(horizon, art.n_channels)
    return art.decode(y_model).T


def _build_artifact(
    tmp_path: Path,
    n_y: int,
    n_u: int,
    horizon: int,
    hidden_size: int,
    depth: int,
    n_channels: int,
    n_controls: int,
    activation: str = "relu",
) -> tuple[Path, eqx.nn.MLP, dict[str, FloatArray]]:
    """Build and save a tiny artifact via ``MLPArtifact.save``, returning its path, MLP, and scalers."""
    rng = np.random.default_rng(_SEED)
    in_size = n_y * n_channels + n_u * n_controls
    out_size = n_channels
    mlp = eqx.nn.MLP(
        in_size=in_size,
        out_size=out_size,
        width_size=hidden_size,
        depth=depth,
        activation=get_activation(activation),
        key=jax.random.PRNGKey(0),
    )
    wrapped = AutoregressivePredictor(
        model=mlp,
        n_y=n_y,
        n_u=n_u,
        horizon=horizon,
        n_channels=n_channels,
        n_controls=n_controls,
        activation=activation,
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

    return artifact, mlp, scalers


def _build_projection_artifact(
    tmp_path: Path,
    n_y: int,
    n_u: int,
    horizon: int,
    hidden_size: int,
    depth: int,
    k: int,
    n_eeg: int,
    n_controls: int,
    activation: str = "relu",
) -> Path:
    """Save a tiny artifact whose model runs in a ``k``-dim PCA latent space; return its path.

    The y-pipeline standardizes the ``n_eeg`` raw EEG channels, then projects onto ``k`` latent
    components via a fixed orthonormal basis ``(k, n_eeg)`` plus mean (standardize-then-project),
    so model space is the ``k``-dimensional latent.
    """
    rng = np.random.default_rng(_SEED + 5)
    q, _ = np.linalg.qr(rng.standard_normal((n_eeg, n_eeg)))
    basis = np.ascontiguousarray(q[:, :k].T)  # orthonormal rows
    mean = rng.standard_normal(n_eeg)

    mlp = eqx.nn.MLP(
        in_size=n_y * k + n_u * n_controls,
        out_size=k,
        width_size=hidden_size,
        depth=depth,
        activation=get_activation(activation),
        key=jax.random.PRNGKey(0),
    )
    wrapped = AutoregressivePredictor(
        model=mlp, n_y=n_y, n_u=n_u, horizon=horizon, n_channels=k, n_controls=n_controls, activation=activation
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


@pytest.mark.parametrize(
    ("n_y", "n_u", "horizon", "hidden_size", "depth", "n_channels", "n_controls", "activation"), _CASES
)
def test_mlp_forward_matches_jax(
    tmp_path: Path,
    n_y: int,
    n_u: int,
    horizon: int,
    hidden_size: int,
    depth: int,
    n_channels: int,
    n_controls: int,
    activation: str,
) -> None:
    """The CasADi MLP forward pass matches the Equinox MLP bit-for-bit."""
    artifact, mlp, _ = _build_artifact(
        tmp_path, n_y, n_u, horizon, hidden_size, depth, n_channels, n_controls, activation
    )
    mlp_artifact = MLPArtifact.load(artifact)

    in_size = n_y * n_channels + n_u * n_controls
    x_sym = ca.SX.sym("x", in_size, 1)
    layers = _extract_mlp_layers(mlp_artifact.model.model)
    fn = ca.Function("f", [x_sym], [_mlp_forward_ca(x_sym, layers, activation)])

    rng = np.random.default_rng(_SEED + 1)
    for _ in range(5):
        x = rng.standard_normal(in_size)
        got = _np2(fn(x)).flatten()
        want = np.asarray(mlp(jnp.asarray(x)))
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize(
    ("n_y", "n_u", "horizon", "hidden_size", "depth", "n_channels", "n_controls", "activation"), _CASES
)
def test_single_step_matches_manual_scan_iteration(
    tmp_path: Path,
    n_y: int,
    n_u: int,
    horizon: int,
    hidden_size: int,
    depth: int,
    n_channels: int,
    n_controls: int,
    activation: str,
) -> None:
    """f_step/f_out match one hand-written iteration of the JAX scan body."""
    artifact, mlp, scalers = _build_artifact(
        tmp_path, n_y, n_u, horizon, hidden_size, depth, n_channels, n_controls, activation
    )
    model = NNSymbolicModel.from_artifact(artifact)

    rng = np.random.default_rng(_SEED + 2)
    y_w = rng.standard_normal((n_y, n_channels))  # model-space y-window (no projection -> standardized channels)
    u_w_raw = rng.standard_normal((n_u, n_controls))
    u_curr = rng.standard_normal(n_controls)

    x0 = np.concatenate([y_w.flatten(), u_w_raw.flatten()])
    x_next = _np2(model.f_step(x0, u_curr)).flatten()
    y_next_ca = _np2(model.f_out(x_next)).flatten()

    new_u_w = np.concatenate([u_w_raw[1:], u_curr[None, :]], axis=0)
    new_u_w_scaled = (new_u_w - scalers["u_mean"]) / scalers["u_scale"]
    # The y-window is already model space, so it feeds the MLP unscaled; only controls are standardized.
    mlp_in = np.concatenate([y_w.flatten(), new_u_w_scaled.flatten()])
    y_next_model = np.asarray(mlp(jnp.asarray(mlp_in)))
    new_y_w_want = np.concatenate([y_w[1:], y_next_model[None, :]], axis=0)
    x_next_want = np.concatenate([new_y_w_want.flatten(), new_u_w.flatten()])
    # f_out decodes model space to raw EEG (no projection -> inverse standardization).
    y_next_raw_want = y_next_model * scalers["y_scale"] + scalers["y_mean"]

    np.testing.assert_allclose(x_next, x_next_want, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(y_next_ca, y_next_raw_want, rtol=1e-10, atol=1e-12)


def test_single_step_matches_jax_with_global_scalar_scalers(tmp_path: Path) -> None:
    """f_step matches the JAX reference when scalers are global scalars (shape (1,)).

    ``global_scaling=True`` trains a single shared mean/scale rather than a per-channel
    vector; the CasADi port must broadcast it across all channels just as numpy does on
    the JAX side, not tile it by ``n_y``/``n_u`` only.
    """
    n_y, n_u, horizon, n_channels, n_controls = 3, 2, 4, 5, 2
    in_size = n_y * n_channels + n_u * n_controls
    mlp = eqx.nn.MLP(
        in_size=in_size, out_size=n_channels, width_size=5, depth=2, activation=jax.nn.relu, key=jax.random.PRNGKey(0)
    )
    wrapped = AutoregressivePredictor(
        model=mlp, n_y=n_y, n_u=n_u, horizon=horizon, n_channels=n_channels, n_controls=n_controls, activation="relu"
    )
    scalers = {  # global (scalar) statistics, as produced by global_scaling=True
        "u_mean": np.array([0.3]),
        "u_scale": np.array([1.7]),
        "y_mean": np.array([-0.9]),
        "y_scale": np.array([2.5]),
    }
    artifact = tmp_path / "art"
    MLPArtifact(
        model=wrapped,
        dt=0.01,
        downsample=1,
        y_pipeline=_standardizer_pipeline(scalers["y_mean"], scalers["y_scale"]),
        u_pipeline=_standardizer_pipeline(scalers["u_mean"], scalers["u_scale"]),
    ).save(artifact)
    model = NNSymbolicModel.from_artifact(artifact)

    rng = np.random.default_rng(_SEED + 9)
    y_w = rng.standard_normal((n_y, n_channels))  # model-space y-window
    u_w_raw = rng.standard_normal((n_u, n_controls))
    u_curr = rng.standard_normal(n_controls)

    x0 = np.concatenate([y_w.flatten(), u_w_raw.flatten()])
    x_next = _np2(model.f_step(x0, u_curr)).flatten()
    y_next_ca = _np2(model.f_out(x_next)).flatten()

    new_u_w = np.concatenate([u_w_raw[1:], u_curr[None, :]], axis=0)
    # The y-window is already model space; the global scalar broadcasts across all channels for u and the decode.
    mlp_in = np.concatenate([y_w.flatten(), ((new_u_w - scalers["u_mean"]) / scalers["u_scale"]).flatten()])
    y_next_raw_want = np.asarray(mlp(jnp.asarray(mlp_in))) * scalers["y_scale"] + scalers["y_mean"]
    np.testing.assert_allclose(y_next_ca, y_next_raw_want, rtol=1e-10, atol=1e-12)


def test_output_slices_most_recent_row_not_oldest(tmp_path: Path) -> None:
    """f_out must return the LAST row of the y-window, not the first.

    Uses distinguishable per-row values so a transposed/reversed slice would be caught,
    unlike random data where it could accidentally still pass.
    """
    n_y, n_u, n_channels, n_controls = 3, 2, 2, 2
    artifact, _mlp, _scalers = _build_artifact(tmp_path, n_y, n_u, 3, 5, 2, n_channels, n_controls)
    model = NNSymbolicModel.from_artifact(artifact)
    art = MLPArtifact.load(artifact)

    y_w = np.arange(n_y * n_channels, dtype=np.float64).reshape(n_y, n_channels)
    u_w = np.zeros((n_u, n_controls))
    x = np.concatenate([y_w.flatten(), u_w.flatten()])

    # f_out decodes the LAST model-space row to raw EEG; distinguishable per-row values catch a
    # reversed slice (which would decode y_w[0] instead).
    got = _np2(model.f_out(x)).flatten()
    np.testing.assert_allclose(got, art.decode(y_w[-1]), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize(
    ("n_y", "n_u", "horizon", "hidden_size", "depth", "n_channels", "n_controls", "activation"), _CASES
)
def test_multistep_rollout_matches_jax_model(
    tmp_path: Path,
    n_y: int,
    n_u: int,
    horizon: int,
    hidden_size: int,
    depth: int,
    n_channels: int,
    n_controls: int,
    activation: str,
) -> None:
    """Chaining f_step/f_out over the horizon matches the JAX AutoregressivePredictor rollout."""
    artifact, _mlp, _scalers = _build_artifact(
        tmp_path, n_y, n_u, horizon, hidden_size, depth, n_channels, n_controls, activation
    )
    model = NNSymbolicModel.from_artifact(artifact)
    art = MLPArtifact.load(artifact)

    rng = np.random.default_rng(_SEED + 3)
    ctx = max(n_y, n_u) + 2
    y_ctx = rng.standard_normal((n_channels, ctx))  # raw EEG context (no projection -> standardized channels)
    u_ctx = rng.standard_normal((ctx, n_controls))
    u_future = rng.standard_normal((horizon, n_controls))

    y_pred_jax = _jax_horizon_rollout(art, y_ctx, u_ctx, u_future)  # (n_eeg, horizon)

    # The shooting state is in model space: encode the raw EEG window (standardize) before the roll.
    x = np.concatenate([art.encode(y_ctx.T[-n_y:]).flatten(), u_ctx[-n_u:].flatten()])
    y_pred_ca = []
    for k in range(horizon):
        x = _np2(model.f_step(x, u_future[k])).flatten()
        y_pred_ca.append(_np2(model.f_out(x)).flatten())
    y_pred_ca_arr = np.stack(y_pred_ca, axis=1)

    np.testing.assert_allclose(y_pred_ca_arr, y_pred_jax, rtol=1e-10, atol=1e-12)


def test_mlp_artifact_round_trip_preserves_exact_weights_and_meta(tmp_path: Path) -> None:
    """MLPArtifact.save/.load round-trips weights/biases and metadata bit-exactly."""
    n_y, n_u, horizon, hidden_size, depth, n_channels, n_controls, activation = 2, 2, 3, 5, 2, 2, 2, "relu"
    artifact, mlp, scalers = _build_artifact(
        tmp_path, n_y, n_u, horizon, hidden_size, depth, n_channels, n_controls, activation
    )

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
    u_std = mlp_artifact.u_pipeline.standardizer
    y_std = mlp_artifact.y_pipeline.standardizer
    assert u_std is not None
    assert y_std is not None
    np.testing.assert_array_equal(u_std.center, scalers["u_mean"])
    np.testing.assert_array_equal(u_std.scale, scalers["u_scale"])
    np.testing.assert_array_equal(y_std.center, scalers["y_mean"])
    np.testing.assert_array_equal(y_std.scale, scalers["y_scale"])


def test_f_out_decodes_latent_state_to_eeg(tmp_path: Path) -> None:
    """With a projection, f_out decodes the last latent row to raw EEG (mirrors MLPArtifact.decode)."""
    n_y, n_u, k, n_eeg, n_controls = 3, 2, 2, 5, 2
    artifact = _build_projection_artifact(tmp_path, n_y, n_u, 3, 4, 2, k, n_eeg, n_controls)
    model = NNSymbolicModel.from_artifact(artifact)
    art = MLPArtifact.load(artifact)

    assert model.n_channels == k  # the state carries latent components
    assert model.n_eeg_channels == n_eeg  # but the output is raw EEG
    assert model.state_shape[0] == n_y * k + n_u * n_controls  # shooting state is latent-sized

    rng = np.random.default_rng(_SEED + 6)
    z_w = rng.standard_normal((n_y, k))
    u_w = rng.standard_normal((n_u, n_controls))
    x = np.concatenate([z_w.flatten(), u_w.flatten()])

    got = _np2(model.f_out(x)).flatten()
    assert got.shape == (n_eeg,)
    np.testing.assert_allclose(got, art.decode(z_w[-1]), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("activation", ["relu", "tanh", "softplus"])
def test_multistep_rollout_matches_jax_model_with_projection(tmp_path: Path, activation: str) -> None:
    """Chaining f_step/f_out over a latent-projection model matches the JAX rollout.

    The CasADi rollout runs entirely in latent space (encoded initial window, latent f_step)
    and f_out decodes each step to EEG; the JAX AutoregressivePredictor does the same, so the
    decoded EEG trajectories must agree bit-for-bit.
    """
    n_y, n_u, horizon = 2, 2, 3
    k, n_eeg, n_controls, hidden, depth = 2, 5, 2, 4, 2
    artifact = _build_projection_artifact(tmp_path, n_y, n_u, horizon, hidden, depth, k, n_eeg, n_controls, activation)
    model = NNSymbolicModel.from_artifact(artifact)
    art = MLPArtifact.load(artifact)

    rng = np.random.default_rng(_SEED + 7)
    ctx = max(n_y, n_u) + 2
    y_ctx = rng.standard_normal((n_eeg, ctx))  # raw EEG context
    u_ctx = rng.standard_normal((ctx, n_controls))
    u_future = rng.standard_normal((horizon, n_controls))

    y_pred_jax = _jax_horizon_rollout(art, y_ctx, u_ctx, u_future)  # (n_eeg, horizon), decoded

    # Encode the raw EEG window to latent for the initial state; controls stay raw.
    z_w = art.encode(y_ctx.T[-n_y:])
    x = np.concatenate([z_w.flatten(), u_ctx[-n_u:].flatten()])
    y_pred_ca = []
    for step in range(horizon):
        x = _np2(model.f_step(x, u_future[step])).flatten()
        y_pred_ca.append(_np2(model.f_out(x)).flatten())  # f_out decodes back to EEG
    y_pred_ca_arr = np.stack(y_pred_ca, axis=1)

    assert y_pred_ca_arr.shape == (n_eeg, horizon)
    np.testing.assert_allclose(y_pred_ca_arr, y_pred_jax, rtol=1e-10, atol=1e-12)
