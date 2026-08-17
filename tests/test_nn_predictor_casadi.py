"""Verify the CasADi NN-predictor port against the NumPy ``MLPArtifact`` reference."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import casadi as ca
import numpy as np
import pytest

from neuro.nn_predictor_casadi import NNSymbolicModel, _mlp_forward_ca
from neuro.predictor.artifact import Activation, MLPArtifact
from neuro.transforms import PCAProjection, Pipeline, Standardizer

if TYPE_CHECKING:
    from pathlib import Path

    from neuro.types import FloatArray


def _standardizer_pipeline(center: FloatArray, scale: FloatArray) -> Pipeline:
    """A single-step standardizer pipeline from raw ``center``/``scale`` arrays."""
    return Pipeline((Standardizer(center=np.asarray(center, np.float64), scale=np.asarray(scale, np.float64)),))


def _random_layers(
    rng: np.random.Generator, in_size: int, out_size: int, hidden_size: int, depth: int
) -> tuple[tuple[FloatArray, FloatArray], ...]:
    """Random ``(weight, bias)`` pairs for an MLP with ``depth`` hidden layers of ``hidden_size``."""
    sizes = [in_size, *([hidden_size] * depth), out_size]
    return tuple(
        (rng.standard_normal((n_out, n_in)) / np.sqrt(n_in), rng.standard_normal(n_out))
        for n_in, n_out in itertools.pairwise(sizes)
    )


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


def _artifact_horizon_rollout(
    art: MLPArtifact, y_ctx: FloatArray, u_ctx: FloatArray, u_future: FloatArray
) -> FloatArray:
    """Reference EEG rollout over one ``horizon`` via the NumPy ``MLPArtifact`` AR loop.

    Mirrors what the CasADi ``f_step``/``f_out`` chain must reproduce: absorb the raw EEG/control
    context into a model-space state, free-run on raw future controls, and decode back to raw
    EEG. ``y_ctx`` is ``(n_eeg_channels, ctx)``; returns shape ``(n_eeg_channels, horizon)``.
    """
    return art.rollout(art.prime(y_ctx.T, u_ctx), u_future[: art.horizon]).T


def _build_artifact(
    tmp_path: Path,
    n_y: int,
    n_u: int,
    horizon: int,
    hidden_size: int,
    depth: int,
    n_channels: int,
    n_controls: int,
    activation: Activation = "relu",
) -> tuple[Path, tuple[tuple[FloatArray, FloatArray], ...], dict[str, FloatArray]]:
    """Build and save a tiny artifact via ``MLPArtifact.save``, returning its path, layers, and scalers."""
    rng = np.random.default_rng(_SEED)
    layers = _random_layers(rng, n_y * n_channels + n_u * n_controls, n_channels, hidden_size, depth)

    scalers = {
        "u_mean": rng.uniform(-1.0, 1.0, n_controls),
        "u_scale": rng.uniform(0.5, 2.0, n_controls),
        "y_mean": rng.uniform(-1.0, 1.0, n_channels),
        "y_scale": rng.uniform(0.5, 2.0, n_channels),
    }

    artifact = tmp_path / "art"
    MLPArtifact(
        layers=layers,
        activation=activation,
        n_y=n_y,
        n_u=n_u,
        horizon=horizon,
        n_channels=n_channels,
        n_controls=n_controls,
        dt=0.01,
        downsample=1,
        y_pipeline=_standardizer_pipeline(scalers["y_mean"], scalers["y_scale"]),
        u_pipeline=_standardizer_pipeline(scalers["u_mean"], scalers["u_scale"]),
    ).save(artifact)

    return artifact, layers, scalers


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
    activation: Activation = "relu",
) -> Path:
    """Save a tiny artifact whose model runs in a ``k``-dim PCA latent space and return its path.

    The y-pipeline standardizes the ``n_eeg`` raw EEG channels, then projects onto ``k`` latent
    components via a fixed orthonormal basis ``(k, n_eeg)`` plus mean (standardize-then-project),
    so model space is the ``k``-dimensional latent.
    """
    rng = np.random.default_rng(_SEED + 5)
    q, _ = np.linalg.qr(rng.standard_normal((n_eeg, n_eeg)))
    basis = np.ascontiguousarray(q[:, :k].T)  # orthonormal rows
    mean = rng.standard_normal(n_eeg)

    layers = _random_layers(rng, n_y * k + n_u * n_controls, k, hidden_size, depth)
    y_pipeline = Pipeline(
        (
            Standardizer(center=rng.uniform(-1.0, 1.0, n_eeg), scale=rng.uniform(0.5, 2.0, n_eeg)),
            PCAProjection(basis=basis, mean=mean),
        )
    )
    u_pipeline = _standardizer_pipeline(rng.uniform(-1.0, 1.0, n_controls), rng.uniform(0.5, 2.0, n_controls))
    artifact = tmp_path / "art_proj"
    MLPArtifact(
        layers=layers,
        activation=activation,
        n_y=n_y,
        n_u=n_u,
        horizon=horizon,
        n_channels=k,
        n_controls=n_controls,
        dt=0.01,
        downsample=1,
        y_pipeline=y_pipeline,
        u_pipeline=u_pipeline,
    ).save(artifact)
    return artifact


@pytest.mark.parametrize(
    ("n_y", "n_u", "horizon", "hidden_size", "depth", "n_channels", "n_controls", "activation"), _CASES
)
def test_mlp_forward_matches_artifact(
    tmp_path: Path,
    n_y: int,
    n_u: int,
    horizon: int,
    hidden_size: int,
    depth: int,
    n_channels: int,
    n_controls: int,
    activation: Activation,
) -> None:
    """The CasADi MLP forward pass matches ``MLPArtifact.forward_1step`` bit-for-bit."""
    artifact, _layers, _ = _build_artifact(
        tmp_path, n_y, n_u, horizon, hidden_size, depth, n_channels, n_controls, activation
    )
    art = MLPArtifact.load(artifact)

    in_size = n_y * n_channels + n_u * n_controls
    x_sym = ca.SX.sym("x", in_size, 1)
    fn = ca.Function("f", [x_sym], [_mlp_forward_ca(x_sym, art.layers, activation)])

    rng = np.random.default_rng(_SEED + 1)
    for _ in range(5):
        x = rng.standard_normal(in_size)
        got = _np2(fn(x)).flatten()
        want = art.forward_1step(x[: n_y * n_channels], x[n_y * n_channels :])
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
    activation: Activation,
) -> None:
    """f_step/f_out match one hand-written iteration of the AR loop."""
    artifact, _layers, scalers = _build_artifact(
        tmp_path, n_y, n_u, horizon, hidden_size, depth, n_channels, n_controls, activation
    )
    model = NNSymbolicModel.from_artifact(artifact)
    art = MLPArtifact.load(artifact)

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
    y_next_model = art.forward_1step(y_w.flatten(), new_u_w_scaled.flatten())
    new_y_w_want = np.concatenate([y_w[1:], y_next_model[None, :]], axis=0)
    x_next_want = np.concatenate([new_y_w_want.flatten(), new_u_w.flatten()])
    # f_out decodes model space to raw EEG (no projection -> inverse standardization).
    y_next_raw_want = y_next_model * scalers["y_scale"] + scalers["y_mean"]

    np.testing.assert_allclose(x_next, x_next_want, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(y_next_ca, y_next_raw_want, rtol=1e-10, atol=1e-12)


def test_single_step_with_global_scalar_scalers(tmp_path: Path) -> None:
    """f_step matches the NumPy reference when scalers are global scalars (shape (1,)).

    ``global_scaling=True`` trains a single shared mean/scale rather than a per-channel
    vector; the CasADi port must broadcast it across all channels just as numpy does, not
    tile it by ``n_y``/``n_u`` only.
    """
    n_y, n_u, horizon, n_channels, n_controls = 3, 2, 4, 5, 2
    rng = np.random.default_rng(_SEED + 8)
    layers = _random_layers(rng, n_y * n_channels + n_u * n_controls, n_channels, 5, 2)
    scalers = {  # global (scalar) statistics, as produced by global_scaling=True
        "u_mean": np.array([0.3]),
        "u_scale": np.array([1.7]),
        "y_mean": np.array([-0.9]),
        "y_scale": np.array([2.5]),
    }
    artifact = tmp_path / "art"
    MLPArtifact(
        layers=layers,
        activation="relu",
        n_y=n_y,
        n_u=n_u,
        horizon=horizon,
        n_channels=n_channels,
        n_controls=n_controls,
        dt=0.01,
        downsample=1,
        y_pipeline=_standardizer_pipeline(scalers["y_mean"], scalers["y_scale"]),
        u_pipeline=_standardizer_pipeline(scalers["u_mean"], scalers["u_scale"]),
    ).save(artifact)
    model = NNSymbolicModel.from_artifact(artifact)
    art = MLPArtifact.load(artifact)

    rng = np.random.default_rng(_SEED + 9)
    y_w = rng.standard_normal((n_y, n_channels))  # model-space y-window
    u_w_raw = rng.standard_normal((n_u, n_controls))
    u_curr = rng.standard_normal(n_controls)

    x0 = np.concatenate([y_w.flatten(), u_w_raw.flatten()])
    x_next = _np2(model.f_step(x0, u_curr)).flatten()
    y_next_ca = _np2(model.f_out(x_next)).flatten()

    new_u_w = np.concatenate([u_w_raw[1:], u_curr[None, :]], axis=0)
    # The y-window is already model space; the global scalar broadcasts across all channels for u and the decode.
    u_w_scaled = (new_u_w - scalers["u_mean"]) / scalers["u_scale"]
    y_next_raw_want = art.forward_1step(y_w.flatten(), u_w_scaled.flatten()) * scalers["y_scale"] + scalers["y_mean"]
    np.testing.assert_allclose(y_next_ca, y_next_raw_want, rtol=1e-10, atol=1e-12)


def test_output_slices_most_recent_row_not_oldest(tmp_path: Path) -> None:
    """f_out must return the LAST row of the y-window, not the first.

    Uses distinguishable per-row values so a transposed/reversed slice would be caught,
    unlike random data where it could accidentally still pass.
    """
    n_y, n_u, n_channels, n_controls = 3, 2, 2, 2
    artifact, _layers, _scalers = _build_artifact(tmp_path, n_y, n_u, 3, 5, 2, n_channels, n_controls)
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
def test_multistep_rollout_matches_artifact_rollout(
    tmp_path: Path,
    n_y: int,
    n_u: int,
    horizon: int,
    hidden_size: int,
    depth: int,
    n_channels: int,
    n_controls: int,
    activation: Activation,
) -> None:
    """Chaining f_step/f_out over the horizon matches ``MLPArtifact.rollout``."""
    artifact, _layers, _scalers = _build_artifact(
        tmp_path, n_y, n_u, horizon, hidden_size, depth, n_channels, n_controls, activation
    )
    model = NNSymbolicModel.from_artifact(artifact)
    art = MLPArtifact.load(artifact)

    rng = np.random.default_rng(_SEED + 3)
    ctx = max(n_y, n_u) + 2
    y_ctx = rng.standard_normal((n_channels, ctx))  # raw EEG context (no projection -> standardized channels)
    u_ctx = rng.standard_normal((ctx, n_controls))
    u_future = rng.standard_normal((horizon, n_controls))

    y_pred_np = _artifact_horizon_rollout(art, y_ctx, u_ctx, u_future)  # (n_eeg, horizon)

    # The shooting state is in model space: encode the raw EEG window (standardize) before the roll.
    x = np.concatenate([art.encode(y_ctx.T[-n_y:]).flatten(), u_ctx[-n_u:].flatten()])
    y_pred_ca = []
    for k in range(horizon):
        x = _np2(model.f_step(x, u_future[k])).flatten()
        y_pred_ca.append(_np2(model.f_out(x)).flatten())
    y_pred_ca_arr = np.stack(y_pred_ca, axis=1)

    np.testing.assert_allclose(y_pred_ca_arr, y_pred_np, rtol=1e-10, atol=1e-12)


def test_mlp_artifact_round_trip_preserves_exact_weights_and_meta(tmp_path: Path) -> None:
    """MLPArtifact.save/.load round-trips weights/biases and metadata bit-exactly."""
    n_y, n_u, horizon, hidden_size, depth, n_channels, n_controls, activation = 2, 2, 3, 5, 2, 2, 2, "relu"
    artifact, want_layers, scalers = _build_artifact(
        tmp_path, n_y, n_u, horizon, hidden_size, depth, n_channels, n_controls, activation
    )

    mlp_artifact = MLPArtifact.load(artifact)

    assert len(mlp_artifact.layers) == len(want_layers)
    for (got_w, got_b), (want_w, want_b) in zip(mlp_artifact.layers, want_layers, strict=True):
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
def test_multistep_rollout_matches_artifact_rollout_with_projection(tmp_path: Path, activation: Activation) -> None:
    """Chaining f_step/f_out over a latent-projection model matches ``MLPArtifact.rollout``.

    The CasADi rollout runs entirely in latent space (encoded initial window, latent f_step)
    and f_out decodes each step to EEG; the NumPy AR loop does the same, so the decoded EEG
    trajectories must agree bit-for-bit.
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

    y_pred_np = _artifact_horizon_rollout(art, y_ctx, u_ctx, u_future)  # (n_eeg, horizon), decoded

    # Encode the raw EEG window to latent for the initial state; controls stay raw.
    z_w = art.encode(y_ctx.T[-n_y:])
    x = np.concatenate([z_w.flatten(), u_ctx[-n_u:].flatten()])
    y_pred_ca = []
    for step in range(horizon):
        x = _np2(model.f_step(x, u_future[step])).flatten()
        y_pred_ca.append(_np2(model.f_out(x)).flatten())  # f_out decodes back to EEG
    y_pred_ca_arr = np.stack(y_pred_ca, axis=1)

    assert y_pred_ca_arr.shape == (n_eeg, horizon)
    np.testing.assert_allclose(y_pred_ca_arr, y_pred_np, rtol=1e-10, atol=1e-12)
