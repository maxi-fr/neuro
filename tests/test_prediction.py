"""Tests for the NN predictor's transform pipeline (standardize-then-project) and artifact.

The y-pipeline maps raw EEG into model space -- a channel :class:`~neuro.transforms.Standardizer`
followed by an optional :class:`~neuro.transforms.PCAProjection` -- so the predictor trains and
rolls out in the reduced latent space and decodes back to raw EEG channels before returning
predictions. These tests pin the raw dataset build, the artifact round-trip (with and without a
projection), and the decode wired into ``NNPredictor``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from neuro.types import FloatArray

# float64 parity is mandatory; enable before any array is created.
jax.config.update("jax_enable_x64", True)  # noqa: FBT003

from neuro.nn_training import (  # noqa: E402
    build_dataset_for_trajectory,
    load_trajectory,
    prepare_datasets,
)
from neuro.prediction import (  # noqa: E402
    AutoregressivePredictor,
    MLPArtifact,
    NNPredictor,
    PredictionWindow,
)
from neuro.transforms import PCAProjection, Pipeline, Standardizer  # noqa: E402

_SEED = 7


def _write_trajectory(path: Path, n_steps: int, n_eeg: int, n_controls: int) -> str:
    """Write a synthetic ``.npz`` trajectory (``y_mea``/``u``) and return its path."""
    rng = np.random.default_rng(_SEED + hash(str(path)) % 1000)
    # Give the EEG a non-zero mean so the standardizer/projection means are exercised.
    y = rng.standard_normal((n_steps, n_eeg)) + np.arange(1.0, n_eeg + 1.0)
    u = rng.standard_normal((n_steps, n_controls))
    np.savez(path, y_mea=y, u=u)
    return str(path)


def _orthonormal_basis(rng: np.random.Generator, k: int, n_eeg: int) -> FloatArray:
    """Return an orthonormal ``(k, n_eeg)`` basis (orthonormal rows)."""
    q, _ = np.linalg.qr(rng.standard_normal((n_eeg, n_eeg)))
    return np.ascontiguousarray(q[:, :k].T)


def _standardizer(rng: np.random.Generator, dim: int) -> Standardizer:
    """A random (non-trivial) standardizer of channel dimension ``dim``."""
    return Standardizer(center=rng.standard_normal(dim), scale=rng.uniform(0.5, 2.0, dim))


def _build_projection_artifact(
    tmp_path: Path,
    *,
    n_y: int,
    n_u: int,
    horizon: int,
    k: int,
    n_eeg: int,
    n_controls: int,
    zero_model: bool = False,
) -> tuple[Path, FloatArray, FloatArray]:
    """Save a projection artifact (model runs in ``k``-dim latent space).

    Returns ``(artifact_path, basis, pca_mean)``.
    """
    rng = np.random.default_rng(_SEED + 1)
    basis = _orthonormal_basis(rng, k, n_eeg)
    pca_mean = rng.standard_normal(n_eeg)

    in_size = n_y * k + n_u * n_controls
    mlp = eqx.nn.MLP(
        in_size=in_size, out_size=k, width_size=4, depth=2, activation=jax.nn.relu, key=jax.random.PRNGKey(0)
    )
    if zero_model:  # a zero model outputs 0 for any input, so each latent prediction is the zero vector
        mlp = jax.tree_util.tree_map(lambda leaf: jnp.zeros_like(leaf) if eqx.is_inexact_array(leaf) else leaf, mlp)
    wrapped = AutoregressivePredictor(
        model=mlp, n_y=n_y, n_u=n_u, horizon=horizon, n_channels=k, n_controls=n_controls, activation="relu"
    )

    y_pipeline = Pipeline((_standardizer(rng, n_eeg), PCAProjection(basis=basis, mean=pca_mean)))
    u_pipeline = Pipeline((_standardizer(rng, n_controls),))
    artifact = tmp_path / "art"
    MLPArtifact(model=wrapped, dt=0.01, downsample=1, y_pipeline=y_pipeline, u_pipeline=u_pipeline).save(artifact)
    return artifact, basis, pca_mean


def test_prepare_datasets_builds_raw_windows(tmp_path: Path) -> None:
    """``prepare_datasets`` returns raw windows with ``n_channels == n_eeg`` (no transform applied)."""
    n_eeg, n_controls, n_steps = 6, 2, 250
    n_y, n_u, horizon = 4, 3, 5
    file = _write_trajectory(tmp_path / "traj.npz", n_steps, n_eeg, n_controls)

    x_full, y_full, n_channels = prepare_datasets([file], n_steps, 1, n_y, n_u, horizon)

    assert n_channels == n_eeg
    assert x_full.shape[1] == n_y * n_eeg + n_u * n_controls + horizon * n_controls

    # The build must be the raw windowing, untouched by any scaler/projection.
    u, y = load_trajectory(file, n_steps, 1)
    x_manual, y_manual = build_dataset_for_trajectory(u, y, n_y, n_u, horizon)
    np.testing.assert_allclose(x_full, x_manual, atol=1e-12)
    np.testing.assert_allclose(y_full, y_manual, atol=1e-12)


def test_artifact_round_trips_latent_projection(tmp_path: Path) -> None:
    """``MLPArtifact`` persists and restores the PCA basis/mean and reports both dimensions."""
    k, n_eeg, n_controls = 3, 6, 2
    artifact, basis, pca_mean = _build_projection_artifact(
        tmp_path, n_y=4, n_u=3, horizon=5, k=k, n_eeg=n_eeg, n_controls=n_controls
    )

    loaded = MLPArtifact.load(artifact)
    assert loaded.latent_basis is not None
    np.testing.assert_array_equal(loaded.latent_basis, basis)
    np.testing.assert_array_equal(loaded.latent_mean, pca_mean)
    assert loaded.n_channels == k  # the model's latent dimension
    assert loaded.n_eeg_channels == n_eeg  # raw EEG channels


def test_artifact_without_projection_is_backward_compatible(tmp_path: Path) -> None:
    """A non-projection artifact loads with ``latent_basis is None`` and ``n_eeg == n_channels``."""
    n_channels, n_controls = 4, 2
    mlp = eqx.nn.MLP(
        in_size=2 * n_channels + 2 * n_controls,
        out_size=n_channels,
        width_size=4,
        depth=1,
        activation=jax.nn.relu,
        key=jax.random.PRNGKey(0),
    )
    wrapped = AutoregressivePredictor(model=mlp, n_y=2, n_u=2, horizon=3, n_channels=n_channels, n_controls=n_controls)
    rng = np.random.default_rng(_SEED)
    artifact = tmp_path / "art"
    MLPArtifact(
        model=wrapped,
        dt=0.01,
        downsample=1,
        y_pipeline=Pipeline((_standardizer(rng, n_channels),)),
        u_pipeline=Pipeline((_standardizer(rng, n_controls),)),
    ).save(artifact)

    loaded = MLPArtifact.load(artifact)
    assert loaded.latent_basis is None
    assert loaded.latent_mean is None
    assert loaded.n_eeg_channels == n_channels
    assert NNPredictor.load(artifact)._n_ch == n_channels  # noqa: SLF001


def test_predict_decodes_latent_predictions_to_eeg(tmp_path: Path) -> None:
    """``NNPredictor.predict`` decodes latent predictions back to raw EEG (full inverse pipeline).

    With a zero model every latent prediction is the zero vector, so every decoded column must
    equal ``artifact.decode(0)`` -- the constant the inverse projection + inverse standardization
    map the origin to. The output also has the raw EEG channel count, not ``k``.
    """
    k, n_eeg, n_controls = 3, 6, 2
    n_y, n_u, horizon = 4, 3, 5
    artifact, _basis, _pca_mean = _build_projection_artifact(
        tmp_path, n_y=n_y, n_u=n_u, horizon=horizon, k=k, n_eeg=n_eeg, n_controls=n_controls, zero_model=True
    )
    predictor = NNPredictor.load(artifact)
    art = MLPArtifact.load(artifact)

    rng = np.random.default_rng(_SEED + 4)
    ctx = max(n_y, n_u) + 2
    n_steps = 2 * horizon
    window = PredictionWindow(
        y_ctx=rng.standard_normal((n_eeg, ctx)),
        u_ctx=rng.standard_normal((ctx, n_controls)),
        u_future=rng.standard_normal((n_steps, n_controls)),
        dt_data=predictor.dt,
    )

    y_pred = predictor.predict(window, horizon_s=n_steps * predictor.dt)

    assert y_pred.shape == (n_eeg, n_steps)
    expected_col = art.decode(np.zeros(k))
    np.testing.assert_allclose(y_pred, np.tile(expected_col[:, None], (1, n_steps)), atol=1e-10)
