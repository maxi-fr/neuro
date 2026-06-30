"""Tests for the fixed PCA latent projection on the NN predictor.

The projection encodes the raw EEG channels onto ``k`` latent components
(``z = (y - mean) @ E.T``) so the predictor trains and rolls out in the reduced space,
and decodes back (``y = z @ E + mean``) before returning predictions. These tests pin
the orthonormality of the basis, the encode wired into ``prepare_datasets``, the artifact
round-trip (and its backward compatibility), and the decode wired into ``NNPredictor``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

# float64 parity is mandatory; enable before any array is created.
jax.config.update("jax_enable_x64", True)  # noqa: FBT003

from neuro.nn_training import (  # noqa: E402
    build_dataset_for_trajectory,
    fit_latent_projection,
    load_trajectory,
    prepare_datasets,
)
from neuro.prediction import (  # noqa: E402
    AutoregressivePredictor,
    MLPArtifact,
    NNPredictor,
    PredictionWindow,
)

_SEED = 7


def _write_trajectory(path: Path, n_steps: int, n_eeg: int, n_controls: int) -> str:
    """Write a synthetic ``.npz`` trajectory (``y_mea``/``u``) and return its path."""
    rng = np.random.default_rng(_SEED + hash(str(path)) % 1000)
    # Give the EEG a non-zero mean so the projection mean is exercised.
    y = rng.standard_normal((n_steps, n_eeg)) + np.arange(1.0, n_eeg + 1.0)
    u = rng.standard_normal((n_steps, n_controls))
    np.savez(path, y_mea=y, u=u)
    return str(path)


def _orthonormal_basis(rng: np.random.Generator, k: int, n_eeg: int) -> np.ndarray:
    """Return an orthonormal ``(k, n_eeg)`` basis (orthonormal rows)."""
    q, _ = np.linalg.qr(rng.standard_normal((n_eeg, n_eeg)))
    return np.ascontiguousarray(q[:, :k].T)


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
) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Save a projection artifact (model runs in ``k``-dim latent space).

    Returns ``(artifact_path, E, mean, y_mean_latent, y_scale_latent)``.
    """
    rng = np.random.default_rng(_SEED + 1)
    basis = _orthonormal_basis(rng, k, n_eeg)
    mean = rng.standard_normal(n_eeg)

    in_size = n_y * k + n_u * n_controls
    mlp = eqx.nn.MLP(
        in_size=in_size, out_size=k, width_size=4, depth=2, activation=jax.nn.relu, key=jax.random.PRNGKey(0)
    )
    if zero_model:  # a zero model outputs 0 for any input, so each latent prediction == y_mean_latent
        mlp = jax.tree_util.tree_map(lambda leaf: jnp.zeros_like(leaf) if eqx.is_inexact_array(leaf) else leaf, mlp)
    wrapped = AutoregressivePredictor(
        model=mlp, n_y=n_y, n_u=n_u, horizon=horizon, C_y=k, C_u=n_controls, activation="relu"
    )

    y_mean_latent = rng.standard_normal(k)
    y_scale_latent = rng.uniform(0.5, 2.0, k)
    artifact = tmp_path / "art"
    MLPArtifact(
        model=wrapped,
        dt=0.01,
        downsample=1,
        u_mean=rng.standard_normal(n_controls),
        u_scale=rng.uniform(0.5, 2.0, n_controls),
        y_mean=y_mean_latent,
        y_scale=y_scale_latent,
        latent_basis=basis,
        latent_mean=mean,
    ).save(artifact)
    return artifact, basis, mean, y_mean_latent, y_scale_latent


def test_fit_latent_projection_orthonormal_and_shapes(tmp_path: Path) -> None:
    """The PCA basis has orthonormal rows and the requested latent dimension."""
    n_eeg, n_controls, n_steps, k = 6, 2, 300, 3
    files = [_write_trajectory(tmp_path / f"traj_{i}.npz", n_steps, n_eeg, n_controls) for i in range(2)]

    basis, mean = fit_latent_projection(files, n_steps, downsample=1, latent_dim=k)

    assert basis.shape == (k, n_eeg)
    assert mean.shape == (n_eeg,)
    np.testing.assert_allclose(basis @ basis.T, np.eye(k), atol=1e-10)
    y_all = np.concatenate([load_trajectory(f, n_steps, 1)[1] for f in files], axis=0)
    np.testing.assert_allclose(mean, y_all.mean(axis=0), atol=1e-10)


def test_fit_latent_projection_requires_a_target(tmp_path: Path) -> None:
    """Neither ``latent_dim`` nor ``explained_variance`` is an error."""
    files = [_write_trajectory(tmp_path / "traj.npz", 200, 5, 2)]
    with pytest.raises(ValueError, match="latent_dim or explained_variance"):
        fit_latent_projection(files, 200, downsample=1)


def test_fit_latent_projection_explained_variance_selects_k(tmp_path: Path) -> None:
    """``explained_variance`` picks a latent dimension within ``[1, n_eeg]``."""
    n_eeg = 8
    files = [_write_trajectory(tmp_path / "traj.npz", 400, n_eeg, 2)]
    basis, _ = fit_latent_projection(files, 400, downsample=1, explained_variance=0.9)
    assert 1 <= basis.shape[0] <= n_eeg
    assert basis.shape[1] == n_eeg


def test_prepare_datasets_projection_encodes_and_reduces_channels(tmp_path: Path) -> None:
    """With a projection, ``prepare_datasets`` encodes EEG and returns ``C_y == k``."""
    n_eeg, n_controls, n_steps, k = 6, 2, 250, 3
    n_y, n_u, horizon = 4, 3, 5
    file = _write_trajectory(tmp_path / "traj.npz", n_steps, n_eeg, n_controls)

    basis, mean = fit_latent_projection([file], n_steps, downsample=1, latent_dim=k)
    x_proj, y_proj, c_y = prepare_datasets([file], n_steps, 1, n_y, n_u, horizon, projection=(basis, mean))

    assert c_y == k
    assert x_proj.shape[1] == n_y * k + n_u * n_controls + horizon * n_controls

    # The encode must be exactly (y - mean) @ E.T fed through the unchanged windowing.
    u, y = load_trajectory(file, n_steps, 1)
    z = (y - mean) @ basis.T
    x_manual, y_manual = build_dataset_for_trajectory(u, z, n_y, n_u, horizon)
    np.testing.assert_allclose(x_proj, x_manual, atol=1e-12)
    np.testing.assert_allclose(y_proj, y_manual, atol=1e-12)

    # Without a projection the channel count is the raw EEG dimension.
    _, _, c_y_raw = prepare_datasets([file], n_steps, 1, n_y, n_u, horizon)
    assert c_y_raw == n_eeg


def test_artifact_round_trips_latent_projection(tmp_path: Path) -> None:
    """``MLPArtifact`` persists and restores the basis/mean and reports both dimensions."""
    k, n_eeg, n_controls = 3, 6, 2
    artifact, basis, mean, _, _ = _build_projection_artifact(
        tmp_path, n_y=4, n_u=3, horizon=5, k=k, n_eeg=n_eeg, n_controls=n_controls
    )

    loaded = MLPArtifact.load(artifact)
    assert loaded.latent_basis is not None
    np.testing.assert_array_equal(loaded.latent_basis, basis)
    np.testing.assert_array_equal(loaded.latent_mean, mean)
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
    wrapped = AutoregressivePredictor(model=mlp, n_y=2, n_u=2, horizon=3, C_y=n_channels, C_u=n_controls)
    rng = np.random.default_rng(_SEED)
    artifact = tmp_path / "art"
    MLPArtifact(
        model=wrapped,
        dt=0.01,
        downsample=1,
        u_mean=rng.standard_normal(n_controls),
        u_scale=rng.uniform(0.5, 2.0, n_controls),
        y_mean=rng.standard_normal(n_channels),
        y_scale=rng.uniform(0.5, 2.0, n_channels),
    ).save(artifact)

    loaded = MLPArtifact.load(artifact)
    assert loaded.latent_basis is None
    assert loaded.latent_mean is None
    assert loaded.n_eeg_channels == n_channels
    assert NNPredictor.load(artifact)._n_ch == n_channels  # noqa: SLF001


def test_predict_decodes_latent_predictions_to_eeg(tmp_path: Path) -> None:
    """``NNPredictor.predict`` decodes latent predictions back to raw EEG (mean included).

    With a zero model every latent prediction equals ``y_mean_latent``, so every decoded
    column must equal ``y_mean_latent @ E + mean`` -- which fails if the decode drops the
    PCA mean. The output also has the raw EEG channel count, not ``k``.
    """
    k, n_eeg, n_controls = 3, 6, 2
    n_y, n_u, horizon = 4, 3, 5
    artifact, basis, mean, y_mean_latent, _ = _build_projection_artifact(
        tmp_path, n_y=n_y, n_u=n_u, horizon=horizon, k=k, n_eeg=n_eeg, n_controls=n_controls, zero_model=True
    )
    predictor = NNPredictor.load(artifact)

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
    expected_col = y_mean_latent @ basis + mean
    np.testing.assert_allclose(y_pred, np.tile(expected_col[:, None], (1, n_steps)), atol=1e-10)
