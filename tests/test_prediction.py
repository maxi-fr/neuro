from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from neuro.types import FloatArray


jax.config.update("jax_enable_x64", True)  # noqa: FBT003

from neuro.nn_training import (  # noqa: E402
    build_dataset_for_trajectory,
    load_trajectory,
    prepare_datasets,
)
from neuro.prediction import (  # noqa: E402
    AutoregressivePredictor,
    MLPArtifact,
)
from neuro.transforms import PCAProjection, Pipeline, Standardizer  # noqa: E402

_SEED = 7


def _write_trajectory(path: Path, n_steps: int, n_eeg: int, n_controls: int) -> str:
    """Write a synthetic ``.npz`` trajectory (``sensor_0.y_mea``/``controller.u``) and return its path."""
    rng = np.random.default_rng(_SEED + hash(str(path)) % 1000)

    y = rng.standard_normal((n_steps, n_eeg)) + np.arange(1.0, n_eeg + 1.0)
    u = rng.standard_normal((n_steps, n_controls))
    # ty reads the dotted-key **kwargs as candidates for savez's `allow_pickle` keyword.
    np.savez(path, **{"sensor_0.y_mea": y, "controller.u": u})  # ty:ignore[invalid-argument-type]
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

    x_full, y_full, n_channels = prepare_datasets([file], n_steps, 1, n_y, n_u, horizon, 1e-4)

    assert n_channels == n_eeg
    assert x_full.shape[1] == n_y * n_eeg + n_u * n_controls + horizon * n_controls

    u, y = load_trajectory(file, n_steps, 1, 1e-4)
    x_manual, y_manual = build_dataset_for_trajectory(u, y, n_y, n_u, horizon)
    np.testing.assert_allclose(x_full, x_manual, atol=1e-12)
    np.testing.assert_allclose(y_full, y_manual, atol=1e-12)


def test_prepare_datasets_supports_optional_n_steps(tmp_path: Path) -> None:
    """``load_trajectory`` and ``prepare_datasets`` read the complete trajectory when ``n_steps`` is None."""
    n_eeg, n_controls, n_steps = 6, 2, 250
    n_y, n_u, horizon = 4, 3, 5
    file = _write_trajectory(tmp_path / "traj.npz", n_steps, n_eeg, n_controls)

    u, y = load_trajectory(file, None, 1, 1e-4)
    assert u.shape == (n_steps, n_controls)
    assert y.shape == (n_steps, n_eeg)

    x_full, _y_full, n_channels = prepare_datasets([file], None, 1, n_y, n_u, horizon, 1e-4)
    assert n_channels == n_eeg
    assert x_full.shape[0] == n_steps - horizon - max(n_y - 1, n_u)


def test_artifact_round_trips_latent_projection(tmp_path: Path) -> None:
    """``MLPArtifact`` persists and restores the PCA basis/mean and reports both dimensions."""
    k, n_eeg, n_controls = 3, 6, 2
    artifact, basis, pca_mean = _build_projection_artifact(
        tmp_path, n_y=4, n_u=3, horizon=5, k=k, n_eeg=n_eeg, n_controls=n_controls
    )

    loaded = MLPArtifact.load(artifact)
    pca = loaded.y_pipeline.pca
    assert pca is not None
    np.testing.assert_array_equal(pca.basis, basis)
    np.testing.assert_array_equal(pca.mean, pca_mean)
    assert loaded.n_channels == k
    assert loaded.n_eeg_channels == n_eeg


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
    assert loaded.y_pipeline.pca is None
    assert loaded.n_eeg_channels == n_channels


def test_load_trajectory_and_prepare_datasets_with_cutoff_hz(tmp_path: Path) -> None:
    """load_trajectory and prepare_datasets apply custom lowpass cutoff_hz when supplied."""
    n_eeg, n_controls, n_steps = 4, 2, 200
    n_y, n_u, horizon = 3, 2, 4
    file = _write_trajectory(tmp_path / "traj.npz", n_steps, n_eeg, n_controls)

    u, y = load_trajectory(file, n_steps, 2, 1e-4, cutoff_hz=45.0)
    assert u.shape == (n_steps // 2, n_controls)
    assert y.shape == (n_steps // 2, n_eeg)

    x_full, y_full, n_channels = prepare_datasets([file], n_steps, 2, n_y, n_u, horizon, 1e-4, cutoff_hz=45.0)
    assert n_channels == n_eeg
    x_manual, y_manual = build_dataset_for_trajectory(u, y, n_y, n_u, horizon)
    np.testing.assert_allclose(x_full, x_manual, atol=1e-12)
    np.testing.assert_allclose(y_full, y_manual, atol=1e-12)
