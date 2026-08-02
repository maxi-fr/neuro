from __future__ import annotations

from typing import Literal

import numpy as np
import pytest

from neuro.transforms import PCAProjection, Pipeline, Standardizer

_SEED = 11


@pytest.mark.parametrize("kind", ["standard", "robust"])
@pytest.mark.parametrize("global_scaling", [False, True])
def test_standardizer_round_trip(kind: Literal["standard", "robust"], global_scaling: bool) -> None:  # noqa: FBT001
    """``inverse_transform`` undoes ``transform`` for both scaler kinds and scopes."""
    rng = np.random.default_rng(_SEED)
    x = rng.standard_normal((200, 4)) * np.array([1.0, 5.0, 0.2, 3.0]) + 2.0

    std = Standardizer.fit(x, kind=kind, global_scaling=global_scaling)
    z = std.transform(x)
    np.testing.assert_allclose(std.inverse_transform(z), x, atol=1e-10)

    expected_len = 1 if global_scaling else 4
    assert std.center.shape == (expected_len,)
    assert std.scale.shape == (expected_len,)


def test_standardizer_matches_sklearn_standard() -> None:
    """Per-channel standard scaling matches mean/population-std."""
    rng = np.random.default_rng(_SEED + 1)
    x = rng.standard_normal((500, 3)) * np.array([2.0, 0.5, 4.0]) + np.array([1.0, -3.0, 7.0])
    std = Standardizer.fit(x, kind="standard")
    np.testing.assert_allclose(std.center, x.mean(axis=0), atol=1e-12)
    np.testing.assert_allclose(std.scale, x.std(axis=0), atol=1e-12)


def test_standardizer_zero_scale_is_set_to_one() -> None:
    """A constant channel gets scale 1 (not 0), so it passes through unchanged."""
    x = np.column_stack([np.full(10, 5.0), np.arange(10.0)])
    std = Standardizer.fit(x, kind="standard")
    assert std.scale[0] == 1.0
    np.testing.assert_allclose(std.transform(x)[:, 0], 0.0, atol=1e-12)


def test_pca_projection_orthonormal_and_none_when_full() -> None:
    """PCA has orthonormal rows / requested dim, and is ``None`` when it would be a no-op."""
    rng = np.random.default_rng(_SEED + 2)
    x = rng.standard_normal((300, 6))

    pca = PCAProjection.fit(x, latent_dim=3)
    assert pca is not None
    assert pca.basis.shape == (3, 6)
    assert pca.mean.shape == (6,)
    np.testing.assert_allclose(pca.basis @ pca.basis.T, np.eye(3), atol=1e-10)

    assert PCAProjection.fit(x, latent_dim=6) is None
    assert PCAProjection.fit(x, latent_dim=9) is None


def test_pipeline_standardize_then_pca_round_trips_in_subspace() -> None:
    """A standardize-then-PCA pipeline reconstructs data that already lies in the subspace."""
    rng = np.random.default_rng(_SEED + 3)

    latent = rng.standard_normal((400, 2))
    embed = rng.standard_normal((2, 5))
    x = latent @ embed + np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    std = Standardizer.fit(x, kind="standard")
    pca = PCAProjection.fit(std.transform(x), latent_dim=2)
    assert pca is not None
    pipeline = Pipeline((std, pca))

    z = pipeline.transform(x)
    assert z.shape == (400, 2)
    np.testing.assert_allclose(pipeline.inverse_transform(z), x, atol=1e-8)


def test_pipeline_accessors() -> None:
    """``standardizer``/``pca`` return the right step or ``None``."""
    std = Standardizer(center=np.zeros(3), scale=np.ones(3))
    pca = PCAProjection(basis=np.eye(2, 3), mean=np.zeros(3))

    both = Pipeline((std, pca))
    assert both.standardizer is std
    assert both.pca is pca

    only_std = Pipeline((std,))
    assert only_std.standardizer is std
    assert only_std.pca is None


def test_pipeline_serialization_round_trip() -> None:
    """``step_tags`` + ``array_dict`` reconstruct an identical pipeline via ``from_serialized``."""
    rng = np.random.default_rng(_SEED + 4)
    std = Standardizer(center=rng.standard_normal(4), scale=rng.uniform(0.5, 2.0, 4))
    pca = PCAProjection(basis=rng.standard_normal((2, 4)), mean=rng.standard_normal(4))
    pipeline = Pipeline((std, pca))

    tags = pipeline.step_tags()
    arrays = pipeline.array_dict("y")
    assert tags == ["standardizer", "pca"]

    restored = Pipeline.from_serialized("y", tags, arrays)
    x = rng.standard_normal((10, 4))
    np.testing.assert_allclose(restored.transform(x), pipeline.transform(x), atol=1e-12)
