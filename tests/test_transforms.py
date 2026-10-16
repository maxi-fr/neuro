from __future__ import annotations

import numpy as np
import pytest

from neuro.transforms import Standardizer

_SEED = 11


def test_standardizer_round_trip() -> None:
    """Per-channel mean/std scaling preserves values through a round trip."""
    rng = np.random.default_rng(_SEED)
    x = rng.standard_normal((200, 4)) * np.array([1.0, 5.0, 0.2, 3.0]) + 2.0

    std = Standardizer.fit(x)
    z = std.transform(x)
    np.testing.assert_allclose(std.inverse_transform(z), x, atol=1e-10)

    assert std.center.shape == (4,)
    assert std.scale.shape == (4,)


def test_standardizer_matches_standard() -> None:
    """Fixed per-channel scaling matches mean/population-std."""
    rng = np.random.default_rng(_SEED + 1)
    x = rng.standard_normal((500, 3)) * np.array([2.0, 0.5, 4.0]) + np.array([1.0, -3.0, 7.0])
    std = Standardizer.fit(x)
    np.testing.assert_allclose(std.center, x.mean(axis=0), atol=1e-12)
    np.testing.assert_allclose(std.scale, x.std(axis=0), atol=1e-12)


def test_standardizer_zero_scale_is_set_to_one() -> None:
    """A constant channel uses scale one and transforms to zero."""
    x = np.column_stack([np.full(10, 5.0), np.arange(10.0)])
    std = Standardizer.fit(x)
    assert std.scale[0] == 1.0
    np.testing.assert_allclose(std.transform(x)[:, 0], 0.0, atol=1e-12)


@pytest.mark.parametrize("prefix", ["y", "u"])
def test_standardizer_serialization_round_trip(prefix: str) -> None:
    """``arrays`` + ``from_arrays`` reconstructs an identical Standardizer."""
    rng = np.random.default_rng(_SEED + 2)
    center = rng.standard_normal(4)
    scale = rng.uniform(0.5, 2.0, 4)
    std = Standardizer(center=center, scale=scale)

    arrs = std.arrays(prefix=prefix)
    restored = Standardizer.from_arrays(arrs, prefix=prefix)
    np.testing.assert_allclose(restored.center, std.center)
    np.testing.assert_allclose(restored.scale, std.scale)


def test_structured_standardizer_matches_flattened_statistics() -> None:
    """Each channel-frequency coordinate retains its own mean/std, matching flattened statistics."""
    rng = np.random.default_rng(_SEED + 3)
    x = rng.standard_normal((40, 3, 5))
    structured = Standardizer.fit(x)
    flattened = Standardizer.fit(x.reshape(x.shape[0], -1))

    np.testing.assert_allclose(structured.center.reshape(-1), flattened.center)
    np.testing.assert_allclose(structured.scale.reshape(-1), flattened.scale)
    np.testing.assert_allclose(
        structured.transform(x).reshape(x.shape[0], -1), flattened.transform(x.reshape(x.shape[0], -1))
    )
