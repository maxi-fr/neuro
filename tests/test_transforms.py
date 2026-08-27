from __future__ import annotations

from typing import Literal

import numpy as np
import pytest

from neuro.transforms import Standardizer

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

    # Both scopes fit one statistic per column; global scaling pools it, then broadcasts back,
    # so a consumer checking the fitted width against the output width never special-cases it.
    assert std.center.shape == (4,)
    assert std.scale.shape == (4,)
    if global_scaling:
        assert len(np.unique(std.center)) == 1
        assert len(np.unique(std.scale)) == 1


def test_standardizer_matches_standard() -> None:
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
