from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neuro.config import EegMsGeometry
from neuro.metrics import DEFAULT_HOP_S, METRICS
from neuro.predictor.losses import EegMsLoss, LossContext

_SEED = 42


@pytest.mark.parametrize("fs", [50.0, 100.0])
@pytest.mark.parametrize("hop_s", [DEFAULT_HOP_S, 0.02])
def test_eeg_ms_windows_pin_against_numpy_metrics(fs: float, hop_s: float) -> None:
    """The JAX twin reproduces METRICS['eeg_ms'] window-by-window in raw units down to ~1e-12."""
    rng = np.random.default_rng(_SEED)
    n_channels = 4
    span_s = 1.0
    span_steps = round(span_s * fs)
    window_s = METRICS["eeg_ms"].window_s

    # 1. Test identity standardizer: x is already in raw units
    x_raw = rng.standard_normal((n_channels, span_steps))
    _, want = METRICS["eeg_ms"](x_raw, fs, hop_s=hop_s)

    loss_fn = EegMsLoss(
        weight=1.0,
        span_steps=span_steps,
        start_epoch=0,
        geometry=EegMsGeometry(window_s=window_s, hop_s=hop_s),
    )

    with jax.enable_x64():
        x_arr = jnp.asarray(x_raw.T[np.newaxis, ...], dtype=jnp.float64)  # (1, span_steps, n_channels)
        ctx_identity = LossContext(
            y_center=jnp.zeros(n_channels, dtype=jnp.float64),
            y_scale=jnp.ones(n_channels, dtype=jnp.float64),
            fs=fs,
            epoch=None,
        )

        got = np.asarray(loss_fn.windowed_power(x_arr, ctx_identity)[0])

    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)

    # 2. Test non-trivial standardizer affine recovery
    center = rng.standard_normal(n_channels)
    scale = np.abs(rng.standard_normal(n_channels)) + 0.5
    x_std = (x_raw - center[:, None]) / scale[:, None]

    with jax.enable_x64():
        ctx_scaled = LossContext(
            y_center=jnp.asarray(center, dtype=jnp.float64),
            y_scale=jnp.asarray(scale, dtype=jnp.float64),
            fs=fs,
            epoch=None,
        )
        x_std_arr = jnp.asarray(x_std.T[np.newaxis, ...], dtype=jnp.float64)
        got_scaled = np.asarray(loss_fn.windowed_power(x_std_arr, ctx_scaled)[0])

    np.testing.assert_allclose(got_scaled, want, rtol=1e-12, atol=1e-12)


def test_eeg_ms_loss_identical_inputs_give_zero() -> None:
    """When pred and true trajectories are identical, EegMsLoss evaluates to zero."""
    rng = np.random.default_rng(_SEED + 1)
    batch, span_steps, n_channels = 2, 50, 4

    with jax.enable_x64():
        x = jnp.asarray(rng.standard_normal((batch, span_steps, n_channels)), dtype=jnp.float64)

        ctx = LossContext(
            y_center=jnp.zeros(n_channels, dtype=jnp.float64),
            y_scale=jnp.ones(n_channels, dtype=jnp.float64),
            fs=50.0,
            epoch=None,
        )
        loss_fn = EegMsLoss(
            weight=1.0,
            span_steps=span_steps,
            start_epoch=0,
            geometry=EegMsGeometry(window_s=5 / 50.0, hop_s=2 / 50.0),
        )

        loss, diag = loss_fn(x, x, ctx)
    assert float(loss) == pytest.approx(0.0, abs=1e-12)
    assert diag == {}


def test_eeg_ms_loss_gradient_finite_and_nonzero() -> None:
    """EegMsLoss backpropagates finite and non-zero gradients with respect to pred."""
    rng = np.random.default_rng(_SEED + 2)
    batch, span_steps, n_channels = 2, 50, 4

    with jax.enable_x64():
        pred = jnp.asarray(rng.standard_normal((batch, span_steps, n_channels)), dtype=jnp.float64)
        true = jnp.asarray(rng.standard_normal((batch, span_steps, n_channels)), dtype=jnp.float64)

        ctx = LossContext(
            y_center=jnp.asarray(rng.standard_normal(n_channels), dtype=jnp.float64),
            y_scale=jnp.asarray(np.abs(rng.standard_normal(n_channels)) + 0.5, dtype=jnp.float64),
            fs=50.0,
            epoch=None,
        )
        loss_fn = EegMsLoss(
            weight=1.0,
            span_steps=span_steps,
            start_epoch=0,
            geometry=EegMsGeometry(window_s=5 / 50.0, hop_s=2 / 50.0),
        )

        grad_fn = jax.grad(lambda p: loss_fn(p, true, ctx)[0])
        grad = grad_fn(pred)

    assert jnp.all(jnp.isfinite(grad))
    assert not jnp.all(grad == 0.0)
