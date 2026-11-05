from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.observable_predictability.features import (
    CONTROL_HORIZONS_S,
    HealthyConnectivityReference,
    compute_connectivity_penalty,
    compute_horizon_penalty,
    extract_connectivity_frames,
)
from experiments.observable_predictability.metrics import RolloutIdentity, windowed
from neuro.config import EegMsGeometry
from neuro.predictor.losses import EegMsLoss, LossContext

_SEED = 42


@pytest.mark.parametrize("fs", [50.0, 100.0])
@pytest.mark.parametrize("hop_s", [0.05, 0.02])
def test_eeg_ms_windows_pin_against_numpy_metrics(fs: float, hop_s: float) -> None:
    """Verify the JAX loss against a NumPy trailing mean-square calculation."""
    rng = np.random.default_rng(_SEED)
    n_channels = 4
    span_s = 1.0
    span_steps = round(span_s * fs)
    window_s = 0.1

    # 1. Test identity standardizer: x is already in raw units
    x_raw = rng.standard_normal((n_channels, span_steps))
    _, want = windowed(x_raw, fs, lambda block, _: (block**2).mean(axis=1), window_s=window_s, hop_s=hop_s)

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


# --- Ticket 03: Healthy connectivity interval reference and excess penalty -----------------


def test_healthy_connectivity_reference_enforces_reference_split_and_bounds() -> None:
    rng = np.random.default_rng(801)
    n_pairs = 6
    ref_values = rng.uniform(-0.4, 0.4, size=(3, 20, n_pairs))
    identities = [RolloutIdentity("healthy", s, "interictal", 0, "zero", split="reference") for s in range(3)]

    conn_ref = HealthyConnectivityReference.fit(ref_values, identities=identities)
    assert conn_ref.q05.shape == (n_pairs,)
    assert conn_ref.q95.shape == (n_pairs,)
    assert np.all(conn_ref.q95 > conn_ref.q05)

    bad_ident = [RolloutIdentity("healthy", 0, "interictal", 0, "zero", split="dev")]
    with pytest.raises(ValueError, match="split='reference'"):
        HealthyConnectivityReference.fit(ref_values[:1], identities=bad_ident)


def test_connectivity_interval_penalty_zero_inside_quadratic_outside_and_masks_coverage() -> None:
    n_pairs = 6
    # Use exact healthy bounds of -0.2 and 0.3.
    q05 = np.full(n_pairs, -0.2, dtype=np.float64)
    q95 = np.full(n_pairs, 0.3, dtype=np.float64)
    conn_ref = HealthyConnectivityReference(
        q05=q05,
        q95=q95,
        pairs=((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)),
        window_s=1.02,
        pair_valid_fraction=np.ones(n_pairs, dtype=np.float64),
    )

    # Three complete Frames test inside, above, and below the interval.
    # Two incomplete Frames test that changing pair coverage invalidates the penalty.
    values = np.array(
        [
            [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3],
            [0.55, 0.55, 0.55, 0.55, 0.55, 0.55],
            [-0.70, -0.70, -0.70, -0.70, -0.70, -0.70],
            [0.55, 0.55, 0.55, 0.55, 0.55, np.nan],
            [0.55, 0.55, 0.55, 0.55, np.nan, np.nan],
        ],
        dtype=np.float64,
    )
    valid_mask = np.isfinite(values)

    res = compute_connectivity_penalty(values, conn_ref, valid_mask=valid_mask)

    assert res.penalty[0] == pytest.approx(0.0, abs=1e-12)
    assert res.penalty[1] == pytest.approx(0.0625, rel=1e-12)
    assert res.penalty[2] == pytest.approx(0.25, rel=1e-12)
    assert np.isnan(res.penalty[3])
    assert np.isnan(res.penalty[4])
    assert np.array_equal(res.frame_valid, np.array([True, True, True, False, False]))
    assert res.pair_valid_fraction == pytest.approx(27.0 / 30.0)
    assert res.frame_valid_fraction == pytest.approx(0.60)


def test_connectivity_end_to_end_extraction_and_horizon_penalty() -> None:
    rng = np.random.default_rng(802)
    fs = 50.0
    n_channels = 4
    ref_ext = [
        extract_connectivity_frames(
            rng.standard_normal((75, n_channels)),
            fs,
            history=rng.standard_normal((60, n_channels)),
            window_s=1.02,
        )
        for _ in range(3)
    ]
    conn_ref = HealthyConnectivityReference.fit(ref_ext)

    eval_ext = extract_connectivity_frames(
        rng.standard_normal((75, n_channels)),
        fs,
        history=rng.standard_normal((60, n_channels)),
        window_s=1.02,
    )
    penalty_res = compute_connectivity_penalty(eval_ext, conn_ref)
    assert penalty_res.frame_valid_fraction == pytest.approx(1.0)
    assert np.all(penalty_res.penalty >= 0.0)

    for h in CONTROL_HORIZONS_S:
        h_pen = float(compute_horizon_penalty(eval_ext.times, penalty_res.penalty, h))
        assert np.isfinite(h_pen)
        assert h_pen >= 0.0
