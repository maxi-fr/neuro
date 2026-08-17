from __future__ import annotations

import numpy as np
import pytest
import torch

from neuro.metrics import DEFAULT_HOP_S, METRICS
from neuro.predictor.losses import EegMsLoss, LossContext

_SEED = 42


@pytest.mark.parametrize("fs", [50.0, 100.0])
@pytest.mark.parametrize("hop_s", [DEFAULT_HOP_S, 0.02])
def test_eeg_ms_windows_pin_against_numpy_metrics(fs: float, hop_s: float) -> None:
    """The torch twin reproduces METRICS['eeg_ms'] window-by-window in raw units down to ~1e-12."""
    rng = np.random.default_rng(_SEED)
    n_channels = 4
    span_s = 1.0
    span_steps = round(span_s * fs)
    window_s = METRICS["eeg_ms"].window_s
    n_window = round(window_s * fs)
    n_hop = round(hop_s * fs)

    # 1. Test identity standardizer: x is already in raw units
    x_raw = rng.standard_normal((n_channels, span_steps))
    _, want = METRICS["eeg_ms"](x_raw, fs, hop_s=hop_s)

    loss_fn = EegMsLoss(
        weight=1.0,
        span_steps=span_steps,
        start_epoch=0,
        n_window=n_window,
        n_hop=n_hop,
    )

    x_tensor = torch.as_tensor(x_raw.T[np.newaxis, ...], dtype=torch.float64)  # (1, span_steps, n_channels)
    ctx_identity = LossContext(
        y_center=torch.zeros(n_channels, dtype=torch.float64),
        y_scale=torch.ones(n_channels, dtype=torch.float64),
        fs=fs,
        epoch=None,
    )

    got = loss_fn.windowed_power(x_tensor, ctx_identity)[0].numpy()

    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)

    # 2. Test non-trivial standardizer affine recovery
    center = rng.standard_normal(n_channels)
    scale = np.abs(rng.standard_normal(n_channels)) + 0.5
    x_std = (x_raw - center[:, None]) / scale[:, None]

    ctx_scaled = LossContext(
        y_center=torch.as_tensor(center, dtype=torch.float64),
        y_scale=torch.as_tensor(scale, dtype=torch.float64),
        fs=fs,
        epoch=None,
    )
    x_std_tensor = torch.as_tensor(x_std.T[np.newaxis, ...], dtype=torch.float64)
    got_scaled = loss_fn.windowed_power(x_std_tensor, ctx_scaled)[0].numpy()

    np.testing.assert_allclose(got_scaled, want, rtol=1e-12, atol=1e-12)


def test_eeg_ms_loss_identical_inputs_give_zero() -> None:
    """When pred and true trajectories are identical, EegMsLoss evaluates to zero."""
    rng = np.random.default_rng(_SEED + 1)
    batch, span_steps, n_channels = 2, 50, 4
    x = torch.as_tensor(rng.standard_normal((batch, span_steps, n_channels)), dtype=torch.float64)

    ctx = LossContext(
        y_center=torch.zeros(n_channels, dtype=torch.float64),
        y_scale=torch.ones(n_channels, dtype=torch.float64),
        fs=50.0,
        epoch=None,
    )
    loss_fn = EegMsLoss(
        weight=1.0,
        span_steps=span_steps,
        start_epoch=0,
        n_window=5,
        n_hop=2,
    )

    loss, diag = loss_fn(x, x, ctx)
    assert float(loss) == 0.0
    assert diag == {}


def test_eeg_ms_loss_gradient_finite_and_nonzero() -> None:
    """EegMsLoss backpropagates finite and non-zero gradients with respect to pred."""
    rng = np.random.default_rng(_SEED + 2)
    batch, span_steps, n_channels = 2, 50, 4
    pred = torch.tensor(rng.standard_normal((batch, span_steps, n_channels)), dtype=torch.float64, requires_grad=True)
    true = torch.tensor(rng.standard_normal((batch, span_steps, n_channels)), dtype=torch.float64)

    ctx = LossContext(
        y_center=torch.randn(n_channels, dtype=torch.float64),
        y_scale=torch.rand(n_channels, dtype=torch.float64) + 0.5,
        fs=50.0,
        epoch=None,
    )
    loss_fn = EegMsLoss(
        weight=1.0,
        span_steps=span_steps,
        start_epoch=0,
        n_window=5,
        n_hop=2,
    )

    loss, _ = loss_fn(pred, true, ctx)
    loss.backward()

    assert pred.grad is not None
    assert torch.all(torch.isfinite(pred.grad))
    assert not torch.all(pred.grad == 0.0)
