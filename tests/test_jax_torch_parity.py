"""Numerical parity verification between PyTorch reference and JAX implementations."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import torch

if TYPE_CHECKING:
    from collections.abc import Iterator

from neuro.config import CurriculumMSESpec, EegMsSpec, StftGeometry, StftSpec
from neuro.predictor.cnn import ObservableCNNModel as JaxObservableCNN
from neuro.predictor.cnn import WaveformCNNModel as JaxWaveformCNN
from neuro.predictor.losses import (
    CurriculumMSE as JaxCurriculumMSE,
)
from neuro.predictor.losses import (
    EegMsLoss as JaxEegMsLoss,
)
from neuro.predictor.losses import (
    LossContext as JaxLossContext,
)
from neuro.predictor.losses import (
    StftLoss as JaxStftLoss,
)
from neuro.predictor.mlp import ObservableMLPModel as JaxObservableMLP
from neuro.predictor.mlp import WaveformMLPModel as JaxWaveformMLP
from neuro.predictor.module import AutoregressiveCNN as TorchCNN
from neuro.predictor.module import AutoregressiveMLP as TorchMLP
from neuro.spectral import LOG_FLOOR
from neuro.transforms import Standardizer


@pytest.fixture(autouse=True)
def _enable_x64() -> Iterator[None]:
    """Run cross-framework comparison tests in double precision where appropriate."""
    with jax.enable_x64():
        yield


def test_waveform_mlp_forward_parity() -> None:
    """Waveform MLP unrolled rollouts match between PyTorch and JAX to float32 tolerance."""
    rng = np.random.default_rng(42)
    n_y, n_u, horizon, n_c, n_m, hidden = 3, 2, 5, 4, 2, 8

    t_model = TorchMLP(
        n_y=n_y,
        n_u=n_u,
        horizon=horizon,
        n_channels=n_c,
        n_controls=n_m,
        n_outputs=n_c,
        hidden_size=hidden,
        depth=1,
        activation="tanh",
        residual=True,
    )
    meta, arrays = t_model.to_checkpoint()
    j_model = JaxWaveformMLP.from_checkpoint(meta, arrays)

    B = 4
    y_hist = rng.standard_normal((B, n_y, n_c)).astype(np.float32)
    u_hist = rng.standard_normal((B, n_u, n_m)).astype(np.float32)
    u_future = rng.standard_normal((B, horizon, n_m)).astype(np.float32)

    with torch.no_grad():
        t_out = t_model(torch.as_tensor(y_hist), torch.as_tensor(u_hist), torch.as_tensor(u_future)).numpy()
    j_out = np.asarray(j_model.rollout(jnp.asarray(y_hist), jnp.asarray(u_hist), jnp.asarray(u_future)))

    np.testing.assert_allclose(j_out, t_out, rtol=1e-5, atol=1e-6)


def test_observable_mlp_forward_parity() -> None:
    """Observable MLP unrolled rollouts match between PyTorch and JAX to float32 tolerance."""
    rng = np.random.default_rng(43)
    geometry = StftGeometry(n_segment=16, n_hop=4, band_hz=(4.0, 20.0), n_bin_pool=2)
    n_vals = geometry.n_values(50.0)
    n_y, n_u, horizon, n_c, n_m, hidden = 2, 2, 4, 3, 2, 8
    n_out = n_c * n_vals

    t_model = TorchMLP(
        n_y=n_y,
        n_u=n_u,
        horizon=horizon,
        n_channels=n_c,
        n_controls=n_m,
        n_outputs=n_out,
        hidden_size=hidden,
        depth=1,
        activation="relu",
        residual=True,
        geometry=geometry,
    )
    meta, arrays = t_model.to_checkpoint()
    j_model = JaxObservableMLP.from_checkpoint(meta, arrays)

    B = 3
    y_hist = rng.standard_normal((B, n_y, n_c, n_vals)).astype(np.float32)
    u_hist = rng.standard_normal((B, n_u, n_m)).astype(np.float32)
    u_future = rng.standard_normal((B, horizon, n_m)).astype(np.float32)

    with torch.no_grad():
        t_out = t_model(torch.as_tensor(y_hist), torch.as_tensor(u_hist), torch.as_tensor(u_future)).numpy()
    j_out = np.asarray(j_model.rollout(jnp.asarray(y_hist), jnp.asarray(u_hist), jnp.asarray(u_future)))

    np.testing.assert_allclose(j_out, t_out, rtol=1e-5, atol=1e-6)


def test_waveform_cnn_forward_parity() -> None:
    """Waveform CNN unrolled rollouts match between PyTorch and JAX to float32 tolerance."""
    rng = np.random.default_rng(44)
    n_y, n_u, horizon, n_c, n_m, hidden = 4, 3, 4, 3, 2, 6

    t_model = TorchCNN(
        n_y=n_y,
        n_u=n_u,
        horizon=horizon,
        n_channels=n_c,
        n_controls=n_m,
        hidden_size=hidden,
        depth=2,
        kernel_size=3,
        activation="tanh",
        residual=True,
    )
    meta, arrays = t_model.to_checkpoint()
    j_model = JaxWaveformCNN.from_checkpoint(meta, arrays)

    B = 2
    y_hist = rng.standard_normal((B, n_y, n_c)).astype(np.float32)
    u_hist = rng.standard_normal((B, n_u, n_m)).astype(np.float32)
    u_future = rng.standard_normal((B, horizon, n_m)).astype(np.float32)

    with torch.no_grad():
        t_out = t_model(torch.as_tensor(y_hist), torch.as_tensor(u_hist), torch.as_tensor(u_future)).numpy()
    j_out = np.asarray(j_model.rollout(jnp.asarray(y_hist), jnp.asarray(u_hist), jnp.asarray(u_future)))

    np.testing.assert_allclose(j_out, t_out, rtol=1e-5, atol=1e-6)


def test_observable_cnn_forward_parity() -> None:
    """Observable CNN unrolled rollouts match between PyTorch and JAX to float32 tolerance."""
    rng = np.random.default_rng(45)
    geometry = StftGeometry(n_segment=16, n_hop=4, band_hz=(4.0, 20.0), n_bin_pool=2)
    n_vals = geometry.n_values(50.0)
    n_y, n_u, horizon, n_c, n_m, hidden = 3, 2, 4, 2, 2, 6

    n_out = n_c * n_vals
    t_model = TorchCNN(
        n_y=n_y,
        n_u=n_u,
        horizon=horizon,
        n_channels=n_c,
        n_controls=n_m,
        n_outputs=n_out,
        hidden_size=hidden,
        depth=2,
        kernel_size=3,
        frequency_kernel_size=3,
        activation="relu",
        residual=True,
        geometry=geometry,
    )
    meta, arrays = t_model.to_checkpoint()
    j_model = JaxObservableCNN.from_checkpoint(meta, arrays)

    B = 2
    y_hist = rng.standard_normal((B, n_y, n_c, n_vals)).astype(np.float32)
    u_hist = rng.standard_normal((B, n_u, n_m)).astype(np.float32)
    u_future = rng.standard_normal((B, horizon, n_m)).astype(np.float32)

    with torch.no_grad():
        t_out = t_model(torch.as_tensor(y_hist), torch.as_tensor(u_hist), torch.as_tensor(u_future)).numpy()
    j_out = np.asarray(j_model.rollout(jnp.asarray(y_hist), jnp.asarray(u_hist), jnp.asarray(u_future)))

    np.testing.assert_allclose(j_out, t_out, rtol=1e-5, atol=1e-6)


def _torch_curriculum_mse(pred: torch.Tensor, true: torch.Tensor, L: int) -> float:
    return float(torch.mean((pred[:, :L] - true[:, :L]) ** 2))


def _torch_stft_spectrogram(x: torch.Tensor, n_segment: int, n_hop: int, fs: float) -> torch.Tensor:
    segments = x.unfold(dimension=-1, size=n_segment, step=n_hop)
    window = torch.hann_window(n_segment, periodic=True, dtype=x.dtype, device=x.device)
    spectrum = torch.fft.rfft(segments * window, n=n_segment, dim=-1)
    psd = (spectrum.real**2 + spectrum.imag**2) / (fs * (window**2).sum())
    fold = torch.full((psd.shape[-1],), 2.0, dtype=psd.dtype, device=psd.device)
    fold[0] = 1.0
    if n_segment % 2 == 0:
        fold[-1] = 1.0
    return psd * fold


def _torch_stft_loss(pred: torch.Tensor, true: torch.Tensor, geom: StftGeometry, fs: float) -> float:
    bin_lo, bin_hi = geom.bin_range(fs)
    p_pred = _torch_stft_spectrogram(pred.movedim(1, -1), geom.n_segment, geom.n_hop, fs=fs)[..., bin_lo:bin_hi]
    p_true = _torch_stft_spectrogram(true.movedim(1, -1), geom.n_segment, geom.n_hop, fs=fs)[..., bin_lo:bin_hi]
    if geom.n_bin_pool > 1:
        n_g = p_pred.shape[-1] // geom.n_bin_pool
        p_pred = p_pred[..., : n_g * geom.n_bin_pool].reshape(*p_pred.shape[:-1], n_g, geom.n_bin_pool).mean(dim=-1)
        p_true = p_true[..., : n_g * geom.n_bin_pool].reshape(*p_true.shape[:-1], n_g, geom.n_bin_pool).mean(dim=-1)
    log_p_pred = torch.log(p_pred + LOG_FLOOR)
    log_p_true = torch.log(p_true + LOG_FLOOR)
    return float(torch.mean((log_p_pred - log_p_true) ** 2))


def _torch_eeg_ms_loss(pred: torch.Tensor, true: torch.Tensor, w_size: int, w_step: int) -> float:
    m_pred = pred.movedim(1, -1).unfold(dimension=-1, size=w_size, step=w_step).pow(2).mean(dim=-1)
    m_true = true.movedim(1, -1).unfold(dimension=-1, size=w_size, step=w_step).pow(2).mean(dim=-1)
    log_r = torch.log(m_pred + LOG_FLOOR) - torch.log(m_true + LOG_FLOOR)
    return float(torch.mean(log_r**2))


def test_curriculum_mse_loss_parity() -> None:
    """CurriculumMSE evaluates to identical values between PyTorch reference and JAX."""
    rng = np.random.default_rng(46)
    B, T, C = 4, 10, 3
    pred = rng.standard_normal((B, T, C)).astype(np.float32)
    true = rng.standard_normal((B, T, C)).astype(np.float32)

    spec = CurriculumMSESpec(weight=1.5, span_s=0.2, curr_start=0, curr_end=50)
    j_loss = JaxCurriculumMSE.from_spec(spec, fs=50.0)
    j_ctx = JaxLossContext(y_center=jnp.zeros(C), y_scale=jnp.ones(C), fs=50.0, epoch=25)

    L = j_loss.trusted_length(25)
    t_val = _torch_curriculum_mse(torch.as_tensor(pred), torch.as_tensor(true), L)
    j_val, j_diag = j_loss(jnp.asarray(pred), jnp.asarray(true), j_ctx)

    np.testing.assert_allclose(float(j_val), t_val, rtol=1e-6, atol=1e-7)
    assert j_diag["L"] == L


def test_stft_loss_parity() -> None:
    """StftLoss evaluates to identical values between PyTorch reference and JAX."""
    rng = np.random.default_rng(47)
    B, T, C = 2, 32, 2
    pred = rng.standard_normal((B, T, C)).astype(np.float32)
    true = rng.standard_normal((B, T, C)).astype(np.float32)

    spec = StftSpec(weight=1.0, n_span=32, n_segment=16, n_hop=4)
    j_loss = JaxStftLoss.from_spec(spec, fs=50.0)
    j_ctx = JaxLossContext(y_center=jnp.zeros(C), y_scale=jnp.ones(C), fs=50.0, epoch=0)

    t_val = _torch_stft_loss(torch.as_tensor(pred), torch.as_tensor(true), spec.geometry(), 50.0)
    j_val, _ = j_loss(jnp.asarray(pred), jnp.asarray(true), j_ctx)

    np.testing.assert_allclose(float(j_val), t_val, rtol=1e-5, atol=1e-6)


def test_eeg_ms_loss_parity() -> None:
    """EegMsLoss evaluates to identical values between PyTorch reference and JAX."""
    rng = np.random.default_rng(48)
    B, T, C = 3, 20, 2
    pred = rng.standard_normal((B, T, C)).astype(np.float32)
    true = rng.standard_normal((B, T, C)).astype(np.float32)

    spec = EegMsSpec(weight=1.0, span_s=0.4, window_s=0.1, hop_s=0.04)
    j_loss = JaxEegMsLoss.from_spec(spec, fs=50.0)
    j_ctx = JaxLossContext(y_center=jnp.zeros(C), y_scale=jnp.ones(C), fs=50.0, epoch=0)

    t_val = _torch_eeg_ms_loss(torch.as_tensor(pred), torch.as_tensor(true), w_size=5, w_step=2)
    j_val, _ = j_loss(jnp.asarray(pred), jnp.asarray(true), j_ctx)

    np.testing.assert_allclose(float(j_val), t_val, rtol=1e-6, atol=1e-7)


def test_bptt_gradient_and_adamw_step_parity() -> None:
    """BPTT gradients and one step of AdamW update match between PyTorch and JAX."""
    rng = np.random.default_rng(49)
    n_y, n_u, horizon, n_c, n_m, hidden = 2, 2, 3, 2, 1, 4

    t_model = TorchMLP(
        n_y=n_y,
        n_u=n_u,
        horizon=horizon,
        n_channels=n_c,
        n_controls=n_m,
        n_outputs=n_c,
        hidden_size=hidden,
        depth=1,
        activation="tanh",
        residual=False,
    )
    meta, arrays = t_model.to_checkpoint()
    j_model = JaxWaveformMLP.from_checkpoint(meta, arrays)

    B = 2
    y_hist = rng.standard_normal((B, n_y, n_c)).astype(np.float32)
    u_hist = rng.standard_normal((B, n_u, n_m)).astype(np.float32)
    u_future = rng.standard_normal((B, horizon, n_m)).astype(np.float32)
    target = rng.standard_normal((B, horizon, n_c)).astype(np.float32)

    # 1. PyTorch gradient
    t_opt = torch.optim.AdamW(t_model.parameters(), lr=1e-3, weight_decay=1e-2)
    t_pred = t_model(torch.as_tensor(y_hist), torch.as_tensor(u_hist), torch.as_tensor(u_future))
    t_loss = torch.mean((t_pred - torch.as_tensor(target)) ** 2)
    t_opt.zero_grad()
    t_loss.backward()

    # 2. JAX gradient
    def loss_fn(m: JaxWaveformMLP) -> jax.Array:
        p = m.rollout(jnp.asarray(y_hist), jnp.asarray(u_hist), jnp.asarray(u_future))
        return jnp.mean((p - jnp.asarray(target)) ** 2)

    j_loss, j_grads = eqx.filter_value_and_grad(loss_fn)(j_model)

    # Check loss parity
    np.testing.assert_allclose(float(j_loss), float(t_loss.detach()), rtol=1e-5, atol=1e-6)

    # Check gradient parity for layer 0 and layer 1
    t_params = [p for p in t_model.parameters() if p.requires_grad]
    assert t_params[0].grad is not None
    assert t_params[1].grad is not None
    assert t_params[2].grad is not None
    assert t_params[3].grad is not None
    t_g_w0 = t_params[0].grad.numpy()  # layer.0.weight
    t_g_b0 = t_params[1].grad.numpy()  # layer.0.bias
    t_g_w1 = t_params[2].grad.numpy()  # layer.1.weight
    t_g_b1 = t_params[3].grad.numpy()  # layer.1.bias

    np.testing.assert_allclose(np.asarray(j_grads.weights[0]), t_g_w0, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(np.asarray(j_grads.biases[0]), t_g_b0, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(np.asarray(j_grads.weights[1]), t_g_w1, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(np.asarray(j_grads.biases[1]), t_g_b1, rtol=1e-4, atol=1e-5)

    # 3. One AdamW step parity
    t_opt.step()

    j_opt = optax.adamw(learning_rate=1e-3, weight_decay=1e-2)
    opt_state = j_opt.init(eqx.filter(j_model, eqx.is_array))
    updates, _ = j_opt.update(j_grads, opt_state, eqx.filter(j_model, eqx.is_array))
    j_updated = eqx.apply_updates(j_model, updates)

    # Compare updated weights
    t_w0_new = t_params[0].detach().numpy()
    j_w0_new = np.asarray(j_updated.weights[0])
    np.testing.assert_allclose(j_w0_new, t_w0_new, rtol=1e-5, atol=1e-6)
