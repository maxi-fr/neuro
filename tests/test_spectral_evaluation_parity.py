from __future__ import annotations

from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neuro.config import StftGeometry
from neuro.predictor.data import reduce_trajectory_to_frames
from neuro.predictor.evaluation import evaluate_spectral_free_run
from neuro.predictor.losses import (
    CurriculumMSE,
    LossContext,
    StftLoss,
)
from neuro.spectral import (
    compute_forecast_log_power_frames,
    compute_log_power_frames,
)

if TYPE_CHECKING:
    from neuro.predictor.inference import InferencePredictor
    from neuro.types import FloatArray


def _stft_geometry() -> StftGeometry:
    return StftGeometry(
        n_segment=16,
        n_hop=4,
        band_hz=[2.0, 20.0],
        n_bin_pool=2,
        kernel="hann",
        kernel_width=3,
    )


def test_identical_log_power_produces_identical_loss_regardless_of_standardization() -> None:
    """Waveform STFT loss and unstandardized spectral loss produce identical error."""
    geom = _stft_geometry()
    fs = 50.0
    supp = geom.sample_support_steps(fs)
    prefix_len = supp - geom.n_hop
    n_frames = 5
    n_channels = 2
    span_steps = n_frames * geom.n_hop

    rng = np.random.default_rng(101)
    # Measurement history up to forecast boundary
    history_np = rng.standard_normal((prefix_len + 10, n_channels))
    # True future and predicted future EEG
    true_future_np = rng.standard_normal((span_steps, n_channels))
    pred_future_np = true_future_np + 0.1 * rng.standard_normal((span_steps, n_channels))

    # Compute ground truth and predicted log-power frames via waveform path
    true_frames = compute_forecast_log_power_frames(history_np, true_future_np, geom, fs=fs)
    pred_frames = compute_forecast_log_power_frames(history_np, pred_future_np, geom, fs=fs)
    assert true_frames.shape == (n_frames, n_channels, geom.n_values(fs))
    assert pred_frames.shape == (n_frames, n_channels, geom.n_values(fs))

    # Evaluate waveform spectral MSE directly
    waveform_mse = float(np.mean((pred_frames - true_frames) ** 2))

    # Now create arbitrary standardizers (per-coordinate center and scale)
    n_values = geom.n_values(fs)
    y_center = rng.uniform(-5.0, 5.0, size=(n_channels, n_values))
    y_scale = rng.uniform(0.5, 3.0, size=(n_channels, n_values))  # per-coordinate scales

    # Standardize predictions and targets for spectral predictor
    pred_std = (pred_frames - y_center) / y_scale
    true_std = (true_frames - y_center) / y_scale

    # Standardized MSE differs from unstandardized MSE because of per-coordinate scaling
    std_mse = float(np.mean((pred_std - true_std) ** 2))
    assert not np.isclose(std_mse, waveform_mse)

    # Recovered unstandardized spectral MSE must match waveform MSE to machine precision
    recovered_pred = pred_std * y_scale + y_center
    recovered_true = true_std * y_scale + y_center
    spectral_mse = float(np.mean((recovered_pred - recovered_true) ** 2))

    np.testing.assert_allclose(spectral_mse, waveform_mse, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("gapped", [False, True])
@pytest.mark.parametrize("window", ["hann", "hann_poisson"])
@pytest.mark.parametrize("asymmetric", [False, True])
def test_jax_unstandardized_spectral_loss_matches_stft_loss(window: str, *, asymmetric: bool, gapped: bool) -> None:
    """Both Loss paths agree for every taper when boundary history is aligned."""
    geom = StftGeometry(n_segment=4, n_hop=8) if gapped else _stft_geometry()
    fs = 50.0
    supp = geom.sample_support_steps(fs)
    prefix_len = max(0, supp - geom.n_hop)
    n_frames = 6
    n_channels = 2
    span_steps = n_frames * geom.n_hop

    geom = geom.model_copy(update={"window": window, "asymmetric_window": asymmetric})
    rng = np.random.default_rng(202)
    history_np = rng.standard_normal((prefix_len, n_channels))
    true_future_np = rng.standard_normal((span_steps, n_channels))
    pred_future_np = true_future_np + 0.2 * rng.standard_normal((span_steps, n_channels))

    # Compute target and predicted frames
    true_frames_np = compute_forecast_log_power_frames(history_np, true_future_np, geom, fs=fs)
    pred_frames_np = compute_forecast_log_power_frames(history_np, pred_future_np, geom, fs=fs)

    # Setup JAX tensors
    y_center_np = rng.uniform(-2.0, 2.0, size=(n_channels, geom.n_values(fs)))
    y_scale_np = rng.uniform(0.8, 2.5, size=(n_channels, geom.n_values(fs)))

    # Waveform context: dummy raw EEG center/scale = 0/1
    ctx_waveform = LossContext(
        y_center=jnp.zeros(n_channels),
        y_scale=jnp.ones(n_channels),
        fs=fs,
        epoch=0,
        y_hist=jnp.asarray(history_np)[None, ...],
    )
    stft_loss_fn = StftLoss(
        weight=1.0,
        span_steps=span_steps,
        start_epoch=0,
        geometry=geom,
    )

    pred_eeg_torch = jnp.asarray(pred_future_np)[None, ...]
    true_eeg_torch = jnp.asarray(true_future_np)[None, ...]
    loss_waveform, _ = stft_loss_fn(pred_eeg_torch, true_eeg_torch, ctx_waveform)

    # Spectral context: carry log-power center and scale
    ctx_spectral = LossContext(
        y_center=jnp.asarray(y_center_np).reshape(-1),
        y_scale=jnp.asarray(y_scale_np).reshape(-1),
        fs=geom.frame_rate(fs),
        epoch=None,
    )
    spectral_loss_fn = CurriculumMSE(
        weight=1.0,
        span_steps=n_frames,
        start_epoch=0,
        curr_start=0,
        curr_end=0,
        unstandardized=True,
    )

    # Standardize frames as fed to spectral predictor
    pred_frames_flat = pred_frames_np.reshape(n_frames, -1)
    true_frames_flat = true_frames_np.reshape(n_frames, -1)
    y_center_flat = y_center_np.reshape(-1)
    y_scale_flat = y_scale_np.reshape(-1)

    pred_frames_std = jnp.asarray((pred_frames_flat - y_center_flat) / y_scale_flat)[None, ...]
    true_frames_std = jnp.asarray((true_frames_flat - y_center_flat) / y_scale_flat)[None, ...]

    loss_spectral, _ = spectral_loss_fn(pred_frames_std, true_frames_std, ctx_spectral)

    np.testing.assert_allclose(float(loss_waveform), float(loss_spectral), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("gapped", [False, True])
def test_aligned_frames_spanning_forecast_boundary(*, gapped: bool) -> None:
    """Frame 1 uses trailing history and forecast future, matching the continuous recording."""
    geom = _stft_geometry()
    fs = 50.0
    supp = geom.sample_support_steps(fs)
    prefix_len = supp - geom.n_hop
    if gapped:
        geom = StftGeometry(n_segment=4, n_hop=8)
        supp = geom.sample_support_steps(fs)
        prefix_len = max(0, supp - geom.n_hop)
    rng = np.random.default_rng(303)
    y_full = rng.standard_normal((100, 2))

    all_frames = compute_log_power_frames(y_full, geom, fs=fs)

    anchor_frame_idx = 5
    # Sample index corresponding to the end of anchor_frame_idx:
    anchor_sample_idx = supp - 1 + anchor_frame_idx * geom.n_hop
    Q = 4

    history = y_full[anchor_sample_idx - prefix_len + 1 : anchor_sample_idx + 1]
    future = y_full[anchor_sample_idx + 1 : anchor_sample_idx + 1 + Q * geom.n_hop]

    forecast_frames = compute_forecast_log_power_frames(history, future, geom, fs=fs)
    assert len(forecast_frames) == Q
    # Must match the exact consecutive frames from the continuous recording
    np.testing.assert_allclose(forecast_frames, all_frames[anchor_frame_idx + 1 : anchor_frame_idx + 1 + Q])


@pytest.mark.parametrize("gapped", [False, True])
def test_free_run_spectral_evaluation_parity(monkeypatch: pytest.MonkeyPatch, *, gapped: bool) -> None:
    """Both evaluator branches score equivalent predictions at identical physical anchors."""
    geom = StftGeometry(n_segment=4, n_hop=8) if gapped else _stft_geometry()
    fs = 50.0
    rng = np.random.default_rng(404)
    y_raw = rng.standard_normal((100, 2))
    u_raw = np.arange(100, dtype=np.float64)[:, None]
    eval_frames = 3
    eval_steps = eval_frames * geom.n_hop
    prefix = max(0, geom.sample_support_steps(fs) - geom.n_hop)
    anchors: list[list[int]] = []

    def waveform_prediction(t0: int) -> FloatArray:
        """Produce the same deterministic waveform forecast for either adapter."""
        return np.stack([y_raw[t0] * (0.95**i) for i in range(1, eval_steps + 1)])

    class MockWaveformModel:
        priming_steps = 4
        n_channels = 2
        dt = 1.0 / fs

        def free_run(self, y_hists: FloatArray, u_hists: FloatArray, u_futures: FloatArray) -> FloatArray:
            """Record forecast anchors and return deterministic EEG predictions."""
            del y_hists, u_hists
            t0s = u_futures[:, 0, 0].astype(int).tolist()
            anchors.append(t0s)
            return np.stack([waveform_prediction(t0) for t0 in t0s])

    class MockObservableModel:
        priming_steps = 4
        n_outputs = 2 * geom.n_values(fs)
        geometry = geom
        dt = geom.n_hop / fs

        def free_run(self, y_hists: FloatArray, u_hists: FloatArray, u_futures: FloatArray) -> FloatArray:
            """Return the log-power Frames of the equivalent waveform forecasts."""
            del y_hists, u_hists
            t0s = u_futures[:, 0, 0].astype(int).tolist()
            anchors.append(t0s)
            return np.stack(
                [
                    compute_forecast_log_power_frames(
                        y_raw[t0 - prefix + 1 : t0 + 1], waveform_prediction(t0), geom, fs=fs
                    )
                    for t0 in t0s
                ]
            )

    monkeypatch.setattr("neuro.predictor.evaluation.ObservableMLPModel", MockObservableModel)
    results = [
        evaluate_spectral_free_run(
            cast("InferencePredictor", model),
            [(u_raw, y_raw)],
            eval_frames,
            geometry=geom,
            fs=fs,
            step_stride=1,
            start_frame=4,
        )
        for model in (MockWaveformModel(), MockObservableModel())
    ]
    assert anchors[0] == anchors[1]
    assert anchors[0][0] == geom.sample_support_steps(fs) - 1 + 4 * geom.n_hop
    assert results[0].pooled > 0
    np.testing.assert_allclose(results[0].per_step, results[1].per_step, rtol=1e-12)
    assert results[0].pooled == pytest.approx(results[1].pooled)


def test_stft_rejects_insufficient_boundary_history() -> None:
    """Supplied measurement history must cover the first forecast Frame."""
    geom = _stft_geometry()
    loss = StftLoss(weight=1.0, span_steps=40, start_epoch=0, geometry=geom)
    ctx = LossContext(
        y_center=jnp.zeros(2),
        y_scale=jnp.ones(2),
        fs=50.0,
        epoch=None,
        y_hist=jnp.zeros((1, 4, 2)),
    )
    with pytest.raises(ValueError, match="history"):
        loss.log_spectrogram(jnp.ones((1, 40, 2)), ctx)


def test_curriculum_mse_unstandardized_matches_raw_mse_and_backpropagates() -> None:
    """CurriculumMSE with unstandardized=True matches unstandardized MSE and backpropagates."""
    n_frames = 4
    n_channels = 2
    n_values = 3
    n_features = n_channels * n_values

    rng = np.random.default_rng(505)
    pred_data = rng.standard_normal((2, n_frames, n_features))
    true_data = rng.standard_normal((2, n_frames, n_features))
    y_center = rng.uniform(-3.0, 3.0, size=(n_features,))
    y_scale = rng.uniform(0.5, 2.5, size=(n_features,))

    pred_tensor = jnp.asarray(pred_data, dtype=jnp.float32)
    true_tensor = jnp.asarray(true_data, dtype=jnp.float32)

    ctx = LossContext(
        y_center=jnp.asarray(y_center, dtype=jnp.float32),
        y_scale=jnp.asarray(y_scale, dtype=jnp.float32),
        fs=10.0,
        epoch=10,
    )

    curr_loss_fn = CurriculumMSE(
        weight=1.0,
        span_steps=n_frames,
        start_epoch=0,
        curr_start=0,
        curr_end=0,
        unstandardized=True,
    )

    loss_curr, _ = curr_loss_fn(pred_tensor, true_tensor, ctx)
    expected_unstd_loss = jnp.mean((ctx.to_raw(pred_tensor) - ctx.to_raw(true_tensor)) ** 2)

    assert float(loss_curr) == pytest.approx(float(expected_unstd_loss))

    # Verify backpropagation produces gradients scaled by y_scale ** 2
    gradient = jax.grad(lambda pred: curr_loss_fn(pred, true_tensor, ctx)[0])(pred_tensor)
    # Expected gradient d/d(pred) of 1/N * sum(( (pred - true) * scale ) ** 2)
    # is 2/N * (pred - true) * (scale ** 2)
    expected_grad = 2.0 / pred_data.size * (pred_data - true_data) * (y_scale**2)
    np.testing.assert_allclose(np.asarray(gradient), expected_grad, rtol=1e-5, atol=1e-5)


def test_waveform_default_start_converts_sample_priming_to_frames() -> None:
    """Sample Priming does not unnecessarily exclude short valid evaluation recordings."""
    geom = StftGeometry(n_segment=50, n_hop=5)
    rng = np.random.default_rng(606)
    y = rng.standard_normal((100, 2))
    u = np.arange(100, dtype=np.float64)[:, None]

    class Model:
        priming_steps = 50
        n_channels = 2
        dt = 0.02

        def free_run(self, y_hists: FloatArray, u_hists: FloatArray, u_futures: FloatArray) -> FloatArray:
            """Verify the earliest legal Frame anchor and repeat the latest EEG sample."""
            del u_hists
            assert u_futures[0, 0, 0] == 54
            return np.repeat(y_hists[:, -1:, :], u_futures.shape[1], axis=1)

    result = evaluate_spectral_free_run(cast("InferencePredictor", Model()), [(u, y)], 3, geometry=geom)
    assert np.isfinite(result.pooled)


def test_unstandardized_curriculum_mask_matches_prefix_under_jit() -> None:
    """Dynamic JAX curriculum masks preserve raw-unit Losses and their gradients."""
    pred = jnp.arange(12, dtype=jnp.float32).reshape(1, 4, 3)
    true = jnp.ones_like(pred)
    ctx = LossContext(y_center=jnp.zeros(3), y_scale=jnp.array([0.5, 2.0, 3.0]), fs=10.0)
    loss = CurriculumMSE(weight=1.0, span_steps=4, start_epoch=0, curr_start=0, curr_end=0, unstandardized=True)
    mask = jnp.array([1.0, 1.0, 0.0, 0.0])
    actual, gradient = jax.jit(jax.value_and_grad(lambda p: loss(p, true, ctx, step_mask=mask)[0]))(pred)
    expected, expected_gradient = jax.value_and_grad(lambda p: jnp.mean(((p[:, :2] - true[:, :2]) * ctx.y_scale) ** 2))(
        pred
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-6)
