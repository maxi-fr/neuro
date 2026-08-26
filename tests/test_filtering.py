from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import sosfreqz

from neuro.config import StftGeometry
from neuro.filtering import (
    AntiAliasEstimator,
    LowPassEstimator,
    ObservableEstimator,
    antialias_filter,
    causal_filter,
    design_bandpass_sos,
    design_lowpass_sos,
    group_delay_s,
    lowpass_filter,
)
from neuro.spectral import compute_log_power_frames

_SEED = 7
_FS = 1e4
_DOWNSAMPLE = 100


def _plant_signal(n_steps: int, n_channels: int) -> np.ndarray:
    """Broadband multichannel signal at the plant rate."""
    rng = np.random.default_rng(_SEED)
    t = np.arange(n_steps) / _FS
    slow = np.sin(2 * np.pi * 8.0 * t)[:, None] * np.linspace(1.0, 2.0, n_channels)
    fast = np.sin(2 * np.pi * 900.0 * t)[:, None] * np.linspace(0.5, 1.5, n_channels)
    return slow + fast + 0.1 * rng.standard_normal((n_steps, n_channels))


def test_online_estimator_matches_offline_decimation() -> None:
    """Sample-by-sample filtering then ZOH striding equals ``antialias_filter`` then striding.

    This is the property the whole change rests on: the predictor is fit on the offline path and
    served by the online one, so they must produce the same decimated sequence.
    """
    n_steps, n_channels = 3000, 4
    y = _plant_signal(n_steps, n_channels)

    offline = antialias_filter(y, _FS, _DOWNSAMPLE)[::_DOWNSAMPLE]

    estimator = AntiAliasEstimator(dt=1.0 / _FS, downsample=_DOWNSAMPLE)
    u = np.zeros(3)
    online = np.stack([estimator.evaluate(k / _FS, y[k], u)[0] for k in range(n_steps)])[::_DOWNSAMPLE]

    assert online.shape == offline.shape
    np.testing.assert_allclose(online, offline, atol=1e-12)


def test_estimator_logs_the_filtered_measurement() -> None:
    """``x_hat`` is logged and is the filtered signal, not the raw measurement."""
    y = _plant_signal(500, 3)
    estimator = AntiAliasEstimator(dt=1.0 / _FS, downsample=_DOWNSAMPLE)
    u = np.zeros(3)

    logs = [estimator.evaluate(k / _FS, y[k], u)[1].x_hat for k in range(len(y))]

    np.testing.assert_allclose(np.stack(logs), antialias_filter(y, _FS, _DOWNSAMPLE), atol=1e-12)
    assert not np.allclose(np.stack(logs), y)


def test_lowpass_estimator_matches_offline_filter() -> None:
    """Sample-by-sample filtering with LowPassEstimator equals lowpass_filter offline."""
    cutoff_hz = 45.0
    n_steps, n_channels = 1000, 3
    y = _plant_signal(n_steps, n_channels)

    offline = lowpass_filter(y, _FS, cutoff_hz)

    estimator = LowPassEstimator(dt=1.0 / _FS, cutoff_hz=cutoff_hz)
    u = np.zeros(3)
    online = np.stack([estimator.evaluate(k / _FS, y[k], u)[0] for k in range(n_steps)])

    np.testing.assert_allclose(online, offline, atol=1e-12)


def test_lowpass_estimator_from_config_and_logging() -> None:
    """LowPassEstimator instantiates from config and logs x_hat."""
    estimator = LowPassEstimator.from_config({"dt": 1e-4, "cutoff_hz": 45.0})
    assert estimator.sos.shape == (2, 6)
    y_sample = np.array([1.0, 2.0])
    u_sample = np.zeros(2)
    x_hat, log = estimator.update(0.0, y_sample, u_sample)
    np.testing.assert_array_equal(x_hat, log.x_hat)


def test_observable_estimator_matches_offline_reduction() -> None:
    """ObservableEstimator emits Frames that equal the offline canonical reduction on the hop grid."""
    n_steps, n_channels = 6000, 3
    y = _plant_signal(n_steps, n_channels)
    fs_dec = _FS / _DOWNSAMPLE

    geometry = StftGeometry(
        n_segment=20,
        n_hop=5,
        band_hz=(4.0, 40.0),
        n_bin_pool=2,
        kernel="hann",
        kernel_width=3,
    )
    sample_support = (geometry.kernel_width - 1) * geometry.n_hop + geometry.n_segment

    offline_dec = antialias_filter(y, _FS, _DOWNSAMPLE)[::_DOWNSAMPLE]
    offline_frames = compute_log_power_frames(offline_dec, geometry, fs=fs_dec)
    assert len(offline_frames) > 0

    estimator = ObservableEstimator(dt=1.0 / _FS, geometry=geometry, downsample=_DOWNSAMPLE)
    u = np.zeros(2)
    online_frames = np.stack([estimator.evaluate(k / _FS, y[k], u)[0] for k in range(n_steps)])

    for i in range(len(offline_frames)):
        step_idx = (sample_support - 1 + i * geometry.n_hop) * _DOWNSAMPLE
        np.testing.assert_allclose(online_frames[step_idx], offline_frames[i], atol=1e-12)


def test_observable_estimator_warmup_and_hold() -> None:
    """ObservableEstimator emits unprimed NaN until sample support is reached, and holds between hops."""
    n_steps, n_channels = 4000, 2
    y = _plant_signal(n_steps, n_channels)
    geometry = StftGeometry(n_segment=15, n_hop=5, kernel="boxcar", kernel_width=2)
    sample_support = (geometry.kernel_width - 1) * geometry.n_hop + geometry.n_segment  # 20
    first_frame_step = (sample_support - 1) * _DOWNSAMPLE  # 1900

    estimator = ObservableEstimator(dt=1.0 / _FS, geometry=geometry, downsample=_DOWNSAMPLE)
    u = np.zeros(2)
    online_frames = [estimator.evaluate(k / _FS, y[k], u)[0] for k in range(n_steps)]

    # Before first frame: all NaN
    for k in range(first_frame_step):
        assert np.isnan(online_frames[k]).all()

    # At first frame: valid and finite
    assert np.isfinite(online_frames[first_frame_step]).all()

    # Held between hops: steps 1900 to 2399 hold the exact same frame
    next_frame_step = first_frame_step + geometry.n_hop * _DOWNSAMPLE  # 2400
    for k in range(first_frame_step, next_frame_step):
        np.testing.assert_array_equal(online_frames[k], online_frames[first_frame_step])

    # At next hop: fresh frame
    assert not np.array_equal(online_frames[next_frame_step], online_frames[first_frame_step])


def test_observable_estimator_from_config_and_logging() -> None:
    """ObservableEstimator instantiates from config and logs x_hat."""
    cfg = {
        "dt": 1e-4,
        "downsample": 100,
        "geometry": {
            "n_segment": 20,
            "n_hop": 5,
            "band_hz": [4.0, 30.0],
            "n_bin_pool": 1,
            "kernel": "boxcar",
            "kernel_width": 1,
        },
    }
    estimator = ObservableEstimator.from_config(cfg)
    assert estimator.sample_support == 20
    y_sample = np.array([1.0, 2.0, 3.0])
    u_sample = np.zeros(2)
    x_hat, log = estimator.update(0.0, y_sample, u_sample)
    np.testing.assert_array_equal(x_hat, log.x_hat)


def test_filter_is_causal() -> None:
    """Samples after ``k`` cannot change the output at ``k`` (``sosfilt``, not ``sosfiltfilt``)."""
    y = _plant_signal(1000, 2)
    cut = 400
    y_perturbed = y.copy()
    y_perturbed[cut:] += 50.0

    filtered = antialias_filter(y, _FS, _DOWNSAMPLE)
    filtered_perturbed = antialias_filter(y_perturbed, _FS, _DOWNSAMPLE)

    np.testing.assert_allclose(filtered[:cut], filtered_perturbed[:cut], atol=1e-12)
    assert not np.allclose(filtered[cut:], filtered_perturbed[cut:])


def test_filter_starts_from_zero_state() -> None:
    """Both paths start the filter at zero state, so a prefix of the signal fixes its own output."""
    y = _plant_signal(1000, 2)
    np.testing.assert_allclose(
        antialias_filter(y[:300], _FS, _DOWNSAMPLE),
        antialias_filter(y, _FS, _DOWNSAMPLE)[:300],
        atol=1e-12,
    )


def test_design_is_low_pass_at_the_decimated_nyquist() -> None:
    """The response is flat at DC, -3 dB at the new Nyquist and negligible at the old one."""
    nyquist_new = _FS / (2 * _DOWNSAMPLE)
    sos = design_lowpass_sos(_FS, nyquist_new)

    _, h = sosfreqz(sos, worN=[0.0, nyquist_new, _FS / 2], fs=_FS)
    gain = np.abs(h)

    np.testing.assert_allclose(gain[0], 1.0, atol=1e-12)
    np.testing.assert_allclose(gain[1], 1.0 / np.sqrt(2.0), rtol=1e-6)
    assert gain[2] < 1e-6


def test_downsample_one_is_a_passthrough() -> None:
    """No decimation means no anti-alias filter, so the signal is returned untouched."""
    y = _plant_signal(200, 2)
    np.testing.assert_array_equal(antialias_filter(y, _FS, 1), y)


# --- the bandwidth axis --------------------------------------------------------------------


def _channels_first(freq_hz: float, n_samples: int, n_channels: int = 3) -> np.ndarray:
    """A tone as ``(n_channels, n_samples)`` -- the layout the metric path filters in."""
    t = np.arange(n_samples) / _FS
    return np.tile(np.sin(2 * np.pi * freq_hz * t), (n_channels, 1))


def test_the_antialias_design_is_the_lowpass_design_at_the_implied_cutoff() -> None:
    """AntiAliasEstimator configures the lowpass design at the implied cutoff."""
    estimator = AntiAliasEstimator(dt=1.0 / _FS, downsample=_DOWNSAMPLE)
    np.testing.assert_array_equal(
        estimator.sos,
        design_lowpass_sos(_FS, _FS / (2 * _DOWNSAMPLE)),
    )


def test_causal_filter_opens_with_no_step_from_zero() -> None:
    """A constant comes back as itself everywhere.

    Zero-state ``sosfilt`` would ramp up from 0 over the first time constants. That opening
    transient is near-identical across rollouts branched from one parent, so it would inflate
    short-lookahead predictability for a reason that has nothing to do with the plant.
    """
    constant = np.full((3, 500), 5.0)

    np.testing.assert_allclose(causal_filter(constant, design_lowpass_sos(_FS, 45.0)), constant, atol=1e-9)


def test_causal_filter_never_sees_the_future() -> None:
    signals = _channels_first(8.0, 1000)
    perturbed = signals.copy()
    perturbed[:, 600:] += 50.0

    sos = design_lowpass_sos(_FS, 45.0)
    filtered, filtered_perturbed = causal_filter(signals, sos), causal_filter(perturbed, sos)

    np.testing.assert_allclose(filtered[:, :600], filtered_perturbed[:, :600], atol=1e-12)
    assert not np.allclose(filtered[:, 600:], filtered_perturbed[:, 600:])


def test_the_lowpass_keeps_a_tone_below_the_cutoff_and_kills_one_above() -> None:
    sos = design_lowpass_sos(_FS, 45.0)

    passed = causal_filter(_channels_first(8.0, 4000), sos)
    stopped = causal_filter(_channels_first(400.0, 4000), sos)

    assert np.ptp(passed[:, 2000:]) == pytest.approx(2.0, rel=0.05)
    assert np.ptp(stopped[:, 2000:]) < 0.01


def test_the_bandpass_admits_the_seizure_band_and_rejects_either_side() -> None:
    sos = design_bandpass_sos(_FS, (3.0, 12.0))
    settled = slice(20000, None)

    in_band = causal_filter(_channels_first(7.0, 30000), sos)
    below = causal_filter(_channels_first(0.5, 30000), sos)
    above = causal_filter(_channels_first(60.0, 30000), sos)

    assert np.ptp(in_band[:, settled]) == pytest.approx(2.0, rel=0.1)
    assert np.ptp(below[:, settled]) < 0.2
    assert np.ptp(above[:, settled]) < 0.2


def test_group_delay_is_positive_and_grows_as_the_cutoff_falls() -> None:
    """The latency a controller pays for bandwidth -- what stops the sweep bottoming out at DC."""
    delays = [group_delay_s(design_lowpass_sos(_FS, cutoff), _FS) for cutoff in (500.0, 100.0, 45.0, 10.0)]

    assert all(d > 0 for d in delays)
    assert delays == sorted(delays)


def test_group_delay_matches_the_measured_lag_of_a_slow_tone() -> None:
    """Measured against the thing it claims to predict, not against another formula."""
    sos = design_lowpass_sos(_FS, 45.0)
    signals = _channels_first(5.0, 20000)

    filtered = causal_filter(signals, sos)
    settled = slice(10000, 14000)
    lag = np.argmax([np.dot(signals[0, settled], np.roll(filtered[0], -shift)[settled]) for shift in range(200)])

    assert lag / _FS == pytest.approx(group_delay_s(sos, _FS), abs=1e-3)
