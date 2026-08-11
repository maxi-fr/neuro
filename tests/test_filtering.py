from __future__ import annotations

import numpy as np
from scipy.signal import sosfreqz

from neuro.filtering import AntiAliasEstimator, antialias_filter, design_antialias_sos

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
    sos = design_antialias_sos(_FS, _DOWNSAMPLE)
    nyquist_new = _FS / (2 * _DOWNSAMPLE)

    _, h = sosfreqz(sos, worN=[0.0, nyquist_new, _FS / 2], fs=_FS)
    gain = np.abs(h)

    np.testing.assert_allclose(gain[0], 1.0, atol=1e-12)
    np.testing.assert_allclose(gain[1], 1.0 / np.sqrt(2.0), rtol=1e-6)
    assert gain[2] < 1e-6


def test_downsample_one_is_a_passthrough() -> None:
    """No decimation means no anti-alias filter, so the signal is returned untouched."""
    y = _plant_signal(200, 2)
    np.testing.assert_array_equal(antialias_filter(y, _FS, 1), y)
