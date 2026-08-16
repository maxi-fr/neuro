from __future__ import annotations

import numpy as np
import pytest

from neuro.metrics import (
    BASELINE_FINE_HOP_S,
    BASELINE_FINE_UNTIL_S,
    DEFAULT_HOP_S,
    METRICS,
    Metric,
    baseline_grid,
    envelope,
    sample_at,
    seizure_state,
    windowed,
)

_FS = 1000.0
_RNG = np.random.default_rng(0)


def _sine(freq_hz: float, duration_s: float, amplitude: float = 1.0, n_channels: int = 4) -> np.ndarray:
    t = np.arange(round(duration_s * _FS)) / _FS
    return np.tile(amplitude * np.sin(2.0 * np.pi * freq_hz * t), (n_channels, 1))


# --- windowing semantics -------------------------------------------------------------------


def test_times_are_window_end_so_the_first_lands_exactly_on_the_window_length() -> None:
    times, _ = windowed(np.zeros((2, 1000)), _FS, lambda b, _fs: b.sum(axis=1), window_s=0.1, hop_s=0.05)

    assert times[0] == pytest.approx(0.1)
    assert np.allclose(np.diff(times), 0.05)


def test_window_count_matches_the_trailing_grid() -> None:
    # 1.0 s of samples, 0.1 s window, 0.05 s hop -> starts at 0, 50, ..., 900 -> 19 windows.
    times, values = windowed(np.zeros((2, 1000)), _FS, lambda b, _fs: b.sum(axis=1), window_s=0.1, hop_s=0.05)

    assert len(times) == values.shape[1] == 19


def test_windows_are_causal_and_never_see_the_future() -> None:
    # Silent for the first second, then a unit step. A trailing window ending at t <= 1.0 s
    # can only contain silence; a non-causal (centred) window would leak the step backwards.
    signals = np.zeros((3, 2000))
    signals[:, 1000:] = 1.0

    times, values = windowed(signals, _FS, METRICS["eeg_ms"].reduce, window_s=0.1, hop_s=0.05)

    assert np.all(values[:, times <= 1.0] == 0.0)
    assert values[:, -1] == pytest.approx(1.0)


def test_the_first_window_covering_the_step_is_the_one_ending_just_after_it() -> None:
    signals = np.zeros((1, 2000))
    signals[:, 1000:] = 1.0

    times, values = windowed(signals, _FS, METRICS["eeg_ms"].reduce, window_s=0.1, hop_s=0.05)
    first_nonzero = times[np.flatnonzero(values[0] > 0.0)[0]]

    assert first_nonzero == pytest.approx(1.05)


def test_a_signal_shorter_than_the_window_yields_an_empty_grid() -> None:
    times, values = windowed(np.zeros((2, 50)), _FS, lambda b, _fs: np.ptp(b, axis=1), window_s=0.1, hop_s=0.05)

    assert times.shape == (0,)
    assert values.shape == (2, 0)


def test_a_signal_exactly_one_window_long_yields_one_point() -> None:
    times, values = windowed(np.zeros((2, 100)), _FS, lambda b, _fs: b.sum(axis=1), window_s=0.1, hop_s=0.05)

    assert len(times) == values.shape[1] == 1
    assert times[0] == pytest.approx(0.1)


def test_each_window_sees_exactly_window_samples() -> None:
    widths: list[int] = []
    windowed(
        np.zeros((2, 1000)),
        _FS,
        lambda b, _fs: widths.append(b.shape[1]) or np.zeros(b.shape[0]),
        window_s=0.1,
        hop_s=0.05,
    )

    assert set(widths) == {100}


# --- individual metrics --------------------------------------------------------------------


def test_block_ptp_of_a_sine_is_twice_its_amplitude() -> None:
    _, values = METRICS["block_ptp"](_sine(20.0, 1.0, amplitude=3.0), _FS)

    assert values == pytest.approx(6.0, rel=1e-2)


def test_eeg_ms_of_a_sine_is_half_its_squared_amplitude() -> None:
    _, values = METRICS["eeg_ms"](_sine(20.0, 1.0, amplitude=2.0), _FS)

    assert values == pytest.approx(2.0, rel=1e-2)


def test_line_length_is_a_per_sample_density_so_it_does_not_grow_with_the_window() -> None:
    signals = _sine(10.0, 3.0)
    short = Metric("ll_short", 0.1, METRICS["line_length"].reduce)
    long = Metric("ll_long", 0.5, METRICS["line_length"].reduce)

    _, short_values = short(signals, _FS)
    _, long_values = long(signals, _FS)

    assert np.median(short_values) == pytest.approx(np.median(long_values), rel=1e-2)


def test_line_length_grows_with_frequency_at_fixed_amplitude() -> None:
    _, slow = METRICS["line_length"](_sine(5.0, 2.0), _FS)
    _, fast = METRICS["line_length"](_sine(20.0, 2.0), _FS)

    assert np.median(fast) > 3.0 * np.median(slow)


def test_band_power_passes_an_in_band_tone_and_rejects_an_out_of_band_one() -> None:
    _, in_band = METRICS["band_power"](_sine(6.0, 3.0), _FS)
    _, out_of_band = METRICS["band_power"](_sine(40.0, 3.0), _FS)

    assert np.median(in_band) > 100.0 * np.median(out_of_band)


def test_fc_strength_separates_identical_channels_from_independent_ones() -> None:
    identical = np.tile(_RNG.normal(size=(1, 3000)), (6, 1))
    independent = _RNG.normal(size=(6, 3000))

    _, locked = METRICS["fc_strength"](identical, _FS)
    _, unlocked = METRICS["fc_strength"](independent, _FS)

    assert np.median(locked) == pytest.approx(1.0, abs=1e-6)
    assert abs(np.median(unlocked)) < 0.2


def test_fc_strength_is_blind_to_the_sign_a_channel_sees_a_shared_source_with() -> None:
    """The EEG forward operator is signed: an inverted channel is synchronous, not unrelated."""
    source = _RNG.normal(size=(1, 3000))
    flipped = np.tile(source, (6, 1)) * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])[:, None]

    _, values = METRICS["fc_strength"](flipped, _FS)

    assert np.median(values) == pytest.approx(1.0, abs=1e-6)


def test_spectral_centroid_recovers_the_frequency_of_a_pure_tone() -> None:
    _, values = METRICS["spectral_centroid"](_sine(10.0, 3.0), _FS)

    assert np.median(values) == pytest.approx(10.0, abs=1.5)


def test_spectral_centroid_rises_when_the_tone_does() -> None:
    _, low = METRICS["spectral_centroid"](_sine(6.0, 3.0), _FS)
    _, high = METRICS["spectral_centroid"](_sine(30.0, 3.0), _FS)

    assert np.median(high) > np.median(low) + 10.0


# --- registry ------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(METRICS))
def test_every_metric_returns_a_finite_series_on_the_same_grid(name: str) -> None:
    signals = _RNG.normal(size=(8, 3000)) + _sine(8.0, 3.0, n_channels=8)

    times, values = METRICS[name](signals, _FS)

    assert values.shape == (8, len(times))
    assert len(times) > 0
    assert np.all(np.isfinite(values))
    assert times[0] == pytest.approx(METRICS[name].window_s)
    assert np.allclose(np.diff(times), DEFAULT_HOP_S)


@pytest.mark.parametrize("name", sorted(METRICS))
def test_every_metric_accepts_an_arbitrary_channel_subset(name: str) -> None:
    signals = _RNG.normal(size=(62, 3000)) + _sine(8.0, 3.0, n_channels=62)
    subset = np.array([27, 55, 12, 14])

    times_all, _ = METRICS[name](signals, _FS)
    times_subset, values_subset = METRICS[name](signals[subset], _FS)

    assert np.array_equal(times_all, times_subset)
    assert np.all(np.isfinite(values_subset))


@pytest.mark.parametrize("name", sorted(METRICS))
def test_registry_keys_match_the_metric_names(name: str) -> None:
    assert METRICS[name].name == name


# --- the raw-signal baselines --------------------------------------------------------------

_SETTLED = slice(round(1.5 * _FS), None)


def test_envelope_recovers_the_amplitude_of_an_in_band_tone() -> None:
    """The pi/2 scaling is what makes this the amplitude rather than the mean-rectified value."""
    values = envelope(_sine(7.0, 4.0, amplitude=2.5), _FS)

    assert np.mean(values[:, _SETTLED]) == pytest.approx(2.5, rel=0.02)


def test_envelope_follows_an_amplitude_step() -> None:
    t = np.arange(round(8.0 * _FS)) / _FS
    signals = np.tile((1.0 + 2.0 * (t > 4.0)) * np.sin(2.0 * np.pi * 7.0 * t), (3, 1))

    values = envelope(signals, _FS)

    assert np.mean(values[:, round(3.5 * _FS)]) == pytest.approx(1.0, rel=0.02)
    assert np.mean(values[:, round(7.5 * _FS)]) == pytest.approx(3.0, rel=0.02)


def test_envelope_rejects_a_tone_outside_the_band() -> None:
    in_band = envelope(_sine(7.0, 4.0), _FS)
    out_of_band = envelope(_sine(60.0, 4.0), _FS)

    assert np.mean(out_of_band[:, _SETTLED]) < 0.01 * np.mean(in_band[:, _SETTLED])


def test_envelope_is_causal_so_a_burst_casts_no_shadow_before_itself() -> None:
    """A Hilbert envelope would leak the burst backwards -- and inflate the baseline it feeds."""
    signals = _sine(7.0, 4.0)
    signals[:, : round(2.0 * _FS)] = 0.0

    values = envelope(signals, _FS)

    assert np.max(values[:, : round(2.0 * _FS)]) < 1e-9
    assert np.mean(values[:, round(3.5 * _FS) :]) == pytest.approx(1.0, rel=0.02)


def test_baseline_grid_is_fine_early_and_the_shared_grid_later() -> None:
    grid = baseline_grid(3.0, hop_s=DEFAULT_HOP_S)
    fine, coarse = grid[grid < BASELINE_FINE_UNTIL_S], grid[grid >= BASELINE_FINE_UNTIL_S]

    assert grid[0] == pytest.approx(BASELINE_FINE_HOP_S)
    assert np.allclose(np.diff(fine), BASELINE_FINE_HOP_S)
    assert np.allclose(np.diff(coarse), DEFAULT_HOP_S)
    assert grid[-1] == pytest.approx(3.0)


def test_sample_at_reads_the_instant_a_windowed_value_would_be_stamped() -> None:
    """Sample ``j`` is stamped at ``(j + 1) / fs``, the same right-edge convention as `windowed`."""
    signals = np.zeros((2, 2000))
    signals[:, 1000:] = 1.0

    values = sample_at(signals, _FS, np.array([0.999, 1.0, 1.001]))

    assert np.array_equal(values, [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])


def test_sample_at_keeps_the_channel_axis() -> None:
    signals = _RNG.normal(size=(62, 3000))

    values = sample_at(signals, _FS, baseline_grid(3.0))

    assert values.shape == (62, len(baseline_grid(3.0)))


def test_seizure_state_times_are_branch_referenced_and_start_at_zero() -> None:
    """The region LFP leads the branch by one window, so the first causal point is lookahead 0."""
    times, state = seizure_state(np.zeros((76, 4000)), _FS, window_s=1.0, hop_s=0.05)

    assert times[0] == pytest.approx(0.0)
    assert times[-1] == pytest.approx(3.0)
    assert len(times) == len(state) == 61
