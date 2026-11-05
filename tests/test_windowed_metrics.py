from __future__ import annotations

import numpy as np
import pytest

from experiments.observable_predictability.features import (
    CONNECTIVITY_CANDIDATE_WINDOWS_S,
    CONTROL_HORIZONS_S,
    REPRESENTATIONS,
    baseline_stft_geometry,
    compute_seizure_state,
    condition_waveform_pair,
    connectivity_pair_indices,
    extract_connectivity_frames,
    extract_spectral_frames,
    select_connectivity_support_window,
)
from experiments.observable_predictability.metrics import (
    RolloutIdentity,
    windowed,
)
from neuro.filtering import AntiAliasEstimator

_FS = 1000.0
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

    times, values = windowed(signals, _FS, lambda block, _: (block**2).mean(axis=1), window_s=0.1, hop_s=0.05)

    assert np.all(values[:, times <= 1.0] == 0.0)
    assert values[:, -1] == pytest.approx(1.0)


def test_the_first_window_covering_the_step_is_the_one_ending_just_after_it() -> None:
    signals = np.zeros((1, 2000))
    signals[:, 1000:] = 1.0

    times, values = windowed(signals, _FS, lambda block, _: (block**2).mean(axis=1), window_s=0.1, hop_s=0.05)
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


def test_baseline_stft_geometry_matches_november_protocol() -> None:
    geom = baseline_stft_geometry()

    assert geom.n_segment == 25
    assert geom.n_hop == 3
    assert geom.band_hz == (1.0, 25.0)
    assert geom.n_bin_pool == 1
    assert geom.kernel == "linear"
    assert geom.kernel_width == 1
    assert geom.window == "hann"
    assert geom.asymmetric_window is True
    assert geom.sample_support_steps(50.0) == 25
    assert geom.n_values(50.0) == 12


def test_condition_scalp_waveform_matches_antialias_estimator_and_continuous_join() -> None:
    rng = np.random.default_rng(17)
    fs_raw = 1000.0
    target_fs = 50.0
    downsample = 20
    n_hist_raw = 600  # 0.60 s -> 30 samples at 50 Hz
    n_cont_raw = 900  # 0.90 s -> 45 samples at 50 Hz
    n_channels = 4

    history_raw = rng.standard_normal((n_hist_raw, n_channels))
    continuation_raw = rng.standard_normal((n_cont_raw, n_channels))
    joined_raw = np.concatenate([history_raw, continuation_raw], axis=0)

    # 1. Split conditioning with history warm-up vs continuous join conditioning
    _, post_from_split = condition_waveform_pair(
        continuation_raw, fs_raw, history_raw=history_raw, cutoff_hz=25.0, target_fs=target_fs
    )
    _, full_from_joined = condition_waveform_pair(joined_raw, fs_raw, cutoff_hz=25.0, target_fs=target_fs)
    n_hist_50 = n_hist_raw // downsample

    assert post_from_split.shape == (n_cont_raw // downsample, n_channels)
    np.testing.assert_allclose(post_from_split, full_from_joined[n_hist_50:], rtol=1e-13, atol=1e-13)

    # 2. Sample-for-sample parity with online AntiAliasEstimator
    estimator = AntiAliasEstimator(dt=1.0 / fs_raw, downsample=downsample)
    u_zero = np.zeros(1)
    online_decimated: list[np.ndarray] = []
    for step, sample in enumerate(joined_raw):
        x_hat, _ = estimator.update(step / fs_raw, sample, u_zero)
        if (step + 1) % downsample == 0:
            online_decimated.append(x_hat.copy())
    online_post = np.stack(online_decimated[n_hist_50:], axis=0)
    np.testing.assert_allclose(post_from_split, online_post, rtol=1e-13, atol=1e-13)

    # 3. Already conditioned 50 Hz input passes through unchanged
    _, passthrough = condition_waveform_pair(post_from_split, target_fs)
    np.testing.assert_array_equal(passthrough, post_from_split)


def test_spectral_and_seizure_state_extraction_match_continuous_joined_signal() -> None:
    rng = np.random.default_rng(23)
    fs_50 = 50.0
    n_hist = 61  # 25 + 12 * 3 samples -> branch aligns with continuous Frame grid
    n_cont = 75  # 1.50 s at 50 Hz
    n_channels = 4

    history_50 = rng.standard_normal((n_hist, n_channels))
    cont_50 = rng.standard_normal((n_cont, n_channels))
    joined_50 = np.concatenate([history_50, cont_50], axis=0)

    times_split, frames_split = extract_spectral_frames(cont_50, fs_50, history=history_50)
    times_full, frames_full = extract_spectral_frames(joined_50, fs_50)

    t_branch = n_hist / fs_50
    post_mask = times_full > t_branch + 1e-9
    np.testing.assert_allclose(times_split, times_full[post_mask] - t_branch, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(frames_split, frames_full[post_mask], rtol=1e-12, atol=1e-12)

    # Regional LFP Seizure State continuity across branch boundary
    fs_lfp = 500.0
    lfp_hist = rng.uniform(-2.0, 2.0, size=(1000 + 8 * 30, 8))
    lfp_cont = rng.uniform(-5.0, 5.0, size=(round(1.50 * fs_lfp), 8))
    lfp_joined = np.concatenate([lfp_hist, lfp_cont], axis=0)

    s_times_split, s_split = compute_seizure_state(lfp_cont, fs_lfp, lfp_history=lfp_hist, hop_s=0.06, window_s=2.0)
    s_times_full, s_full = compute_seizure_state(lfp_joined, fs_lfp, hop_s=0.06, window_s=2.0)
    lfp_branch_s = lfp_hist.shape[0] / fs_lfp
    lfp_post_mask = s_times_full > lfp_branch_s + 1e-9

    np.testing.assert_allclose(s_times_split, s_times_full[lfp_post_mask] - lfp_branch_s, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(s_split, s_full[lfp_post_mask], rtol=1e-12, atol=1e-12)
    assert "seizure_state" not in REPRESENTATIONS
    assert REPRESENTATIONS == ("waveform", "spectral", "connectivity")


def test_connectivity_extraction_pair_count_bounds_continuity_and_lag_sign() -> None:
    fs = 50.0
    n_channels = 8
    assert len(connectivity_pair_indices(n_channels)) == 28

    rng = np.random.default_rng(31)
    n_hist = 60  # 1.20 s at 50 Hz
    n_cont = 75  # 1.50 s at 50 Hz
    history = rng.standard_normal((n_hist, 4))
    continuation = rng.standard_normal((n_cont, 4))
    joined = np.concatenate([history, continuation], axis=0)

    ext_split = extract_connectivity_frames(continuation, fs, history=history, window_s=1.02)
    ext_full = extract_connectivity_frames(joined, fs, window_s=1.02)

    t_branch = n_hist / fs
    post_mask = ext_full.times > t_branch + 1e-9
    np.testing.assert_allclose(ext_split.times, ext_full.times[post_mask] - t_branch, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(ext_split.values, ext_full.values[post_mask], rtol=1e-12, atol=1e-12)
    assert ext_split.values.shape == (25, 6)
    assert np.all(ext_split.valid_mask)
    assert np.all((ext_split.values >= -1.0) & (ext_split.values <= 1.0))

    # Zero-lag vs known quarter-cycle lag vs reversed lag at 10 Hz
    t = np.arange(150, dtype=np.float64) / fs
    freq = 10.0
    sig_base = np.sin(2.0 * np.pi * freq * t)
    sig_zero_lag = 2.5 * sig_base
    sig_lag_pos = np.sin(2.0 * np.pi * freq * t - 0.5 * np.pi)
    sig_lag_neg = np.sin(2.0 * np.pi * freq * t + 0.5 * np.pi)

    zero_lag_ext = extract_connectivity_frames(
        np.column_stack([sig_base, sig_zero_lag]), fs, window_s=1.02, band_hz=(9.0, 11.0)
    )
    pos_lag_ext = extract_connectivity_frames(
        np.column_stack([sig_base, sig_lag_pos]), fs, window_s=1.02, band_hz=(9.0, 11.0)
    )
    neg_lag_ext = extract_connectivity_frames(
        np.column_stack([sig_base, sig_lag_neg]), fs, window_s=1.02, band_hz=(9.0, 11.0)
    )
    swapped_ext = extract_connectivity_frames(
        np.column_stack([sig_lag_pos, sig_base]), fs, window_s=1.02, band_hz=(9.0, 11.0)
    )

    assert np.max(np.abs(zero_lag_ext.values)) < 1e-12
    assert np.all(pos_lag_ext.values > 0.3)
    np.testing.assert_allclose(neg_lag_ext.values, -pos_lag_ext.values, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(swapped_ext.values, -pos_lag_ext.values, rtol=1e-12, atol=1e-12)


def test_connectivity_validity_masks_insufficient_segments_and_low_power() -> None:
    fs = 50.0
    rng = np.random.default_rng(41)
    signals = rng.standard_normal((100, 3))

    # 1. Single-segment window (window_s=0.50 -> M=1 < M_min=4) is completely masked
    single_seg = extract_connectivity_frames(signals, fs, window_s=0.50, min_segments=4)
    assert np.all(single_seg.n_segments == 1)
    assert not np.any(single_seg.valid_mask)
    assert np.all(np.isnan(single_seg.values))

    # 2. Zero/below-floor power on channel 1 masks pairs (0, 1) and (1, 2) while keeping pair (0, 2) valid
    low_power = signals.copy()
    low_power[:, 1] = 0.0
    masked_ext = extract_connectivity_frames(low_power, fs, window_s=1.02)
    assert masked_ext.pairs == ((0, 1), (0, 2), (1, 2))
    assert not np.any(masked_ext.valid_mask[:, 0])
    assert np.all(masked_ext.valid_mask[:, 1])
    assert not np.any(masked_ext.valid_mask[:, 2])
    assert np.all(np.isnan(masked_ext.values[:, 0]))
    assert np.all(np.isfinite(masked_ext.values[:, 1]))
    assert np.all(np.isnan(masked_ext.values[:, 2]))


def test_select_connectivity_support_window_enforces_dev_split_and_reports_horizon_ratios() -> None:
    fs = 50.0
    rng = np.random.default_rng(53)
    # 2 states, 3 replicates, 75 post-branch samples (1.50 s), 4 channels, with 150-sample (3.0 s) pre-branch history
    base_state = rng.standard_normal((2, 1, 75, 4))
    dev_waveforms = np.tile(base_state, (1, 3, 1, 1)) + 0.05 * rng.standard_normal((2, 3, 75, 4))
    dev_histories = np.tile(rng.standard_normal((2, 1, 150, 4)), (1, 3, 1, 1))

    selection = select_connectivity_support_window(
        dev_waveforms,
        fs,
        dev_histories=dev_histories,
        repeatability_target=0.50,
    )
    assert selection.selected_window_s in CONNECTIVITY_CANDIDATE_WINDOWS_S
    assert selection.qualified is True
    assert set(selection.candidate_spreads) == set(CONNECTIVITY_CANDIDATE_WINDOWS_S)
    for h in CONTROL_HORIZONS_S:
        assert selection.support_to_horizon_ratio[h] == pytest.approx(selection.selected_window_s / h)

    bad_ident = [RolloutIdentity("seizure", 1, "ictal", 0, "zero", split="eval")]
    with pytest.raises(ValueError, match="split='dev'"):
        select_connectivity_support_window(dev_waveforms.reshape(6, 75, 4), fs, identities=bad_ident)
