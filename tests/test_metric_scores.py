from __future__ import annotations

import numpy as np
import pytest

from neuro.metrics import (
    METRICS,
    RAW_SERIES,
    Ensemble,
    controllability,
    coupling,
    envelope,
    latency_s,
    raw_series_grid,
    sample_at,
    score_raw_store,
    score_store,
    separability,
    sigma_ens,
    state_predictability_r2,
    state_readout_r2,
    variance_ratio,
)

_RNG = np.random.default_rng(0)


def _ensemble(by_state: np.ndarray) -> Ensemble:
    """Build an ensemble from a ``(n_states, n_replicates, n_times)`` block."""
    n_states, n_replicates, n_times = by_state.shape
    return Ensemble(
        times=np.arange(n_times, dtype=np.float64),
        values=by_state.reshape(n_states * n_replicates, n_times),
        n_replicates=n_replicates,
    )


# --- ensemble bookkeeping ------------------------------------------------------------------


def test_by_state_restores_the_state_grouping() -> None:
    block = _RNG.normal(size=(4, 5, 3))

    ens = _ensemble(block)

    assert ens.n_states == 4
    assert np.array_equal(ens.by_state, block)


def test_a_ragged_ensemble_is_rejected() -> None:
    with pytest.raises(ValueError, match="not a whole number of states"):
        Ensemble(times=np.zeros(3), values=np.zeros((7, 3)), n_replicates=4)


# --- predictability ------------------------------------------------------------------------


def test_r2_is_one_when_the_future_is_fixed_by_the_state() -> None:
    block = np.tile(_RNG.normal(size=(4, 1, 3)), (1, 5, 1))

    assert variance_ratio(_ensemble(block).values, 5) == pytest.approx(np.ones(3))


def test_r2_is_zero_when_every_state_has_the_same_replicates() -> None:
    """States that carry no information leave the conditional variance equal to the total."""
    block = np.tile(_RNG.normal(size=(1, 5, 3)), (4, 1, 1))

    assert variance_ratio(_ensemble(block).values, 5) == pytest.approx(np.zeros(3), abs=1e-12)


def test_r2_is_the_between_state_share_of_the_total_variance() -> None:
    ens = _ensemble(_RNG.normal(size=(4, 5, 3)))

    between = ens.by_state.mean(axis=1).var(axis=0)
    total = ens.values.var(axis=0)

    assert variance_ratio(ens.values, ens.n_replicates) == pytest.approx(between / total)


def test_r2_falls_as_the_within_state_spread_grows() -> None:
    means = _RNG.normal(size=(4, 1, 3))
    tight = _ensemble(means + 0.1 * _RNG.normal(size=(4, 200, 3)))
    loose = _ensemble(means + 2.0 * _RNG.normal(size=(4, 200, 3)))

    assert np.all(variance_ratio(tight.values, tight.n_replicates) > variance_ratio(loose.values, loose.n_replicates))


def test_sigma_ens_measures_the_within_state_spread_not_the_state_offsets() -> None:
    offsets = np.array([0.0, 100.0, -100.0, 50.0]).reshape(4, 1, 1)
    block = offsets + _RNG.normal(scale=3.0, size=(4, 4000, 2))

    assert sigma_ens(_ensemble(block)) == pytest.approx(np.full(2, 3.0), rel=0.05)


# --- separability --------------------------------------------------------------------------


def test_cohens_d_matches_the_pooled_sd_definition_on_state_means() -> None:
    healthy = _ensemble(_RNG.normal(loc=1.0, size=(8, 4, 2)))
    saturated = _ensemble(_RNG.normal(loc=5.0, size=(8, 4, 2)))

    h, s = healthy.by_state.mean(axis=1), saturated.by_state.mean(axis=1)
    pooled = np.sqrt((h.var(axis=0, ddof=1) + s.var(axis=0, ddof=1)) / 2.0)

    assert separability(healthy, saturated).cohens_d == pytest.approx((s.mean(axis=0) - h.mean(axis=0)) / pooled)


def test_direction_points_down_for_a_metric_the_seizure_raises() -> None:
    healthy = _ensemble(np.full((8, 4, 2), 1.0))
    saturated = _ensemble(np.full((8, 4, 2), 5.0) + 1e-6 * _RNG.normal(size=(8, 4, 2)))

    assert np.array_equal(separability(healthy, saturated).direction, [-1.0, -1.0])


def test_direction_points_up_for_a_metric_the_seizure_lowers() -> None:
    healthy = _ensemble(np.full((8, 4, 2), 5.0))
    saturated = _ensemble(np.full((8, 4, 2), 1.0) + 1e-6 * _RNG.normal(size=(8, 4, 2)))

    assert np.array_equal(separability(healthy, saturated).direction, [1.0, 1.0])


# --- controllability -----------------------------------------------------------------------


def test_d_ctrl_is_positive_when_stimulation_moves_the_metric_toward_healthy() -> None:
    zero = _ensemble(_RNG.normal(loc=5.0, size=(4, 50, 2)))
    stim = _ensemble(zero.by_state - 1.0)

    score = controllability(zero, stim, direction=np.array([-1.0, -1.0]), gap=np.array([4.0, 4.0]))

    assert np.all(score.d_ctrl > 0)
    assert score.delta_bar == pytest.approx(np.full(2, -1.0))


def test_d_ctrl_is_negative_when_stimulation_drives_the_metric_the_wrong_way() -> None:
    zero = _ensemble(_RNG.normal(loc=5.0, size=(4, 50, 2)))
    stim = _ensemble(zero.by_state + 1.0)

    score = controllability(zero, stim, direction=np.array([-1.0, -1.0]), gap=np.array([4.0, 4.0]))

    assert np.all(score.d_ctrl < 0)


def test_d_ctrl_is_the_effect_over_the_unpaired_ensemble_spread() -> None:
    zero = _ensemble(_RNG.normal(loc=5.0, scale=2.0, size=(4, 500, 2)))
    stim = _ensemble(zero.by_state - 1.0)

    score = controllability(zero, stim, direction=np.array([-1.0, -1.0]), gap=np.array([4.0, 4.0]))

    assert score.d_ctrl == pytest.approx(1.0 / sigma_ens(zero))


def test_the_paired_sd_vanishes_for_a_constant_offset_while_sigma_ens_does_not() -> None:
    zero = _ensemble(_RNG.normal(loc=5.0, scale=2.0, size=(4, 500, 2)))
    stim = _ensemble(zero.by_state - 1.0)

    score = controllability(zero, stim, direction=np.array([-1.0, -1.0]), gap=np.array([4.0, 4.0]))

    assert score.paired_sd == pytest.approx(np.zeros(2), abs=1e-12)
    assert np.all(sigma_ens(zero) > 1.0)


def test_relative_is_one_when_stimulation_closes_the_whole_gap() -> None:
    zero = _ensemble(np.full((4, 10, 2), 5.0))
    stim = _ensemble(np.full((4, 10, 2), 1.0))

    score = controllability(zero, stim, direction=np.array([-1.0, -1.0]), gap=np.array([4.0, 4.0]))

    assert score.relative == pytest.approx(np.ones(2))


# --- state readout -------------------------------------------------------------------------


def test_readout_r2_is_one_when_the_metric_is_a_function_of_the_state() -> None:
    state = np.tile(np.arange(10) / 10.0, 50)

    score = state_readout_r2(2.0 * state + 1.0, state, n_bins=10)

    assert score.r2 == pytest.approx(1.0)


def test_readout_r2_is_zero_when_the_metric_ignores_the_state() -> None:
    state = np.tile(np.arange(10) / 10.0, 2000)
    metric = _RNG.normal(size=state.shape)

    score = state_readout_r2(metric, state, n_bins=10)

    assert score.r2 == pytest.approx(0.0, abs=0.02)


def test_readout_r2_matches_a_hand_computed_two_bin_split() -> None:
    """Low state -> {0, 2}, high state -> {10, 14}: within var (1 + 4)/2 = 2.5, total var 32.75."""
    state = np.array([0.0, 0.0, 1.0, 1.0])
    metric = np.array([0.0, 2.0, 10.0, 14.0])

    score = state_readout_r2(metric, state, n_bins=2)

    assert score.r2 == pytest.approx(1.0 - 2.5 / 32.75)
    assert score.explained_var == pytest.approx(30.25)


def test_readout_carries_a_channel_axis_through_independently() -> None:
    state = np.tile(np.arange(10) / 10.0, 40)
    metric = np.stack([2.0 * state + 1.0, _RNG.normal(size=state.shape)], axis=1)

    score = state_readout_r2(metric, state, n_bins=10)

    assert score.r2.shape == (2,)
    assert score.r2[0] == pytest.approx(1.0)
    assert score.r2[1] == pytest.approx(0.0, abs=0.15)


def test_readout_survives_the_point_mass_at_zero_state() -> None:
    """Every pre-onset window has s = 0, so quantile edges coincide and bins would come out empty."""
    state = np.concatenate([np.zeros(300), np.repeat(np.linspace(0.2, 1.0, 5), 40)])
    metric = np.stack([2.0 * state + 1.0, _RNG.normal(size=state.shape)], axis=1)

    score = state_readout_r2(metric, state, n_bins=10)

    assert score.bin_metric.shape == (len(score.bin_state), 2)
    assert np.isfinite(score.r2).all()
    assert score.r2[0] > 0.9
    assert score.r2[1] == pytest.approx(0.0, abs=0.15)


def test_readout_returns_the_calibration_curve_the_notebook_plots() -> None:
    state = np.tile(np.arange(10) / 10.0, 50)

    score = state_readout_r2(3.0 * state, state, n_bins=10)

    assert score.bin_state == pytest.approx(np.arange(10) / 10.0)
    assert score.bin_metric == pytest.approx(3.0 * np.arange(10) / 10.0)


# --- predictability against a stated denominator ---------------------------------------------


def test_state_predictability_is_one_when_the_state_fixes_the_future() -> None:
    block = np.tile(_RNG.normal(size=(4, 1, 3)), (1, 50, 1))

    assert state_predictability_r2(_ensemble(block), 1.0) == pytest.approx(np.ones(3))


def test_state_predictability_is_zero_when_the_forecast_error_fills_the_state_range() -> None:
    block = _RNG.normal(scale=2.0, size=(4, 4000, 2))

    score = state_predictability_r2(_ensemble(block), 4.0)

    assert score == pytest.approx(np.zeros(2), abs=0.05)


def test_state_predictability_goes_negative_when_the_noise_exceeds_the_state_range() -> None:
    """Unbounded below on purpose: it says the metric cannot resolve the states it must steer between."""
    block = _RNG.normal(scale=4.0, size=(4, 4000, 2))

    assert np.all(state_predictability_r2(_ensemble(block), 1.0) < -10.0)


def test_state_predictability_uses_the_same_denominator_at_every_branch() -> None:
    """Unlike variance_ratio, whose denominator is whatever spread the trajectories happened to have."""
    tight = _ensemble(0.01 * _RNG.normal(size=(4, 1, 2)) + _RNG.normal(scale=0.5, size=(4, 500, 2)))
    wide = _ensemble(10.0 * _RNG.normal(size=(4, 1, 2)) + _RNG.normal(scale=0.5, size=(4, 500, 2)))

    assert state_predictability_r2(tight, 4.0) == pytest.approx(state_predictability_r2(wide, 4.0), abs=0.1)
    assert np.all(
        variance_ratio(tight.values, tight.n_replicates) < variance_ratio(wide.values, wide.n_replicates) - 0.5
    )


def test_state_predictability_takes_the_scalar_explained_var_of_a_pooled_metric() -> None:
    """A pooled metric's V_state is one number, not one per lookahead -- the notebook's own path."""
    state = np.tile(np.arange(10) / 10.0, 50)
    v_state = state_readout_r2(2.0 * state + 1.0, state, n_bins=10).explained_var
    ens = _ensemble(_RNG.normal(scale=0.1, size=(4, 200, 3)))

    score = state_predictability_r2(ens, v_state)

    assert np.ndim(v_state) == 0
    assert score == pytest.approx(1.0 - sigma_ens(ens) ** 2 / float(v_state))


def test_state_predictability_scores_a_per_channel_metric_against_its_own_state_range() -> None:
    """One explained_var per channel, one curve per channel -- the trailing axes both sides carry."""
    ens = Ensemble(times=np.arange(3, dtype=np.float64), values=_RNG.normal(size=(20, 2, 3)), n_replicates=5)

    score = state_predictability_r2(ens, np.array([4.0, 1.0]))

    assert score.shape == (2, 3)
    assert score[0] == pytest.approx(state_predictability_r2(_ensemble(ens.by_state[:, :, 0, :]), 4.0))
    assert score[1] == pytest.approx(state_predictability_r2(_ensemble(ens.by_state[:, :, 1, :]), 1.0))


# --- coupling: does moving the metric move the seizure? --------------------------------------


def test_coupling_is_one_when_the_state_shift_tracks_the_metric_shift() -> None:
    zero_m, zero_s = _ensemble(_RNG.normal(size=(4, 40, 3))), _ensemble(_RNG.normal(size=(4, 40, 3)))
    delta = _RNG.normal(size=(160, 3))
    stim_m = Ensemble(zero_m.times, zero_m.values + delta, 40)
    stim_s = Ensemble(zero_s.times, zero_s.values + 2.0 * delta, 40)

    assert coupling(zero_m, stim_m, zero_s, stim_s) == pytest.approx(np.ones(3))


def test_coupling_is_minus_one_when_the_metric_moves_opposite_to_the_seizure() -> None:
    """The tes_field_geometry quadrant: the objective is steered, the seizure goes the other way."""
    zero_m, zero_s = _ensemble(_RNG.normal(size=(4, 40, 3))), _ensemble(_RNG.normal(size=(4, 40, 3)))
    delta = _RNG.normal(size=(160, 3))
    stim_m = Ensemble(zero_m.times, zero_m.values + delta, 40)
    stim_s = Ensemble(zero_s.times, zero_s.values - delta, 40)

    assert coupling(zero_m, stim_m, zero_s, stim_s) == pytest.approx(-np.ones(3))


def test_coupling_is_near_zero_when_the_metric_moves_and_the_seizure_does_not_follow() -> None:
    """A metric the actuator drives confidently, with no bearing on the seizure -- the §2.3 failure."""
    zero_m, zero_s = _ensemble(_RNG.normal(size=(4, 500, 2))), _ensemble(_RNG.normal(size=(4, 500, 2)))
    stim_m = Ensemble(zero_m.times, zero_m.values + _RNG.normal(size=(2000, 2)), 500)
    stim_s = Ensemble(zero_s.times, zero_s.values + _RNG.normal(size=(2000, 2)), 500)

    assert np.all(np.abs(coupling(zero_m, stim_m, zero_s, stim_s)) < 0.1)


def test_coupling_scores_each_channel_against_the_one_network_state() -> None:
    """The state is one scalar per rollout, so a per-channel metric gets one correlation per channel."""
    zero_s = _ensemble(_RNG.normal(size=(4, 40, 3)))
    delta_s = _RNG.normal(size=(160, 3))
    stim_s = Ensemble(zero_s.times, zero_s.values + delta_s, 40)

    zero_m = Ensemble(zero_s.times, _RNG.normal(size=(160, 2, 3)), 40)
    # channel 0 is shifted exactly with the state, channel 1 independently of it
    delta_m = np.stack([delta_s, _RNG.normal(size=(160, 3))], axis=1)
    stim_m = Ensemble(zero_m.times, zero_m.values + delta_m, 40)

    rho = coupling(zero_m, stim_m, zero_s, stim_s)

    assert rho.shape == (2, 3)
    assert rho[0] == pytest.approx(np.ones(3))
    assert np.all(np.abs(rho[1]) < 0.4)


def test_coupling_ignores_the_mean_effect_and_scores_only_the_covariation() -> None:
    """The mean shift is what d_ctrl and d_state already report; this axis asks the other question."""
    zero_m, zero_s = _ensemble(_RNG.normal(size=(4, 200, 2))), _ensemble(_RNG.normal(size=(4, 200, 2)))
    delta = _RNG.normal(size=(800, 2))
    stim_m = Ensemble(zero_m.times, zero_m.values + delta, 200)
    stim_s = Ensemble(zero_s.times, zero_s.values + 0.5 * delta + _RNG.normal(size=(800, 2)), 200)

    shifted_m = Ensemble(zero_m.times, stim_m.values - 5.0, 200)
    shifted_s = Ensemble(zero_s.times, stim_s.values - 3.0, 200)

    assert coupling(zero_m, stim_m, zero_s, stim_s) == pytest.approx(coupling(zero_m, shifted_m, zero_s, shifted_s))


# --- the variance ratio on its own ----------------------------------------------------------


def test_variance_ratio_carries_trailing_axes_through_untouched() -> None:
    """One formula serves a scalar metric series and the per-channel raw-signal baselines."""
    block = _RNG.normal(size=(3, 4, 8, 20))
    values = block.reshape(12, 8, 20)

    per_channel = variance_ratio(values, n_replicates=4)

    assert per_channel.shape == (8, 20)
    for channel in range(8):
        expected = variance_ratio(_ensemble(block[:, :, channel, :]).values, 4)
        assert per_channel[channel] == pytest.approx(expected)


# --- scoring a store -----------------------------------------------------------------------

_ALL62 = {"all62": np.arange(62)}
_SETS = {"all62": np.arange(62), "focal": np.array([27, 55, 12, 14])}


def test_score_store_returns_one_ensemble_per_metric_and_channel_set() -> None:
    store = _RNG.normal(size=(6, 62, 3000))

    times, scored = score_store(store, 1000.0, metrics=METRICS, channel_sets=_SETS, n_replicates=3)

    assert set(scored) == {(metric, channels) for metric in METRICS for channels in _SETS}
    assert times["block_ptp"][0] == pytest.approx(METRICS["block_ptp"].window_s)
    for (metric, _), ens in scored.items():
        assert ens.n_states == 2
        assert ens.values.shape == (6, len(times[metric]))


def test_score_store_honours_a_channel_subset() -> None:
    store = _RNG.normal(size=(4, 62, 2000))

    times, scored = score_store(store, 1000.0, metrics=METRICS, channel_sets=_SETS, n_replicates=2)

    full, focal = scored["eeg_ms", "all62"], scored["eeg_ms", "focal"]
    assert np.array_equal(full.times, focal.times)
    assert not np.allclose(full.values, focal.values)
    assert focal.values.shape == full.values.shape
    assert len(times["eeg_ms"]) == full.values.shape[1]


def test_score_store_scores_one_pass_exactly_as_the_metric_scores_one_rollout() -> None:
    """The row-major sweep must not change any number, only how often the store is read."""
    store = _RNG.normal(size=(4, 62, 2000))

    _, scored = score_store(store, 1000.0, metrics=METRICS, channel_sets=_ALL62, n_replicates=2)

    for name, metric in METRICS.items():
        _, per_channel = metric(store[2], 1000.0)
        assert scored[name, "all62"].values[2] == pytest.approx(per_channel.mean(axis=0))


def test_score_store_keeps_the_channel_axis_when_asked_not_to_pool() -> None:
    """The per-channel path must be the same numbers, un-averaged -- not a second definition."""
    store = _RNG.normal(size=(4, 62, 2000))

    _, pooled = score_store(store, 1000.0, metrics=METRICS, channel_sets=_ALL62, n_replicates=2)
    _, per_channel = score_store(store, 1000.0, metrics=METRICS, channel_sets=_ALL62, n_replicates=2, pool=False)

    for name in METRICS:
        values = per_channel[name, "all62"].values
        assert values.shape == (4, 62, len(pooled[name, "all62"].times))
        assert values.mean(axis=1) == pytest.approx(pooled[name, "all62"].values)


def test_a_cutoff_strips_the_high_frequency_content_line_length_lives_on() -> None:
    t = np.arange(2000) / 1000.0
    store = np.tile(np.sin(2 * np.pi * 6.0 * t) + np.sin(2 * np.pi * 200.0 * t), (4, 62, 1))

    _, raw = score_store(store, 1000.0, metrics=METRICS, channel_sets=_ALL62, n_replicates=2)
    _, filtered = score_store(store, 1000.0, metrics=METRICS, channel_sets=_ALL62, n_replicates=2, cutoff_hz=45.0)

    assert np.median(filtered["line_length", "all62"].values) < 0.1 * np.median(raw["line_length", "all62"].values)


# --- the window-free series ------------------------------------------------------------------

_RAW_SETS = {"all6": np.arange(6), "focal": np.array([1, 4])}


def _raw_store(by_state: np.ndarray) -> np.ndarray:
    """Flatten a ``(n_states, n_replicates, n_channels, n_samples)`` block into store row order."""
    return by_state.reshape(-1, *by_state.shape[2:])


def _scored_raw(store: np.ndarray, n_replicates: int) -> dict[tuple[str, str], Ensemble]:
    return score_raw_store(
        store,
        1000.0,
        times=raw_series_grid(1.0),
        series=RAW_SERIES,
        channel_sets=_RAW_SETS,
        n_replicates=n_replicates,
    )


def test_score_raw_store_returns_one_ensemble_per_series_and_channel_set() -> None:
    store = _RNG.normal(size=(4, 6, 1000))

    scored = _scored_raw(store, 2)

    assert set(scored) == {(name, channels) for name in RAW_SERIES for channels in _RAW_SETS}
    assert scored["waveform", "all6"].values.shape == (4, 6, len(raw_series_grid(1.0)))
    assert scored["envelope", "focal"].values.shape == (4, 2, len(raw_series_grid(1.0)))


def test_score_raw_store_keeps_the_channel_axis_so_a_signed_waveform_does_not_cancel() -> None:
    """Channels straddling a source see it with opposite sign, so their mean cancels what is there."""
    signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])[:, None]
    store = signs * _RNG.normal(size=(4, 1, 1000))

    values = _scored_raw(store, 2)["waveform", "all6"].values

    assert np.abs(values).mean() > 0.5
    assert values.mean(axis=1) == pytest.approx(np.zeros((4, len(raw_series_grid(1.0)))))


def test_score_raw_store_samples_the_transform_rather_than_reducing_a_window() -> None:
    store = _RNG.normal(size=(2, 6, 1000))
    times = raw_series_grid(1.0)

    scored = _scored_raw(store, 2)

    assert scored["waveform", "all6"].values[1] == pytest.approx(sample_at(store[1], 1000.0, times))
    assert scored["envelope", "all6"].values[1] == pytest.approx(sample_at(envelope(store[1], 1000.0), 1000.0, times))


def test_a_raw_series_is_predictable_when_every_replicate_of_a_state_is_identical() -> None:
    block = np.tile(_RNG.normal(size=(2, 1, 6, 1000)), (1, 3, 1, 1))

    ens = _scored_raw(_raw_store(block), 3)["waveform", "all6"]
    scores = variance_ratio(ens.values, ens.n_replicates)

    assert scores == pytest.approx(np.ones_like(scores))


def test_a_raw_series_is_unpredictable_when_the_states_carry_no_information() -> None:
    block = np.tile(_RNG.normal(size=(1, 3, 6, 1000)), (2, 1, 1, 1))

    ens = _scored_raw(_raw_store(block), 3)["waveform", "all6"]
    scores = variance_ratio(ens.values, ens.n_replicates)

    assert scores == pytest.approx(np.zeros_like(scores), abs=1e-12)


def test_latency_is_the_window_for_a_metric_and_the_group_delay_for_a_raw_series() -> None:
    """The window-free series still make the controller wait -- for their filters, not a window."""
    assert latency_s("band_power", 1000.0) == pytest.approx(METRICS["band_power"].window_s)
    assert latency_s("waveform", 1000.0) == 0.0
    assert latency_s("envelope", 1000.0) > 0.0
