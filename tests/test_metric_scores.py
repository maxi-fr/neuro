from __future__ import annotations

import numpy as np
import pytest

from neuro.metrics import (
    METRICS,
    Ensemble,
    baseline_grid,
    baseline_r2,
    controllability,
    predictability_r2,
    scalp_region_correlation,
    score_store,
    separability,
    sigma_ens,
    spread_reference,
    variance_ratio,
)

_RNG = np.random.default_rng(0)


def _ensemble(by_parent: np.ndarray) -> Ensemble:
    """Build an ensemble from a ``(n_parents, n_children, n_times)`` block."""
    n_parents, n_children, n_times = by_parent.shape
    return Ensemble(
        times=np.arange(n_times, dtype=np.float64),
        values=by_parent.reshape(n_parents * n_children, n_times),
        n_children=n_children,
    )


# --- ensemble bookkeeping ------------------------------------------------------------------


def test_by_parent_restores_the_parent_grouping() -> None:
    block = _RNG.normal(size=(4, 5, 3))

    ens = _ensemble(block)

    assert ens.n_parents == 4
    assert np.array_equal(ens.by_parent, block)


def test_a_ragged_ensemble_is_rejected() -> None:
    with pytest.raises(ValueError, match="not a whole number of parents"):
        Ensemble(times=np.zeros(3), values=np.zeros((7, 3)), n_children=4)


# --- predictability ------------------------------------------------------------------------


def test_r2_is_one_when_the_future_is_fixed_by_the_parent() -> None:
    block = np.tile(_RNG.normal(size=(4, 1, 3)), (1, 5, 1))

    assert predictability_r2(_ensemble(block)) == pytest.approx(np.ones(3))


def test_r2_is_zero_when_every_parent_has_the_same_children() -> None:
    """Parents that carry no information leave the conditional variance equal to the total."""
    block = np.tile(_RNG.normal(size=(1, 5, 3)), (4, 1, 1))

    assert predictability_r2(_ensemble(block)) == pytest.approx(np.zeros(3), abs=1e-12)


def test_r2_is_the_between_parent_share_of_the_total_variance() -> None:
    ens = _ensemble(_RNG.normal(size=(4, 5, 3)))

    between = ens.by_parent.mean(axis=1).var(axis=0)
    total = ens.values.var(axis=0)

    assert predictability_r2(ens) == pytest.approx(between / total)


def test_r2_falls_as_the_within_parent_spread_grows() -> None:
    means = _RNG.normal(size=(4, 1, 3))
    tight = _ensemble(means + 0.1 * _RNG.normal(size=(4, 200, 3)))
    loose = _ensemble(means + 2.0 * _RNG.normal(size=(4, 200, 3)))

    assert np.all(predictability_r2(tight) > predictability_r2(loose))


def test_sigma_ens_measures_the_within_parent_spread_not_the_parent_offsets() -> None:
    offsets = np.array([0.0, 100.0, -100.0, 50.0]).reshape(4, 1, 1)
    block = offsets + _RNG.normal(scale=3.0, size=(4, 4000, 2))

    assert sigma_ens(_ensemble(block)) == pytest.approx(np.full(2, 3.0), rel=0.05)


def test_spread_reference_returns_the_p5_p95_band() -> None:
    block = np.arange(100, dtype=np.float64).reshape(4, 25, 1)

    low, high = spread_reference(_ensemble(block))

    assert low == pytest.approx(np.percentile(np.arange(100), 5))
    assert high == pytest.approx(np.percentile(np.arange(100), 95))


# --- separability --------------------------------------------------------------------------


def test_cohens_d_matches_the_pooled_sd_definition_on_parent_means() -> None:
    healthy = _ensemble(_RNG.normal(loc=1.0, size=(8, 4, 2)))
    saturated = _ensemble(_RNG.normal(loc=5.0, size=(8, 4, 2)))

    h, s = healthy.by_parent.mean(axis=1), saturated.by_parent.mean(axis=1)
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
    stim = _ensemble(zero.by_parent - 1.0)

    score = controllability(zero, stim, direction=np.array([-1.0, -1.0]), gap=np.array([4.0, 4.0]))

    assert np.all(score.d_ctrl > 0)
    assert score.delta_bar == pytest.approx(np.full(2, -1.0))


def test_d_ctrl_is_negative_when_stimulation_drives_the_metric_the_wrong_way() -> None:
    zero = _ensemble(_RNG.normal(loc=5.0, size=(4, 50, 2)))
    stim = _ensemble(zero.by_parent + 1.0)

    score = controllability(zero, stim, direction=np.array([-1.0, -1.0]), gap=np.array([4.0, 4.0]))

    assert np.all(score.d_ctrl < 0)


def test_d_ctrl_is_the_effect_over_the_unpaired_ensemble_spread() -> None:
    zero = _ensemble(_RNG.normal(loc=5.0, scale=2.0, size=(4, 500, 2)))
    stim = _ensemble(zero.by_parent - 1.0)

    score = controllability(zero, stim, direction=np.array([-1.0, -1.0]), gap=np.array([4.0, 4.0]))

    assert score.d_ctrl == pytest.approx(1.0 / sigma_ens(zero))


def test_the_paired_sd_vanishes_for_a_constant_offset_while_sigma_ens_does_not() -> None:
    zero = _ensemble(_RNG.normal(loc=5.0, scale=2.0, size=(4, 500, 2)))
    stim = _ensemble(zero.by_parent - 1.0)

    score = controllability(zero, stim, direction=np.array([-1.0, -1.0]), gap=np.array([4.0, 4.0]))

    assert score.paired_sd == pytest.approx(np.zeros(2), abs=1e-12)
    assert np.all(sigma_ens(zero) > 1.0)


def test_relative_is_one_when_stimulation_closes_the_whole_gap() -> None:
    zero = _ensemble(np.full((4, 10, 2), 5.0))
    stim = _ensemble(np.full((4, 10, 2), 1.0))

    score = controllability(zero, stim, direction=np.array([-1.0, -1.0]), gap=np.array([4.0, 4.0]))

    assert score.relative == pytest.approx(np.ones(2))


# --- observability -------------------------------------------------------------------------


def test_scalp_region_correlation_is_one_for_an_affine_pair() -> None:
    scalp = _ensemble(_RNG.normal(size=(2, 3, 40)))
    region = _ensemble(3.0 * scalp.by_parent + 7.0)

    assert scalp_region_correlation(scalp, region) == pytest.approx(np.ones(6))


def test_scalp_region_correlation_is_near_zero_for_unrelated_series() -> None:
    scalp = _ensemble(_RNG.normal(size=(2, 3, 4000)))
    region = _ensemble(_RNG.normal(size=(2, 3, 4000)))

    assert np.all(np.abs(scalp_region_correlation(scalp, region)) < 0.2)


# --- the variance ratio on its own ----------------------------------------------------------


def test_variance_ratio_carries_trailing_axes_through_untouched() -> None:
    """One formula serves a scalar metric series and the per-channel raw-signal baselines."""
    block = _RNG.normal(size=(3, 4, 8, 20))
    values = block.reshape(12, 8, 20)

    per_channel = variance_ratio(values, n_children=4)

    assert per_channel.shape == (8, 20)
    for channel in range(8):
        expected = predictability_r2(_ensemble(block[:, :, channel, :]))
        assert per_channel[channel] == pytest.approx(expected)


# --- scoring a store -----------------------------------------------------------------------

_ALL62 = {"all62": np.arange(62)}
_SETS = {"all62": np.arange(62), "focal": np.array([27, 55, 12, 14])}


def test_score_store_returns_one_ensemble_per_metric_and_channel_set() -> None:
    store = _RNG.normal(size=(6, 62, 3000))

    times, scored = score_store(store, 1000.0, metrics=METRICS, channel_sets=_SETS, n_children=3)

    assert set(scored) == {(metric, channels) for metric in METRICS for channels in _SETS}
    assert times["block_ptp"][0] == pytest.approx(METRICS["block_ptp"].window_s)
    for (metric, _), ens in scored.items():
        assert ens.n_parents == 2
        assert ens.values.shape == (6, len(times[metric]))


def test_score_store_honours_a_channel_subset() -> None:
    store = _RNG.normal(size=(4, 62, 2000))

    times, scored = score_store(store, 1000.0, metrics=METRICS, channel_sets=_SETS, n_children=2)

    full, focal = scored["eeg_ms", "all62"], scored["eeg_ms", "focal"]
    assert np.array_equal(full.times, focal.times)
    assert not np.allclose(full.values, focal.values)
    assert focal.values.shape == full.values.shape
    assert len(times["eeg_ms"]) == full.values.shape[1]


def test_score_store_scores_one_pass_exactly_as_the_metric_scores_one_rollout() -> None:
    """The row-major sweep must not change any number, only how often the store is read."""
    store = _RNG.normal(size=(4, 62, 2000))

    _, scored = score_store(store, 1000.0, metrics=METRICS, channel_sets=_ALL62, n_children=2)

    for name, metric in METRICS.items():
        _, expected = metric(store[2], 1000.0)
        assert scored[name, "all62"].values[2] == pytest.approx(expected)


def test_a_cutoff_strips_the_high_frequency_content_line_length_lives_on() -> None:
    t = np.arange(2000) / 1000.0
    store = np.tile(np.sin(2 * np.pi * 6.0 * t) + np.sin(2 * np.pi * 200.0 * t), (4, 62, 1))

    _, raw = score_store(store, 1000.0, metrics=METRICS, channel_sets=_ALL62, n_children=2)
    _, filtered = score_store(store, 1000.0, metrics=METRICS, channel_sets=_ALL62, n_children=2, cutoff_hz=45.0)

    assert np.median(filtered["line_length", "all62"].values) < 0.1 * np.median(raw["line_length", "all62"].values)


# --- the raw-signal baselines ---------------------------------------------------------------


def _baseline_store(by_parent: np.ndarray) -> np.ndarray:
    """Flatten a ``(n_parents, n_children, n_channels, n_samples)`` block into store row order."""
    return by_parent.reshape(-1, *by_parent.shape[2:])


def test_baseline_r2_scores_both_rungs_on_the_raw_signal() -> None:
    store = _RNG.normal(size=(4, 6, 1000))
    times = baseline_grid(1.0)

    scores = baseline_r2(store, 1000.0, times=times, n_children=2)

    assert set(scores) == {"waveform", "envelope"}
    assert all(value.shape == (6, len(times)) for value in scores.values())


def test_baseline_r2_drops_the_envelope_under_a_cutoff() -> None:
    """Its band is fixed, so the bandwidth axis would redefine it rather than filter it."""
    store = _RNG.normal(size=(4, 6, 1000))

    scores = baseline_r2(store, 1000.0, times=baseline_grid(1.0), n_children=2, cutoff_hz=45.0)

    assert set(scores) == {"waveform"}


def test_baseline_r2_is_one_when_every_child_of_a_parent_is_identical() -> None:
    block = np.tile(_RNG.normal(size=(2, 1, 6, 1000)), (1, 3, 1, 1))

    scores = baseline_r2(_baseline_store(block), 1000.0, times=baseline_grid(1.0), n_children=3)

    assert scores["waveform"] == pytest.approx(np.ones_like(scores["waveform"]))


def test_baseline_r2_is_zero_when_the_parents_carry_no_information() -> None:
    block = np.tile(_RNG.normal(size=(1, 3, 6, 1000)), (2, 1, 1, 1))

    scores = baseline_r2(_baseline_store(block), 1000.0, times=baseline_grid(1.0), n_children=3)

    assert scores["waveform"] == pytest.approx(np.zeros_like(scores["waveform"]), abs=1e-12)
