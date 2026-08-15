from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from neuro.ensembles import (
    ARMS,
    RAW,
    SWEPT_METRICS,
    Arm,
    Branch,
    EnsembleConfig,
    PlantPair,
    ScoreArchive,
    build_plants,
    cutoff_key,
    eeg_path,
    generate,
    region_path,
    reseed,
    run_child,
    run_parent,
    run_segment,
    score_ensemble_dir,
)
from neuro.metrics import METRICS

if TYPE_CHECKING:
    from pathlib import Path

_ZERO = (0.0, 0.0, 0.0)
_D1 = (2.0, 0.0, -2.0)


@pytest.fixture(scope="module")
def plants() -> PlantPair:
    return build_plants()


# --- the branching gotcha ------------------------------------------------------------------


def test_two_segments_reproduce_one_long_run(plants: PlantPair) -> None:
    """Branching preserves x, the delay history and the clock -- the whole point of deepcopy."""
    split = reseed(plants.seizure, 7)
    stitched = np.concatenate([run_segment(split, 250, _ZERO), run_segment(split, 250, _ZERO)], axis=1)

    whole = run_segment(reseed(plants.seizure, 7), 500, _ZERO)

    assert np.array_equal(stitched, whole)


def test_reseed_carries_the_full_state_and_only_swaps_the_noise(plants: PlantPair) -> None:
    parent = reseed(plants.seizure, 7)
    run_segment(parent, 300, _ZERO)

    child = reseed(parent, 99)

    assert np.array_equal(child.x, parent.x)
    assert np.array_equal(child.history, parent.history)
    assert child.next_update_time == parent.next_update_time
    assert not np.array_equal(run_segment(child, 100, _ZERO), run_segment(parent, 100, _ZERO))


def test_a_snapshot_is_not_advanced_by_the_children_branched_off_it(plants: PlantPair) -> None:
    snapshot = reseed(plants.seizure, 7)
    run_segment(snapshot, 300, _ZERO)
    x_before, clock_before = snapshot.x.copy(), snapshot.next_update_time

    for seed in range(3):
        run_segment(reseed(snapshot, seed), 200, _ZERO)

    assert np.array_equal(snapshot.x, x_before)
    assert snapshot.next_update_time == clock_before


def test_the_same_seed_and_arm_reproduce_the_same_rollout(plants: PlantPair) -> None:
    snapshot = reseed(plants.seizure, 7)
    run_segment(snapshot, 200, _ZERO)

    first = run_segment(reseed(snapshot, 42), 200, _D1)
    second = run_segment(reseed(snapshot, 42), 200, _D1)

    assert np.array_equal(first, second)


def test_sharing_a_seed_across_arms_is_far_tighter_than_sharing_none(plants: PlantPair) -> None:
    """Common random numbers: the paired arm difference must be dwarfed by the noise spread."""
    snapshot = reseed(plants.seizure, 7)
    run_segment(snapshot, 2000, _ZERO)

    paired = run_segment(reseed(snapshot, 42), 1000, _D1) - run_segment(reseed(snapshot, 42), 1000, _ZERO)
    unpaired = run_segment(reseed(snapshot, 43), 1000, _ZERO) - run_segment(reseed(snapshot, 42), 1000, _ZERO)

    assert np.abs(paired).mean() < 0.1 * np.abs(unpaired).mean()


# --- plants --------------------------------------------------------------------------------


def test_the_two_plants_differ_only_in_the_excitatory_gain(plants: PlantPair) -> None:
    assert np.array_equal(plants.healthy.x, plants.seizure.x)
    assert np.array_equal(plants.healthy.history, plants.seizure.history)
    assert np.array_equal(plants.healthy.w_weights, plants.seizure.w_weights)
    assert plants.healthy.K == plants.seizure.K
    assert plants.healthy.net_params.sigma == plants.seizure.net_params.sigma
    assert np.array_equal(np.unique(plants.healthy.net_params.A), [3.25])
    assert np.array_equal(np.unique(plants.seizure.net_params.A), [3.25, 3.4, 3.6])


def test_the_arms_are_kcl_valid_on_the_three_electrode_montage(plants: PlantPair) -> None:
    assert plants.seizure.n_controls == 3
    for arm in ARMS:
        assert len(arm.current) == plants.seizure.n_controls
        assert sum(arm.current) == pytest.approx(0.0)


# --- parents and children ------------------------------------------------------------------


def test_run_parent_snapshots_at_every_branch_time(plants: PlantPair) -> None:
    cfg = EnsembleConfig(
        n_parents=1,
        n_children=1,
        rollout_s=0.1,
        branches=(Branch("early", "seizure", 0.6), Branch("late", "seizure", 1.1)),
        chunk_s=0.25,
    )

    parent = run_parent(plants, cfg, "seizure", 0)

    assert set(parent.snapshots) == {"early", "late"}
    assert parent.snapshots["early"].next_update_time == pytest.approx(0.6)
    assert parent.snapshots["late"].next_update_time == pytest.approx(1.1)
    assert parent.onsets.shape == (len(plants.connectome.region_labels),)
    assert 0 <= parent.n_seizing_final <= len(plants.connectome.region_labels)


def test_run_child_returns_scalp_at_full_rate_and_region_decimated(plants: PlantPair) -> None:
    cfg = EnsembleConfig(rollout_s=0.5, region_fs=1000.0)
    snapshot = reseed(plants.seizure, 7)
    run_segment(snapshot, 200, _ZERO)

    eeg, region = run_child(snapshot, plants, cfg, seed=3, arm=Arm("zero", _ZERO))

    assert eeg.shape == (62, 5000)
    assert region.shape == (len(plants.connectome.region_labels), 500)
    assert region.dtype == np.float32
    assert np.all(np.isfinite(eeg))


# --- generation end to end -----------------------------------------------------------------


def test_generate_writes_one_store_per_branch_and_arm_plus_a_manifest(plants: PlantPair, tmp_path: Path) -> None:
    cfg = EnsembleConfig(
        n_parents=1,
        n_children=2,
        rollout_s=0.3,
        branches=(Branch("pre_onset", "seizure", 1.0),),
        region_fs=1000.0,
        chunk_s=0.5,
    )

    manifest = generate(tmp_path, cfg, plants)

    for arm in cfg.arms:
        eeg = np.load(eeg_path(tmp_path, "pre_onset", arm.name), mmap_mode="r")
        region = np.load(region_path(tmp_path, "pre_onset", arm.name), mmap_mode="r")
        assert eeg.shape == (2, 62, 3000)
        assert region.shape == (2, 76, 300)
        assert np.all(np.isfinite(np.asarray(eeg)))
        assert np.any(np.asarray(eeg) != 0.0)

    assert manifest == json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["channel_sets"]["lTCI"] == [27, 55, 12, 14]
    assert [c["name"] for c in manifest["arms"]] == ["zero", "d1"]
    assert manifest["parents"][0]["seed"] == cfg.parent_seed("seizure", 0)
    assert len(manifest["parents"][0]["onsets"]) == 76


def test_generated_arms_share_their_noise_realisation(plants: PlantPair, tmp_path: Path) -> None:
    """The stored arms must be paired, or every controllability score loses its power.

    Two arms carrying the same (null) command isolate the seed from the stimulus: any
    difference at all would mean the arm loop advanced the noise stream between them.
    """
    cfg = EnsembleConfig(
        n_parents=1,
        n_children=2,
        rollout_s=0.3,
        branches=(Branch("pre_onset", "seizure", 1.0),),
        arms=(Arm("first", _ZERO), Arm("second", _ZERO)),
        chunk_s=0.5,
    )

    generate(tmp_path, cfg, plants)
    first = np.load(eeg_path(tmp_path, "pre_onset", "first"))
    second = np.load(eeg_path(tmp_path, "pre_onset", "second"))

    assert np.array_equal(first, second)
    assert not np.array_equal(first[0], first[1])


# --- scoring across the bandwidth axis -------------------------------------------------------

_CUTOFFS: tuple[float | None, ...] = (None, 100.0, 20.0)


@pytest.fixture(scope="module")
def scored(plants: PlantPair, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, ScoreArchive]:
    """One tiny generated ensemble, scored over three bandwidths.

    Two parents, not one: with a single parent the within-parent variance *is* the total, so
    every R2 in the archive would be identically zero and nothing about it would be testable.
    """
    out_dir = tmp_path_factory.mktemp("scored")
    cfg = EnsembleConfig(
        n_parents=2,
        n_children=2,
        rollout_s=0.6,
        branches=(Branch("pre_onset", "seizure", 1.0),),
        chunk_s=0.5,
    )
    generate(out_dir, cfg, plants)
    return out_dir, score_ensemble_dir(out_dir, cutoffs=_CUTOFFS)


def test_only_the_swept_metrics_are_rescored_under_a_filter(scored: tuple[Path, ScoreArchive]) -> None:
    """A low-pass redefines a band integral and a centroid, so sweeping those is not one metric."""
    _, archive = scored
    keyed = {(metric, cutoff) for _, _, metric, _, cutoff in archive.series if cutoff != RAW}

    assert {metric for metric, _ in keyed} == set(SWEPT_METRICS)
    assert {cutoff for _, cutoff in keyed} == {cutoff_key(c) for c in _CUTOFFS if c is not None}


def test_every_metric_is_scored_on_the_unfiltered_signal(scored: tuple[Path, ScoreArchive]) -> None:
    _, archive = scored

    for name in METRICS:
        assert archive.ensemble("pre_onset", "zero", name, "all62").n_states == 2


def test_the_region_store_is_read_for_the_seizure_state_only(scored: tuple[Path, ScoreArchive]) -> None:
    """No metric is scored in region space -- that was the superseded observability axis's reference."""
    _, archive = scored

    assert set(archive.states) == {(branch, arm) for branch, arm, _, _, _ in archive.series}
    assert archive.state("pre_onset", "zero").values.shape[1] == len(archive.state_times)


def test_a_narrower_band_strips_the_content_line_length_lives_on(scored: tuple[Path, ScoreArchive]) -> None:
    """Half of why R2 alone cannot rank the bandwidth axis: the filter removes signal, not just noise."""
    _, archive = scored
    raw = archive.ensemble("pre_onset", "zero", "line_length", "all62")
    narrow = archive.ensemble("pre_onset", "zero", "line_length", "all62", cutoff_key(20.0))

    assert np.median(narrow.values) < np.median(raw.values)


def test_baselines_cover_every_cutoff_but_the_envelope_only_the_raw_signal(
    scored: tuple[Path, ScoreArchive],
) -> None:
    _, archive = scored
    keys = {(name, cutoff) for _, name, cutoff in archive.baselines}

    assert {cutoff for name, cutoff in keys if name == "waveform"} == {cutoff_key(c) for c in _CUTOFFS}
    assert {cutoff for name, cutoff in keys if name == "envelope"} == {RAW}


def test_a_baseline_is_stored_per_channel_so_the_channel_sets_come_free(
    scored: tuple[Path, ScoreArchive],
) -> None:
    _, archive = scored

    per_channel = archive.baselines["pre_onset", "waveform", RAW]
    focal = archive.baseline("pre_onset", "waveform", "lTCI")

    assert per_channel.shape == (62, len(archive.baseline_times))
    assert focal == pytest.approx(per_channel[[27, 55, 12, 14]].mean(axis=0))
    assert not np.allclose(focal, archive.baseline("pre_onset", "waveform", "all62"))


def test_the_cached_archive_reloads_to_float32_precision(scored: tuple[Path, ScoreArchive]) -> None:
    """Series round-trip through float32: they are scores off a float32 store, not raw data."""
    out_dir, archive = scored

    reloaded = score_ensemble_dir(out_dir, cutoffs=_CUTOFFS)

    assert set(reloaded.series) == set(archive.series)
    assert set(reloaded.baselines) == set(archive.baselines)
    assert np.array_equal(reloaded.baseline_times, archive.baseline_times)
    for key, values in archive.series.items():
        assert reloaded.series[key] == pytest.approx(values, rel=1e-6)
    for key, values in archive.baselines.items():
        assert np.array_equal(reloaded.baselines[key], values)


def test_the_raw_scalp_series_keep_the_channel_axis_and_pool_on_read(
    scored: tuple[Path, ScoreArchive],
) -> None:
    """One traversal serves both views: per-channel on disk, channel-mean through `pool=True`."""
    _, archive = scored

    per_channel = archive.ensemble("pre_onset", "zero", "eeg_ms", "all62", pool=False)
    pooled = archive.ensemble("pre_onset", "zero", "eeg_ms", "all62")

    assert per_channel.values.shape == (4, 62, len(archive.times["eeg_ms"]))
    assert pooled.values == pytest.approx(per_channel.values.mean(axis=1))


def test_the_seizure_state_is_stored_for_every_branch_and_arm(scored: tuple[Path, ScoreArchive]) -> None:
    """Both arms, because d_state -- does the probe move the seizure -- is a paired difference."""
    _, archive = scored

    assert set(archive.states) == {("pre_onset", "zero"), ("pre_onset", "d1")}
    for arm in ("zero", "d1"):
        state = archive.state("pre_onset", arm)
        assert state.values.shape == (4, len(archive.state_times))
        assert np.all((state.values >= 0.0) & (state.values <= 1.0))
