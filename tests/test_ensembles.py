from __future__ import annotations

import importlib.util
import json
import math
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.signal import decimate

from experiments.observable_predictability.archive import EVALUATION_SCORES_FILE, SCHEMA_VERSION, load_evaluation_bundle
from experiments.observable_predictability.ensemble import (
    ARM_ZERO,
    Arm,
    Branch,
    EnsembleConfig,
    PlantPair,
    all_arms,
    build_plants,
    default_plant_params,
    eeg_history_path,
    eeg_path,
    generate,
    generate_iter,
    load_manifest,
    load_region_rollout,
    load_scalp_rollout,
    manifest_identities,
    region_history_path,
    region_path,
    reseed,
    resolve_identity,
    run_child,
    run_parent,
    run_segment,
)
from experiments.observable_predictability.features import condition_waveform_pair
from experiments.observable_predictability.metrics import (
    RolloutIdentity,
    validate_identities,
)
from neuro.plant.seizure import SPREAD_WINDOW_S

if TYPE_CHECKING:
    from collections.abc import Sequence

    from neuro.types import FloatArray

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_predictability_experiment.py"


def runner_main(argv: Sequence[str] | None = None) -> int:
    """Load and execute scripts/run_predictability_experiment.py main entry point."""
    spec = importlib.util.spec_from_file_location("run_predictability_experiment", _SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return int(module.main(argv))


_ZERO = (0.0, 0.0, 0.0)
_D1 = (2.0, 0.0, -2.0)
ARM_D1 = Arm("d1", _D1)
ARM_STIM = Arm("stim", _D1)


@pytest.fixture(scope="module")
def plants() -> PlantPair:
    """Build the shared healthy and seizure Plant pair from repo-local configuration."""
    return build_plants()


# --- the branching gotcha ------------------------------------------------------------------


def test_two_segments_reproduce_one_long_run(plants: PlantPair) -> None:
    """Branching preserves x, the delay history and the clock -- the whole point of deepcopy."""
    split = reseed(plants.seizure, 7)
    stitched = np.concatenate([run_segment(split, 250, _ZERO), run_segment(split, 250, _ZERO)], axis=1)

    whole = run_segment(reseed(plants.seizure, 7), 500, _ZERO)

    assert np.array_equal(stitched, whole)


def test_reseed_carries_the_full_state_and_only_swaps_the_noise(plants: PlantPair) -> None:
    """Verify reseed keeps state, delay history, and clock while changing future noise."""
    parent = reseed(plants.seizure, 7)
    run_segment(parent, 300, _ZERO)

    child = reseed(parent, 99)

    assert np.array_equal(child.x, parent.x)
    assert np.array_equal(child.history, parent.history)
    assert child.next_update_time == parent.next_update_time
    assert not np.array_equal(run_segment(child, 100, _ZERO), run_segment(parent, 100, _ZERO))


def test_a_snapshot_is_not_advanced_by_the_children_branched_off_it(plants: PlantPair) -> None:
    """Verify branching children from a snapshot leaves the snapshot state untouched."""
    snapshot = reseed(plants.seizure, 7)
    run_segment(snapshot, 300, _ZERO)
    x_before, clock_before = snapshot.x.copy(), snapshot.next_update_time

    for seed in range(3):
        run_segment(reseed(snapshot, seed), 200, _ZERO)

    assert np.array_equal(snapshot.x, x_before)
    assert snapshot.next_update_time == clock_before


def test_the_same_seed_and_arm_reproduce_the_same_rollout(plants: PlantPair) -> None:
    """Verify identical replicate seed and arm produce bit-identical Rollouts."""
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


# --- plants & dry-run ----------------------------------------------------------------------


def test_the_two_plants_differ_only_in_the_excitatory_gain(plants: PlantPair) -> None:
    """Verify healthy and seizure Plants share state, connectome, and noise, differing only in A."""
    assert np.array_equal(plants.healthy.x, plants.seizure.x)
    assert np.array_equal(plants.healthy.history, plants.seizure.history)
    assert np.array_equal(plants.healthy.w_weights, plants.seizure.w_weights)
    assert plants.healthy.K == plants.seizure.K
    assert plants.healthy.net_params.sigma == plants.seizure.net_params.sigma
    assert np.array_equal(np.unique(plants.healthy.net_params.A), [3.25])
    assert np.array_equal(np.unique(plants.seizure.net_params.A), [3.25, 3.4, 3.6])


def test_default_plant_params_are_repo_local_and_self_contained() -> None:
    """Verify default_plant_params has no external path references and initializes build_plants."""
    params = default_plant_params()
    serialized = json.dumps(params)
    assert "Closed-loop/Simulation/params.json" not in serialized
    assert params["regime"] == "seizure"
    pair = build_plants(params)
    assert pair.leadfield.shape == (62, 76)


def test_dry_run_verifies_plants_and_writes_no_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify --dry-run initializes both Plants from repo-local config without creating files."""
    out_dir = tmp_path / "dry_run_out"
    code = runner_main(["--dry-run", "--out", str(out_dir)])
    captured = capsys.readouterr().out

    assert code == 0
    assert "repo-local default_plant_params()" in captured
    assert "leadfield  : 62 channels x 76 regions" in captured
    assert not out_dir.exists()


def test_the_arms_are_kcl_valid_on_the_three_electrode_montage(plants: PlantPair) -> None:
    """Verify every configured arm obeys Kirchhoff's Current Law and the Control Budget."""
    assert plants.seizure.n_controls == 3
    for arm in (*all_arms(EnsembleConfig().branches), ARM_STIM):
        assert len(arm.current) == plants.seizure.n_controls
        assert sum(arm.current) == pytest.approx(0.0)
        assert all(-2.0 <= c <= 2.0 for c in arm.current)


# --- parents and children ------------------------------------------------------------------


def test_run_parent_snapshots_at_every_branch_time(plants: PlantPair) -> None:
    """Verify run_parent records snapshots and histories at each branch time."""
    cfg = EnsembleConfig(
        n_parents=1,
        n_children=1,
        rollout_s=0.1,
        branches=(
            Branch("early", "seizure", 1.2, arms=(ARM_ZERO,)),
            Branch("late", "seizure", 1.75, arms=(ARM_ZERO,)),
        ),
        chunk_s=0.25,
    )

    parent = run_parent(plants, cfg, "seizure", 0)

    assert set(parent.snapshots) == {"early", "late"}
    assert parent.snapshots["early"].next_update_time == pytest.approx(1.2)
    assert parent.snapshots["late"].next_update_time == pytest.approx(1.75)
    assert parent.onsets.shape == (len(plants.connectome.region_labels),)
    assert 0 <= parent.n_seizing_final <= len(plants.connectome.region_labels)
    assert parent.history_eeg["early"].shape == (62, 12_000)
    assert parent.history_eeg["late"].shape == (62, 17_500)


def test_run_parent_carries_one_window_of_pre_branch_history(plants: PlantPair) -> None:
    """Full rate, not decimated: run_child decimates across the branch instant in one pass."""
    cfg = EnsembleConfig(
        n_parents=1,
        n_children=1,
        rollout_s=0.1,
        branches=(Branch("late", "seizure", 1.75, arms=(ARM_ZERO,)),),
        chunk_s=0.25,
    )

    parent = run_parent(plants, cfg, "seizure", 0)

    assert parent.pre_regions["late"].shape == (len(plants.connectome.region_labels), 10_000)
    assert parent.pre_regions["late"].dtype == np.float64


def test_a_branch_at_exactly_the_pre_branch_window_fills_it_with_nothing_to_spare(plants: PlantPair) -> None:
    """``pre_onset`` branches at ``t_branch == SPREAD_WINDOW_S``, so the whole parent run is its history."""
    cfg = EnsembleConfig(
        n_parents=1,
        n_children=1,
        rollout_s=0.2,
        branches=(
            Branch("pre_onset", "seizure", SPREAD_WINDOW_S, arms=(ARM_ZERO,)),
            Branch("late", "seizure", 1.75, arms=(ARM_ZERO,)),
        ),
        chunk_s=0.3,
    )

    parent = run_parent(plants, cfg, "seizure", 0)
    pre_region = parent.pre_regions["pre_onset"]
    _, region = run_child(parent.snapshots["pre_onset"], plants, cfg, seed=1, arm=ARM_ZERO, pre_region=pre_region)

    assert pre_region.shape[1] == round(SPREAD_WINDOW_S / plants.seizure.dt)
    assert region.shape[1] == round((SPREAD_WINDOW_S + cfg.rollout_s) * cfg.region_fs)


@pytest.mark.parametrize("chunk_s", [0.5, 0.4])
def test_the_pre_branch_window_is_the_trailing_history_whatever_the_chunking(plants: PlantPair, chunk_s: float) -> None:
    """Taking the window off the covering chunks only must be bit-identical to slicing the whole run."""
    branches = (Branch("late", "seizure", 1.75, arms=(ARM_ZERO,)),)

    def parent(chunk: float) -> FloatArray:
        cfg = EnsembleConfig(n_parents=1, n_children=1, rollout_s=0.1, branches=branches, chunk_s=chunk)
        return run_parent(plants, cfg, "seizure", 0).pre_regions["late"]

    assert np.array_equal(parent(chunk_s), parent(1.75))


def test_run_child_returns_scalp_at_full_rate_and_region_decimated(plants: PlantPair) -> None:
    """Verify run_child returns full-rate Raw EEG and decimated regional LFP."""
    cfg = EnsembleConfig(rollout_s=0.5, region_fs=1000.0)
    snapshot = reseed(plants.seizure, 7)
    run_segment(snapshot, 200, _ZERO)
    n_nodes = len(plants.connectome.region_labels)
    pre_region = np.zeros((n_nodes, 10_000), dtype=np.float64)

    eeg, region = run_child(snapshot, plants, cfg, seed=3, arm=Arm("zero", _ZERO), pre_region=pre_region)

    assert eeg.shape == (62, 5000)
    assert region.shape == (n_nodes, 1500)
    assert region.dtype == np.float32
    assert np.abs(region[:, :400]).max() < 1e-6
    assert np.all(np.isfinite(eeg))


def test_the_region_rollout_is_decimated_in_one_pass_across_the_branch(plants: PlantPair) -> None:
    """Decimating the two segments apart puts two anti-alias edge transients on the branch instant."""
    cfg = EnsembleConfig(rollout_s=0.5, region_fs=1000.0)
    dt = plants.seizure.dt
    factor = round(1.0 / (dt * cfg.region_fs))
    assert factor > 1, "a decimation factor of 1 cannot show the seam"

    snapshot = reseed(plants.seizure, 7)
    run_segment(snapshot, 2000, _ZERO)
    pre_region = run_segment(reseed(snapshot, 11), round(SPREAD_WINDOW_S / dt), _ZERO)

    _, region = run_child(snapshot, plants, cfg, seed=3, arm=ARM_ZERO, pre_region=pre_region)

    rollout = run_segment(reseed(snapshot, 3), round(cfg.rollout_s / dt), _ZERO)
    one_pass = decimate(np.concatenate([pre_region, rollout], axis=1), factor, axis=1)
    two_pass = np.concatenate([decimate(pre_region, factor, axis=1), decimate(rollout, factor, axis=1)], axis=1)
    seam = round(SPREAD_WINDOW_S * cfg.region_fs)

    assert region.shape[1] == round((SPREAD_WINDOW_S + cfg.rollout_s) * cfg.region_fs)
    assert np.array_equal(region, one_pass.astype(np.float32))
    assert np.abs(two_pass - one_pass)[:, seam - 5 : seam + 5].max() > 1e-3


def test_the_production_join_still_decimates_to_the_stored_region_length(plants: PlantPair) -> None:
    """One decimation of the joined signal must land on _build_stores' n_region, not one short."""
    cfg = EnsembleConfig()
    factor = round(1.0 / (plants.seizure.dt * cfg.region_fs))
    n_joined = round((SPREAD_WINDOW_S + cfg.rollout_s) / plants.seizure.dt)

    assert (factor, n_joined) == (10, 40_000)
    assert math.ceil(n_joined / factor) == round((SPREAD_WINDOW_S + cfg.rollout_s) * cfg.region_fs) == 4000


# --- generation, splits, paired arms, and fail-fast validation -----------------------------


def test_generate_writes_one_store_per_branch_and_arm_plus_a_manifest(plants: PlantPair, tmp_path: Path) -> None:
    """Verify generate writes continuation and history stores plus a schema_version=2 manifest."""
    cfg = EnsembleConfig(
        n_parents=1,
        n_children=2,
        rollout_s=0.3,
        branches=(Branch("pre_onset", "seizure", 1.75, arms=(ARM_ZERO, ARM_D1)),),
        reference_seeds=(101,),
        dev_seeds=(201,),
        region_fs=1000.0,
        chunk_s=0.5,
    )

    manifest = generate(tmp_path, cfg, plants)

    for arm in cfg.branches[0].arms:
        eeg = np.load(eeg_path(tmp_path, "pre_onset", arm.name), mmap_mode="r")
        region = np.load(region_path(tmp_path, "pre_onset", arm.name), mmap_mode="r")
        assert eeg.shape == (2, 62, 3000)
        assert region.shape == (2, 76, 1300)
        assert np.all(np.isfinite(np.asarray(eeg)))
        assert np.any(np.asarray(eeg) != 0.0)

    assert manifest == load_manifest(tmp_path)
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["channel_sets"]["lTCI"] == [27, 55, 12, 14]
    assert [c["name"] for c in manifest["arms"]] == ["zero", "d1"]
    assert manifest["branches"][0]["arms"] == ["zero", "d1"]
    assert manifest["pre_branch_s"] == SPREAD_WINDOW_S
    assert manifest["parents"][0]["seed"] == cfg.parent_seed("seizure", 0)
    assert len(manifest["parents"][0]["onsets"]) == 76


def test_conditioned_eeg_store_matches_conditioning_the_raw_replay(plants: PlantPair, tmp_path: Path) -> None:
    """Verify stored 50 Hz samples and history match offline conditioning exactly."""
    cfg = EnsembleConfig(
        n_parents=1,
        n_children=2,
        rollout_s=0.3,
        branches=(Branch("pre_onset", "seizure", 1.75, arms=(ARM_ZERO, ARM_D1)),),
    )
    raw_dir = tmp_path / "raw"
    compact_dir = tmp_path / "compact"
    raw_manifest = generate(raw_dir, cfg, plants)
    generate(compact_dir, replace(cfg, store_conditioned_eeg=True), plants)
    compact_manifest = load_manifest(compact_dir)
    assert compact_manifest["fs"] == 50
    for identity in manifest_identities(raw_manifest):
        history, continuation = load_scalp_rollout(raw_dir, identity, raw_manifest)
        expected_history, expected_continuation = condition_waveform_pair(
            continuation.T, raw_manifest["fs"], history_raw=history.T
        )
        stored_history, stored_continuation = load_scalp_rollout(compact_dir, identity, compact_manifest)
        np.testing.assert_array_equal(stored_history.T, expected_history)
        np.testing.assert_array_equal(stored_continuation.T, expected_continuation)


def test_zero_current_cli_scores_seizure_only_evaluation(tmp_path: Path) -> None:
    """Verify zero-current evaluation skips healthy replays and still fits healthy references."""
    code = runner_main(
        [
            "--generate",
            "--score",
            "--out",
            str(tmp_path),
            "--n-reference",
            "1",
            "--n-dev",
            "1",
            "--n-parents",
            "1",
            "--n-children",
            "2",
            "--rollout",
            "0.60",
            "--pre-branch",
            "1.20",
            "--branch-times",
            "1.2,1.2,1.5,1.8,2.0",
        ]
    )
    assert code == 0
    manifest = load_manifest(tmp_path)
    assert {identity.plant for identity in manifest_identities(manifest, split="reference")} == {"healthy"}
    assert {identity.plant for identity in manifest_identities(manifest, split="dev")} == {"healthy"}
    assert {identity.plant for identity in manifest_identities(manifest, split="eval")} == {"seizure"}
    assert {branch["plant"] for branch in manifest["branches"]} == {"seizure"}
    assert {identity.arm for identity in manifest_identities(manifest)} == {"zero"}
    assert not eeg_path(tmp_path, "healthy", "zero").exists()
    bundle = load_evaluation_bundle(tmp_path)
    assert set(bundle.potential_predictability) == {"waveform", "spectral", "connectivity"}
    with np.load(tmp_path / EVALUATION_SCORES_FILE) as archive:
        assert not any(key.startswith(("spread|", "paired|")) for key in archive.files)


def test_atomic_manifest_is_only_published_after_all_stores_complete(plants: PlantPair, tmp_path: Path) -> None:
    """Verify manifest.json does not exist on disk until generate_iter finishes flushing."""
    cfg = EnsembleConfig(
        n_parents=2,
        n_children=1,
        rollout_s=0.1,
        branches=(Branch("pre_ictal", "seizure", 1.2, arms=(ARM_ZERO, ARM_STIM)),),
        reference_seeds=(101,),
        dev_seeds=(201,),
        chunk_s=0.6,
    )
    iterator = generate_iter(tmp_path, cfg, plants)
    first_update = next(iterator)
    with pytest.raises(FileExistsError):
        next(generate_iter(tmp_path, cfg, plants))

    assert first_update["schema_version"] == SCHEMA_VERSION
    assert not (tmp_path / "manifest.json").exists()

    for _ in iterator:
        pass

    assert (tmp_path / "manifest.json").exists()
    assert not (tmp_path / "manifest.json.tmp").exists()
    committed_manifest = (tmp_path / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError, match="fresh output directory"):
        generate(tmp_path, cfg, plants)
    assert (tmp_path / "manifest.json").read_bytes() == committed_manifest


def test_fail_fast_validation_rejects_corrupted_manifest_metadata(plants: PlantPair, tmp_path: Path) -> None:
    """Verify load_manifest rejects schema, duplicate/missing/reordered identity, and pairing errors."""
    cfg = EnsembleConfig(
        n_parents=1,
        n_children=2,
        rollout_s=0.2,
        branches=(Branch("pre_ictal", "seizure", 1.2, arms=(ARM_ZERO, ARM_STIM)),),
        reference_seeds=(101,),
        dev_seeds=(201,),
        chunk_s=0.6,
    )
    manifest = generate(tmp_path, cfg, plants)
    manifest_file = tmp_path / "manifest.json"

    bad_schema = deepcopy(manifest)
    bad_schema["schema_version"] = 1
    manifest_file.write_text(json.dumps(bad_schema), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_manifest(tmp_path)

    dup_manifest = deepcopy(manifest)
    dup_manifest["identities"].append(dup_manifest["identities"][0])
    manifest_file.write_text(json.dumps(dup_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate rollout identity key"):
        load_manifest(tmp_path)

    short_manifest = deepcopy(manifest)
    short_manifest["identities"] = short_manifest["identities"][:-1]
    manifest_file.write_text(json.dumps(short_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="identity count mismatch"):
        load_manifest(tmp_path)

    swapped_manifest = deepcopy(manifest)
    swapped_manifest["identities"][-1], swapped_manifest["identities"][-2] = (
        swapped_manifest["identities"][-2],
        swapped_manifest["identities"][-1],
    )
    manifest_file.write_text(json.dumps(swapped_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        load_manifest(tmp_path)

    bad_pair = deepcopy(manifest)
    bad_pair["identities"][-1]["replicate_seed"] += 999
    manifest_file.write_text(json.dumps(bad_pair), encoding="utf-8")
    with pytest.raises(ValueError, match="Mismatched replicate_seed"):
        load_manifest(tmp_path)

    partial_parents = deepcopy(manifest)
    partial_parents["splits"]["eval"]["parents"] = []
    manifest_file.write_text(json.dumps(partial_parents), encoding="utf-8")
    with pytest.raises(ValueError, match="Partial manifest"):
        load_manifest(tmp_path)


def test_fail_fast_validation_rejects_missing_stores_or_unknown_keys(plants: PlantPair, tmp_path: Path) -> None:
    """Verify load_manifest and resolve_identity reject missing files and unknown keys."""
    cfg = EnsembleConfig(
        n_parents=1,
        n_children=1,
        rollout_s=0.2,
        branches=(Branch("pre_ictal", "seizure", 1.2, arms=(ARM_ZERO, ARM_STIM)),),
        reference_seeds=(101,),
        dev_seeds=(201,),
        chunk_s=0.6,
    )
    manifest = generate(tmp_path, cfg, plants)

    with pytest.raises(KeyError, match="not found in manifest"):
        resolve_identity(manifest, ("seizure", 999999, "pre_ictal", 0, "zero"))

    target_store = eeg_path(tmp_path, "pre_ictal", "stim", "eval")
    backup_store = tmp_path / "backup_stim.npy"
    target_store.replace(backup_store)
    with pytest.raises(FileNotFoundError, match="Missing Rollout store"):
        load_manifest(tmp_path)
    backup_store.replace(target_store)

    target_hist = eeg_history_path(tmp_path, "pre_ictal", "eval")
    backup_hist = tmp_path / "backup_hist.npy"
    target_hist.replace(backup_hist)
    with pytest.raises(FileNotFoundError, match="Missing pre-branch history store"):
        load_manifest(tmp_path)
    backup_hist.replace(target_hist)

    assert region_history_path(tmp_path, "pre_ictal", "eval").exists()


def test_generated_arms_share_their_noise_realisation(plants: PlantPair, tmp_path: Path) -> None:
    """Two arms carrying the same (null) command must produce identical trajectories."""
    cfg = EnsembleConfig(
        n_parents=1,
        n_children=2,
        rollout_s=0.3,
        branches=(Branch("pre_onset", "seizure", 1.75, arms=(Arm("first", _ZERO), Arm("second", _ZERO))),),
        reference_seeds=(),
        dev_seeds=(),
        chunk_s=0.5,
    )

    generate(tmp_path, cfg, plants)
    first = np.load(eeg_path(tmp_path, "pre_onset", "first"))
    second = np.load(eeg_path(tmp_path, "pre_onset", "second"))

    assert np.array_equal(first, second)
    assert not np.array_equal(first[0], first[1])


def test_default_branches_use_zero_current() -> None:
    """Verify the default branch schedule uses zero current throughout."""
    cfg = EnsembleConfig()

    by_name = {b.name: [a.name for a in b.arms] for b in cfg.branches}
    assert by_name == {
        "healthy": ["zero"],
        "pre_onset": ["zero"],
        "ez_ignited": ["zero"],
        "mid_spread": ["zero"],
        "saturated": ["zero"],
    }
    assert cfg.n_parents == 16
    assert cfg.n_children == 8
    assert cfg.pre_branch_s == 3.5
    assert len(validate_identities(cfg.expected_identities())) == len(cfg.expected_identities())


def test_all_arms_deduplicates_across_branches_in_first_seen_order() -> None:
    """Verify all_arms preserves first-seen order without duplicates."""
    branches = (Branch("a", "seizure", 1.0, (ARM_ZERO, ARM_D1)), Branch("b", "seizure", 2.0, (ARM_D1, ARM_STIM)))
    assert all_arms(branches) == (ARM_ZERO, ARM_D1, ARM_STIM)


def test_ensemble_config_rejects_a_branch_too_early_for_its_pre_branch_history() -> None:
    """Verify EnsembleConfig rejects branches shorter than SPREAD_WINDOW_S or overlapping split seeds."""
    with pytest.raises(ValueError, match="shorter than the"):
        EnsembleConfig(branches=(Branch("too_short", "seizure", 0.3, arms=(ARM_ZERO,)),))

    with pytest.raises(ValueError, match="pairwise disjoint"):
        EnsembleConfig(reference_seeds=(101, 102), dev_seeds=(102, 202))

    with pytest.raises(ValueError, match="Duplicate rollout identity key"):
        validate_identities(
            [
                RolloutIdentity("seizure", 69, "ictal", 0, "zero"),
                RolloutIdentity("seizure", 69, "ictal", 0, "zero"),
            ]
        )
