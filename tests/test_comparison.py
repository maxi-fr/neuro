from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from simulate.config import load_config

from neuro.comparison import (
    ArmSpec,
    ComparisonManifest,
    arm_config,
    check_arms_are_paired,
    control_bound,
    control_metrics,
    expand_grid,
    lfp_logging,
    load_manifest,
    manifest_hash,
    read_rows,
    spread_metrics,
    summarize,
    write_rows,
)
from neuro.connectome import Connectome
from neuro.seizure import EZ_REGIONS, PZ_REGIONS
from neuro.validation import validate_simulation_config

if TYPE_CHECKING:
    from neuro.types import FloatArray

_ROOT = Path(__file__).resolve().parent.parent
_ORACLE = "configs/simulation/jansen_rit_oracle_mpc.yaml"
_UNCONTROLLED = "configs/simulation/uncontrolled.yaml"
_THRESHOLD = "configs/simulation/threshold_control.yaml"


@pytest.fixture(scope="module")
def connectome() -> Connectome:
    return Connectome.from_config({})


def _manifest(**overrides: Any) -> ComparisonManifest:  # noqa: ANN401 -- mirrors the manifest's mixed field types
    fields: dict[str, Any] = {
        "base": _ORACLE,
        "seeds": [7000, 7001],
        "arms": {"tracking": ArmSpec(), "terminal": ArmSpec(patch={"controller": {"problem": {"w_y_terminal": 10.0}}})},
        "t_end": 4.0,
    }
    return ComparisonManifest(**(fields | overrides))


def test_arm_patch_is_merged_without_touching_the_base_config() -> None:
    manifest = _manifest()

    problem = arm_config(manifest, "terminal")["controller"]["problem"]

    assert problem["w_y_terminal"] == 10.0
    assert problem["w_y"] == load_config(_ROOT / _ORACLE)["controller"]["problem"]["w_y"]
    assert "w_y_terminal" not in arm_config(manifest, "tracking")["controller"]["problem"]


def test_scoring_forces_the_lfp_log_the_spread_metrics_read() -> None:
    config = load_config(_ROOT / "configs/simulation/threshold_control.yaml")

    assert "log" not in config["dynamics"]  # the arm's own config never asked for it
    assert lfp_logging(config)["dynamics"]["log"] == "lfp"
    assert "log" not in config["dynamics"]  # ... and the caller's config is left alone


def test_grid_is_seed_major_so_a_truncated_run_still_covers_every_arm() -> None:
    grid = expand_grid(_manifest())

    assert [(cell.arm, cell.seed) for cell in grid] == [
        ("tracking", 7000),
        ("terminal", 7000),
        ("tracking", 7001),
        ("terminal", 7001),
    ]
    assert all(cell.config["dynamics"]["seed"] == cell.seed for cell in grid)
    assert all(cell.config["t_end"] == 4.0 for cell in grid)


def test_every_grid_cell_is_a_consistent_simulation_config() -> None:
    for cell in expand_grid(_manifest()):
        validate_simulation_config(cell.config)


def test_unstimulated_arm_keeps_the_plant_it_drops_the_stimulation_from() -> None:
    manifest = _manifest(arms={"tracking": ArmSpec(), "uncontrolled": ArmSpec(config=_UNCONTROLLED)})
    grid = expand_grid(manifest)

    check_arms_are_paired(grid)  # dropping `stimulation` alone does not unpair the seeds

    uncontrolled = next(cell.config for cell in grid if cell.arm == "uncontrolled")
    assert "stimulation" not in uncontrolled["dynamics"]
    assert uncontrolled["dynamics"]["params"] == load_config(_ROOT / _ORACLE)["dynamics"]["params"]


def test_pairing_check_rejects_arms_that_do_not_share_the_plant() -> None:
    manifest = _manifest()
    manifest.arms["terminal"].patch["dynamics"] = {"connectome": {"K": 0.9}}

    with pytest.raises(ValueError, match="do not share the Plant block"):
        check_arms_are_paired(expand_grid(manifest))


def test_pairing_check_rejects_a_bigger_control_budget() -> None:
    manifest = _manifest()
    manifest.arms["terminal"].patch["controller"] = {"problem": {"u_max": 4.0}}

    with pytest.raises(ValueError, match="disagree on the Control Budget"):
        check_arms_are_paired(expand_grid(manifest))


def test_pairing_check_reads_the_budget_as_a_peak_so_arms_may_differ_on_n_u() -> None:
    # Two electrodes carrying 2.0 each is the same Control Budget as one carrying 2.0; the user
    # said arms may differ on n_u, so the shape of `u_max` must not decide the comparison.
    manifest = _manifest()
    manifest.arms["terminal"].patch["controller"] = {"problem": {"u_max": [2.0, -2.0]}}

    check_arms_are_paired(expand_grid(manifest))


def test_pairing_check_covers_the_threshold_arm_it_cannot_read_a_u_max_from() -> None:
    manifest = _manifest(arms={"tracking": ArmSpec(), "threshold": ArmSpec(config=_THRESHOLD)})
    over_budget = deepcopy(manifest)
    over_budget.arms["threshold"].patch["controller"] = {"amplitude": [8.0, 0.0, -8.0]}

    check_arms_are_paired(expand_grid(manifest))  # the shipped burst amplitude matches u_max
    with pytest.raises(ValueError, match="disagree on the Control Budget"):
        check_arms_are_paired(expand_grid(over_budget))


def test_an_unstimulated_arm_has_no_budget_to_compare() -> None:
    assert control_bound(load_config(_ROOT / _UNCONTROLLED)) is None
    assert control_bound(load_config(_ROOT / _ORACLE)) == pytest.approx(2.0)


def test_manifest_hash_tracks_the_content_not_the_object() -> None:
    assert manifest_hash(_manifest()) == manifest_hash(_manifest())
    assert manifest_hash(_manifest()) != manifest_hash(_manifest(seeds=[7000, 7002]))


def test_shipped_manifests_expand_into_valid_paired_grids() -> None:
    for path in sorted((_ROOT / "configs" / "comparison").glob("*.yaml")):
        grid = expand_grid(load_manifest(path))
        check_arms_are_paired(grid)
        for cell in grid:
            validate_simulation_config(cell.config)


def _seizing_lfp(dt: float, connectome: Connectome, regions: list[str]) -> FloatArray:
    """A run in which only ``regions`` ring loudly, shape ``(n_samples, n_nodes)``."""
    t = np.arange(0.0, 6.0, dt)
    lfp = np.zeros((len(t), len(connectome.region_labels)))
    for name in regions:
        lfp[:, connectome.region_index[name]] = 30.0 * np.sin(2 * np.pi * 5 * t)
    return lfp


def test_spread_metrics_separate_the_ez_from_the_healthy_remainder(connectome: Connectome) -> None:
    dt = 0.01
    lfp = _seizing_lfp(dt, connectome, [*EZ_REGIONS])

    metrics = spread_metrics(lfp, dt, connectome, threshold=5.0)

    assert metrics["ez_ptp_mv"] == pytest.approx(60.0, rel=1e-2)
    assert metrics["ez_recruited"] == len(EZ_REGIONS)
    assert metrics["pz_recruited"] == 0.0
    assert metrics["healthy_ptp_mv"] == 0.0
    assert metrics["pz_ptp_mv"] == 0.0
    assert metrics["seizure_burden"] == pytest.approx(len(EZ_REGIONS) / len(connectome.region_labels))


def test_spread_metrics_censor_an_unrecruited_zone_instead_of_dropping_it(connectome: Connectome) -> None:
    dt = 0.01
    ez_only = spread_metrics(_seizing_lfp(dt, connectome, [*EZ_REGIONS]), dt, connectome, threshold=5.0)
    both = spread_metrics(_seizing_lfp(dt, connectome, [*EZ_REGIONS, *PZ_REGIONS]), dt, connectome, threshold=5.0)

    # A PZ that never seizes scores the worst possible onset, not NaN: the arm that prevented
    # recruitment must not vanish from the column that the arm which caused it appears in.
    assert np.isfinite(ez_only["t_pz"])
    assert ez_only["t_pz"] > both["t_pz"]


def test_control_metrics_score_effort_charge_and_the_kirchhoff_residual() -> None:
    us = np.tile(np.array([[1.0, 0.0, -1.0]]), (100, 1))

    metrics = control_metrics(us, u_max=2.0, dt=0.01)

    assert metrics["mean_amplitude"] == pytest.approx(2.0 / 3.0 / 2.0)
    assert metrics["delivered_charge"] == pytest.approx(2.0)
    assert metrics["kirchhoff_max"] == pytest.approx(0.0, abs=1e-12)


def test_summarize_averages_paired_seeds_and_drops_failed_runs() -> None:
    rows = [
        {"run": "a_s1", "arm": "a", "seed": 1, "error": "", "seizure_burden": 0.2},
        {"run": "a_s2", "arm": "a", "seed": 2, "error": "", "seizure_burden": 0.4},
        {"run": "a_s3", "arm": "a", "seed": 3, "error": "RuntimeError: diverged"},
    ]

    summary = summarize(rows)

    assert len(summary) == 1
    assert summary[0]["n_seeds"] == 2
    assert summary[0]["seizure_burden"] == pytest.approx(0.3)
    assert summary[0]["seizure_burden_sd"] == pytest.approx(0.1)


def test_rows_survive_the_round_trip_a_resume_reads_them_back_through(tmp_path: Path) -> None:
    rows = [
        {"run": "a_s1", "arm": "a", "seed": 7000, "error": "", "seizure_burden": 0.25},
        {"run": "b_s1", "arm": "b", "seed": 7000, "error": "RuntimeError: diverged"},
    ]
    write_rows(rows, tmp_path / "rows.csv")

    restored = read_rows(tmp_path / "rows.csv")

    assert [(row["arm"], row["seed"]) for row in restored] == [("a", 7000), ("b", 7000)]
    assert restored[0]["seizure_burden"] == pytest.approx(0.25)
    assert np.isnan(restored[1]["seizure_burden"])
