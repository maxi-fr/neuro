from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from simulate.config import load_config

from neuro.validation import validate_simulation_config

if TYPE_CHECKING:
    from types import ModuleType

_ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"{name}_mod", _ROOT / "scripts" / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script() -> ModuleType:
    return _load_script("run_cost_comparison")


@pytest.fixture(scope="module")
def base() -> dict[str, Any]:
    return load_config(_ROOT / "configs" / "simulation" / "jansen_rit_oracle_mpc.yaml")


def test_grid_sweeps_horizons_on_tracking_only(script: ModuleType, base: dict[str, Any]) -> None:
    grid = script.build_grid(base, arms=("tracking", "terminal"), horizons=(2, 50), seeds=(69, 7), baselines=())
    horizons = {(cell["arm"], cell["horizon"]) for cell in grid}
    assert horizons == {("tracking", 2), ("tracking", 50), ("terminal", 50)}
    assert len(grid) == 6
    assert all(cell["config"]["controller"]["problem"]["horizon"] == cell["horizon"] for cell in grid)


def test_grid_skips_hinge_arms_below_their_segment(script: ModuleType, base: dict[str, Any]) -> None:
    grid = script.build_grid(base, arms=("psd_hinge", "observable_hinge"), horizons=(2, 5), seeds=(69,), baselines=())
    assert grid == []


def test_grid_applies_the_arm_weights_and_pairs_the_seeds(script: ModuleType, base: dict[str, Any]) -> None:
    grid = script.build_grid(base, arms=("l1_effort",), horizons=(50,), seeds=(69, 7), baselines=())
    assert [cell["seed"] for cell in grid] == [69, 7]
    assert [cell["config"]["dynamics"]["seed"] for cell in grid] == [69, 7]
    problem = grid[0]["config"]["controller"]["problem"]
    assert problem["w_u_l1"] == 10.0
    assert problem["w_u"] == 0.0
    # The base config is not mutated, so a later arm starts from the same weights.
    assert "w_u_l1" not in base["controller"]["problem"]


def test_uncontrolled_baseline_drops_the_stimulation_but_keeps_the_plant(
    script: ModuleType, base: dict[str, Any]
) -> None:
    grid = script.build_grid(base, arms=(), horizons=(50,), seeds=(69,), baselines=("uncontrolled",))
    cfg = grid[0]["config"]
    assert "stimulation" not in cfg["dynamics"]
    assert cfg["dynamics"]["params"] == base["dynamics"]["params"]
    assert cfg["controller"]["class_path"] == "neuro.control.zero.ZeroController"


def test_dc_baselines_are_opposite_polarities_of_the_same_montage(script: ModuleType, base: dict[str, Any]) -> None:
    grid = script.build_grid(base, arms=(), horizons=(50,), seeds=(69,), baselines=("dc_plus", "dc_minus"))
    plus, minus = (cell["config"]["controller"]["amplitude"] for cell in grid)
    assert plus == [-a for a in minus]
    assert sum(plus) == 0.0
    for cell in grid:
        assert cell["config"]["controller"]["burst_duration"] > cell["config"]["t_end"]


def test_every_grid_cell_is_a_consistent_simulation_config(script: ModuleType, base: dict[str, Any]) -> None:
    grid = script.build_grid(base, arms=tuple(script.ARMS), horizons=(2, 50), seeds=(69,), baselines=script.BASELINES)
    for cell in grid:
        validate_simulation_config(cell["config"])


def test_spread_metrics_separate_the_ez_from_the_healthy_remainder(script: ModuleType) -> None:
    dt, n_nodes = 0.01, 5
    t = np.arange(0, 6.0, dt)
    lfp = np.zeros((len(t), n_nodes))
    lfp[:, 0] = 30.0 * np.sin(2 * np.pi * 5 * t)  # one loudly seizing region
    groups = {"ez": [0], "pz": [1], "healthy": [2, 3, 4]}

    metrics = script.spread_metrics(lfp, dt, groups)

    assert metrics["ez_recruited"] == 1.0
    assert metrics["pz_recruited"] == 0.0
    assert metrics["ez_ptp_mv"] == pytest.approx(60.0, rel=1e-2)
    assert metrics["healthy_ptp_mv"] == 0.0
    assert np.isnan(metrics["pz_onset_s"])
    assert metrics["seizure_burden"] == pytest.approx(1.0 / n_nodes)


def test_control_metrics_score_effort_charge_and_the_kirchhoff_residual(script: ModuleType) -> None:
    us = np.tile(np.array([[1.0, 0.0, -1.0]]), (100, 1))

    metrics = script.control_metrics(us, u_max=2.0, dt=0.01)

    assert metrics["mean_amplitude"] == pytest.approx(2.0 / 3.0 / 2.0)
    assert metrics["delivered_charge"] == pytest.approx(2.0)
    assert metrics["kirchhoff_max"] == pytest.approx(0.0, abs=1e-12)


def test_summarize_averages_paired_seeds_and_drops_failed_runs(script: ModuleType) -> None:
    rows = [
        {"run": "a_s1", "arm": "a", "horizon": 50, "seed": 1, "error": "", "seizure_burden": 0.2},
        {"run": "a_s2", "arm": "a", "horizon": 50, "seed": 2, "error": "", "seizure_burden": 0.4},
        {"run": "a_s3", "arm": "a", "horizon": 50, "seed": 3, "error": "RuntimeError: diverged"},
    ]

    summary = script.summarize(rows)

    assert len(summary) == 1
    assert summary[0]["n_seeds"] == 2
    assert summary[0]["seizure_burden"] == pytest.approx(0.3)
    assert summary[0]["seizure_burden_sd"] == pytest.approx(0.1)


def test_amplitude_bias_reports_the_ratio_the_rollout_underestimates_by() -> None:
    module = _load_script("jansen_rit_horizon_error")
    actual = np.zeros((4, 10, 3))
    actual[:, ::2] = 1.0  # peak-to-peak 1.0 per region
    predicted = 0.75 * actual

    bias = module.amplitude_bias(predicted, actual, {"all": [0, 1, 2]})

    assert bias["ptp_ratio_all"] == pytest.approx(0.75)
    assert np.isnan(module.amplitude_bias(predicted[:, :1], actual[:, :1], {"all": [0]})["ptp_ratio_all"])
