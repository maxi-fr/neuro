"""Compare Waveform against Observable Predictors on the closed loop at a matched Control Horizon."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import multiprocessing
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from simulate.config import load_config
from simulate.simulation import Simulation

from neuro.connectome import Connectome
from neuro.seizure import EZ_REGIONS, PZ_REGIONS, spread_profile_from_lfp
from neuro.validation import validate_simulation_config

if TYPE_CHECKING:
    from simulate.logger.base import BaseLogger

    from neuro.types import FloatArray

WAVEFORM_CONFIG = Path("configs/simulation/cmp_waveform_1p5s.yaml")
OBSERVABLE_CONFIG = Path("configs/simulation/cmp_observable_1p5s.yaml")

# Every arm plans over the same 1.5 s of Plant time, so the Control Horizon is not a confound:
# the waveform arm reaches it with 75 knots of 0.02 s, the observable arms with 15 Frames of
# 0.1 s. Only the Predictor and the rate it decides at differ.
HORIZON_S = 1.5

# Each arm is one Predictor dropped into an otherwise identical loop. Arms `observable_mlp` and
# `observable_dmd` share a config and a geometry and differ only in the Trainer that produced the
# checkpoint, so a gap between those two is attributable to backprop against closed-form SVD.
ARMS: dict[str, dict[str, Any]] = {
    "waveform_mlp": {"config": WAVEFORM_CONFIG, "artifact": "artifacts/cmp_waveform_mlp_1p5s/model"},
    "observable_mlp": {"config": OBSERVABLE_CONFIG, "artifact": "artifacts/cmp_observable_mlp_hop5/model"},
    "observable_dmd": {"config": OBSERVABLE_CONFIG, "artifact": "artifacts/cmp_observable_dmd_hop5/model"},
}

# Held-out Plant realisations: the seeds of data/experiment_excited_long/test, never seen by any
# Trainer. Paired across arms as common random numbers, so seed-to-seed variance drops out.
SEEDS = (5000, 5001, 5002, 5003, 5004, 5005, 5006)

# Open-loop reference points on the identical Plant. Without them a seizure burden is a number
# with nothing to be better than.
BASELINES = ("uncontrolled", "dc_plus")
DC_AMPLITUDE = 2.0
DC_TRIGGER_WINDOW = 0.2


def _baseline_config(base: dict[str, Any], kind: str) -> dict[str, Any]:
    """Swap the MPC loop for an open-loop baseline, leaving the Plant block untouched."""
    cfg = copy.deepcopy(base)
    plant_dt = float(cfg["dynamics"]["dt"])
    control_dt = float(cfg["controller"]["dt"])
    cfg["sensors"] = {
        "class_path": "simulate.sensor.GaussianSensor",
        "dt": plant_dt,
        "std_dev": 0.0,
        "measurement": {"class_path": "neuro.eeg.EEGMeasurement"},
    }
    cfg["estimator"] = {"class_path": "simulate.estimator.IdentityEstimator", "dt": plant_dt}

    if kind == "uncontrolled":
        cfg["dynamics"].pop("stimulation", None)  # -> NullStim, one electrode that projects nothing
        cfg["controller"] = {"class_path": "neuro.control.zero.ZeroController", "dt": control_dt, "n_u": 1}
        return cfg

    # A threshold controller with an unreachably low threshold and a burst longer than the run is
    # a constant current; it switches on one trigger window in, once the ptp buffer has filled.
    sign = 1.0 if kind == "dc_plus" else -1.0
    cfg["controller"] = {
        "class_path": "neuro.control.threshold.AmplitudeThresholdController",
        "dt": control_dt,
        "amplitude": [sign * DC_AMPLITUDE, 0.0, -sign * DC_AMPLITUDE],
        "threshold": 1e-9,
        "window": DC_TRIGGER_WINDOW,
        "burst_duration": float(cfg["t_end"]) + 1.0,
        "n_u": 3,
    }
    return cfg


def _arm_config(arm: str, t_end: float | None) -> dict[str, Any]:
    """Load one arm's base config and inject its checkpoint, leaving the shared blocks alone."""
    spec = ARMS[arm]
    cfg = load_config(spec["config"])
    cfg["controller"]["problem"]["artifact"] = spec["artifact"]
    if t_end is not None:
        cfg["t_end"] = t_end
    return cfg


def build_grid(
    *,
    arms: tuple[str, ...],
    seeds: tuple[int, ...],
    baselines: tuple[str, ...],
    t_end: float | None,
) -> list[dict[str, Any]]:
    """Expand the arms and baselines into one cell per (arm, seed).

    The baselines are generated from the waveform arm's config because the Plant block is
    identical across arms; only its Estimator and controller are replaced.
    """
    grid: list[dict[str, Any]] = []
    for arm in arms:
        cfg = _arm_config(arm, t_end)
        for seed in seeds:
            run = copy.deepcopy(cfg)
            run["dynamics"]["seed"] = seed
            grid.append(
                {
                    "run": f"{arm}_s{seed}",
                    "arm": arm,
                    "seed": seed,
                    "u_max": float(run["controller"]["problem"]["u_max"]),
                    "config": run,
                }
            )
    if baselines:
        base = _arm_config("waveform_mlp", t_end)
        for kind in baselines:
            cfg = _baseline_config(base, kind)
            for seed in seeds:
                run = copy.deepcopy(cfg)
                run["dynamics"]["seed"] = seed
                grid.append({"run": f"{kind}_s{seed}", "arm": kind, "seed": seed, "u_max": DC_AMPLITUDE, "config": run})
    # Seed-major, so a grid cut short by its time budget still covers every arm on the seeds it
    # reached: an arm comparison missing two seeds is usable, one missing two arms is not.
    grid.sort(key=lambda cell: seeds.index(cell["seed"]))
    return grid


def _region_groups(connectome: Connectome) -> dict[str, list[int]]:
    """Region indices of the EZ, the PZ, and the healthy remainder that side effects show up in."""
    ez = [connectome.region_index[name] for name in EZ_REGIONS]
    pz = [connectome.region_index[name] for name in PZ_REGIONS]
    rest = sorted(set(range(len(connectome.region_labels))) - set(ez) - set(pz))
    return {"ez": ez, "pz": pz, "healthy": rest}


def spread_metrics(lfp: FloatArray, dt: float, groups: dict[str, list[int]]) -> dict[str, float]:
    """Score Seizure Burden, per-group amplitude and EZ/PZ recruitment from the region LFP.

    Parameters
    ----------
    lfp
        Region LFP as the logger stores it, shape ``(n_samples, n_nodes)``.
    """
    profile = spread_profile_from_lfp(np.asarray(lfp).T, dt)
    metrics = {"seizure_burden": profile.burden(), "n_seizing_final": float(profile.n_seizing()[-1])}
    for name, idx in groups.items():
        metrics[f"{name}_ptp_mv"] = float(np.mean(profile.ptp[idx]))
    for name in ("ez", "pz"):
        onsets = profile.onsets[groups[name]]
        recruited = np.isfinite(onsets)
        metrics[f"{name}_recruited"] = float(np.count_nonzero(recruited))
        metrics[f"{name}_onset_s"] = float(np.mean(onsets[recruited])) if recruited.any() else float("nan")
    return metrics


def control_metrics(us: FloatArray, u_max: float, dt: float) -> dict[str, float]:
    """Score effort, delivered charge and the Kirchhoff residual of the applied control."""
    u = np.asarray(us, dtype=np.float64)
    return {
        "mean_amplitude": float(np.mean(np.abs(u) / u_max)),
        "delivered_charge": float(np.sum(np.abs(u)) * dt),
        "kirchhoff_max": float(np.max(np.abs(np.sum(u, axis=1)))),
    }


def solver_metrics(logger: BaseLogger, stride: int, control_dt: float) -> dict[str, float]:
    """Score solve time and convergence over the control steps that actually ran a solve.

    ``realtime_factor`` is the mean solve time against the decision period the arm has to meet;
    the arms decide at different rates, so a raw solve time does not compare across them.
    """
    available = {name for component, name in logger.signals() if component == "controller"}
    if not {"solve_time", "success", "warmup"} <= available:
        return {}
    warmup = logger.signal("controller", "warmup")[::stride].reshape(-1).astype(bool)
    success = logger.signal("controller", "success")[::stride].reshape(-1).astype(bool)
    solve_time = logger.signal("controller", "solve_time")[::stride].reshape(-1)
    solved = ~warmup
    if not solved.any():
        return {
            "solve_success_rate": float("nan"),
            "solve_time_mean_s": float("nan"),
            "solve_time_p95_s": float("nan"),
            "realtime_factor": float("nan"),
        }
    return {
        "solve_success_rate": float(np.mean(success[solved])),
        "solve_time_mean_s": float(np.mean(solve_time[solved])),
        "solve_time_p95_s": float(np.quantile(solve_time[solved], 0.95)),
        "realtime_factor": float(np.mean(solve_time[solved]) / control_dt),
    }


def score_run(config: dict[str, Any], u_max: float) -> dict[str, float]:
    """Run one Simulation to completion and reduce its logs to the comparison metrics."""
    sim = Simulation.from_config(config)
    groups = _region_groups(Connectome.from_config(config["dynamics"]["connectome"]))
    started = time.perf_counter()
    # Log to a scratch directory so the full-rate region LFP never becomes resident, and is
    # discarded with the directory rather than kept for every cell of the grid.
    with tempfile.TemporaryDirectory(prefix="predictor_comparison_", ignore_cleanup_errors=True) as log_dir:
        sim.run(output_dir=log_dir, use_mmap=True)
        wall_time = time.perf_counter() - started
        if sim.logger is None:
            msg = "Simulation logger is missing after run."
            raise RuntimeError(msg)
        control_dt = float(config["controller"]["dt"])
        stride = max(1, round(control_dt / sim.dt))
        return {
            "wall_time_s": wall_time,
            **spread_metrics(sim.logger.signal("dynamics", "lfp"), sim.dt, groups),
            **control_metrics(sim.logger.signal("controller", "u"), u_max, sim.dt),
            **solver_metrics(sim.logger, stride, control_dt),
        }


def _run_worker(cell: dict[str, Any]) -> dict[str, Any]:
    """Score one grid cell; a failure is recorded as a row rather than sinking the batch."""
    row: dict[str, Any] = {"run": cell["run"], "arm": cell["arm"], "seed": cell["seed"], "error": ""}
    try:
        row |= score_run(cell["config"], cell["u_max"])
    except Exception as exc:  # noqa: BLE001 -- one diverged arm must not lose the rest of the grid
        row["error"] = f"{type(exc).__name__}: {exc}"
        print(f"  {cell['run']}: {row['error']}")
    return row


def _write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    """Write metric rows as CSV, unioning the columns across arms."""
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average each numeric metric over the paired seeds of every arm."""
    cells: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not row["error"]:
            cells.setdefault(row["arm"], []).append(row)

    summary = []
    for arm, group in cells.items():
        entry: dict[str, Any] = {"arm": arm, "n_seeds": len(group)}
        for key in [k for k, v in group[0].items() if isinstance(v, float)]:
            values = np.asarray([row[key] for row in group], dtype=np.float64)
            finite = np.isfinite(values).any()
            entry[key] = float(np.nanmean(values)) if finite else float("nan")
            entry[f"{key}_sd"] = float(np.nanstd(values)) if finite else float("nan")
        summary.append(entry)
    return sorted(summary, key=lambda entry: entry["arm"])


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Predictor comparison grid."""
    parser = argparse.ArgumentParser(description="Compare Waveform and Observable Predictors at a matched horizon.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for rows.csv and summary.csv.")
    parser.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS), help="Predictor arms to run.")
    parser.add_argument("--baselines", nargs="*", default=list(BASELINES), choices=list(BASELINES))
    parser.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS), help="Plant seeds, paired across arms.")
    parser.add_argument("--t-end", type=float, default=None, help="Override the run length in seconds.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel simulation processes.")
    parser.add_argument("--dry-run", action="store_true", help="Validate every cell and list the grid, run nothing.")
    return parser.parse_args()


def main() -> None:
    """Expand the grid, run it, and write the per-run and per-arm metric tables."""
    args = parse_args()
    grid = build_grid(arms=tuple(args.arms), seeds=tuple(args.seeds), baselines=tuple(args.baselines), t_end=args.t_end)
    for cell in grid:
        validate_simulation_config(cell["config"])
    print(f"{len(grid)} runs over {len({cell['arm'] for cell in grid})} arms and {len(args.seeds)} seeds")
    if args.dry_run:
        for cell in grid:
            print(f"  {cell['run']}")
        return

    stamp = datetime.now(UTC).astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = args.output_dir or Path("results/predictor_comparison") / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # Each finished run is flushed to rows.csv immediately: the grid takes hours, and a crash
    # part-way through must not cost the runs that already succeeded.
    rows: list[dict[str, Any]] = []
    if args.workers > 1:
        with multiprocessing.Pool(processes=min(args.workers, len(grid))) as pool:
            for row in pool.imap_unordered(_run_worker, grid):
                rows.append(row)
                _write_rows(rows, out_dir / "rows.csv")
                print(f"  [{len(rows)}/{len(grid)}] {row['run']}", flush=True)
    else:
        for cell in grid:
            rows.append(_run_worker(cell))
            _write_rows(rows, out_dir / "rows.csv")
            print(f"  [{len(rows)}/{len(grid)}] {rows[-1]['run']}", flush=True)
    summary = summarize(rows)
    _write_rows(summary, out_dir / "summary.csv")
    (out_dir / "grid.json").write_text(
        json.dumps({"arms": args.arms, "seeds": args.seeds, "horizon_s": HORIZON_S}, indent=2), encoding="utf-8"
    )
    print(f"Wrote {out_dir}/rows.csv ({len(rows)} runs) and summary.csv ({len(summary)} cells)")
    for entry in summary:
        print(f"  {entry['arm']:>20s} burden={entry['seizure_burden']:.4f} +- {entry['seizure_burden_sd']:.4f}")


if __name__ == "__main__":
    main()
