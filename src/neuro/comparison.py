# One manifest, one grid, one metric row per (arm, seed). Every closed-loop comparison in the
# repo -- Cost arms, Predictor arms, the seed screen, the sweep's closed-loop objective -- goes
# through here, so two runs can never disagree about what Seizure Burden means.

from __future__ import annotations

import copy
import csv
import hashlib
import json
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import Field
from simulate.config import deep_merge, load_config
from simulate.simulation import Simulation

from neuro.config import StrictConfig
from neuro.connectome import Connectome
from neuro.seizure import EZ_REGIONS, PZ_REGIONS, SEIZURE_PTP_MV, spread_profile_from_lfp, spread_summary
from neuro.validation import validate_simulation_config

if TYPE_CHECKING:
    from simulate.logger.base import BaseLogger

    from neuro.types import FloatArray


class ArmSpec(StrictConfig):
    """One arm: the simulation config it runs, plus the patch that makes it this arm."""

    config: str | None = None
    patch: dict[str, Any] = Field(default_factory=dict)


class ComparisonManifest(StrictConfig):
    """A whole comparison: the shared base config, the arms, and the Plant seeds they are paired on."""

    base: str
    seeds: list[int]
    arms: dict[str, ArmSpec]
    t_end: float | None = None
    seizure_ptp_mv: float = SEIZURE_PTP_MV


@dataclass(frozen=True)
class Cell:
    """One grid cell: an arm evaluated on one Plant seed."""

    run: str
    arm: str
    seed: int
    u_max: float
    config: dict[str, Any] = field(repr=False)


def load_manifest(path: Path) -> ComparisonManifest:
    """Read a comparison manifest YAML into its schema."""
    return ComparisonManifest.model_validate(load_config(path))


def manifest_hash(manifest: ComparisonManifest) -> str:
    """Content hash of a manifest, so a resumed grid cannot silently mix two of them."""
    payload = json.dumps(manifest.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def control_bound(config: dict[str, Any]) -> float | None:
    """Read the arm's peak per-electrode current, the Control Budget the effort metric divides by.

    MPC arms carry it in the Problem and the threshold controller in its burst amplitude. An
    unstimulated arm has no bound to speak of and reports ``None``, which is what tells the pairing
    check to leave it out of the budget comparison rather than compare it against a fiction.

    Reading the *peak* rather than the field as written is what lets arms differ on ``n_u``: the
    same budget spread over three electrodes is a list, over one a scalar, and both reduce here.
    """
    controller = config["controller"]
    if "problem" in controller:
        return float(np.max(np.abs(controller["problem"]["u_max"])))
    if "amplitude" in controller:
        return float(np.max(np.abs(controller["amplitude"])))
    return None


def arm_config(manifest: ComparisonManifest, arm: str) -> dict[str, Any]:
    """Resolve one arm to a full simulation config: its base, its patch, the manifest's run length."""
    spec = manifest.arms[arm]
    config = deep_merge(load_config(Path(spec.config or manifest.base)), spec.patch)
    if manifest.t_end is not None:
        config["t_end"] = manifest.t_end
    return config


def lfp_logging(config: dict[str, Any]) -> dict[str, Any]:
    """Copy ``config`` with region-LFP logging on, the signal every spread metric is read from.

    An arm's own config need not ask for it, and the sweep's eval config need not either: forcing it
    here rather than at each call site is what keeps the two paths scoring the same signal.
    """
    return config | {"dynamics": config["dynamics"] | {"log": "lfp"}}


def expand_grid(manifest: ComparisonManifest) -> list[Cell]:
    """Expand the manifest into one Cell per (arm, seed), seed-major.

    Seed-major so a grid cut short by its time budget still covers every arm on the seeds it
    reached: an arm comparison missing two seeds is usable, one missing two arms is not.
    """
    grid: list[Cell] = []
    for seed in manifest.seeds:
        for arm in manifest.arms:
            config = arm_config(manifest, arm)
            config["dynamics"]["seed"] = seed
            grid.append(
                Cell(run=f"{arm}_s{seed}", arm=arm, seed=seed, u_max=control_bound(config) or 1.0, config=config)
            )
    return grid


def check_arms_are_paired(grid: list[Cell]) -> None:
    """Reject a grid whose arms do not share a Plant, a Control Budget, or a Kirchhoff constraint.

    Paired seeds only cancel seed-to-seed variance if every arm integrates the same Plant, so the
    dynamics blocks must agree everywhere except ``stimulation``, which an unstimulated arm drops
    by design, and ``log``, which chooses what is recorded rather than what is integrated. The budget check stops an arm winning because it was handed more current: it covers
    the threshold arm's burst amplitude too, and compares peak magnitudes so that arms differing
    only in ``n_u`` still pair. Kirchhoff is an MPC-only constraint and is compared among the arms
    that have one.

    Raises
    ------
    ValueError
        If the arms disagree on the Plant, the Control Budget, or the Kirchhoff constraint.
    """
    # One seed's slice of the grid: every arm appears in it exactly once, and the seed itself is
    # then constant across the arms being compared, which is the pairing the check is about.
    first_seed = grid[0].seed
    by_arm = {cell.arm: cell.config for cell in grid if cell.seed == first_seed}

    def plant(config: dict[str, Any]) -> str:
        block = {key: value for key, value in config["dynamics"].items() if key not in {"stimulation", "log"}}
        return json.dumps(block, sort_keys=True, default=str)

    plants = {arm: plant(config) for arm, config in by_arm.items()}
    reference = next(iter(plants))
    mismatched = sorted(arm for arm, digest in plants.items() if digest != plants[reference])
    if mismatched:
        msg = f"Arms {mismatched} do not share the Plant block of {reference!r}; their seeds are not paired."
        raise ValueError(msg)

    budgets = {arm: bound for arm, config in by_arm.items() if (bound := control_bound(config)) is not None}
    if len(set(budgets.values())) > 1:
        msg = f"Arms disagree on the Control Budget: {budgets}."
        raise ValueError(msg)

    kirchhoff = {
        arm: config["controller"]["problem"].get("kirchhoff")
        for arm, config in by_arm.items()
        if "problem" in config["controller"]
    }
    if len(set(kirchhoff.values())) > 1:
        msg = f"Arms disagree on the Kirchhoff constraint: {kirchhoff}."
        raise ValueError(msg)


def _region_groups(connectome: Connectome) -> dict[str, list[int]]:
    """Region indices of the EZ, the PZ, and the healthy remainder that side effects show up in."""
    ez = [connectome.region_index[name] for name in EZ_REGIONS]
    pz = [connectome.region_index[name] for name in PZ_REGIONS]
    rest = sorted(set(range(len(connectome.region_labels))) - set(ez) - set(pz))
    return {"ez": ez, "pz": pz, "healthy": rest}


def spread_metrics(
    lfp: FloatArray, dt: float, connectome: Connectome, *, threshold: float = SEIZURE_PTP_MV
) -> dict[str, float]:
    """Score Seizure Burden, the spread schedule and per-group amplitude from the region LFP.

    Onsets come from :func:`neuro.seizure.spread_summary` and are censored at the run duration, as
    :meth:`neuro.seizure.SpreadSummary.score` censors them: a zone that never seizes scores the
    worst onset the run can show rather than NaN. Left as NaN, the arm that prevented recruitment
    would drop out of the very column the arm that caused it appears in, and averaging over the
    recruited regions alone would rank one region recruited late below three recruited early. An
    onset equal to the run length therefore reads "not recruited"; the ``*_recruited`` counts say
    which of the two it was.

    Parameters
    ----------
    lfp
        Region LFP as the logger stores it, shape ``(n_samples, n_nodes)``.
    """
    lfp = np.asarray(lfp)
    profile = spread_profile_from_lfp(lfp.T, dt, threshold=threshold)
    summary = spread_summary(profile, connectome)
    duration = float(lfp.shape[0] * dt)

    def censored(onset: float) -> float:
        return duration if not np.isfinite(onset) else onset

    metrics = {
        "seizure_burden": profile.burden(),
        "n_seizing_final": float(profile.n_seizing()[-1]),
        "t_pz": censored(summary.t_pz),
        "t_left_half": censored(summary.t_left_half),
        "frac_left": summary.frac_left,
    }
    for name, idx in _region_groups(connectome).items():
        metrics[f"{name}_ptp_mv"] = float(np.mean(profile.ptp[idx]))
        metrics[f"{name}_recruited"] = float(profile.n_seizing(idx)[-1])
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

    ``realtime_factor`` is the p95 solve time against the decision period the arm has to meet: the
    arms decide at different rates, so a raw solve time does not compare across them, and anything
    above 1.0 is a controller that cannot run online.
    """
    available = {name for component, name in logger.signals() if component == "controller"}
    if not {"solve_time", "success", "warmup"} <= available:
        return {}
    warmup = logger.signal("controller", "warmup")[::stride].reshape(-1).astype(bool)
    success = logger.signal("controller", "success")[::stride].reshape(-1).astype(bool)
    solve_time = logger.signal("controller", "solve_time")[::stride].reshape(-1)
    solved = ~warmup
    if not solved.any():
        keys = ("solve_success_rate", "solve_time_mean_s", "solve_time_p95_s", "realtime_factor")
        return dict.fromkeys(keys, float("nan"))
    p95 = float(np.quantile(solve_time[solved], 0.95))
    return {
        "solve_success_rate": float(np.mean(success[solved])),
        "solve_time_mean_s": float(np.mean(solve_time[solved])),
        "solve_time_p95_s": p95,
        "realtime_factor": p95 / control_dt,
    }


def score_run(config: dict[str, Any], u_max: float, *, threshold: float = SEIZURE_PTP_MV) -> dict[str, float]:
    """Run one Simulation to completion and reduce its logs to the comparison metrics."""
    config = lfp_logging(config)
    sim = Simulation.from_config(config)
    connectome = Connectome.from_config(config["dynamics"]["connectome"])
    started = time.perf_counter()
    # Log to a scratch directory so the full-rate region LFP never becomes resident, and is
    # discarded with the directory rather than kept for every cell of the grid.
    with tempfile.TemporaryDirectory(prefix="comparison_", ignore_cleanup_errors=True) as log_dir:
        sim.run(output_dir=log_dir, use_mmap=True)
        wall_time = time.perf_counter() - started
        if sim.logger is None:
            msg = "Simulation logger is missing after run."
            raise RuntimeError(msg)
        control_dt = float(config["controller"]["dt"])
        stride = max(1, round(control_dt / sim.dt))
        return {
            "wall_time_s": wall_time,
            **spread_metrics(sim.logger.signal("dynamics", "lfp"), sim.dt, connectome, threshold=threshold),
            **control_metrics(sim.logger.signal("controller", "u"), u_max, sim.dt),
            **solver_metrics(sim.logger, stride, control_dt),
        }


def score_cell(cell: Cell, *, threshold: float = SEIZURE_PTP_MV) -> dict[str, Any]:
    """Score one Cell; a failure is recorded as a row rather than sinking the whole grid."""
    row: dict[str, Any] = {"run": cell.run, "arm": cell.arm, "seed": cell.seed, "error": ""}
    try:
        row |= score_run(cell.config, cell.u_max, threshold=threshold)
    except Exception as exc:  # noqa: BLE001 -- one diverged arm must not lose the rest of the grid
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average each numeric metric over the paired seeds of every arm, with its spread."""
    cells: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not row["error"]:
            cells.setdefault(row["arm"], []).append(row)

    summary = []
    for arm, group in cells.items():
        entry: dict[str, Any] = {"arm": arm, "n_seeds": len(group)}
        for key in [key for key, value in group[0].items() if isinstance(value, float)]:
            values = np.asarray([row.get(key, np.nan) for row in group], dtype=np.float64)
            finite = np.isfinite(values).any()
            entry[key] = float(np.nanmean(values)) if finite else float("nan")
            entry[f"{key}_sd"] = float(np.nanstd(values)) if finite else float("nan")
        summary.append(entry)
    return sorted(summary, key=lambda entry: entry["arm"])


def write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    """Write metric rows as CSV, unioning the columns across arms."""
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Read back a ``rows.csv``, restoring the types a resume compares and re-summarizes against."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        rows: list[dict[str, Any]] = list(csv.DictReader(handle))
    for row in rows:
        row["seed"] = int(row["seed"])
        for key, value in row.items():
            if key not in {"run", "arm", "error", "seed"}:
                row[key] = float(value) if value else float("nan")
    return rows


def validate_grid(grid: list[Cell]) -> None:
    """Check every cell's wiring, and the arms' shared Plant and Control Budget, before any run."""
    for cell in grid:
        validate_simulation_config(copy.deepcopy(cell.config))
    check_arms_are_paired(grid)
