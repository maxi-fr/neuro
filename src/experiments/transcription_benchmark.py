from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
from simulate.config import load_config
from simulate.simulation import Simulation
from trajopt.mpc import MPC
from trajopt.program import WarmStart
from trajopt.transcription.ipopt import Ipopt
from trajopt.transcription.single_shooting import SingleShooting

from experiments.closed_loop_metrics import current_hold_durations, delivered_charge, solve_attempt_mask
from neuro.config import StftGeometry
from neuro.control.multiple_shooting import MultipleShooting
from neuro.control.narx import NarxShooting, narx_dimensions
from neuro.control.problems import NeuroProblem, build_observable_problem, build_waveform_problem
from neuro.control.solvers import IPOPT_DEFAULTS
from neuro.filtering import antialias_filter
from neuro.predictor.base import AutoregressiveModel
from neuro.spectral import HealthyReference, compute_log_power_frames

if TYPE_CHECKING:
    from trajopt.transcription.result import Solver

logger = logging.getLogger(__name__)

BENCHMARK_DIR = Path("artifacts/thesis/transcription_benchmark")
UNCONTROLLED_DIR = Path("artifacts/thesis/sweep_2026_11/observable_benchmark/runs")
SNAPSHOTS_PATH = BENCHMARK_DIR / "snapshots.npz"
CELLS_DIR = BENCHMARK_DIR / "cells"

SEEDS: tuple[int, ...] = (7100, 7101, 7102, 7103, 7104, 7105, 7106, 7107, 7108, 7109)
SNAPSHOT_TIMES_S: tuple[float, ...] = (2.40, 2.70, 3.00, 3.30, 3.60, 3.90, 4.20, 4.50, 4.80, 5.10)
WARMUP_TIME_S: float = 2.10
HORIZONS_S: tuple[float, ...] = (0.30, 0.60, 1.02, 1.50)
TRANSCRIPTIONS: tuple[str, ...] = ("single_shooting", "narx", "multiple_shooting")
MAX_CPU_TIME_S: float = 120.0
CLOSED_LOOP_MEDIAN_CAP_S: float = 40.0
MAX_JAC_BYTES: float = 12.0e9
_EARLY_TIMEOUT_SNAPSHOTS: int = 3
_IPOPT_CPU_TIMEOUT_STATUS: int = -4
_SUBPROCESS_TIMEOUT_RC: int = -9
_SINGLE_ITER_TIMEOUT_MULTIPLIER: float = 3.0
CLOSED_LOOP_SEED: int = 7100
CLOSED_LOOP_START_S: float = 1.38
CLOSED_LOOP_REPLAN_DT: float = 0.06
CLOSED_LOOP_N_REPLANS: int = 20
CLOSED_LOOP_T_END_S: float = CLOSED_LOOP_START_S + CLOSED_LOOP_N_REPLANS * CLOSED_LOOP_REPLAN_DT

_SPECTRAL_GEOM = StftGeometry(
    n_segment=25,
    n_hop=3,
    band_hz=(1.0, 25.0),
    window="hann",
    asymmetric_window=True,
    n_bin_pool=1,
    kernel="linear",
    kernel_width=1,
)


@dataclass(frozen=True)
class PredictorSpec:
    """Problem builder parameters matching a Section 6.3.3 controller."""

    name: str
    kind: str
    dt: float
    model_path: Path
    reference_path: Path
    w_u: float
    w_hinge: float = 0.0
    w_hinge_terminal: float | None = None
    w_y: float = 0.0
    w_y_terminal: float | None = None


PREDICTORS: dict[str, PredictorSpec] = {
    "spectral_dmdc": PredictorSpec(
        name="spectral_dmdc",
        kind="observable",
        dt=0.06,
        model_path=BENCHMARK_DIR / "models/observable_dmdc.npz",
        reference_path=BENCHMARK_DIR / "references/healthy_psd_baseline_observable.npz",
        w_u=0.1,
        w_hinge=10.0,
        w_hinge_terminal=10.0,
    ),
    "spectral_cnn": PredictorSpec(
        name="spectral_cnn",
        kind="observable",
        dt=0.06,
        model_path=BENCHMARK_DIR / "models/observable_cnn.npz",
        reference_path=BENCHMARK_DIR / "references/healthy_psd_baseline_observable.npz",
        w_u=0.3,
        w_hinge=10.0,
        w_hinge_terminal=50.0,
    ),
    "waveform_mlp": PredictorSpec(
        name="waveform_mlp",
        kind="waveform",
        dt=0.02,
        model_path=BENCHMARK_DIR / "models/waveform_mlp.npz",
        reference_path=BENCHMARK_DIR / "references/healthy_waveform_reference.npz",
        w_u=100.0,
        w_y=1.0,
        w_y_terminal=1.0,
    ),
}


def prepare_snapshots(out_path: Path = SNAPSHOTS_PATH) -> Path:
    """Extract ten primed seizure snapshots across reserved uncontrolled trajectories."""
    wave_hists: list[np.ndarray] = []
    spec_hists: list[np.ndarray] = []
    max_wave = 16
    max_spec = 16
    fs_50 = 50.0
    support = _SPECTRAL_GEOM.sample_support_steps(fs_50)

    def extract_at(y_mea: np.ndarray, t_snap: float) -> tuple[np.ndarray, np.ndarray]:
        y_50 = antialias_filter(np.asarray(y_mea, dtype=np.float64), 10_000.0, 200)[::200]
        frames = compute_log_power_frames(y_50, _SPECTRAL_GEOM, fs=fs_50)
        sample_idx = round(t_snap / 0.02)
        frame_idx = (sample_idx - (support - 1)) // _SPECTRAL_GEOM.n_hop
        if sample_idx < max_wave - 1 or frame_idx < max_spec - 1:
            msg = f"Snapshot time {t_snap} s is too early for priming windows."
            raise ValueError(msg)
        y_w = y_50[sample_idx - max_wave + 1 : sample_idx + 1]
        y_s = frames[frame_idx - max_spec + 1 : frame_idx + 1]
        return y_w, y_s

    for seed, t_snap in zip(SEEDS, SNAPSHOT_TIMES_S, strict=True):
        log_path = UNCONTROLLED_DIR / f"uncontrolled_s{seed}/log.npz"
        with np.load(log_path) as data:
            y_w, y_s = extract_at(data["sensor_0.y_mea"], t_snap)
        wave_hists.append(y_w)
        spec_hists.append(y_s)

    with np.load(UNCONTROLLED_DIR / f"uncontrolled_s{SEEDS[0]}/log.npz") as data:
        warmup_w, warmup_s = extract_at(data["sensor_0.y_mea"], WARMUP_TIME_S)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        seeds=np.asarray(SEEDS, dtype=np.int64),
        times_s=np.asarray(SNAPSHOT_TIMES_S, dtype=np.float64),
        wave_hists=np.stack(wave_hists, axis=0),
        spec_hists=np.stack(spec_hists, axis=0),
        warmup_wave=warmup_w,
        warmup_spec=warmup_s,
        warmup_time_s=np.asarray(WARMUP_TIME_S, dtype=np.float64),
    )
    return out_path


def _build_problem(spec: PredictorSpec, horizon_steps: int) -> NeuroProblem:
    """Instantiate the NeuroProblem for ``spec`` at ``horizon_steps``."""
    ref = HealthyReference.load(spec.reference_path)
    if spec.kind == "observable":
        return build_observable_problem(
            spec.model_path,
            horizon=horizon_steps,
            u_max=2.0,
            w_u=spec.w_u,
            w_u_l1=0.0,
            w_hinge=spec.w_hinge,
            w_hinge_terminal=spec.w_hinge_terminal,
            reference=ref,
            kirchhoff=True,
        )
    return build_waveform_problem(
        spec.model_path,
        horizon=horizon_steps,
        control_block_steps=1,
        u_max=2.0,
        w_y=spec.w_y,
        w_y_terminal=spec.w_y_terminal,
        w_u=spec.w_u,
        w_hinge=0.0,
        w_u_l1=0.0,
        reference=ref,
        kirchhoff=True,
    )


def _require_ar(model: object) -> AutoregressiveModel:
    """Ensure ``model`` is an ``AutoregressiveModel``."""
    if not isinstance(model, AutoregressiveModel):
        msg = f"Expected AutoregressiveModel, got {type(model).__name__}"
        raise TypeError(msg)
    return model


def _state_from_history(model: AutoregressiveModel, wave_hist: np.ndarray, spec_hist: np.ndarray) -> np.ndarray:
    """Prime an ``AutoregressiveModel`` state from unstimulated output history."""
    state = np.asarray(model.initial_state(), dtype=np.float64)
    u_zero = np.zeros(int(model.m), dtype=np.float64)
    hist = spec_hist[-int(model.n_history) :] if hasattr(model, "geometry") else wave_hist[-int(model.n_history) :]
    for y in hist:
        state = np.asarray(model.absorb(state, y, u_zero), dtype=np.float64)
    if not model.is_ready(state):
        msg = "Primed snapshot state is not ready."
        raise RuntimeError(msg)
    return state


def _make_solver(transcription: str) -> Solver:
    """Build the configured Ipopt transcription solver."""
    ipopt_opts = {**IPOPT_DEFAULTS, "max_cpu_time": MAX_CPU_TIME_S}
    if transcription == "single_shooting":
        return SingleShooting(solver=Ipopt(options=ipopt_opts))
    if transcription == "narx":
        return NarxShooting(solver=Ipopt(options=ipopt_opts))
    if transcription == "multiple_shooting":
        return MultipleShooting(solver=Ipopt(options=ipopt_opts))
    msg = f"Unknown transcription: {transcription}"
    raise ValueError(msg)


def _solver_config_dict(transcription: str) -> dict[str, Any]:
    """Build a serializable solver config dict for ``TrajOptMPCController``."""
    ipopt_cfg = {
        "class_path": "trajopt.transcription.ipopt.Ipopt",
        "options": {**IPOPT_DEFAULTS, "max_cpu_time": MAX_CPU_TIME_S},
    }
    if transcription == "single_shooting":
        return {"class_path": "trajopt.transcription.single_shooting.SingleShooting", "solver": ipopt_cfg}
    if transcription == "narx":
        return {"class_path": "neuro.control.narx.NarxShooting", "solver": ipopt_cfg}
    if transcription == "multiple_shooting":
        return {"class_path": "neuro.control.multiple_shooting.MultipleShooting", "solver": ipopt_cfg}
    msg = f"Unknown transcription: {transcription}"
    raise ValueError(msg)


def _cell_dimensions(problem: NeuroProblem, transcription: str) -> tuple[int, int, int]:
    """Return ``(n_vars, n_eq, est_jac_bytes)`` for ``transcription``."""
    model = _require_ar(problem.model)
    H = int(problem.N) - 1
    m = int(model.m)
    n_h = int(model.n)
    n_o = int(model.n_outputs)
    if transcription == "single_shooting":
        n_vars = H * m
        n_eq = H
        est_jac_bytes = n_eq * n_vars * 8
        return n_vars, n_eq, est_jac_bytes
    if transcription == "narx":
        n_vars, n_defects, n_user = narx_dimensions(problem)
        n_y = int(model.n_y)
        n_u = int(model.n_u)
        nnz_approx = H * n_o * (1 + n_y * n_o + n_u * m) + n_user * m
        return n_vars, n_defects + n_user, int(nnz_approx * 8)
    n_vars = (H + 1) * n_h + H * m
    n_eq = (H + 1) * n_h + H
    nnz_approx = n_h + H * (2 * n_h + n_o * n_h + m)
    return n_vars, n_eq, int(nnz_approx * 8)


def _run_closed_loop_slice(spec: PredictorSpec, horizon_steps: int, transcription: str) -> dict[str, Any]:
    """Run a 20-replan closed-loop simulation slice on seed 7100."""
    base = load_config(BENCHMARK_DIR / "base_simulation.yaml")
    cfg = copy.deepcopy(base)
    cfg["t_end"] = CLOSED_LOOP_T_END_S
    cfg["dynamics"]["seed"] = CLOSED_LOOP_SEED

    if spec.kind == "waveform":
        cfg["estimator"] = {
            "class_path": "neuro.filtering.AntiAliasEstimator",
            "dt": 0.0001,
            "downsample": 200,
        }
        problem_cfg: dict[str, Any] = {
            "class_path": "neuro.control.problems.build_waveform_problem",
            "artifact": str(spec.model_path),
            "horizon": horizon_steps,
            "control_block_steps": 1,
            "u_max": 2.0,
            "w_y": spec.w_y,
            "w_y_terminal": spec.w_y_terminal,
            "w_u": spec.w_u,
            "w_hinge": 0.0,
            "w_u_l1": 0.0,
            "reference": str(spec.reference_path),
            "kirchhoff": True,
        }
    else:
        problem_cfg = {
            "class_path": "neuro.control.problems.build_observable_problem",
            "artifact": str(spec.model_path),
            "horizon": horizon_steps,
            "u_max": 2.0,
            "w_u": spec.w_u,
            "w_u_l1": 0.0,
            "w_hinge": spec.w_hinge,
            "w_hinge_terminal": spec.w_hinge_terminal,
            "reference": str(spec.reference_path),
            "kirchhoff": True,
        }

    cfg["controller"] = {
        "class_path": "neuro.control.mpc.TrajOptMPCController",
        "dt": spec.dt,
        "replan_dt": CLOSED_LOOP_REPLAN_DT,
        "control_start_s": CLOSED_LOOP_START_S,
        "solver": _solver_config_dict(transcription),
        "problem": problem_cfg,
    }

    sim = Simulation.from_config(cfg)
    with tempfile.TemporaryDirectory(prefix="tx_bench_", ignore_cleanup_errors=True) as log_dir:
        sim.run(output_dir=log_dir, use_mmap=True)
        if sim.logger is None:
            msg = "Simulation logger missing after run."
            raise RuntimeError(msg)
        warmup = sim.logger.signal("controller", "warmup")[1]
        attempted = sim.logger.signal("controller", "solve_attempted")[1]
        mask = solve_attempt_mask(warmup, attempted)
        solve_times = np.asarray(sim.logger.signal("controller", "solve_time")[1], dtype=np.float64).reshape(-1)[mask]
        ctrl_times = np.asarray(sim.logger.signal("controller", "controller_time")[1], dtype=np.float64).reshape(-1)[
            mask
        ]
        successes = np.asarray(sim.logger.signal("controller", "success")[1], dtype=bool).reshape(-1)[mask]
        costs = np.asarray(sim.logger.signal("controller", "cost")[1], dtype=np.float64).reshape(-1)[mask]
        t_ctrl, u_ctrl = sim.logger.signal("controller", "u")
        t_arr = np.asarray(t_ctrl, dtype=np.float64).reshape(-1)
        u_arr = np.asarray(u_ctrl, dtype=np.float64).reshape(len(t_arr), -1)
        charge = delivered_charge(u_arr, current_hold_durations(t_arr, CLOSED_LOOP_T_END_S))

    return {
        "n_replans": int(np.sum(mask)),
        "success_count": int(np.sum(successes)),
        "solve_times_s": [float(x) for x in solve_times],
        "steady_solve_times_s": [float(x) for x in solve_times[1:]] if len(solve_times) > 1 else [],
        "controller_times_s": [float(x) for x in ctrl_times],
        "costs": [float(x) for x in costs],
        "charge_mas": float(charge),
    }


def _run_open_loop_snapshots(
    mpc: MPC,
    snap_data: dict[str, Any],
    *,
    warmup_ok: bool,
    warmup_wall: float,
) -> tuple[list[dict[str, Any]], str]:
    """Run paired cold-started open-loop solves across the saved seizure snapshots."""
    problem = mpc.problem
    model = _require_ar(problem.model)
    wave_hists = snap_data["wave_hists"]
    spec_hists = snap_data["spec_hists"]
    seeds = snap_data["seeds"]
    times_s = snap_data["times_s"]
    open_rows: list[dict[str, Any]] = []
    status = "ok"

    for idx, (seed, time_s) in enumerate(zip(seeds, times_s, strict=True)):
        x_snap = _state_from_history(model, wave_hists[idx], spec_hists[idx])
        x_jax = jnp.asarray(x_snap, dtype=jnp.float64)
        mpc.measure(x_jax, 0.0)
        mpc._ws = WarmStart.cold(problem, x_jax)  # noqa: SLF001 -- reset cold primal/dual guess for paired open-loop snapshots
        started = time.perf_counter()
        res = mpc.solve()
        wall_s = time.perf_counter() - started
        open_rows.append(
            {
                "snapshot_idx": idx,
                "seed": seed,
                "time_s": time_s,
                "wall_s": float(wall_s),
                "success": bool(res.success),
                "ipopt_status": int(res.status),
                "message": str(res.message),
                "iterations": int(res.iterations),
                "cost": float(res.cost),
                "constraint_violation": float(res.constraint_violation),
            }
        )
        if not res.success and wall_s > _SINGLE_ITER_TIMEOUT_MULTIPLIER * MAX_CPU_TIME_S:
            status = "timeout_single_iter"
            break
        if (
            len(open_rows) == _EARLY_TIMEOUT_SNAPSHOTS
            and not warmup_ok
            and warmup_wall >= MAX_CPU_TIME_S
            and all(not r["success"] and r["ipopt_status"] == _IPOPT_CPU_TIMEOUT_STATUS for r in open_rows)
        ):
            status = "timeout"
            break
    return open_rows, status


def run_cell(predictor: str, horizon_s: float, transcription: str, out_json: Path) -> dict[str, Any]:
    """Run one ``(predictor, horizon_s, transcription)`` benchmark cell."""
    spec = PREDICTORS[predictor]
    horizon_steps = round(horizon_s / spec.dt)
    problem = _build_problem(spec, horizon_steps)
    n_vars, n_eq, est_jac_bytes = _cell_dimensions(problem, transcription)

    result: dict[str, Any] = {
        "predictor": predictor,
        "horizon_s": float(horizon_s),
        "horizon_steps": int(horizon_steps),
        "transcription": transcription,
        "n_vars": int(n_vars),
        "n_eq": int(n_eq),
        "est_jac_gb": float(est_jac_bytes / 1e9),
        "status": "ok",
        "warmup_wall_s": None,
        "open_loop": [],
        "closed_loop": None,
    }

    if est_jac_bytes > MAX_JAC_BYTES:
        result["status"] = "skipped_oom"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(result, indent=2))
        return result

    with np.load(SNAPSHOTS_PATH) as raw_snaps:
        snap_data = {
            "wave_hists": np.asarray(raw_snaps["wave_hists"], dtype=np.float64),
            "spec_hists": np.asarray(raw_snaps["spec_hists"], dtype=np.float64),
            "warmup_wave": np.asarray(raw_snaps["warmup_wave"], dtype=np.float64),
            "warmup_spec": np.asarray(raw_snaps["warmup_spec"], dtype=np.float64),
            "seeds": [int(s) for s in raw_snaps["seeds"]],
            "times_s": [float(t) for t in raw_snaps["times_s"]],
        }

    model = _require_ar(problem.model)
    x_warmup = _state_from_history(model, snap_data["warmup_wave"], snap_data["warmup_spec"])
    mpc = MPC(problem, _make_solver(transcription), x0=jnp.asarray(x_warmup, dtype=jnp.float64))

    t0 = time.perf_counter()
    res_w = mpc.solve()
    warmup_wall = time.perf_counter() - t0
    result["warmup_wall_s"] = float(warmup_wall)
    result["warmup_success"] = bool(res_w.success)
    result["warmup_status"] = int(res_w.status)

    if not res_w.success and warmup_wall > _SINGLE_ITER_TIMEOUT_MULTIPLIER * MAX_CPU_TIME_S:
        result["status"] = "timeout_single_iter"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(result, indent=2))
        return result

    open_rows, status = _run_open_loop_snapshots(
        mpc,
        snap_data,
        warmup_ok=bool(res_w.success),
        warmup_wall=float(warmup_wall),
    )
    result["open_loop"] = open_rows
    result["status"] = status

    ol_times = [r["wall_s"] for r in open_rows]
    ol_succ = [r["success"] for r in open_rows]
    if any(ol_succ) and statistics.median(ol_times) < CLOSED_LOOP_MEDIAN_CAP_S:
        result["closed_loop"] = _run_closed_loop_slice(spec, horizon_steps, transcription)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2))
    return result


def _cell_id(predictor: str, horizon_s: float, transcription: str) -> str:
    return f"{predictor}_h{round(horizon_s * 100):03d}_{transcription}"


def _record_fallback(
    spec: PredictorSpec,
    horizon_s: float,
    transcription: str,
    status: str,
    out_json: Path,
) -> dict[str, Any]:
    """Build and persist a fallback cell entry when execution is skipped, timed out, or fails."""
    h_steps = round(horizon_s / spec.dt)
    prob = _build_problem(spec, h_steps)
    n_vars, n_eq, est_jac_bytes = _cell_dimensions(prob, transcription)
    fallback: dict[str, Any] = {
        "predictor": spec.name,
        "horizon_s": float(horizon_s),
        "horizon_steps": int(h_steps),
        "transcription": transcription,
        "n_vars": int(n_vars),
        "n_eq": int(n_eq),
        "est_jac_gb": float(est_jac_bytes / 1e9),
        "status": status,
        "warmup_wall_s": None,
        "open_loop": [],
        "closed_loop": None,
    }
    out_json.write_text(json.dumps(fallback, indent=2))
    return fallback


def _run_worker_cell(spec: PredictorSpec, horizon_s: float, transcription: str, out_json: Path) -> dict[str, Any]:
    """Execute one benchmark cell in an isolated subprocess with timeout protection."""
    cmd = [
        sys.executable,
        "-m",
        "experiments.transcription_benchmark",
        "--worker",
        "--predictor",
        spec.name,
        "--horizon-s",
        f"{horizon_s:.2f}",
        "--transcription",
        transcription,
        "--out-json",
        str(out_json),
    ]
    logger.info("Running cell %s", out_json.stem)
    try:
        proc = subprocess.run(cmd, check=False, timeout=1800)  # noqa: S603 -- invokes fixed internal module CLI
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        rc = _SUBPROCESS_TIMEOUT_RC
    if rc != 0 or not out_json.exists():
        status = "timeout_subprocess" if rc == _SUBPROCESS_TIMEOUT_RC else f"failed_rc_{rc}"
        return _record_fallback(spec, horizon_s, transcription, status, out_json)
    return json.loads(out_json.read_text())


def run_all_cells() -> list[dict[str, Any]]:
    """Orchestrate all 36 cells in isolated subprocesses."""
    if not SNAPSHOTS_PATH.exists():
        prepare_snapshots(SNAPSHOTS_PATH)
    CELLS_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    infeasible_pairs: set[tuple[str, str]] = set()
    skip_statuses = {"skipped_oom", "timeout_single_iter", "timeout_subprocess"}

    for predictor, spec in PREDICTORS.items():
        for horizon_s in HORIZONS_S:
            for transcription in TRANSCRIPTIONS:
                cid = _cell_id(predictor, horizon_s, transcription)
                out_json = CELLS_DIR / f"{cid}.json"
                if out_json.exists():
                    cell = json.loads(out_json.read_text())
                elif (predictor, transcription) in infeasible_pairs:
                    logger.info("Skipping %s (smaller horizon already OOM/single-iter timeout)", cid)
                    cell = _record_fallback(spec, horizon_s, transcription, "skipped_infeasible", out_json)
                else:
                    cell = _run_worker_cell(spec, horizon_s, transcription, out_json)
                if cell["status"] in skip_statuses or str(cell["status"]).startswith("failed_rc_"):
                    infeasible_pairs.add((predictor, transcription))
                results.append(cell)
    return results


def aggregate_results() -> Path:
    """Write summary CSV and JSON from completed cell JSONs."""
    rows: list[dict[str, Any]] = []
    for predictor in PREDICTORS:
        for horizon_s in HORIZONS_S:
            for transcription in TRANSCRIPTIONS:
                cid = _cell_id(predictor, horizon_s, transcription)
                path = CELLS_DIR / f"{cid}.json"
                if not path.exists():
                    continue
                cell = json.loads(path.read_text())
                ol = cell["open_loop"]
                cl = cell["closed_loop"]
                if ol:
                    ol_times = [r["wall_s"] for r in ol]
                    ol_iters = [r["iterations"] for r in ol]
                    ol_costs = [r["cost"] for r in ol if r["success"]]
                    ol_viols = [r["constraint_violation"] for r in ol if r["success"]]
                    ol_conv = sum(1 for r in ol if r["success"])
                    ol_med = statistics.median(ol_times)
                    ol_iqr = float(np.percentile(ol_times, 75) - np.percentile(ol_times, 25))
                    ol_iter_med = statistics.median(ol_iters)
                    ol_cost_med = statistics.median(ol_costs) if ol_costs else float("nan")
                    ol_viol_max = max(ol_viols) if ol_viols else float("nan")
                else:
                    ol_conv = 0
                    ol_med = float("nan")
                    ol_iqr = float("nan")
                    ol_iter_med = float("nan")
                    ol_cost_med = float("nan")
                    ol_viol_max = float("nan")

                if cl and cl["steady_solve_times_s"]:
                    cl_med = statistics.median(cl["steady_solve_times_s"])
                    cl_succ = f"{cl['success_count']}/{cl['n_replans']}"
                else:
                    cl_med = float("nan")
                    cl_succ = "--"

                rows.append(
                    {
                        "predictor": cell["predictor"],
                        "horizon_s": cell["horizon_s"],
                        "horizon_steps": cell["horizon_steps"],
                        "transcription": cell["transcription"],
                        "n_vars": cell["n_vars"],
                        "n_eq": cell["n_eq"],
                        "est_jac_gb": round(cell["est_jac_gb"], 3),
                        "status": cell["status"],
                        "warmup_wall_s": cell["warmup_wall_s"],
                        "ol_converged": ol_conv,
                        "ol_total": len(ol),
                        "ol_median_s": ol_med,
                        "ol_iqr_s": ol_iqr,
                        "ol_iter_median": ol_iter_med,
                        "ol_cost_median": ol_cost_med,
                        "ol_viol_max": ol_viol_max,
                        "cl_median_s": cl_med,
                        "cl_success": cl_succ,
                    }
                )

    csv_path = BENCHMARK_DIR / "summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_json = BENCHMARK_DIR / "summary.json"
    summary_json.write_text(json.dumps(rows, indent=2))
    return csv_path


def main() -> None:
    """CLI entry point for snapshot preparation, isolated worker, and aggregation."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description="MPC transcription benchmark runner.")
    parser.add_argument("--prepare-snapshots", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--predictor", type=str)
    parser.add_argument("--horizon-s", type=float)
    parser.add_argument("--transcription", type=str)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    if args.prepare_snapshots:
        path = prepare_snapshots()
        logger.info("Wrote snapshots to %s", path)
    elif args.worker:
        if not (args.predictor and args.horizon_s is not None and args.transcription and args.out_json):
            parser.error("--worker requires --predictor, --horizon-s, --transcription, and --out-json")
        run_cell(args.predictor, args.horizon_s, args.transcription, args.out_json)
    elif args.run_all:
        run_all_cells()
        csv_path = aggregate_results()
        logger.info("Wrote summary to %s", csv_path)
    elif args.aggregate:
        csv_path = aggregate_results()
        logger.info("Wrote summary to %s", csv_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
