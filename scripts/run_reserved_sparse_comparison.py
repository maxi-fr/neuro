from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml
from summarize_sparse_stimulation import trace_metrics

from experiments import comparison

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts" / "sparse_stimulation_102"
EXPERIMENT = ROOT / "artifacts" / "sparse_stimulation_reserved"
SEEDS = list(range(7100, 7110))
METRICS = (
    "seizure_burden",
    "n_seizing_final",
    "delivered_charge",
    "intentional_off_s",
    "fallback_off_s",
    "fallback_holds",
    "bouts",
    "solve_failures",
    "replan_time_mean_s",
)


def sha256(path: Path) -> str:
    """Hash one experiment input."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(weight: float) -> Path:
    """Snapshot the selected configs and inputs, retaining an existing experiment on resume."""
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    manifest_path = EXPERIMENT / "comparison_manifest.yaml"
    if manifest_path.exists():
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        saved_weight = manifest["arms"]["combined_candidates"]["patch"]["controller"]["w_active"]
        if saved_weight != weight:
            msg = f"Existing experiment uses activation weight {saved_weight}, not {weight}."
            raise ValueError(msg)
        return manifest_path
    for name in ("base_simulation.yaml", "model.npz", "healthy_reference.npz"):
        shutil.copy2(SOURCE / name, EXPERIMENT / name)
    arms = {}
    for name in ("quadratic_osqp", "l1_osqp", "combined_candidates"):
        config_path = EXPERIMENT / name / "config.yaml"
        config_path.parent.mkdir(exist_ok=True)
        source_name = "activation_candidates" if name == "combined_candidates" else name
        config = yaml.safe_load((SOURCE / source_name / "config.yaml").read_text(encoding="utf-8"))
        problem = config["controller"]["problem"]
        problem["artifact"] = (EXPERIMENT / "model.npz").relative_to(ROOT).as_posix()
        problem["reference"] = (EXPERIMENT / "healthy_reference.npz").relative_to(ROOT).as_posix()
        if name in {"l1_osqp", "combined_candidates"}:
            problem["w_u_l1"] = 0.60
            problem["l1_mode"] = "epigraph"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        spec = {"config": config_path.relative_to(ROOT).as_posix()}
        if name == "combined_candidates":
            spec["patch"] = {"controller": {"w_active": weight}}
        arms[name] = spec
    manifest = {
        "base": (EXPERIMENT / "base_simulation.yaml").relative_to(ROOT).as_posix(),
        "seeds": SEEDS,
        "t_end": 12.0,
        "arms": arms,
    }
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return manifest_path


def record_inputs(manifest_path: Path) -> None:
    """Record input hashes, code revision, local changes, and solver versions."""
    files = [
        manifest_path,
        EXPERIMENT / "base_simulation.yaml",
        EXPERIMENT / "model.npz",
        EXPERIMENT / "healthy_reference.npz",
        *(EXPERIMENT / name / "config.yaml" for name in ("quadratic_osqp", "l1_osqp", "combined_candidates")),
        ROOT / "data" / "roast_field_projection_3d.npz",
        ROOT / "uv.lock",
        *sorted((ROOT / "src").rglob("*.py")),
        ROOT / "scripts" / "run_comparison.py",
        ROOT / "scripts" / "summarize_sparse_stimulation.py",
        Path(__file__).resolve(),
    ]
    hashes = {path.relative_to(ROOT).as_posix(): sha256(path) for path in files}
    path = EXPERIMENT / "input_hashes.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != hashes:
            msg = "Experiment inputs or source files changed since the previous run."
            raise ValueError(msg)
        return
    path.write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S607 -- git is available on PATH
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    patch = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--", "src", "scripts"],  # noqa: S607 -- git is available on PATH
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    (EXPERIMENT / "source.patch").write_bytes(patch)
    shutil.copy2(Path(__file__).resolve(), EXPERIMENT / "run_reserved_sparse_comparison.py")
    shutil.copy2(ROOT / "uv.lock", EXPERIMENT / "uv.lock")
    metadata = {
        "revision": revision,
        "python": sys.version,
        "packages": {name: version(name) for name in ("jax", "osqp", "casadi", "cyipopt", "trajopt")},
        "command": subprocess.list2cmdline(sys.argv),
    }
    (EXPERIMENT / "environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def summarize(batch: Path) -> None:
    """Write sparse outcomes and paired differences for the completed reserved runs."""
    rows = comparison.read_rows(batch / "rows.csv")
    expected = {(arm, seed) for arm in ("quadratic_osqp", "l1_osqp", "combined_candidates") for seed in SEEDS}
    successful = {(row["arm"], row["seed"]) for row in rows if not row["error"]}
    if successful != expected or len(rows) != len(expected):
        failures = [f"{row['run']}: {row['error']}" for row in rows if row["error"]]
        msg = f"Incomplete comparison. Retry the command to resume. Failures: {failures}"
        raise RuntimeError(msg)
    for row in rows:
        row.update(trace_metrics(batch / "runs" / row["run"]))
    comparison.write_rows(rows, batch / "sparse_rows.csv")
    comparison.write_rows(comparison.summarize(rows), batch / "sparse_summary.csv")
    by_seed = {(row["arm"], row["seed"]): row for row in rows}
    paired = []
    for seed in SEEDS:
        baseline = by_seed["l1_osqp", seed]
        combined = by_seed["combined_candidates", seed]
        entry = {"seed": seed}
        entry.update({metric: combined[metric] - baseline[metric] for metric in METRICS})
        paired.append(entry)
    comparison.write_rows(paired, batch / "paired_combined_minus_l1.csv")
    means = {metric: float(np.mean([row[metric] for row in paired])) for metric in METRICS}
    (batch / "paired_means.json").write_text(json.dumps(means, indent=2), encoding="utf-8")
    print(f"Combined minus L1 mean intentional off time: {means['intentional_off_s']:.3f} s")
    report = [
        "# Reserved-seed sparse stimulation comparison",
        "",
        "## Motivation",
        "",
        "Compare exact L1 and combined L1 plus grouped-candidate activation on the reserved Plant seeds.",
        "",
        "## Experimental Setup",
        "",
        "Seeds 7100-7109, 12 s simulations, 17 holds at 60 ms, L1 weight 0.60, and two workers.",
        "Resolved configurations, input hashes, environment and per-run traces are retained in this experiment.",
        "",
        "## Results",
        "",
        f"Combined minus L1 mean intentional off time: {means['intentional_off_s']:.3f} s.",
        "See the paired CSV and sparse summary for per-seed outcomes, fallback counts and attempted-replan timing.",
        "",
        "## Actionable Recommendation / Decision",
        "",
        "Interpret the paired off-time difference alongside seizure burden, final seizing-region counts and charge.",
        "",
    ]
    (EXPERIMENT / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Results: {batch}")


def main() -> None:
    """Run the paired reserved-seed comparison with two workers and resume completed simulations."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "--activation-weight", type=float, required=True, help="Selected grouped-candidate activation weight."
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and list the 30 simulations.")
    args = parser.parse_args()
    if not np.isfinite(args.activation_weight) or args.activation_weight <= 0:
        parser.error("--activation-weight must be finite and positive")
    os.chdir(ROOT)
    manifest = prepare(args.activation_weight)
    record_inputs(manifest)
    env = os.environ.copy()
    env.update(
        JAX_ENABLE_X64="true", JAX_PLATFORMS="cpu", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1"
    )
    batch = EXPERIMENT / "reserved"
    command = [
        sys.executable,
        "scripts/run_comparison.py",
        str(manifest),
        "--output-dir",
        str(batch),
        "--workers",
        "2",
        "--resume",
    ]
    if args.dry_run:
        command.append("--dry-run")
    subprocess.run(command, cwd=ROOT, env=env, check=True)  # noqa: S603 -- command and manifest are constructed locally
    if not args.dry_run:
        summarize(batch)


if __name__ == "__main__":
    main()
