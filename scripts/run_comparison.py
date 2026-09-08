# Run one closed-loop comparison manifest: every arm on every seed, scored into one table.
#
# The comparison itself lives in the manifest, not here -- configs/comparison/*.yaml name the
# arms and the seeds, so a Cost comparison and a Predictor comparison are the same code.

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

# One simulation spends its time in small BLAS calls that do not parallelise: measured at 12 sim s,
# a run took 59.5 s/sim s while holding seven cores and 57.3 s/sim s pinned to one. The seven cores
# bought nothing. Pinning here, before torch is imported anywhere below, turns each worker into one
# core so the Pool actually scales.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import multiprocessing

import torch
import yaml

from neuro import comparison

if TYPE_CHECKING:
    from neuro.comparison import Cell, ComparisonManifest

_THRESHOLD_MV = 0.0


def _init_worker(threshold: float) -> None:
    """Pin each worker to one thread and hand it the Seizure Threshhold the manifest set."""
    # The env vars set at import get torch most of the way, but a probe still measured 2.6 cores
    # per run. Saying it directly leaves nothing for the workers to fight over.
    torch.set_num_threads(1)
    global _THRESHOLD_MV  # noqa: PLW0603 -- a Pool initializer has no other way to seed its workers
    _THRESHOLD_MV = threshold


def _score(cell: Cell) -> dict[str, Any]:
    """Score one Cell at the worker's configured Seizure Threshhold."""
    return comparison.score_cell(cell, threshold=_THRESHOLD_MV)


def _git_sha() -> str:
    """Return the working tree's commit, so a table can be traced back to the code that made it."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607 -- git is on PATH by assumption
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"
    return sha


def _write_provenance(out_dir: Path, manifest: ComparisonManifest) -> None:
    """Record the manifest, its hash, the commit, and every arm's fully resolved config."""
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "manifest": manifest.model_dump(mode="json"),
                "manifest_hash": comparison.manifest_hash(manifest),
                "git_sha": _git_sha(),
                "created": datetime.now(UTC).astimezone().isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    resolved = out_dir / "configs"
    resolved.mkdir(exist_ok=True)
    for arm in manifest.arms:
        config = comparison.arm_config(manifest, arm)
        (resolved / f"{arm}.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _resume_rows(out_dir: Path, manifest: ComparisonManifest) -> list[dict[str, Any]]:
    """Load the rows an interrupted grid already finished, refusing to mix two manifests.

    Raises
    ------
    ValueError
        If the output directory was written by a different manifest.
    """
    stamp = out_dir / "manifest.json"
    if not stamp.exists():
        return []
    recorded = json.loads(stamp.read_text(encoding="utf-8")).get("manifest_hash")
    if recorded != comparison.manifest_hash(manifest):
        msg = f"{out_dir} was written by manifest {recorded}, not {comparison.manifest_hash(manifest)}."
        raise ValueError(msg)
    return [row for row in comparison.read_rows(out_dir / "rows.csv") if not row["error"]]


def _run_grid(
    pending: list[Cell],
    rows: list[dict[str, Any]],
    *,
    out_dir: Path,
    threshold: float,
    workers: int,
) -> None:
    """Score the pending cells into ``rows``, flushing rows.csv after each one.

    The grid takes hours, so a crash part-way through must not cost the runs that already
    succeeded: every finished row is on disk before the next run starts.
    """
    total = len(rows) + len(pending)

    def record(row: dict[str, Any]) -> None:
        rows.append(row)
        comparison.write_rows(rows, out_dir / "rows.csv")
        note = row["error"] or f"burden={row['seizure_burden']:.3f}"
        print(f"  [{len(rows)}/{total}] {row['run']}: {note}", flush=True)

    if workers > 1 and pending:
        with multiprocessing.Pool(
            processes=min(workers, len(pending)), initializer=_init_worker, initargs=(threshold,)
        ) as pool:
            for row in pool.imap_unordered(_score, pending):
                record(row)
        return
    _init_worker(threshold)
    for cell in pending:
        record(_score(cell))


def _subset[K, V](available: dict[K, V], wanted: list[K], what: str) -> dict[K, V]:
    """Narrow a manifest field to ``wanted``, refusing anything the manifest does not name.

    A run must never reach for an arm or a seed outside its manifest: the whole point of running
    every comparison off one manifest is that two runs of it are comparable.

    Raises
    ------
    ValueError
        If ``wanted`` names something the manifest does not have.
    """
    unknown = sorted(str(key) for key in wanted if key not in available)
    if unknown:
        msg = f"Manifest has no {what} {unknown}; it defines {sorted(str(key) for key in available)}."
        raise ValueError(msg)
    return {key: available[key] for key in wanted}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for a comparison run."""
    parser = argparse.ArgumentParser(description="Run one closed-loop comparison manifest.")
    parser.add_argument("manifest", type=Path, help="Comparison manifest YAML.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for rows.csv and summary.csv.")
    parser.add_argument("--arms", nargs="*", default=None, help="Run only these arms of the manifest.")
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="Run only these seeds of the manifest.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel simulation processes.")
    parser.add_argument("--resume", action="store_true", help="Skip the (arm, seed) cells --output-dir already has.")
    parser.add_argument("--dry-run", action="store_true", help="Validate every cell and list the grid, run nothing.")
    return parser.parse_args()


def main() -> None:
    """Expand the manifest, run its grid, and write the per-run and per-arm metric tables."""
    args = parse_args()
    manifest = comparison.load_manifest(args.manifest)
    if args.arms is not None:
        manifest = manifest.model_copy(update={"arms": _subset(manifest.arms, args.arms, "arms")})
    if args.seeds is not None:
        manifest = manifest.model_copy(
            update={"seeds": list(_subset(dict.fromkeys(manifest.seeds), args.seeds, "seeds"))}
        )

    grid = comparison.expand_grid(manifest)
    comparison.validate_grid(grid)
    print(f"{len(grid)} runs over {len(manifest.arms)} arms and {len(manifest.seeds)} seeds")
    if args.dry_run:
        for cell in grid:
            print(f"  {cell.run}")
        return

    stamp = datetime.now(UTC).astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = args.output_dir or Path("results/comparison") / args.manifest.stem / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _resume_rows(out_dir, manifest) if args.resume else []
    done = {(row["arm"], row["seed"]) for row in rows}
    pending = [cell for cell in grid if (cell.arm, cell.seed) not in done]
    _write_provenance(out_dir, manifest)
    if done:
        print(f"resuming: {len(done)} runs already scored, {len(pending)} to go")

    _run_grid(pending, rows, out_dir=out_dir, threshold=manifest.seizure_ptp_mv, workers=args.workers)

    summary = comparison.summarize(rows)
    comparison.write_rows(summary, out_dir / "summary.csv")
    print(f"Wrote {out_dir}/rows.csv ({len(rows)} runs) and summary.csv ({len(summary)} arms)")
    for entry in summary:
        print(f"  {entry['arm']:>20s} burden={entry['seizure_burden']:.3f} +- {entry['seizure_burden_sd']:.3f}")


if __name__ == "__main__":
    main()
