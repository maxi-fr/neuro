"""Pool several Predictor-comparison passes over disjoint seeds into one table."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_predictor_comparison import _write_rows, summarize

_TEXT_FIELDS = {"run", "arm", "error"}


def _coerce(row: dict[str, str]) -> dict[str, Any]:
    """Restore the numeric types the CSV round-trip flattened to strings."""
    out: dict[str, Any] = {}
    for key, value in row.items():
        if key in _TEXT_FIELDS:
            out[key] = value
        elif key == "seed":
            out[key] = int(value)
        else:
            out[key] = float(value) if value else float("nan")
    return out


def merge_rows(roots: list[Path]) -> list[dict[str, Any]]:
    """Collect every run from the passes, keyed by run name so a repeated seed is not counted twice."""
    seen: dict[str, dict[str, Any]] = {}
    for root in roots:
        for row in csv.DictReader((root / "rows.csv").open(encoding="utf-8")):
            seen[row["run"]] = _coerce(row)
    return sorted(seen.values(), key=lambda row: (row["arm"], row["seed"]))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the merge."""
    parser = argparse.ArgumentParser(description="Pool Predictor-comparison passes over disjoint seeds.")
    parser.add_argument("roots", type=Path, nargs="+", help="Pass directories, each holding a rows.csv.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Pool the passes, then write the combined per-run and per-arm tables."""
    args = parse_args()
    rows = merge_rows(args.roots)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(rows, args.output_dir / "rows.csv")
    summary = summarize(rows)
    _write_rows(summary, args.output_dir / "summary.csv")
    print(f"{len(rows)} runs -> {len(summary)} arms in {args.output_dir}")
    for entry in summary:
        print(f"  {entry['arm']:16s} n={entry['n_seeds']:<3d} burden={entry['seizure_burden']:.4f}")


if __name__ == "__main__":
    main()
