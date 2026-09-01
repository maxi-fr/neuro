from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import ModuleType

_ROOT = Path(__file__).resolve().parent


def _load_run_module() -> ModuleType:
    """Import ``run_cost_comparison`` by path so its ``summarize`` is reused rather than copied."""
    spec = importlib.util.spec_from_file_location("run_cost_comparison", _ROOT / "run_cost_comparison.py")
    if spec is None or spec.loader is None:
        msg = "cannot load run_cost_comparison.py"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _coerce(row: dict[str, str]) -> dict[str, Any]:
    """Restore the numeric types the CSV round-trip flattened to strings."""
    out: dict[str, Any] = {}
    for key, value in row.items():
        if key in {"run", "arm", "error"}:
            out[key] = value
        elif key in {"horizon", "seed"}:
            out[key] = int(value)
        else:
            out[key] = float(value) if value else float("nan")
    return out


def merge_rows(roots: list[Path], seeds: set[int]) -> list[dict[str, Any]]:
    """Collect every successful run from the split grid directories, deduplicated by run name.

    The grid was run in several passes after a memory failure, so the same run may appear in more
    than one ``rows.csv``; the runs are deterministic per seed, so any copy will do.
    """
    seen: dict[str, dict[str, Any]] = {}
    for root in roots:
        for path in sorted(root.glob("*/rows.csv")):
            for row in csv.DictReader(path.open(encoding="utf-8")):
                if row["error"] or (seeds and int(row["seed"]) not in seeds):
                    continue
                seen[row["run"]] = _coerce(row)
    return sorted(seen.values(), key=lambda row: (row["arm"], row["horizon"], row["seed"]))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the merge."""
    parser = argparse.ArgumentParser(description="Merge the split Cost-comparison passes into one table.")
    parser.add_argument("--root", type=Path, default=Path("results/cost_comparison"))
    parser.add_argument("--seeds", type=int, nargs="*", default=[7, 17, 31, 69, 90, 123, 555])
    parser.add_argument("--output-dir", type=Path, default=Path("results/cost_comparison/merged"))
    return parser.parse_args()


def main() -> None:
    """Merge every pass's rows, then write the pooled per-run and per-cell tables."""
    args = parse_args()
    run_module = _load_run_module()
    rows = merge_rows([args.root], set(args.seeds))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_module._write_rows(rows, args.output_dir / "rows.csv")  # noqa: SLF001 -- one writer, shared with the grid
    summary = run_module.summarize(rows)
    run_module._write_rows(summary, args.output_dir / "summary.csv")  # noqa: SLF001 -- as above
    print(f"{len(rows)} runs -> {len(summary)} cells in {args.output_dir}")
    for entry in summary:
        print(f"  {entry['arm']:20s} h={entry['horizon']:<3d} n={entry['n_seeds']}")


if __name__ == "__main__":
    main()
