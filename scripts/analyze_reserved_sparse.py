from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "artifacts" / "sparse_stimulation_reserved" / "reserved"
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


def main() -> None:
    """Compute paired bootstrap intervals for combined minus exact L1 outcomes."""
    with (BATCH / "paired_combined_minus_l1.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected_seeds = set(range(7100, 7110))
    if {int(row["seed"]) for row in rows} != expected_seeds or len(rows) != len(expected_seeds):
        msg = "Paired comparison must contain each of the ten reserved seeds exactly once."
        raise ValueError(msg)
    sample_indices = np.random.default_rng(42).integers(0, len(rows), size=(10000, len(rows)))
    intervals = []
    for metric in METRICS:
        differences = np.asarray([float(row[metric]) for row in rows])
        if not np.isfinite(differences).all():
            msg = f"Non-finite paired differences for {metric}."
            raise ValueError(msg)
        low, high = np.percentile(differences[sample_indices].mean(axis=1), [2.5, 97.5])
        intervals.append({"metric": metric, "mean_difference": differences.mean(), "ci_low": low, "ci_high": high})
    with (BATCH / "paired_intervals.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["metric", "mean_difference", "ci_low", "ci_high"])
        writer.writeheader()
        writer.writerows(intervals)
    print(f"Wrote paired 95% percentile bootstrap intervals to {BATCH / 'paired_intervals.csv'}")


if __name__ == "__main__":
    main()
