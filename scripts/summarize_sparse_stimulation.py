from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from experiments import comparison
from experiments.closed_loop_metrics import current_hold_durations

_OFF_CURRENT_MA = 1e-3
_MIN_HOLD_S = 1e-10


def trace_metrics(run: Path) -> dict[str, float]:
    """Measure applied holds and full replan latency separately from the solver interval."""
    config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    with np.load(run / "log.npz", allow_pickle=False) as log:
        times = log["controller.t"].ravel()
        attempts = log["controller.solve_attempted"].ravel().astype(bool)
        if not attempts.any():
            msg = f"No replans were attempted in {run}"
            raise ValueError(msg)
        replan_times = log["controller.replan_time"].ravel()[attempts]
        if not np.isfinite(replan_times).all():
            msg = f"A replan is missing its elapsed time in {run}"
            raise ValueError(msg)
        release = float(times[attempts][0])
        holds = current_hold_durations(np.maximum(times, release), config["t_end"])
        applied = holds > _MIN_HOLD_S
        off = np.max(np.abs(log["controller.u"]), axis=1) <= _OFF_CURRENT_MA
        fallback = log["controller.fallback"].ravel().astype(bool)
        active = applied & ~off
        status = log["controller.status"].ravel()[attempts].astype(str)
        return {
            "first_replan_s": release,
            "replan_time_mean_s": float(np.mean(replan_times)),
            "intentional_off_s": float(holds[applied & off & ~fallback].sum()),
            "fallback_off_s": float(holds[applied & off & fallback].sum()),
            "fallback_holds": float(np.count_nonzero(applied & fallback)),
            "bouts": float(np.count_nonzero(active & ~np.r_[False, active[:-1]])),
            "accepted_limits": float(np.count_nonzero(status == "FEASIBLE_AT_LIMIT")),
        }


def main() -> None:
    """Write per-run and per-arm sparse stimulation metrics beside the comparison results."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("batch", type=Path)
    batch = parser.parse_args().batch
    rows = comparison.read_rows(batch / "rows.csv")
    for row in rows:
        if not row["error"]:
            row.update(trace_metrics(batch / "runs" / row["run"]))
    comparison.write_rows(rows, batch / "sparse_rows.csv")
    comparison.write_rows(comparison.summarize(rows), batch / "sparse_summary.csv")


if __name__ == "__main__":
    main()
