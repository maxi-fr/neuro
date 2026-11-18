from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import yaml

from neuro.plant.seizure import spread_profile_from_lfp

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "artifacts" / "sparse_stimulation_reserved"
BATCH = EXPERIMENT / "reserved"
LIMIT_TOL = 1e-6
EXPECTED_RUNS = 30
EXPECTED_REPLANS = 190
GATE_THRESHOLD = 0.5


def main() -> None:  # noqa: PLR0915 -- validation and atomic packaging follow the same run loop
    """Validate and compact reserved logs while retaining plans, statuses and seizure profiles."""
    with (BATCH / "rows.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != EXPECTED_RUNS or any(row["error"] for row in rows):
        msg = "All 30 reserved runs must succeed before packaging."
        raise ValueError(msg)
    audit = []
    for row in rows:
        run = BATCH / "runs" / row["run"]
        log_path = run / "log.npz"
        config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
        with np.load(log_path, allow_pickle=False) as log:
            attempted = log["controller.solve_attempted"].astype(bool)
            accepted = attempted & log["controller.success"].astype(bool)
            current = log["controller.u"]
            planned = log["controller.planned_u"][accepted]
            if attempted.sum() != EXPECTED_REPLANS or planned.shape[1:] != (17, 3) or not np.isfinite(planned).all():
                msg = f"Unexpected replan or plan shape in {run.name}."
                raise ValueError(msg)
            if np.max(np.abs(current)) > 2 + LIMIT_TOL or np.max(np.abs(planned)) > 2 + LIMIT_TOL:
                msg = f"Current bound exceeded in {run.name}."
                raise ValueError(msg)
            if np.max(np.abs(current.sum(axis=1))) > LIMIT_TOL or np.max(np.abs(planned.sum(axis=2))) > LIMIT_TOL:
                msg = f"Kirchhoff balance failed in {run.name}."
                raise ValueError(msg)
            np.testing.assert_allclose(current[accepted], planned[:, 0], atol=LIMIT_TOL, rtol=0)
            if row["arm"] == "combined_candidates":
                gates = log["controller.planned_active"][accepted]
                if np.max(np.abs(planned[gates < GATE_THRESHOLD]), initial=0.0) > LIMIT_TOL:
                    msg = f"An inactive hold carries current in {run.name}."
                    raise ValueError(msg)
            audit.append(
                {
                    "run": row["run"],
                    "replans": int(attempted.sum()),
                    "accepted": int(accepted.sum()),
                    "fallback_holds": int(log["controller.fallback"].sum()),
                    "max_current_ma": float(np.max(np.abs(current))),
                    "max_kirchhoff_ma": float(np.max(np.abs(current.sum(axis=1)))),
                }
            )
            if "spread_ptp" in log.files:
                continue
            profile = spread_profile_from_lfp(log["dynamics.lfp"].T, config["dynamics"]["dt"], threshold=5.0)
            np.testing.assert_allclose(profile.burden(), float(row["seizure_burden"]), atol=1e-12, rtol=0)
            np.testing.assert_equal(profile.n_seizing()[-1], float(row["n_seizing_final"]))
            estimator_times = log["estimator.t"]
            controller_times = log["controller.t"]
            indices = np.searchsorted(estimator_times, controller_times - 1e-10)
            np.testing.assert_allclose(estimator_times[indices], controller_times, atol=1e-9, rtol=0)
            compact = {key: log[key] for key in log.files if key.startswith("controller.")}
            compact.update(
                {
                    "estimator.t": estimator_times[indices],
                    "estimator.x_hat": log["estimator.x_hat"][indices],
                    "spread_t": profile.times,
                    "spread_ptp": profile.ptp,
                    "spread_onsets": profile.onsets,
                }
            )
        temporary = run / "log.compact.npz"
        np.savez_compressed(temporary, **compact)
        with np.load(temporary, allow_pickle=False) as result:
            np.testing.assert_equal(set(result.files), set(compact))
            np.testing.assert_array_equal(result["controller.u"], current)
        temporary.replace(log_path)
        print(f"Packaged {run.name}", flush=True)
    with (BATCH / "plan_audit.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)


if __name__ == "__main__":
    main()
