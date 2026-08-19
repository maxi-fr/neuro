from __future__ import annotations

import argparse
import time
from pathlib import Path

from neuro.ensembles import (
    PLANT_CONFIG,
    EnsembleConfig,
    PlantPair,
    all_arms,
    build_plants,
    generate_iter,
)
from neuro.seizure import SPREAD_WINDOW_S

# Generates the ensemble that docs/predictability_controllability_experiment.md scores: 16 healthy
# and 16 seizure parents, each snapshotted at its branch points and branched into 8 children run
# under per-branch stimulation arms on a shared seed. Nothing is scored here -- the notebooks do that.
_DESCRIPTION = "Generate the predictability/controllability ensemble."


def simulated_seconds(cfg: EnsembleConfig) -> float:
    """Total simulated time the run will integrate, parents and children together."""
    total = 0.0
    for plant in ("healthy", "seizure"):
        branches = cfg.branches_of(plant)
        if not branches:
            continue
        parent_s = max(b.t_branch for b in branches)
        child_s = sum(len(b.arms) * cfg.n_children * cfg.rollout_s for b in branches)
        total += cfg.n_parents * (parent_s + child_s)
    return total


def stored_bytes(cfg: EnsembleConfig, plants: PlantPair) -> int:
    """Bytes the stores will occupy: full-rate EEG plus decimated region LFP."""
    n_pairs = sum(len(b.arms) for b in cfg.branches)
    n_channels, n_nodes = plants.leadfield.shape
    eeg = cfg.n_rollouts * n_channels * round(cfg.rollout_s / plants.seizure.dt) * 8
    region = cfg.n_rollouts * n_nodes * round((SPREAD_WINDOW_S + cfg.rollout_s) * cfg.region_fs) * 4
    return n_pairs * (eeg + region)


def parse_args() -> argparse.Namespace:
    """Parse the generation CLI arguments."""
    parser = argparse.ArgumentParser(description=_DESCRIPTION)
    parser.add_argument("--out", type=Path, required=True, help="Output directory for the stores and manifest.")
    parser.add_argument("--config", type=Path, default=PLANT_CONFIG, help="Plant config to take the dynamics from.")
    parser.add_argument("--n-parents", type=int, default=16, help="Distinct plant seeds per plant kind.")
    parser.add_argument("--n-children", type=int, default=8, help="Noise realisations branched off each snapshot.")
    parser.add_argument("--rollout", type=float, default=3.0, help="Child rollout length in seconds.")
    parser.add_argument("--region-fs", type=float, default=1000.0, help="Rate the stored region LFP is decimated to.")
    parser.add_argument("--dry-run", action="store_true", help="Print the cost and storage plan, then stop.")
    return parser.parse_args()


def main() -> None:
    """Generate the ensemble, reporting progress and the measured cost as it goes."""
    args = parse_args()
    cfg = EnsembleConfig(
        n_parents=args.n_parents,
        n_children=args.n_children,
        rollout_s=args.rollout,
        region_fs=args.region_fs,
    )

    plants = build_plants(args.config)
    sim_s = simulated_seconds(cfg)
    n_parent_runs = cfg.n_parents * len({b.plant for b in cfg.branches})
    n_rollouts_total = sum(len(b.arms) * cfg.n_rollouts for b in cfg.branches)

    print(f"plant      : {args.config}")
    print(f"branches   : {', '.join(f'{b.name}@{b.t_branch:g}s' for b in cfg.branches)}")
    print(f"arms       : {', '.join(f'{a.name}={list(a.current)}' for a in all_arms(cfg.branches))}")
    print(f"rollouts   : {n_rollouts_total}")
    print(f"simulated  : {sim_s:.0f} s over {n_parent_runs} parent runs")
    print(f"storage    : {stored_bytes(cfg, plants) / 1e9:.1f} GB into {args.out}")
    if args.dry_run:
        return

    t0 = time.perf_counter()
    for done, _ in enumerate(generate_iter(args.out, cfg, plants), start=1):
        elapsed = time.perf_counter() - t0
        # The final yield follows the last parent, so it reports 100% rather than overshooting.
        frac = min(done / n_parent_runs, 1.0)
        eta = elapsed / frac - elapsed
        print(f"parent {min(done, n_parent_runs)}/{n_parent_runs} | {elapsed:.0f} s elapsed | {eta:.0f} s left")

    wall = time.perf_counter() - t0
    print(f"\ndone in {wall:.0f} s ({wall / sim_s:.1f}x realtime), manifest at {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
