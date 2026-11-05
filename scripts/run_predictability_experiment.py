from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from experiments.observable_predictability.ensemble import (
    BRANCHES,
    DEFAULT_EVALUATION_PROTOCOL,
    DEFAULT_PRE_BRANCH_S,
    SPLITS,
    EnsembleConfig,
    PlantPair,
    all_arms,
    build_plants,
    generate_iter,
    load_or_score_ensemble,
)
from neuro.plant.seizure import A_HEALTHY, SPREAD_WINDOW_S, build_seizure_a_gains

if TYPE_CHECKING:
    from collections.abc import Sequence

_DESCRIPTION = "Generate healthy calibration and zero-current seizure evaluation replays."


def simulated_seconds(cfg: EnsembleConfig) -> float:
    """Compute total simulated time in seconds across all splits, parents, and children."""
    total = 0.0
    for split in SPLITS:
        n_rep = cfg.split_n_replicates(split)
        for plant in ("healthy", "seizure"):
            branches = cfg.branches_for_split(split, plant)
            seeds = cfg.split_parent_seeds(split, plant)
            if not branches or not seeds:
                continue
            parent_s = max(b.t_branch for b in branches)
            child_s = sum(len(b.arms) * n_rep * cfg.rollout_s for b in branches)
            total += len(seeds) * (parent_s + child_s)
    return total


def stored_bytes(cfg: EnsembleConfig, plants: PlantPair) -> int:
    """Compute bytes occupied by the configured EEG and regional LFP stores across all splits."""
    dt = plants.seizure.dt
    n_channels, n_nodes = plants.leadfield.shape
    eeg_factor = round(1 / (dt * DEFAULT_EVALUATION_PROTOCOL.target_fs)) if cfg.store_conditioned_eeg else 1
    n_eeg = round(cfg.rollout_s / dt) // eeg_factor
    n_region = round((SPREAD_WINDOW_S + cfg.rollout_s) * cfg.region_fs)

    total = 0
    for split in SPLITS:
        n_rep = cfg.split_n_replicates(split)
        for plant in ("healthy", "seizure"):
            branches = cfg.branches_for_split(split, plant)
            seeds = cfg.split_parent_seeds(split, plant)
            if not branches or not seeds:
                continue
            n_parents_split = len(seeds)
            n_rollouts_split = n_parents_split * n_rep
            for branch in branches:
                hist_s = cfg.branch_history_s(branch)
                hist_eeg_samples = (round(hist_s / dt) + eeg_factor - 1) // eeg_factor
                hist_eeg_bytes = n_parents_split * n_channels * hist_eeg_samples * 8
                hist_reg_bytes = n_parents_split * n_nodes * round(hist_s * cfg.region_fs) * 4
                cont_eeg_bytes = n_rollouts_split * n_channels * n_eeg * 8
                cont_reg_bytes = n_rollouts_split * n_nodes * n_region * 4
                total += hist_eeg_bytes + hist_reg_bytes + len(branch.arms) * (cont_eeg_bytes + cont_reg_bytes)
    return total


def verify_plants(plants: PlantPair) -> None:
    """Verify healthy and seizure Plant parameters, Leadfield projection, and montage."""
    expected_seizure_a = build_seizure_a_gains(plants.connectome)
    if not np.allclose(plants.healthy.net_params.A, A_HEALTHY):
        msg = "Healthy Plant A gains do not match A_HEALTHY."
        raise ValueError(msg)
    if not np.allclose(plants.seizure.net_params.A, expected_seizure_a):
        msg = "Seizure Plant A gains do not match build_seizure_a_gains."
        raise ValueError(msg)
    n_nodes = len(plants.connectome.region_labels)
    if plants.leadfield.shape != (len(plants.channel_labels), n_nodes):
        msg = f"Leadfield shape {plants.leadfield.shape} does not match ({len(plants.channel_labels)}, {n_nodes})."
        raise ValueError(msg)
    if not np.all(np.isfinite(plants.leadfield)):
        msg = "Leadfield contains non-finite entries."
        raise ValueError(msg)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for ensemble dry-run, generation, and scoring."""
    parser = argparse.ArgumentParser(description=_DESCRIPTION)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/predictability_ensemble"),
        help="Output directory for stores and manifest.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional simulation config YAML; defaults to repo-local default_plant_params().",
    )
    parser.add_argument("--n-parents", type=int, default=16, help="Seizure evaluation parent seeds.")
    parser.add_argument("--n-reference", type=int, default=2, help="Healthy reference parent seeds.")
    parser.add_argument("--n-dev", type=int, default=2, help="Healthy development parent seeds.")
    parser.add_argument("--n-children", type=int, default=8, help="Noise replicates branched off each snapshot.")
    parser.add_argument("--rollout", type=float, default=3.0, help="Child Rollout duration in seconds.")
    parser.add_argument("--pre-branch", type=float, default=DEFAULT_PRE_BRANCH_S, help="Pre-branch history in seconds.")
    parser.add_argument("--region-fs", type=float, default=1000.0, help="Decimated regional LFP sampling rate in Hz.")
    parser.add_argument(
        "--branch-times",
        type=str,
        default=None,
        help="Optional comma-separated t_branch overrides (seconds) for BRANCHES.",
    )
    parser.add_argument("--generate", action="store_true", help="Generate the ensemble stores and manifest.")
    parser.add_argument("--score", action="store_true", help="Score the generated ensemble directory.")
    parser.add_argument("--dry-run", action="store_true", help="Verify Plants and print plan without generating files.")
    return parser.parse_args(argv)


def _build_config(args: argparse.Namespace) -> EnsembleConfig:
    """Configure the zero-current seizure-only experiment."""
    branches = BRANCHES
    if args.branch_times is not None:
        times = [float(x.strip()) for x in args.branch_times.split(",") if x.strip()]
        if len(times) != len(BRANCHES):
            msg = f"--branch-times requires {len(BRANCHES)} values, got {len(times)}."
            raise ValueError(msg)
        branches = tuple(replace(b, t_branch=t) for b, t in zip(BRANCHES, times, strict=True))

    reference_seeds = tuple(101 + i for i in range(args.n_reference))
    dev_seeds = tuple(201 + i for i in range(args.n_dev))
    return EnsembleConfig(
        n_parents=args.n_parents,
        n_children=args.n_children,
        rollout_s=args.rollout,
        pre_branch_s=args.pre_branch,
        branches=branches,
        reference_seeds=reference_seeds,
        dev_seeds=dev_seeds,
        region_fs=args.region_fs,
        store_conditioned_eeg=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Plan, generate, or score an evaluation ensemble and report progress."""
    args = parse_args(argv)
    cfg = _build_config(args)
    protocol = DEFAULT_EVALUATION_PROTOCOL
    plants = build_plants(args.config)
    verify_plants(plants)

    identities = cfg.expected_identities()
    sim_s = simulated_seconds(cfg)
    n_eval_parent_runs = sum(len(cfg.split_parent_seeds("eval", plant)) for plant in ("healthy", "seizure"))
    source_label = "repo-local default_plant_params()" if args.config is None else str(args.config)

    print(f"plant      : {source_label} (healthy A={A_HEALTHY}, seizure EZ/PZ verified)")
    print(f"leadfield  : {plants.leadfield.shape[0]} channels x {plants.leadfield.shape[1]} regions")
    print(f"branches   : {', '.join(f'{b.name}@{b.t_branch:g}s' for b in cfg.branches)}")
    print(f"arms       : {', '.join(f'{a.name}={list(a.current)}' for a in all_arms(cfg.branches))}")
    print(
        f"splits     : reference={len(cfg.reference_seeds)}, "
        f"dev={len(cfg.dev_seeds)} healthy parents, eval={n_eval_parent_runs} total parents"
    )
    print(f"history    : {cfg.pre_branch_s:g} s pre-branch, {cfg.rollout_s:g} s rollout")
    print(
        f"geometry   : waveform={protocol.target_fs:g} Hz (cutoff={protocol.cutoff_hz:g} Hz), "
        f"spectral={protocol.stft_geometry.n_segment} samples/{protocol.stft_geometry.n_hop} hop, "
        f"connectivity={protocol.connectivity_band_hz} Hz at {protocol.hop_s:g} s hop"
    )
    print(f"identities : {len(identities)} total rollouts")
    print(f"simulated  : {sim_s:.1f} s")
    print(f"storage    : {stored_bytes(cfg, plants) / 1e9:.3f} GB into {args.out}")
    if args.dry_run:
        return 0

    t0 = time.perf_counter()
    if args.generate or not args.score:
        for done, _ in enumerate(generate_iter(args.out, cfg, plants), start=1):
            elapsed = time.perf_counter() - t0
            frac = min(done / max(n_eval_parent_runs, 1), 1.0)
            eta = elapsed / frac - elapsed if frac > 0 else 0.0
            print(
                f"parent {min(done, n_eval_parent_runs)}/{n_eval_parent_runs} | "
                f"{elapsed:.0f} s elapsed | {eta:.0f} s left"
            )

    if args.score:
        bundle = load_or_score_ensemble(args.out, protocol=protocol, rebuild=args.score)
        print(f"scored     : W_conn={bundle.metadata.selected_connectivity_window_s:g} s")

    wall = time.perf_counter() - t0
    print(f"\ndone in {wall:.1f} s ({wall / max(sim_s, 1e-9):.2f}x realtime), manifest at {args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
