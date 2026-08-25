from __future__ import annotations

import argparse
from pathlib import Path

from neuro.checkpoint import RolloutCheckpoint, load_rollout
from neuro.predictor.data import load_trajectory
from neuro.predictor.esn_module import ESNModule
from neuro.predictor.evaluation import accumulate_rollout_errors, nmse
from neuro.predictor.module import AutoregressiveMLP

_REPORT_STEPS = (1, 5, 10, 20, 40, 60, 80, 100, 125, 150)


def _load_module(path: Path) -> AutoregressiveMLP | ESNModule:
    """Load the torch rollout Predictor whose checkpoint ``path`` names."""
    ckpt = load_rollout(path)
    if ckpt.model_type == "mlp":
        return AutoregressiveMLP.load(path)
    return ESNModule.load(path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the rollout-horizon probe."""
    parser = argparse.ArgumentParser(
        description="Measure per-step NMSE and power_ratio of identified predictors (MLP/ESN) rolled out far past training horizon."
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        nargs="+",
        required=True,
        help="Checkpoint basename(s), e.g. artifacts/mlp_model artifacts/esn_model",
    )
    parser.add_argument("--data", type=Path, required=True, help="Directory of held-out .npz trajectories.")
    parser.add_argument("--max-steps", type=int, default=150, help="Longest rollout to evaluate, in model steps.")
    parser.add_argument("--stride", type=int, default=25, help="Stride between rollout start windows.")
    return parser.parse_args()


def main() -> None:
    """Roll predictor(s) out to --max-steps on held-out data and print per-step NMSE and power_ratio."""
    args = parse_args()
    checkpoint_paths: list[Path] = args.artifact
    ckpts: list[RolloutCheckpoint] = [load_rollout(p) for p in checkpoint_paths]
    predictors = [_load_module(p) for p in checkpoint_paths]

    files = sorted(str(p) for p in args.data.glob("*.npz"))
    if not files:
        msg = f"no .npz trajectories found in {args.data}"
        raise SystemExit(msg)

    # Shared start window alignment across all predictors
    global_start = max(c.priming_steps for c in ckpts)

    # One resampling drives every checkpoint, so they must agree on it or the co-evaluation is a lie.
    base_ckpt = ckpts[0]
    mismatched = [
        str(p)
        for p, c in zip(checkpoint_paths, ckpts, strict=True)
        if (c.downsample, c.dt) != (base_ckpt.downsample, base_ckpt.dt)
    ]
    if mismatched:
        msg = (
            f"checkpoints co-evaluated in one run must share downsample/dt; "
            f"{mismatched} differ from {checkpoint_paths[0]} "
            f"(downsample={base_ckpt.downsample}, dt={base_ckpt.dt})"
        )
        raise SystemExit(msg)

    # Load trajectories once
    print(f"Loading {len(files)} trajectories from {args.data}...", flush=True)
    trajectories = []
    for f in files:
        u, y = load_trajectory(f, None, base_ckpt.downsample, base_ckpt.dt / base_ckpt.downsample)
        trajectories.append((u, y))

    print(f"Loaded {len(trajectories)} trajectories.", flush=True)

    for path, ckpt, predictor in zip(checkpoint_paths, ckpts, predictors, strict=True):
        sq_err, power, pred_power = accumulate_rollout_errors(
            predictor, trajectories, args.max_steps, stride=args.stride, start=global_start
        )

        per_step_nmse = nmse(sq_err, power)
        power_ratio = pred_power / power

        print(
            f"\nCheckpoint: {path} ({ckpt.model_type}), native horizon {ckpt.horizon}, dt {ckpt.dt:.4f} s",
            flush=True,
        )
        print(f"{'step':>6} {'lookahead':>10} {'NMSE':>9} {'power_ratio':>13}", flush=True)
        for k in _REPORT_STEPS:
            if k <= args.max_steps:
                print(
                    f"{k:>6} {k * ckpt.dt:>9.2f}s {per_step_nmse[k - 1]:>9.4f} {power_ratio[k - 1]:>13.4f}", flush=True
                )


if __name__ == "__main__":
    main()
