from __future__ import annotations

import argparse
from pathlib import Path

from neuro.artifacts import accumulate_rollout_errors, load_any_artifact, nmse
from neuro.predictor.data import load_trajectory

_REPORT_STEPS = (1, 5, 10, 20, 40, 60, 80, 100, 125, 150)


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
        help="Artifact basename(s), e.g. artifacts/mlp_model artifacts/esn_model",
    )
    parser.add_argument("--data", type=Path, required=True, help="Directory of held-out .npz trajectories.")
    parser.add_argument("--max-steps", type=int, default=150, help="Longest rollout to evaluate, in model steps.")
    parser.add_argument("--stride", type=int, default=25, help="Stride between rollout start windows.")
    return parser.parse_args()


def main() -> None:
    """Roll artifact(s) out to --max-steps on held-out data and print per-step NMSE and power_ratio."""
    args = parse_args()
    artifact_paths: list[Path] = args.artifact
    artifacts = [load_any_artifact(p) for p in artifact_paths]

    files = sorted(str(p) for p in args.data.glob("*.npz"))
    if not files:
        msg = f"no .npz trajectories found in {args.data}"
        raise SystemExit(msg)

    # Shared start window alignment across all artifacts
    global_start = max(a.priming_steps for a in artifacts)

    # One resampling drives every artifact, so they must agree on it or the co-evaluation is a lie.
    base_art = artifacts[0]
    mismatched = [
        str(p)
        for p, a in zip(artifact_paths, artifacts, strict=True)
        if (a.downsample, a.dt) != (base_art.downsample, base_art.dt)
    ]
    if mismatched:
        msg = (
            f"artifacts co-evaluated in one run must share downsample/dt; "
            f"{mismatched} differ from {artifact_paths[0]} "
            f"(downsample={base_art.downsample}, dt={base_art.dt})"
        )
        raise SystemExit(msg)

    # Load trajectories once
    print(f"Loading {len(files)} trajectories from {args.data}...", flush=True)
    trajectories = []
    for f in files:
        u, y = load_trajectory(f, None, base_art.downsample, base_art.dt / base_art.downsample)
        trajectories.append((u, y))

    print(f"Loaded {len(trajectories)} trajectories.", flush=True)

    for path, art in zip(artifact_paths, artifacts, strict=True):
        sq_err, power, pred_power = accumulate_rollout_errors(
            art, trajectories, args.max_steps, stride=args.stride, start=global_start
        )

        per_step_nmse = nmse(sq_err, power)
        power_ratio = pred_power / power

        print(
            f"\nArtifact: {path} ({art.model_type}), native horizon {art.horizon}, dt {art.dt:.4f} s",
            flush=True,
        )
        print(f"{'step':>6} {'lookahead':>10} {'NMSE':>9} {'power_ratio':>13}", flush=True)
        for k in _REPORT_STEPS:
            if k <= args.max_steps:
                print(
                    f"{k:>6} {k * art.dt:>9.2f}s {per_step_nmse[k - 1]:>9.4f} {power_ratio[k - 1]:>13.4f}", flush=True
                )


if __name__ == "__main__":
    main()
