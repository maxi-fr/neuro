from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from neuro.artifacts import RolloutArtifact, load_rollout_artifact
from neuro.esn import ESNArtifact
from neuro.predictor.artifact import MLPArtifact
from neuro.predictor.data import load_trajectory
from neuro.predictor.esn_module import ESNModule
from neuro.predictor.evaluation import accumulate_rollout_errors, nmse
from neuro.predictor.module import AutoregressiveMLP

if TYPE_CHECKING:
    from neuro.types import Predictor

_REPORT_STEPS = (1, 5, 10, 20, 40, 60, 80, 100, 125, 150)


def _as_module(art: RolloutArtifact) -> Predictor:
    """Rebuild the torch module twin of a loaded artifact for protocol-based evaluation."""
    if isinstance(art, MLPArtifact):
        return AutoregressiveMLP.from_artifact(art)
    if isinstance(art, ESNArtifact):
        return ESNModule(
            w_res=art.w_res,
            w_in=art.w_in,
            w_out=art.w_out,
            leak_rate=art.leak_rate,
            priming_steps=art.priming_steps,
            horizon=art.horizon,
            dt=art.dt,
            y_std=art.y_std,
            u_std=art.u_std,
        )
    msg = f"unsupported artifact type {type(art).__name__}"
    raise ValueError(msg)


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
    """Roll predictor(s) out to --max-steps on held-out data and print per-step NMSE and power_ratio."""
    args = parse_args()
    artifact_paths: list[Path] = args.artifact
    artifacts = [load_rollout_artifact(p) for p in artifact_paths]
    predictors = [_as_module(art) for art in artifacts]

    files = sorted(str(p) for p in args.data.glob("*.npz"))
    if not files:
        msg = f"no .npz trajectories found in {args.data}"
        raise SystemExit(msg)

    # Shared start window alignment across all predictors
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

    for path, art, predictor in zip(artifact_paths, artifacts, predictors, strict=True):
        sq_err, power, pred_power = accumulate_rollout_errors(
            predictor, trajectories, args.max_steps, stride=args.stride, start=global_start
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
