from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from neuro.nn_training import load_trajectory
from neuro.prediction import AutoregressivePredictor, MLPArtifact

if TYPE_CHECKING:
    from neuro.types import FloatArray

_REPORT_STEPS = (1, 5, 10, 20, 40, 60, 80, 100, 125, 150)


def extend_horizon(artifact: MLPArtifact, max_steps: int) -> AutoregressivePredictor:
    """Rebuild the artifact's predictor with a longer rollout, reusing the trained 1-step MLP."""
    p = artifact.model
    return AutoregressivePredictor(
        model=p.model,
        n_y=p.n_y,
        n_u=p.n_u,
        horizon=max_steps,
        n_channels=p.n_channels,
        n_controls=p.n_controls,
        activation=p.activation,
    )


def rollout_errors(  # noqa: PLR0913
    artifact: MLPArtifact,
    predictor: AutoregressivePredictor,
    u: FloatArray,
    y: FloatArray,
    max_steps: int,
    stride: int,
) -> tuple[FloatArray, FloatArray]:
    """Return per-step squared error and true power, both summed over windows and channels."""
    n_y, n_u = artifact.n_y, artifact.n_u
    z = artifact.encode(y)
    w = artifact.u_pipeline.transform(u)
    start = max(n_y, n_u)

    sq_err = np.zeros(max_steps)
    power = np.zeros(max_steps)
    for t0 in range(start, len(y) - max_steps, stride):
        x = np.concatenate(
            [
                z[t0 - n_y : t0].reshape(-1),
                w[t0 - n_u : t0].reshape(-1),
                w[t0 : t0 + max_steps].reshape(-1),
            ]
        )
        pred = np.asarray(predictor(jnp.asarray(x))).reshape(max_steps, -1)
        y_pred = artifact.decode(pred)
        y_true = y[t0 : t0 + max_steps]
        sq_err += ((y_pred - y_true) ** 2).sum(axis=1)
        power += (y_true**2).sum(axis=1)
    return sq_err, power


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the rollout-horizon probe."""
    parser = argparse.ArgumentParser(
        description="Measure per-step NMSE of an identified predictor rolled out far past its training horizon."
    )
    parser.add_argument("--artifact", type=Path, required=True, help="Artifact basename, e.g. artifacts/roast/x/model")
    parser.add_argument("--data", type=Path, required=True, help="Directory of held-out .npz trajectories.")
    parser.add_argument("--max-steps", type=int, default=150, help="Longest rollout to evaluate, in model steps.")
    parser.add_argument("--stride", type=int, default=25, help="Stride between rollout start windows.")
    return parser.parse_args()


def main() -> None:
    """Roll the artifact out to --max-steps on held-out data and print per-step NMSE."""
    args = parse_args()
    artifact = MLPArtifact.load(args.artifact)
    predictor = extend_horizon(artifact, args.max_steps)

    files = sorted(str(p) for p in args.data.glob("*.npz"))
    if not files:
        msg = f"no .npz trajectories found in {args.data}"
        raise SystemExit(msg)

    sq_err = np.zeros(args.max_steps)
    power = np.zeros(args.max_steps)
    for f in files:
        u, y = load_trajectory(f, None, artifact.downsample, artifact.dt / artifact.downsample)
        e, p = rollout_errors(artifact, predictor, u, y, args.max_steps, args.stride)
        sq_err += e
        power += p

    nmse = sq_err / power
    print(f"{len(files)} trajectories, native horizon {artifact.horizon}, dt {artifact.dt:.4f} s")
    print(f"{'step':>6} {'lookahead':>10} {'NMSE':>9}")
    for k in _REPORT_STEPS:
        if k <= args.max_steps:
            print(f"{k:>6} {k * artifact.dt:>9.2f}s {nmse[k - 1]:>9.4f}")


if __name__ == "__main__":
    main()
