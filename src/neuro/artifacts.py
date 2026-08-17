from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from neuro.esn import ESNArtifact
from neuro.esn_predictor_casadi import ESNSymbolicModel
from neuro.nn_predictor_casadi import NNSymbolicModel
from neuro.predictor.artifact import MLPArtifact

if TYPE_CHECKING:
    from neuro.types import FloatArray, SymbolicModel

PredictorArtifact = MLPArtifact | ESNArtifact


def load_any_artifact(artifact_path: str | Path) -> PredictorArtifact:
    """Load a single-``.npz`` predictor artifact (MLP or ESN) from disk."""
    p = Path(artifact_path)
    npz_path = p.with_suffix(".npz")
    with np.load(npz_path) as npz:
        model_type = json.loads(str(npz["meta"]))["model_type"]
    if model_type == "mlp":
        return MLPArtifact.load(p)
    if model_type == "esn":
        return ESNArtifact.load(p)
    msg = f"unsupported model_type {model_type!r} in {npz_path}"
    raise ValueError(msg)


def build_symbolic_model(art: PredictorArtifact) -> SymbolicModel:
    """Build the appropriate symbolic model bridge for a predictor artifact."""
    if isinstance(art, ESNArtifact):
        return ESNSymbolicModel(art)
    return NNSymbolicModel(art)


def accumulate_rollout_errors(
    art: PredictorArtifact,
    trajectories: list[tuple[FloatArray, FloatArray]],
    steps: int,
    *,
    stride: int = 25,
    start: int | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Accumulate per-step squared error, true power and predicted power over free-run windows.

    ``start`` overrides the first window index, so several artifacts can share one t0 grid.
    The whole t0 grid of a trajectory is rolled out in one batched call.
    """
    grid_start = art.priming_steps if start is None else start
    k = art.priming_steps
    sq_err = np.zeros(steps, dtype=np.float64)
    power = np.zeros(steps, dtype=np.float64)
    pred_power = np.zeros(steps, dtype=np.float64)

    for u, y in trajectories:
        t0s = range(grid_start, len(y) - steps, stride)
        if not t0s:
            continue

        states = art.prime_many(
            np.stack([y[t0 - k : t0] for t0 in t0s]),
            np.stack([u[t0 - k : t0] for t0 in t0s]),
        )
        y_pred = art.rollout_many(states, np.stack([u[t0 : t0 + steps] for t0 in t0s]))
        y_true = np.stack([y[t0 : t0 + steps] for t0 in t0s])

        sq_err += ((y_pred - y_true) ** 2).sum(axis=(0, 2))
        power += (y_true**2).sum(axis=(0, 2))
        pred_power += (y_pred**2).sum(axis=(0, 2))

    return sq_err, power, pred_power


def nmse(sq_err: FloatArray | float, power: FloatArray | float) -> FloatArray:
    """Normalize squared error by the true signal's energy, elementwise (``inf`` where it is silent).

    The reference is the uncentered second moment ``sum(y_true ** 2)``, not the variance, so
    ``1.0`` is the score of the zero predictor. This is the repo's single NMSE definition; every
    reported NMSE -- per horizon step, pooled, teacher-forced or free-running -- goes through here.
    """
    err = np.asarray(sq_err, dtype=np.float64)
    ref = np.asarray(power, dtype=np.float64)
    return np.divide(err, ref, out=np.full_like(err, np.inf), where=ref > 0)


class RolloutNMSE(NamedTuple):
    """Free-run rollout NMSE, resolved per horizon step and pooled over the whole horizon."""

    pooled: float
    per_step: FloatArray


def evaluate_rollouts(
    art: PredictorArtifact,
    val_trajs: list[tuple[FloatArray, FloatArray]],
    horizon: int,
    step_stride: int = 25,
) -> RolloutNMSE:
    """Evaluate free-run rollout NMSE per horizon step and pooled over every step and window."""
    sq_err, power, _ = accumulate_rollout_errors(art, val_trajs, horizon, stride=step_stride)
    return RolloutNMSE(pooled=float(nmse(sq_err.sum(), power.sum())), per_step=nmse(sq_err, power))
