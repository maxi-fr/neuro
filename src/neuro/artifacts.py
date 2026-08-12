from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from neuro.esn import ESNArtifact
from neuro.esn_predictor_casadi import ESNSymbolicModel
from neuro.nn_predictor_casadi import NNSymbolicModel
from neuro.prediction import MLPArtifact

if TYPE_CHECKING:
    from neuro.types import FloatArray, SymbolicModel

PredictorArtifact = MLPArtifact | ESNArtifact


def load_any_artifact(artifact_path: str | Path) -> PredictorArtifact:
    """Load an artifact from disk, auto-detecting MLP vs ESN via its JSON sidecar."""
    p = Path(artifact_path)
    json_path = p.with_suffix(".json")
    if json_path.exists():
        meta: dict[str, object] = json.loads(json_path.read_text())
        if meta.get("model_type") == "esn":
            return ESNArtifact.load(p)
    return MLPArtifact.load(p)


def build_symbolic_model(art: PredictorArtifact) -> SymbolicModel:
    """Build the appropriate symbolic model bridge for a predictor artifact."""
    if isinstance(art, ESNArtifact):
        return ESNSymbolicModel(art)
    return NNSymbolicModel(art)


def prime_symbolic_state(art: PredictorArtifact, y_hist: FloatArray, u_hist: FloatArray) -> FloatArray:
    """Prime an initial state in the *CasADi* convention: model-space y, but **raw** u.

    ``MLPArtifact.prime`` follows the JAX convention and carries a model-space u tail, which
    ``NNSymbolicModel.step`` would then z-score a second time. The ESN state is a bare ``h``, so it
    is convention-free and this is just :meth:`prime`.
    """
    if isinstance(art, MLPArtifact):
        z_hist = art.encode(np.asarray(y_hist, dtype=np.float64))
        u_arr = np.asarray(u_hist, dtype=np.float64)
        return np.concatenate([z_hist[-art.n_y :].reshape(-1), u_arr[-art.n_u :].reshape(-1)])
    return art.prime(y_hist, u_hist)


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
    """
    grid_start = art.priming_steps if start is None else start
    sq_err = np.zeros(steps, dtype=np.float64)
    power = np.zeros(steps, dtype=np.float64)
    pred_power = np.zeros(steps, dtype=np.float64)

    for u, y in trajectories:
        for t0 in range(grid_start, len(y) - steps, stride):
            state = art.prime(y[t0 - art.priming_steps : t0], u[t0 - art.priming_steps : t0])
            y_pred = art.rollout(state, u[t0 : t0 + steps])
            y_true = y[t0 : t0 + steps]

            sq_err += ((y_pred - y_true) ** 2).sum(axis=1)
            power += (y_true**2).sum(axis=1)
            pred_power += (y_pred**2).sum(axis=1)

    return sq_err, power, pred_power


def evaluate_rollouts(
    art: PredictorArtifact,
    val_trajs: list[tuple[FloatArray, FloatArray]],
    horizon: int,
    step_stride: int = 25,
) -> float:
    """Evaluate rollout NMSE pooled over every horizon step and window."""
    sq_err, power, _ = accumulate_rollout_errors(art, val_trajs, horizon, stride=step_stride)
    total_power = float(power.sum())
    return float(sq_err.sum() / total_power) if total_power > 0 else float("inf")
