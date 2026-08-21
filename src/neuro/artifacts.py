from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from neuro.esn import ESNArtifact
from neuro.esn_predictor_casadi import ESNSymbolicModel
from neuro.nn_predictor_casadi import NNSymbolicModel
from neuro.observable import ObservableArtifact
from neuro.observable_casadi import ObservableSymbolicModel
from neuro.predictor.artifact import MLPArtifact

if TYPE_CHECKING:
    from collections.abc import Iterator

    from neuro.types import FloatArray, ObservableModel, SymbolicModel

RolloutArtifact = MLPArtifact | ESNArtifact
"""Artifacts that free-run on the sample grid; the free-run scores below only accept these."""
PredictorArtifact = RolloutArtifact | ObservableArtifact

# Energies are mean squares in mV**2, so this floors the log of a prediction that has collapsed
# to silence rather than of a genuinely quiet one.
_ENERGY_EPS = 1e-12


def load_any_artifact(artifact_path: str | Path) -> PredictorArtifact:
    """Load a single-``.npz`` predictor artifact (MLP, ESN or observable) from disk."""
    p = Path(artifact_path)
    npz_path = p.with_suffix(".npz")
    with np.load(npz_path) as npz:
        model_type = json.loads(str(npz["meta"]))["model_type"]
    if model_type == "mlp":
        return MLPArtifact.load(p)
    if model_type == "esn":
        return ESNArtifact.load(p)
    if model_type == "observable":
        return ObservableArtifact.load(p)
    msg = f"unsupported model_type {model_type!r} in {npz_path}"
    raise ValueError(msg)


def load_rollout_artifact(artifact_path: str | Path) -> RolloutArtifact:
    """Load an artifact that free-runs on the sample grid, rejecting an observable one."""
    art = load_any_artifact(artifact_path)
    if isinstance(art, ObservableArtifact):
        msg = f"{artifact_path} is an observable artifact; it forecasts the Observable and never a waveform."
        raise TypeError(msg)
    return art


def build_symbolic_model(art: PredictorArtifact) -> SymbolicModel | ObservableModel:
    """Build the appropriate symbolic model bridge; the MPC branches on which of the two it gets."""
    if isinstance(art, ESNArtifact):
        return ESNSymbolicModel(art)
    if isinstance(art, ObservableArtifact):
        return ObservableSymbolicModel(art)
    return NNSymbolicModel(art)


def rollout_batches(
    art: RolloutArtifact,
    trajectories: list[tuple[FloatArray, FloatArray]],
    steps: int,
    *,
    stride: int = 25,
    start: int | None = None,
) -> Iterator[tuple[FloatArray, FloatArray]]:
    """Yield ``(y_pred, y_true)`` of shape ``(n_windows, steps, n_channels)``, one batch per trajectory.

    The whole t0 grid of a trajectory is primed and rolled out in one batched call, so every
    free-run score reads the same windows off one traversal rather than re-rolling per metric.
    ``start`` overrides the first window index, so several artifacts can share one t0 grid.
    """
    grid_start = art.priming_steps if start is None else start
    k = art.priming_steps

    for u, y in trajectories:
        t0s = range(grid_start, len(y) - steps, stride)
        if not t0s:
            continue

        states = art.prime_many(
            np.stack([y[t0 - k : t0] for t0 in t0s]),
            np.stack([u[t0 - k : t0] for t0 in t0s]),
        )
        y_pred = art.rollout_many(states, np.stack([u[t0 : t0 + steps] for t0 in t0s]))
        yield y_pred, np.stack([y[t0 : t0 + steps] for t0 in t0s])


def accumulate_rollout_errors(
    art: RolloutArtifact,
    trajectories: list[tuple[FloatArray, FloatArray]],
    steps: int,
    *,
    stride: int = 25,
    start: int | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Accumulate per-step squared error, true power and predicted power over free-run windows."""
    sq_err = np.zeros(steps, dtype=np.float64)
    power = np.zeros(steps, dtype=np.float64)
    pred_power = np.zeros(steps, dtype=np.float64)

    for y_pred, y_true in rollout_batches(art, trajectories, steps, stride=stride, start=start):
        sq_err += ((y_pred - y_true) ** 2).sum(axis=(0, 2))
        power += (y_true**2).sum(axis=(0, 2))
        pred_power += (y_pred**2).sum(axis=(0, 2))

    return sq_err, power, pred_power


def nmse(sq_err: FloatArray | float, power: FloatArray | float) -> FloatArray:
    """Normalize squared error by the true signal's energy, elementwise (``inf`` where it is silent).

    The reference is the uncentered second moment ``sum(y_true ** 2)``, not the variance, so
    ``1.0`` is the score of the zero predictor. This is the repo's single NMSE definition; every
    reported NMSE -- per horizon step, pooled, one-step state-absorbed or free-running -- goes through here.
    """
    err = np.asarray(sq_err, dtype=np.float64)
    ref = np.asarray(power, dtype=np.float64)
    return np.divide(err, ref, out=np.full_like(err, np.inf), where=ref > 0)


class RolloutNMSE(NamedTuple):
    """Free-run rollout NMSE, resolved per horizon step and pooled over the whole horizon."""

    pooled: float
    per_step: FloatArray


def evaluate_rollouts(
    art: RolloutArtifact,
    val_trajs: list[tuple[FloatArray, FloatArray]],
    horizon: int,
    step_stride: int = 25,
) -> RolloutNMSE:
    """Evaluate free-run rollout NMSE per horizon step and pooled over every step and window."""
    sq_err, power, _ = accumulate_rollout_errors(art, val_trajs, horizon, stride=step_stride)
    return RolloutNMSE(pooled=float(nmse(sq_err.sum(), power.sum())), per_step=nmse(sq_err, power))


def window_energy(y: FloatArray, window_steps: int, hop_steps: int) -> FloatArray:
    """Cross-channel mean square of ``(n_windows, steps, n_channels)`` per trailing window.

    Returns shape ``(n_windows, n_positions)``, one energy per window position along the horizon.
    """
    starts = range(0, y.shape[1] - window_steps + 1, hop_steps)
    return np.stack([(y[:, s : s + window_steps] ** 2).mean(axis=(1, 2)) for s in starts], axis=1)


class LogEnergyError(NamedTuple):
    """Free-run error in the functional the MPC actually costs, per window position and pooled."""

    pooled: float
    per_position: FloatArray


def evaluate_log_energy(  # noqa: PLR0913
    art: RolloutArtifact,
    val_trajs: list[tuple[FloatArray, FloatArray]],
    horizon: int,
    *,
    window_steps: int,
    hop_steps: int,
    step_stride: int = 25,
) -> LogEnergyError:
    """Mean squared log-ratio of predicted to true windowed energy over free-run windows.

    The MPC costs ``sumsqr(y)`` over its horizon, never the waveform, so this scores the quantity
    the controller consumes: a phase-scrambled rollout carrying the right energy course is worth
    the same to it, while waveform NMSE saturates at 1.0 once phase decorrelates and stops
    separating models. Log-space because energy spans orders of magnitude between interictal and
    ictal, and because the controller responds to the ratio rather than the difference.

    Resolved per window rather than pooled over windows first: pooling the numerator and
    denominator would let over- and under-prediction cancel across windows, which a model that is
    right only on average would score perfectly.

    Lower is better and ``0.0`` is exact; unlike NMSE it is unbounded above, so a prediction that
    decays to silence is scored as the failure it is instead of tying with every other one.

    Raises
    ------
    ValueError
        If ``horizon`` is shorter than one window, which would leave nothing to score.
    """
    if horizon < window_steps:
        msg = f"horizon ({horizon}) is shorter than the energy window ({window_steps} steps)."
        raise ValueError(msg)

    total: FloatArray | None = None
    n_windows = 0
    for y_pred, y_true in rollout_batches(art, val_trajs, horizon, stride=step_stride):
        log_ratio = np.log(window_energy(y_pred, window_steps, hop_steps) + _ENERGY_EPS) - np.log(
            window_energy(y_true, window_steps, hop_steps) + _ENERGY_EPS
        )
        sq = (log_ratio**2).sum(axis=0)
        total = sq if total is None else total + sq
        n_windows += y_pred.shape[0]

    if total is None:
        msg = "No validation trajectory is long enough to hold one free-run window."
        raise ValueError(msg)

    per_position = total / n_windows
    return LogEnergyError(pooled=float(per_position.mean()), per_position=per_position)
