from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from neuro.predictor.data import frame_aligned_controls, reduce_trajectory_to_frames
from neuro.predictor.inference import InferencePredictor, ObservableCNNModel, ObservableMLPModel
from neuro.spectral import compute_forecast_log_power_frames

if TYPE_CHECKING:
    from collections.abc import Iterator

    from neuro.config import StftGeometry
    from neuro.types import FloatArray


def rollout_batches(
    model: InferencePredictor,
    trajectories: list[tuple[FloatArray, FloatArray]],
    steps: int,
    *,
    stride: int = 25,
    start: int | None = None,
) -> Iterator[tuple[FloatArray, FloatArray]]:
    """Yield predictions and targets using the shared causal Rollout boundary convention.

    Control history ends at t0 - 1; the candidate Current at t0 acts on the first prediction.
    Waveform samples and Observable Frames retain their output axes.
    """
    for pred, true, _, _ in _rollout_windows(model, trajectories, steps, stride=stride, start=start):
        yield pred, true


def _rollout_windows(
    model: InferencePredictor,
    trajectories: list[tuple[FloatArray, FloatArray]],
    steps: int,
    *,
    stride: int,
    start: int | None,
) -> Iterator[tuple[FloatArray, FloatArray, range, FloatArray]]:
    """Batch causal Rollouts while retaining their anchors and recording for spectral history."""
    k = model.priming_steps
    grid_start = k if start is None else start

    for u, y in trajectories:
        t0s = range(grid_start, len(y) - steps, stride)
        if not t0s:
            continue

        y_pred = np.asarray(
            model.free_run(
                np.stack([y[t0 - k + 1 : t0 + 1] for t0 in t0s]),
                np.stack([u[t0 - k : t0] for t0 in t0s]),
                np.stack([u[t0 : t0 + steps] for t0 in t0s]),
            )
        )
        yield y_pred, np.stack([y[t0 + 1 : t0 + 1 + steps] for t0 in t0s]), t0s, y


def nmse(sq_err: FloatArray | float, power: FloatArray | float) -> FloatArray:
    """Normalize squared error by the true signal's energy, elementwise (``inf`` where it is silent).

    The reference is the uncentered second moment ``sum(y_true ** 2)``, not the variance, so
    ``1.0`` is the score of the zero predictor. This is the repo's single NMSE definition; every
    reported NMSE -- per horizon step, pooled, one-step state-absorbed or free-running -- goes through here.
    """
    err = np.asarray(sq_err, dtype=np.float64)
    ref = np.asarray(power, dtype=np.float64)
    return np.divide(err, ref, out=np.full_like(err, np.inf), where=ref > 0)


class ObservableFrameMSE(NamedTuple):
    """Free-run log-power Frame MSE on held-out trajectories, per step and pooled."""

    pooled: float
    per_step: FloatArray


def evaluate_observable_free_run(
    model: InferencePredictor,
    val_trajs: list[tuple[FloatArray, FloatArray]],
    eval_steps: int,
    *,
    step_stride: int = 1,
) -> ObservableFrameMSE:
    """Evaluate causal Frame MSE with the same histories and future targets as training.

    Parameters
    ----------
    model : InferencePredictor
        The inference predictor adapter.
    val_trajs : list[tuple[FloatArray, FloatArray]]
        Held-out validation trajectories in Frame space, shape ``(n_frames, n_controls)`` and
        ``(n_frames, n_channels, n_values)``.
    eval_steps : int
        Free-run horizon in Frames.
    step_stride : int, optional
        Spacing between window start anchors. Defaults to 1.

    Returns
    -------
    ObservableFrameMSE
        Pooled and per-step MSE over raw log-power Frames.
    """
    sq_err = np.zeros(eval_steps, dtype=np.float64)
    n_windows = 0

    for y_pred, y_true in rollout_batches(model, val_trajs, eval_steps, stride=step_stride):
        value_axes = (0, *range(2, y_pred.ndim))
        sq_err += ((y_pred - y_true) ** 2).sum(axis=value_axes)
        n_windows += len(y_pred)

    if n_windows == 0:
        msg = "No validation trajectory is long enough to hold one free-run window."
        raise ValueError(msg)

    per_step = sq_err / (n_windows * model.n_outputs)
    return ObservableFrameMSE(pooled=float(per_step.mean()), per_step=per_step)


def evaluate_spectral_free_run(  # noqa: PLR0913 -- Predictor and physical evaluation grid
    model: InferencePredictor,
    val_trajs: list[tuple[FloatArray, FloatArray]],
    eval_frames: int,
    *,
    geometry: StftGeometry | None = None,
    fs: float | None = None,
    step_stride: int = 1,
    start_frame: int | None = None,
) -> ObservableFrameMSE:
    """Score raw EEG recordings on a common physical Frame grid in unstandardized log-power.

    ``val_trajs`` contains raw controls and EEG, shaped ``(n_samples, n_controls)`` and
    ``(n_samples, n_channels)``. ``step_stride`` and ``start_frame`` are in Frames for both
    Predictor kinds. The default start is the earliest Frame anchor satisfying Priming;
    when comparing models, pass the same ``start_frame``
    large enough for both. Geometry and raw sampling rate must match across comparisons.
    Inference adapters already undo output standardization.
    """
    observable = isinstance(model, (ObservableMLPModel, ObservableCNNModel))
    geom = getattr(model, "geometry", None) if geometry is None else geometry
    if geom is None:
        msg = "Spectral evaluation requires an StftGeometry."
        raise ValueError(msg)
    rate = fs if fs is not None else (geom.n_hop if observable else 1) / model.dt
    support = geom.sample_support_steps(rate)
    minimum = (
        model.priming_steps if observable else max(0, math.ceil((model.priming_steps - (support - 1)) / geom.n_hop))
    )
    first = minimum if start_frame is None else start_frame
    start = support - 1 + first * geom.n_hop
    if first < minimum:
        msg = "Evaluation start precedes the Predictor's required Priming history."
        raise ValueError(msg)
    if observable:
        if geom != model.geometry:
            msg = "Evaluation geometry must match the Observable Predictor geometry."
            raise ValueError(msg)
        frames = [
            (frame_aligned_controls(u, geom, fs=rate), reduce_trajectory_to_frames(y, geom, rate)) for u, y in val_trajs
        ]
        windows = _rollout_windows(model, frames, eval_frames, stride=step_stride, start=first)
    else:
        windows = _rollout_windows(
            model,
            val_trajs,
            eval_frames * geom.n_hop,
            stride=step_stride * geom.n_hop,
            start=start,
        )
    sq_err = np.zeros(eval_frames, dtype=np.float64)
    n_windows = 0
    prefix_len = support - geom.n_hop
    for predictions, targets, anchors, recording in windows:
        for pred, true, t0 in zip(predictions, targets, anchors, strict=True):
            pred_frames = np.asarray(pred)
            true_frames = np.asarray(true)
            if not observable:
                history = recording[t0 - prefix_len + 1 : t0 + 1]
                pred_frames = compute_forecast_log_power_frames(history, pred_frames, geom, fs=rate)
                true_frames = compute_forecast_log_power_frames(history, true_frames, geom, fs=rate)
            sq_err += ((pred_frames - true_frames) ** 2).sum(axis=(1, 2))
            n_windows += 1
    if n_windows == 0:
        msg = "No validation trajectory is long enough to hold one free-run window."
        raise ValueError(msg)
    per_step = sq_err / (n_windows * val_trajs[0][1].shape[1] * geom.n_values(rate))
    return ObservableFrameMSE(pooled=float(per_step.mean()), per_step=per_step)
