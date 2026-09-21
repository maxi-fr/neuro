from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import jax.numpy as jnp
import numpy as np

from neuro.metrics import DEFAULT_HOP_S, METRICS
from neuro.predictor.data import load_trajectory
from neuro.predictor.inference import InferencePredictor, ObservableCNNModel, ObservableMLPModel
from neuro.predictor.replay import predict
from neuro.run_view import Run
from neuro.spectral import compute_log_power_frames

if TYPE_CHECKING:
    from collections.abc import Iterator

    from neuro.config import StftGeometry
    from neuro.types import FloatArray

# Energies are mean squares in mV**2, so this floors the log of a prediction that has collapsed
# to silence rather than of a genuinely quiet one.
_ENERGY_EPS = 1e-12


def rollout_batches(
    model: InferencePredictor,
    trajectories: list[tuple[FloatArray, FloatArray]],
    steps: int,
    *,
    stride: int = 25,
    start: int | None = None,
) -> Iterator[tuple[FloatArray, FloatArray]]:
    """Yield one trajectory batch of predictions and targets with all output axes preserved.

    The whole t0 grid of a trajectory is primed and rolled out in one stateless jax ``free_run``
    call, so every free-run score reads the same windows off one traversal rather than re-rolling
    per metric. ``start`` overrides the first window index, so several models can share one t0
    grid. The scores live on the sample grid -- one output per position -- so the waveform MLP is
    the intended subject, not the observable predictor, whose ``free_run`` emits one Frame per
    position.
    """
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
        yield y_pred, np.stack([y[t0 + 1 : t0 + 1 + steps] for t0 in t0s])


def accumulate_rollout_errors(
    model: InferencePredictor,
    trajectories: list[tuple[FloatArray, FloatArray]],
    steps: int,
    *,
    stride: int = 25,
    start: int | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Accumulate per-step errors over free-run windows, summing every output axis after time."""
    sq_err = np.zeros(steps, dtype=np.float64)
    power = np.zeros(steps, dtype=np.float64)
    pred_power = np.zeros(steps, dtype=np.float64)

    for y_pred, y_true in rollout_batches(model, trajectories, steps, stride=stride, start=start):
        value_axes = (0, *range(2, y_pred.ndim))
        sq_err += ((y_pred - y_true) ** 2).sum(axis=value_axes)
        power += (y_true**2).sum(axis=value_axes)
        pred_power += (y_pred**2).sum(axis=value_axes)

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


def evaluate_free_run(
    model: InferencePredictor,
    val_trajs: list[tuple[FloatArray, FloatArray]],
    eval_steps: int,
    fs: float,
) -> tuple[RolloutNMSE, LogEnergyError]:
    """Score free-run rollouts and the windowed-energy course over the held-out trajectories.

    Every training arm runs the same evaluation: waveform NMSE via :func:`evaluate_rollouts` and
    the windowed-energy course the MPC costs via :func:`evaluate_log_energy`. The energy course
    follows the metrics layer's own eeg_ms convention rather than a knob of its own, clamped
    where the evaluation horizon is too short to hold one window.
    """
    energy_window = min(max(1, round(METRICS["eeg_ms"].window_s * fs)), eval_steps)
    energy_hop = max(1, round(DEFAULT_HOP_S * fs))
    rollout = evaluate_rollouts(model, val_trajs, eval_steps)
    log_energy = evaluate_log_energy(model, val_trajs, eval_steps, window_steps=energy_window, hop_steps=energy_hop)
    return rollout, log_energy


class RolloutNMSE(NamedTuple):
    """Free-run rollout NMSE, resolved per horizon step and pooled over the whole horizon."""

    pooled: float
    per_step: FloatArray


def evaluate_rollouts(
    model: InferencePredictor,
    val_trajs: list[tuple[FloatArray, FloatArray]],
    horizon: int,
    step_stride: int = 25,
) -> RolloutNMSE:
    """Evaluate free-run rollout NMSE per horizon step and pooled over every step and window."""
    sq_err, power, _ = accumulate_rollout_errors(model, val_trajs, horizon, stride=step_stride)
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
    model: InferencePredictor,
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
    for y_pred, y_true in rollout_batches(model, val_trajs, horizon, stride=step_stride):
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
    """Evaluate free-run Frame MSE per horizon step and pooled over all validation windows.

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
    k = model.priming_steps
    sq_err = np.zeros(eval_steps, dtype=np.float64)
    n_windows = 0

    for u, y in val_trajs:
        t0s = list(range(k, len(y) - eval_steps + 1, step_stride))
        if not t0s:
            continue

        y_pred = np.asarray(
            model.free_run(
                np.stack([y[t0 - k : t0] for t0 in t0s]),
                np.stack([u[t0 - k : t0] for t0 in t0s]),
                np.stack([u[t0 : t0 + eval_steps] for t0 in t0s]),
            )
        )
        y_true = np.stack([y[t0 : t0 + eval_steps] for t0 in t0s])
        value_axes = (0, *range(2, y_pred.ndim))
        sq_err += ((y_pred - y_true) ** 2).sum(axis=value_axes)
        n_windows += len(t0s)

    if n_windows == 0:
        msg = "No validation trajectory is long enough to hold one free-run window."
        raise ValueError(msg)

    per_step = sq_err / (n_windows * model.n_outputs)
    return ObservableFrameMSE(pooled=float(per_step.mean()), per_step=per_step)


def free_run_stats(
    free_run: RolloutNMSE | ObservableFrameMSE, log_energy: LogEnergyError | None
) -> dict[str, float | list[float]]:
    """Serialize the free-run scores under keys naming the metric the kind actually computed."""
    prefix = "nmse_rollout" if isinstance(free_run, RolloutNMSE) else "frame_mse"
    stats: dict[str, float | list[float]] = {
        prefix: free_run.pooled,
        f"{prefix}_per_step": free_run.per_step.tolist(),
    }
    if log_energy is not None:
        stats["log_energy"] = log_energy.pooled
        stats["log_energy_per_position"] = log_energy.per_position.tolist()
    return stats


class EligibilityStatus(StrEnum):
    """Candidate eligibility verdict for MPC Control Horizon deployment."""

    PASS = "PASS"  # noqa: S105 -- candidate eligibility verdict, not a password
    FAIL = "FAIL"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class NumericalTolerances:
    """Documented numerical tolerances for floating-point agreement and forecast reconstruction."""

    reconstruction_atol: float = 1e-5
    reconstruction_rtol: float = 1e-4
    energy_eps: float = _ENERGY_EPS


@dataclass(frozen=True)
class ScientificThresholds:
    """Declared scientific acceptance thresholds fixed from held-out calibration."""

    max_growth_ratio: float = 3.0
    max_nmse: float = 1.25
    max_spectral_error: float | None = None
    min_windows: int = 1
    min_reference_energy: float = 1e-8


@dataclass(frozen=True)
class HorizonEligibility:
    """Result of candidate eligibility determination."""

    status: EligibilityStatus
    reasons: list[str] = field(default_factory=list)
    passed: bool = False


@dataclass(frozen=True)
class HorizonMetadata:
    """Provenance and configuration accompanying Control Horizon evaluation."""

    checkpoint: str = ""
    data_partition: str = "val"
    preprocessing: dict[str, Any] = field(default_factory=dict)
    sample_rate: float = 50.0
    horizon: int = 75
    dt: float = 0.02
    reference_identity: str = ""


@dataclass
class ControlHorizonReport:
    """Complete evaluation report for a Predictor over the complete Control Horizon."""

    metadata: HorizonMetadata
    conditioning: str
    n_evaluated_windows: int
    n_excluded_windows: int
    total_windows: int
    lookahead_steps: FloatArray
    lookahead_seconds: FloatArray
    rmse: FloatArray
    nmse: FloatArray
    persistence_rmse: FloatArray
    persistence_nmse: FloatArray
    amplitude_growth: FloatArray
    energy_growth: FloatArray
    spectral_error: FloatArray | None
    one_step_metrics: dict[str, float | None]
    terminal_metrics: dict[str, float | None]
    max_growth_ratio: float
    reference_energy: float
    eligibility: HorizonEligibility
    reconstruction: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize report to a dictionary with array values converted to lists."""
        return {
            "metadata": asdict(self.metadata),
            "conditioning": self.conditioning,
            "n_evaluated_windows": self.n_evaluated_windows,
            "n_excluded_windows": self.n_excluded_windows,
            "total_windows": self.total_windows,
            "lookahead_steps": self.lookahead_steps.tolist(),
            "lookahead_seconds": self.lookahead_seconds.tolist(),
            "rmse": self.rmse.tolist(),
            "nmse": self.nmse.tolist(),
            "persistence_rmse": self.persistence_rmse.tolist(),
            "persistence_nmse": self.persistence_nmse.tolist(),
            "amplitude_growth": self.amplitude_growth.tolist(),
            "energy_growth": self.energy_growth.tolist(),
            "spectral_error": self.spectral_error.tolist() if self.spectral_error is not None else None,
            "one_step_metrics": self.one_step_metrics,
            "terminal_metrics": self.terminal_metrics,
            "max_growth_ratio": self.max_growth_ratio,
            "reference_energy": self.reference_energy,
            "eligibility": {
                "status": self.eligibility.status.value,
                "reasons": self.eligibility.reasons,
                "passed": self.eligibility.passed,
            },
            "reconstruction": self.reconstruction,
        }


def evaluate_control_horizon(  # noqa: C901, PLR0912, PLR0913, PLR0915
    model: InferencePredictor,
    recordings: list[tuple[FloatArray, FloatArray]] | list[Run] | Run | Path | tuple[FloatArray, FloatArray],
    horizon: int,
    dt: float | None = None,
    *,
    conditioning: Literal["actual", "planned", "zero"] = "actual",
    geometry: StftGeometry | None = None,
    stride: int = 1,
    start: int | None = None,
    thresholds: ScientificThresholds | None = None,
    tolerances: NumericalTolerances | None = None,
    checkpoint_id: str = "",
    data_partition: str = "val",
    reference_identity: str = "",
) -> ControlHorizonReport:
    """Evaluate Predictor accuracy and stability over the complete Control Horizon."""
    thresh = thresholds or ScientificThresholds()
    tol = tolerances or NumericalTolerances()
    sample_dt = float(getattr(model, "dt", 0.02) if dt is None else dt)
    geom = getattr(model, "geometry", None) if geometry is None else geometry

    runs_or_trajs: list[Run | tuple[FloatArray, FloatArray]]
    if isinstance(recordings, Run):
        runs_or_trajs = [recordings]
    elif isinstance(recordings, (str, Path)):
        path = Path(recordings)
        if (path / "config.yaml").is_file() and (path / "log.npz").is_file():
            runs_or_trajs = [Run.load(path)]
        elif path.is_dir():
            npz_files = sorted(path.glob("*.npz"))
            runs_or_trajs = [
                load_trajectory(str(f), None, getattr(model, "downsample", 1), sample_dt) for f in npz_files
            ]
        else:
            runs_or_trajs = [load_trajectory(str(path), None, getattr(model, "downsample", 1), sample_dt)]
    elif isinstance(recordings, tuple) and len(recordings) == 2 and isinstance(recordings[0], np.ndarray):  # noqa: PLR2004 -- (u, y) pair
        runs_or_trajs = [recordings]
    elif isinstance(recordings, list):
        runs_or_trajs = list(recordings)
    else:
        msg = f"Unsupported recordings format: {type(recordings)}"
        raise TypeError(msg)

    n_evaluated_windows = 0
    n_excluded_windows = 0
    all_pred: list[FloatArray] = []
    all_true: list[FloatArray] = []
    all_persist: list[FloatArray] = []
    all_reconstruction_errors: list[float] = []

    k = model.priming_steps
    grid_start = k if start is None else start

    for item in runs_or_trajs:
        planned_u: FloatArray | None = None
        predicted_y: FloatArray | None = None
        if isinstance(item, Run):
            times, u = item.signal("controller", "u")
            y = item.measurements()
            planned_u = item.arrays.get("controller.planned_u")
            predicted_y = item.arrays.get("controller.predicted_y")
        else:
            u, y = item
            u = np.asarray(u, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
            times = np.arange(len(y), dtype=np.float64) * sample_dt

        n_samples = len(y)
        y_flat = y.reshape(n_samples, -1) if y.ndim > 2 else y  # noqa: PLR2004 -- 2D (time, channels) array

        state = model.initial_state()
        previous = np.zeros(model.m, dtype=np.float64)

        for i in range(n_samples):
            state = model.absorb(state, y_flat[i], previous)
            previous = u[i]

            if i < grid_start:
                continue
            if (i - grid_start) % stride != 0:
                continue

            if i + horizon >= n_samples:
                n_excluded_windows += 1
                continue

            if conditioning == "actual":
                u_future = u[i : i + horizon]
            elif conditioning == "zero":
                u_future = np.zeros((horizon, model.m), dtype=np.float64)
            elif conditioning == "planned":
                if planned_u is None or i >= len(planned_u):
                    n_excluded_windows += 1
                    continue
                u_future = planned_u[i]
            else:
                msg = f"Unknown conditioning mode: {conditioning!r}"
                raise ValueError(msg)

            pred_raw = np.asarray(predict(model, jnp.asarray(state), jnp.asarray(u_future), float(times[i]), sample_dt))
            pred_future = pred_raw[1:]
            true_future = y_flat[i + 1 : i + 1 + horizon]
            persist_future = np.broadcast_to(y_flat[i : i + 1], (horizon, y_flat.shape[1]))

            all_pred.append(pred_future)
            all_true.append(true_future)
            all_persist.append(persist_future)
            n_evaluated_windows += 1

            if conditioning == "planned" and predicted_y is not None and i < len(predicted_y):
                rec_err = float(np.max(np.abs(pred_raw - predicted_y[i])))
                all_reconstruction_errors.append(rec_err)

    total_windows = n_evaluated_windows + n_excluded_windows
    lookahead_steps = np.arange(1, horizon + 1, dtype=np.float64)
    lookahead_seconds = np.asarray(lookahead_steps * sample_dt, dtype=np.float64)

    spectral_error: FloatArray | None = None
    if n_evaluated_windows == 0:
        rmse = np.zeros(horizon, dtype=np.float64)
        nmse_vals = np.full(horizon, np.nan, dtype=np.float64)
        persistence_rmse = np.zeros(horizon, dtype=np.float64)
        persistence_nmse = np.full(horizon, np.nan, dtype=np.float64)
        amplitude_growth = np.full(horizon, np.nan, dtype=np.float64)
        energy_growth = np.full(horizon, np.nan, dtype=np.float64)
        max_growth_ratio = float("nan")
        mean_ref_energy = 0.0
    else:
        y_preds = np.stack(all_pred, axis=0)
        y_trues = np.stack(all_true, axis=0)
        y_persists = np.stack(all_persist, axis=0)
        w_count, h_count, p_dim = y_preds.shape

        sq_err = ((y_preds - y_trues) ** 2).sum(axis=(0, 2))
        true_power = (y_trues**2).sum(axis=(0, 2))
        pred_power = (y_preds**2).sum(axis=(0, 2))
        persist_sq_err = ((y_persists - y_trues) ** 2).sum(axis=(0, 2))

        mean_ref_energy = float(true_power.sum() / (w_count * h_count * p_dim))
        rmse = np.sqrt(sq_err / (w_count * p_dim))
        persistence_rmse = np.sqrt(persist_sq_err / (w_count * p_dim))

        if mean_ref_energy <= thresh.min_reference_energy:
            nmse_vals = np.full(horizon, np.nan, dtype=np.float64)
            persistence_nmse = np.full(horizon, np.nan, dtype=np.float64)
            amplitude_growth = np.full(horizon, np.nan, dtype=np.float64)
            energy_growth = np.full(horizon, np.nan, dtype=np.float64)
            max_growth_ratio = float("nan")
        else:
            nmse_vals = np.divide(sq_err, true_power, out=np.full_like(sq_err, np.nan), where=true_power > 0)
            persistence_nmse = np.divide(
                persist_sq_err, true_power, out=np.full_like(persist_sq_err, np.nan), where=true_power > 0
            )
            pred_rms = np.sqrt(pred_power / (w_count * p_dim))
            true_rms = np.sqrt(true_power / (w_count * p_dim))
            amplitude_growth = np.divide(pred_rms, true_rms, out=np.full_like(pred_rms, np.nan), where=true_rms > 0)
            energy_growth = np.divide(
                pred_power, true_power, out=np.full_like(pred_power, np.nan), where=true_power > 0
            )
            finite_growth = amplitude_growth[np.isfinite(amplitude_growth)]
            max_growth_ratio = float(np.max(finite_growth)) if finite_growth.size > 0 else float("nan")

        if isinstance(model, (ObservableMLPModel, ObservableCNNModel)):
            spectral_error = ((y_preds - y_trues) ** 2).mean(axis=(0, 2))
        elif geom is not None:
            fs = 1.0 / sample_dt
            support = geom.sample_support_steps(fs)
            if horizon >= support:
                pred_frames = [compute_log_power_frames(y_preds[w], geom, fs=fs) for w in range(w_count)]
                true_frames = [compute_log_power_frames(y_trues[w], geom, fs=fs) for w in range(w_count)]
                if pred_frames[0].shape[0] > 0:
                    pf_arr = np.stack(pred_frames, axis=0)
                    tf_arr = np.stack(true_frames, axis=0)
                    spectral_error = ((pf_arr - tf_arr) ** 2).mean(axis=(0, 2, 3))

    one_step_metrics: dict[str, float | None] = {
        "rmse": float(rmse[0]) if n_evaluated_windows > 0 else float("nan"),
        "nmse": float(nmse_vals[0]) if n_evaluated_windows > 0 else float("nan"),
        "persistence_rmse": float(persistence_rmse[0]) if n_evaluated_windows > 0 else float("nan"),
        "persistence_nmse": float(persistence_nmse[0]) if n_evaluated_windows > 0 else float("nan"),
        "amplitude_growth": float(amplitude_growth[0]) if n_evaluated_windows > 0 else float("nan"),
        "growth_ratio": float(amplitude_growth[0]) if n_evaluated_windows > 0 else float("nan"),
        "energy_growth": float(energy_growth[0]) if n_evaluated_windows > 0 else float("nan"),
        "spectral_error": float(spectral_error[0]) if spectral_error is not None and len(spectral_error) > 0 else None,
    }
    terminal_metrics: dict[str, float | None] = {
        "rmse": float(rmse[-1]) if n_evaluated_windows > 0 else float("nan"),
        "nmse": float(nmse_vals[-1]) if n_evaluated_windows > 0 else float("nan"),
        "persistence_rmse": float(persistence_rmse[-1]) if n_evaluated_windows > 0 else float("nan"),
        "persistence_nmse": float(persistence_nmse[-1]) if n_evaluated_windows > 0 else float("nan"),
        "amplitude_growth": float(amplitude_growth[-1]) if n_evaluated_windows > 0 else float("nan"),
        "growth_ratio": float(amplitude_growth[-1]) if n_evaluated_windows > 0 else float("nan"),
        "energy_growth": float(energy_growth[-1]) if n_evaluated_windows > 0 else float("nan"),
        "spectral_error": float(spectral_error[-1]) if spectral_error is not None and len(spectral_error) > 0 else None,
    }

    reconstruction: dict[str, Any] | None = None
    if all_reconstruction_errors:
        max_rec_err = float(np.max(all_reconstruction_errors))
        reconstruction = {
            "max_error": max_rec_err,
            "is_reconstructed": bool(max_rec_err <= tol.reconstruction_atol),
            "atol": tol.reconstruction_atol,
        }

    reasons: list[str] = []
    status = EligibilityStatus.PASS

    if n_evaluated_windows < thresh.min_windows:
        status = EligibilityStatus.INSUFFICIENT_EVIDENCE
        reasons.append(
            f"Insufficient horizon coverage: {n_evaluated_windows} valid windows evaluated "
            f"(minimum required: {thresh.min_windows})"
        )
    elif mean_ref_energy <= thresh.min_reference_energy:
        status = EligibilityStatus.INSUFFICIENT_EVIDENCE
        reasons.append(
            f"Near-zero reference energy ({mean_ref_energy:.2e} <= {thresh.min_reference_energy:.2e}): "
            "normalized error and growth ratios are undefined"
        )
    else:
        if conditioning == "planned" and reconstruction is not None and not reconstruction["is_reconstructed"]:
            status = EligibilityStatus.FAIL
            reasons.append(
                f"Saved-plan forecast reconstruction failed: max error {reconstruction['max_error']:.2e} "
                f"exceeds tolerance {reconstruction['atol']:.2e}"
            )
        if conditioning in ("actual", "zero"):
            if np.isnan(max_growth_ratio) or max_growth_ratio > thresh.max_growth_ratio:
                status = EligibilityStatus.FAIL
                reasons.append(
                    f"Amplitude growth ratio {max_growth_ratio:.2f} exceeds threshold {thresh.max_growth_ratio:.2f}"
                )
            if np.isnan(terminal_metrics["nmse"]) or terminal_metrics["nmse"] > thresh.max_nmse:
                status = EligibilityStatus.FAIL
                reasons.append(f"Terminal NMSE {terminal_metrics['nmse']:.2f} exceeds threshold {thresh.max_nmse:.2f}")
            if (
                spectral_error is not None
                and thresh.max_spectral_error is not None
                and float(np.mean(spectral_error)) > thresh.max_spectral_error
            ):
                status = EligibilityStatus.FAIL
                reasons.append(
                    f"Spectral error {float(np.mean(spectral_error)):.2f} exceeds threshold {thresh.max_spectral_error:.2f}"
                )

    passed = status == EligibilityStatus.PASS
    eligibility = HorizonEligibility(status=status, reasons=reasons, passed=passed)

    metadata = HorizonMetadata(
        checkpoint=checkpoint_id or type(model).__name__,
        data_partition=data_partition,
        preprocessing={"downsample": getattr(model, "downsample", 1), "dt": sample_dt},
        sample_rate=1.0 / sample_dt,
        horizon=horizon,
        dt=sample_dt,
        reference_identity=reference_identity,
    )

    return ControlHorizonReport(
        metadata=metadata,
        conditioning=conditioning,
        n_evaluated_windows=n_evaluated_windows,
        n_excluded_windows=n_excluded_windows,
        total_windows=total_windows,
        lookahead_steps=lookahead_steps,
        lookahead_seconds=lookahead_seconds,
        rmse=rmse,
        nmse=nmse_vals,
        persistence_rmse=persistence_rmse,
        persistence_nmse=persistence_nmse,
        amplitude_growth=amplitude_growth,
        energy_growth=energy_growth,
        spectral_error=spectral_error,
        one_step_metrics=one_step_metrics,
        terminal_metrics=terminal_metrics,
        max_growth_ratio=max_growth_ratio,
        reference_energy=mean_ref_energy,
        eligibility=eligibility,
        reconstruction=reconstruction,
    )


def check_candidate_eligibility(  # noqa: PLR0913 -- model, trajectories, horizon, dt, thresholds, tolerances
    model_or_config: InferencePredictor | dict[str, Any],
    trajectories: list[tuple[FloatArray, FloatArray]]
    | list[Run]
    | Run
    | Path
    | tuple[FloatArray, FloatArray]
    | None = None,
    *,
    horizon: int | None = None,
    dt: float | None = None,
    thresholds: ScientificThresholds | None = None,
    tolerances: NumericalTolerances | None = None,
) -> HorizonEligibility:
    """Check candidate eligibility over the Control Horizon before closed-loop deployment."""
    thresh = thresholds or ScientificThresholds()
    tol = tolerances or NumericalTolerances()
    if trajectories is None:
        return HorizonEligibility(
            status=EligibilityStatus.INSUFFICIENT_EVIDENCE,
            reasons=["Insufficient horizon coverage or unavailable calibration: no validation data provided"],
            passed=False,
        )

    if isinstance(model_or_config, dict):
        problem = (
            model_or_config["controller"]["problem"]
            if "controller" in model_or_config and "problem" in model_or_config["controller"]
            else model_or_config
        )
        artifact_path = problem.get("artifact")
        if not artifact_path:
            return HorizonEligibility(
                status=EligibilityStatus.INSUFFICIENT_EVIDENCE,
                reasons=["Config does not specify a Predictor artifact to evaluate"],
                passed=False,
            )
        model = InferencePredictor.load(Path(artifact_path))
        eval_horizon = int(problem.get("horizon", 75)) if horizon is None else horizon
        eval_dt = float(problem.get("dt", getattr(model, "dt", 0.02))) if dt is None else dt
    else:
        model = model_or_config
        eval_horizon = getattr(model, "horizon", 75) if horizon is None else horizon
        eval_dt = float(getattr(model, "dt", 0.02)) if dt is None else dt

    report = evaluate_control_horizon(
        model,
        trajectories,
        horizon=eval_horizon,
        dt=eval_dt,
        conditioning="actual",
        thresholds=thresh,
        tolerances=tol,
    )
    return report.eligibility
