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
from neuro.spectral import HealthyReference, compute_log_power_frames

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
    """Batch causal Rollouts from measurements through t0 and targets starting at t0 + 1.

    Control history ends at t0 - 1; the candidate Current at t0 acts on the first prediction.
    Waveform samples and Observable Frames retain their output axes.
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
    """Criteria fixed on held-out calibration, whose identity is required before qualification."""

    max_growth_ratio: float = 3.0
    max_nmse: float = 1.25
    max_spectral_error: float | None = None
    min_windows: int = 1
    min_reference_energy: float = 1e-8
    calibration_identity: str = ""


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
    scientific_thresholds: ScientificThresholds | None = None
    growth_reference: str = "recorded_future"


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


@dataclass
class _HorizonWindows:
    """Aligned predictions, recorded futures, and causal histories for one evaluation."""

    predictions: list[FloatArray] = field(default_factory=list)
    targets: list[FloatArray] = field(default_factory=list)
    histories: list[FloatArray] = field(default_factory=list)
    reconstruction_errors: list[float] = field(default_factory=list)
    reconstructed: list[bool] = field(default_factory=list)
    excluded: int = 0


def _load_horizon_recordings(
    recordings: list[tuple[FloatArray, FloatArray]] | list[Run] | Run | Path | tuple[FloatArray, FloatArray],
    model: InferencePredictor,
    dt: float,
) -> list[Run | tuple[FloatArray, FloatArray]]:
    """Load recordings using their raw sample period before checkpoint decimation."""
    if isinstance(recordings, Run):
        return [recordings]
    if isinstance(recordings, Path):
        if (recordings / "config.yaml").is_file() and (recordings / "log.npz").is_file():
            return [Run.load(recordings)]
        paths = sorted(recordings.glob("*.npz")) if recordings.is_dir() else [recordings]
        downsample = getattr(model, "downsample", 1)
        return [load_trajectory(str(path), None, downsample, dt / downsample) for path in paths]
    if isinstance(recordings, tuple):
        return [recordings]
    return list(recordings)


def _future_controls(
    conditioning: str, controls: FloatArray, planned: FloatArray | None, index: int, horizon: int
) -> FloatArray | None:
    """Select the declared input sequence, excluding missing or incomplete saved plans."""
    if conditioning == "actual":
        return controls[index : index + horizon]
    if conditioning == "zero":
        return np.zeros((horizon, controls.shape[1]), dtype=np.float64)
    if planned is None or index >= len(planned):
        return None
    plan = planned[index]
    return plan if len(plan) == horizon and np.isfinite(plan).all() else None


def _collect_horizon_windows(  # noqa: PLR0913 -- model, recordings, and declared evaluation grid
    model: InferencePredictor,
    recordings: list[Run | tuple[FloatArray, FloatArray]],
    horizon: int,
    dt: float,
    *,
    conditioning: str,
    stride: int,
    start: int | None,
    history_steps: int,
    tolerances: NumericalTolerances,
) -> _HorizonWindows:
    """Absorb causal measurements and retain only complete, sufficiently primed futures."""
    windows = _HorizonWindows()
    grid_start = model.priming_steps if start is None else start
    for item in recordings:
        if isinstance(item, Run):
            times, controls = item.signal("controller", "u")
            measurements = item.measurements()
            planned = item.arrays.get("controller.planned_u")
            saved = item.arrays.get("controller.predicted_y")
        else:
            controls, measurements = item
            times = np.arange(len(measurements), dtype=np.float64) * dt
            planned, saved = None, None
        y = np.asarray(measurements, dtype=np.float64).reshape(len(measurements), -1)
        state = model.initial_state()
        previous = np.zeros(model.m, dtype=np.float64)
        for index in range(len(y)):
            state = model.absorb(state, y[index], previous)
            previous = controls[index]
            if index < grid_start or (index - grid_start) % stride:
                continue
            future = _future_controls(conditioning, controls, planned, index, horizon)
            if index + horizon >= len(y) or index + 1 < history_steps or not model.is_ready(state) or future is None:
                windows.excluded += 1
                continue
            prediction = np.asarray(predict(model, jnp.asarray(state), jnp.asarray(future), float(times[index]), dt))
            windows.predictions.append(prediction[1:])
            windows.targets.append(y[index + 1 : index + horizon + 1])
            windows.histories.append(y[index - history_steps + 1 : index + 1])
            if conditioning == "planned" and saved is not None and index < len(saved):
                windows.reconstruction_errors.append(float(np.max(np.abs(prediction - saved[index]))))
                windows.reconstructed.append(
                    bool(
                        np.allclose(
                            prediction,
                            saved[index],
                            atol=tolerances.reconstruction_atol,
                            rtol=tolerances.reconstruction_rtol,
                        )
                    )
                )
    return windows


def _horizon_metrics(
    windows: _HorizonWindows, horizon: int, conditioning: str, min_energy: float
) -> tuple[dict[str, FloatArray], float]:
    """Score matched accuracy, or counterfactual growth relative to measured history."""
    metrics = {
        name: np.full(horizon, np.nan)
        for name in ("rmse", "nmse", "persistence_rmse", "persistence_nmse", "amplitude_growth", "energy_growth")
    }
    if not windows.predictions:
        return metrics, 0.0
    predicted = np.stack(windows.predictions)
    recorded = np.stack(windows.targets)
    history = np.stack(windows.histories)
    pred_power = np.mean(predicted**2, axis=(0, 2))
    ref_power = np.mean(recorded**2, axis=(0, 2)) if conditioning == "actual" else np.full(horizon, np.mean(history**2))
    reference_energy = float(np.mean(ref_power))
    valid = ref_power > min_energy
    metrics["energy_growth"] = np.divide(pred_power, ref_power, out=np.full(horizon, np.nan), where=valid)
    metrics["amplitude_growth"] = np.sqrt(metrics["energy_growth"])
    if conditioning == "actual":
        errors = np.mean((predicted - recorded) ** 2, axis=(0, 2))
        persistence_errors = np.mean((history[:, -1:, :] - recorded) ** 2, axis=(0, 2))
        metrics["rmse"] = np.sqrt(errors)
        metrics["persistence_rmse"] = np.sqrt(persistence_errors)
        metrics["nmse"] = np.divide(errors, ref_power, out=np.full(horizon, np.nan), where=valid)
        metrics["persistence_nmse"] = np.divide(
            persistence_errors, ref_power, out=np.full(horizon, np.nan), where=valid
        )
    return metrics, reference_energy


def _spectral_errors(
    windows: _HorizonWindows, model: InferencePredictor, geometry: StftGeometry | None, fs: float
) -> FloatArray | None:
    """Score causal Frames ending at every predicted sample, with the configured Frame Kernel."""
    if not windows.predictions:
        return None
    if isinstance(model, (ObservableMLPModel, ObservableCNNModel)):
        return np.mean((np.stack(windows.predictions) - np.stack(windows.targets)) ** 2, axis=(0, 2))
    if geometry is None:
        return None
    support = geometry.sample_support_steps(fs)
    errors = []
    for prediction, target, history in zip(windows.predictions, windows.targets, windows.histories, strict=True):
        predicted = np.concatenate([history[-(support - 1) :], prediction])
        recorded = np.concatenate([history[-(support - 1) :], target])
        errors.append(
            [
                float(
                    np.mean(
                        (
                            compute_log_power_frames(predicted[i : i + support], geometry, fs=fs)
                            - compute_log_power_frames(recorded[i : i + support], geometry, fs=fs)
                        )
                        ** 2
                    )
                )
                for i in range(len(prediction))
            ]
        )
    return np.mean(np.asarray(errors), axis=0)


def _metric_snapshot(
    metrics: dict[str, FloatArray], spectral: FloatArray | None, index: int
) -> dict[str, float | None]:
    """Expose one lookahead, representing unavailable measurements as None."""
    result = {name: float(values[index]) if np.isfinite(values[index]) else None for name, values in metrics.items()}
    result["growth_ratio"] = result["amplitude_growth"]
    result["spectral_error"] = float(spectral[index]) if spectral is not None else None
    return result


def _horizon_eligibility(  # noqa: PLR0913 -- declared criteria and their required independent measurements
    thresholds: ScientificThresholds | None,
    metrics: dict[str, FloatArray],
    spectral: FloatArray | None,
    n_windows: int,
    reference_energy: float,
    *,
    conditioning: str,
) -> HorizonEligibility:
    """Require calibrated criteria and every requested measurement before checking acceptance limits."""
    criteria = thresholds or ScientificThresholds()
    missing = []
    if thresholds is None or not criteria.calibration_identity.strip():
        missing.append("Unavailable calibration: supply criteria fixed on held-out data and their calibration identity")
    if conditioning != "actual":
        missing.append("Counterfactual diagnostics cannot establish actual-input candidate accuracy")
    if n_windows < criteria.min_windows:
        missing.append(f"Insufficient horizon coverage: {n_windows} valid windows, minimum {criteria.min_windows}")
    if reference_energy <= criteria.min_reference_energy:
        missing.append("Near-zero reference energy: normalized error and growth ratios are undefined")
    if criteria.max_spectral_error is not None and (spectral is None or not np.isfinite(spectral).all()):
        missing.append("Required spectral evidence is unavailable over the complete Control Horizon")
    if missing:
        return HorizonEligibility(EligibilityStatus.INSUFFICIENT_EVIDENCE, missing)
    reasons = []
    growth = metrics["amplitude_growth"]
    if not np.isfinite(growth).all() or np.max(growth) > criteria.max_growth_ratio:
        reasons.append(f"Amplitude growth ratio exceeds threshold {criteria.max_growth_ratio:.2f} or is undefined")
    terminal_nmse = metrics["nmse"][-1]
    if not np.isfinite(terminal_nmse) or terminal_nmse > criteria.max_nmse:
        reasons.append(f"Terminal NMSE {terminal_nmse:.2f} exceeds threshold {criteria.max_nmse:.2f} or is undefined")
    if (
        spectral is not None
        and criteria.max_spectral_error is not None
        and np.mean(spectral) > criteria.max_spectral_error
    ):
        reasons.append(f"Spectral error exceeds threshold {criteria.max_spectral_error:.2f}")
    return HorizonEligibility(
        EligibilityStatus.FAIL if reasons else EligibilityStatus.PASS, reasons, passed=not reasons
    )


def evaluate_control_horizon(  # noqa: PLR0913 -- evaluation inputs, geometry, and provenance are independent
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
    """Report causal lookahead metrics and qualify only calibrated actual-input evaluations.

    Spectral Frames end at every lookahead, including the first and terminal predictions.
    Planned and zero-input reports withhold Plant accuracy and measure growth relative to
    observed history. Their reconstruction or growth diagnostics never qualify a candidate.
    """
    sample_dt = float(getattr(model, "dt", 0.02) if dt is None else dt)
    geom = getattr(model, "geometry", None) if geometry is None else geometry
    tol = tolerances or NumericalTolerances()
    observable = isinstance(model, (ObservableMLPModel, ObservableCNNModel))
    history_steps = max(1, model.priming_steps)
    if geom is not None and not observable:
        history_steps = max(history_steps, geom.sample_support_steps(1 / sample_dt) - 1)
    windows = _collect_horizon_windows(
        model,
        _load_horizon_recordings(recordings, model, sample_dt),
        horizon,
        sample_dt,
        conditioning=conditioning,
        stride=stride,
        start=start,
        history_steps=history_steps,
        tolerances=tol,
    )
    metrics, reference_energy = _horizon_metrics(
        windows, horizon, conditioning, (thresholds or ScientificThresholds()).min_reference_energy
    )
    spectral = _spectral_errors(windows, model, geom, 1 / sample_dt) if conditioning == "actual" else None
    reconstruction = None
    if windows.reconstruction_errors:
        reconstruction = {
            "max_error": float(np.max(windows.reconstruction_errors)),
            "is_reconstructed": all(windows.reconstructed),
            "atol": tol.reconstruction_atol,
            "rtol": tol.reconstruction_rtol,
        }
    steps = np.arange(1, horizon + 1, dtype=np.float64)
    return ControlHorizonReport(
        metadata=HorizonMetadata(
            checkpoint=checkpoint_id or type(model).__name__,
            data_partition=data_partition,
            preprocessing={"downsample": getattr(model, "downsample", 1), "dt": sample_dt},
            sample_rate=1 / sample_dt,
            horizon=horizon,
            dt=sample_dt,
            reference_identity=reference_identity,
            scientific_thresholds=thresholds,
            growth_reference="recorded_future" if conditioning == "actual" else "measurement_history",
        ),
        conditioning=conditioning,
        n_evaluated_windows=len(windows.predictions),
        n_excluded_windows=windows.excluded,
        total_windows=len(windows.predictions) + windows.excluded,
        lookahead_steps=steps,
        lookahead_seconds=steps * sample_dt,
        rmse=metrics["rmse"],
        nmse=metrics["nmse"],
        persistence_rmse=metrics["persistence_rmse"],
        persistence_nmse=metrics["persistence_nmse"],
        amplitude_growth=metrics["amplitude_growth"],
        energy_growth=metrics["energy_growth"],
        spectral_error=spectral,
        one_step_metrics=_metric_snapshot(metrics, spectral, 0),
        terminal_metrics=_metric_snapshot(metrics, spectral, -1),
        max_growth_ratio=float(np.max(metrics["amplitude_growth"])),
        reference_energy=reference_energy,
        eligibility=_horizon_eligibility(
            thresholds, metrics, spectral, len(windows.predictions), reference_energy, conditioning=conditioning
        ),
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
    """Check calibrated candidate criteria using the configured reference geometry and controller period."""
    if trajectories is None:
        return HorizonEligibility(
            status=EligibilityStatus.INSUFFICIENT_EVIDENCE,
            reasons=["Insufficient horizon coverage or unavailable calibration: no validation data provided"],
            passed=False,
        )

    geometry = None
    checkpoint_id = ""
    reference_identity = ""
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
        checkpoint_id = str(artifact_path)
        eval_horizon = int(problem.get("horizon", 75)) if horizon is None else horizon
        controller = model_or_config.get("controller", problem)
        eval_dt = float(controller.get("dt", getattr(model, "dt", 0.02))) if dt is None else dt
        reference = problem.get("reference")
        if reference is not None:
            reference_identity = str(reference)
            envelope = HealthyReference.load(reference).observable
            geometry = envelope.geometry if envelope is not None else None
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
        geometry=geometry,
        checkpoint_id=checkpoint_id,
        reference_identity=reference_identity,
        thresholds=thresholds,
        tolerances=tolerances,
    )
    return report.eligibility
