from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from neuro.predictor.checkpoint import load_meta
from neuro.provenance import TrainingProvenance, plant_fingerprint
from neuro.spectral import ObservableEnvelope, PsdEnvelope

if TYPE_CHECKING:
    from collections.abc import Mapping

    from neuro.config import StftGeometry

_FILTERING_ESTIMATORS = frozenset(
    {"neuro.filtering.AntiAliasEstimator", "neuro.filtering.LowPassEstimator", "neuro.filtering.ObservableEstimator"}
)
_PREDICTIVE_CONTROLLERS = frozenset({"neuro.control.mpc.TrajOptMPCController"})
_REL_TOL = 1e-9


class ConfigConsistencyError(ValueError):
    """A simulation config wires components together on terms they do not agree on."""


def _estimator_cutoff_hz(estimator: Mapping[str, Any]) -> float | None:
    """Apply -3 dB cutoff the configured estimator online, or ``None`` if it does not filter."""
    match estimator["class_path"]:
        case "neuro.filtering.AntiAliasEstimator":
            return (1.0 / float(estimator["dt"])) / (2 * int(estimator["downsample"]))
        case "neuro.filtering.LowPassEstimator":
            return float(estimator["cutoff_hz"])
        case "neuro.filtering.ObservableEstimator":
            if "cutoff_hz" in estimator and estimator["cutoff_hz"] is not None:
                return float(estimator["cutoff_hz"])
            return (1.0 / float(estimator["dt"])) / (2 * int(estimator.get("downsample", 1)))
        case _:
            return None


def _training_cutoff_hz(dt: float, downsample: int, provenance: TrainingProvenance) -> float | None:
    """Apply -3 dB cutoff ``load_trajectory`` before striding the predictor's training data."""
    if provenance.cutoff_hz is not None:
        return provenance.cutoff_hz
    return None if downsample == 1 else (downsample / dt) / (2 * downsample)


def _cutoffs_agree(online: float | None, offline: float | None) -> bool:
    """Whether the loop low-passes with the same filter the training data was decimated with."""
    if online is None or offline is None:
        return online is None and offline is None
    return math.isclose(online, offline, rel_tol=_REL_TOL)


def _describe(cutoff_hz: float | None) -> str:
    return "no filter" if cutoff_hz is None else f"{cutoff_hz:g} Hz"


def _check_rates(config: Mapping[str, Any]) -> None:
    """Require a filtering estimator and the sensors feeding it to run at the plant rate."""
    estimator = config["estimator"]
    if estimator["class_path"] not in _FILTERING_ESTIMATORS:
        return

    plant_dt, estimator_dt = float(config["dynamics"]["dt"]), float(estimator["dt"])
    if not math.isclose(estimator_dt, plant_dt, rel_tol=_REL_TOL):
        msg = (
            f"estimator.dt ({estimator_dt}) must equal dynamics.dt ({plant_dt}): the estimator's low-pass "
            f"is designed for the plant rate, and running it slower filters an already-decimated stream."
        )
        raise ConfigConsistencyError(msg)

    raw_sensors = config["sensors"]
    sensors = raw_sensors if isinstance(raw_sensors, list) else [raw_sensors]
    for i, sensor in enumerate(sensors):
        sensor_dt = float(sensor["dt"])
        if not math.isclose(sensor_dt, estimator_dt, rel_tol=_REL_TOL):
            msg = (
                f"sensors[{i}].dt ({sensor_dt}) must equal estimator.dt ({estimator_dt}): a slower sensor "
                f"feeds the low-pass a held staircase rather than the signal it was designed for."
            )
            raise ConfigConsistencyError(msg)


def _model_geometry(raw: Any) -> StftGeometry:  # noqa: ANN401 -- a checkpoint mapping or an already-built geometry
    """Rebuild the Observable geometry a checkpoint or config section carries."""
    from neuro.config import StftGeometry  # noqa: PLC0415 -- neuro.config imports neuro.metrics, which imports this

    return StftGeometry.model_validate(raw)


def _check_geometry_matches(actual: StftGeometry, expected: StftGeometry, what: str) -> None:
    """Raise ``ConfigConsistencyError`` naming every Observable geometry field ``what`` disagrees on."""
    if actual == expected:
        return
    differing = [
        f"{field} ({getattr(actual, field)!r} vs {getattr(expected, field)!r})"
        for field in type(expected).model_fields
        if getattr(actual, field) != getattr(expected, field)
    ]
    msg = f"{what} geometry does not match the predictor checkpoint: {', '.join(differing)}."
    raise ConfigConsistencyError(msg)


def _check_observable_geometry_agreement(
    estimator: Mapping[str, Any],
    model_geom: StftGeometry,
) -> None:
    """Ensure estimator geometry agrees with the predictor checkpoint's recorded geometry."""
    est_geom_raw = estimator.get("geometry")
    if est_geom_raw is None:
        msg = "estimator 'neuro.filtering.ObservableEstimator' requires a 'geometry' section."
        raise ConfigConsistencyError(msg)

    _check_geometry_matches(_model_geometry(est_geom_raw), model_geom, "estimator")


def _check_observable_psd_reference(
    obs_envelope: ObservableEnvelope,
    controller_dt: float,
    meta: Mapping[str, Any] | None,
) -> None:
    """Validate Observable envelope sample rate, channels, and geometry against the model."""
    expected_dt = obs_envelope.geometry.n_hop / obs_envelope.fs
    if not math.isclose(controller_dt, expected_dt, rel_tol=_REL_TOL):
        msg = (
            f"controller.dt ({controller_dt}) must match Observable reference dt "
            f"({expected_dt:g} s from hop={obs_envelope.geometry.n_hop} at fs={obs_envelope.fs:g} Hz)."
        )
        raise ConfigConsistencyError(msg)

    if meta is None:
        return

    n_channels = int(meta["n_channels"])
    if obs_envelope.power.shape[0] != n_channels:
        msg = (
            f"envelope channel count ({obs_envelope.power.shape[0]}) must equal predictor channel count ({n_channels})."
        )
        raise ConfigConsistencyError(msg)

    model_geom = _model_geometry(meta["geometry"])
    model_fs = model_geom.n_hop / float(meta["dt"])
    if not math.isclose(obs_envelope.fs, model_fs, rel_tol=_REL_TOL):
        msg = f"envelope sampling rate ({obs_envelope.fs:g} Hz) must equal predictor sampling rate ({model_fs:g} Hz)."
        raise ConfigConsistencyError(msg)

    _check_geometry_matches(obs_envelope.geometry, model_geom, "envelope")


def _check_waveform_psd_reference(
    envelope: PsdEnvelope,
    controller_dt: float,
    meta: Mapping[str, Any] | None,
) -> None:
    """Validate waveform PSD envelope sample rate and channels against the model."""
    if not math.isclose(controller_dt, 1.0 / envelope.fs, rel_tol=_REL_TOL):
        msg = f"controller.dt ({controller_dt}) must match spectral reference dt ({1.0 / envelope.fs:g} s from fs={envelope.fs:g} Hz)."
        raise ConfigConsistencyError(msg)

    if meta is not None:
        n_channels = int(meta["n_channels"])
        if envelope.power.shape[0] != n_channels:
            msg = (
                f"envelope channel count ({envelope.power.shape[0]}) must equal predictor channel count ({n_channels})."
            )
            raise ConfigConsistencyError(msg)
        model_fs = 1.0 / float(meta["dt"])
        if not math.isclose(envelope.fs, model_fs, rel_tol=_REL_TOL):
            msg = f"envelope sampling rate ({envelope.fs:g} Hz) must equal predictor sampling rate ({model_fs:g} Hz)."
            raise ConfigConsistencyError(msg)


def _envelope_path(problem: Mapping[str, Any], key: str) -> Path | None:
    """Resolve the envelope path ``problem`` spells under ``key``, or ``None`` when it configures none."""
    configured = problem.get(key)
    if configured is None:
        return None

    path = Path(configured)
    if not path.exists():
        msg = f"spectral reference envelope not found: {path}"
        raise ConfigConsistencyError(msg)
    return path


def _check_observable_predictor(
    config: Mapping[str, Any],
    problem: Mapping[str, Any],
    meta: Mapping[str, Any],
    controller_dt: float,
    plant_dt: float,
) -> float:
    """Validate the Observable predictor rates, estimator kind, geometry, and control-support."""
    if "geometry" not in meta:
        msg = f"problem is {problem.get('class_path')} but checkpoint {problem['artifact']!r} carries no Observable geometry."
        raise ConfigConsistencyError(msg)

    downsample = int(meta["downsample"])
    model_geom = _model_geometry(meta["geometry"])
    expected_dt = model_geom.n_hop * downsample * plant_dt
    if not math.isclose(controller_dt, expected_dt, rel_tol=_REL_TOL):
        msg = (
            f"controller.dt ({controller_dt}) must equal hop ({model_geom.n_hop}) x downsample ({downsample}) x "
            f"dynamics.dt ({plant_dt}) = {expected_dt:g} s: exactly one fresh Frame is absorbed per tick."
        )
        raise ConfigConsistencyError(msg)

    estimator = config["estimator"]
    if estimator["class_path"] != "neuro.filtering.ObservableEstimator":
        msg = f"predictor checkpoint {problem['artifact']!r} is an Observable model but estimator is {estimator['class_path']!r}, not 'neuro.filtering.ObservableEstimator'."
        raise ConfigConsistencyError(msg)

    _check_observable_geometry_agreement(estimator, model_geom)

    n_u = int(meta["n_u"])
    min_n_u = model_geom.min_past_controls()
    if n_u < min_n_u:
        msg = (
            f"predictor n_u ({n_u}) violates the control-support rule: must be >= {min_n_u} "
            f"(kernel_width={model_geom.kernel_width}, segment={model_geom.n_segment}, hop={model_geom.n_hop}) "
            f"so the past-control window covers the Frame's sample support."
        )
        raise ConfigConsistencyError(msg)

    return plant_dt * downsample


def _check_waveform_predictor(
    config: Mapping[str, Any],
    problem: Mapping[str, Any],
    controller_dt: float,
    plant_dt: float,
    downsample: int,
) -> float:
    """Validate the waveform predictor rates and estimator kind."""
    expected_dt = downsample * plant_dt
    if not math.isclose(controller_dt, expected_dt, rel_tol=_REL_TOL):
        msg = (
            f"controller.dt ({controller_dt}) must equal the predictor's native dt ({expected_dt:g} s), "
            f"which is dynamics.dt ({plant_dt}) x downsample ({downsample}): the MPC steps the model once "
            "per control step."
        )
        raise ConfigConsistencyError(msg)

    if config["estimator"]["class_path"] == "neuro.filtering.ObservableEstimator":
        msg = f"predictor checkpoint {problem['artifact']!r} is a waveform model but estimator is 'neuro.filtering.ObservableEstimator'."
        raise ConfigConsistencyError(msg)

    return controller_dt


def _check_predictor(config: Mapping[str, Any]) -> None:
    """Require the loop's rate, anti-alias filter, horizon, plant and geometry to match the predictor's."""
    controller = config["controller"]
    if controller["class_path"] not in _PREDICTIVE_CONTROLLERS:
        return

    problem = controller["problem"]
    meta = load_meta(problem["artifact"])
    provenance = TrainingProvenance.from_meta(meta)
    plant_dt = float(config["dynamics"]["dt"])
    downsample = int(meta["downsample"])
    controller_dt = float(controller["dt"])

    is_observable = "geometry" in meta or problem.get("class_path") == "neuro.control.mpc.build_observable_problem"

    if is_observable:
        effective_decimated_dt = _check_observable_predictor(config, problem, meta, controller_dt, plant_dt)
        envelope_path = _envelope_path(problem, "envelope_ref")
        if envelope_path is not None:
            _check_observable_psd_reference(ObservableEnvelope.load(envelope_path), controller_dt, meta)
    else:
        effective_decimated_dt = _check_waveform_predictor(config, problem, controller_dt, plant_dt, downsample)
        envelope_path = _envelope_path(problem, "psd_ref")
        if envelope_path is not None:
            _check_waveform_psd_reference(PsdEnvelope.load(envelope_path), controller_dt, meta)

    online = _estimator_cutoff_hz(config["estimator"])
    offline = _training_cutoff_hz(effective_decimated_dt, downsample, provenance)
    if not _cutoffs_agree(online, offline):
        msg = (
            f"the estimator applies {_describe(online)} but the predictor was identified on data decimated "
            f"with {_describe(offline)}; the loop would feed the model a signal it was not fit on."
        )
        raise ConfigConsistencyError(msg)

    horizon = problem.get("horizon")
    if horizon is not None and int(horizon) > int(meta["horizon"]):
        warnings.warn(
            f"controller.problem.horizon ({horizon}) exceeds the predictor's trained horizon "
            f"({meta['horizon']}); the MPC would cost a free run longer than any the model was ever fit against.",
            stacklevel=2,
        )

    if provenance.plant_fingerprint is not None and provenance.plant_fingerprint != plant_fingerprint(config):
        warnings.warn(
            "the plant this config simulates is not the one the predictor was identified on "
            "(dynamics or sensors differ); the prediction model is off-plant.",
            stacklevel=2,
        )


def validate_simulation_config(config: Mapping[str, Any]) -> None:
    """Check the couplings between a simulation config's components before anything is built.

    Only the couplings that fail *silently* are checked here: a rate or an anti-alias filter that
    disagrees with the predictor's produces a plausible-looking run rather than an exception.
    Mismatched channel and electrode counts are left to the components, which already raise on
    them within the first few steps.

    Raises
    ------
    ConfigConsistencyError
        If the loop's rates, filter or horizon contradict the predictor checkpoint's.
    """
    _check_rates(config)
    _check_predictor(config)
