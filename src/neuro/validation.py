from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from neuro.checkpoint import ObservableCheckpoint, load_any
from neuro.observable import envelope_log_reference, load_envelope
from neuro.provenance import plant_fingerprint
from neuro.spectral import PsdEnvelope

if TYPE_CHECKING:
    from collections.abc import Mapping

    from neuro.provenance import TrainingProvenance

_FILTERING_ESTIMATORS = frozenset({"neuro.filtering.AntiAliasEstimator", "neuro.filtering.LowPassEstimator"})
_PREDICTIVE_CONTROLLERS = frozenset(
    {"neuro.control.nonlinear_mpc.MPCController", "neuro.control.linear_mpc.LinearMPCController"}
)
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


def _check_psd_reference(controller: Mapping[str, Any], controller_dt: float) -> None:
    """Validate spectral reference envelope and geometry against controller configuration."""
    psd_ref_path = controller.get("psd_ref")
    if psd_ref_path is None:
        return

    psd_path = Path(psd_ref_path)
    if not psd_path.exists():
        msg = f"spectral reference envelope not found: {psd_path}"
        raise ConfigConsistencyError(msg)
    envelope = PsdEnvelope.load(psd_path)

    if not math.isclose(controller_dt, 1.0 / envelope.fs, rel_tol=_REL_TOL):
        msg = (
            f"controller.dt ({controller_dt}) must match spectral reference dt "
            f"({1.0 / envelope.fs:g} s from fs={envelope.fs:g} Hz)."
        )
        raise ConfigConsistencyError(msg)

    for key, declared, actual in (
        ("psd_window_s", controller.get("psd_window_s"), envelope.window),
        ("psd_hop_s", controller.get("psd_hop_s"), envelope.hop),
    ):
        if declared is None:
            continue
        steps = round(float(declared) / controller_dt)
        if steps != actual:
            msg = (
                f"controller.{key} ({declared} s = {steps} steps) does not match the spectral "
                f"reference geometry ({actual} steps)."
            )
            raise ConfigConsistencyError(msg)


def _check_observable_geometry(ckpt: ObservableCheckpoint, controller: Mapping[str, Any]) -> None:
    """Require the geometry recorded in an observable checkpoint to match the reference envelope's.

    :func:`_check_psd_reference` only sees what the YAML declares, not what the predictor was fit on.
    """
    psd_ref_path = controller.get("psd_ref")
    if psd_ref_path is None:
        return

    geometry, fs = ckpt.geometry, ckpt.fs
    try:
        envelope = load_envelope(psd_ref_path, geometry)
    except (ValueError, TypeError) as exc:
        msg = f"the envelope at {psd_ref_path} does not carry a {geometry.kind} reference: {exc}"
        raise ConfigConsistencyError(msg) from exc

    for field, recorded, measured in (
        ("fs", fs, envelope.fs),
        ("segment length", geometry.segment_steps(fs), envelope.window),
        ("hop", geometry.hop_steps(fs), envelope.hop),
        ("channel count", ckpt.n_channels, envelope.power.shape[0]),
    ):
        if not math.isclose(recorded, measured, rel_tol=_REL_TOL):
            msg = (
                f"the observable predictor was trained at {field} {recorded} but the reference "
                f"envelope at {psd_ref_path} was measured at {measured}; the Cost would score a "
                f"forecast against a reference built on another grid."
            )
            raise ConfigConsistencyError(msg)

    # The envelope records no bin range, pooling or Frame Kernel width, so those are checked
    # indirectly: reducing it onto the checkpoint's grid must yield exactly the readout's width.
    try:
        reference = envelope_log_reference(envelope, geometry, fs)
    except (IndexError, TypeError) as exc:
        msg = f"the envelope at {psd_ref_path} cannot be reduced onto the checkpoint's frame grid: {exc}"
        raise ConfigConsistencyError(msg) from exc
    if reference.shape != (ckpt.n_channels, ckpt.n_values):
        msg = (
            f"the checkpoint's scored bin range and pooling leave {reference.shape} reference values "
            f"but its readout emits {(ckpt.n_channels, ckpt.n_values)}."
        )
        raise ConfigConsistencyError(msg)


def _check_predictor(config: Mapping[str, Any]) -> None:
    """Require the loop's rate, anti-alias filter, horizon, plant and current range to match the predictor's."""
    controller = config["controller"]
    if controller["class_path"] not in _PREDICTIVE_CONTROLLERS:
        return

    ckpt = load_any(controller["artifact"])
    provenance = ckpt.provenance

    controller_dt = float(controller["dt"])
    if not math.isclose(controller_dt, ckpt.dt, rel_tol=_REL_TOL):
        msg = (
            f"controller.dt ({controller_dt}) must equal the predictor's native dt ({ckpt.dt}), which is "
            f"dynamics.dt x {ckpt.downsample}: the MPC steps the model once per control step."
        )
        raise ConfigConsistencyError(msg)

    online = _estimator_cutoff_hz(config["estimator"])
    offline = _training_cutoff_hz(ckpt.dt, ckpt.downsample, provenance)
    if not _cutoffs_agree(online, offline):
        msg = (
            f"the estimator applies {_describe(online)} but the predictor was identified on data decimated "
            f"with {_describe(offline)}; the loop would feed the model a signal it was not fit on."
        )
        raise ConfigConsistencyError(msg)

    horizon = controller.get("horizon")
    if horizon is not None and int(horizon) > ckpt.horizon:
        warnings.warn(
            f"controller.horizon ({horizon}) exceeds the predictor's trained horizon ({ckpt.horizon}); "
            f"the MPC would cost a free run longer than any the model was ever fit against.",
            stacklevel=2,
        )

    _check_psd_reference(controller, controller_dt)
    if isinstance(ckpt, ObservableCheckpoint):
        _check_observable_geometry(ckpt, controller)

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
