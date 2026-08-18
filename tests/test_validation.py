from __future__ import annotations

import dataclasses
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import yaml

from neuro.predictor.artifact import MLPArtifact
from neuro.provenance import (
    TrainingProvenance,
    check_excitation_alignment,
    data_plant_fingerprint,
    plant_fingerprint,
)
from neuro.transforms import Standardizer
from neuro.validation import ConfigConsistencyError, validate_simulation_config

if TYPE_CHECKING:
    from pathlib import Path

_PLANT_DT = 1e-4
_DOWNSAMPLE = 200
_N_CHANNELS = 2
_N_CONTROLS = 3


def _plant() -> dict[str, Any]:
    """The plant blocks shared by the training data and the closed-loop config under test."""
    return {
        "dynamics": {"class_path": "neuro.jansen_rit.JansenRitDynamics", "dt": _PLANT_DT, "params": {"K": 0.6}},
        "sensors": {"class_path": "simulate.sensor.GaussianSensor", "dt": _PLANT_DT, "std_dev": 0.0},
    }


def _artifact(tmp_path: Path, provenance: TrainingProvenance) -> Path:
    """Save a tiny MLP artifact carrying ``provenance`` and return its suffix-less stem."""
    rng = np.random.default_rng(0)
    in_size = 2 * _N_CHANNELS + 2 * _N_CONTROLS
    stem = tmp_path / "model"
    MLPArtifact(
        layers=((rng.normal(size=(_N_CHANNELS, in_size)), rng.normal(size=_N_CHANNELS)),),
        activation="relu",
        n_y=2,
        n_u=2,
        horizon=50,
        n_channels=_N_CHANNELS,
        n_controls=_N_CONTROLS,
        dt=_PLANT_DT * _DOWNSAMPLE,
        downsample=_DOWNSAMPLE,
        y_std=Standardizer(center=np.zeros(_N_CHANNELS), scale=np.ones(_N_CHANNELS)),
        u_std=Standardizer(center=np.zeros(_N_CONTROLS), scale=np.ones(_N_CONTROLS)),
        provenance=provenance,
    ).save(stem)
    return stem


@pytest.fixture
def config(tmp_path: Path) -> dict[str, Any]:
    """A closed-loop config that agrees with its predictor on every coupling."""
    cfg = _plant()
    provenance = TrainingProvenance(cutoff_hz=None, plant_fingerprint=plant_fingerprint(cfg))
    return {
        "t_end": 1.0,
        **cfg,
        "estimator": {
            "class_path": "neuro.filtering.AntiAliasEstimator",
            "dt": _PLANT_DT,
            "downsample": _DOWNSAMPLE,
        },
        "controller": {
            "class_path": "neuro.control.MPCController",
            "dt": _PLANT_DT * _DOWNSAMPLE,
            "artifact": str(_artifact(tmp_path, provenance)),
            "horizon": 50,
        },
    }


def test_consistent_config_passes(config: dict[str, Any]) -> None:
    """A config matching its predictor on rate, filter, horizon, plant and current range validates cleanly."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_simulation_config(config)


def test_controller_dt_must_match_predictor(config: dict[str, Any]) -> None:
    """A controller stepping at a rate other than the predictor's native dt is rejected."""
    config["controller"]["dt"] = _PLANT_DT * 100
    with pytest.raises(ConfigConsistencyError, match="native dt"):
        validate_simulation_config(config)


def test_estimator_downsample_must_match_predictor(config: dict[str, Any]) -> None:
    """An anti-alias filter designed for a different decimation is rejected."""
    config["estimator"]["downsample"] = 100
    with pytest.raises(ConfigConsistencyError, match="decimated with"):
        validate_simulation_config(config)


def test_unfiltered_estimator_rejected(config: dict[str, Any]) -> None:
    """Feeding the predictor an unfiltered measurement it was never fit on is rejected."""
    config["estimator"] = {"class_path": "simulate.estimator.IdentityEstimator", "dt": _PLANT_DT}
    with pytest.raises(ConfigConsistencyError, match="no filter"):
        validate_simulation_config(config)


def test_explicit_training_cutoff_must_match(config: dict[str, Any], tmp_path: Path) -> None:
    """A predictor trained behind an explicit cutoff needs that same cutoff online."""
    provenance = TrainingProvenance(cutoff_hz=12.0, plant_fingerprint=plant_fingerprint(_plant()))
    config["controller"]["artifact"] = str(_artifact(tmp_path, provenance))
    with pytest.raises(ConfigConsistencyError, match="12 Hz"):
        validate_simulation_config(config)


def test_estimator_must_run_at_plant_rate(config: dict[str, Any]) -> None:
    """A low-pass running slower than the plant filters an already-decimated stream."""
    config["estimator"]["dt"] = _PLANT_DT * 2
    with pytest.raises(ConfigConsistencyError, match=r"estimator.dt"):
        validate_simulation_config(config)


def test_sensor_must_run_at_estimator_rate(config: dict[str, Any]) -> None:
    """A sensor slower than the low-pass feeds it a held staircase."""
    config["sensors"]["dt"] = _PLANT_DT * 10
    with pytest.raises(ConfigConsistencyError, match=r"sensors\[0\].dt"):
        validate_simulation_config(config)


def test_horizon_beyond_trained_horizon_rejected(config: dict[str, Any]) -> None:
    """The MPC may not cost a free run longer than the predictor was fit against."""
    config["controller"]["horizon"] = 51
    with pytest.raises(ConfigConsistencyError, match=r"trained horizon"):
        validate_simulation_config(config)


def test_plant_mismatch_warns(config: dict[str, Any]) -> None:
    """Deploying onto a plant other than the identified one warns but does not block."""
    config["dynamics"]["params"]["K"] = 0.9
    with pytest.warns(UserWarning, match="off-plant"):
        validate_simulation_config(config)


def test_seed_and_log_do_not_count_as_a_different_plant(config: dict[str, Any]) -> None:
    """The seed picks a realisation and ``log`` picks an output, so neither changes the plant."""
    config["dynamics"]["seed"] = 4242
    config["dynamics"]["log"] = "lfp"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_simulation_config(config)


def test_artifact_without_provenance_skips_those_checks(config: dict[str, Any], tmp_path: Path) -> None:
    """Artifacts written before provenance was recorded still validate on what they do carry."""
    config["controller"]["artifact"] = str(_artifact(tmp_path, TrainingProvenance()))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_simulation_config(config)


def test_data_plant_fingerprint_reads_the_batch_base(tmp_path: Path) -> None:
    """A trajectory directory is fingerprinted from the full config the batch's first entry carries."""
    overrides = [_plant(), {"dynamics": {"seed": 71}}]
    (tmp_path / "exp.yaml").write_text(yaml.safe_dump({"experiments": overrides}))
    assert data_plant_fingerprint(tmp_path) == plant_fingerprint(_plant())


def test_data_plant_fingerprint_absent_without_a_config(tmp_path: Path) -> None:
    """A directory that does not record its generating config yields no fingerprint."""
    assert data_plant_fingerprint(tmp_path) is None


def test_undecimated_predictor_needs_no_filter(config: dict[str, Any], tmp_path: Path) -> None:
    """A predictor trained on undecimated data is correctly paired with an unfiltered estimator."""
    provenance = TrainingProvenance(plant_fingerprint=plant_fingerprint(_plant()))
    art = MLPArtifact.load(_artifact(tmp_path, provenance))
    dataclasses.replace(art, dt=_PLANT_DT, downsample=1).save(tmp_path / "undecimated")

    config["estimator"] = {"class_path": "simulate.estimator.IdentityEstimator", "dt": _PLANT_DT}
    config["controller"]["dt"] = _PLANT_DT
    config["controller"]["artifact"] = str(tmp_path / "undecimated")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_simulation_config(config)


def _excitation(hold_ms: list[float]) -> dict[str, Any]:
    """A generating config whose excitation holds each amplitude for ``hold_ms``."""
    return {
        **_plant(),
        "controller": {
            "class_path": "neuro.control.WaveformController",
            "dt": _PLANT_DT,
            "input_type": "ras",
            "hold_ms": hold_ms,
        },
    }


def test_ragged_excitation_holds_warn(tmp_path: Path) -> None:
    """A hold shorter than the predictor's step switches the input off the grid it is strided on."""
    (tmp_path / "exp.yaml").write_text(yaml.safe_dump({"experiments": [_excitation([10.0, 100.0])]}))
    with pytest.warns(UserWarning, match=r"\[10\.0\] ms"):
        check_excitation_alignment(tmp_path, _DOWNSAMPLE)


def test_aligned_excitation_holds_are_quiet(tmp_path: Path) -> None:
    """Holds that are whole multiples of the predictor's step raise nothing."""
    (tmp_path / "exp.yaml").write_text(yaml.safe_dump({"experiments": [_excitation([20.0, 100.0])]}))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_excitation_alignment(tmp_path, _DOWNSAMPLE)
