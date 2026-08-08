from __future__ import annotations

import pytest
from pydantic import ValidationError

from neuro.config import (
    CategoricalParam,
    FloatParam,
    IntParam,
    LogUniformParam,
    NNPredictorConfig,
    resolve_data_files,
)
from neuro.connectome import Connectome
from neuro.control import ZeroController
from neuro.eeg import EEGMeasurement
from neuro.jansen_rit import JansenRitDynamics


def test_defaults_applied_for_missing_sections() -> None:
    """Test Defaults applied for missing sections."""
    cfg = NNPredictorConfig.from_dict({})
    assert cfg.simulation.n_steps is None
    assert cfg.model.n_y == 5
    assert cfg.training.epochs == 100
    assert cfg.model.latent_dim is None
    assert cfg.sweep is None


def test_known_keys_parsed() -> None:
    """Test Known keys parsed."""
    raw = {
        "simulation": {"dt": 1e-4, "downsample": 100, "n_steps": 50000, "data_path": "data/x"},
        "model": {"n_y": 14, "latent_dim": 16},
        "training": {"epochs": 5, "scaler": "robust"},
    }
    cfg = NNPredictorConfig.from_dict(raw)
    assert cfg.simulation.downsample == 100
    assert cfg.simulation.n_steps == 50000
    assert cfg.model.n_y == 14
    assert cfg.model.latent_dim == 16
    assert cfg.training.scaler == "robust"


@pytest.mark.parametrize(
    "raw",
    [
        {"trainng": {}},
        {"model": {"n_yy": 3}},
        {"sweep": {"model": {"depth": {"typ": "int", "low": 0, "high": 5}}}},
        {"sweep": {"trials": 5}},
        {"training": {"curriculum_alpha_min": 1.0}},
        {"training": {"curriculum_decay_fraction": 0.5}},
    ],
)
def test_unknown_keys_rejected(raw: dict) -> None:
    """Test Unknown keys rejected."""
    with pytest.raises(ValidationError):
        NNPredictorConfig.from_dict(raw)


def test_wrong_scalar_type_rejected() -> None:
    """Test Wrong scalar type rejected."""
    with pytest.raises(ValidationError):
        NNPredictorConfig.from_dict({"model": {"n_y": "not-an-int"}})


def test_sweep_section_typed() -> None:
    """Test Sweep section typed."""
    raw = {
        "sweep": {
            "n_trials": 7,
            "model": {
                "depth": {"type": "int", "low": 0, "high": 5},
                "hidden_size": {"type": "categorical", "choices": [64, 128]},
            },
            "training": {
                "learning_rate": {"type": "loguniform", "low": 1e-4, "high": 1e-2},
                "weight_decay": {"type": "float", "low": 1e-4, "high": 1e-1},
            },
        }
    }
    cfg = NNPredictorConfig.from_dict(raw)
    assert cfg.sweep is not None
    assert cfg.sweep.n_trials == 7
    assert isinstance(cfg.sweep.model["depth"], IntParam)
    assert isinstance(cfg.sweep.model["hidden_size"], CategoricalParam)
    assert isinstance(cfg.sweep.training["learning_rate"], LogUniformParam)
    assert isinstance(cfg.sweep.training["weight_decay"], FloatParam)


def test_sweep_unknown_param_type_rejected() -> None:
    """Test Sweep unknown param type rejected."""
    with pytest.raises(ValidationError):
        NNPredictorConfig.from_dict({"sweep": {"model": {"x": {"type": "bogus", "low": 0, "high": 1}}}})


@pytest.mark.parametrize(
    "raw",
    [
        {"simulation": {"dt": 0}},
        {"simulation": {"downsample": 0}},
        {"model": {"depth": -1}},
        {"model": {"n_y": 0}},
        {"training": {"learning_rate": 0}},
        {"training": {"train_split": 1.0}},
        {"training": {"scaler": "standrd"}},
        {"model": {"latent_dim": 0}},
        {"training": {"curriculum_start_epoch": -1}},
        {"training": {"curriculum_end_epoch": -1}},
        {"training": {"curriculum_start_epoch": 100, "curriculum_end_epoch": 50}},
        {"sweep": {"training": {"lr": {"type": "loguniform", "low": 0, "high": 1}}}},
        {"sweep": {"model": {"n_y": {"type": "int", "low": 10, "high": 5}}}},
    ],
)
def test_value_constraints_rejected(raw: dict) -> None:
    """Test Value constraints rejected."""
    with pytest.raises(ValidationError):
        NNPredictorConfig.from_dict(raw)


def test_valid_boundaries_accepted() -> None:
    """Test Valid boundaries accepted."""
    cfg = NNPredictorConfig.from_dict(
        {
            "model": {"depth": 0, "horizon": 20},
            "training": {"w_psd": 0.0, "curriculum_start_epoch": 5, "curriculum_end_epoch": 5},
        }
    )
    assert cfg.model.depth == 0
    assert cfg.training.curriculum_start_epoch == 5
    assert cfg.training.curriculum_end_epoch == 5


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        (
            {
                "model": {"n_y": 10},
                "sweep": {"model": {"n_y": {"type": "int", "low": 5, "high": 15}}},
            },
            r"Overlap: \['n_y'\]",
        ),
        (
            {
                "training": {"learning_rate": 1e-4, "epochs": 50},
                "sweep": {"training": {"epochs": {"type": "int", "low": 10, "high": 100}}},
            },
            r"Overlap: \['epochs'\]",
        ),
    ],
)
def test_sweep_overlap_rejected(raw: dict, match: str) -> None:
    """Test Sweep overlap rejected."""
    with pytest.raises(ValidationError, match=match):
        NNPredictorConfig.from_dict(raw)


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        (
            {
                "sweep": {"model": {"not_a_param": {"type": "int", "low": 1, "high": 5}}},
            },
            r"Keys \['not_a_param'\] in 'sweep.model' are not valid",
        ),
        (
            {
                "sweep": {"training": {"lr": {"type": "loguniform", "low": 1e-5, "high": 1e-3}}},
            },
            r"Keys \['lr'\] in 'sweep.training' are not valid",
        ),
    ],
)
def test_sweep_invalid_keys_rejected(raw: dict, match: str) -> None:
    """Test Sweep invalid keys rejected."""
    with pytest.raises(ValidationError, match=match):
        NNPredictorConfig.from_dict(raw)


def test_resolve_data_files_missing_path() -> None:
    """Test Resolve data files missing path."""
    cfg = NNPredictorConfig.from_dict({})
    with pytest.raises(ValueError, match="data_path not specified"):
        resolve_data_files(cfg)


def test_zero_controller_rejects_unknown_key() -> None:
    """Test Zero controller rejects unknown key."""
    with pytest.raises(ValidationError, match="ZeroController"):
        ZeroController.from_config({"dt": 0.1, "n_uu": 2})


def test_eeg_measurement_rejects_unknown_key() -> None:
    """Test Eeg measurement rejects unknown key."""
    with pytest.raises(ValidationError, match="EEGMeasurement"):
        EEGMeasurement.from_config({"n_nodes": 76, "speeed": 1})


def test_connectome_rejects_unknown_key() -> None:
    """Test Connectome rejects unknown key."""
    with pytest.raises(ValidationError, match="Connectome"):
        Connectome.from_config({"speed": 50.0, "speeed": 50.0})


def test_jansen_rit_dynamics_rejects_unknown_key() -> None:
    """Test Jansen rit dynamics rejects unknown key."""
    with pytest.raises(ValidationError, match="JansenRitDynamics"):
        JansenRitDynamics.from_config({"dt": 1e-4, "connectome": {"K": 1.0}, "seedd": 1})
