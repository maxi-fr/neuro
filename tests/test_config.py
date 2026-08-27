from __future__ import annotations

import pytest
from pydantic import ValidationError

from neuro.config import (
    CategoricalParam,
    CurriculumMSESpec,
    FloatParam,
    IntParam,
    LogUniformParam,
    LossSpecs,
    ModelConfig,
    NNPredictorConfig,
    SimulationConfig,
    StftSpec,
    TrainingConfig,
    expand_dotted_dict,
    resolve_data_files,
)
from neuro.connectome import Connectome
from neuro.control.zero import ZeroController
from neuro.eeg import EEGMeasurement
from neuro.jansen_rit import JansenRitDynamics

_VALID_TRAINING = {
    "eval_horizon_s": 0.2,
    "losses": {
        "curriculum_mse": {
            "weight": 1.0,
            "span_s": 0.2,
            "curr_start": 0,
            "curr_end": 10,
        }
    },
}


def test_defaults_applied_for_missing_sections() -> None:
    """Test Defaults applied for missing sections."""
    cfg = NNPredictorConfig.from_dict({"training": _VALID_TRAINING})
    assert cfg.simulation.n_steps is None
    assert cfg.model.n_y == 5
    assert cfg.training.epochs == 100
    assert cfg.sweep is None


def test_known_keys_parsed() -> None:
    """Test Known keys parsed."""
    raw = {
        "simulation": {"dt": 1e-4, "downsample": 100, "n_steps": 50000, "data_path": "data/x", "cutoff_hz": 45.0},
        "model": {"n_y": 14, "hidden_size": 64},
        "training": {
            **_VALID_TRAINING,
            "epochs": 5,
            "scaler": "robust",
        },
    }
    cfg = NNPredictorConfig.from_dict(raw)
    assert cfg.simulation.downsample == 100
    assert cfg.simulation.n_steps == 50000
    assert cfg.simulation.cutoff_hz == 45.0
    assert cfg.model.n_y == 14
    assert cfg.model.hidden_size == 64
    assert cfg.training.scaler == "robust"


@pytest.mark.parametrize(
    "raw",
    [
        {"trainng": {}},
        {"model": {"n_yy": 3}},
        {"model": {"horizon": 5}},
        {"sweep": {"model": {"depth": {"typ": "int", "low": 0, "high": 5}}}},
        {"sweep": {"trials": 5}},
        {"training": {**_VALID_TRAINING, "w_psd": 0.1}},
        {"training": {**_VALID_TRAINING, "curriculum_start_epoch": 10}},
        {"training": {**_VALID_TRAINING, "curriculum_end_epoch": 10}},
        {"training": {**_VALID_TRAINING, "curriculum_max_steps": 10}},
        {"training": {**_VALID_TRAINING, "curriculum_alpha_min": 1.0}},
        {"training": {**_VALID_TRAINING, "curriculum_decay_fraction": 0.5}},
        {"model": {"latent_dim": 20}},
        {"training": {"eval_horizon_s": 0.2, "losses": {"unknown_loss": {"weight": 1.0, "span_s": 0.2}}}},
    ],
)
def test_unknown_keys_rejected(raw: dict) -> None:
    """Test Unknown keys rejected."""
    with pytest.raises(ValidationError):
        NNPredictorConfig.from_dict(raw)


def test_wrong_scalar_type_rejected() -> None:
    """Test Wrong scalar type rejected."""
    with pytest.raises(ValidationError):
        NNPredictorConfig.from_dict({"model": {"n_y": "not-an-int"}, "training": _VALID_TRAINING})


def test_sweep_section_typed() -> None:
    """Test Sweep section typed."""
    raw = {
        "training": _VALID_TRAINING,
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
        },
    }
    cfg = NNPredictorConfig.from_dict(raw)
    assert cfg.sweep is not None
    assert cfg.sweep.n_trials == 7
    assert isinstance(cfg.sweep.model["depth"], IntParam)
    assert isinstance(cfg.sweep.model["hidden_size"], CategoricalParam)
    assert isinstance(cfg.sweep.training["learning_rate"], LogUniformParam)
    assert isinstance(cfg.sweep.training["weight_decay"], FloatParam)


def test_sweep_objective_defaults_to_log_energy() -> None:
    """Test the sweep objective defaults to the log-energy metric rather than waveform NMSE."""
    cfg = NNPredictorConfig.from_dict({"training": _VALID_TRAINING, "sweep": {"n_trials": 3}})
    assert cfg.sweep is not None
    assert cfg.sweep.objective == "log_energy"


def test_closed_loop_objective_without_its_section_rejected() -> None:
    """Test asking for the closed-loop objective without configuring the evaluation is rejected."""
    with pytest.raises(ValidationError, match=r"requires a 'sweep\.closed_loop' section"):
        NNPredictorConfig.from_dict({"training": _VALID_TRAINING, "sweep": {"objective": "closed_loop"}})


def test_sweep_unknown_param_type_rejected() -> None:
    """Test Sweep unknown param type rejected."""
    with pytest.raises(ValidationError):
        NNPredictorConfig.from_dict(
            {"training": _VALID_TRAINING, "sweep": {"model": {"x": {"type": "bogus", "low": 0, "high": 1}}}}
        )


@pytest.mark.parametrize(
    "raw",
    [
        {"training": _VALID_TRAINING, "simulation": {"dt": 0}},
        {"training": _VALID_TRAINING, "simulation": {"downsample": 0}},
        {"training": _VALID_TRAINING, "model": {"depth": -1}},
        {"training": _VALID_TRAINING, "model": {"n_y": 0}},
        {"training": {**_VALID_TRAINING, "learning_rate": 0}},
        {"training": {**_VALID_TRAINING, "train_split": 1.0}},
        {"training": {**_VALID_TRAINING, "scaler": "standrd"}},
        {"training": {**_VALID_TRAINING, "warmup_epochs": -1}},
        {"training": {**_VALID_TRAINING, "epochs": 10, "warmup_epochs": 10}},
        {"training": {"eval_horizon_s": 0.0, "losses": _VALID_TRAINING["losses"]}},
        {"training": {"eval_horizon_s": 0.2, "losses": {}}},
        {
            "training": {
                "eval_horizon_s": 0.2,
                "losses": {
                    "curriculum_mse": {
                        "weight": 1.0,
                        "span_s": 0.2,
                        "curr_start": 100,
                        "curr_end": 50,
                    }
                },
            }
        },
        {
            "simulation": {"dt": 0.01, "downsample": 1},
            "training": {
                "eval_horizon_s": 0.2,
                "losses": {
                    "curriculum_mse": {"weight": 1.0, "span_s": 0.2, "curr_start": 0, "curr_end": 10},
                    "eeg_ms": {"weight": 1.0, "span_s": 0.05, "window_s": 0.1},
                },
            },
        },
        {
            "simulation": {"dt": 0.01, "downsample": 1},
            "training": {
                "eval_horizon_s": 0.2,
                "losses": {
                    "curriculum_mse": {"weight": 1.0, "span_s": 0.2, "curr_start": 0, "curr_end": 10},
                    "stft": {"weight": 1.0, "n_span": 20, "n_segment": 40, "n_hop": 10},
                },
            },
        },
        {
            "simulation": {"dt": 0.01, "downsample": 1},
            "training": {
                "eval_horizon_s": 0.2,
                "losses": {
                    "curriculum_mse": {"weight": 1.0, "span_s": 0.2, "curr_start": 0, "curr_end": 10},
                    "stft": {"weight": 1.0, "n_span": 20, "n_segment": 10, "n_hop": 5, "kernel_width": 4},
                },
            },
        },
        {
            "simulation": {"dt": 0.01, "downsample": 1},
            "training": {
                "eval_horizon_s": 0.2,
                "losses": {
                    "curriculum_mse": {"weight": 1.0, "span_s": 0.2, "curr_start": 0, "curr_end": 10},
                    "stft": {"weight": 1.0, "n_span": 20, "n_segment": 10, "n_hop": 5, "kernel_width": 3},
                },
            },
        },
        {
            "simulation": {"dt": 0.01, "downsample": 1},
            "training": {
                "eval_horizon_s": 0.2,
                "losses": {
                    "curriculum_mse": {"weight": 1.0, "span_s": 0.2, "curr_start": 0, "curr_end": 10},
                    "stft": {"weight": 1.0, "n_span": 20, "n_segment": 10, "n_hop": 10, "band_hz": [60.0, 80.0]},
                },
            },
        },
        {
            "simulation": {"dt": 0.01, "downsample": 1},
            "training": {
                "eval_horizon_s": 0.2,
                "losses": {
                    "curriculum_mse": {"weight": 1.0, "span_s": 0.2, "curr_start": 0, "curr_end": 10},
                    "stft": {"weight": 1.0, "n_span": 20, "n_segment": 10, "n_hop": 10, "band_hz": [12.0, 3.0]},
                },
            },
        },
        {
            "simulation": {"dt": 0.01, "downsample": 1},
            "training": {
                "eval_horizon_s": 0.2,
                "losses": {
                    "curriculum_mse": {"weight": 1.0, "span_s": 0.2, "curr_start": 0, "curr_end": 10},
                    "stft": {
                        "weight": 1.0,
                        "n_span": 20,
                        "n_segment": 10,
                        "n_hop": 10,
                        "band_hz": [10.0, 20.0],
                        "n_bin_pool": 4,
                    },
                },
            },
        },
        {
            "training": {
                "eval_horizon_s": 0.2,
                "losses": {
                    "curriculum_mse": {
                        "weight": 1.0,
                        "span_s": 0.2,
                        "curr_start": 0,
                        "curr_end": 10,
                        "start_epoch": 5,
                    }
                },
            }
        },
        {"training": _VALID_TRAINING, "sweep": {"training": {"lr": {"type": "loguniform", "low": 0, "high": 1}}}},
        {"training": _VALID_TRAINING, "sweep": {"model": {"n_y": {"type": "int", "low": 10, "high": 5}}}},
    ],
)
def test_value_constraints_rejected(raw: dict) -> None:
    """Test Value constraints rejected."""
    with pytest.raises(ValidationError):
        NNPredictorConfig.from_dict(raw)


def test_stft_bin_range_excludes_dc_and_clips_to_the_band() -> None:
    """bin_range drops the DC bin and keeps only rfft bins inside band_hz."""
    fs = 50.0
    full = StftSpec(weight=1.0, n_span=50, n_segment=50, n_hop=25)
    assert full.bin_range(fs) == (1, 26)
    assert full.n_segment_frames(full.n_span) == 1  # segment == span: the Welch endpoint

    hopped = StftSpec(weight=1.0, n_span=50, n_segment=25, n_hop=12)
    assert hopped.n_segment_frames(hopped.n_span) == 3

    banded = StftSpec(weight=1.0, n_span=50, n_segment=50, n_hop=25, band_hz=(3.0, 12.0))
    assert banded.bin_range(fs) == (3, 13)  # 1 Hz per bin at n_segment = 50


def test_valid_boundaries_accepted() -> None:
    """Test Valid boundaries accepted."""
    cfg = NNPredictorConfig.from_dict(
        {
            "model": {"depth": 0},
            "training": {
                "eval_horizon_s": 0.2,
                "losses": {
                    "curriculum_mse": {"weight": 1.0, "span_s": 0.2, "curr_start": 5, "curr_end": 5},
                },
            },
        }
    )
    assert cfg.model.depth == 0
    assert cfg.training.losses is not None
    assert cfg.training.losses.curriculum_mse is not None
    assert cfg.training.losses.curriculum_mse.curr_start == 5
    assert cfg.training.losses.curriculum_mse.curr_end == 5


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        (
            {
                "model": {"n_y": 10},
                "training": _VALID_TRAINING,
                "sweep": {"model": {"n_y": {"type": "int", "low": 5, "high": 15}}},
            },
            r"Overlap: \['n_y'\]",
        ),
        (
            {
                "training": {**_VALID_TRAINING, "learning_rate": 1e-4, "epochs": 50},
                "sweep": {"training": {"epochs": {"type": "int", "low": 10, "high": 100}}},
            },
            r"Overlap: \['epochs'\]",
        ),
        (
            {
                "training": _VALID_TRAINING,
                "sweep": {
                    "training": {
                        "losses.curriculum_mse.span_s": {"type": "float", "low": 0.1, "high": 0.5},
                    }
                },
            },
            r"Overlap: \['losses\.curriculum_mse\.span_s'\]",
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
                "training": _VALID_TRAINING,
                "sweep": {"model": {"not_a_param": {"type": "int", "low": 1, "high": 5}}},
            },
            r"Keys \['not_a_param'\] in 'sweep.model' are not valid",
        ),
        (
            {
                "training": _VALID_TRAINING,
                "sweep": {"training": {"lr": {"type": "loguniform", "low": 1e-5, "high": 1e-3}}},
            },
            r"Keys \['lr'\] in 'sweep.training' are not valid",
        ),
        (
            {
                "training": _VALID_TRAINING,
                "sweep": {"training": {"losses.bogus.weight": {"type": "float", "low": 0.1, "high": 1.0}}},
            },
            r"Keys \['losses\.bogus\.weight'\] in 'sweep.training' are not valid",
        ),
        (
            {
                "training": _VALID_TRAINING,
                "sweep": {"training": {"losses.eeg_ms.weight": {"type": "float", "low": 0.1, "high": 1.0}}},
            },
            r"Loss 'eeg_ms' referenced in 'sweep\.training\.losses\.eeg_ms\.weight' is not configured",
        ),
    ],
)
def test_sweep_invalid_keys_rejected(raw: dict, match: str) -> None:
    """Test Sweep invalid keys rejected."""
    with pytest.raises(ValidationError, match=match):
        NNPredictorConfig.from_dict(raw)


def test_sweep_valid_dotted_path_accepted() -> None:
    """Test valid dotted path for optional/default field on configured loss is accepted."""
    cfg = NNPredictorConfig.from_dict(
        {
            "training": _VALID_TRAINING,
            "sweep": {
                "n_trials": 5,
                "training": {
                    "losses.curriculum_mse.start_epoch": {"type": "int", "low": 0, "high": 50},
                },
            },
        }
    )
    assert cfg.sweep is not None
    assert "losses.curriculum_mse.start_epoch" in cfg.sweep.training


def test_expand_dotted_dict_nests_sweep_overrides() -> None:
    """The dotted keys the sweep suggests expand into the nesting deep_merge expects."""
    assert expand_dotted_dict({"losses.eeg_ms.weight": 0.3, "batch_size": 64}) == {
        "losses": {"eeg_ms": {"weight": 0.3}},
        "batch_size": 64,
    }


def test_resolve_data_files_missing_path() -> None:
    """Test Resolve data files missing path."""
    cfg = NNPredictorConfig.from_dict({"training": _VALID_TRAINING})
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


def test_sweep_without_closed_loop_section() -> None:
    """Test omitting closed_loop leaves it unset rather than silently defaulted."""
    cfg = NNPredictorConfig.from_dict({"training": _VALID_TRAINING, "sweep": {"n_trials": 10}})
    assert cfg.sweep is not None
    assert cfg.sweep.closed_loop is None


def test_sweep_objective_validated_against_the_waveform_candidates() -> None:
    """A waveform sweep may not name an observable-only candidate."""
    with pytest.raises(ValidationError, match="not a candidate"):
        NNPredictorConfig.from_dict({"training": _VALID_TRAINING, "sweep": {"objective": "val_log_mse"}})


def test_observable_predictor_rejects_reduction_losses() -> None:
    """An observable predictor rejects reduction losses (stft, eeg_ms)."""
    raw = {
        "training": {
            "eval_horizon_s": 0.2,
            "losses": {
                "curriculum_mse": {"weight": 1.0, "span_s": 0.2, "curr_start": 0, "curr_end": 10},
                "stft": {"weight": 1.0, "n_span": 80, "n_segment": 64, "n_hop": 16, "band_hz": [4.0, 30.0]},
            },
        },
        "observable": {
            "n_segment": 64,
            "n_hop": 16,
            "band_hz": [4.0, 30.0],
            "n_bin_pool": 2,
            "kernel_width": 5,
        },
    }
    with pytest.raises(ValidationError, match="does not support reduction losses"):
        NNPredictorConfig.from_dict(raw)


def test_observable_sweep_objective_validation() -> None:
    """An observable sweep validates its objective against observable candidates."""
    base = {
        "model": {"n_u": 8},
        "training": _VALID_TRAINING,
        "observable": {
            "n_segment": 64,
            "n_hop": 16,
            "band_hz": [4.0, 30.0],
            "n_bin_pool": 2,
            "kernel_width": 5,
        },
    }
    # Accepts valid observable candidates
    cfg1 = NNPredictorConfig.from_dict({**base, "sweep": {"objective": "val_log_mse"}})
    assert cfg1.sweep is not None
    assert cfg1.sweep.objective == "val_log_mse"

    cfg2 = NNPredictorConfig.from_dict({**base, "sweep": {"objective": "val_loss"}})
    assert cfg2.sweep is not None
    assert cfg2.sweep.objective == "val_loss"

    # Rejects waveform-only candidate
    with pytest.raises(ValidationError, match=r"sweep\.objective 'rollout_nmse' is not a candidate"):
        NNPredictorConfig.from_dict({**base, "sweep": {"objective": "rollout_nmse"}})


def test_control_support_rule_validation() -> None:
    """The control-support rule requires n_u >= kernel_width - 1 + ceil(segment / hop)."""
    # segment=64, hop=16, kernel_width=5 -> min_n_u = 5 - 1 + 4 = 8
    base = {
        "training": _VALID_TRAINING,
        "observable": {
            "n_segment": 64,
            "n_hop": 16,
            "band_hz": [4.0, 30.0],
            "n_bin_pool": 2,
            "kernel_width": 5,
        },
    }
    # Accepting conforming config
    cfg = NNPredictorConfig.from_dict({**base, "model": {"n_u": 8}})
    assert cfg.model.n_u == 8

    # Rejecting violating config naming offending values
    with pytest.raises(
        ValidationError,
        match=r"model\.n_u \(7\) violates the control-support rule: must be >= 8 \(kernel_width=5, segment=64, hop=16\)",
    ):
        NNPredictorConfig.from_dict({**base, "model": {"n_u": 7}})


def test_curriculum_span_and_eval_horizon_must_hold_at_least_one_frame() -> None:
    """The curriculum span and eval horizon must resolve to at least one Frame at the Frame rate."""
    # dt = 1e-4, downsample = 200 -> fs = 50 Hz. Hop = 25 -> fs_frame = 2 Hz.
    # At fs_frame = 2 Hz, span_s = 0.1s gives round(0.1 * 2) = 0 frames (< 1).
    base = {
        "simulation": {"dt": 1e-4, "downsample": 200},
        "model": {"n_u": 3},
        "observable": {
            "n_segment": 50,
            "n_hop": 25,
            "kernel_width": 1,
        },
    }
    # span_s = 0.1s -> 0 frames -> rejected naming span_s, frame rate, hop, fs
    with pytest.raises(
        ValidationError,
        match=r"loss 'curriculum_mse' span \(0\.1 s\) resolves to 0 frame\(s\) at frame rate 2 Hz \(hop=25, fs=50 Hz\)",
    ):
        NNPredictorConfig.from_dict(
            {
                **base,
                "training": {
                    "eval_horizon_s": 1.0,
                    "losses": {
                        "curriculum_mse": {"weight": 1.0, "span_s": 0.1, "curr_start": 0, "curr_end": 10},
                    },
                },
            }
        )

    # eval_horizon_s = 0.1s -> 0 frames -> rejected naming eval_horizon_s, frame rate, hop, fs
    with pytest.raises(
        ValidationError,
        match=r"training\.eval_horizon_s \(0\.1 s\) resolves to 0 frame\(s\) at frame rate 2 Hz \(hop=25, fs=50 Hz\)",
    ):
        NNPredictorConfig.from_dict(
            {
                **base,
                "training": {
                    "eval_horizon_s": 0.1,
                    "losses": {
                        "curriculum_mse": {"weight": 1.0, "span_s": 1.0, "curr_start": 0, "curr_end": 10},
                    },
                },
            }
        )

    # Conforming config with span_s = 1.0s (2 frames) and eval_horizon_s = 1.0s (2 frames) accepted
    conforming = NNPredictorConfig.from_dict(
        {
            **base,
            "training": {
                "eval_horizon_s": 1.0,
                "losses": {
                    "curriculum_mse": {"weight": 1.0, "span_s": 1.0, "curr_start": 0, "curr_end": 10},
                },
            },
        }
    )
    assert conforming.training.losses is not None
