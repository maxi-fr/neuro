from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from neuro.config import (
    SEED_TIERS,
    CategoricalParam,
    ClosedLoopEvalConfig,
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
    resolve_seeds,
    resolve_simulation_config,
)
from neuro.connectome import Connectome
from neuro.control.zero import ZeroController
from neuro.eeg import EEGMeasurement
from neuro.jansen_rit import JansenRitDynamics
from neuro.seizure import build_seizure_a_gains

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
    assert cfg.training.device == "auto"
    assert cfg.sweep is None


def test_known_keys_parsed() -> None:
    """Test Known keys parsed."""
    raw = {
        "simulation": {"dt": 1e-4, "downsample": 100, "n_steps": 50000, "data_path": "data/x", "cutoff_hz": 45.0},
        "model": {"n_y": 14, "hidden_size": 64},
        "training": {
            **_VALID_TRAINING,
            "epochs": 15,
            "scaler": "robust",
            "device": "cuda",
        },
    }
    cfg = NNPredictorConfig.from_dict(raw)
    assert cfg.simulation.downsample == 100
    assert cfg.simulation.n_steps == 50000
    assert cfg.simulation.cutoff_hz == 45.0
    assert cfg.model.n_y == 14
    assert cfg.model.hidden_size == 64
    assert cfg.training.scaler == "robust"
    assert cfg.training.device == "cuda"
    assert cfg.training.epochs == 15


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


def test_stft_bin_range_includes_dc_unless_manually_excluded() -> None:
    """bin_range includes DC when band_hz is None or starts at 0, and excludes DC when band_hz starts > 0."""
    fs = 50.0
    full = StftSpec(weight=1.0, n_span=50, n_segment=50, n_hop=25)
    assert full.bin_range(fs) == (0, 26)  # DC included by default
    assert full.n_segment_frames(full.n_span) == 1  # segment == span: the Welch endpoint

    dc_explicit = StftSpec(weight=1.0, n_span=50, n_segment=50, n_hop=25, band_hz=(0.0, 25.0))
    assert dc_explicit.bin_range(fs) == (0, 26)

    dc_excluded = StftSpec(weight=1.0, n_span=50, n_segment=50, n_hop=25, band_hz=(1.0, 25.0))
    assert dc_excluded.bin_range(fs) == (1, 26)  # DC manually excluded

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


def test_simulation_config_fs_infers_downsample() -> None:
    """SimulationConfig derives downsample from sampling frequency fs and dt."""
    cfg = SimulationConfig(fs=50.0)
    assert cfg.downsample == 200
    assert cfg.fs == 50.0

    cfg100 = SimulationConfig(fs=100.0)
    assert cfg100.downsample == 100

    cfg_agree = SimulationConfig(fs=50.0, downsample=200)
    assert cfg_agree.downsample == 200

    with pytest.raises(ValidationError, match="implies downsample=200"):
        SimulationConfig(fs=50.0, downsample=100)


def test_seed_tiers_resolution() -> None:
    """Seed tiers resolve correctly to small, medium, and large presets."""
    assert resolve_seeds("small") == SEED_TIERS["small"]
    assert resolve_seeds("medium") == [7000, 7001, 7002, 7004, 7005]
    assert resolve_seeds("large") == SEED_TIERS["large"]
    assert resolve_seeds([1, 2, 3]) == [1, 2, 3]
    assert resolve_seeds(None) == SEED_TIERS["medium"]

    with pytest.raises(ValueError, match="Unknown seed tier 'invalid'"):
        resolve_seeds("invalid")


def test_closed_loop_eval_config_defaults_and_tiers() -> None:
    """ClosedLoopEvalConfig defaults seeds to medium, t_end to 12.0s, and resolves seed tiers."""
    cfg = ClosedLoopEvalConfig(simulation_config="sim.yaml")
    assert cfg.seeds == [7000, 7001, 7002, 7004, 7005]
    assert cfg.t_end == 12.0
    assert cfg.seizure_ptp_mv == 5.0
    assert cfg.max_seizing_regions == 5

    cfg_small = ClosedLoopEvalConfig(simulation_config="sim.yaml", seeds="small")  # type: ignore[arg-type]
    assert cfg_small.seeds == [7000, 7001]


def test_jansen_rit_regime_healthy_and_seizure() -> None:
    """JansenRitDynamics builds healthy and seizure regimes without manual parameter arrays."""
    dyn_h = JansenRitDynamics.from_config({"regime": "healthy", "seed": 42})
    assert np.allclose(dyn_h.net_params.A, 3.25)
    assert dyn_h.net_params.sigma == 280.0
    assert dyn_h.conn.K == 0.60
    assert dyn_h.dt == 1e-4

    dyn_s = JansenRitDynamics.from_config({"regime": "seizure", "seed": 42})
    assert np.allclose(dyn_s.net_params.A, build_seizure_a_gains(dyn_s.conn))
    assert dyn_s.net_params.sigma == 280.0

    with pytest.raises(ValidationError):
        JansenRitDynamics.from_config({"regime": "unknown"})


def test_resolve_simulation_config_expands_defaults() -> None:
    """resolve_simulation_config populates t_end, reference, and eeg sensors when omitted."""
    cfg = {"dynamics": {"dt": 1e-4}}
    resolved = resolve_simulation_config(cfg)
    assert resolved["t_end"] == 12.0
    assert resolved["reference"]["class_path"] == "simulate.reference.StepReference"
    assert resolved["reference"]["dt"] == 1e-4
    assert resolved["reference"]["step_value"] == 0.0
    assert resolved["sensors"]["class_path"] == "simulate.sensor.GaussianSensor"
    assert resolved["sensors"]["measurement"]["class_path"] == "neuro.eeg.EEGMeasurement"


def test_resolve_simulation_config_immutability() -> None:
    """resolve_simulation_config does not mutate its input mapping."""
    original = {"dynamics": {"dt": 1e-4, "regime": "healthy"}, "sensors": "eeg"}
    resolved = resolve_simulation_config(original)
    assert "t_end" not in original
    assert original["sensors"] == "eeg"
    assert original["dynamics"] == {"dt": 1e-4, "regime": "healthy"}
    assert resolved["t_end"] == 12.0
    assert isinstance(resolved["sensors"], dict)


def test_resolve_simulation_config_sensors_shorthands() -> None:
    """resolve_simulation_config expands 'eeg' and 'oracle' sensor shorthands."""
    cfg_eeg = resolve_simulation_config({"dynamics": {"dt": 2e-4}, "sensors": "eeg"})
    assert cfg_eeg["sensors"]["class_path"] == "simulate.sensor.GaussianSensor"
    assert cfg_eeg["sensors"]["dt"] == 2e-4

    cfg_oracle = resolve_simulation_config({"dynamics": {"dt": 1e-4}, "sensors": "oracle"})
    assert cfg_oracle["sensors"]["class_path"] == "neuro.predictor.oracle.FullStateSensor"

    with pytest.raises(ValueError, match="Unknown sensors shorthand"):
        resolve_simulation_config({"sensors": "unknown"})


def test_resolve_simulation_config_dynamics_regimes() -> None:
    """resolve_simulation_config expands healthy and seizure dynamics regimes."""
    cfg_h = resolve_simulation_config({"dynamics": {"regime": "healthy"}})
    assert cfg_h["dynamics"]["class_path"] == "neuro.jansen_rit.JansenRitDynamics"
    assert cfg_h["dynamics"]["params"]["A"] == 3.25
    assert cfg_h["dynamics"]["params"]["sigma"] == 280.0
    assert cfg_h["dynamics"]["connectome"] == {"speed": 50.0, "K": 0.60}

    cfg_s = resolve_simulation_config({"dynamics": {"regime": "seizure"}})
    assert len(cfg_s["dynamics"]["params"]["A"]) == 76
    assert cfg_s["dynamics"]["params"]["sigma"] == 280.0

    with pytest.raises(ValueError, match="Unknown dynamics regime"):
        resolve_simulation_config({"dynamics": {"regime": "unknown"}})


def test_resolve_simulation_config_oracle_propagation() -> None:
    """resolve_simulation_config propagates plant settings to oracle estimator and problem."""
    cfg = {
        "dynamics": {
            "regime": "seizure",
            "stimulation": {"model": "roast_3d", "field_projection_path": "data/roast_field_projection_3d.npz"},
        },
        "sensors": "oracle",
        "estimator": {"class_path": "neuro.predictor.oracle.JansenRitOracleEstimator", "dt": 1e-4},
        "controller": {
            "class_path": "neuro.control.mpc.TrajOptMPCController",
            "dt": 0.02,
            "problem": {
                "class_path": "neuro.predictor.jansen_rit.build_jansen_rit_problem",
                "horizon": 10,
            },
        },
    }
    resolved = resolve_simulation_config(cfg)
    assert resolved["estimator"]["connectome"] == resolved["dynamics"]["connectome"]
    assert resolved["estimator"]["params"] == resolved["dynamics"]["params"]
    assert resolved["controller"]["problem"]["connectome"] == resolved["dynamics"]["connectome"]
    assert resolved["controller"]["problem"]["params"] == resolved["dynamics"]["params"]
    assert resolved["controller"]["problem"]["stimulation"] == resolved["dynamics"]["stimulation"]
