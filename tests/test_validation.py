from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import yaml

from neuro.config import StftGeometry
from neuro.predictor.module import AutoregressiveMLP
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


def _module(
    provenance: TrainingProvenance, *, dt: float = _PLANT_DT * _DOWNSAMPLE, downsample: int = _DOWNSAMPLE
) -> AutoregressiveMLP:
    """A tiny depth-0 MLP carrying ``provenance`` at the given rate and decimation."""
    model = AutoregressiveMLP(
        n_y=2,
        n_u=2,
        horizon=50,
        n_channels=_N_CHANNELS,
        n_controls=_N_CONTROLS,
        n_outputs=_N_CHANNELS,
        hidden_size=0,
        depth=0,
        activation="relu",
        dt=dt,
        y_std=Standardizer(center=np.zeros(_N_CHANNELS), scale=np.ones(_N_CHANNELS)),
        u_std=Standardizer(center=np.zeros(_N_CONTROLS), scale=np.ones(_N_CONTROLS)),
    )
    model.downsample = downsample
    model.provenance = provenance
    return model


def _artifact(tmp_path: Path, provenance: TrainingProvenance) -> Path:
    """Save a tiny MLP checkpoint carrying ``provenance`` and return its suffix-less stem."""
    stem = tmp_path / "model"
    _module(provenance).save(stem)
    return stem


def _obs_module(
    provenance: TrainingProvenance,
    geom: StftGeometry,
    *,
    dt: float = 0.10,
    downsample: int = _DOWNSAMPLE,
    n_u: int = 4,
) -> AutoregressiveMLP:
    """A tiny depth-0 observable MLP carrying ``provenance`` and ``geom``."""
    fs = 1.0 / (_PLANT_DT * downsample)
    n_values = geom.n_values(fs)
    n_outputs = _N_CHANNELS * n_values
    model = AutoregressiveMLP(
        n_y=2,
        n_u=n_u,
        horizon=50,
        n_channels=_N_CHANNELS,
        n_controls=_N_CONTROLS,
        n_outputs=n_outputs,
        hidden_size=0,
        depth=0,
        activation="relu",
        dt=dt,
        geometry=geom,
        y_std=Standardizer(center=np.zeros(n_outputs), scale=np.ones(n_outputs)),
        u_std=Standardizer(center=np.zeros(_N_CONTROLS), scale=np.ones(_N_CONTROLS)),
    )
    model.downsample = downsample
    model.provenance = provenance
    return model


def _obs_artifact(
    tmp_path: Path,
    provenance: TrainingProvenance,
    geom: StftGeometry,
    *,
    dt: float = 0.10,
    downsample: int = _DOWNSAMPLE,
    n_u: int = 4,
) -> Path:
    """Save a tiny observable MLP checkpoint and return its suffix-less stem."""
    stem = tmp_path / "obs_model"
    _obs_module(provenance, geom, dt=dt, downsample=downsample, n_u=n_u).save(stem)
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
            "class_path": "neuro.control.mpc.TrajOptMPCController",
            "dt": _PLANT_DT * _DOWNSAMPLE,
            "problem": {
                "class_path": "neuro.control.mpc.build_waveform_problem",
                "artifact": str(_artifact(tmp_path, provenance)),
                "horizon": 50,
            },
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
    config["controller"]["problem"]["artifact"] = str(_artifact(tmp_path, provenance))
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


def test_horizon_beyond_trained_horizon_warns(config: dict[str, Any]) -> None:
    """The MPC costing a free run longer than the predictor was fit against warns."""
    config["controller"]["problem"]["horizon"] = 51
    with pytest.warns(UserWarning, match=r"trained horizon"):
        validate_simulation_config(config)


def _psd_npz(tmp_path: Path, *, fs: float = 50.0, L: int = 50, R: int = 25) -> Path:
    """Save a synthetic healthy PSD reference npz."""
    npz_path = tmp_path / "healthy_psd.npz"
    np.savez(
        npz_path,
        Pref=np.ones((2, L // 2 + 1)),
        freqs=np.linspace(0, fs / 2, L // 2 + 1),
        fs=fs,
        L=L,
        R=R,
        quantile=0.9,
        n_windows=100,
        plant_fingerprint="dummy",
    )
    return npz_path


def test_consistent_psd_reference_passes(config: dict[str, Any], tmp_path: Path) -> None:
    """A PSD reference agreeing on rate passes cleanly."""
    config["controller"]["problem"]["psd_ref"] = str(_psd_npz(tmp_path, fs=50.0, L=50, R=25))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_simulation_config(config)


def test_missing_psd_reference_raises(config: dict[str, Any], tmp_path: Path) -> None:
    """A missing PSD reference npz raises ConfigConsistencyError."""
    config["controller"]["problem"]["psd_ref"] = str(tmp_path / "absent.npz")
    with pytest.raises(ConfigConsistencyError, match="spectral reference envelope not found"):
        validate_simulation_config(config)


def test_psd_reference_rate_mismatch_raises(config: dict[str, Any], tmp_path: Path) -> None:
    """A sampling rate mismatch between controller and PSD reference raises ConfigConsistencyError."""
    config["controller"]["problem"]["psd_ref"] = str(_psd_npz(tmp_path, fs=100.0))
    with pytest.raises(ConfigConsistencyError, match="must match spectral reference dt"):
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
    """Checkpoints written before provenance was recorded still validate on what they do carry."""
    config["controller"]["problem"]["artifact"] = str(_artifact(tmp_path, TrainingProvenance()))
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
    _module(provenance, dt=_PLANT_DT, downsample=1).save(tmp_path / "undecimated")

    config["estimator"] = {"class_path": "simulate.estimator.IdentityEstimator", "dt": _PLANT_DT}
    config["controller"]["dt"] = _PLANT_DT
    config["controller"]["problem"]["artifact"] = str(tmp_path / "undecimated")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_simulation_config(config)


def _excitation(hold_ms: list[float]) -> dict[str, Any]:
    """A generating config whose excitation holds each amplitude for ``hold_ms``."""
    return {
        **_plant(),
        "controller": {
            "class_path": "neuro.control.schedule.ScheduleController",
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


def test_observable_estimator_and_envelope_validation(tmp_path: Path) -> None:
    """Observable estimator must run at plant rate and controller dt must equal hop * downsample * plant_dt."""
    geom = StftGeometry(n_segment=20, n_hop=5, band_hz=(4.0, 16.0), n_bin_pool=2, kernel="boxcar", kernel_width=1)
    env_path = tmp_path / "obs_env.npz"
    np.savez_compressed(
        env_path,
        Pref_frames=np.full((2, 2), -2.0),
        fs=50.0,
        n_segment=20,
        n_hop=5,
        band_hz=np.array([4.0, 16.0]),
        n_bin_pool=2,
        kernel="boxcar",
        kernel_width=1,
    )
    # Expected dt = hop * downsample * plant_dt = 5 * 200 * 1e-4 = 0.10s.
    provenance = TrainingProvenance(cutoff_hz=25.0, plant_fingerprint=plant_fingerprint(_plant()))
    art = _obs_artifact(tmp_path, provenance, geom, dt=0.10, downsample=200, n_u=4)

    sim_cfg: dict[str, Any] = {
        **_plant(),
        "estimator": {
            "class_path": "neuro.filtering.ObservableEstimator",
            "dt": _PLANT_DT,
            "downsample": 200,
            "geometry": geom.model_dump(),
        },
        "controller": {
            "class_path": "neuro.control.mpc.TrajOptMPCController",
            "dt": 0.10,
            "problem": {
                "class_path": "neuro.control.mpc.build_observable_problem",
                "artifact": str(art),
                "envelope_ref": str(env_path),
            },
        },
    }

    # Valid config passes validation cleanly
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_simulation_config(sim_cfg)

    # Mismatched estimator dt raises naming offending values
    bad_estimator_cfg = {**sim_cfg, "estimator": {**sim_cfg["estimator"], "dt": 0.01}}
    with pytest.raises(ConfigConsistencyError, match=r"estimator\.dt \(0\.01\) must equal dynamics\.dt \(0\.0001\)"):
        validate_simulation_config(bad_estimator_cfg)

    # Mismatched controller dt raises naming hop, downsample, plant_dt
    bad_controller_cfg = {**sim_cfg, "controller": {**sim_cfg["controller"], "dt": 0.05}}
    with pytest.raises(
        ConfigConsistencyError,
        match=r"controller\.dt \(0\.05\) must equal hop \(5\) x downsample \(200\) x dynamics\.dt \(0\.0001\) = 0\.1 s",
    ):
        validate_simulation_config(bad_controller_cfg)


def test_observable_geometry_agreement_between_estimator_and_checkpoint(tmp_path: Path) -> None:
    """The Observable geometry in the estimator must agree with the checkpoint geometry."""
    geom = StftGeometry(n_segment=20, n_hop=5, band_hz=(4.0, 16.0), n_bin_pool=2, kernel="boxcar", kernel_width=1)
    provenance = TrainingProvenance(cutoff_hz=25.0, plant_fingerprint=plant_fingerprint(_plant()))
    art = _obs_artifact(tmp_path, provenance, geom, dt=0.10, downsample=200, n_u=4)

    base_cfg: dict[str, Any] = {
        **_plant(),
        "estimator": {
            "class_path": "neuro.filtering.ObservableEstimator",
            "dt": _PLANT_DT,
            "downsample": 200,
            "geometry": geom.model_dump(),
        },
        "controller": {
            "class_path": "neuro.control.mpc.TrajOptMPCController",
            "dt": 0.10,
            "problem": {
                "class_path": "neuro.control.mpc.build_observable_problem",
                "artifact": str(art),
            },
        },
    }

    # Disagreeing n_segment raises naming conflicting values
    cfg = {**base_cfg, "estimator": {**base_cfg["estimator"], "geometry": {**geom.model_dump(), "n_segment": 40}}}
    with pytest.raises(
        ConfigConsistencyError,
        match=r"estimator geometry does not match the predictor checkpoint: n_segment \(40 vs 20\)",
    ):
        validate_simulation_config(cfg)

    # Disagreeing n_bin_pool raises naming conflicting values
    cfg = {**base_cfg, "estimator": {**base_cfg["estimator"], "geometry": {**geom.model_dump(), "n_bin_pool": 1}}}
    with pytest.raises(
        ConfigConsistencyError,
        match=r"estimator geometry does not match the predictor checkpoint: n_bin_pool \(1 vs 2\)",
    ):
        validate_simulation_config(cfg)


def test_observable_envelope_geometry_agreement(tmp_path: Path) -> None:
    """The healthy envelope's geometry must agree with the model's recorded geometry."""
    geom = StftGeometry(n_segment=20, n_hop=5, band_hz=(4.0, 16.0), n_bin_pool=2, kernel="boxcar", kernel_width=1)
    provenance = TrainingProvenance(cutoff_hz=25.0, plant_fingerprint=plant_fingerprint(_plant()))
    art = _obs_artifact(tmp_path, provenance, geom, dt=0.10, downsample=200, n_u=4)

    # Save envelope with mismatched band_hz
    bad_env_path = tmp_path / "bad_env.npz"
    np.savez_compressed(
        bad_env_path,
        Pref_frames=np.full((2, 2), -2.0),
        fs=50.0,
        n_segment=20,
        n_hop=5,
        band_hz=np.array([2.0, 10.0]),
        n_bin_pool=2,
        kernel="boxcar",
        kernel_width=1,
    )

    sim_cfg: dict[str, Any] = {
        **_plant(),
        "estimator": {
            "class_path": "neuro.filtering.ObservableEstimator",
            "dt": _PLANT_DT,
            "downsample": 200,
            "geometry": geom.model_dump(),
        },
        "controller": {
            "class_path": "neuro.control.mpc.TrajOptMPCController",
            "dt": 0.10,
            "problem": {
                "class_path": "neuro.control.mpc.build_observable_problem",
                "artifact": str(art),
                "envelope_ref": str(bad_env_path),
            },
        },
    }

    with pytest.raises(
        ConfigConsistencyError,
        match=r"envelope geometry does not match the predictor checkpoint: band_hz \(\(2\.0, 10\.0\) vs \(4\.0, 16\.0\)\)",
    ):
        validate_simulation_config(sim_cfg)


def test_envelope_channel_count_and_sampling_rate_validation(tmp_path: Path) -> None:
    """The healthy envelope's channel count and sampling rate must agree with the model's."""
    geom = StftGeometry(n_segment=20, n_hop=5, band_hz=(4.0, 16.0), n_bin_pool=2, kernel="boxcar", kernel_width=1)
    provenance = TrainingProvenance(cutoff_hz=25.0, plant_fingerprint=plant_fingerprint(_plant()))
    art = _obs_artifact(tmp_path, provenance, geom, dt=0.10, downsample=200, n_u=4)

    # Envelope with 4 channels instead of 2
    bad_ch_path = tmp_path / "bad_ch.npz"
    np.savez_compressed(
        bad_ch_path,
        Pref_frames=np.full((4, 2), -2.0),
        fs=50.0,
        n_segment=20,
        n_hop=5,
        band_hz=np.array([4.0, 16.0]),
        n_bin_pool=2,
        kernel="boxcar",
        kernel_width=1,
    )

    sim_cfg: dict[str, Any] = {
        **_plant(),
        "estimator": {
            "class_path": "neuro.filtering.ObservableEstimator",
            "dt": _PLANT_DT,
            "downsample": 200,
            "geometry": geom.model_dump(),
        },
        "controller": {
            "class_path": "neuro.control.mpc.TrajOptMPCController",
            "dt": 0.10,
            "problem": {
                "class_path": "neuro.control.mpc.build_observable_problem",
                "artifact": str(art),
                "envelope_ref": str(bad_ch_path),
            },
        },
    }

    with pytest.raises(
        ConfigConsistencyError, match=r"envelope channel count \(4\) must equal predictor channel count \(2\)"
    ):
        validate_simulation_config(sim_cfg)

    # Envelope with 100 Hz fs instead of 50 Hz
    bad_fs_path = tmp_path / "bad_fs.npz"
    np.savez_compressed(
        bad_fs_path,
        Pref_frames=np.full((2, geom.n_values(100.0)), -2.0),
        fs=100.0,
        n_segment=20,
        n_hop=5,
        band_hz=np.array([4.0, 16.0]),
        n_bin_pool=2,
        kernel="boxcar",
        kernel_width=1,
    )
    sim_cfg["controller"]["problem"]["envelope_ref"] = str(bad_fs_path)
    with pytest.raises(
        ConfigConsistencyError, match=r"controller\.dt \(0\.1\) must match Observable reference dt \(0\.05 s"
    ):
        validate_simulation_config(sim_cfg)


def test_control_support_rule_in_checkpoint_validation(tmp_path: Path) -> None:
    """A checkpoint violating n_u >= kernel_width - 1 + ceil(segment / hop) is rejected."""
    # segment=20, hop=5, kernel_width=1 -> min_n_u = 4. Set n_u = 2 in checkpoint.
    geom = StftGeometry(n_segment=20, n_hop=5, band_hz=(4.0, 16.0), n_bin_pool=2, kernel="boxcar", kernel_width=1)
    provenance = TrainingProvenance(cutoff_hz=25.0, plant_fingerprint=plant_fingerprint(_plant()))
    bad_art = _obs_artifact(tmp_path, provenance, geom, dt=0.10, downsample=200, n_u=2)

    sim_cfg: dict[str, Any] = {
        **_plant(),
        "estimator": {
            "class_path": "neuro.filtering.ObservableEstimator",
            "dt": _PLANT_DT,
            "downsample": 200,
            "geometry": geom.model_dump(),
        },
        "controller": {
            "class_path": "neuro.control.mpc.TrajOptMPCController",
            "dt": 0.10,
            "problem": {
                "class_path": "neuro.control.mpc.build_observable_problem",
                "artifact": str(bad_art),
            },
        },
    }

    with pytest.raises(
        ConfigConsistencyError,
        match=r"predictor n_u \(2\) violates the control-support rule: must be >= 4 \(kernel_width=1, segment=20, hop=5\)",
    ):
        validate_simulation_config(sim_cfg)


def test_mismatched_model_and_estimator_kinds(tmp_path: Path) -> None:
    """Coupling an observable checkpoint to an anti-alias estimator or vice-versa is rejected."""
    geom = StftGeometry(n_segment=20, n_hop=5, kernel="boxcar", kernel_width=1)
    provenance = TrainingProvenance(cutoff_hz=25.0, plant_fingerprint=plant_fingerprint(_plant()))
    obs_art = _obs_artifact(tmp_path, provenance, geom, dt=0.10, downsample=200, n_u=4)
    wave_art = _artifact(tmp_path, provenance)

    # Observable checkpoint with AntiAliasEstimator
    bad_cfg1: dict[str, Any] = {
        **_plant(),
        "estimator": {
            "class_path": "neuro.filtering.AntiAliasEstimator",
            "dt": _PLANT_DT,
            "downsample": 200,
        },
        "controller": {
            "class_path": "neuro.control.mpc.TrajOptMPCController",
            "dt": 0.10,
            "problem": {
                "class_path": "neuro.control.mpc.build_observable_problem",
                "artifact": str(obs_art),
            },
        },
    }
    with pytest.raises(
        ConfigConsistencyError, match=r"is an Observable model but estimator is 'neuro.filtering.AntiAliasEstimator'"
    ):
        validate_simulation_config(bad_cfg1)

    # Waveform checkpoint with ObservableEstimator
    bad_cfg2: dict[str, Any] = {
        **_plant(),
        "estimator": {
            "class_path": "neuro.filtering.ObservableEstimator",
            "dt": _PLANT_DT,
            "downsample": 200,
            "geometry": geom.model_dump(),
        },
        "controller": {
            "class_path": "neuro.control.mpc.TrajOptMPCController",
            "dt": _PLANT_DT * _DOWNSAMPLE,
            "problem": {
                "class_path": "neuro.control.mpc.build_waveform_problem",
                "artifact": str(wave_art),
            },
        },
    }
    with pytest.raises(
        ConfigConsistencyError, match=r"is a waveform model but estimator is 'neuro.filtering.ObservableEstimator'"
    ):
        validate_simulation_config(bad_cfg2)
