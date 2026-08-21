"""Cover the observable training loop end to end: targets, artifact hand-off and the blind arm."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

from neuro.artifacts import load_any_artifact
from neuro.config import (
    ModelConfig,
    NNPredictorConfig,
    ObservableSpec,
    SimulationConfig,
    StftGeometry,
    TrainingConfig,
)
from neuro.observable import ObservableArtifact, log_observable
from neuro.predictor.data import prepare_datasets
from neuro.predictor.observable_train import build_targets, prepare_observable_data, train_observable

if TYPE_CHECKING:
    from pathlib import Path

_SEED = 13
_DT = 0.02
_T, _N_EEG, _N_CONTROLS = 400, 3, 2
_HORIZON = 16
_GEOMETRY = StftGeometry(n_segment=8, n_hop=4)


def _write_trajectories(tmp_path: Path) -> list[str]:
    """Two synthetic trajectories -- sinusoids plus a control-driven drift -- in the loader's npz layout."""
    rng = np.random.default_rng(_SEED)
    files = []
    for i in range(2):
        u = np.cumsum(rng.standard_normal((_T, _N_CONTROLS)) * 0.1, axis=0)
        phase = np.arange(_T)[:, None] * _DT * 2 * np.pi * np.array([7.0, 11.0, 13.0])
        y = np.sin(phase) + 0.3 * u[:, :1] + 0.05 * rng.standard_normal((_T, _N_EEG))
        path = tmp_path / f"sim_{i:03d}.npz"
        np.savez(path, **{"sensor_0.y_mea": y, "controller.u": u})  # ty: ignore[invalid-argument-type]
        files.append(str(path))
    return files


def _config(*, control_blind: bool = False) -> NNPredictorConfig:
    """A tiny but complete observable-predictor config."""
    return NNPredictorConfig(
        simulation=SimulationConfig(dt=_DT, downsample=1),
        model=ModelConfig(n_y=3, n_u=2),
        training=TrainingConfig.model_validate(
            {
                "epochs": 3,
                "batch_size": 64,
                "learning_rate": 1e-2,
                "weight_decay": 0.0,
                "train_split": 0.5,
                "seed": _SEED,
                "patience": 50,
                "eval_horizon_s": _HORIZON * _DT,
            }
        ),
        observable=ObservableSpec(
            horizon=_HORIZON,
            stft=_GEOMETRY,
            z_dim=6,
            lift_hidden=8,
            lift_depth=1,
            transition_hidden=8,
            transition_depth=1,
            control_blind=control_blind,
        ),
    )


def test_training_targets_are_the_shared_observable_of_the_true_future(tmp_path: Path) -> None:
    """The targets the trainer regresses on are ``log_observable`` of the same windows, flattened."""
    files = _write_trajectories(tmp_path)
    data = prepare_datasets(files, None, 1, 3, 2, _HORIZON, _DT, 0.5, scaler="standard", global_scaling=False)

    targets = build_targets(data.Y_train, data.y_std, _GEOMETRY, horizon=_HORIZON, n_channels=_N_EEG, fs=1.0 / _DT)
    n_frames = _GEOMETRY.n_frames(_HORIZON, 1.0 / _DT)
    assert targets.shape == (data.Y_train.shape[0], n_frames, _N_EEG * _GEOMETRY.n_values(1.0 / _DT))

    raw = data.y_std.inverse_transform(data.Y_train[:3].reshape(3, _HORIZON, _N_EEG))
    expected = log_observable(raw, _GEOMETRY, 1.0 / _DT).reshape(3, n_frames, -1)
    np.testing.assert_allclose(targets[:3], expected, rtol=1e-12, atol=1e-12)


def test_train_observable_writes_a_loadable_artifact(tmp_path: Path) -> None:
    """A run produces an artifact that ``load_any_artifact`` dispatches on and stats that persist."""
    result = train_observable(_config(), _write_trajectories(tmp_path))

    assert len(result.train_losses) == 3
    assert result.n_independent_samples > 0
    assert result.artifact.geometry == _GEOMETRY
    assert result.artifact.horizon == _HORIZON

    result.save(tmp_path)
    loaded = load_any_artifact(tmp_path / "model")
    assert isinstance(loaded, ObservableArtifact)
    assert loaded.geometry == _GEOMETRY
    stats = json.loads((tmp_path / "training_stats.json").read_text())
    assert stats["n_independent_samples"] == result.n_independent_samples
    assert stats["du_sensitivity"] == result.du_sensitivity


def test_control_blind_arm_zeroes_the_future_control_block(tmp_path: Path) -> None:
    """The blind arm is the same architecture and targets on zeroed future controls.

    Whether it scores worse than the full model is the decision 11 gate's measurement, not a unit
    test: zeroing the *inputs* does not zero the weights, so the sensitivity ordering depends on
    trained weights. All that is pinned here is that the ablation touches the control block alone.
    """
    files = _write_trajectories(tmp_path)
    full = prepare_observable_data(_config(), files)
    blind = prepare_observable_data(_config(control_blind=True), files)

    n_hist = 3 * _N_EEG + 2 * _N_CONTROLS
    assert np.count_nonzero(blind.x_train[:, n_hist:]) == 0
    np.testing.assert_array_equal(blind.x_train[:, :n_hist], full.x_train[:, :n_hist])
    np.testing.assert_array_equal(blind.targets_train, full.targets_train)
