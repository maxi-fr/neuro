"""Seam 4 -- the sweep seam: config-named objective selection and per-trial candidate recording."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import optuna
import pytest

from neuro.config import (
    ClosedLoopEvalConfig,
    CurriculumMSESpec,
    ESNModelConfig,
    ESNPredictorConfig,
    ESNSweepConfig,
    ESNTrainingConfig,
    FloatParam,
    IntParam,
    LossSpecs,
    ModelConfig,
    NNPredictorConfig,
    NNSweepConfig,
    ObservableSpec,
    SimulationConfig,
    StftGeometry,
    TrainingConfig,
)
from neuro.predictor import sweep as sweep_module
from neuro.predictor.sweep import GridSweep, OptunaSweep

if TYPE_CHECKING:
    from pathlib import Path

_WAVE_DT, _T = 1e-3, 200
_OBS_DT, _OBS_T = 0.02, 400
_WAVE_HORIZON = 3
_OBS_HORIZON = 16
_OBS_GEOMETRY = StftGeometry(n_segment=8, n_hop=4)


def _wave_config(
    *,
    objective: str = "log_energy",
    sweep_model: dict[str, Any] | None = None,
    closed_loop: ClosedLoopEvalConfig | None = None,
) -> NNPredictorConfig:
    """A tiny but complete waveform config whose sweep names ``objective``."""
    span_s = _WAVE_HORIZON / _WAVE_DT
    losses = LossSpecs(curriculum_mse=CurriculumMSESpec(weight=1.0, span_s=span_s, curr_start=0, curr_end=2))
    return NNPredictorConfig(
        simulation=SimulationConfig(dt=_WAVE_DT, downsample=1),
        model=ModelConfig(n_y=2, n_u=2, hidden_size=4),
        training=TrainingConfig.model_validate(
            {
                "epochs": 3,
                "batch_size": 64,
                "learning_rate": 1e-2,
                "weight_decay": 0.0,
                "train_split": 0.5,
                "seed": 21,
                "patience": 50,
                "eval_horizon_s": span_s,
                "losses": losses,
            }
        ),
        sweep=NNSweepConfig(objective=objective, model=sweep_model or {}, closed_loop=closed_loop),
    )


def _obs_config(*, objective: str = "val_loss", closed_loop: ClosedLoopEvalConfig | None = None) -> NNPredictorConfig:
    """A tiny but complete observable config whose sweep names ``objective``."""
    return NNPredictorConfig(
        simulation=SimulationConfig(dt=_OBS_DT, downsample=1),
        model=ModelConfig(n_y=3, n_u=2),
        training=TrainingConfig.model_validate(
            {
                "epochs": 3,
                "batch_size": 64,
                "learning_rate": 1e-2,
                "weight_decay": 0.0,
                "train_split": 0.5,
                "seed": 21,
                "patience": 50,
                "eval_horizon_s": _OBS_HORIZON * _OBS_DT,
            }
        ),
        observable=ObservableSpec(
            horizon=_OBS_HORIZON,
            stft=_OBS_GEOMETRY,
            z_dim=6,
            lift_hidden=8,
            lift_depth=1,
            transition_hidden=8,
            transition_depth=1,
        ),
        sweep=NNSweepConfig(objective=objective, closed_loop=closed_loop),
    )


def _esn_config(
    *, objective: str = "rollout_nmse", closed_loop: ClosedLoopEvalConfig | None = None
) -> ESNPredictorConfig:
    """A tiny but complete ESN config whose sweep names ``objective``."""
    return ESNPredictorConfig(
        simulation=SimulationConfig(dt=_WAVE_DT, downsample=1),
        model=ESNModelConfig(
            reservoir_size=50,
            spectral_radius=0.9,
            leak_rate=0.1,
            density=0.1,
            input_scaling=0.5,
            priming_steps=10,
            ridge_lambda=1e-3,
            noise_sigma=0.0,
            horizon=8,
        ),
        training=ESNTrainingConfig(train_split=0.5, seed=21),
        sweep=ESNSweepConfig(objective=objective, closed_loop=closed_loop),
    )


def _closed_loop_cfg() -> ClosedLoopEvalConfig:
    """A minimal closed-loop evaluation section; the evaluation itself is stubbed in these tests."""
    return ClosedLoopEvalConfig(
        simulation_config="sim.yaml",
        seeds=[69],
        t_end=5.0,
        seizure_ptp_mv=5.0,
        max_seizing_regions=5,
    )


class _StubResult:
    """A ``train`` stand-in carrying candidates and a no-op save."""

    def __init__(self, candidates: dict[str, float]) -> None:
        """Store the candidates this fake Trainer reports."""
        self.candidates = candidates

    def save(self, artifact_dir: Path) -> None:
        """No-op: the wiring tests never train, so nothing is written."""


def _stub_train(monkeypatch: pytest.MonkeyPatch, candidates: dict[str, float]) -> list[Any]:
    """Replace ``sweep.train`` with a stub and capture the configs it is handed."""

    def fake_train(cfg: object, data_files: list[str], *, seed_offset: int = 0) -> _StubResult:
        captured.append(cfg)
        return _StubResult(candidates)

    captured: list[Any] = []
    monkeypatch.setattr(sweep_module, "train", fake_train)
    return captured


def _stub_closed_loop(monkeypatch: pytest.MonkeyPatch, score: float = 0.07) -> None:
    """Replace the closed-loop evaluation with a fixed score and summary."""

    def fake_eval(trial_dir: Path, eval_cfg: ClosedLoopEvalConfig) -> tuple[float, dict[str, float]]:
        return score, {"seizure_burden": score, "suppressed_seeds": 1.0, "total_seeds": 2.0}

    monkeypatch.setattr(sweep_module, "evaluate_closed_loop_suppression", fake_eval)


def test_optuna_sweep_records_every_candidate_and_scores_the_named_objective(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The waveform arm records all Trainer candidates on the trial and returns the named one."""
    candidates = {"log_energy": 0.31, "val_loss": 0.24, "rollout_nmse": 0.19}
    captured = _stub_train(monkeypatch, candidates)
    trial = optuna.trial.FixedTrial({})

    value = OptunaSweep(_wave_config(objective="rollout_nmse"), ["sim_0.npz"], tmp_path).objective(trial)

    assert value == pytest.approx(0.19)
    assert trial.user_attrs == candidates
    assert captured[0].model.n_y == 2  # the base config reaches the trainer


def test_optuna_sweep_merges_suggested_params_into_the_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A trial's suggested search-space value reaches the trainer inside the merged config."""
    captured = _stub_train(monkeypatch, {"log_energy": 0.1, "val_loss": 0.2, "rollout_nmse": 0.3})
    cfg = _wave_config(sweep_model={"depth": IntParam(type="int", low=0, high=3)})
    trial = optuna.trial.FixedTrial({"depth": 0})

    OptunaSweep(cfg, ["sim_0.npz"], tmp_path).objective(trial)

    assert captured[0].model.depth == 0
    assert captured[0].model.hidden_size == 4  # untouched base values survive the merge


def test_optuna_sweep_observable_arm_scores_its_named_objective(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The observable arm records ``{val_loss, val_log_mse}`` and returns the named one."""
    candidates = {"val_loss": 0.18, "val_log_mse": 0.42}
    captured = _stub_train(monkeypatch, candidates)
    trial = optuna.trial.FixedTrial({})

    value = OptunaSweep(_obs_config(objective="val_log_mse"), ["sim_0.npz"], tmp_path).objective(trial)

    assert value == pytest.approx(0.42)
    assert trial.user_attrs == candidates
    assert captured[0].observable is not None


def test_grid_sweep_records_every_candidate_and_scores_the_named_objective(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ESN cell records both Trainer candidates and returns the named one."""
    candidates = {"rollout_nmse": 0.15, "log_energy": 0.23}
    captured = _stub_train(monkeypatch, candidates)
    trial = optuna.trial.FixedTrial({"spectral_radius": 1.1, "leak_rate": 0.2, "density": 0.05})

    value = GridSweep(_esn_config(objective="log_energy"), ["sim_0.npz"], tmp_path).objective(
        trial, reservoir_size=100, ridge_lambda=0.01
    )

    assert value == pytest.approx(0.23)
    assert trial.user_attrs == candidates
    cfg = captured[0]
    assert cfg.model.reservoir_size == 100  # the outer-grid cell reaches the trainer
    assert cfg.model.ridge_lambda == 0.01
    assert cfg.model.spectral_radius == 1.1  # the inner Optuna suggestion reaches the trainer
    assert cfg.model.leak_rate == 0.2
    assert cfg.model.density == 0.05


@pytest.mark.parametrize(
    ("config", "sweep_factory"),
    [
        (_wave_config(objective="closed_loop", closed_loop=_closed_loop_cfg()), OptunaSweep),
        (_obs_config(objective="closed_loop", closed_loop=_closed_loop_cfg()), OptunaSweep),
        (_esn_config(objective="closed_loop", closed_loop=_closed_loop_cfg()), GridSweep),
    ],
)
def test_closed_loop_objective_works_for_all_three_predictor_kinds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: object,
    sweep_factory: type,
) -> None:
    """``closed_loop`` scores the sweep-level evaluation, for waveform, observable and ESN alike."""
    candidates = {"log_energy": 0.31, "val_loss": 0.24, "rollout_nmse": 0.19}
    _stub_train(monkeypatch, candidates)
    _stub_closed_loop(monkeypatch)
    trial = optuna.trial.FixedTrial({"spectral_radius": 0.9, "leak_rate": 0.1, "density": 0.1})

    sweep = sweep_factory(config, ["sim_0.npz"], tmp_path)
    if isinstance(sweep, GridSweep):
        value = sweep.objective(trial, reservoir_size=50, ridge_lambda=1e-3)
    else:
        value = sweep.objective(trial)

    assert value == pytest.approx(0.07)
    assert trial.user_attrs["closed_loop"] == pytest.approx(0.07)
    assert trial.user_attrs["seizure_burden"] == pytest.approx(0.07)
    assert trial.user_attrs["log_energy"] == pytest.approx(0.31)  # trainer candidates stay recorded


def test_objective_missing_from_candidates_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A config-named objective the trained result cannot report fails loudly, not with a KeyError."""
    _stub_train(monkeypatch, {"rollout_nmse": 0.15, "log_energy": 0.23})  # e.g. a ridge fit, no epoch loop

    with pytest.raises(ValueError, match="val_loss"):
        OptunaSweep(_wave_config(objective="val_loss"), ["sim_0.npz"], tmp_path).objective(optuna.trial.FixedTrial({}))


def test_grid_sweep_run_returns_one_best_per_outer_cell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``run`` executes the outer grid and returns each cell's best trial with its recorded candidates."""
    candidates = {"rollout_nmse": 0.15, "log_energy": 0.23}
    _stub_train(monkeypatch, candidates)
    cfg = _esn_config(objective="log_energy").model_copy(
        update={"sweep": ESNSweepConfig(objective="log_energy", reservoir_sizes=[50], lambdas=[1e-3], n_trials=1)}
    )

    results = GridSweep(cfg, ["sim_0.npz"], tmp_path).run()

    assert len(results) == 1
    result = results[0]
    assert result.reservoir_size == 50
    assert result.ridge_lambda == 1e-3
    assert result.value == pytest.approx(0.23)
    assert result.candidates["rollout_nmse"] == pytest.approx(0.15)
    assert result.candidates["log_energy"] == pytest.approx(0.23)
