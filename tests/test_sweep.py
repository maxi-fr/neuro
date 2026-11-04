"""Seam 4 -- the sweep seam: config-named objective selection and per-trial candidate recording."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import optuna
import pytest

from neuro.config import (
    CategoricalParam,
    CurriculumMSESpec,
    FloatParam,
    IntParam,
    LossSpecs,
    ModelConfig,
    NNPredictorConfig,
    NNSweepConfig,
    SimulationConfig,
    StftGeometry,
    TrainingConfig,
)
from neuro.predictor import sweep as sweep_module
from neuro.predictor.gradient import TrainingTimeLimitError
from neuro.predictor.sweep import OptunaSweep

if TYPE_CHECKING:
    from pathlib import Path

_WAVE_DT, _T = 1e-3, 200
_WAVE_HORIZON = 3


def _wave_config(
    *,
    sweep_model: dict[str, Any] | None = None,
) -> NNPredictorConfig:
    """Build a small Waveform sweep configuration."""
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
                "losses": losses,
            }
        ),
        sweep=NNSweepConfig(model=sweep_model or {}),
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


def test_optuna_sweep_records_every_candidate_and_scores_the_named_objective(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The waveform arm records all Trainer candidates on the trial and returns the named one."""
    candidates = {"val_loss": 0.24}
    captured = _stub_train(monkeypatch, candidates)
    trial = optuna.trial.FixedTrial({})

    value = OptunaSweep(_wave_config(), ["sim_0.npz"], tmp_path).objective(trial)

    assert value == pytest.approx(0.24)
    assert trial.user_attrs == candidates
    assert captured[0].model.n_y == 2  # the base config reaches the trainer


def test_optuna_sweep_merges_suggested_params_into_the_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A trial's suggested search-space value reaches the trainer inside the merged config."""
    captured = _stub_train(monkeypatch, {"val_loss": 0.2})
    cfg = _wave_config(sweep_model={"depth": IntParam(type="int", low=0, high=3)})
    trial = optuna.trial.FixedTrial({"depth": 0})

    OptunaSweep(cfg, ["sim_0.npz"], tmp_path).objective(trial)

    assert captured[0].model.depth == 0
    assert captured[0].model.hidden_size == 4  # untouched base values survive the merge


def test_sweep_runs_25_trials_with_15_seeded_random_startup_trials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The persisted sweep starts with fifteen random trials before using multivariate TPE."""
    cfg = _wave_config()
    sweep = OptunaSweep(cfg, [], tmp_path)

    def objective(trial: optuna.trial.Trial) -> float:
        """Score two interacting parameters without training a Predictor."""
        x = trial.suggest_float("x", -1.0, 1.0)
        y = trial.suggest_float("y", -1.0, 1.0)
        return (x - y) ** 2 + x**2

    monkeypatch.setattr(sweep, "objective", objective)
    study = sweep.run()
    random_study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=cfg.training.seed))
    random_study.optimize(objective, n_trials=16)

    assert len(study.trials) == 25
    assert all(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
    assert [trial.params for trial in study.trials[:15]] == [trial.params for trial in random_study.trials[:15]]
    assert study.trials[15].params != random_study.trials[15].params
    assert isinstance(study.sampler, optuna.samplers.TPESampler)
    assert study.sampler._multivariate  # noqa: SLF001 -- Optuna has no public accessor for this sampler setting
    assert (tmp_path / "nn_predictor_sweep.db").is_file()


def test_objective_missing_from_candidates_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A config-named objective the trained result cannot report fails loudly, not with a KeyError."""
    _stub_train(monkeypatch, {})

    with pytest.raises(ValueError, match="val_loss"):
        OptunaSweep(_wave_config(), ["sim_0.npz"], tmp_path).objective(optuna.trial.FixedTrial({}))


def test_observable_optuna_sweep_ranks_on_val_loss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Rank the Observable trial on its validation Loss."""
    candidates = {"val_loss": 0.042}
    _stub_train(monkeypatch, candidates)
    trial = optuna.trial.FixedTrial({})

    geometry = StftGeometry(n_segment=64, n_hop=16, band_hz=[4.0, 30.0], n_bin_pool=2, kernel_width=5)
    fs_frame = 250.0 / geometry.n_hop
    span_s = 4 / fs_frame
    losses = LossSpecs(curriculum_mse=CurriculumMSESpec(weight=1.0, span_s=span_s, curr_start=0, curr_end=2))
    cfg = NNPredictorConfig(
        simulation=SimulationConfig(dt=0.004, downsample=1),
        model=ModelConfig(n_y=2, n_u=8, hidden_size=4),
        training=TrainingConfig.model_validate(
            {
                "epochs": 3,
                "batch_size": 64,
                "learning_rate": 1e-2,
                "weight_decay": 0.0,
                "train_split": 0.5,
                "seed": 21,
                "patience": 50,
                "losses": losses,
            }
        ),
        observable=geometry,
        sweep=NNSweepConfig(objective="val_loss"),
    )

    value = OptunaSweep(cfg, ["sim_0.npz"], tmp_path).objective(trial)
    assert value == pytest.approx(0.042)
    assert trial.user_attrs == candidates


@pytest.mark.parametrize("architecture", ["cnn", "mlp"])
@pytest.mark.parametrize(("depth", "head_depth"), [(1, 0), (1, 1), (2, 0), (2, 1)])
def test_cnn_sweep_only_samples_active_tapering_and_head_width(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, architecture: Literal["cnn", "mlp"], depth: int, head_depth: int
) -> None:
    """CNN trials omit inactive dimensions regardless of config order and save canonical settings."""
    captured = _stub_train(monkeypatch, {"val_loss": 0.1})
    cfg = _wave_config(
        sweep_model={
            "head_width": CategoricalParam(type="categorical", choices=["trunk", "geometric_mean"]),
            "tapering_ratio": FloatParam(type="float", low=0.5, high=1.0),
            "depth": IntParam(type="int", low=1, high=2),
            "head_depth": IntParam(type="int", low=0, high=1),
        }
    )
    cfg = cfg.model_copy(update={"model": ModelConfig(architecture=architecture, n_y=5)})
    params: dict[str, Any] = {"depth": depth, "head_depth": head_depth}
    if architecture != "cnn" or depth > 1:
        params["tapering_ratio"] = 0.5
    if architecture != "cnn" or head_depth > 0:
        params["head_width"] = "geometric_mean"
    trial = optuna.trial.FixedTrial(params)

    OptunaSweep(cfg, ["sim_0.npz"], tmp_path).objective(trial)

    assert trial.params == params
    assert captured[0].model.tapering_ratio == params.get("tapering_ratio", 1.0)
    assert captured[0].model.head_width == params.get("head_width", "trunk")


def test_training_time_limit_prunes_trial_and_study_continues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Runtime projection records an Optuna prune with its evidence and permits the next trial."""
    calls = []

    def fake_train(cfg: object, data_files: list[str], *, seed_offset: int = 0) -> _StubResult:
        """Simulate a slow first training run followed by an affordable candidate."""
        calls.append(seed_offset)
        if len(calls) == 1:
            raise TrainingTimeLimitError(3, 410.0, 29810.0, 25200.0)
        return _StubResult({"val_loss": 0.1})

    monkeypatch.setattr(sweep_module, "train", fake_train)
    sweep = OptunaSweep(_wave_config(), ["sim_0.npz"], tmp_path)
    study = optuna.create_study()
    study.optimize(sweep.objective, n_trials=2)

    assert [trial.state for trial in study.trials] == [
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.COMPLETE,
    ]
    assert study.trials[0].user_attrs == {
        "pruning_reason": "training_time",
        "completed_epochs": 3,
        "elapsed_training_s": 410.0,
        "projected_training_s": 29810.0,
        "training_time_limit_s": 25200.0,
    }
    assert not (tmp_path / "trial_0" / "model.npz").exists()
    assert calls == [0, 1]
