from __future__ import annotations

from typing import TYPE_CHECKING, Any

import optuna
import yaml
from simulate.config import deep_merge

from neuro.closed_loop_eval import evaluate_closed_loop_suppression
from neuro.config import (
    CLOSED_LOOP_OBJECTIVE,
    ModelConfig,
    NNPredictorConfig,
    NNSweepConfig,
    StftGeometry,
    TrainingConfig,
    expand_dotted_dict,
)
from neuro.predictor.train import train

if TYPE_CHECKING:
    from pathlib import Path

    from optuna.trial import BaseTrial

    from neuro.config import ClosedLoopEvalConfig


def score_trial(
    trial: BaseTrial,
    *,
    candidates: dict[str, float],
    objective: str,
    trial_dir: Path,
    closed_loop: ClosedLoopEvalConfig | None,
) -> float:
    """Record every candidate on the trial and return the config-named objective.

    ``closed_loop`` is a sweep-level candidate: when its evaluation section is configured it is
    computed on every trial (not only when named), so a finished study can be re-ranked on it.
    Raises ``ValueError`` when the named objective is not among the recorded candidates, which
    guards the per-fit candidate sets (``val_loss`` on a ridge fit that never trains).
    """
    recorded = dict(candidates)
    for name, value in recorded.items():
        trial.set_user_attr(name, value)
    if closed_loop is not None:
        score, summary = evaluate_closed_loop_suppression(trial_dir, closed_loop)
        trial.set_user_attr(CLOSED_LOOP_OBJECTIVE, score)
        for name, value in summary.items():
            trial.set_user_attr(name, value)
        recorded[CLOSED_LOOP_OBJECTIVE] = score
    if objective not in recorded:
        msg = (
            f"sweep.objective {objective!r} is not among this run's candidates ({sorted(recorded)}); "
            "the objective must be one the Trainer reports, or 'closed_loop'."
        )
        raise ValueError(msg)
    return recorded[objective]


def _run_trial(
    config: NNPredictorConfig,
    data_files: list[str],
    trial: BaseTrial,
    trial_dir: Path,
    sweep: NNSweepConfig,
) -> float:
    """Train one trial at ``config``, persist it under ``trial_dir`` and return the named objective.

    Shared body of the two sweep objectives; a NaN training loss prunes the trial instead of
    failing the study.
    """
    trial_dir.mkdir(parents=True, exist_ok=True)
    with (trial_dir / "trial_config.yaml").open("w") as f:
        yaml.dump(config.model_dump(exclude={"sweep"}), f)
    try:
        result = train(config, data_files, seed_offset=trial.number)
    except ValueError as exc:
        if "NaN" in str(exc):
            raise optuna.TrialPruned from exc
        raise
    result.save(trial_dir)
    return score_trial(
        trial,
        candidates=result.candidates,
        objective=sweep.objective,
        trial_dir=trial_dir,
        closed_loop=sweep.closed_loop,
    )


class OptunaSweep:
    """Optuna sweep seam serving the two NN predictor kinds: waveform and observable.

    Each trial suggests the ``sweep.model`` and ``sweep.training`` search dimensions, merges them
    into the base config, trains through the unified entry point, records every Trainer candidate
    on the trial, and returns the ``sweep.objective`` the study minimizes.
    """

    cfg: NNPredictorConfig
    sweep: NNSweepConfig
    data_files: list[str]
    artifact_dir: Path

    def __init__(self, cfg: NNPredictorConfig, data_files: list[str], artifact_dir: Path) -> None:
        """Store the base config, the data files and the artifact directory for trial outputs."""
        if cfg.sweep is None:
            msg = "OptunaSweep requires a config with a 'sweep' section."
            raise ValueError(msg)
        self.cfg = cfg
        self.sweep = cfg.sweep
        self.data_files = data_files
        self.artifact_dir = artifact_dir

    def objective(self, trial: BaseTrial) -> float:
        """Train one trial's hyperparameters and return the named objective, recording every candidate."""
        model_overrides = {name: spec.suggest(trial, name) for name, spec in self.sweep.model.items()}
        training_overrides = {name: spec.suggest(trial, name) for name, spec in self.sweep.training.items()}
        update_dict: dict[str, Any] = {
            "model": ModelConfig.model_validate(
                deep_merge(self.cfg.model.model_dump(), expand_dotted_dict(model_overrides))
            ),
            "training": TrainingConfig.model_validate(
                deep_merge(self.cfg.training.model_dump(), expand_dotted_dict(training_overrides))
            ),
        }
        if self.sweep.observable and self.cfg.observable is not None:
            observable_overrides = {name: spec.suggest(trial, name) for name, spec in self.sweep.observable.items()}
            update_dict["observable"] = StftGeometry.model_validate(
                deep_merge(self.cfg.observable.model_dump(), expand_dotted_dict(observable_overrides))
            )
        config = self.cfg.model_copy(update=update_dict)
        trial_dir = self.artifact_dir / f"trial_{trial.number}"
        return _run_trial(config, self.data_files, trial, trial_dir, self.sweep)

    def run(self) -> optuna.Study:
        """Create or resume the sqlite-backed study and run ``sweep.n_trials`` evaluations."""
        study_name = "nn_predictor_sweep"
        db_path = self.artifact_dir / f"{study_name}.db"
        storage = f"sqlite:///{db_path.resolve()}"
        study = optuna.create_study(study_name=study_name, storage=storage, direction="minimize", load_if_exists=True)
        study.optimize(self.objective, n_trials=self.sweep.n_trials)
        return study
