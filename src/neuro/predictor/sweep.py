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
from neuro.predictor.gradient import TrainingTimeLimitError
from neuro.predictor.plotting import plot_rollout_comparison, plot_training_curves
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

    Shared body of the two sweep objectives; projected runtime, NaN Losses, and memory errors prune trials.
    """
    trial_dir.mkdir(parents=True, exist_ok=True)
    with (trial_dir / "trial_config.yaml").open("w") as f:
        yaml.dump(config.model_dump(exclude={"sweep"}), f)
    try:
        result = train(config, data_files, seed_offset=trial.number)
    except TrainingTimeLimitError as exc:
        trial.set_user_attr("pruning_reason", "training_time")
        trial.set_user_attr("completed_epochs", exc.completed_epochs)
        trial.set_user_attr("elapsed_training_s", exc.elapsed_s)
        trial.set_user_attr("projected_training_s", exc.projected_s)
        trial.set_user_attr("training_time_limit_s", exc.limit_s)
        raise optuna.TrialPruned(str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        err_msg = str(exc).lower()
        if "nan" in err_msg or "out of memory" in err_msg or "cuda" in err_msg:
            raise optuna.TrialPruned from exc
        raise
    result.save(trial_dir)
    plot_training_curves(result, trial_dir / "loss_curve.png")
    plot_rollout_comparison(result, trial_dir / "comparison.png")
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
        """Sample active CNN dimensions, train the resolved config, and record every objective candidate."""
        model_overrides = {
            name: spec.suggest(trial, name)
            for name, spec in self.sweep.model.items()
            if name not in {"tapering_ratio", "head_width"}
        }
        is_cnn = model_overrides.get("architecture", self.cfg.model.architecture) == "cnn"
        for name, inactive, canonical in (
            ("tapering_ratio", model_overrides.get("depth", self.cfg.model.depth) == 1, 1.0),
            ("head_width", model_overrides.get("head_depth", self.cfg.model.head_depth) == 0, "trunk"),
        ):
            if name in self.sweep.model:
                model_overrides[name] = (
                    canonical if is_cnn and inactive else self.sweep.model[name].suggest(trial, name)
                )
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
        """Run the configured trial budget with seeded multivariate TPE and timeout-resilient SQLite storage."""
        study_name = "nn_predictor_sweep"
        db_path = self.artifact_dir / f"{study_name}.db"
        storage = optuna.storages.RDBStorage(
            f"sqlite:///{db_path.resolve()}",
            engine_kwargs={"connect_args": {"timeout": 60}},
        )
        sampler = optuna.samplers.TPESampler(n_startup_trials=15, multivariate=True, seed=self.cfg.training.seed)
        study = optuna.create_study(
            study_name=study_name, storage=storage, direction="minimize", load_if_exists=True, sampler=sampler
        )
        study.optimize(self.objective, n_trials=self.sweep.n_trials)
        return study
