from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import optuna
import yaml
from simulate.config import deep_merge

from neuro.closed_loop_eval import evaluate_closed_loop_suppression
from neuro.config import (
    CLOSED_LOOP_OBJECTIVE,
    ESNModelConfig,
    ESNPredictorConfig,
    ESNSweepConfig,
    FloatParam,
    ModelConfig,
    NNPredictorConfig,
    NNSweepConfig,
    ParamSpec,
    TrainingConfig,
    expand_dotted_dict,
)
from neuro.predictor.train import train

if TYPE_CHECKING:
    from pathlib import Path

    from optuna.trial import BaseTrial

    from neuro.config import ClosedLoopEvalConfig


# The continuous reservoir hyperparameters the ESN sweep searches when ``sweep.model`` is empty.
# ``noise_sigma`` is deliberately absent: the unified entry point rejects it, so searching it
# would prune every suggested trial above zero.
_DEFAULT_ESN_SEARCH: dict[str, ParamSpec] = {
    "spectral_radius": FloatParam(type="float", low=0.1, high=1.5),
    "leak_rate": FloatParam(type="float", low=0.01, high=1.0),
    "density": FloatParam(type="float", low=0.01, high=0.5),
}


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
    config: NNPredictorConfig | ESNPredictorConfig,
    data_files: list[str],
    trial: BaseTrial,
    trial_dir: Path,
    sweep: NNSweepConfig | ESNSweepConfig,
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
        config = self.cfg.model_copy(
            update={
                "model": ModelConfig.model_validate(
                    deep_merge(self.cfg.model.model_dump(), expand_dotted_dict(model_overrides))
                ),
                "training": TrainingConfig.model_validate(
                    deep_merge(self.cfg.training.model_dump(), expand_dotted_dict(training_overrides))
                ),
            }
        )
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


@dataclass(frozen=True)
class GridResult:
    """The best inner-Optuna outcome of one outer-grid (reservoir_size, ridge_lambda) cell.

    ``candidates`` holds every candidate recorded on the best trial, so a finished grid can be
    re-ranked on an objective other than the one minimized without re-running it.
    """

    reservoir_size: int
    ridge_lambda: float
    value: float
    params: dict[str, Any]
    candidates: dict[str, float]


class GridSweep:
    """Grid-plus-Optuna sweep seam serving the ESN predictor.

    The outer grid iterates ``sweep.reservoir_sizes`` x ``sweep.lambdas``; for each cell an
    in-memory Optuna study searches the continuous reservoir hyperparameters in ``sweep.model``
    (or the defaults). Every trial trains through the unified entry point, records every Trainer
    candidate, and returns the ``sweep.objective`` the study minimizes.
    """

    cfg: ESNPredictorConfig
    sweep: ESNSweepConfig
    data_files: list[str]
    artifact_dir: Path

    def __init__(self, cfg: ESNPredictorConfig, data_files: list[str], artifact_dir: Path) -> None:
        """Store the base config, the data files and the artifact directory for trial outputs."""
        if cfg.sweep is None:
            msg = "GridSweep requires a config with a 'sweep' section."
            raise ValueError(msg)
        self.cfg = cfg
        self.sweep = cfg.sweep
        self.data_files = data_files
        self.artifact_dir = artifact_dir

    def objective(self, trial: BaseTrial, *, reservoir_size: int, ridge_lambda: float) -> float:
        """Train one cell trial at fixed (reservoir_size, ridge_lambda) and return the named objective."""
        search = self.sweep.model or _DEFAULT_ESN_SEARCH
        overrides = {name: spec.suggest(trial, name) for name, spec in search.items()}
        config = self.cfg.model_copy(
            update={
                "model": ESNModelConfig.model_validate(
                    deep_merge(
                        self.cfg.model.model_dump(),
                        {"reservoir_size": reservoir_size, "ridge_lambda": ridge_lambda, **overrides},
                    )
                )
            }
        )
        trial_dir = self.artifact_dir / f"n{reservoir_size}_lam{ridge_lambda:g}_trial_{trial.number}"
        return _run_trial(config, self.data_files, trial, trial_dir, self.sweep)

    def run(self) -> list[GridResult]:
        """Run the outer grid; one in-memory Optuna study per (reservoir_size, ridge_lambda) cell."""
        results: list[GridResult] = []
        for reservoir_size in self.sweep.reservoir_sizes:
            for ridge_lambda in self.sweep.lambdas:
                study = optuna.create_study(direction="minimize")
                study.optimize(
                    lambda trial, n=reservoir_size, lam=ridge_lambda: self.objective(
                        trial, reservoir_size=n, ridge_lambda=lam
                    ),
                    n_trials=self.sweep.n_trials,
                )
                best = study.best_trial
                value = best.value
                if value is None:  # every trial pruned
                    msg = f"all {self.sweep.n_trials} trials for N={reservoir_size}, lambda={ridge_lambda} were pruned."
                    raise RuntimeError(msg)
                results.append(
                    GridResult(
                        reservoir_size=reservoir_size,
                        ridge_lambda=ridge_lambda,
                        value=value,
                        params=dict(best.params),
                        candidates={k: float(v) for k, v in best.user_attrs.items() if isinstance(v, (int, float))},
                    )
                )
        return results
