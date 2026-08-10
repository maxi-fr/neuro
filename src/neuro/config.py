from __future__ import annotations

import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from typing import Self

    import optuna


def parse_array(val: Any) -> Any:  # noqa: ANN401
    """Parse a configuration value that might be an array or a path to an array."""
    if isinstance(val, str) and (val.endswith((".npy", ".npz", ".npv"))):
        loaded = np.load(val)
        if isinstance(loaded, np.ndarray):
            return loaded

        keys = list(loaded.keys())
        if keys:
            return loaded[keys[0]]
        msg = f"NPZ file {val} is empty"
        raise ValueError(msg)
    return val


class StrictConfig(BaseModel):
    """Base for config schemas: reject unknown keys, coerce values, and stay immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())


class ModelConfig(StrictConfig):
    """MLP predictor architecture settings."""

    n_y: int = Field(default=5, ge=1)
    n_u: int = Field(default=5, ge=1)
    horizon: int = Field(default=5, ge=1)
    hidden_size: int = Field(default=128, ge=1)
    depth: int = Field(default=2, ge=0)
    activation: str = "relu"
    latent_dim: int | None = Field(default=None, ge=1)


class SimulationConfig(StrictConfig):
    """Trajectory-loading settings shared by the NN-predictor pipeline."""

    dt: float = Field(default=1e-4, gt=0)
    downsample: int = Field(default=1, ge=1)
    n_steps: int | None = Field(default=None, ge=1)
    data_path: str | None = None


class TrainingConfig(StrictConfig):
    """Optimisation and scaling settings for the NN predictor."""

    epochs: int = Field(default=100, ge=1)
    batch_size: int = Field(default=128, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    train_split: float = Field(default=0.8, gt=0, lt=1)
    curriculum_start_epoch: int = Field(default=0, ge=0)
    curriculum_end_epoch: int = Field(default=80, ge=0)
    curriculum_max_steps: int | None = Field(default=None, ge=1)
    seed: int = Field(default=69, ge=0)
    w_psd: float = Field(default=0.0, ge=0)
    w_fc: float = Field(default=0.0, ge=0)
    patience: int = Field(default=50, ge=1)
    scaler: Literal["standard", "robust"] = "standard"
    global_scaling: bool = False


class CategoricalParam(StrictConfig):
    """Categorical Optuna search dimension."""

    type: Literal["categorical"]
    choices: list[Any]

    def suggest(self, trial: optuna.Trial, name: str) -> Any:  # noqa: ANN401
        """Suggest a value for ``name`` from the fixed set of choices."""
        return trial.suggest_categorical(name, self.choices)


class _RangeParam(StrictConfig):
    """Base for numeric search dimensions: enforces an ordered ``[low, high]`` range."""

    low: float
    high: float

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        if self.high < self.low:
            msg = f"high ({self.high}) must be >= low ({self.low})"
            raise ValueError(msg)
        return self


class IntParam(_RangeParam):
    """Integer Optuna search dimension."""

    type: Literal["int"]
    low: int
    high: int
    log: bool = False

    def suggest(self, trial: optuna.Trial, name: str) -> int:
        """Suggest an integer for ``name`` within ``[low, high]``."""
        return trial.suggest_int(name, self.low, self.high, log=self.log)


class FloatParam(_RangeParam):
    """Floating-point Optuna search dimension."""

    type: Literal["float"]
    log: bool = False

    def suggest(self, trial: optuna.Trial, name: str) -> float:
        """Suggest a float for ``name`` within ``[low, high]``."""
        return trial.suggest_float(name, self.low, self.high, log=self.log)


class LogUniformParam(_RangeParam):
    """Log-uniform Optuna search dimension."""

    type: Literal["loguniform"]
    low: float = Field(gt=0)
    high: float = Field(gt=0)

    def suggest(self, trial: optuna.Trial, name: str) -> float:
        """Suggest a float for ``name`` within ``[low, high]``, sampled log-uniformly."""
        return trial.suggest_float(name, self.low, self.high, log=True)


ParamSpec = Annotated[
    CategoricalParam | IntParam | FloatParam | LogUniformParam,
    Field(discriminator="type"),
]


class ClosedLoopEvalConfig(StrictConfig):
    """Settings for closed-loop evaluation in Optuna hyperparameter sweeps."""

    simulation_config: str
    seeds: list[int]
    t_end: float = Field(gt=0)
    seizure_ptp_mv: float = Field(gt=0)
    max_seizing_regions: int = Field(ge=0)


class SweepConfig(StrictConfig):
    """Optuna sweep settings: trial count, output dir and per-group search spaces."""

    n_trials: int = Field(default=20, ge=1)
    artifact: str | None = None
    model: dict[str, ParamSpec] = Field(default_factory=dict)
    training: dict[str, ParamSpec] = Field(default_factory=dict)
    closed_loop: ClosedLoopEvalConfig | None = None


class NNPredictorConfig(StrictConfig):
    """Fully-resolved, validated configuration for the NN-predictor pipeline."""

    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    artifact: str | None = None
    sweep: SweepConfig | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> NNPredictorConfig:
        """Build a typed config from a raw YAML mapping, rejecting unknown keys everywhere."""
        return cls.model_validate({} if data is None else data)

    @model_validator(mode="after")
    def _validate_curriculum_epochs(self) -> Self:
        if self.training.curriculum_end_epoch < self.training.curriculum_start_epoch:
            msg = (
                f"curriculum_end_epoch ({self.training.curriculum_end_epoch}) must be "
                f">= curriculum_start_epoch ({self.training.curriculum_start_epoch})."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_sweep_exclusivity_and_keys(self) -> Self:
        if self.sweep is None:
            return self

        sweep_model_keys = set(self.sweep.model.keys())
        invalid_model_keys = sweep_model_keys - set(self.model.__class__.model_fields.keys())
        if invalid_model_keys:
            msg = f"Keys {sorted(invalid_model_keys)} in 'sweep.model' are not valid 'model' parameters."
            raise ValueError(msg)

        overlap_model = sweep_model_keys & self.model.model_fields_set
        if overlap_model:
            msg = f"Parameters cannot be defined in both regular 'model' and 'sweep.model'. Overlap: {sorted(overlap_model)}"
            raise ValueError(msg)

        sweep_training_keys = set(self.sweep.training.keys())
        invalid_training_keys = sweep_training_keys - set(self.training.__class__.model_fields.keys())
        if invalid_training_keys:
            msg = f"Keys {sorted(invalid_training_keys)} in 'sweep.training' are not valid 'training' parameters."
            raise ValueError(msg)

        overlap_training = sweep_training_keys & self.training.model_fields_set
        if overlap_training:
            msg = f"Parameters cannot be defined in both regular 'training' and 'sweep.training'. Overlap: {sorted(overlap_training)}"
            raise ValueError(msg)

        return self


def load_config(path: Path) -> NNPredictorConfig:
    """Load and strictly validate an NN-predictor YAML config from ``path``."""
    if not path.exists():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)
    with path.open() as f:
        raw = yaml.safe_load(f)
    return NNPredictorConfig.from_dict(raw)


def resolve_data_files(config: NNPredictorConfig, data_path_override: str | None = None) -> list[str]:
    """Resolve and validate the ``.npz`` training files from the config or an override."""
    data_path_str = data_path_override or config.simulation.data_path
    if not data_path_str:
        msg = "data_path not specified in config or arguments."
        raise ValueError(msg)
    data_path = Path(data_path_str)
    if not data_path.is_dir():
        msg = f"data_path is not a valid directory: {data_path}"
        raise ValueError(msg)
    data_files = sorted(str(p) for p in data_path.glob("*.npz"))
    if not data_files:
        msg = f"No .npz data files found in: {data_path}"
        raise ValueError(msg)
    return data_files


def resolve_artifact_dir(artifact: str | None, default_prefix: str) -> Path:
    """Resolve (and create) the artifact output directory."""
    if artifact is None:
        local_now = datetime.datetime.now(datetime.UTC).astimezone()
        artifact = f"artifacts/{default_prefix}_{local_now.strftime('%Y-%m-%d_%H-%M-%S')}"
    artifact_dir = Path(artifact)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir
