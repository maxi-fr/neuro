from __future__ import annotations

import datetime
import types
import typing
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from neuro.metrics import DEFAULT_HOP_S, METRICS
from neuro.provenance import check_excitation_alignment

if TYPE_CHECKING:
    from typing import Self

    import optuna

_MIN_PSD_SPAN_STEPS = 2


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
    hidden_size: int = Field(default=128, ge=1)
    depth: int = Field(default=2, ge=0)
    activation: Literal["relu", "tanh", "softplus"] = "relu"


class SimulationConfig(StrictConfig):
    """Trajectory-loading settings shared by the NN-predictor pipeline."""

    dt: float = Field(default=1e-4, gt=0)
    downsample: int = Field(default=1, ge=1)
    n_steps: int | None = Field(default=None, ge=1)
    data_path: str | None = None
    cutoff_hz: float | None = Field(default=None, gt=0)


class LossSpec(StrictConfig):
    """Base for one additive loss term: its weight, its epoch gate and its rollout span."""

    weight: float = Field(ge=0)
    span_s: float = Field(gt=0)
    start_epoch: int = Field(default=0, ge=0)

    def span_steps(self, fs: float) -> int:
        """Rollout length in steps implied by span_s at fs, >= 1."""
        return max(1, round(self.span_s * fs))


class CurriculumMSESpec(LossSpec):
    """Curriculum MSE loss spec: ramps trusted rollout length over curr_start -> curr_end."""

    curr_start: int = Field(ge=0)
    curr_end: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_curr_epochs(self) -> Self:
        if self.curr_end < self.curr_start:
            msg = f"curr_end ({self.curr_end}) must be >= curr_start ({self.curr_start})."
            raise ValueError(msg)
        return self


class PSDSpec(LossSpec):
    """Welch PSD matching loss spec."""


class MetricLossSpec(LossSpec):
    """Base for a metric twin: window geometry in seconds, as metrics.py defines it.

    Both geometry fields default to ``None``, meaning "whatever grid ``metrics.py`` scores
    ``name`` on"; the resolution happens in the ``*_steps`` methods, so the defaults live in
    exactly one place -- :mod:`neuro.metrics` -- rather than being copied here.
    """

    window_s: float | None = Field(default=None, gt=0)
    hop_s: float | None = Field(default=None, gt=0)

    def window_steps(self, fs: float, name: str) -> int:
        """Trailing-window length in steps at ``fs``, defaulting to ``METRICS[name]``'s window."""
        window_s = self.window_s if self.window_s is not None else METRICS[name].window_s
        return round(window_s * fs)

    def hop_steps(self, fs: float) -> int:
        """Window spacing in steps at ``fs``, defaulting to the metrics layer's own hop."""
        hop_s = self.hop_s if self.hop_s is not None else DEFAULT_HOP_S
        return round(hop_s * fs)


class EegMsSpec(MetricLossSpec):
    """EEG mean square metric loss spec."""


class LossSpecs(StrictConfig):
    """Config-declared set of additive loss terms."""

    curriculum_mse: CurriculumMSESpec | None = None
    psd: PSDSpec | None = None
    eeg_ms: EegMsSpec | None = None

    def active(self) -> dict[str, LossSpec]:
        """Return a mapping of non-None configured loss specs."""
        return {name: spec for name in self.__class__.model_fields if (spec := getattr(self, name)) is not None}

    @model_validator(mode="after")
    def _validate_non_empty(self) -> Self:
        if not self.active():
            msg = "At least one loss must be configured in 'training.losses'."
            raise ValueError(msg)
        return self


class TrainingConfig(StrictConfig):
    """Optimisation and scaling settings for the NN predictor."""

    epochs: int = Field(default=100, ge=1)
    warmup_epochs: int = Field(default=0, ge=0)
    batch_size: int = Field(default=128, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    train_split: float = Field(default=0.8, gt=0, lt=1)
    seed: int = Field(default=69, ge=0)
    patience: int = Field(default=50, ge=1)
    scaler: Literal["standard", "robust"] = "standard"
    global_scaling: bool = False
    device: Literal["cpu", "cuda"] = "cpu"
    eval_horizon_s: float = Field(gt=0)
    losses: LossSpecs

    @model_validator(mode="after")
    def _validate_warmup(self) -> Self:
        if self.warmup_epochs >= self.epochs:
            msg = f"warmup_epochs ({self.warmup_epochs}) must be < epochs ({self.epochs})."
            raise ValueError(msg)
        return self


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
    amplitude_weight: float = Field(default=0.0, ge=0)


class NNSweepConfig(StrictConfig):
    """Optuna sweep settings for the NN predictor: trial count, output dir and per-group search spaces."""

    n_trials: int = Field(default=20, ge=1)
    artifact: str | None = None
    objective: Literal["log_energy", "val_loss", "closed_loop", "rollout_nmse"] = "log_energy"
    model: dict[str, ParamSpec] = Field(default_factory=dict)
    training: dict[str, ParamSpec] = Field(default_factory=dict)
    closed_loop: ClosedLoopEvalConfig | None = None

    @model_validator(mode="after")
    def _validate_objective(self) -> Self:
        if self.objective == "closed_loop" and self.closed_loop is None:
            msg = "objective 'closed_loop' requires a 'sweep.closed_loop' section."
            raise ValueError(msg)
        return self


class NNPredictorConfig(StrictConfig):
    """Fully-resolved, validated configuration for the NN-predictor pipeline."""

    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig
    artifact: str | None = None
    sweep: NNSweepConfig | None = None

    @property
    def fs(self) -> float:
        """Effective sampling frequency in Hz after downsampling."""
        return 1.0 / (self.simulation.dt * self.simulation.downsample)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> NNPredictorConfig:
        """Build a typed config from a raw YAML mapping, rejecting unknown keys everywhere."""
        return cls.model_validate({} if data is None else data)

    @model_validator(mode="after")
    def _validate_losses_and_horizon(self) -> Self:
        fs = self.fs
        active = self.training.losses.active()
        if all(spec.start_epoch > 0 for spec in active.values()):
            msg = "At least one loss must have start_epoch = 0; otherwise epoch 0 has no gradient."
            raise ValueError(msg)

        if self.training.losses.psd is not None:
            psd_steps = self.training.losses.psd.span_steps(fs)
            if psd_steps < _MIN_PSD_SPAN_STEPS:
                msg = f"psd.span_steps ({psd_steps}) must be >= {_MIN_PSD_SPAN_STEPS} at fs={fs} Hz."
                raise ValueError(msg)

        for name, spec in active.items():
            if isinstance(spec, MetricLossSpec):
                n_window = spec.window_steps(fs, name)
                n_hop = spec.hop_steps(fs)
                span_steps = spec.span_steps(fs)
                if not (1 <= n_window <= span_steps):
                    msg = (
                        f"Metric loss '{name}' requires 1 <= round(window_s * fs) <= span_steps, "
                        f"got window_steps={n_window}, span_steps={span_steps} at fs={fs} Hz."
                    )
                    raise ValueError(msg)
                if n_hop < 1:
                    msg = f"Metric loss '{name}' hop_s resolves to < 1 sample at fs={fs} Hz."
                    raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_sweep_exclusivity_and_keys(self) -> Self:
        if self.sweep is None:
            return self

        _validate_sweep_overlap_and_keys(self.sweep.model, self.model, "model")
        _validate_sweep_overlap_and_keys(self.sweep.training, self.training, "training")
        return self


def _resolve_field_model(cls: type[BaseModel], field_name: str) -> type[BaseModel] | None:
    if field_name not in cls.model_fields:
        return None
    annotation = cls.model_fields[field_name].annotation

    if annotation is None:
        return None
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    origin = typing.get_origin(annotation)
    if origin is typing.Union or origin is types.UnionType:
        for arg in typing.get_args(annotation):
            if isinstance(arg, type) and issubclass(arg, BaseModel):
                return arg
    return None


def _validate_sweep_overlap_and_keys(
    sweep_dict: dict[str, ParamSpec],
    target_config: StrictConfig,
    target_name: str,
) -> None:
    invalid_keys: list[str] = []
    overlap_keys: list[str] = []

    for sweep_key in sweep_dict:
        parts = sweep_key.split(".")
        curr_cls: type[BaseModel] | None = target_config.__class__
        curr_obj: Any = target_config

        for i, part in enumerate(parts):
            if curr_cls is None or part not in curr_cls.model_fields:
                invalid_keys.append(sweep_key)
                break

            if i == len(parts) - 1:
                if curr_obj is not None and part in curr_obj.model_fields_set:
                    overlap_keys.append(sweep_key)
            else:
                if curr_obj is not None:
                    curr_obj = getattr(curr_obj, part)
                    if curr_obj is None:
                        msg = (
                            f"Loss '{part}' referenced in 'sweep.{target_name}.{sweep_key}' "
                            f"is not configured in '{target_name}.{'.'.join(parts[: i + 1])}'."
                        )
                        raise ValueError(msg)
                curr_cls = _resolve_field_model(curr_cls, part)

    if invalid_keys:
        msg = f"Keys {sorted(invalid_keys)} in 'sweep.{target_name}' are not valid '{target_name}' parameters."
        raise ValueError(msg)

    if overlap_keys:
        msg = (
            f"Parameters cannot be defined in both regular '{target_name}' and 'sweep.{target_name}'."
            f" Overlap: {sorted(overlap_keys)}"
        )
        raise ValueError(msg)


def expand_dotted_dict(flat: dict[str, Any]) -> dict[str, Any]:
    """Expand flat dotted keys like ``{'losses.eeg_ms.weight': 0.3}`` into nested mappings."""
    nested: dict[str, Any] = {}
    for key, value in flat.items():
        parts = key.split(".")
        curr = nested
        for part in parts[:-1]:
            curr = curr.setdefault(part, {})
        curr[parts[-1]] = value
    return nested


class ESNModelConfig(StrictConfig):
    """ESN predictor architecture settings."""

    reservoir_size: int = Field(default=500, ge=1)
    spectral_radius: float = Field(default=0.9, gt=0)
    leak_rate: float = Field(default=0.1, gt=0, le=1)
    density: float = Field(default=0.1, gt=0, le=1)
    input_scaling: float = Field(default=0.1, gt=0)
    washout: int = Field(default=100, ge=0)
    ridge_lambda: float = Field(default=1e-3, ge=0)
    noise_sigma: float = Field(default=0.0, ge=0)
    horizon: int = Field(default=50, ge=1)


class ESNTrainingConfig(StrictConfig):
    """Training settings for ESN predictor."""

    train_split: float = Field(default=0.8, gt=0, lt=1)
    seed: int = Field(default=69, ge=0)
    scaler: Literal["standard", "robust"] = "standard"
    global_scaling: bool = False


class ESNSweepConfig(StrictConfig):
    """Optuna sweep settings for ESN predictor."""

    reservoir_sizes: list[int] = Field(default_factory=lambda: [100, 250, 500, 1000])
    lambdas: list[float] = Field(default_factory=lambda: [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0])
    n_trials: int = Field(default=50, ge=1)
    model: dict[str, ParamSpec] = Field(default_factory=dict)


class ESNPredictorConfig(StrictConfig):
    """Fully-resolved, validated configuration for the ESN-predictor pipeline."""

    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    model: ESNModelConfig = Field(default_factory=ESNModelConfig)
    training: ESNTrainingConfig = Field(default_factory=ESNTrainingConfig)
    artifact: str | None = None
    sweep: ESNSweepConfig | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ESNPredictorConfig:
        """Build a typed config from a raw YAML mapping, rejecting unknown keys everywhere."""
        return cls.model_validate({} if data is None else data)

    @model_validator(mode="after")
    def _validate_sweep_exclusivity_and_keys(self) -> Self:
        if self.sweep is None:
            return self
        _validate_sweep_overlap_and_keys(self.sweep.model, self.model, "model")
        return self


def load_config(path: Path) -> NNPredictorConfig:
    """Load and strictly validate an NN-predictor YAML config from ``path``."""
    if not path.exists():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)
    with path.open() as f:
        raw = yaml.safe_load(f)
    return NNPredictorConfig.from_dict(raw)


def load_esn_config(path: Path) -> ESNPredictorConfig:
    """Load and strictly validate an ESN-predictor YAML config from ``path``."""
    if not path.exists():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)
    with path.open() as f:
        raw = yaml.safe_load(f)
    return ESNPredictorConfig.from_dict(raw)


def resolve_data_files(
    config: NNPredictorConfig | ESNPredictorConfig, data_path_override: str | None = None
) -> list[str]:
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
    check_excitation_alignment(data_path, config.simulation.downsample)
    return data_files


def resolve_artifact_dir(artifact: str | None, default_prefix: str) -> Path:
    """Resolve (and create) the artifact output directory."""
    if artifact is None:
        local_now = datetime.datetime.now(datetime.UTC).astimezone()
        artifact = f"artifacts/{default_prefix}_{local_now.strftime('%Y-%m-%d_%H-%M-%S')}"
    artifact_dir = Path(artifact)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir
