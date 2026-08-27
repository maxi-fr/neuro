from __future__ import annotations

import datetime
import math
import types
import typing
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from neuro.metrics import DEFAULT_HOP_S, METRICS
from neuro.provenance import check_excitation_alignment

if TYPE_CHECKING:
    from typing import Self

    import optuna

_MIN_SMOOTHED_FRAMES = 2


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
    residual: bool = True


class SimulationConfig(StrictConfig):
    """Trajectory-loading settings shared by the NN-predictor pipeline."""

    dt: float = Field(default=1e-4, gt=0)
    downsample: int = Field(default=1, ge=1)
    n_steps: int | None = Field(default=None, ge=1)
    data_path: str | None = None
    cutoff_hz: float | None = Field(default=None, gt=0)


class LossSpec(StrictConfig):
    """Base for one additive loss term: its weight and its epoch gate."""

    weight: float = Field(ge=0)
    start_epoch: int = Field(default=0, ge=0)

    def span_steps(self, fs: float) -> int:
        """Rollout length in steps this term scores at ``fs``."""
        raise NotImplementedError


class SecondsSpanSpec(LossSpec):
    """A loss term whose rollout span is declared in seconds and rounded at ``fs``."""

    span_s: float = Field(gt=0)

    def span_steps(self, fs: float) -> int:
        """Rollout length in steps implied by span_s at fs, >= 1."""
        return max(1, round(self.span_s * fs))


class CurriculumMSESpec(SecondsSpanSpec):
    """Curriculum MSE loss spec: ramps trusted rollout length over curr_start -> curr_end."""

    curr_start: int = Field(ge=0)
    curr_end: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_curr_epochs(self) -> Self:
        if self.curr_end < self.curr_start:
            msg = f"curr_end ({self.curr_end}) must be >= curr_start ({self.curr_start})."
            raise ValueError(msg)
        return self


class ObservableGeometry(StrictConfig):
    """Base for the grid an Observable is reduced onto: segment support, hop and per-channel width.

    The single source of truth shared by the offline target, the training Loss and the MPC Cost.
    ``span_steps`` is passed in rather than stored, so one geometry serves any Control Horizon.
    """

    kind: ClassVar[str]

    def segment_steps(self, fs: float) -> int:
        """Length in samples of the Segment one Frame reduces."""
        raise NotImplementedError

    def hop_steps(self, fs: float) -> int:
        """Spacing in samples between consecutive Segments."""
        raise NotImplementedError

    def sample_support_steps(self, fs: float) -> int:
        """Sample support in samples of a single Frame at ``fs``."""
        raise NotImplementedError

    def n_frames(self, span_steps: int, fs: float) -> int:
        """Count the Observable Frames a length-``span_steps`` span holds at ``fs``."""
        raise NotImplementedError

    def frame_supports(self, span_steps: int, fs: float) -> tuple[tuple[int, int], ...]:
        """Half-open ``[start, end)`` sample support of every Frame, in span-relative indices."""
        raise NotImplementedError

    def n_values(self, fs: float) -> int:
        """Count the scored values a Frame carries per channel."""
        raise NotImplementedError

    def check_span(self, span_steps: int, fs: float) -> None:
        """Raise ``ValueError`` if ``span_steps`` cannot hold at least one Frame at ``fs``."""
        raise NotImplementedError


class StftGeometry(ObservableGeometry):
    """Spectrogram geometry in samples, since at fs = 50 Hz seconds do not resolve it.

    ``kernel_width = 1`` and ``n_bin_pool = 1`` mean "no pooling"; the Welch endpoint is
    ``n_segment = span_steps``, which yields a single frame.
    """

    kind: ClassVar[str] = "stft"

    n_segment: int = Field(gt=0)
    n_hop: int = Field(gt=0)
    band_hz: tuple[float, float] | None = None
    n_bin_pool: int = Field(default=1, ge=1)
    kernel: Literal["boxcar", "triangular", "hann"] = "boxcar"
    kernel_width: int = Field(default=1, ge=1)

    def bin_range(self, fs: float) -> tuple[int, int]:
        """Half-open rfft bin index range scored at ``fs``; DC is always excluded."""
        n_bins = self.n_segment // 2 + 1
        if self.band_hz is None:
            return 1, n_bins
        lo_hz, hi_hz = self.band_hz
        lo = max(1, math.ceil(lo_hz * self.n_segment / fs))
        hi = min(n_bins, math.floor(hi_hz * self.n_segment / fs) + 1)
        return lo, max(lo, hi)

    def segment_steps(self, fs: float) -> int:  # noqa: ARG002 -- geometry is already in samples
        """Length in samples of the Segment one Frame reduces."""
        return self.n_segment

    def hop_steps(self, fs: float) -> int:  # noqa: ARG002 -- geometry is already in samples
        """Spacing in samples between consecutive Segments."""
        return self.n_hop

    def sample_support_steps(self, fs: float) -> int:  # noqa: ARG002 -- geometry is already in samples
        """Sample support in samples of a single Frame."""
        return (self.kernel_width - 1) * self.n_hop + self.n_segment

    def n_segment_frames(self, span_steps: int) -> int:
        """Count the frames the segment grid extracts from the span, before the Frame Kernel."""
        if span_steps < self.n_segment:
            return 0
        return (span_steps - self.n_segment) // self.n_hop + 1

    def n_frames(self, span_steps: int, fs: float) -> int:
        """Count the frames left after the Frame Kernel has consumed its valid support."""
        if span_steps < self.sample_support_steps(fs):
            return 0
        return self.n_segment_frames(span_steps) - self.kernel_width + 1

    def frame_supports(self, span_steps: int, fs: float) -> tuple[tuple[int, int], ...]:
        """Sample support of every smoothed Frame: the ``kernel_width`` segments it pools."""
        return tuple(
            (m * self.n_hop, (m + self.kernel_width - 1) * self.n_hop + self.n_segment)
            for m in range(self.n_frames(span_steps, fs))
        )

    def n_values(self, fs: float) -> int:
        """Pooled in-band bin count per channel; ``pool_bins`` drops the trailing remainder."""
        bin_lo, bin_hi = self.bin_range(fs)
        return (bin_hi - bin_lo) // self.n_bin_pool

    def check_span(self, span_steps: int, fs: float) -> None:
        """Raise if the span is too short for the segment grid or for the Frame Kernel."""
        if self.n_segment > span_steps:
            msg = f"stft.n_segment ({self.n_segment}) must be <= the span ({span_steps} steps)."
            raise ValueError(msg)
        n_frames = self.n_segment_frames(span_steps)
        if self.kernel_width > n_frames:
            msg = f"stft.kernel_width ({self.kernel_width}) must be <= the frame count ({n_frames})."
            raise ValueError(msg)
        # A kernel that collapses the frame axis to one output is Welch with fewer effective dof.
        n_out = self.n_frames(span_steps, fs)
        if self.kernel_width > 1 and n_out < _MIN_SMOOTHED_FRAMES:
            msg = f"stft frame kernel leaves {n_out} frame(s); a kernel must leave >= {_MIN_SMOOTHED_FRAMES}."
            raise ValueError(msg)

    @model_validator(mode="after")
    def _validate_band(self) -> Self:
        if self.band_hz is not None and self.band_hz[0] >= self.band_hz[1]:
            msg = f"stft.band_hz must be increasing, got {self.band_hz}."
            raise ValueError(msg)
        return self


class EegMsGeometry(ObservableGeometry):
    """Trailing mean-square window geometry in seconds, as metrics.py defines it.

    Both fields default to ``None``, meaning "whatever grid ``metrics.py`` scores ``eeg_ms`` on";
    the resolution happens in the ``*_steps`` methods, so the defaults live in exactly one place --
    :mod:`neuro.metrics` -- rather than being copied here.
    """

    kind: ClassVar[str] = "eeg_ms"

    window_s: float | None = Field(default=None, gt=0)
    hop_s: float | None = Field(default=None, gt=0)

    def window_steps(self, fs: float) -> int:
        """Trailing-window length in steps at ``fs``, defaulting to the ``eeg_ms`` metric's window."""
        window_s = self.window_s if self.window_s is not None else METRICS["eeg_ms"].window_s
        return round(window_s * fs)

    def segment_steps(self, fs: float) -> int:
        """Length in samples of the Segment one Frame reduces."""
        return self.window_steps(fs)

    def hop_steps(self, fs: float) -> int:
        """Window spacing in steps at ``fs``, defaulting to the metrics layer's own hop."""
        hop_s = self.hop_s if self.hop_s is not None else DEFAULT_HOP_S
        return round(hop_s * fs)

    def sample_support_steps(self, fs: float) -> int:
        """Sample support in samples of a single Frame."""
        return self.window_steps(fs)

    def n_frames(self, span_steps: int, fs: float) -> int:
        """Count the trailing windows the hop grid extracts from the span."""
        if span_steps < self.window_steps(fs):
            return 0
        return (span_steps - self.window_steps(fs)) // self.hop_steps(fs) + 1

    def frame_supports(self, span_steps: int, fs: float) -> tuple[tuple[int, int], ...]:
        """Sample support of every trailing window, in span-relative indices."""
        n_window, n_hop = self.window_steps(fs), self.hop_steps(fs)
        return tuple((m * n_hop, m * n_hop + n_window) for m in range(self.n_frames(span_steps, fs)))

    def n_values(self, fs: float) -> int:  # noqa: ARG002 -- one mean square per channel, at any rate
        """One mean-square value per channel."""
        return 1

    def check_span(self, span_steps: int, fs: float) -> None:
        """Raise if the span cannot hold one trailing window, or the hop resolves below a sample."""
        n_window, n_hop = self.window_steps(fs), self.hop_steps(fs)
        if not 1 <= n_window <= span_steps:
            msg = (
                f"eeg_ms requires 1 <= round(window_s * fs) <= the span, got window_steps={n_window}, "
                f"span={span_steps} at fs={fs} Hz."
            )
            raise ValueError(msg)
        if n_hop < 1:
            msg = f"eeg_ms hop_s resolves to < 1 sample at fs={fs} Hz."
            raise ValueError(msg)


class StftSpec(StftGeometry, LossSpec):
    """Spectrogram matching loss spec: the shared :class:`StftGeometry`, scored over ``n_span`` steps."""

    n_span: int = Field(gt=0)

    def span_steps(self, fs: float) -> int:  # noqa: ARG002 -- geometry is already in samples
        """Rollout length in steps, declared directly."""
        return self.n_span

    def geometry(self) -> StftGeometry:
        """Return the Observable geometry this loss scores, without its weight and epoch gate."""
        return StftGeometry(**{name: getattr(self, name) for name in StftGeometry.model_fields})

    @model_validator(mode="after")
    def _validate_span(self) -> Self:
        # The sample-rate-dependent checks are the band and pooling ones, which
        # NNPredictorConfig runs once it knows fs; the span itself is pure sample arithmetic.
        if self.n_segment > self.n_span:
            msg = f"stft.n_segment ({self.n_segment}) must be <= n_span ({self.n_span})."
            raise ValueError(msg)
        n_frames = self.n_segment_frames(self.n_span)
        if self.kernel_width > n_frames:
            msg = f"stft.kernel_width ({self.kernel_width}) must be <= the frame count ({n_frames})."
            raise ValueError(msg)
        if self.kernel_width > 1 and n_frames - self.kernel_width + 1 < _MIN_SMOOTHED_FRAMES:
            msg = (
                f"stft frame kernel leaves {n_frames - self.kernel_width + 1} frame(s); "
                f"a kernel must leave >= {_MIN_SMOOTHED_FRAMES}."
            )
            raise ValueError(msg)
        return self


class EegMsSpec(EegMsGeometry, SecondsSpanSpec):
    """EEG mean square metric loss spec: the shared :class:`EegMsGeometry` over a span in seconds."""

    def geometry(self) -> EegMsGeometry:
        """Return the Observable geometry this loss scores, without its weight and epoch gate."""
        return EegMsGeometry(**{name: getattr(self, name) for name in EegMsGeometry.model_fields})


class LossSpecs(StrictConfig):
    """Config-declared set of additive loss terms."""

    curriculum_mse: CurriculumMSESpec | None = None
    stft: StftSpec | None = None
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
    """Optimisation and scaling settings for the NN predictor.

    ``fit`` names the algorithm the unified entry point routes to: ``gradient_descent`` serves any
    torch module over the base protocol, ``ridge`` serves only the Ridge-Fittable depth-0 waveform
    MLP and depth-0 observable MLP. A fit the configured model does not support fails at build time
    in :func:`neuro.predictor.train.train`, not mid-fit.
    """

    fit: Literal["gradient_descent", "ridge"] = "gradient_descent"
    ridge_lambda: float = Field(default=0.0, ge=0)
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
    # ``None`` only on the observable path, whose loss is a fixed MSE on the frame grid rather
    # than a configured set of terms; NNPredictorConfig requires one of the two.
    losses: LossSpecs | None = None

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

    def suggest(self, trial: optuna.trial.BaseTrial, name: str) -> Any:  # noqa: ANN401
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

    def suggest(self, trial: optuna.trial.BaseTrial, name: str) -> int:
        """Suggest an integer for ``name`` within ``[low, high]``."""
        return trial.suggest_int(name, self.low, self.high, log=self.log)


class FloatParam(_RangeParam):
    """Floating-point Optuna search dimension."""

    type: Literal["float"]
    log: bool = False

    def suggest(self, trial: optuna.trial.BaseTrial, name: str) -> float:
        """Suggest a float for ``name`` within ``[low, high]``."""
        return trial.suggest_float(name, self.low, self.high, log=self.log)


class LogUniformParam(_RangeParam):
    """Log-uniform Optuna search dimension."""

    type: Literal["loguniform"]
    low: float = Field(gt=0)
    high: float = Field(gt=0)

    def suggest(self, trial: optuna.trial.BaseTrial, name: str) -> float:
        """Suggest a float for ``name`` within ``[low, high]``, sampled log-uniformly."""
        return trial.suggest_float(name, self.low, self.high, log=True)


ParamSpec = Annotated[
    CategoricalParam | IntParam | FloatParam | LogUniformParam,
    Field(discriminator="type"),
]


# Per-model candidate sets the sweep objective is validated against; ``closed_loop`` is the
# sweep-level candidate every kind can rank on too.
WAVEFORM_TRAINER_CANDIDATES = frozenset({"log_energy", "val_loss", "rollout_nmse"})
OBSERVABLE_TRAINER_CANDIDATES = frozenset({"val_loss", "val_log_mse"})
CLOSED_LOOP_OBJECTIVE = "closed_loop"


class ClosedLoopEvalConfig(StrictConfig):
    """Settings for closed-loop evaluation in Optuna hyperparameter sweeps."""

    simulation_config: str
    seeds: list[int]
    t_end: float = Field(gt=0)
    seizure_ptp_mv: float = Field(gt=0)
    max_seizing_regions: int = Field(ge=0)
    amplitude_weight: float = Field(default=0.0, ge=0)


class NNSweepConfig(StrictConfig):
    """Optuna sweep settings for the two NN predictors: trial count, output dir and search spaces.

    ``objective`` names the Trainer candidate the study minimizes. It is a plain string here
    because the sweep alone does not know the model kind; :class:`NNPredictorConfig` validates it
    against the waveform/observable candidate sets (plus the sweep-level ``closed_loop``), so a
    mismatched name fails at build time.
    """

    n_trials: int = Field(default=20, ge=1)
    artifact: str | None = None
    objective: str = "log_energy"
    model: dict[str, ParamSpec] = Field(default_factory=dict)
    training: dict[str, ParamSpec] = Field(default_factory=dict)
    observable: dict[str, ParamSpec] = Field(default_factory=dict)
    closed_loop: ClosedLoopEvalConfig | None = None

    @model_validator(mode="after")
    def _validate_objective(self) -> Self:
        if self.objective == "closed_loop" and self.closed_loop is None:
            msg = "objective 'closed_loop' requires a 'sweep.closed_loop' section."
            raise ValueError(msg)
        return self


def _validate_observable_losses(losses: LossSpecs | None, geometry: StftGeometry, fs: float) -> None:
    if losses is None:
        return
    active = losses.active()
    reduction_losses = sorted(set(active.keys()) & {"stft", "eeg_ms"})
    if reduction_losses:
        msg = (
            f"observable predictor does not support reduction losses ({', '.join(repr(k) for k in reduction_losses)})."
        )
        raise ValueError(msg)
    if all(spec.start_epoch > 0 for spec in active.values()):
        msg = "At least one loss must have start_epoch = 0; otherwise epoch 0 has no gradient."
        raise ValueError(msg)
    fs_frame = fs / geometry.n_hop
    for name, spec in active.items():
        if isinstance(spec, SecondsSpanSpec) and round(spec.span_s * fs_frame) < 1:
            msg = (
                f"loss '{name}' span ({spec.span_s} s) resolves to {round(spec.span_s * fs_frame)} frame(s) "
                f"at frame rate {fs_frame:g} Hz (hop={geometry.n_hop}, fs={fs:g} Hz); must hold at least 1 Frame."
            )
            raise ValueError(msg)


def _validate_waveform_losses(losses: LossSpecs | None, fs: float) -> None:
    if losses is None:
        return
    active = losses.active()
    if all(spec.start_epoch > 0 for spec in active.values()):
        msg = "At least one loss must have start_epoch = 0; otherwise epoch 0 has no gradient."
        raise ValueError(msg)

    stft = losses.stft
    if stft is not None:
        bin_lo, bin_hi = stft.bin_range(fs)
        n_bins = bin_hi - bin_lo
        if n_bins < 1:
            msg = f"stft leaves no frequency bins at fs={fs} Hz for band_hz={stft.band_hz}."
            raise ValueError(msg)
        if stft.n_bin_pool > n_bins:
            msg = f"stft.n_bin_pool ({stft.n_bin_pool}) exceeds the {n_bins} in-band bin(s) at fs={fs} Hz."
            raise ValueError(msg)

    for spec in active.values():
        if isinstance(spec, ObservableGeometry):
            spec.check_span(spec.span_steps(fs), fs)


class NNPredictorConfig(StrictConfig):
    """Fully-resolved, validated configuration for the NN-predictor pipeline."""

    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig
    observable: StftGeometry | None = None
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
        if self.observable is not None:
            _validate_observable_losses(self.training.losses, self.observable, self.fs)
            min_n_u = self.observable.kernel_width - 1 + math.ceil(self.observable.n_segment / self.observable.n_hop)
            if self.model.n_u < min_n_u:
                msg = (
                    f"model.n_u ({self.model.n_u}) violates the control-support rule: must be >= {min_n_u} "
                    f"(kernel_width={self.observable.kernel_width}, segment={self.observable.n_segment}, "
                    f"hop={self.observable.n_hop}) so the past-control window covers the Frame's sample support."
                )
                raise ValueError(msg)
            fs_frame = self.fs / self.observable.n_hop
            eval_frames = round(self.training.eval_horizon_s * fs_frame)
            if eval_frames < 1:
                msg = (
                    f"training.eval_horizon_s ({self.training.eval_horizon_s} s) resolves to {eval_frames} "
                    f"frame(s) at frame rate {fs_frame:g} Hz (hop={self.observable.n_hop}, fs={self.fs:g} Hz); "
                    "must hold at least 1 Frame."
                )
                raise ValueError(msg)
        else:
            _validate_waveform_losses(self.training.losses, self.fs)
        return self

    @model_validator(mode="after")
    def _validate_sweep_exclusivity_and_keys(self) -> Self:
        if self.sweep is None:
            return self

        _validate_sweep_overlap_and_keys(self.sweep.model, self.model, "model")
        _validate_sweep_overlap_and_keys(self.sweep.training, self.training, "training")
        if self.observable is not None and self.sweep.observable:
            _validate_sweep_overlap_and_keys(self.sweep.observable, self.observable, "observable")
        candidates = OBSERVABLE_TRAINER_CANDIDATES if self.observable is not None else WAVEFORM_TRAINER_CANDIDATES
        _validate_sweep_objective(self.sweep.objective, candidates)
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


def _validate_sweep_objective(objective: str, candidates: frozenset[str]) -> None:
    """Raise unless the named sweep objective is one the configured model's Trainer can report.

    The Trainer candidates are per-model kind -- waveform ``{log_energy, val_loss, rollout_nmse}``
    or observable ``{val_loss, val_log_mse}`` -- and ``closed_loop`` is the sweep-level candidate
    every kind can rank on. Failing here catches a mismatched objective at build time instead of
    on the first trial.
    """
    if objective in candidates or objective == CLOSED_LOOP_OBJECTIVE:
        return
    msg = (
        f"sweep.objective {objective!r} is not a candidate the configured Trainer reports "
        f"({sorted(candidates)}), nor the sweep-level 'closed_loop'."
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
