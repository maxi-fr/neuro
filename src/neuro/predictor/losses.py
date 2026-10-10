from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp

from neuro.config import (
    CurriculumMSESpec,
    EegMsGeometry,
    EegMsSpec,
    LossSpec,
    LossSpecs,
    StftGeometry,
    StftSpec,
)
from neuro.schedules import curriculum_completion, curriculum_span
from neuro.spectral import LOG_FLOOR

if TYPE_CHECKING:
    from collections.abc import Sequence


class LossContext(eqx.Module):
    """Unit recovery parameters, sample rate, and schedule clock for loss computation."""

    y_center: jax.Array
    y_scale: jax.Array
    fs: float = eqx.field(static=True)
    epoch: int | None = eqx.field(static=True, default=None)

    def to_raw(self, x: jax.Array) -> jax.Array:
        """Map standardized channel tensor back to raw units."""
        return x * self.y_scale + self.y_center


class Loss(Protocol):
    """Protocol for an additive predictor loss term."""

    @property
    def name(self) -> str:
        """Unique identifier for the loss term."""
        ...

    @property
    def weight(self) -> float:
        """Scalar multiplier in total loss."""
        ...

    @property
    def span_steps(self) -> int:
        """Number of rollout steps required by this loss."""
        ...

    @property
    def start_epoch(self) -> int:
        """First training epoch where this loss contributes to the gradient."""
        ...

    def __call__(
        self,
        pred: jax.Array,
        true: jax.Array,
        ctx: LossContext,
        step_mask: jax.Array | None = None,
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Compute unweighted loss tensor and diagnostic metrics."""
        ...


@dataclass(frozen=True)
class CurriculumMSE:
    """Curriculum mean squared error loss on standardized channels."""

    weight: float
    span_steps: int
    start_epoch: int
    curr_start: int
    curr_end: int
    name: str = "curriculum_mse"

    @classmethod
    def from_spec(cls, spec: CurriculumMSESpec, fs: float, name: str = "curriculum_mse") -> CurriculumMSE:
        """Instantiate from a CurriculumMSESpec at sampling rate ``fs``."""
        return cls(
            weight=spec.weight,
            span_steps=spec.span_steps(fs),
            start_epoch=spec.start_epoch,
            curr_start=spec.curr_start,
            curr_end=spec.curr_end,
            name=name,
        )

    def trusted_length(self, epoch: int | None) -> int:
        """Score the shared rounded curriculum Span, or the full Span for validation."""
        return curriculum_span(self.span_steps, self.curr_start, self.curr_end, epoch)

    def __call__(
        self,
        pred: jax.Array,
        true: jax.Array,
        ctx: LossContext,
        step_mask: jax.Array | None = None,
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Compute MSE across channels over the trusted rollout prefix."""
        if step_mask is not None:
            mask = step_mask[: self.span_steps]
            # Average SE over batch and channel/frequency dimensions, leaving step dimension
            diff_sq = (pred[:, : self.span_steps] - true[:, : self.span_steps]) ** 2
            reduce_axes = (0, *range(2, diff_sq.ndim))
            step_se = jnp.mean(diff_sq, axis=reduce_axes)
            mse = jnp.sum(mask * step_se) / jnp.maximum(jnp.sum(mask), 1.0)
            return mse, {"L": jnp.sum(mask)}

        length = self.trusted_length(ctx.epoch)
        mse = jnp.mean((pred[:, :length] - true[:, :length]) ** 2)
        return mse, {"L": float(length)}


@dataclass(frozen=True)
class StftLoss:
    """Log-spectrogram matching loss on hopped segments in raw units."""

    weight: float
    span_steps: int
    start_epoch: int
    geometry: StftGeometry
    name: str = "stft"

    @classmethod
    def from_spec(cls, spec: StftSpec, fs: float, name: str = "stft") -> StftLoss:
        """Instantiate from a StftSpec at sampling rate ``fs``."""
        return cls(
            weight=spec.weight,
            span_steps=spec.span_steps(fs),
            start_epoch=spec.start_epoch,
            geometry=spec.geometry(),
            name=name,
        )

    def log_spectrogram(self, x: jax.Array, ctx: LossContext) -> jax.Array:
        """Pooled log power per batch, output axis, Frame, and frequency bin in raw units."""
        geom = self.geometry
        bin_lo, bin_hi = geom.bin_range(ctx.fs)
        raw = jnp.moveaxis(ctx.to_raw(x[:, : self.span_steps]), 1, -1)
        power = spectrogram(raw, geom.n_segment, geom.n_hop, fs=ctx.fs)[..., bin_lo:bin_hi]
        power = pool_bins(power, geom.n_bin_pool)
        weights = frame_kernel(geom.kernel, geom.kernel_width, power)
        power = smooth_frames(power, weights)
        return jnp.log(power + LOG_FLOOR)

    def __call__(
        self,
        pred: jax.Array,
        true: jax.Array,
        ctx: LossContext,
        step_mask: jax.Array | None = None,
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Compute mean squared log-power difference over frames, channels, and bins."""
        del step_mask
        residual = self.log_spectrogram(pred, ctx) - self.log_spectrogram(true, ctx)
        weights = frame_kernel(self.geometry.kernel, self.geometry.kernel_width, residual)
        diag = {
            "M_out": float(self.geometry.n_frames(self.span_steps, ctx.fs)),
            "K_eff": jnp.sum(weights) ** 2 / jnp.sum(weights**2),
        }
        return jnp.mean(residual**2), diag


@dataclass(frozen=True)
class EegMsLoss:
    """Log-space mean square power matching on hopped segments in raw units."""

    weight: float
    span_steps: int
    start_epoch: int
    geometry: EegMsGeometry
    name: str = "eeg_ms"

    @classmethod
    def from_spec(cls, spec: EegMsSpec, fs: float, name: str = "eeg_ms") -> EegMsLoss:
        """Instantiate from an EegMsSpec at sampling rate ``fs``."""
        return cls(
            weight=spec.weight,
            span_steps=spec.span_steps(fs),
            start_epoch=spec.start_epoch,
            geometry=spec.geometry(),
            name=name,
        )

    def windowed_power(self, x: jax.Array, ctx: LossContext) -> jax.Array:
        """Mean-square power per trailing window, preserving every output axis in raw units."""
        raw = jnp.moveaxis(ctx.to_raw(x[:, : self.span_steps]), 1, -1)
        w_size = self.geometry.window_steps(ctx.fs)
        w_step = self.geometry.hop_steps(ctx.fs)
        t_len = raw.shape[-1]
        n_win = (t_len - w_size) // w_step + 1
        idx = jnp.arange(w_size)[None, :] + jnp.arange(n_win)[:, None] * w_step
        windows = raw[..., idx]
        return jnp.mean(windows**2, axis=-1)

    def __call__(
        self,
        pred: jax.Array,
        true: jax.Array,
        ctx: LossContext,
        step_mask: jax.Array | None = None,
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Compute log-space MSE between windowed mean-square power courses."""
        del step_mask
        m_pred = self.windowed_power(pred, ctx)
        m_true = self.windowed_power(true, ctx)
        log_ratio = jnp.log(m_pred + LOG_FLOOR) - jnp.log(m_true + LOG_FLOOR)
        loss = jnp.mean(log_ratio**2)
        return loss, {}


_LOSS_FACTORIES: dict[type[LossSpec], Any] = {
    CurriculumMSESpec: CurriculumMSE.from_spec,
    StftSpec: StftLoss.from_spec,
    EegMsSpec: EegMsLoss.from_spec,
}


def build_losses(specs: LossSpecs | dict[str, LossSpec], fs: float) -> list[Loss]:
    """Instantiate Loss terms from configuration specs at sampling rate ``fs``."""
    spec_dict = specs.active() if isinstance(specs, LossSpecs) else specs
    losses: list[Loss] = []
    for name, spec in spec_dict.items():
        factory = _LOSS_FACTORIES.get(type(spec))
        if factory is None:
            msg = f"Unknown loss spec type: {type(spec)}"
            raise TypeError(msg)
        losses.append(factory(spec, fs, name))
    return losses


def loss_eligibility_start(loss: Loss) -> int:
    """Combine Loss activation with shared curriculum completion epoch."""
    if loss.weight <= 0.0:
        return 0
    start = loss.start_epoch
    if isinstance(loss, CurriculumMSE):
        return max(start, curriculum_completion(loss.span_steps, loss.curr_start, loss.curr_end))
    return start


def eligibility_start_epoch(losses: Sequence[Loss]) -> int:
    """First training epoch where every enabled Loss is active and every curriculum at full Span."""
    return max((loss_eligibility_start(loss) for loss in losses), default=0)


def specs_eligibility_start(specs: LossSpecs | dict[str, LossSpec] | None, fs: float) -> int:
    """First training epoch where every enabled Loss is active and every curriculum at full Span from specs."""
    if specs is None:
        return 0
    return eligibility_start_epoch(build_losses(specs, fs))


def spectrogram(x: jax.Array, n_segment: int, n_hop: int, fs: float) -> jax.Array:
    """Hopped periodograms of ``x`` ``(..., n)`` -> ``(..., n_frames, n_segment // 2 + 1)``."""
    t_len = x.shape[-1]
    n_frames = (t_len - n_segment) // n_hop + 1
    idx = jnp.arange(n_segment)[None, :] + jnp.arange(n_frames)[:, None] * n_hop
    segments = x[..., idx]

    window = 0.5 - 0.5 * jnp.cos(2.0 * jnp.pi * jnp.arange(n_segment, dtype=x.dtype) / n_segment)
    spectrum = jnp.fft.rfft(segments * window, n=n_segment, axis=-1)
    psd = (spectrum.real**2 + spectrum.imag**2) / (fs * jnp.sum(window**2))

    fold = jnp.full(psd.shape[-1], 2.0, dtype=psd.dtype)
    fold = fold.at[0].set(1.0)
    if n_segment % 2 == 0:
        fold = fold.at[-1].set(1.0)
    return psd * fold


def pool_bins(power: jax.Array, n_bin_pool: int) -> jax.Array:
    """Mean power over consecutive groups of ``n_bin_pool`` bins, dropping the trailing remainder."""
    if n_bin_pool == 1:
        return power
    n_groups = power.shape[-1] // n_bin_pool
    grouped = power[..., : n_groups * n_bin_pool].reshape(*power.shape[:-1], n_groups, n_bin_pool)
    return jnp.mean(grouped, axis=-1)


def frame_kernel(kernel: str, width: int, like: jax.Array) -> jax.Array:
    """Return normalized non-negative smoothing weights along the frame axis."""
    dtype = like.dtype
    if width == 1:
        return jnp.ones(1, dtype=dtype)
    if kernel == "boxcar":
        weights = jnp.ones(width, dtype=dtype)
    elif kernel == "triangular":
        N = width + 2
        n = jnp.arange(1, width + 1, dtype=dtype)
        weights = 1.0 - jnp.abs(2.0 * n - (N - 1.0)) / (N - 1.0)
    elif kernel == "hann":
        N = width + 2
        n = jnp.arange(1, width + 1, dtype=dtype)
        weights = 0.5 - 0.5 * jnp.cos(2.0 * jnp.pi * n / (N - 1.0))
    elif kernel == "exponential":
        weights = jnp.exp(jnp.linspace(-1.0, 0.0, width, dtype=dtype))
    elif kernel == "linear":
        weights = jnp.arange(1, width + 1, dtype=dtype)
    else:
        msg = f"Unknown frame kernel: {kernel!r}"
        raise ValueError(msg)
    return weights / jnp.sum(weights)


def smooth_frames(power: jax.Array, weights: jax.Array) -> jax.Array:
    """Convolve power ``(..., n_frames, n_bins)`` along the frame axis, valid support only."""
    if weights.size == 1:
        return power
    moved = jnp.moveaxis(power, -2, -1)
    flat = moved.reshape(-1, 1, moved.shape[-1])
    w = weights.reshape(1, 1, -1)
    res = jax.lax.conv_general_dilated(flat, w, (1,), "VALID", dimension_numbers=("NCH", "OIH", "NCH"))
    reshaped = res.reshape(*moved.shape[:-1], res.shape[-1])
    return jnp.moveaxis(reshaped, -1, -2)


def total_loss(
    losses: Sequence[Loss],
    pred: jax.Array,
    true: jax.Array,
    ctx: LossContext,
    step_mask: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, Any]]:
    """Compute weighted sum of active loss terms and unweighted diagnostics."""
    total = jnp.zeros((), dtype=pred.dtype)
    comps: dict[str, Any] = {}

    for loss in losses:
        if loss.weight <= 0.0 or (ctx.epoch is not None and ctx.epoch < loss.start_epoch):
            comps[loss.name] = None
            continue
        val, diag = loss(pred, true, ctx, step_mask=step_mask)
        total = total + loss.weight * val
        comps[loss.name] = val
        comps.update(diag)

    return total, comps
