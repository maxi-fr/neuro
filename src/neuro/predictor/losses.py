from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import torch

from neuro.config import CurriculumMSESpec, EegMsSpec, LossSpec, LossSpecs, StftSpec
from neuro.spectral import LOG_FLOOR

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor


@dataclass(frozen=True)
class LossContext:
    """Unit recovery parameters, sample rate and schedule clock for loss computation."""

    y_center: Tensor
    y_scale: Tensor
    fs: float
    epoch: int | None

    def to_raw(self, x: Tensor) -> Tensor:
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

    def __call__(self, pred: Tensor, true: Tensor, ctx: LossContext) -> tuple[Tensor, dict[str, float]]:
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
        """Rollout prefix scored at ``epoch``: 1 -> ``span_steps``, linear over the curriculum window.

        ``None`` is the terminal schedule (validation), which trusts the whole span.
        """
        if epoch is None:
            return self.span_steps
        span = max(self.curr_end - self.curr_start, 1)
        frac = min(max((epoch - self.curr_start) / span, 0.0), 1.0)
        return round(1 + (self.span_steps - 1) * frac)

    def __call__(self, pred: Tensor, true: Tensor, ctx: LossContext) -> tuple[Tensor, dict[str, float]]:
        """Compute MSE across channels over the trusted rollout prefix."""
        length = self.trusted_length(ctx.epoch)
        mse = torch.mean((pred[:, :length] - true[:, :length]) ** 2)
        return mse, {"L": float(length)}


@dataclass(frozen=True)
class StftLoss:
    """Log-spectrogram matching loss on hopped segments in raw units.

    Pooling happens only before the log (frequency bins, then a frame kernel) and only over the
    batch after the square, so a signed log residual is never averaged over anything.
    """

    weight: float
    span_steps: int
    start_epoch: int
    n_segment: int
    n_hop: int
    bin_lo: int
    bin_hi: int
    n_bin_pool: int
    kernel: str
    kernel_width: int
    name: str = "stft"

    @classmethod
    def from_spec(cls, spec: StftSpec, fs: float, name: str = "stft") -> StftLoss:
        """Instantiate from a StftSpec at sampling rate ``fs``."""
        bin_lo, bin_hi = spec.bin_range(fs)
        return cls(
            weight=spec.weight,
            span_steps=spec.span_steps(fs),
            start_epoch=spec.start_epoch,
            n_segment=spec.n_segment,
            n_hop=spec.n_hop,
            bin_lo=bin_lo,
            bin_hi=bin_hi,
            n_bin_pool=spec.n_bin_pool,
            kernel=spec.kernel,
            kernel_width=spec.kernel_width,
            name=name,
        )

    def log_spectrogram(self, x: Tensor, ctx: LossContext) -> Tensor:
        """Pooled log power per ``(batch, channel, frame, bin)`` in raw units."""
        raw = ctx.to_raw(x[:, : self.span_steps]).transpose(1, 2)
        power = spectrogram(raw, self.n_segment, self.n_hop, fs=ctx.fs)[..., self.bin_lo : self.bin_hi]
        power = pool_bins(power, self.n_bin_pool)
        power = smooth_frames(power, frame_kernel(self.kernel, self.kernel_width, power))
        return torch.log(power + LOG_FLOOR)

    def __call__(self, pred: Tensor, true: Tensor, ctx: LossContext) -> tuple[Tensor, dict[str, float]]:
        """Compute mean squared log-power difference over frames, channels and bins."""
        residual = self.log_spectrogram(pred, ctx) - self.log_spectrogram(true, ctx)
        weights = frame_kernel(self.kernel, self.kernel_width, residual)
        n_frames = (self.span_steps - self.n_segment) // self.n_hop + 1
        diag = {
            "M_out": float(n_frames - self.kernel_width + 1),
            "K_eff": float(weights.sum() ** 2 / (weights**2).sum()),
        }
        return torch.mean(residual**2), diag


@dataclass(frozen=True)
class EegMsLoss:
    """Log-space mean square power matching on hopped segments in raw units.

    The geometry mirrors the trailing power window of the ``eeg_ms`` observable, so that what the
    loss scores is what the controller later reads; the loss itself sees the whole rollout at once.
    """

    weight: float
    span_steps: int
    start_epoch: int
    n_window: int
    n_hop: int
    name: str = "eeg_ms"

    @classmethod
    def from_spec(cls, spec: EegMsSpec, fs: float, name: str = "eeg_ms") -> EegMsLoss:
        """Instantiate from an EegMsSpec at sampling rate ``fs``."""
        return cls(
            weight=spec.weight,
            span_steps=spec.span_steps(fs),
            start_epoch=spec.start_epoch,
            n_window=spec.window_steps(fs, name),
            n_hop=spec.hop_steps(fs),
            name=name,
        )

    def windowed_power(self, x: Tensor, ctx: LossContext) -> Tensor:
        """Mean-square power per trailing window in raw units: ``(B, span, C) -> (B, C, n_windows)``."""
        raw = ctx.to_raw(x[:, : self.span_steps]).transpose(1, 2)
        return raw.unfold(dimension=-1, size=self.n_window, step=self.n_hop).pow(2).mean(dim=-1)

    def __call__(self, pred: Tensor, true: Tensor, ctx: LossContext) -> tuple[Tensor, dict[str, float]]:
        """Compute log-space MSE between windowed mean-square power courses."""
        m_pred = self.windowed_power(pred, ctx)
        m_true = self.windowed_power(true, ctx)
        log_ratio = torch.log(m_pred + LOG_FLOOR) - torch.log(m_true + LOG_FLOOR)
        loss = torch.mean(log_ratio**2)
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


def total_loss(losses: Sequence[Loss], pred: Tensor, true: Tensor, ctx: LossContext) -> tuple[Tensor, dict[str, float]]:
    """Compute weighted sum of active loss terms and unweighted per-term diagnostics."""
    total = torch.zeros((), dtype=pred.dtype, device=pred.device)
    comps: dict[str, float] = {}

    for loss in losses:
        comps[loss.name] = 0.0
        if ctx.epoch is not None and ctx.epoch < loss.start_epoch:
            continue
        val, diag = loss(pred, true, ctx)
        total = total + loss.weight * val
        comps[loss.name] = float(val.detach())
        comps.update(diag)

    return total, comps


def spectrogram(x: Tensor, n_segment: int, n_hop: int, fs: float) -> Tensor:
    """Hopped periodograms of ``x`` ``(..., n)`` -> ``(..., n_frames, n_segment // 2 + 1)``.

    Matches ``scipy.signal.spectrogram(x, fs, window=hann(n_segment, sym=False),
    noverlap=n_segment - n_hop, detrend=False)``: one-sided, density-scaled, no per-segment
    detrend -- the DC bin is dropped by the caller instead, so that a segment-length sweep is not
    also a sweep over an implicit high-pass.
    """
    segments = x.unfold(dimension=-1, size=n_segment, step=n_hop)
    window = torch.hann_window(n_segment, periodic=True, dtype=x.dtype, device=x.device)
    spectrum = torch.fft.rfft(segments * window, n=n_segment, dim=-1)
    psd = (spectrum.real**2 + spectrum.imag**2) / (fs * (window**2).sum())

    # One-sided: every bin carries its negative-frequency twin, except DC and (even n) Nyquist.
    fold = torch.full((psd.shape[-1],), 2.0, dtype=psd.dtype, device=psd.device)
    fold[0] = 1.0
    if n_segment % 2 == 0:
        fold[-1] = 1.0
    return psd * fold


def pool_bins(power: Tensor, n_bin_pool: int) -> Tensor:
    """Mean power over consecutive groups of ``n_bin_pool`` bins, dropping the trailing remainder."""
    if n_bin_pool == 1:
        return power
    n_groups = power.shape[-1] // n_bin_pool
    grouped = power[..., : n_groups * n_bin_pool].reshape(*power.shape[:-1], n_groups, n_bin_pool)
    return grouped.mean(dim=-1)


def frame_kernel(kernel: str, width: int, like: Tensor) -> Tensor:
    """Normalised non-negative smoothing weights along the frame axis.

    Endpoints are kept strictly positive, so a width-``n`` taper really pools ``n`` frames.
    """
    if width == 1:
        return torch.ones(1, dtype=like.dtype, device=like.device)
    if kernel == "boxcar":
        weights = torch.ones(width, dtype=like.dtype, device=like.device)
    elif kernel == "triangular":
        weights = torch.bartlett_window(width + 2, periodic=False, dtype=like.dtype, device=like.device)[1:-1]
    else:
        weights = torch.hann_window(width + 2, periodic=False, dtype=like.dtype, device=like.device)[1:-1]
    return weights / weights.sum()


def smooth_frames(power: Tensor, weights: Tensor) -> Tensor:
    """Convolve power ``(..., n_frames, n_bins)`` along the frame axis, valid support only."""
    if weights.numel() == 1:
        return power
    moved = power.movedim(-2, -1)
    flat = moved.reshape(-1, 1, moved.shape[-1])
    smoothed = torch.nn.functional.conv1d(flat, weights.reshape(1, 1, -1))
    return smoothed.reshape(*moved.shape[:-1], smoothed.shape[-1]).movedim(-1, -2)
