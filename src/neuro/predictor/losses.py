from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import torch

from neuro.config import CurriculumMSESpec, EegMsSpec, LossSpec, LossSpecs, PSDSpec

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor


_EPS = 1e-8


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
class PSDLoss:
    """Welch power spectral density log-ratio matching loss."""

    weight: float
    span_steps: int
    start_epoch: int
    name: str = "psd"

    @classmethod
    def from_spec(cls, spec: PSDSpec, fs: float, name: str = "psd") -> PSDLoss:
        """Instantiate from a PSDSpec at sampling rate ``fs``."""
        return cls(
            weight=spec.weight,
            span_steps=spec.span_steps(fs),
            start_epoch=spec.start_epoch,
            name=name,
        )

    def __call__(self, pred: Tensor, true: Tensor, ctx: LossContext) -> tuple[Tensor, dict[str, float]]:
        """Compute mean squared log-PSD difference across channels."""
        pred_slice = pred[:, : self.span_steps]
        true_slice = true[:, : self.span_steps]
        n_channels = pred_slice.shape[-1]
        pred_pool = pred_slice.permute(2, 0, 1).reshape(n_channels, -1)
        true_pool = true_slice.permute(2, 0, 1).reshape(n_channels, -1)
        log_ratio = torch.log(welch_psd(pred_pool, self.span_steps, fs=ctx.fs) + _EPS) - torch.log(
            welch_psd(true_pool, self.span_steps, fs=ctx.fs) + _EPS
        )
        psd = torch.mean(log_ratio**2)
        return psd, {}


@dataclass(frozen=True)
class EegMsLoss:
    """Log-space mean square power matching on causal trailing windows in raw units."""

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
        log_ratio = torch.log(m_pred + _EPS) - torch.log(m_true + _EPS)
        loss = torch.mean(log_ratio**2)
        return loss, {}


_LOSS_FACTORIES: dict[type[LossSpec], Any] = {
    CurriculumMSESpec: CurriculumMSE.from_spec,
    PSDSpec: PSDLoss.from_spec,
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


def welch_psd(x: Tensor, nperseg: int, fs: float = 1.0) -> Tensor:
    """Welch power spectral density of ``x`` ``(..., n)`` -> ``(..., nperseg // 2 + 1)``.

    A differentiable replica of ``scipy.signal.welch(x, fs, nperseg=nperseg, noverlap=0, axis=-1)``
    under that call's defaults: ``detrend="constant"``, a periodic Hann window, density scaling,
    one-sided spectrum and mean averaging over segments.
    """
    n_segments = x.shape[-1] // nperseg
    segments = x[..., : n_segments * nperseg].reshape(*x.shape[:-1], n_segments, nperseg)
    segments = segments - segments.mean(dim=-1, keepdim=True)  # detrend="constant", before windowing

    window = torch.hann_window(nperseg, periodic=True, dtype=x.dtype, device=x.device)
    spectrum = torch.fft.rfft(segments * window, n=nperseg, dim=-1)
    psd = (spectrum.real**2 + spectrum.imag**2) / (fs * (window**2).sum())

    # One-sided: every bin carries its negative-frequency twin, except DC and (even nperseg) Nyquist.
    fold = torch.full((psd.shape[-1],), 2.0, dtype=psd.dtype, device=psd.device)
    fold[0] = 1.0
    if nperseg % 2 == 0:
        fold[-1] = 1.0
    return (psd * fold).mean(dim=-2)
