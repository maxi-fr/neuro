from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

if TYPE_CHECKING:
    from torch import Tensor

    from neuro.predictor.module import AutoregressiveMLP

_EPS = 1e-8


def curriculum_mask(
    epoch: int, horizon: int, max_steps: int | None = None, start_epoch: int = 0, end_epoch: int = 80
) -> Tensor:
    """Per-step loss mask for one epoch: a prefix of ones ramping ``1 -> min(max_steps, horizon)``.

    The mask keeps its full ``horizon`` width at every epoch; steps beyond the current trusted
    rollout length (and beyond ``max_steps``, which defaults to the whole horizon) stay zero.
    The ramp is linear in epoch over ``[start_epoch, end_epoch]``.
    """
    target = min(max_steps or horizon, horizon)
    span = max(end_epoch - start_epoch, 1)
    frac = min(max((epoch - start_epoch) / span, 0.0), 1.0)
    length = max(1, min(round(1 + (target - 1) * frac), target))

    mask = torch.zeros(horizon, dtype=torch.float64)
    mask[:length] = 1.0
    return mask


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


def predictor_loss(
    model: AutoregressiveMLP, pred_traj: Tensor, true_traj: Tensor, step_mask: Tensor, w_psd: float
) -> tuple[Tensor, dict[str, float]]:
    """Masked channel-space MSE plus a log-PSD term, over a batch of rollouts.

    ``pred_traj`` is the model's own output ``(batch, horizon, n_channels)`` (latent under PCA);
    it is decoded to channel space with the module's decode buffers before scoring against
    ``true_traj`` ``(batch, horizon, n_eeg_channels)``. ``step_mask`` is the curriculum mask; its
    last entry gates the PSD term, so the spectrum only contributes at full rollout length.
    Returns the weighted total and the unweighted ``mse``/``psd`` components for logging.
    """
    if model.decode_basis is None:
        pred_channels = pred_traj
    else:
        # Buffer access goes through nn.Module.__getattr__, which is typed `Tensor | Module`.
        basis = cast("Tensor", model.decode_basis)
        mean = cast("Tensor", model.decode_mean)
        pred_channels = pred_traj @ basis + mean

    per_step_se = torch.mean((pred_channels - true_traj) ** 2, dim=(0, 2))
    mse = torch.sum(step_mask * per_step_se) / torch.sum(step_mask)

    _, horizon, n_channels = true_traj.shape
    if horizon > 1:
        pred_pool = pred_channels.permute(2, 0, 1).reshape(n_channels, -1)
        true_pool = true_traj.permute(2, 0, 1).reshape(n_channels, -1)
        log_ratio = torch.log(welch_psd(pred_pool, horizon) + _EPS) - torch.log(welch_psd(true_pool, horizon) + _EPS)
        psd = step_mask[-1] * torch.mean(log_ratio**2)
    else:
        psd = torch.zeros((), dtype=mse.dtype, device=mse.device)

    return mse + w_psd * psd, {"mse": float(mse.detach()), "psd": float(psd.detach())}
