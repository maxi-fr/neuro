"""Pin the torch losses: the spectrogram against SciPy, and curriculum MSE scheduling."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch
from scipy.signal import spectrogram as scipy_spectrogram
from scipy.signal.windows import hann

from neuro.config import CurriculumMSESpec, EegMsSpec, LossSpecs, SecondsSpanSpec, StftSpec
from neuro.predictor.losses import (
    CurriculumMSE,
    EegMsLoss,
    Loss,
    LossContext,
    StftLoss,
    build_losses,
    frame_kernel,
    spectrogram,
    total_loss,
)
from neuro.predictor.module import AutoregressiveMLP

_SEED = 3


def _model(
    n_y: int,
    n_u: int,
    horizon: int,
    k: int,
    n_controls: int,
) -> AutoregressiveMLP:
    torch.manual_seed(_SEED)
    return AutoregressiveMLP(
        n_y=n_y,
        n_u=n_u,
        horizon=horizon,
        n_channels=k,
        n_controls=n_controls,
        hidden_size=8,
        depth=1,
        activation="tanh",
    )


@pytest.mark.parametrize("fs", [1.0, 250.0])
@pytest.mark.parametrize("n_segment", [8, 9])
def test_spectrogram_matches_scipy(n_segment: int, fs: float) -> None:
    """The primitive reproduces ``scipy.signal.spectrogram`` under the geometry the loss uses.

    Both parities matter: the Nyquist bin is only left undoubled when ``n_segment`` is even.
    """
    rng = np.random.default_rng(_SEED)
    n_hop = 3
    x = rng.standard_normal((4, 5 * n_segment))

    _, _, want = scipy_spectrogram(
        x,
        fs=fs,
        window=hann(n_segment, sym=False),
        nperseg=n_segment,
        noverlap=n_segment - n_hop,
        detrend=False,
        scaling="density",
        mode="psd",
        axis=-1,
    )
    got = spectrogram(torch.as_tensor(x), n_segment, n_hop, fs=fs).numpy()

    assert got.shape == (4, want.shape[-1], n_segment // 2 + 1)
    np.testing.assert_allclose(got, np.moveaxis(want, -2, -1), rtol=1e-10, atol=0.0)


def _stft_ctx(c: int, fs: float = 1.0, epoch: int | None = 0) -> LossContext:
    return LossContext(
        y_center=torch.zeros(c, dtype=torch.float64),
        y_scale=torch.ones(c, dtype=torch.float64),
        fs=fs,
        epoch=epoch,
    )


def test_stft_welch_endpoint_scores_each_sample_separately() -> None:
    """With one frame per rollout the loss is Welch's geometry, but never pooled over the batch.

    Two rollouts whose spectra err in opposite directions must not cancel -- the defect the old
    batch-pooled PSD term had.
    """
    n_span, c = 32, 2
    t = torch.arange(n_span, dtype=torch.float64)
    tone = torch.sin(2 * torch.pi * 0.2 * t).reshape(1, n_span, 1).repeat(1, 1, c)
    true = torch.cat([tone, tone], dim=0)
    pred = torch.cat([tone * 2.0, tone * 0.5], dim=0)  # +6 dB and -6 dB: exact cancellation if pooled

    loss_fn = StftLoss(
        weight=1.0,
        span_steps=n_span,
        start_epoch=0,
        n_segment=n_span,
        n_hop=n_span,
        bin_lo=1,
        bin_hi=n_span // 2 + 1,
        n_bin_pool=1,
        kernel="boxcar",
        kernel_width=1,
    )
    value, diag = loss_fn(pred, true, _stft_ctx(c))

    assert diag["M_out"] == 1.0
    assert float(value) > 1.0  # a pooled loss would report ~0 here


@pytest.mark.parametrize(("kernel", "want"), [("boxcar", 1.0), ("triangular", 0.75), ("hann", 0.667)])
def test_frame_kernel_effective_dof(kernel: str, want: float) -> None:
    """K_eff / n follows the shape table: boxcar keeps every degree of freedom, Hann two thirds."""
    width = 200
    weights = frame_kernel(kernel, width, torch.zeros(1, dtype=torch.float64))
    k_eff = float(weights.sum() ** 2 / (weights**2).sum())
    assert k_eff / width == pytest.approx(want, abs=0.01)


def test_stft_frame_kernel_pools_before_the_log() -> None:
    """A frame kernel shortens the frame axis and is reported through M_out and K_eff."""
    n_span, c, n_segment, n_hop, width = 40, 2, 8, 4, 3
    rng = np.random.default_rng(_SEED + 7)
    pred = torch.as_tensor(rng.standard_normal((3, n_span, c)))
    true = torch.as_tensor(rng.standard_normal((3, n_span, c)))

    loss_fn = StftLoss(
        weight=1.0,
        span_steps=n_span,
        start_epoch=0,
        n_segment=n_segment,
        n_hop=n_hop,
        bin_lo=1,
        bin_hi=n_segment // 2 + 1,
        n_bin_pool=1,
        kernel="hann",
        kernel_width=width,
    )
    n_frames = (n_span - n_segment) // n_hop + 1
    assert loss_fn.log_spectrogram(pred, _stft_ctx(c)).shape == (3, c, n_frames - width + 1, n_segment // 2)

    _, diag = loss_fn(pred, true, _stft_ctx(c))
    assert diag["M_out"] == float(n_frames - width + 1)
    assert diag["K_eff"] == pytest.approx(8 / 3)  # [0.5, 1, 0.5] weights


def test_curriculum_ramps_the_trusted_prefix_then_holds() -> None:
    """L holds at 1 until curr_start, ramps 1 -> span_steps by curr_end, then holds there."""
    span_steps, curr_start, curr_end = 20, 10, 90
    loss_fn = CurriculumMSE(weight=1.0, span_steps=span_steps, start_epoch=0, curr_start=curr_start, curr_end=curr_end)

    lengths = [loss_fn.trusted_length(e) for e in range(curr_end + 1)]
    assert lengths[: curr_start + 1] == [1] * (curr_start + 1)
    assert lengths[-1] == span_steps
    assert all(b >= a for a, b in itertools.pairwise(lengths))
    assert loss_fn.trusted_length(curr_end + 500) == span_steps
    assert loss_fn.trusted_length(None) == span_steps  # terminal schedule


def test_curriculum_zero_width_window_jumps_to_full_span() -> None:
    """With curr_start == curr_end the trusted prefix jumps 1 -> span_steps (no divide-by-zero)."""
    pivot, span_steps = 30, 20
    loss_fn = CurriculumMSE(weight=1.0, span_steps=span_steps, start_epoch=0, curr_start=pivot, curr_end=pivot)
    assert loss_fn.trusted_length(pivot) == 1
    assert loss_fn.trusted_length(pivot + 1) == span_steps


def test_curriculum_scores_only_the_trusted_prefix() -> None:
    """Only the trusted prefix reaches the MSE, and its width is reported as the 'L' diagnostic."""
    rng = np.random.default_rng(_SEED + 3)
    batch, span_steps, c = 4, 10, 3
    pred = torch.as_tensor(rng.standard_normal((batch, span_steps, c)))
    true = torch.as_tensor(rng.standard_normal((batch, span_steps, c)))
    ctx = LossContext(
        y_center=torch.zeros(c, dtype=torch.float64),
        y_scale=torch.ones(c, dtype=torch.float64),
        fs=1.0,
        epoch=0,
    )

    loss_fn = CurriculumMSE(weight=1.0, span_steps=span_steps, start_epoch=0, curr_start=0, curr_end=100)
    value, diag = loss_fn(pred, true, ctx)

    assert diag["L"] == 1.0
    # At L = 1 the loss must equal the plain MSE of the first step alone.
    expected = torch.mean((pred[:, 0] - true[:, 0]) ** 2)
    torch.testing.assert_close(value, expected)


def test_stft_is_gated_off_until_start_epoch() -> None:
    """total_loss gates off the spectral term when ctx.epoch < stft.start_epoch."""
    rng = np.random.default_rng(_SEED + 1)
    n_y, n_u, horizon, c, n_controls, batch, w_stft = 2, 2, 6, 5, 2, 32, 0.1
    model = _model(n_y, n_u, horizon, c, n_controls)

    x = torch.as_tensor(rng.standard_normal((batch, n_y * c + n_u * n_controls + horizon * n_controls)))
    true_traj = torch.as_tensor(rng.standard_normal((batch, horizon, c)))
    pred_traj = model(x).reshape(batch, horizon, c)

    losses: list[Loss] = [
        CurriculumMSE(weight=1.0, span_steps=horizon, start_epoch=0, curr_start=0, curr_end=0),
        StftLoss(
            weight=w_stft,
            span_steps=horizon,
            start_epoch=10,
            n_segment=horizon,
            n_hop=horizon,
            bin_lo=1,
            bin_hi=horizon // 2 + 1,
            n_bin_pool=1,
            kernel="boxcar",
            kernel_width=1,
        ),
    ]

    ctx_gated = LossContext(
        y_center=torch.zeros(c, dtype=torch.float64),
        y_scale=torch.ones(c, dtype=torch.float64),
        fs=1.0,
        epoch=5,
    )
    ctx_active = LossContext(
        y_center=torch.zeros(c, dtype=torch.float64),
        y_scale=torch.ones(c, dtype=torch.float64),
        fs=1.0,
        epoch=10,
    )

    total_gated, comps_gated = total_loss(losses, pred_traj, true_traj, ctx_gated)
    total_active, comps_active = total_loss(losses, pred_traj, true_traj, ctx_active)

    assert comps_gated["stft"] == 0.0
    assert float(total_gated.detach()) == pytest.approx(comps_gated["curriculum_mse"])
    assert comps_active["stft"] > 0.0
    assert float(total_active.detach()) == pytest.approx(comps_active["curriculum_mse"] + w_stft * comps_active["stft"])


def test_build_losses_instantiates_from_specs() -> None:
    """build_losses correctly instantiates loss terms from LossSpecs and dict mappings."""
    fs = 100.0
    specs = LossSpecs(
        curriculum_mse=CurriculumMSESpec(weight=1.0, span_s=0.2, curr_start=0, curr_end=10, start_epoch=0),
        stft=StftSpec(weight=0.1, n_span=20, n_segment=20, n_hop=20, start_epoch=5),
        eeg_ms=EegMsSpec(weight=0.5, span_s=0.2, window_s=0.1, hop_s=0.01, start_epoch=2),
    )

    losses = build_losses(specs, fs)
    assert len(losses) == 3
    assert isinstance(losses[0], CurriculumMSE)
    assert losses[0].span_steps == 20
    assert losses[0].curr_end == 10

    assert isinstance(losses[1], StftLoss)
    assert losses[1].span_steps == 20
    assert losses[1].start_epoch == 5

    assert isinstance(losses[2], EegMsLoss)
    assert losses[2].span_steps == 20
    assert losses[2].n_window == 10
    assert losses[2].n_hop == 1

    # Also test passing a plain dict
    losses_dict = build_losses(specs.active(), fs)
    assert len(losses_dict) == 3


def test_build_losses_unknown_spec_raises() -> None:
    """build_losses raises TypeError when encountering an unrecognized LossSpec subclass."""

    class DummySpec(SecondsSpanSpec):
        pass

    with pytest.raises(TypeError, match="Unknown loss spec type"):
        build_losses({"dummy": DummySpec(weight=1.0, span_s=0.1)}, fs=100.0)
