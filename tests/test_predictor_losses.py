"""Pin the torch losses: the Welch replica against SciPy and curriculum MSE scheduling."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch
from scipy.signal import welch

from neuro.config import CurriculumMSESpec, EegMsSpec, LossSpec, LossSpecs, PSDSpec
from neuro.predictor.losses import (
    CurriculumMSE,
    EegMsLoss,
    Loss,
    LossContext,
    PSDLoss,
    build_losses,
    total_loss,
    welch_psd,
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
@pytest.mark.parametrize("nperseg", [8, 9])
def test_welch_psd_matches_scipy(nperseg: int, fs: float) -> None:
    """The replica reproduces ``scipy.signal.welch`` under the call the trainer makes.

    Both parities matter: the Nyquist bin is only left undoubled when ``nperseg`` is even.
    """
    rng = np.random.default_rng(_SEED)
    x = rng.standard_normal((4, 5 * nperseg))

    _, want = welch(x, fs=fs, nperseg=nperseg, noverlap=0, axis=-1)
    got = welch_psd(torch.as_tensor(x), nperseg, fs=fs).numpy()

    assert got.shape == (4, nperseg // 2 + 1)
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=0.0)


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


def test_psd_is_gated_off_until_start_epoch() -> None:
    """total_loss gates off the PSD term when ctx.epoch < psd.start_epoch."""
    rng = np.random.default_rng(_SEED + 1)
    n_y, n_u, horizon, c, n_controls, batch, w_psd = 2, 2, 6, 5, 2, 32, 0.1
    model = _model(n_y, n_u, horizon, c, n_controls)

    x = torch.as_tensor(rng.standard_normal((batch, n_y * c + n_u * n_controls + horizon * n_controls)))
    true_traj = torch.as_tensor(rng.standard_normal((batch, horizon, c)))
    pred_traj = model(x).reshape(batch, horizon, c)

    losses: list[Loss] = [
        CurriculumMSE(weight=1.0, span_steps=horizon, start_epoch=0, curr_start=0, curr_end=0),
        PSDLoss(weight=w_psd, span_steps=horizon, start_epoch=10),
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

    assert comps_gated["psd"] == 0.0
    assert float(total_gated.detach()) == pytest.approx(comps_gated["curriculum_mse"])
    assert comps_active["psd"] > 0.0
    assert float(total_active.detach()) == pytest.approx(comps_active["curriculum_mse"] + w_psd * comps_active["psd"])


def test_build_losses_instantiates_from_specs() -> None:
    """build_losses correctly instantiates loss terms from LossSpecs and dict mappings."""
    fs = 100.0
    specs = LossSpecs(
        curriculum_mse=CurriculumMSESpec(weight=1.0, span_s=0.2, curr_start=0, curr_end=10, start_epoch=0),
        psd=PSDSpec(weight=0.1, span_s=0.2, start_epoch=5),
        eeg_ms=EegMsSpec(weight=0.5, span_s=0.2, window_s=0.1, hop_s=0.01, start_epoch=2),
    )

    losses = build_losses(specs, fs)
    assert len(losses) == 3
    assert isinstance(losses[0], CurriculumMSE)
    assert losses[0].span_steps == 20
    assert losses[0].curr_end == 10

    assert isinstance(losses[1], PSDLoss)
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

    class DummySpec(LossSpec):
        pass

    with pytest.raises(TypeError, match="Unknown loss spec type"):
        build_losses({"dummy": DummySpec(weight=1.0, span_s=0.1)}, fs=100.0)
