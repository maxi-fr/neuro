"""Pin the stateless jax ``rollout`` batching and the free-run scoring it feeds."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from neuro.predictor.evaluation import accumulate_rollout_errors
from neuro.predictor.inference import WaveformMLPModel
from neuro.predictor.module import AutoregressiveMLP
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from neuro.types import FloatArray

_SEED = 31
_N_EEG, _N_CONTROLS = 6, 2
_STEPS = 12


def _model() -> WaveformMLPModel:
    """A random waveform jax model, its weights fixed so the rollout is reproducible."""
    rng = np.random.default_rng(_SEED)
    y_std = Standardizer(center=rng.standard_normal(_N_EEG), scale=rng.uniform(0.5, 2.0, _N_EEG))
    u_std = Standardizer(center=rng.standard_normal(_N_CONTROLS), scale=rng.uniform(0.5, 2.0, _N_CONTROLS))
    n_y, n_u = 4, 3
    in_size = n_y * _N_EEG + n_u * _N_CONTROLS
    layers = tuple(
        (rng.standard_normal((out, inp)) / np.sqrt(inp), rng.standard_normal(out) * 0.1)
        for inp, out in itertools.pairwise([in_size, 7, _N_EEG])
    )
    module = AutoregressiveMLP(
        n_y=n_y,
        n_u=n_u,
        horizon=_STEPS,
        n_channels=_N_EEG,
        n_controls=_N_CONTROLS,
        hidden_size=7,
        depth=1,
        activation="tanh",
        dt=0.02,
        y_std=y_std,
        u_std=u_std,
    )
    linears = [m for m in module.layers if isinstance(m, torch.nn.Linear)]
    with torch.no_grad():
        for lin, (w, b) in zip(linears, layers, strict=True):
            lin.weight.copy_(torch.as_tensor(w, dtype=torch.float32))
            lin.bias.copy_(torch.as_tensor(b, dtype=torch.float32))
    return WaveformMLPModel.from_checkpoint(*module.to_checkpoint())


def _batch(model: WaveformMLPModel, n_batch: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Independently drawn ``(y_hists, u_hists, u_futures)`` -- no two batch members share a history."""
    rng = np.random.default_rng(_SEED + 100)
    k = model.priming_steps
    return (
        rng.standard_normal((n_batch, k, model.n_channels)),
        rng.standard_normal((n_batch, k, model.n_controls)),
        rng.standard_normal((n_batch, _STEPS, model.n_controls)),
    )


@pytest.mark.parametrize("n_batch", [1, 5])
def test_rollout_batches_match_a_loop_of_single_rollouts(n_batch: int) -> None:
    """The stateless jax ``rollout`` batched over members equals one rollout per member."""
    model = _model()
    y_hists, u_hists, u_futures = _batch(model, n_batch)

    batched = np.asarray(model.free_run(y_hists, u_hists, u_futures))
    looped = np.stack(
        [np.asarray(model.free_run(y_hists[i][None], u_hists[i][None], u_futures[i][None]))[0] for i in range(n_batch)]
    )

    assert batched.shape == (n_batch, _STEPS, model.n_channels)
    np.testing.assert_allclose(batched, looped, rtol=1e-5, atol=1e-6)


def test_batch_members_do_not_share_state() -> None:
    """Under one shared future control, distinct histories must still give distinct rollouts."""
    model = _model()
    y_hists, u_hists, u_futures = _batch(model, 4)
    u_shared = np.repeat(u_futures[:1], 4, axis=0)

    preds = np.asarray(model.free_run(y_hists, u_hists, u_shared))

    for i, j in itertools.combinations(range(4), 2):
        assert float(np.abs(preds[i] - preds[j]).max()) > 1e-6


def test_accumulate_rollout_errors_matches_per_window_loop() -> None:
    """The batched accumulator reproduces the per-window free run on the jax model."""
    model = _model()
    rng = np.random.default_rng(_SEED + 200)
    trajs = [
        (rng.standard_normal((90, model.n_controls)), rng.standard_normal((90, model.n_channels))),
        (rng.standard_normal((70, model.n_controls)), rng.standard_normal((70, model.n_channels))),
    ]

    sq_err, power, pred_power = accumulate_rollout_errors(model, trajs, _STEPS, stride=7)

    ref = np.zeros((3, _STEPS), dtype=np.float64)
    k = model.priming_steps
    for u, y in trajs:
        for t0 in range(k, len(y) - _STEPS, 7):
            y_pred = np.asarray(model.free_run(y[t0 - k : t0][None], u[t0 - k : t0][None], u[t0 : t0 + _STEPS][None]))[
                0
            ]
            y_true = y[t0 : t0 + _STEPS]
            ref[0] += ((y_pred - y_true) ** 2).sum(axis=1)
            ref[1] += (y_true**2).sum(axis=1)
            ref[2] += (y_pred**2).sum(axis=1)

    np.testing.assert_allclose(np.stack([sq_err, power, pred_power]), ref, rtol=1e-5, atol=1e-6)
