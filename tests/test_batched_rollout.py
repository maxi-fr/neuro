"""Pin the batched ``prime_many``/``rollout_many`` against a loop of the scalar methods."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from _predictor_reference import mlp_prime, mlp_rollout

from neuro.checkpoint import MLPCheckpoint
from neuro.predictor.evaluation import (
    accumulate_rollout_errors,
    evaluate_log_energy,
    evaluate_rollouts,
    nmse,
    window_energy,
)
from neuro.predictor.module import AutoregressiveMLP
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from neuro.types import FloatArray

_SEED = 31
_N_EEG, _N_CONTROLS = 6, 2
_STEPS = 12


def _standardizers(rng: np.random.Generator) -> tuple[Standardizer, Standardizer]:
    """Non-trivial ``(y_std, u_std)``."""
    y_std = Standardizer(center=rng.standard_normal(_N_EEG), scale=rng.uniform(0.5, 2.0, _N_EEG))
    u_std = Standardizer(center=rng.standard_normal(_N_CONTROLS), scale=rng.uniform(0.5, 2.0, _N_CONTROLS))
    return y_std, u_std


def _model() -> AutoregressiveMLP:
    """A random MLP module, its weights fixed so the module is reproducible."""
    rng = np.random.default_rng(_SEED)
    y_std, u_std = _standardizers(rng)
    n_y, n_u = 4, 3
    in_size = n_y * _N_EEG + n_u * _N_CONTROLS
    layers = tuple(
        (rng.standard_normal((out, inp)) / np.sqrt(inp), rng.standard_normal(out) * 0.1)
        for inp, out in itertools.pairwise([in_size, 7, _N_EEG])
    )
    model = AutoregressiveMLP(
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
    linears = [m for m in model.layers if isinstance(m, torch.nn.Linear)]
    with torch.no_grad():
        for lin, (w, b) in zip(linears, layers, strict=True):
            lin.weight.copy_(torch.as_tensor(w, dtype=torch.float32))
            lin.bias.copy_(torch.as_tensor(b, dtype=torch.float32))
    return model


def _batch(model: AutoregressiveMLP, n_batch: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Independently drawn ``(y_hists, u_hists, u_futures)`` -- no two batch members share a history."""
    rng = np.random.default_rng(_SEED + 100)
    k = model.priming_steps
    return (
        rng.standard_normal((n_batch, k, model.n_channels)),
        rng.standard_normal((n_batch, k, model.n_controls)),
        rng.standard_normal((n_batch, _STEPS, model.n_controls)),
    )


@pytest.mark.parametrize("n_batch", [1, 5])
def test_prime_many_matches_prime_loop(n_batch: int) -> None:
    """``prime_many`` equals a loop of ``prime`` over per-member histories, to float32 tolerance."""
    model = _model()
    y_hists, u_hists, _ = _batch(model, n_batch)

    batched = model.prime_many(y_hists, u_hists)
    looped = np.stack([model.prime(y_hists[i], u_hists[i]) for i in range(n_batch)])

    assert batched.shape == looped.shape
    rtol, atol = 1e-5, 1e-6
    np.testing.assert_allclose(batched, looped, rtol=rtol, atol=atol)


@pytest.mark.parametrize("n_batch", [1, 5])
def test_rollout_many_matches_rollout_loop(n_batch: int) -> None:
    """``rollout_many`` equals a loop of ``rollout`` from per-member states, to float32 tolerance."""
    model = _model()
    y_hists, u_hists, u_futures = _batch(model, n_batch)

    states = np.stack([model.prime(y_hists[i], u_hists[i]) for i in range(n_batch)])
    batched = model.rollout_many(states, u_futures)
    looped = np.stack([model.rollout(states[i], u_futures[i]) for i in range(n_batch)])

    assert batched.shape == (n_batch, _STEPS, model.n_channels)
    rtol, atol = 1e-5, 1e-6
    np.testing.assert_allclose(batched, looped, rtol=rtol, atol=atol)


def test_batch_members_do_not_share_state() -> None:
    """Under one shared future control, distinct histories must still give distinct rollouts."""
    model = _model()
    y_hists, u_hists, u_futures = _batch(model, 4)
    u_shared = np.repeat(u_futures[:1], 4, axis=0)

    preds = model.rollout_many(model.prime_many(y_hists, u_hists), u_shared)

    for i, j in itertools.combinations(range(4), 2):
        assert float(np.abs(preds[i] - preds[j]).max()) > 1e-6


def test_accumulate_rollout_errors_matches_per_window_loop() -> None:
    """The batched accumulator reproduces the per-window ``prime``/``rollout`` loop on the module."""
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
            y_pred = model.rollout(model.prime(y[t0 - k : t0], u[t0 - k : t0]), u[t0 : t0 + _STEPS])
            y_true = y[t0 : t0 + _STEPS]
            ref[0] += ((y_pred - y_true) ** 2).sum(axis=1)
            ref[1] += (y_true**2).sum(axis=1)
            ref[2] += (y_pred**2).sum(axis=1)

    rtol, atol = 1e-5, 1e-6
    np.testing.assert_allclose(np.stack([sq_err, power, pred_power]), ref, rtol=rtol, atol=atol)


def _checkpoint() -> MLPCheckpoint:
    """The torch-free float64 twin of the module, rebuilt from its buffers."""
    model = _model()
    layers = tuple(
        (m.weight.detach().cpu().numpy().astype(np.float64), m.bias.detach().cpu().numpy().astype(np.float64))
        for m in model.layers
        if isinstance(m, torch.nn.Linear)
    )
    return MLPCheckpoint(
        layers=layers,
        activation=model.activation,
        n_y=model.n_y,
        n_u=model.n_u,
        horizon=model.horizon,
        n_channels=model.n_channels,
        n_controls=model.n_controls,
        hidden_size=model.hidden_size,
        depth=model.depth,
        dt=model.dt,
        downsample=model.downsample,
        y_std=model.y_std,
        u_std=model.u_std,
    )


def _reference_rollout(model: AutoregressiveMLP, y: FloatArray, u: FloatArray, t0: int, steps: int) -> FloatArray:
    """One float64 reference free-run window for the module's checkpoint twin."""
    k = model.priming_steps
    ckpt = _checkpoint()
    return mlp_rollout(ckpt, mlp_prime(ckpt, y[t0 - k : t0], u[t0 - k : t0]), u[t0 : t0 + steps])


def test_evaluation_scores_match_the_float64_reference() -> None:
    """``evaluate_rollouts``/``evaluate_log_energy`` score the module like its float64 checkpoint twin.

    The module carries float32 weights while the checkpoint reference is float64, so the scores
    agree to the module-vs-checkpoint tolerance rather than bit-for-bit.
    """
    model = _model()
    rng = np.random.default_rng(_SEED + 300)
    trajs = [
        (rng.standard_normal((90, model.n_controls)), rng.standard_normal((90, model.n_channels))),
        (rng.standard_normal((70, model.n_controls)), rng.standard_normal((70, model.n_channels))),
    ]

    module_rollout = evaluate_rollouts(model, trajs, _STEPS)
    module_energy = evaluate_log_energy(model, trajs, _STEPS, window_steps=4, hop_steps=2)

    # Reference scores: one float64 free-run per window, then the same pooling as the evaluator.
    k = model.priming_steps
    sq_err = np.zeros(_STEPS, dtype=np.float64)
    power = np.zeros(_STEPS, dtype=np.float64)
    energy_total: FloatArray | None = None
    n_windows = 0
    for u, y in trajs:
        for t0 in range(k, len(y) - _STEPS, 25):
            y_pred = _reference_rollout(model, y, u, t0, _STEPS)
            y_true = y[t0 : t0 + _STEPS]
            sq_err += ((y_pred - y_true) ** 2).sum(axis=1)
            power += (y_true**2).sum(axis=1)
            log_ratio = np.log(window_energy(y_pred[None], 4, 2) + 1e-12) - np.log(
                window_energy(y_true[None], 4, 2) + 1e-12
            )
            energy_total = (log_ratio[0] ** 2) if energy_total is None else energy_total + log_ratio[0] ** 2
            n_windows += 1

    twin_rollout_pooled = float(nmse(sq_err.sum(), power.sum()))
    twin_rollout_per_step = nmse(sq_err, power)
    twin_energy_pooled = float((energy_total / n_windows).mean()) if energy_total is not None else 0.0

    rtol, atol = 1e-4, 1e-5
    assert module_rollout.pooled == pytest.approx(twin_rollout_pooled, rel=rtol, abs=atol)
    np.testing.assert_allclose(module_rollout.per_step, twin_rollout_per_step, rtol=rtol, atol=atol)
    assert module_energy.pooled == pytest.approx(twin_energy_pooled, rel=rtol, abs=atol)
