"""Pin the batched ``prime_many``/``rollout_many`` against a loop of the scalar methods."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import scipy.sparse
import torch
from _predictor_reference import esn_prime, esn_rollout, mlp_prime, mlp_rollout

from neuro.checkpoint import ESNCheckpoint, MLPCheckpoint
from neuro.esn import generate_reservoir
from neuro.predictor.esn_module import ESNModule
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


def _mlp_model() -> AutoregressiveMLP:
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


def _esn_model() -> ESNModule:
    """A random ESN module."""
    rng = np.random.default_rng(_SEED)
    y_std, u_std = _standardizers(rng)
    reservoir_size = 40
    w_res, w_in = generate_reservoir(
        reservoir_size=reservoir_size,
        spectral_radius=0.9,
        density=0.1,
        input_scaling=0.5,
        in_dim=_N_EEG + _N_CONTROLS + 1,
        seed=_SEED,
    )
    return ESNModule(
        w_res=w_res,
        w_in=w_in,
        w_out=rng.uniform(-0.1, 0.1, size=(_N_EEG, reservoir_size + 1)),
        leak_rate=0.3,
        priming_steps=8,
        horizon=_STEPS,
        dt=0.02,
        y_std=y_std,
        u_std=u_std,
    )


def _model(family: str) -> AutoregressiveMLP | ESNModule:
    """Build the requested module family."""
    return _mlp_model() if family == "mlp" else _esn_model()


def _batch(model: AutoregressiveMLP | ESNModule, n_batch: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Independently drawn ``(y_hists, u_hists, u_futures)`` -- no two batch members share a history."""
    rng = np.random.default_rng(_SEED + 100)
    k = model.priming_steps
    return (
        rng.standard_normal((n_batch, k, model.n_channels)),
        rng.standard_normal((n_batch, k, model.n_controls)),
        rng.standard_normal((n_batch, _STEPS, model.n_controls)),
    )


_CASES = ["mlp", "esn"]


@pytest.mark.parametrize("family", _CASES)
@pytest.mark.parametrize("n_batch", [1, 5])
def test_prime_many_matches_prime_loop(family: str, n_batch: int) -> None:
    """``prime_many`` equals a loop of ``prime`` over per-member histories, to float32 tolerance."""
    model = _model(family)
    y_hists, u_hists, _ = _batch(model, n_batch)

    batched = model.prime_many(y_hists, u_hists)
    looped = np.stack([model.prime(y_hists[i], u_hists[i]) for i in range(n_batch)])

    assert batched.shape == looped.shape
    # Batched vs scalar reservoir matmuls differ in the last ulps of float32, so the tolerance
    # follows the module family exactly as it does for the accumulator and the rollout.
    rtol, atol = (1e-4, 1e-5) if family == "esn" else (1e-5, 1e-6)
    np.testing.assert_allclose(batched, looped, rtol=rtol, atol=atol)


@pytest.mark.parametrize("family", _CASES)
@pytest.mark.parametrize("n_batch", [1, 5])
def test_rollout_many_matches_rollout_loop(family: str, n_batch: int) -> None:
    """``rollout_many`` equals a loop of ``rollout`` from per-member states, to float32 tolerance."""
    model = _model(family)
    y_hists, u_hists, u_futures = _batch(model, n_batch)

    states = np.stack([model.prime(y_hists[i], u_hists[i]) for i in range(n_batch)])
    batched = model.rollout_many(states, u_futures)
    looped = np.stack([model.rollout(states[i], u_futures[i]) for i in range(n_batch)])

    assert batched.shape == (n_batch, _STEPS, model.n_channels)
    rtol, atol = (1e-4, 1e-5) if family == "esn" else (1e-5, 1e-6)
    np.testing.assert_allclose(batched, looped, rtol=rtol, atol=atol)


@pytest.mark.parametrize("family", _CASES)
def test_batch_members_do_not_share_state(family: str) -> None:
    """Under one shared future control, distinct histories must still give distinct rollouts."""
    model = _model(family)
    y_hists, u_hists, u_futures = _batch(model, 4)
    u_shared = np.repeat(u_futures[:1], 4, axis=0)

    preds = model.rollout_many(model.prime_many(y_hists, u_hists), u_shared)

    for i, j in itertools.combinations(range(4), 2):
        assert float(np.abs(preds[i] - preds[j]).max()) > 1e-6


@pytest.mark.parametrize("family", _CASES)
def test_accumulate_rollout_errors_matches_per_window_loop(family: str) -> None:
    """The batched accumulator reproduces the per-window ``prime``/``rollout`` loop on the module."""
    model = _model(family)
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

    # Batched vs looped linear algebra differs in the last ulps of float32, and the ESN's
    # reservoir recurrence accumulates them, so the tolerance follows the module family.
    rtol, atol = (1e-4, 1e-5) if family == "esn" else (1e-5, 1e-6)
    np.testing.assert_allclose(np.stack([sq_err, power, pred_power]), ref, rtol=rtol, atol=atol)


def _checkpoint(family: str) -> MLPCheckpoint | ESNCheckpoint:
    """The torch-free float64 twin of the module, rebuilt from its buffers."""
    if family == "mlp":
        model = _mlp_model()
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
    model = _esn_model()
    w_res = scipy.sparse.csr_matrix(model.w_res.to_dense().numpy())
    return ESNCheckpoint(
        w_in=model.w_in.detach().cpu().numpy().astype(np.float64),
        w_out=model.w_out.detach().cpu().numpy().astype(np.float64),
        w_res=w_res,
        dt=model.dt,
        downsample=model.downsample,
        horizon=model.horizon,
        reservoir_size=model.reservoir_size,
        leak_rate=model.leak_rate,
        spectral_radius=model.spectral_radius,
        priming_steps=model.priming_steps,
        input_scaling=model.input_scaling,
        density=model.density,
        noise_sigma=model.noise_sigma,
        ridge_lambda=model.ridge_lambda,
        seed=model.seed,
        y_std=model.y_std,
        u_std=model.u_std,
    )


def _reference_rollout(
    family: str, model: AutoregressiveMLP | ESNModule, y: FloatArray, u: FloatArray, t0: int, steps: int
) -> FloatArray:
    """One float64 reference free-run window for the module's checkpoint twin."""
    k = model.priming_steps
    if family == "mlp":
        ckpt = cast("MLPCheckpoint", _checkpoint(family))
        return mlp_rollout(ckpt, mlp_prime(ckpt, y[t0 - k : t0], u[t0 - k : t0]), u[t0 : t0 + steps])
    ckpt = cast("ESNCheckpoint", _checkpoint(family))
    return esn_rollout(ckpt, esn_prime(ckpt, y[t0 - k : t0], u[t0 - k : t0]), u[t0 : t0 + steps])


@pytest.mark.parametrize("family", _CASES)
def test_evaluation_scores_match_the_float64_reference(family: str) -> None:
    """``evaluate_rollouts``/``evaluate_log_energy`` score the module like its float64 checkpoint twin.

    The module carries float32 weights while the checkpoint reference is float64, so the scores
    agree to the module-vs-checkpoint tolerance rather than bit-for-bit.
    """
    model = _model(family)
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
            y_pred = _reference_rollout(family, model, y, u, t0, _STEPS)
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

    rtol, atol = (1e-3, 1e-4) if family == "esn" else (1e-4, 1e-5)
    assert module_rollout.pooled == pytest.approx(twin_rollout_pooled, rel=rtol, abs=atol)
    np.testing.assert_allclose(module_rollout.per_step, twin_rollout_per_step, rtol=rtol, atol=atol)
    assert module_energy.pooled == pytest.approx(twin_energy_pooled, rel=rtol, abs=atol)
