"""Pin the batched ``prime_many``/``rollout_many`` against a loop of the scalar methods."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
import pytest

from neuro.artifacts import accumulate_rollout_errors
from neuro.esn import ESNArtifact, generate_reservoir
from neuro.predictor.artifact import MLPArtifact
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from neuro.artifacts import RolloutArtifact
    from neuro.types import FloatArray

_SEED = 31
_N_EEG, _N_CONTROLS = 6, 2
_STEPS = 12


def _standardizers(rng: np.random.Generator) -> tuple[Standardizer, Standardizer]:
    """Non-trivial ``(y_std, u_std)``."""
    y_std = Standardizer(center=rng.standard_normal(_N_EEG), scale=rng.uniform(0.5, 2.0, _N_EEG))
    u_std = Standardizer(center=rng.standard_normal(_N_CONTROLS), scale=rng.uniform(0.5, 2.0, _N_CONTROLS))
    return y_std, u_std


def _mlp_artifact() -> MLPArtifact:
    """A random MLP artifact."""
    rng = np.random.default_rng(_SEED)
    y_std, u_std = _standardizers(rng)
    n_y, n_u = 4, 3
    sizes = [n_y * _N_EEG + n_u * _N_CONTROLS, 7, _N_EEG]
    layers = tuple(
        (rng.standard_normal((out, inp)) / np.sqrt(inp), rng.standard_normal(out) * 0.1)
        for inp, out in itertools.pairwise(sizes)
    )
    return MLPArtifact(
        layers=layers,
        activation="tanh",
        n_y=n_y,
        n_u=n_u,
        horizon=_STEPS,
        n_channels=_N_EEG,
        n_controls=_N_CONTROLS,
        dt=0.02,
        downsample=200,
        y_std=y_std,
        u_std=u_std,
    )


def _esn_artifact() -> ESNArtifact:
    """A random ESN artifact."""
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
    return ESNArtifact(
        w_in=w_in,
        w_out=rng.uniform(-0.1, 0.1, size=(_N_EEG, reservoir_size + 1)),
        w_res=w_res,
        dt=0.02,
        downsample=200,
        horizon=_STEPS,
        reservoir_size=reservoir_size,
        leak_rate=0.3,
        spectral_radius=0.9,
        priming_steps=8,
        input_scaling=0.5,
        density=0.1,
        noise_sigma=0.0,
        ridge_lambda=1e-3,
        seed=_SEED,
        y_std=y_std,
        u_std=u_std,
    )


def _artifact(family: str) -> RolloutArtifact:
    """Build the requested artifact family."""
    return _mlp_artifact() if family == "mlp" else _esn_artifact()


def _batch(art: RolloutArtifact, n_batch: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Independently drawn ``(y_hists, u_hists, u_futures)`` -- no two batch members share a history."""
    rng = np.random.default_rng(_SEED + 100)
    k = art.priming_steps
    return (
        rng.standard_normal((n_batch, k, art.n_channels)),
        rng.standard_normal((n_batch, k, art.n_controls)),
        rng.standard_normal((n_batch, _STEPS, art.n_controls)),
    )


_CASES = ["mlp", "esn"]


@pytest.mark.parametrize("family", _CASES)
@pytest.mark.parametrize("n_batch", [1, 5])
def test_prime_many_matches_prime_loop(family: str, n_batch: int) -> None:
    """``prime_many`` equals a loop of ``prime`` over per-member histories to 1e-12."""
    art = _artifact(family)
    y_hists, u_hists, _ = _batch(art, n_batch)

    batched = art.prime_many(y_hists, u_hists)
    looped = np.stack([art.prime(y_hists[i], u_hists[i]) for i in range(n_batch)])

    assert batched.shape == looped.shape
    np.testing.assert_allclose(batched, looped, atol=1e-12)


@pytest.mark.parametrize("family", _CASES)
@pytest.mark.parametrize("n_batch", [1, 5])
def test_rollout_many_matches_rollout_loop(family: str, n_batch: int) -> None:
    """``rollout_many`` equals a loop of ``rollout`` from per-member states to 1e-12."""
    art = _artifact(family)
    y_hists, u_hists, u_futures = _batch(art, n_batch)

    states = np.stack([art.prime(y_hists[i], u_hists[i]) for i in range(n_batch)])
    batched = art.rollout_many(states, u_futures)
    looped = np.stack([art.rollout(states[i], u_futures[i]) for i in range(n_batch)])

    assert batched.shape == (n_batch, _STEPS, art.n_channels)
    np.testing.assert_allclose(batched, looped, atol=1e-12)


@pytest.mark.parametrize("family", _CASES)
def test_batch_members_do_not_share_state(family: str) -> None:
    """Under one shared future control, distinct histories must still give distinct rollouts."""
    art = _artifact(family)
    y_hists, u_hists, u_futures = _batch(art, 4)
    u_shared = np.repeat(u_futures[:1], 4, axis=0)

    preds = art.rollout_many(art.prime_many(y_hists, u_hists), u_shared)

    for i, j in itertools.combinations(range(4), 2):
        assert float(np.abs(preds[i] - preds[j]).max()) > 1e-6


@pytest.mark.parametrize("family", _CASES)
def test_accumulate_rollout_errors_matches_per_window_loop(family: str) -> None:
    """The batched accumulator reproduces the per-window ``prime``/``rollout`` loop."""
    art = _artifact(family)
    rng = np.random.default_rng(_SEED + 200)
    trajs = [
        (rng.standard_normal((90, art.n_controls)), rng.standard_normal((90, art.n_channels))),
        (rng.standard_normal((70, art.n_controls)), rng.standard_normal((70, art.n_channels))),
    ]

    sq_err, power, pred_power = accumulate_rollout_errors(art, trajs, _STEPS, stride=7)

    ref = np.zeros((3, _STEPS), dtype=np.float64)
    k = art.priming_steps
    for u, y in trajs:
        for t0 in range(k, len(y) - _STEPS, 7):
            y_pred = art.rollout(art.prime(y[t0 - k : t0], u[t0 - k : t0]), u[t0 : t0 + _STEPS])
            y_true = y[t0 : t0 + _STEPS]
            ref[0] += ((y_pred - y_true) ** 2).sum(axis=1)
            ref[1] += (y_true**2).sum(axis=1)
            ref[2] += (y_pred**2).sum(axis=1)

    np.testing.assert_allclose(np.stack([sq_err, power, pred_power]), ref, rtol=1e-12, atol=0.0)
