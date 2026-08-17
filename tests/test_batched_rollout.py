"""Pin the batched ``prime_many``/``rollout_many`` against a loop of the scalar methods."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
import pytest

from neuro.artifacts import accumulate_rollout_errors
from neuro.esn import ESNArtifact, generate_reservoir
from neuro.predictor.artifact import MLPArtifact
from neuro.transforms import PCAProjection, Pipeline, Standardizer

if TYPE_CHECKING:
    from neuro.artifacts import PredictorArtifact
    from neuro.types import FloatArray

_SEED = 31
_N_EEG, _N_CONTROLS, _LATENT = 6, 2, 3
_STEPS = 12


def _pipelines(rng: np.random.Generator, latent_dim: int | None) -> tuple[Pipeline, Pipeline]:
    """Non-trivial ``(y_pipeline, u_pipeline)``, the former with an optional PCA step."""
    y_steps: list[Standardizer | PCAProjection] = [
        Standardizer(center=rng.standard_normal(_N_EEG), scale=rng.uniform(0.5, 2.0, _N_EEG))
    ]
    if latent_dim is not None:
        basis, _ = np.linalg.qr(rng.standard_normal((_N_EEG, _N_EEG)))
        y_steps.append(
            PCAProjection(basis=np.ascontiguousarray(basis[:, :latent_dim].T), mean=rng.standard_normal(_N_EEG))
        )
    u_pipeline = Pipeline((Standardizer(center=rng.standard_normal(_N_CONTROLS), scale=rng.uniform(0.5, 2.0, 2)),))
    return Pipeline(tuple(y_steps)), u_pipeline


def _mlp_artifact(latent_dim: int | None) -> MLPArtifact:
    """A random MLP artifact running in ``latent_dim`` (or raw EEG) model space."""
    rng = np.random.default_rng(_SEED)
    y_pipeline, u_pipeline = _pipelines(rng, latent_dim)
    n_y, n_u = 4, 3
    n_channels = latent_dim if latent_dim is not None else _N_EEG
    sizes = [n_y * n_channels + n_u * _N_CONTROLS, 7, n_channels]
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
        n_channels=n_channels,
        n_controls=_N_CONTROLS,
        dt=0.02,
        downsample=200,
        y_pipeline=y_pipeline,
        u_pipeline=u_pipeline,
    )


def _esn_artifact(latent_dim: int | None) -> ESNArtifact:
    """A random ESN artifact running in ``latent_dim`` (or raw EEG) model space."""
    rng = np.random.default_rng(_SEED)
    y_pipeline, u_pipeline = _pipelines(rng, latent_dim)
    n_channels = latent_dim if latent_dim is not None else _N_EEG
    reservoir_size = 40
    w_res, w_in = generate_reservoir(
        reservoir_size=reservoir_size,
        spectral_radius=0.9,
        density=0.1,
        input_scaling=0.5,
        in_dim=n_channels + _N_CONTROLS + 1,
        seed=_SEED,
    )
    return ESNArtifact(
        w_in=w_in,
        w_out=rng.uniform(-0.1, 0.1, size=(n_channels, reservoir_size + 1)),
        w_res=w_res,
        dt=0.02,
        downsample=200,
        horizon=_STEPS,
        reservoir_size=reservoir_size,
        leak_rate=0.3,
        spectral_radius=0.9,
        washout=8,
        input_scaling=0.5,
        density=0.1,
        noise_sigma=0.0,
        ridge_lambda=1e-3,
        seed=_SEED,
        y_pipeline=y_pipeline,
        u_pipeline=u_pipeline,
    )


def _artifact(family: str, latent_dim: int | None) -> PredictorArtifact:
    """Build the requested artifact family in the requested model space."""
    return _mlp_artifact(latent_dim) if family == "mlp" else _esn_artifact(latent_dim)


def _batch(art: PredictorArtifact, n_batch: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Independently drawn ``(y_hists, u_hists, u_futures)`` -- no two batch members share a history."""
    rng = np.random.default_rng(_SEED + 100)
    k = art.priming_steps
    return (
        rng.standard_normal((n_batch, k, art.n_eeg_channels)),
        rng.standard_normal((n_batch, k, art.n_controls)),
        rng.standard_normal((n_batch, _STEPS, art.n_controls)),
    )


_CASES = [(family, latent) for family in ("mlp", "esn") for latent in (None, _LATENT)]


@pytest.mark.parametrize(("family", "latent_dim"), _CASES)
@pytest.mark.parametrize("n_batch", [1, 5])
def test_prime_many_matches_prime_loop(family: str, latent_dim: int | None, n_batch: int) -> None:
    """``prime_many`` equals a loop of ``prime`` over per-member histories to 1e-12."""
    art = _artifact(family, latent_dim)
    y_hists, u_hists, _ = _batch(art, n_batch)

    batched = art.prime_many(y_hists, u_hists)
    looped = np.stack([art.prime(y_hists[i], u_hists[i]) for i in range(n_batch)])

    assert batched.shape == looped.shape
    np.testing.assert_allclose(batched, looped, atol=1e-12)


@pytest.mark.parametrize(("family", "latent_dim"), _CASES)
@pytest.mark.parametrize("n_batch", [1, 5])
def test_rollout_many_matches_rollout_loop(family: str, latent_dim: int | None, n_batch: int) -> None:
    """``rollout_many`` equals a loop of ``rollout`` from per-member states to 1e-12."""
    art = _artifact(family, latent_dim)
    y_hists, u_hists, u_futures = _batch(art, n_batch)

    states = np.stack([art.prime(y_hists[i], u_hists[i]) for i in range(n_batch)])
    batched = art.rollout_many(states, u_futures)
    looped = np.stack([art.rollout(states[i], u_futures[i]) for i in range(n_batch)])

    assert batched.shape == (n_batch, _STEPS, art.n_eeg_channels)
    np.testing.assert_allclose(batched, looped, atol=1e-12)


@pytest.mark.parametrize(("family", "latent_dim"), _CASES)
def test_batch_members_do_not_share_state(family: str, latent_dim: int | None) -> None:
    """Under one shared future control, distinct histories must still give distinct rollouts.

    Sharing ``u_future`` leaves the primed state as the only source of variation, so a batched
    path that broadcasts one member's state across the batch collapses every row here.
    """
    art = _artifact(family, latent_dim)
    y_hists, u_hists, u_futures = _batch(art, 4)
    u_shared = np.repeat(u_futures[:1], 4, axis=0)

    preds = art.rollout_many(art.prime_many(y_hists, u_hists), u_shared)

    for i, j in itertools.combinations(range(4), 2):
        assert float(np.abs(preds[i] - preds[j]).max()) > 1e-6


@pytest.mark.parametrize(("family", "latent_dim"), _CASES)
def test_accumulate_rollout_errors_matches_per_window_loop(family: str, latent_dim: int | None) -> None:
    """The batched accumulator reproduces the per-window ``prime``/``rollout`` loop."""
    art = _artifact(family, latent_dim)
    rng = np.random.default_rng(_SEED + 200)
    trajs = [
        (rng.standard_normal((90, art.n_controls)), rng.standard_normal((90, art.n_eeg_channels))),
        (rng.standard_normal((70, art.n_controls)), rng.standard_normal((70, art.n_eeg_channels))),
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
