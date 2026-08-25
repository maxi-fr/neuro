"""Seam 1 -- the Predictor protocol and the waveform MLP's raw-units runtime surface."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from _predictor_reference import mlp_prime, mlp_rollout

from neuro.checkpoint import MLPCheckpoint
from neuro.predictor.module import AutoregressiveMLP
from neuro.transforms import Standardizer
from neuro.types import Predictor

if TYPE_CHECKING:
    from neuro.types import Activation, FloatArray

_SEED = 17
_N_Y, _N_U, _HORIZON = 3, 2, 5
_N_EEG, _N_CONTROLS, _HIDDEN = 5, 2, 6
_STEPS = 8  # rollout length, deliberately different from the trained horizon
_RTOL, _ATOL = 1e-5, 1e-6  # float32 tolerance


def _random_layers(
    rng: np.random.Generator, in_size: int, out_size: int, hidden_size: int, depth: int
) -> tuple[tuple[FloatArray, FloatArray], ...]:
    """Random ``(W, b)`` pairs for a ``depth``-hidden-layer MLP, scaled so activations stay O(1)."""
    sizes = [in_size, *[hidden_size] * depth, out_size]
    return tuple(
        (
            (rng.standard_normal((n_out, n_in), dtype=np.float32) / np.float32(np.sqrt(n_in))).astype(np.float64),
            (rng.standard_normal(n_out, dtype=np.float32) * np.float32(0.1)).astype(np.float64),
        )
        for n_in, n_out in itertools.pairwise(sizes)
    )


def _params(
    depth: int = 2, activation: Activation = "softplus"
) -> tuple[tuple[tuple[FloatArray, FloatArray], ...], Standardizer, Standardizer]:
    """Random layers and nontrivial standardizers, shared by the module and its checkpoint twin."""
    rng = np.random.default_rng(_SEED)
    y_std = Standardizer(center=rng.uniform(-1.0, 1.0, _N_EEG), scale=rng.uniform(0.5, 2.0, _N_EEG))
    u_std = Standardizer(center=rng.uniform(-1.0, 1.0, _N_CONTROLS), scale=rng.uniform(0.5, 2.0, _N_CONTROLS))
    layers = _random_layers(rng, _N_Y * _N_EEG + _N_U * _N_CONTROLS, _N_EEG, _HIDDEN, depth)
    return layers, y_std, u_std


def _model(depth: int = 2, activation: Activation = "softplus") -> AutoregressiveMLP:
    """The waveform MLP carrying random weights and standardizer buffers."""
    layers, y_std, u_std = _params(depth, activation)
    model = AutoregressiveMLP(
        n_y=_N_Y,
        n_u=_N_U,
        horizon=_HORIZON,
        n_channels=_N_EEG,
        n_controls=_N_CONTROLS,
        hidden_size=_HIDDEN,
        depth=depth,
        activation=activation,
        dt=0.01,
        y_std=y_std,
        u_std=u_std,
    )
    linears = [m for m in model.layers if isinstance(m, torch.nn.Linear)]
    with torch.no_grad():
        for lin, (w, b) in zip(linears, layers, strict=True):
            lin.weight.copy_(torch.as_tensor(w, dtype=torch.float32))
            lin.bias.copy_(torch.as_tensor(b, dtype=torch.float32))
    return model


def _checkpoint(depth: int = 2, activation: Activation = "softplus") -> MLPCheckpoint:
    """The float64 checkpoint twin of the module, for the reference side of a parity test."""
    layers, y_std, u_std = _params(depth, activation)
    return MLPCheckpoint(
        layers=layers,
        activation=activation,
        n_y=_N_Y,
        n_u=_N_U,
        horizon=_HORIZON,
        n_channels=_N_EEG,
        n_controls=_N_CONTROLS,
        hidden_size=_HIDDEN,
        depth=depth,
        dt=0.01,
        downsample=2,
        y_std=y_std,
        u_std=u_std,
    )


def _context(seed: int = _SEED + 1) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Raw EEG history, raw control history and raw future controls, all ending/starting at the seam."""
    rng = np.random.default_rng(seed)
    k = max(_N_Y, _N_U)
    return (
        rng.standard_normal((k, _N_EEG)),
        rng.standard_normal((k, _N_CONTROLS)),
        rng.standard_normal((_STEPS, _N_CONTROLS)),
    )


def _batch(n_batch: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Independently drawn ``(y_hists, u_hists, u_futures)`` -- no two members share a history."""
    rng = np.random.default_rng(_SEED + 100)
    k = max(_N_Y, _N_U)
    return (
        rng.standard_normal((n_batch, k, _N_EEG)),
        rng.standard_normal((n_batch, k, _N_CONTROLS)),
        rng.standard_normal((n_batch, _STEPS, _N_CONTROLS)),
    )


def test_mlp_satisfies_predictor_protocol() -> None:
    """The waveform MLP is a Predictor, and ``rollout_many`` returns ``(B, positions, outputs)``."""
    model = _model()
    assert isinstance(model, Predictor)
    assert model.n_outputs == _N_EEG
    assert model.n_channels == _N_EEG
    assert model.n_controls == _N_CONTROLS
    assert model.dt == 0.01
    assert model.priming_steps == max(_N_Y, _N_U)
    assert model.horizon == _HORIZON

    y_hists, u_hists, u_futures = _batch(5)
    preds = model.rollout_many(model.prime_many(y_hists, u_hists), u_futures)
    assert preds.shape == (5, _STEPS, model.n_outputs)


def test_prime_encodes_and_rollout_decodes_against_the_float64_reference() -> None:
    """``prime``/``rollout`` reproduce the checkpoint's raw-in/raw-out path on raw history."""

    ckpt = _checkpoint()
    model = _model()
    y_hist, u_hist, u_future = _context()

    state = model.prime(y_hist, u_hist)
    want_state = mlp_prime(ckpt, y_hist, u_hist)
    assert state.shape == want_state.shape
    np.testing.assert_allclose(state, want_state, rtol=_RTOL, atol=_ATOL)

    got = model.rollout(state, u_future)
    want = mlp_rollout(ckpt, want_state, u_future)
    assert got.shape == (_STEPS, _N_EEG)
    np.testing.assert_allclose(got, want, rtol=_RTOL, atol=_ATOL)


def test_standardizers_are_buffers_and_round_trip_raw() -> None:
    """The module holds the standardizers as float32 buffers; ``encode``/``decode`` invert."""
    _, y_std, u_std = _params()
    model = _model()
    np.testing.assert_allclose(model.y_std.center, y_std.center, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(model.y_std.scale, y_std.scale, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(model.u_std.center, u_std.center, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(model.u_std.scale, u_std.scale, rtol=_RTOL, atol=_ATOL)

    rng = np.random.default_rng(_SEED + 3)
    y = rng.standard_normal((4, _N_EEG))
    np.testing.assert_allclose(model.decode(model.encode(y)), y, rtol=_RTOL, atol=_ATOL)


@pytest.mark.parametrize("n_batch", [1, 5])
def test_prime_many_matches_a_loop_of_prime(n_batch: int) -> None:
    """``prime_many`` equals a loop of ``prime`` over per-member histories."""
    model = _model()
    y_hists, u_hists, _ = _batch(n_batch)

    batched = model.prime_many(y_hists, u_hists)
    looped = np.stack([model.prime(y_hists[i], u_hists[i]) for i in range(n_batch)])

    assert batched.shape == looped.shape
    np.testing.assert_allclose(batched, looped, rtol=_RTOL, atol=_ATOL)


@pytest.mark.parametrize("n_batch", [1, 5])
def test_rollout_many_matches_a_loop_of_rollout(n_batch: int) -> None:
    """``rollout_many`` equals a loop of ``rollout`` from per-member states."""
    model = _model()
    y_hists, u_hists, u_futures = _batch(n_batch)

    states = np.stack([model.prime(y_hists[i], u_hists[i]) for i in range(n_batch)])
    batched = model.rollout_many(states, u_futures)
    looped = np.stack([model.rollout(states[i], u_futures[i]) for i in range(n_batch)])

    assert batched.shape == (n_batch, _STEPS, model.n_outputs)
    np.testing.assert_allclose(batched, looped, rtol=_RTOL, atol=_ATOL)


def test_rollout_equals_a_loop_of_step() -> None:
    """``rollout`` is exactly one ``step`` per position, emitting raw output at each."""
    model = _model()
    y_hist, u_hist, u_future = _context()

    state = model.prime(y_hist, u_hist)
    got = model.rollout(state, u_future)

    want = np.empty((_STEPS, _N_EEG), dtype=np.float64)
    for t in range(_STEPS):
        state, y = model.step(state, u_future[t])
        want[t] = y

    np.testing.assert_allclose(got, want, rtol=_RTOL, atol=_ATOL)


def test_rollout_accepts_any_length_not_just_the_native_horizon() -> None:
    """``rollout`` is not bounded by the trained ``horizon``; the identity stays available."""
    model = _model()
    y_hist, u_hist, _ = _context()

    state = model.prime(y_hist, u_hist)
    long = model.rollout(state, np.zeros((_HORIZON + 3, _N_CONTROLS)))

    assert long.shape == (_HORIZON + 3, _N_EEG)
