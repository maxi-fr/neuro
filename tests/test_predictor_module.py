"""Pin the torch AutoregressiveMLP against the CasADi bridge and guard the torch-free seam."""

from __future__ import annotations

import itertools
import subprocess
import sys
from typing import TYPE_CHECKING

import casadi as ca
import numpy as np
import pytest
import torch

from neuro.nn_predictor_casadi import NNSymbolicModel
from neuro.predictor.artifact import MLPArtifact
from neuro.predictor.module import AutoregressiveMLP
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from neuro.predictor.artifact import Activation
    from neuro.types import FloatArray

_SEED = 17
_N_Y, _N_U, _HORIZON = 3, 2, 5
_N_EEG, _N_CONTROLS, _HIDDEN = 5, 2, 6


def _random_layers(
    rng: np.random.Generator, in_size: int, out_size: int, hidden_size: int, depth: int
) -> tuple[tuple[FloatArray, FloatArray], ...]:
    """Random ``(W, b)`` pairs for a ``depth``-hidden-layer MLP, scaled so activations stay O(1)."""
    sizes = [in_size, *[hidden_size] * depth, out_size]
    return tuple(
        (rng.standard_normal((n_out, n_in)) / np.sqrt(n_in), rng.standard_normal(n_out) * 0.1)
        for n_in, n_out in itertools.pairwise(sizes)
    )


def _build_artifact(depth: int, activation: Activation) -> MLPArtifact:
    """A random artifact with nontrivial standardizers."""
    rng = np.random.default_rng(_SEED)
    y_std = Standardizer(center=rng.uniform(-1.0, 1.0, _N_EEG), scale=rng.uniform(0.5, 2.0, _N_EEG))
    u_std = Standardizer(center=rng.uniform(-1.0, 1.0, _N_CONTROLS), scale=rng.uniform(0.5, 2.0, _N_CONTROLS))
    return MLPArtifact(
        layers=_random_layers(rng, _N_Y * _N_EEG + _N_U * _N_CONTROLS, _N_EEG, _HIDDEN, depth),
        activation=activation,
        n_y=_N_Y,
        n_u=_N_U,
        horizon=_HORIZON,
        n_channels=_N_EEG,
        n_controls=_N_CONTROLS,
        dt=0.01,
        downsample=2,
        y_std=y_std,
        u_std=u_std,
    )


def _casadi_rollout(art: MLPArtifact, y_hist: FloatArray, u_hist: FloatArray, u_future: FloatArray) -> FloatArray:
    """Chain ``f_step``/``f_out`` over the horizon on RAW controls -> raw EEG ``(horizon, n_channels)``."""
    sym = NNSymbolicModel(art)
    x = np.concatenate([art.encode(y_hist[-art.n_y :]).flatten(), u_hist[-art.n_u :].flatten()])
    preds = []
    for t in range(art.horizon):
        x = np.array(ca.DM(sym.f_step(x, u_future[t]))).flatten()
        preds.append(np.array(ca.DM(sym.f_out(x))).flatten())
    return np.stack(preds)


def _model_space_inputs(art: MLPArtifact, y_hist: FloatArray, u_hist: FloatArray, u_future: FloatArray) -> FloatArray:
    """Assemble the torch input row, controls transformed as standardizer does."""
    return np.concatenate(
        [
            art.encode(y_hist[-art.n_y :]).flatten(),
            art.u_std.transform(u_hist[-art.n_u :]).flatten(),
            art.u_std.transform(u_future).flatten(),
        ]
    )


def _context(art: MLPArtifact, seed: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Raw EEG history, raw control history and raw future controls."""
    rng = np.random.default_rng(seed)
    hist = max(art.n_y, art.n_u)
    return (
        rng.standard_normal((hist, art.n_channels)),
        rng.standard_normal((hist, art.n_controls)),
        rng.standard_normal((art.horizon, art.n_controls)),
    )


def test_prime_rollout_matches_the_training_window_at_the_same_index() -> None:
    """``prime`` + ``rollout`` reproduce the torch rollout of the training window over the same targets."""
    art = _build_artifact(2, "softplus")
    n_y, n_u, horizon = art.n_y, art.n_u, art.horizon
    rng = np.random.default_rng(_SEED + 9)
    t0 = max(n_y, n_u) + 3
    y_raw = rng.standard_normal((t0 + horizon, art.n_channels))
    u_raw = rng.standard_normal((t0 + horizon, art.n_controls))

    k = t0 - 1
    x = np.concatenate(
        [
            art.encode(y_raw[k - n_y + 1 : k + 1]).flatten(),
            art.u_std.transform(u_raw[k - n_u : k]).flatten(),
            art.u_std.transform(u_raw[k : k + horizon]).flatten(),
        ]
    )
    model = AutoregressiveMLP.from_artifact(art)
    pred = model(torch.as_tensor(x)[None, :]).detach().numpy().reshape(horizon, art.n_channels)
    want = art.decode(pred)

    got = art.rollout(art.prime(y_raw[:t0], u_raw[:t0]), u_raw[t0 : t0 + horizon])

    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("activation", ["relu", "tanh", "softplus"])
@pytest.mark.parametrize("depth", [0, 2])
def test_torch_rollout_matches_casadi(depth: int, activation: Activation) -> None:
    """The torch rollout reproduces the CasADi f_step/f_out chain, closing torch == CasADi."""
    art = _build_artifact(depth, activation)
    y_hist, u_hist, u_future = _context(art, _SEED + 1)

    want = _casadi_rollout(art, y_hist, u_hist, u_future)

    model = AutoregressiveMLP.from_artifact(art)
    x = torch.as_tensor(_model_space_inputs(art, y_hist, u_hist, u_future))[None, :]
    pred = model(x).detach().numpy().reshape(art.horizon, art.n_channels)
    got = art.decode(pred)

    assert got.shape == (art.horizon, art.n_channels)
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)


def test_forward_is_row_independent() -> None:
    """Stacking two contexts into one batch predicts the same as running them one at a time."""
    art = _build_artifact(2, "tanh")
    model = AutoregressiveMLP.from_artifact(art)

    rows = [_model_space_inputs(art, *_context(art, _SEED + offset)) for offset in (2, 3)]
    batched = model(torch.as_tensor(np.stack(rows))).detach().numpy()
    singles = np.concatenate([model(torch.as_tensor(row)[None, :]).detach().numpy() for row in rows])

    assert batched.shape == (2, art.horizon * art.n_channels)
    np.testing.assert_array_equal(batched, singles)


@pytest.mark.parametrize("depth", [0, 2])
def test_artifact_round_trip_is_exact(depth: int) -> None:
    """``from_artifact`` then ``to_artifact`` returns bit-identical weights and metadata."""
    art = _build_artifact(depth, "softplus")
    got = AutoregressiveMLP.from_artifact(art).to_artifact(art.dt, art.downsample, art.y_std, art.u_std)

    assert got.meta == art.meta
    for (got_w, got_b), (want_w, want_b) in zip(got.layers, art.layers, strict=True):
        np.testing.assert_array_equal(got_w, want_w)
        np.testing.assert_array_equal(got_b, want_b)


def test_module_layers_sequential() -> None:
    """The 1-step MLP is an nn.Sequential interleaving nn.Linear and activation modules."""
    model = AutoregressiveMLP(
        n_y=2,
        n_u=1,
        horizon=3,
        n_channels=4,
        n_controls=2,
        hidden_size=8,
        depth=2,
        activation="softplus",
    )
    assert isinstance(model.layers, torch.nn.Sequential)
    assert len(model.layers) == 5
    assert isinstance(model.layers[0], torch.nn.Linear)
    assert isinstance(model.layers[1], torch.nn.Softplus)
    assert isinstance(model.layers[2], torch.nn.Linear)
    assert isinstance(model.layers[3], torch.nn.Softplus)
    assert isinstance(model.layers[4], torch.nn.Linear)


@pytest.mark.parametrize(
    "module",
    [
        "neuro.predictor.artifact",
        "neuro.predictor.data",
        "neuro.nn_predictor_casadi",
        "neuro.control",
        "neuro.control.zero",
        "neuro.control.threshold",
        "neuro.control.waveform",
        "neuro.control.nlp",
        "neuro.control.solvers",
        "neuro.control.nonlinear_mpc",
        "neuro.control.linear_mpc",
        "neuro.artifacts",
    ],
)
def test_control_path_never_imports_torch(module: str) -> None:
    """The control path stays torch-free, so the rewrite cannot reach it."""
    code = f"import importlib, sys; importlib.import_module({module!r}); assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603
