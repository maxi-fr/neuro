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

from neuro.checkpoint import MLPCheckpoint
from neuro.nn_predictor_casadi import NNSymbolicModel
from neuro.predictor.module import AutoregressiveMLP
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from neuro.types import Activation, FloatArray

_SEED = 17
_N_Y, _N_U, _HORIZON = 3, 2, 5
_N_EEG, _N_CONTROLS, _HIDDEN = 5, 2, 6


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
    depth: int, activation: Activation
) -> tuple[tuple[tuple[FloatArray, FloatArray], ...], Standardizer, Standardizer]:
    """Random layers and nontrivial standardizers, shared by the module and its checkpoint twin."""
    rng = np.random.default_rng(_SEED)
    y_std = Standardizer(center=rng.uniform(-1.0, 1.0, _N_EEG), scale=rng.uniform(0.5, 2.0, _N_EEG))
    u_std = Standardizer(center=rng.uniform(-1.0, 1.0, _N_CONTROLS), scale=rng.uniform(0.5, 2.0, _N_CONTROLS))
    layers = _random_layers(rng, _N_Y * _N_EEG + _N_U * _N_CONTROLS, _N_EEG, _HIDDEN, depth)
    return layers, y_std, u_std


def _build_module(depth: int, activation: Activation) -> AutoregressiveMLP:
    """A random module with nontrivial standardizers."""
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


def _build_checkpoint(depth: int, activation: Activation) -> MLPCheckpoint:
    """The float64 checkpoint twin of the module, for the CasADi side of a parity test."""
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


def _casadi_rollout(ckpt: MLPCheckpoint, y_hist: FloatArray, u_hist: FloatArray, u_future: FloatArray) -> FloatArray:
    """Chain ``f_step``/``f_out`` over the horizon on RAW controls -> raw EEG ``(horizon, n_channels)``."""
    sym = NNSymbolicModel(ckpt)
    x = np.concatenate([ckpt.y_std.transform(y_hist[-ckpt.n_y :]).flatten(), u_hist[-ckpt.n_u :].flatten()])
    preds = []
    for t in range(ckpt.horizon):
        x = np.array(ca.DM(sym.f_step(x, u_future[t]))).flatten()
        preds.append(np.array(ca.DM(sym.f_out(x))).flatten())
    return np.stack(preds)


def _model_space_inputs(
    ckpt: MLPCheckpoint, y_hist: FloatArray, u_hist: FloatArray, u_future: FloatArray
) -> FloatArray:
    """Assemble the torch input row, controls transformed as standardizer does."""
    return np.concatenate(
        [
            ckpt.y_std.transform(y_hist[-ckpt.n_y :]).flatten(),
            ckpt.u_std.transform(u_hist[-ckpt.n_u :]).flatten(),
            ckpt.u_std.transform(u_future).flatten(),
        ]
    )


def _context(ckpt: MLPCheckpoint, seed: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Raw EEG history, raw control history and raw future controls."""
    rng = np.random.default_rng(seed)
    hist = max(ckpt.n_y, ckpt.n_u)
    return (
        rng.standard_normal((hist, ckpt.n_channels)),
        rng.standard_normal((hist, ckpt.n_controls)),
        rng.standard_normal((ckpt.horizon, ckpt.n_controls)),
    )


def test_prime_rollout_matches_the_training_window_at_the_same_index() -> None:
    """``prime`` + ``rollout`` reproduce the torch rollout of the training window over the same targets."""
    ckpt = _build_checkpoint(2, "softplus")
    n_y, n_u, horizon = ckpt.n_y, ckpt.n_u, ckpt.horizon
    rng = np.random.default_rng(_SEED + 9)
    t0 = max(n_y, n_u) + 3
    y_raw = rng.standard_normal((t0 + horizon, ckpt.n_channels))
    u_raw = rng.standard_normal((t0 + horizon, ckpt.n_controls))

    k = t0 - 1
    x = np.concatenate(
        [
            ckpt.y_std.transform(y_raw[k - n_y + 1 : k + 1]).flatten(),
            ckpt.u_std.transform(u_raw[k - n_u : k]).flatten(),
            ckpt.u_std.transform(u_raw[k : k + horizon]).flatten(),
        ]
    )
    model = _build_module(2, "softplus")
    pred = model(torch.as_tensor(x, dtype=torch.float32)[None, :]).detach().numpy().reshape(horizon, ckpt.n_channels)
    want = ckpt.y_std.inverse_transform(pred)

    got = model.rollout(model.prime(y_raw[:t0], u_raw[:t0]), u_raw[t0 : t0 + horizon])

    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("activation", ["relu", "tanh", "softplus"])
@pytest.mark.parametrize("depth", [0, 2])
def test_torch_rollout_matches_casadi(depth: int, activation: Activation) -> None:
    """The torch rollout reproduces the CasADi f_step/f_out chain, closing torch == CasADi."""
    ckpt = _build_checkpoint(depth, activation)
    y_hist, u_hist, u_future = _context(ckpt, _SEED + 1)

    want = _casadi_rollout(ckpt, y_hist, u_hist, u_future)

    model = _build_module(depth, activation)
    x = torch.as_tensor(_model_space_inputs(ckpt, y_hist, u_hist, u_future), dtype=torch.float32)[None, :]
    pred = model(x).detach().numpy().reshape(ckpt.horizon, ckpt.n_channels)
    got = ckpt.y_std.inverse_transform(pred)

    assert got.shape == (ckpt.horizon, ckpt.n_channels)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


def test_forward_is_row_independent() -> None:
    """Stacking two contexts into one batch predicts the same as running them one at a time."""
    ckpt = _build_checkpoint(2, "tanh")
    model = _build_module(2, "tanh")

    rows = [_model_space_inputs(ckpt, *_context(ckpt, _SEED + offset)) for offset in (2, 3)]
    batched = model(torch.as_tensor(np.stack(rows), dtype=torch.float32)).detach().numpy()
    singles = np.concatenate(
        [model(torch.as_tensor(row, dtype=torch.float32)[None, :]).detach().numpy() for row in rows]
    )

    assert batched.shape == (2, ckpt.horizon * ckpt.n_channels)
    np.testing.assert_allclose(batched, singles, rtol=1e-5, atol=1e-6)


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
        "neuro.predictor.data",
        "neuro.predictor.checkpoint",
        "neuro.checkpoint",
        "neuro.nn_predictor_casadi",
        "neuro.esn",
        "neuro.esn_predictor_casadi",
        "neuro.observable",
        "neuro.observable_casadi",
        "neuro.control",
        "neuro.control.zero",
        "neuro.control.threshold",
        "neuro.control.waveform",
        "neuro.control.nlp",
        "neuro.control.solvers",
        "neuro.control.nonlinear_mpc",
        "neuro.control.linear_mpc",
    ],
)
def test_control_path_never_imports_torch(module: str) -> None:
    """The control path stays torch-free, so the rewrite cannot reach it."""
    code = f"import importlib, sys; importlib.import_module({module!r}); assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603
