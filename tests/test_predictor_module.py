"""Pin the torch waveform training forward and guard the torch-free inference path."""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from neuro.predictor.module import AutoregressiveMLP

if TYPE_CHECKING:
    from neuro.types import Activation, FloatArray

_SEED = 17
_N_Y, _N_U, _HORIZON = 3, 2, 5
_N_EEG, _N_CONTROLS, _HIDDEN = 5, 2, 6


def _model(depth: int = 2, activation: Activation = "softplus") -> AutoregressiveMLP:
    """A random module with nontrivial standardizers."""
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
    )
    with torch.no_grad():
        for module in model.layers:
            if isinstance(module, torch.nn.Linear):
                module.weight.normal_()
                module.bias.normal_()
    return model


def _context(seed: int) -> FloatArray:
    """One model-space input row: standardized history plus standardized future controls."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(_N_Y * _N_EEG + (_N_U + _HORIZON) * _N_CONTROLS).astype(np.float64)


def test_forward_is_row_independent() -> None:
    """Stacking two contexts into one batch predicts the same as running them one at a time."""
    model = _model(2, "tanh")

    rows = [_context(_SEED + offset) for offset in (2, 3)]
    batched = model(torch.as_tensor(np.stack(rows), dtype=torch.float32)).detach().numpy()
    singles = np.concatenate(
        [model(torch.as_tensor(row, dtype=torch.float32)[None, :]).detach().numpy() for row in rows]
    )

    assert batched.shape == (2, _HORIZON * _N_EEG)
    np.testing.assert_allclose(batched, singles, rtol=1e-5, atol=1e-6)


def test_residual_skip_makes_a_zero_mlp_predict_pure_persistence() -> None:
    """With the residual skip, a zero-weight MLP predicts the last window sample unchanged.

    ``z_{t+1} = layers(x) + z_t`` degenerates to the identity when ``layers`` is zero, so the
    whole free run is the constant last window sample -- the persistence prior the skip bakes in.
    """
    model = AutoregressiveMLP(
        n_y=2,
        n_u=1,
        horizon=3,
        n_channels=2,
        n_controls=1,
        hidden_size=4,
        depth=2,
        activation="softplus",
        residual=True,
    )
    with torch.no_grad():
        for module in model.layers:
            if isinstance(module, torch.nn.Linear):
                module.weight.zero_()
                module.bias.zero_()

    rng = np.random.default_rng(11)
    row = rng.standard_normal(model.n_y * 2 + (model.n_u + model.horizon) * 1)
    pred = model(torch.as_tensor(row, dtype=torch.float32)[None, :]).detach().numpy()[0]
    z_t = row[model.n_y * 2 - 2 : model.n_y * 2]  # the last sample of the y-window
    np.testing.assert_allclose(
        pred.reshape(model.horizon, 2), np.broadcast_to(z_t, (model.horizon, 2)), rtol=1e-6, atol=1e-7
    )


def test_without_residual_a_zero_mlp_predicts_zero() -> None:
    """Without the skip the same zero-weight MLP emits zeros: the skip is what carries the level."""
    model = AutoregressiveMLP(
        n_y=2,
        n_u=1,
        horizon=3,
        n_channels=2,
        n_controls=1,
        hidden_size=4,
        depth=2,
        activation="softplus",
        residual=False,
    )
    with torch.no_grad():
        for module in model.layers:
            if isinstance(module, torch.nn.Linear):
                module.weight.zero_()
                module.bias.zero_()

    rng = np.random.default_rng(12)
    row = rng.standard_normal(model.n_y * 2 + (model.n_u + model.horizon) * 1)
    pred = model(torch.as_tensor(row, dtype=torch.float32)[None, :]).detach().numpy()[0]
    np.testing.assert_array_equal(pred, np.zeros(model.horizon * 2))


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
        "neuro.predictor.inference",
        "neuro.control",
        "neuro.control.zero",
        "neuro.control.threshold",
        "neuro.control.schedule",
        "neuro.control.costs",
        "neuro.control.mpc",
    ],
)
def test_inference_path_never_imports_torch(module: str) -> None:
    """The jax inference path stays torch-free, so the rewrite cannot reach it."""
    code = f"import importlib, sys; importlib.import_module({module!r}); assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603


def test_autoregressive_mlp_supports_distinct_output_width() -> None:
    """AutoregressiveMLP unrolls with n_outputs independent of n_channels."""
    model = AutoregressiveMLP(
        n_y=2,
        n_u=1,
        horizon=3,
        n_channels=2,
        n_controls=1,
        n_outputs=6,
        hidden_size=8,
        depth=2,
        activation="softplus",
        residual=True,
    )
    assert model.n_outputs == 6
    assert model.n_channels == 2
    assert model.y_center.shape == (6,)
    assert model.y_scale.shape == (6,)

    # Zero-weight residual predicts persistence of length-6 output
    with torch.no_grad():
        for module in model.layers:
            if isinstance(module, torch.nn.Linear):
                module.weight.zero_()
                module.bias.zero_()

    rng = np.random.default_rng(42)
    row = rng.standard_normal(model.n_y * 6 + (model.n_u + model.horizon) * 1)
    pred = model(torch.as_tensor(row, dtype=torch.float32)[None, :]).detach().numpy()[0]
    assert pred.shape == (3 * 6,)
    z_last = row[6:12]
    np.testing.assert_allclose(pred.reshape(3, 6), np.broadcast_to(z_last, (3, 6)), rtol=1e-6, atol=1e-7)
