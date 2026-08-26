"""Seam 2/3/4 -- the two ABCs and the cross-side parity pin for the waveform predictor."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from neuro.predictor.inference import InferencePredictor, WaveformMLPModel
from neuro.predictor.module import AutoregressiveMLP, TrainingPredictor
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from neuro.types import Activation, FloatArray

_SEED = 17
_N_Y, _N_U, _HORIZON = 3, 2, 5
_N_EEG, _N_CONTROLS, _HIDDEN = 5, 2, 6
_FS = 50.0
_RTOL, _ATOL = 1e-5, 1e-6


def _waveform_model(depth: int, activation: Activation, *, residual: bool) -> AutoregressiveMLP:
    """A random waveform MLP with nontrivial standardizers."""
    rng = np.random.default_rng(_SEED)
    model = AutoregressiveMLP(
        n_y=_N_Y,
        n_u=_N_U,
        horizon=_HORIZON,
        n_channels=_N_EEG,
        n_controls=_N_CONTROLS,
        hidden_size=_HIDDEN,
        depth=depth,
        activation=activation,
        residual=residual,
        dt=0.01,
        y_std=Standardizer(center=rng.uniform(-1.0, 1.0, _N_EEG), scale=rng.uniform(0.5, 2.0, _N_EEG)),
        u_std=Standardizer(center=rng.uniform(-1.0, 1.0, _N_CONTROLS), scale=rng.uniform(0.5, 2.0, _N_CONTROLS)),
    )
    linears = [m for m in model.layers if isinstance(m, torch.nn.Linear)]
    with torch.no_grad():
        for lin in linears:
            lin.weight.normal_()
            lin.weight.data.mul_(float(lin.in_features) ** -0.5)
            lin.bias.normal_()
            lin.bias.data.mul_(0.1)
    return model


def test_waveform_torch_module_is_a_training_predictor_without_a_runtime() -> None:
    """The torch module implements the training ABC and exposes no NumPy runtime methods."""
    model = _waveform_model(2, "softplus", residual=True)
    assert isinstance(model, TrainingPredictor)
    for method in ("prime", "step", "rollout", "prime_many", "rollout_many", "absorb", "is_ready", "initial_state"):
        assert not hasattr(model, method)


def test_waveform_jax_model_is_an_inference_predictor() -> None:
    """The jax adapter implements the inference ABC, the controller's priming seam included."""
    model = WaveformMLPModel.from_checkpoint(*_waveform_model(2, "softplus", residual=True).to_checkpoint())
    assert isinstance(model, InferencePredictor)
    assert model.n_outputs == _N_EEG
    assert model.n_channels == _N_EEG
    assert model.n_controls == _N_CONTROLS


@pytest.mark.parametrize("depth", [0, 1, 2])
@pytest.mark.parametrize("activation", ["relu", "tanh", "softplus"])
@pytest.mark.parametrize("residual", [False, True])
def test_waveform_cross_side_parity(depth: int, activation: Activation, residual: bool) -> None:  # noqa: FBT001
    """The jax ``rollout`` (raw in -> raw out) equals the decoded torch ``forward``.

    This replaces the float64 reference as the correctness pin: the torch module is handed to the
    jax side in memory via ``to_checkpoint``/``from_checkpoint`` and the two recursions are
    compared on the training seam.
    """
    module = _waveform_model(depth, activation, residual=residual)
    jax_model = WaveformMLPModel.from_checkpoint(*module.to_checkpoint())

    rng = np.random.default_rng(_SEED + 100)
    t0 = max(_N_Y, _N_U) + 3
    y_raw = rng.standard_normal((t0 + _HORIZON, _N_EEG))
    u_raw = rng.standard_normal((t0 + _HORIZON, _N_CONTROLS))
    k = t0 - 1

    row = np.concatenate(
        [
            module.y_std.transform(y_raw[k - _N_Y + 1 : k + 1]).reshape(-1),
            module.u_std.transform(u_raw[k - _N_U : k]).reshape(-1),
            module.u_std.transform(u_raw[k : k + _HORIZON]).reshape(-1),
        ]
    )
    with torch.no_grad():
        standardized = module(torch.as_tensor(row, dtype=torch.float32)[None, :]).numpy()[0]
    want = module.y_std.inverse_transform(standardized.reshape(_HORIZON, _N_EEG))

    got = np.asarray(jax_model.free_run(y_raw[:t0][None], u_raw[:t0][None], u_raw[t0 : t0 + _HORIZON][None]))[0]
    assert got.shape == (_HORIZON, _N_EEG)
    np.testing.assert_allclose(got, want, rtol=_RTOL, atol=_ATOL)
