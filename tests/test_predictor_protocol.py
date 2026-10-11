"""Seam 2/3/4 -- the two ABCs and the cross-side parity pin for the waveform predictor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest

from neuro.config import StftGeometry
from neuro.predictor.base import AutoregressiveModel
from neuro.predictor.inference import InferencePredictor
from neuro.predictor.mlp import ObservableMLPModel, WaveformMLPModel
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from neuro.types import Activation

_SEED = 17
_N_Y, _N_U, _HORIZON = 3, 2, 5
_N_EEG, _N_CONTROLS, _HIDDEN = 5, 2, 6
_FS = 50.0
_RTOL, _ATOL = 1e-5, 1e-6


def _waveform_model(depth: int, activation: Activation, *, residual: bool) -> WaveformMLPModel:
    """A random waveform MLP with nontrivial standardizers."""
    rng = np.random.default_rng(_SEED)
    return WaveformMLPModel(
        n_y=_N_Y,
        n_u=_N_U,
        horizon=_HORIZON,
        n_channels=_N_EEG,
        n_controls=_N_CONTROLS,
        n_outputs=_N_EEG,
        hidden_size=_HIDDEN,
        depth=depth,
        activation=activation,
        residual=residual,
        dt=0.01,
        y_std=Standardizer(center=rng.uniform(-1.0, 1.0, _N_EEG), scale=rng.uniform(0.5, 2.0, _N_EEG)),
        u_std=Standardizer(center=rng.uniform(-1.0, 1.0, _N_CONTROLS), scale=rng.uniform(0.5, 2.0, _N_CONTROLS)),
    )


def test_waveform_model_supports_training_and_runtime() -> None:
    """One JAX model provides differentiable training and the controller's runtime interface."""
    model = _waveform_model(2, "softplus", residual=True)
    assert isinstance(model, AutoregressiveModel)
    assert isinstance(model, InferencePredictor)
    for method in ("rollout", "free_run", "absorb", "is_ready", "initial_state"):
        assert callable(getattr(model, method))


def test_waveform_jax_model_is_an_inference_predictor() -> None:
    """The jax adapter implements the inference ABC, the controller's priming seam included."""
    model = WaveformMLPModel.from_checkpoint(*_waveform_model(2, "softplus", residual=True).to_checkpoint())
    assert isinstance(model, InferencePredictor)
    assert model.n_outputs == _N_EEG
    assert model.n_channels == _N_EEG
    assert model.n_controls == _N_CONTROLS
    assert hasattr(InferencePredictor, "output")
    assert hasattr(model, "output")


@pytest.mark.parametrize("depth", [0, 1, 2])
@pytest.mark.parametrize("activation", ["relu", "tanh", "softplus"])
@pytest.mark.parametrize("residual", [False, True])
def test_waveform_cross_side_parity(depth: int, activation: Activation, residual: bool) -> None:  # noqa: FBT001 -- pytest passes parametrized cases positionally
    """Raw-unit JAX inference matches standardized training after unit recovery.

    The portable checkpoint preserves standardizers and the causal control alignment.
    """
    module = _waveform_model(depth, activation, residual=residual)
    jax_model = WaveformMLPModel.from_checkpoint(*module.to_checkpoint())

    rng = np.random.default_rng(_SEED + 100)
    t0 = max(_N_Y, _N_U) + 3
    y_raw = rng.standard_normal((t0 + _HORIZON, _N_EEG))
    u_raw = rng.standard_normal((t0 + _HORIZON, _N_CONTROLS))
    k = t0 - 1

    y_hist = jnp.expand_dims(jnp.asarray(module.y_std.transform(y_raw[k - _N_Y + 1 : k + 1]), dtype=jnp.float32), 0)
    u_hist = jnp.expand_dims(jnp.asarray(module.u_std.transform(u_raw[k - _N_U : k]), dtype=jnp.float32), 0)
    u_future = jnp.expand_dims(jnp.asarray(module.u_std.transform(u_raw[k : k + _HORIZON]), dtype=jnp.float32), 0)
    standardized = np.asarray(module(y_hist, u_hist, u_future))[0]
    want = module.y_std.inverse_transform(standardized)

    got = np.asarray(jax_model.free_run(y_raw[: k + 1][None], u_raw[:k][None], u_raw[k : k + _HORIZON][None]))[0]
    assert got.shape == (_HORIZON, _N_EEG)
    np.testing.assert_allclose(got, want, rtol=_RTOL, atol=_ATOL)


def test_cross_side_parity_with_distinct_output_width() -> None:
    """Standardized training and raw inference agree when output width exceeds channel count."""
    rng = np.random.default_rng(_SEED + 200)
    n_channels, n_outputs, n_controls = 2, 6, 2
    n_y, n_u, horizon = 3, 2, 4
    module = WaveformMLPModel(
        n_y=n_y,
        n_u=n_u,
        horizon=horizon,
        n_channels=n_channels,
        n_controls=n_controls,
        n_outputs=n_outputs,
        hidden_size=8,
        depth=1,
        activation="tanh",
        residual=True,
        dt=0.02,
        y_std=Standardizer(center=rng.uniform(-1.0, 1.0, n_outputs), scale=rng.uniform(0.5, 2.0, n_outputs)),
        u_std=Standardizer(center=rng.uniform(-1.0, 1.0, n_controls), scale=rng.uniform(0.5, 2.0, n_controls)),
    )

    meta, arrays = module.to_checkpoint()
    jax_model = WaveformMLPModel.from_checkpoint(meta, arrays)
    assert jax_model.n_outputs == n_outputs
    assert jax_model.n_channels == n_channels

    t0 = max(n_y, n_u) + 2
    y_raw = rng.standard_normal((t0 + horizon, n_outputs))
    u_raw = rng.standard_normal((t0 + horizon, n_controls))
    k = t0 - 1

    y_hist = jnp.expand_dims(jnp.asarray(module.y_std.transform(y_raw[k - n_y + 1 : k + 1]), dtype=jnp.float32), 0)
    u_hist = jnp.expand_dims(jnp.asarray(module.u_std.transform(u_raw[k - n_u : k]), dtype=jnp.float32), 0)
    u_future = jnp.expand_dims(jnp.asarray(module.u_std.transform(u_raw[k : k + horizon]), dtype=jnp.float32), 0)
    standardized = np.asarray(module(y_hist, u_hist, u_future))[0]
    want = module.y_std.inverse_transform(standardized)

    got = np.asarray(jax_model.free_run(y_raw[: k + 1][None], u_raw[:k][None], u_raw[k : k + horizon][None]))[0]
    assert got.shape == (horizon, n_outputs)
    np.testing.assert_allclose(got, want, rtol=_RTOL, atol=_ATOL)


def _observable_model(depth: int, activation: Activation, *, residual: bool) -> tuple[ObservableMLPModel, StftGeometry]:
    """A random observable MLP with recorded StftGeometry and per-output standardizers."""
    rng = np.random.default_rng(_SEED + 300)
    geometry = StftGeometry(n_segment=64, n_hop=16, band_hz=[4.0, 30.0], n_bin_pool=2, kernel_width=5)
    n_values = geometry.n_values(_FS)
    n_outputs = _N_EEG * n_values
    model = ObservableMLPModel(
        n_y=_N_Y,
        n_u=_N_U,
        horizon=_HORIZON,
        n_channels=_N_EEG,
        n_controls=_N_CONTROLS,
        n_outputs=n_outputs,
        hidden_size=_HIDDEN,
        depth=depth,
        activation=activation,
        residual=residual,
        dt=0.01 * 16,
        y_std=Standardizer(
            center=rng.uniform(-1.0, 1.0, (_N_EEG, n_values)),
            scale=rng.uniform(0.5, 2.0, (_N_EEG, n_values)),
        ),
        u_std=Standardizer(center=rng.uniform(-1.0, 1.0, _N_CONTROLS), scale=rng.uniform(0.5, 2.0, _N_CONTROLS)),
        geometry=geometry,
    )
    return model, geometry


def test_observable_jax_model_is_an_inference_predictor() -> None:
    """The jax observable adapter implements InferencePredictor and carries geometry and priming info."""
    module, geometry = _observable_model(2, "relu", residual=True)
    meta, arrays = module.to_checkpoint()
    model = ObservableMLPModel.from_checkpoint(meta, arrays)
    assert isinstance(model, InferencePredictor)
    assert model.n_channels == _N_EEG
    assert model.n_outputs == module.n_outputs
    assert model.n_controls == _N_CONTROLS
    assert model.geometry == geometry
    assert model.priming_steps == max(_N_Y, _N_U)


def test_observable_state_absorption_and_readiness() -> None:
    """Observable adapter absorbs raw Frames and control, and becomes ready after n_y Frames."""
    module, _ = _observable_model(1, "relu", residual=True)
    meta, arrays = module.to_checkpoint()
    model = ObservableMLPModel.from_checkpoint(meta, arrays)

    state = model.initial_state()
    assert not model.is_ready(state)

    rng = np.random.default_rng(42)
    for i in range(_N_Y):
        raw_frame = rng.standard_normal((model.n_channels, model.n_outputs // model.n_channels))
        raw_control = rng.standard_normal(_N_CONTROLS)
        state = model.absorb(state, raw_frame, raw_control)
        if i < _N_Y - 1:
            assert not model.is_ready(state)
        else:
            assert model.is_ready(state)


@pytest.mark.parametrize("depth", [0, 1, 2])
@pytest.mark.parametrize("activation", ["relu", "tanh", "softplus"])
@pytest.mark.parametrize("residual", [False, True])
def test_observable_cross_side_parity(depth: int, activation: Activation, residual: bool) -> None:  # noqa: FBT001 -- pytest passes parametrized cases positionally
    """The jax observable free_run equals the unstandardized standardized training forward on raw Frames."""
    module, _ = _observable_model(depth, activation, residual=residual)
    meta, arrays = module.to_checkpoint()
    jax_model = ObservableMLPModel.from_checkpoint(meta, arrays)

    rng = np.random.default_rng(_SEED + 400)
    t0 = max(_N_Y, _N_U) + 3
    y_raw = rng.standard_normal((t0 + _HORIZON, module.n_channels, module.n_outputs // module.n_channels))
    u_raw = rng.standard_normal((t0 + _HORIZON, _N_CONTROLS))
    k = t0 - 1

    y_hist = jnp.expand_dims(jnp.asarray(module.y_std.transform(y_raw[k - _N_Y + 1 : k + 1]), dtype=jnp.float32), 0)
    u_hist = jnp.expand_dims(jnp.asarray(module.u_std.transform(u_raw[k - _N_U : k]), dtype=jnp.float32), 0)
    u_future = jnp.expand_dims(jnp.asarray(module.u_std.transform(u_raw[k : k + _HORIZON]), dtype=jnp.float32), 0)
    standardized = np.asarray(module(y_hist, u_hist, u_future))[0]
    want = module.y_std.inverse_transform(standardized)

    got = np.asarray(jax_model.free_run(y_raw[: k + 1][None], u_raw[:k][None], u_raw[k : k + _HORIZON][None]))[0]
    assert got.shape == (_HORIZON, module.n_channels, module.n_outputs // module.n_channels)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)
