"""Seam 1 -- the two-sided exchange checkpoint, tested at the ``.npz`` boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from neuro.config import StftGeometry
from neuro.predictor.inference import InferencePredictor
from neuro.predictor.mlp import ObservableMLPModel, WaveformMLPModel
from neuro.provenance import TrainingProvenance
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from pathlib import Path

    from neuro.types import FloatArray

_SEED = 23
_N_Y, _N_U, _HORIZON = 3, 2, 4
_N_EEG, _N_CONTROLS, _HIDDEN = 4, 2, 6


def _mlp_module() -> WaveformMLPModel:
    """A random MLP module with nontrivial standardizers and recorded provenance."""
    rng = np.random.default_rng(_SEED)
    model = WaveformMLPModel(
        n_y=_N_Y,
        n_u=_N_U,
        horizon=_HORIZON,
        n_channels=_N_EEG,
        n_controls=_N_CONTROLS,
        n_outputs=_N_EEG,
        hidden_size=_HIDDEN,
        depth=2,
        activation="softplus",
        dt=0.01,
        y_std=Standardizer(center=rng.uniform(-1.0, 1.0, _N_EEG), scale=rng.uniform(0.5, 2.0, _N_EEG)),
        u_std=Standardizer(center=rng.uniform(-1.0, 1.0, _N_CONTROLS), scale=rng.uniform(0.5, 2.0, _N_CONTROLS)),
    )
    model.downsample = 2
    model.provenance = TrainingProvenance(cutoff_hz=100.0, plant_fingerprint="abc")
    return model


def _waveform_context(seed: int) -> tuple[FloatArray, FloatArray]:
    """One continuous raw trajectory long enough to hold a primed free run and its training seam."""
    rng = np.random.default_rng(seed)
    t0 = max(_N_Y, _N_U) + 3
    return (
        rng.standard_normal((t0 + _HORIZON, _N_EEG)),
        rng.standard_normal((t0 + _HORIZON, _N_CONTROLS)),
    )


def test_training_save_jax_rollout_reproduces_the_decoded_training_forward(tmp_path: Path) -> None:
    """A portable checkpoint loaded on the jax side free-runs the same raw samples as ``forward``.

    Both the jax ``free_run`` and standardized ``forward`` share the same physical time alignment and
    shift the candidate control before predicting, agreeing to float32 tolerance.
    """
    module = _mlp_module()
    path = tmp_path / "mlp"
    module.save(path)
    jax_model = WaveformMLPModel.load(path)

    y_raw, u_raw = _waveform_context(_SEED + 9)
    t0 = max(_N_Y, _N_U) + 3
    k = t0 - 1
    y_hist = jnp.expand_dims(jnp.asarray(module.y_std.transform(y_raw[k - _N_Y + 1 : k + 1]), dtype=jnp.float32), 0)
    u_hist = jnp.expand_dims(jnp.asarray(module.u_std.transform(u_raw[k - _N_U : k]), dtype=jnp.float32), 0)
    u_future = jnp.expand_dims(jnp.asarray(module.u_std.transform(u_raw[k : k + _HORIZON]), dtype=jnp.float32), 0)
    standardized = np.asarray(module(y_hist, u_hist, u_future))[0]
    want = module.y_std.inverse_transform(standardized)

    got = np.asarray(jax_model.free_run(y_raw[: k + 1][None], u_raw[:k][None], u_raw[k : k + _HORIZON][None]))[0]
    np.testing.assert_allclose(got, want, rtol=2e-5, atol=2e-5)


def test_jax_save_training_load_round_trips_weights_buffers_and_metadata(tmp_path: Path) -> None:
    """A JAX model saved through the jax side reloads to the same weights, buffers and metadata."""
    module = _mlp_module()
    training_meta, training_arrays = module.to_checkpoint()
    jax_model = WaveformMLPModel.from_checkpoint(training_meta, training_arrays)
    path = tmp_path / "roundtrip"
    jax_model.save(path)

    loaded = WaveformMLPModel.load(path)

    for got, want in zip(loaded.weights, module.weights, strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    for got, want in zip(loaded.biases, module.biases, strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    np.testing.assert_array_equal(loaded.y_std.center, module.y_std.center)
    np.testing.assert_array_equal(loaded.y_std.scale, module.y_std.scale)
    np.testing.assert_array_equal(loaded.u_std.center, module.u_std.center)
    np.testing.assert_array_equal(loaded.u_std.scale, module.u_std.scale)
    assert loaded.activation == module.activation
    assert loaded.horizon == module.horizon
    assert loaded.dt == module.dt
    assert loaded.downsample == module.downsample
    assert loaded.provenance == module.provenance


def test_observable_training_save_jax_load_round_trips_geometry_and_per_output_standardizers(tmp_path: Path) -> None:
    """An observable portable checkpoint carries geometry and per-output standardizers that jax load restores."""
    rng = np.random.default_rng(_SEED + 50)
    geometry = StftGeometry(n_segment=64, n_hop=16, band_hz=[4.0, 30.0], n_bin_pool=2, kernel_width=3)
    n_outputs = _N_EEG * geometry.n_values(50.0)
    module = ObservableMLPModel(
        n_y=_N_Y,
        n_u=_N_U,
        horizon=_HORIZON,
        n_channels=_N_EEG,
        n_controls=_N_CONTROLS,
        n_outputs=n_outputs,
        hidden_size=_HIDDEN,
        depth=1,
        activation="relu",
        dt=0.01 * 16,
        y_std=Standardizer(
            center=rng.uniform(-1.0, 1.0, (_N_EEG, geometry.n_values(50.0))),
            scale=rng.uniform(0.5, 2.0, (_N_EEG, geometry.n_values(50.0))),
        ),
        u_std=Standardizer(center=rng.uniform(-1.0, 1.0, _N_CONTROLS), scale=rng.uniform(0.5, 2.0, _N_CONTROLS)),
        geometry=geometry,
    )

    path = tmp_path / "obs_model"
    module.save(path)

    # Test polymorphic load via InferencePredictor.load and direct ObservableMLPModel.load
    jax_model_poly = InferencePredictor.load(path)
    assert isinstance(jax_model_poly, ObservableMLPModel)
    assert jax_model_poly.geometry == geometry
    assert jax_model_poly.n_outputs == n_outputs
    np.testing.assert_array_equal(
        np.asarray(jax_model_poly.y_center).reshape(module.y_std.center.shape), module.y_std.center
    )
    np.testing.assert_array_equal(
        np.asarray(jax_model_poly.y_scale).reshape(module.y_std.scale.shape), module.y_std.scale
    )

    # Test reloading back to ObservableMLPModel
    reloaded_training = ObservableMLPModel.load(path)
    assert reloaded_training.geometry == geometry
    assert reloaded_training.n_outputs == n_outputs
    np.testing.assert_array_equal(reloaded_training.y_std.center, module.y_std.center)
    np.testing.assert_array_equal(reloaded_training.y_std.scale, module.y_std.scale)
