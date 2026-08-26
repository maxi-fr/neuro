"""Seam 1 -- the two-sided exchange checkpoint, tested at the ``.npz`` boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from neuro.config import StftGeometry
from neuro.predictor.inference import ObservableModel, WaveformMLPModel
from neuro.predictor.module import AutoregressiveMLP
from neuro.predictor.observable_module import StepwiseObservableMLP
from neuro.provenance import TrainingProvenance
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from neuro.types import FloatArray

_SEED = 23
_N_Y, _N_U, _HORIZON = 3, 2, 4
_N_EEG, _N_CONTROLS, _HIDDEN = 4, 2, 6


def _mlp_module() -> AutoregressiveMLP:
    """A random MLP module with nontrivial standardizers and recorded provenance."""
    rng = np.random.default_rng(_SEED)
    model = AutoregressiveMLP(
        n_y=_N_Y,
        n_u=_N_U,
        horizon=_HORIZON,
        n_channels=_N_EEG,
        n_controls=_N_CONTROLS,
        hidden_size=_HIDDEN,
        depth=2,
        activation="softplus",
        dt=0.01,
        y_std=Standardizer(center=rng.uniform(-1.0, 1.0, _N_EEG), scale=rng.uniform(0.5, 2.0, _N_EEG)),
        u_std=Standardizer(center=rng.uniform(-1.0, 1.0, _N_CONTROLS), scale=rng.uniform(0.5, 2.0, _N_CONTROLS)),
    )
    with torch.no_grad():
        for module in model.layers:
            if isinstance(module, torch.nn.Linear):
                module.weight.normal_()
                module.bias.normal_()
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


def test_torch_save_jax_rollout_reproduces_the_decoded_torch_forward(tmp_path: Path) -> None:
    """A torch checkpoint loaded on the jax side free-runs the same raw samples as ``forward``.

    The jax ``rollout`` primes on raw history ending at ``t0 - 1`` and shifts each future control
    in *after* predicting; the torch ``forward`` shifts in *before*. The two orders are the same
    seam, so the decoded ``forward`` and the jax rollout agree to float32 tolerance.
    """
    module = _mlp_module()
    path = tmp_path / "mlp"
    module.save(path)
    jax_model = WaveformMLPModel.load(path)

    y_raw, u_raw = _waveform_context(_SEED + 9)
    t0 = max(_N_Y, _N_U) + 3
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
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


def test_jax_save_torch_load_round_trips_weights_buffers_and_metadata(tmp_path: Path) -> None:
    """A torch model saved through the jax side reloads to the same weights, buffers and metadata."""
    module = _mlp_module()
    torch_meta, torch_arrays = module.to_checkpoint()
    jax_model = WaveformMLPModel.from_checkpoint(torch_meta, torch_arrays)
    path = tmp_path / "roundtrip"
    jax_model.save(path)

    loaded = AutoregressiveMLP.load(path)

    want_linears = [m for m in module.layers if isinstance(m, torch.nn.Linear)]
    got_linears = [m for m in loaded.layers if isinstance(m, torch.nn.Linear)]
    for got, want in zip(got_linears, want_linears, strict=True):
        np.testing.assert_array_equal(got.weight.detach().numpy(), want.weight.detach().numpy())
        np.testing.assert_array_equal(got.bias.detach().numpy(), want.bias.detach().numpy())
    np.testing.assert_array_equal(loaded.y_std.center, module.y_std.center)
    np.testing.assert_array_equal(loaded.y_std.scale, module.y_std.scale)
    np.testing.assert_array_equal(loaded.u_std.center, module.u_std.center)
    np.testing.assert_array_equal(loaded.u_std.scale, module.u_std.scale)
    assert loaded.activation == module.activation
    assert loaded.horizon == module.horizon
    assert loaded.dt == module.dt
    assert loaded.downsample == module.downsample
    assert loaded.provenance == module.provenance


def test_observable_torch_save_jax_rollout_reproduces_the_decoded_torch_forward(
    tmp_path: Path, make_observable_model: Callable[..., ObservableModel]
) -> None:
    """An observable torch checkpoint loaded on the jax side free-runs the same raw Frames."""
    geometry = StftGeometry(n_segment=8, n_hop=4)
    jax_model = make_observable_model(geometry, horizon=16)
    path = tmp_path / "observable"
    jax_model.save(path)
    torch_model = StepwiseObservableMLP.load(path)

    rng = np.random.default_rng(_SEED + 5)
    k = jax_model.priming_steps
    y_hist = rng.standard_normal((k, jax_model.n_channels))
    u_hist = rng.standard_normal((k, jax_model.n_controls))
    u_future = rng.standard_normal((jax_model.horizon, jax_model.n_controls))

    row = np.concatenate(
        [
            torch_model.y_std.transform(y_hist[-jax_model.n_y :]).reshape(-1),
            torch_model.u_std.transform(u_hist[-jax_model.n_u :]).reshape(-1),
            torch_model.u_std.transform(u_future).reshape(-1),
        ]
    )
    with torch.no_grad():
        standardized = torch_model(torch.as_tensor(row, dtype=torch.float32)[None, :]).numpy()[0]
    want = torch_model.l_std.inverse_transform(standardized)

    got = np.asarray(jax_model.free_run(y_hist[None], u_hist[None], u_future[None]))[0]
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


def test_observable_jax_save_torch_load_round_trips_weights_buffers_and_geometry(
    tmp_path: Path, make_observable_model: Callable[..., ObservableModel]
) -> None:
    """An observable model written on the jax side reloads on torch with the same buffers."""
    geometry = StftGeometry(n_segment=8, n_hop=4, n_bin_pool=2, kernel="hann", kernel_width=2, band_hz=(2.0, 20.0))
    model = make_observable_model(geometry, horizon=20)
    path = tmp_path / "observable_roundtrip"
    model.save(path)

    loaded = StepwiseObservableMLP.load(path)
    assert loaded.geometry == geometry
    assert loaded.z_dim == model.z_dim
    assert loaded.n_frames() == model.n_frames(model.horizon)
    assert loaded.fs == pytest.approx(model.fs)
    assert loaded.downsample == model.downsample
    # The torch side stores the standardizers as float32 buffers, so the float64 jax arrays come
    # back rounded to float32 precision rather than bit-identically.
    np.testing.assert_allclose(loaded.y_std.center, np.asarray(model.y_center), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(loaded.y_std.scale, np.asarray(model.y_scale), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(loaded.l_std.center, np.asarray(model.l_center), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(loaded.l_std.scale, np.asarray(model.l_scale), rtol=1e-5, atol=1e-6)
