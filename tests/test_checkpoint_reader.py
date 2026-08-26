"""The torch-free checkpoint reader: numpy weights + metadata from the numpy-checkpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from neuro.checkpoint import (
    MLPCheckpoint,
    ObservableCheckpoint,
    load_any,
    load_mlp,
    load_observable,
    load_rollout,
)
from neuro.config import StftGeometry
from neuro.predictor.checkpoint import save_checkpoint
from neuro.predictor.module import AutoregressiveMLP
from neuro.predictor.observable_module import StepwiseObservableMLP
from neuro.provenance import TrainingProvenance
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_SEED = 23


def _mlp_module() -> AutoregressiveMLP:
    """A random MLP module with nontrivial standardizers and recorded provenance."""
    rng = np.random.default_rng(_SEED)
    model = AutoregressiveMLP(
        n_y=3,
        n_u=2,
        horizon=4,
        n_channels=4,
        n_controls=2,
        hidden_size=6,
        depth=2,
        activation="softplus",
        dt=0.01,
        y_std=Standardizer(center=rng.uniform(-1.0, 1.0, 4), scale=rng.uniform(0.5, 2.0, 4)),
        u_std=Standardizer(center=rng.uniform(-1.0, 1.0, 2), scale=rng.uniform(0.5, 2.0, 2)),
    )
    with torch.no_grad():
        for module in model.layers:
            if isinstance(module, torch.nn.Linear):
                module.weight.normal_()
                module.bias.normal_()
    model.downsample = 2
    model.provenance = TrainingProvenance(cutoff_hz=100.0, plant_fingerprint="abc")
    return model


def test_mlp_reader_round_trips_weights_standardizers_and_metadata(tmp_path: Path) -> None:
    """``load_mlp`` yields what ``AutoregressiveMLP.save`` wrote, exactly as the module holds it."""
    module = _mlp_module()
    path = tmp_path / "mlp"
    module.save(path)

    got = load_mlp(path)
    want = MLPCheckpoint(
        layers=tuple(
            (m.weight.detach().cpu().numpy(), m.bias.detach().cpu().numpy())
            for m in module.layers
            if isinstance(m, torch.nn.Linear)
        ),
        activation=module.activation,
        n_y=module.n_y,
        n_u=module.n_u,
        horizon=module.horizon,
        n_channels=module.n_channels,
        n_controls=module.n_controls,
        hidden_size=module.hidden_size,
        depth=module.depth,
        dt=module.dt,
        downsample=module.downsample,
        y_std=module.y_std,
        u_std=module.u_std,
        residual=module.residual,
        provenance=module.provenance,
    )

    assert got.model_type == "mlp"
    assert got.activation == module.activation
    assert got.residual == module.residual
    assert got.n_y == module.n_y
    assert got.n_u == module.n_u
    assert got.horizon == module.horizon
    assert got.n_channels == module.n_channels
    assert got.n_controls == module.n_controls
    assert got.hidden_size == 6
    assert got.depth == 2
    assert got.dt == pytest.approx(module.dt)
    assert got.downsample == module.downsample
    assert got.priming_steps == max(module.n_y, module.n_u)
    assert not got.is_linear
    assert got.provenance == module.provenance
    for (got_w, got_b), (want_w, want_b) in zip(got.layers, want.layers, strict=True):
        np.testing.assert_array_equal(got_w, want_w)
        np.testing.assert_array_equal(got_b, want_b)
    np.testing.assert_array_equal(got.y_std.center, want.y_std.center)
    np.testing.assert_array_equal(got.y_std.scale, want.y_std.scale)
    np.testing.assert_array_equal(got.u_std.center, want.u_std.center)
    np.testing.assert_array_equal(got.u_std.scale, want.u_std.scale)


def test_mlp_checkpoint_save_load_round_trips_recorded_provenance(tmp_path: Path) -> None:
    """``MLPCheckpoint.save`` then ``load_mlp`` preserves the recorded metadata bit-exactly."""
    module = _mlp_module()
    ckpt = MLPCheckpoint(
        layers=tuple(
            (m.weight.detach().cpu().numpy(), m.bias.detach().cpu().numpy())
            for m in module.layers
            if isinstance(m, torch.nn.Linear)
        ),
        activation=module.activation,
        n_y=module.n_y,
        n_u=module.n_u,
        horizon=module.horizon,
        n_channels=module.n_channels,
        n_controls=module.n_controls,
        hidden_size=module.hidden_size,
        depth=module.depth,
        dt=module.dt,
        downsample=module.downsample,
        y_std=module.y_std,
        u_std=module.u_std,
        residual=module.residual,
        provenance=module.provenance,
    )
    path = tmp_path / "mlp_ckpt"
    ckpt.save(path)

    got = load_mlp(path)
    assert got.provenance == module.provenance
    assert got.residual == module.residual
    # The torch module loader reads the same layout the dataclass save wrote.
    assert AutoregressiveMLP.load(path).n_y == module.n_y


@pytest.mark.parametrize(
    "geometry", [StftGeometry(n_segment=8, n_hop=4), StftGeometry(n_segment=10, n_hop=5, n_bin_pool=2)]
)
def test_observable_reader_round_trips_weights_standardizers_and_geometry(
    geometry: StftGeometry,
    make_observable_checkpoint: Callable[..., ObservableCheckpoint],
    tmp_path: Path,
) -> None:
    """``load_observable`` yields what the dataclass save wrote, readable by the torch module too."""
    ckpt = make_observable_checkpoint(geometry, horizon=16)
    path = tmp_path / "obs"
    ckpt.save(path)

    got = load_observable(path)
    assert got.model_type == "observable"
    assert got.geometry == ckpt.geometry
    assert got.residual == ckpt.residual
    assert got.fs == pytest.approx(ckpt.fs)
    assert got.z_dim == ckpt.z_dim
    assert got.n_values == ckpt.n_values
    assert got.n_frames() == ckpt.n_frames()
    for got_block, saved_block in ((got.lift, ckpt.lift), (got.transition, ckpt.transition)):
        for (w, b), (w2, b2) in zip(got_block, saved_block, strict=True):
            np.testing.assert_array_equal(w, w2)
            np.testing.assert_array_equal(b, b2)
    np.testing.assert_array_equal(got.readout[0], ckpt.readout[0])
    np.testing.assert_array_equal(got.readout[1], ckpt.readout[1])
    for got_std, saved_std in ((got.y_std, ckpt.y_std), (got.u_std, ckpt.u_std), (got.l_std, ckpt.l_std)):
        np.testing.assert_array_equal(got_std.center, saved_std.center)
        np.testing.assert_array_equal(got_std.scale, saved_std.scale)
    # The torch module loader reads the same layout the dataclass save wrote.
    assert StepwiseObservableMLP.load(path).n_frames() == ckpt.n_frames()


def test_load_any_dispatches_and_load_rollout_rejects_observable(tmp_path: Path) -> None:
    """``load_any`` switches on ``model_type``; ``load_rollout`` refuses the observable forecast."""
    mlp = tmp_path / "mlp"
    _mlp_module().save(mlp)
    obs = tmp_path / "obs"
    ckpt = ObservableCheckpoint(
        lift=((np.ones((2, 4)), np.zeros(2)),),
        transition=((np.ones((2, 4)), np.zeros(2)),),
        readout=(np.ones((2, 2)), np.zeros(2)),
        activation="relu",
        n_y=1,
        n_u=1,
        horizon=4,
        n_channels=2,
        n_controls=2,
        dt=0.02,
        downsample=1,
        geometry=StftGeometry(n_segment=4, n_hop=2),
        y_std=Standardizer(center=np.zeros(2), scale=np.ones(2)),
        u_std=Standardizer(center=np.zeros(2), scale=np.ones(2)),
        l_std=Standardizer(center=np.zeros(2), scale=np.ones(2)),
    )
    ckpt.save(obs)

    assert isinstance(load_any(mlp), MLPCheckpoint)
    assert isinstance(load_rollout(mlp), MLPCheckpoint)
    assert isinstance(load_any(obs), ObservableCheckpoint)
    with pytest.raises(TypeError, match="observable checkpoint"):
        load_rollout(obs)

    bad = tmp_path / "bad"
    save_checkpoint(bad, meta={"model_type": "unknown"}, arrays={"x": np.zeros(1)})
    with pytest.raises(ValueError, match="unsupported model_type 'unknown'"):
        load_any(bad)


def test_reader_rejects_a_wrong_model_type_for_its_kind(tmp_path: Path) -> None:
    """``load_mlp`` on an observable checkpoint raises instead of misreading the layout."""
    ckpt = ObservableCheckpoint(
        lift=((np.ones((2, 4)), np.zeros(2)),),
        transition=((np.ones((2, 4)), np.zeros(2)),),
        readout=(np.ones((2, 2)), np.zeros(2)),
        activation="relu",
        n_y=1,
        n_u=1,
        horizon=4,
        n_channels=2,
        n_controls=2,
        dt=0.02,
        downsample=1,
        geometry=StftGeometry(n_segment=4, n_hop=2),
        y_std=Standardizer(center=np.zeros(2), scale=np.ones(2)),
        u_std=Standardizer(center=np.zeros(2), scale=np.ones(2)),
        l_std=Standardizer(center=np.zeros(2), scale=np.ones(2)),
    )
    path = tmp_path / "obs"
    ckpt.save(path)
    with pytest.raises(ValueError, match="model_type 'observable', not 'mlp'"):
        load_mlp(path)
