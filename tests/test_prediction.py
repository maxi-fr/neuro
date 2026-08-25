from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from neuro.predictor.data import build_dataset_for_trajectory, load_trajectory, prepare_datasets
from neuro.predictor.module import AutoregressiveMLP
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from pathlib import Path

    from neuro.types import FloatArray

_SEED = 7


def _write_trajectory(path: Path, n_steps: int, n_eeg: int, n_controls: int) -> str:
    """Write a synthetic ``.npz`` trajectory (``sensor_0.y_mea``/``controller.u``) and return its path."""
    rng = np.random.default_rng(_SEED + hash(str(path)) % 1000)

    y = rng.standard_normal((n_steps, n_eeg)) + np.arange(1.0, n_eeg + 1.0)
    u = rng.standard_normal((n_steps, n_controls))
    np.savez(path, **{"sensor_0.y_mea": y, "controller.u": u})  # ty:ignore[invalid-argument-type]
    return str(path)


def _standardizer(rng: np.random.Generator, dim: int) -> Standardizer:
    """A random (non-trivial) standardizer of channel dimension ``dim``."""
    return Standardizer(center=rng.standard_normal(dim), scale=rng.uniform(0.5, 2.0, dim))


def test_prepare_datasets_builds_standardized_windows(tmp_path: Path) -> None:
    """``prepare_datasets`` returns standardized windows matching manual normalization."""
    n_eeg, n_controls, n_steps = 6, 2, 250
    n_y, n_u, horizon = 4, 3, 5
    files = [_write_trajectory(tmp_path / f"traj_{i}.npz", n_steps, n_eeg, n_controls) for i in range(2)]

    data = prepare_datasets(files, n_steps, 1, n_y, n_u, horizon, 1e-4, 0.5, scaler="standard", global_scaling=False)

    assert data.n_channels == n_eeg
    assert data.n_controls == n_controls
    assert data.X_train.shape[1] == n_y * n_eeg + n_u * n_controls + horizon * n_controls
    assert len(data.val_trajs) == 1

    u, y = load_trajectory(files[0], n_steps, 1, 1e-4)
    x_manual, y_manual = build_dataset_for_trajectory(
        data.u_std.transform(u), data.y_std.transform(y), n_y, n_u, horizon
    )
    np.testing.assert_allclose(data.X_train, x_manual, atol=1e-12)
    np.testing.assert_allclose(data.Y_train, y_manual, atol=1e-12)


def test_prepare_datasets_supports_optional_n_steps(tmp_path: Path) -> None:
    """``load_trajectory`` and ``prepare_datasets`` read the complete trajectory when ``n_steps`` is None."""
    n_eeg, n_controls, n_steps = 6, 2, 250
    n_y, n_u, horizon = 4, 3, 5
    files = [_write_trajectory(tmp_path / f"traj_{i}.npz", n_steps, n_eeg, n_controls) for i in range(2)]

    u, y = load_trajectory(files[0], None, 1, 1e-4)
    assert u.shape == (n_steps, n_controls)
    assert y.shape == (n_steps, n_eeg)

    data = prepare_datasets(files, None, 1, n_y, n_u, horizon, 1e-4, 0.5, scaler="standard", global_scaling=False)
    assert data.n_channels == n_eeg
    assert data.X_train.shape[0] == n_steps - horizon - max(n_y - 1, n_u)


def test_prepare_datasets_holds_out_the_validation_trajectories(tmp_path: Path) -> None:
    """The validation windows are exactly the ones built from the held-out trajectories."""
    n_eeg, n_controls, n_steps = 4, 2, 200
    n_y, n_u, horizon = 3, 2, 4
    files = [_write_trajectory(tmp_path / f"traj_{i}.npz", n_steps, n_eeg, n_controls) for i in range(3)]

    data = prepare_datasets(files, n_steps, 1, n_y, n_u, horizon, 1e-4, 0.67, scaler="standard", global_scaling=False)

    assert len(data.val_trajs) == 1
    u_val, y_val = data.val_trajs[0]
    x_manual, y_manual = build_dataset_for_trajectory(
        data.u_std.transform(u_val), data.y_std.transform(y_val), n_y, n_u, horizon
    )
    np.testing.assert_allclose(data.X_val, x_manual, atol=1e-12)
    np.testing.assert_allclose(data.Y_val, y_manual, atol=1e-12)


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    """``AutoregressiveMLP.save``/``load`` persist and restore weights, metadata, and standardizers."""
    n_channels, n_controls = 4, 2
    rng = np.random.default_rng(_SEED)
    y_std = _standardizer(rng, n_channels)
    u_std = _standardizer(rng, n_controls)
    model = AutoregressiveMLP(
        n_y=2,
        n_u=2,
        horizon=3,
        n_channels=n_channels,
        n_controls=n_controls,
        hidden_size=4,
        depth=1,
        activation="relu",
        dt=0.01,
        y_std=y_std,
        u_std=u_std,
    )
    checkpoint = tmp_path / "model"
    model.save(checkpoint)

    loaded = AutoregressiveMLP.load(checkpoint)
    assert loaded.n_channels == n_channels
    assert loaded.n_controls == n_controls
    assert loaded.depth == 1
    np.testing.assert_allclose(loaded.y_std.center, y_std.center)
    np.testing.assert_allclose(loaded.y_std.scale, y_std.scale)
    np.testing.assert_allclose(loaded.u_std.center, u_std.center)
    np.testing.assert_allclose(loaded.u_std.scale, u_std.scale)
    for got, want in zip(
        (m for m in loaded.layers if isinstance(m, torch.nn.Linear)),
        (m for m in model.layers if isinstance(m, torch.nn.Linear)),
        strict=True,
    ):
        np.testing.assert_array_equal(got.weight.detach().numpy(), want.weight.detach().numpy())
        np.testing.assert_array_equal(got.bias.detach().numpy(), want.bias.detach().numpy())


def test_load_trajectory_and_prepare_datasets_with_cutoff_hz(tmp_path: Path) -> None:
    """load_trajectory and prepare_datasets apply custom lowpass cutoff_hz when supplied."""
    n_eeg, n_controls, n_steps = 4, 2, 200
    n_y, n_u, horizon = 3, 2, 4
    files = [_write_trajectory(tmp_path / f"traj_{i}.npz", n_steps, n_eeg, n_controls) for i in range(2)]

    u, y = load_trajectory(files[0], n_steps, 2, 1e-4, cutoff_hz=45.0)
    assert u.shape == (n_steps // 2, n_controls)
    assert y.shape == (n_steps // 2, n_eeg)

    data = prepare_datasets(
        files, n_steps, 2, n_y, n_u, horizon, 1e-4, 0.5, scaler="standard", global_scaling=False, cutoff_hz=45.0
    )
    assert data.n_channels == n_eeg
    x_manual, y_manual = build_dataset_for_trajectory(
        data.u_std.transform(u), data.y_std.transform(y), n_y, n_u, horizon
    )
    np.testing.assert_allclose(data.X_train, x_manual, atol=1e-12)
    np.testing.assert_allclose(data.Y_train, y_manual, atol=1e-12)
