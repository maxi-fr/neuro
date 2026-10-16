from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from neuro.filtering import antialias_filter, lowpass_filter
from neuro.predictor.data import load_trajectory

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("cutoff_hz", [None, 10.0])
@pytest.mark.parametrize("timestamped_controls", [False, True])
def test_decimated_trajectory_releases_full_resolution_storage(
    tmp_path: Path, cutoff_hz: float | None, *, timestamped_controls: bool
) -> None:
    rng = np.random.default_rng(69)
    y = rng.normal(size=(4096, 3))
    u = rng.normal(size=(4096, 2))
    dt = 0.001
    downsample = 16
    samples = 128
    payload = {"sensor_0.y_mea": y, "controller.u": u}
    if timestamped_controls:
        times = np.arange(len(y)) * dt
        payload.update({"sensor_0.t": times, "controller.t": times})
    path = tmp_path / "trajectory.npz"
    np.savez(path, allow_pickle=False, **payload)

    controls, measurements = load_trajectory(str(path), samples, downsample, dt, cutoff_hz=cutoff_hz)
    filtered = (
        antialias_filter(y[: samples * downsample], 1 / dt, downsample)
        if cutoff_hz is None
        else lowpass_filter(y[: samples * downsample], 1 / dt, cutoff_hz)
    )
    np.testing.assert_array_equal(measurements, filtered[::downsample])
    np.testing.assert_array_equal(controls, u[: samples * downsample : downsample])
    for array in (controls, measurements):
        assert array.flags.owndata, "Decimated data must release its full-resolution backing array"
        assert array.flags.c_contiguous
