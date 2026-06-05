"""Tests for the project control/sensing components and orchestrated open-loop runs.

Covers :class:`~neuro.control.ZeroController` and :class:`~neuro.sensing.DirectSensor`
"""

from pathlib import Path

import numpy as np

from neuro.control import ZeroController
from neuro.sensing import DirectSensor

_DT = 0.1
_N_SENSORS_TVB = 65


def test_direct_sensor_broadcasts_scalar_seed() -> None:
    """A scalar seed becomes an n_sensors vector; a vector passes through."""
    sensor = DirectSensor(dt=_DT, n_sensors=4, std_dev=0.0)
    seed_mea, _ = sensor.update(0.0, 0.0)
    np.testing.assert_array_equal(np.atleast_1d(seed_mea), np.zeros(4))
    vec_mea, _ = sensor.update(_DT, np.arange(4, dtype=np.float64))
    np.testing.assert_array_equal(np.atleast_1d(vec_mea), np.arange(4))


def test_zero_controller_outputs_zero_vector() -> None:
    """ZeroController ignores its inputs and returns an (n_u,) zero control."""
    controller = ZeroController(dt=_DT, n_u=3)
    u, _ = controller.update(0.0, ref=1.0, x_hat=np.ones(5))
    np.testing.assert_array_equal(np.atleast_1d(u), np.zeros(3))


def test_zero_controller_from_config() -> None:
    """from_config honours dt and n_u."""
    controller = ZeroController.from_config({"dt": _DT, "n_u": 2})
    assert controller.dt == _DT
    assert controller.n_u == 2
