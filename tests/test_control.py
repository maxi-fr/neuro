"""Tests for the project control/sensing components and orchestrated open-loop runs.

Covers :class:`~neuro.control.ZeroController`, :class:`~neuro.sensing.DirectSensor`,
and an end-to-end open-loop :class:`~simulate.simulation.Simulation` of the
Jansen-Rit plant -- the exact path the stock zero-gain ``PIDController`` /
``GaussianSensor`` combination cannot run, because the orchestrator seeds the loop
with a scalar measurement before the EEG vector is available.
"""

from pathlib import Path

import numpy as np
from simulate.simulation import Simulation

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


def test_orchestrated_jr_open_loop_runs(tmp_path: Path) -> None:
    """A full open-loop JR simulation runs and logs finite EEG with zero control."""
    config = {
        "t_end": 2.0,
        "dynamics": {"class_path": "neuro.jansen_rit_plant.TVBJansenRitDynamics", "dt": _DT, "seed": 0},
        "output": {"class_path": "neuro.jansen_rit_plant.TVBJansenRitOutput", "dt": _DT, "n_nodes": 76},
        "reference": {"class_path": "simulate.reference.StepReference", "dt": _DT, "step_value": 0.0},
        "sensor": {"class_path": "neuro.sensing.DirectSensor", "dt": _DT, "n_sensors": _N_SENSORS_TVB, "std_dev": 0.0},
        "estimator": {"class_path": "simulate.estimator.IdentityEstimator", "dt": _DT},
        "controller": {"class_path": "neuro.control.ZeroController", "dt": _DT, "n_u": 1},
    }
    sim = Simulation.from_config(config)
    sim.run(tmp_path, prefix="t")
    sim.export_results(tmp_path, prefix="t")

    with np.load(tmp_path / "t.npz") as data:
        eeg = data["universal_y"]
        u = data["universal_u"]

    assert eeg.shape[1] == _N_SENSORS_TVB
    assert np.isfinite(eeg).all()
    np.testing.assert_array_equal(u, np.zeros_like(u))
