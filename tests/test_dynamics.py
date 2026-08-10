from __future__ import annotations

import numpy as np
import pytest
from simulate.estimator import IdentityEstimator
from simulate.reference import StepReference
from simulate.sensor import GaussianSensor, full_state_measurement
from simulate.simulation import Simulation

from neuro.connectome import Connectome
from neuro.control import ZeroController
from neuro.eeg import EEGMeasurement, build_eeg_gain
from neuro.jansen_rit import JansenRitDynamics, JansenRitParams, simulate_network
from neuro.stimulation import StimulationModel

_DT = 1e-4
_N_REGIONS_TVB = 76
_N_CHANNELS_TVB = 62
_STATE_DIM = 6


def _toy_connectome(n_nodes: int = 3) -> Connectome:
    """Build a small deterministic connectome for fast, TVB-free tests."""
    rng = np.random.default_rng(0)
    weights = rng.random((n_nodes, n_nodes))
    np.fill_diagonal(weights, 0.0)
    tract_lengths = rng.random((n_nodes, n_nodes)) * 10.0
    speed = 50.0
    labels = np.array([f"r{i}" for i in range(n_nodes)], dtype=np.str_)
    return Connectome(
        K=0.5,
        weights=weights,
        tract_lengths=tract_lengths,
        centres=rng.random((n_nodes, 3)),
        region_labels=labels,
        hemispheres=np.zeros(n_nodes, dtype=np.bool_),
        speed=speed,
        delays=tract_lengths / speed,
        region_index={label: idx for idx, label in enumerate(labels)},
    )


def test_dynamics_matches_simulate_network() -> None:
    """Stepping the component reproduces simulate_network exactly (same seed/params)."""
    conn = _toy_connectome(3)
    params = JansenRitParams(A=3.6, sigma=0.0)
    _t, x_ref = simulate_network(
        dyn=JansenRitDynamics(dt=_DT, params=params, conn=conn, seed=7),
        duration=0.01,
    )
    n_steps = x_ref.shape[2] - 1

    dyn = JansenRitDynamics(dt=_DT, params=params, conn=conn, seed=7)
    for k in range(n_steps):
        out, _ = dyn.evaluate(k * _DT, np.array([0.0]))
        np.testing.assert_allclose(np.asarray(out).reshape(6, 3), x_ref[:, :, k + 1], atol=1e-12)


def test_simulation_matches_simulate_network() -> None:
    """A full orchestrated open-loop Simulation reproduces simulate_network's state trajectory.

    Same plant seed/params/coupling, a zero controller and a noise-free sensor, all at a single
    rate, so the logged ``x`` must equal ``simulate_network``'s ``x_traj`` bit-for-bit.
    """
    conn = _toy_connectome(3)
    params = JansenRitParams(A=3.6, sigma=0.0)
    duration = 0.01
    _t, x_ref = simulate_network(dyn=JansenRitDynamics(dt=_DT, params=params, conn=conn, seed=7), duration=duration)

    sim = Simulation(
        t_end=duration,
        dynamics=JansenRitDynamics(dt=_DT, params=params, conn=conn, seed=7),
        reference=StepReference(dt=_DT, step_value=np.array([0.0])),
        sensors=GaussianSensor(dt=_DT, measurement=full_state_measurement, std_dev=0.0),
        estimator=IdentityEstimator(dt=_DT),
        controller=ZeroController(dt=_DT, n_u=1),
    )
    sim.run()

    assert sim.logger is not None
    orch_x = np.moveaxis(sim.logger.signal("sensor_0", "y_mea"), 0, -1)
    n = x_ref.shape[2]
    assert orch_x.shape[2] >= n
    np.testing.assert_allclose(orch_x[:, :, :n], x_ref, atol=1e-12)


def test_dynamics_from_config_builds_network() -> None:
    """from_config loads the TVB connectome and one step yields a finite (6, N) state."""
    dyn = JansenRitDynamics.from_config(
        {"dt": _DT, "connectome": {"speed": 50.0, "K": 0.5357}, "params": {"A": 3.25}, "seed": 69}
    )
    assert dyn.x.shape == (_STATE_DIM, _N_REGIONS_TVB)
    assert dyn.enforce_zero_sum_current is True

    out, _ = dyn.evaluate(0.0, np.array([0.0]))
    out = np.asarray(out)
    assert out.shape == (_STATE_DIM, _N_REGIONS_TVB)
    assert np.isfinite(out).all()


class _ZeroStim(StimulationModel):
    """Two-electrode montage that drives nothing, so only the KCL check is under test."""

    def __init__(self, n_nodes: int) -> None:
        """Build a zero projection for a 2-electrode montage over ``n_nodes`` regions."""
        self.n_controls = 2
        self.control_labels = np.array(["a", "b"], dtype=np.str_)
        self.n_nodes = n_nodes

    def project(self, u: np.ndarray) -> np.ndarray:
        """Zero drive, whatever the current."""
        del u
        return np.zeros(self.n_nodes)


def _two_electrode_dyn(*, enforce_zero_sum_current: bool = True) -> JansenRitDynamics:
    """Fast synthetic plant with a 2-electrode montage."""
    return JansenRitDynamics(
        dt=_DT,
        params=JansenRitParams(),
        conn=_toy_connectome(3),
        stim=_ZeroStim(3),
        seed=7,
        enforce_zero_sum_current=enforce_zero_sum_current,
    )


def test_dynamics_rejects_nonzero_sum_current() -> None:
    """By default the plant refuses a control that violates Kirchhoff's current law (sum != 0)."""
    dyn = _two_electrode_dyn()
    with pytest.raises(ValueError, match="Kirchhoff"):
        dyn.dynamics(0.0, dyn.x, np.array([2.0, 1.0]))


def test_dynamics_accepts_zero_sum_current() -> None:
    """A balanced (zero-sum) montage current is projected and stepped normally."""
    dyn = _two_electrode_dyn()
    out = dyn.dynamics(0.0, dyn.x, np.array([2.0, -2.0]))
    assert out.shape == dyn.x.shape
    assert np.isfinite(out).all()


def test_dynamics_flag_false_allows_monopolar_current() -> None:
    """With enforce_zero_sum_current=False an unbalanced (monopolar) current is permitted."""
    dyn = _two_electrode_dyn(enforce_zero_sum_current=False)
    out = dyn.dynamics(0.0, dyn.x, np.array([2.0, 1.0]))
    assert np.isfinite(out).all()


def test_from_config_can_disable_zero_sum_check() -> None:
    """The Kirchhoff check is config-toggleable via ``enforce_zero_sum_current``."""
    dyn = JansenRitDynamics.from_config(
        {"dt": _DT, "connectome": {"speed": 50.0, "K": 0.5357}, "enforce_zero_sum_current": False}
    )
    assert dyn.enforce_zero_sum_current is False


def test_eeg_measurement_applies_gain() -> None:
    """EEGMeasurement maps the network state to eeg = gain @ (x2 - x3)."""
    m = EEGMeasurement()
    x_flat = np.arange(6 * m.n_nodes, dtype=np.float64)
    x_grid = x_flat.reshape(6, m.n_nodes)
    expected = m.gain @ (x_grid[1] - x_grid[2])

    eeg = m(0.0, x_flat, np.array([0.0]))
    np.testing.assert_allclose(eeg, expected)
    assert eeg.shape == (_N_CHANNELS_TVB,)


def test_eeg_measurement_full_forward_operator() -> None:
    """The forward operator is the (62, 76) TVB lead field."""
    m = EEGMeasurement()
    assert m.gain.shape == (_N_CHANNELS_TVB, _N_REGIONS_TVB)


def test_eeg_measurement_n_nodes_slices_gain() -> None:
    """n_nodes restricts the operator to the leading regions but keeps all 62 channels."""
    m = EEGMeasurement(n_nodes=2)
    assert m.gain.shape == (_N_CHANNELS_TVB, 2)

    x_flat = np.arange(6 * 2, dtype=np.float64)
    eeg = m(0.0, x_flat, np.array([0.0]))
    assert eeg.shape == (_N_CHANNELS_TVB,)


def test_eeg_measurement_selected_channels() -> None:
    """selected_channels restricts the measurement to a channel subset."""
    m = EEGMeasurement(selected_channels=[1, 3])
    x_flat = np.arange(6 * m.n_nodes, dtype=np.float64)
    x_grid = x_flat.reshape(6, m.n_nodes)
    expected = (m.gain @ (x_grid[1] - x_grid[2]))[[1, 3]]

    eeg = m(0.0, x_flat, np.array([0.0]))
    np.testing.assert_allclose(eeg, expected)
    assert eeg.shape == (2,)


def test_eeg_measurement_selected_channels_by_name() -> None:
    """Channel names resolve to the channel indices."""
    m = EEGMeasurement(selected_channels=["F3", "P3"])
    _, channel_labels = build_eeg_gain()
    channel_index = {label: idx for idx, label in enumerate(channel_labels)}
    assert m.selected_channels is not None
    np.testing.assert_array_equal(
        m.selected_channels,
        [channel_index["F3"], channel_index["P3"]],
    )
