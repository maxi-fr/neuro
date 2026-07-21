"""Tests for the project control components.

Covers :class:`~neuro.control.ZeroController`,
:class:`~neuro.control.StimWindowController`, and :func:`~neuro.control.build_input_schedule`.
"""

import numpy as np
import pytest

from neuro.control import StimWindowController, ZeroController, build_input_schedule

_DT = 0.1


def test_zero_controller_outputs_zero_vector() -> None:
    """ZeroController ignores its inputs and returns an (n_u,) zero control."""
    controller = ZeroController(dt=_DT, n_u=3)
    u, _ = controller.update(0.0, ref=np.array([1.0]), x_hat=np.ones(5))
    np.testing.assert_array_equal(np.atleast_1d(u), np.zeros(3))


def test_zero_controller_from_config() -> None:
    """from_config honours dt and n_u."""
    controller = ZeroController.from_config({"dt": _DT, "n_u": 2})
    assert controller.dt == _DT
    assert controller.n_u == 2


def test_stim_window_controller_holds_amplitude_inside_window() -> None:
    """The fixed amplitude is emitted for onset <= t < offset, zero elsewhere."""
    controller = StimWindowController(dt=_DT, onset=1.0, offset=2.0, amplitude=1.5)

    u_before, log_before = controller.update(0.5, ref=np.array([0.0]), x_hat=np.array([0.0]))
    u_onset, log_onset = controller.update(1.0, ref=np.array([0.0]), x_hat=np.array([0.0]))
    u_inside, _ = controller.update(1.5, ref=np.array([0.0]), x_hat=np.array([0.0]))
    u_offset, log_offset = controller.update(2.0, ref=np.array([0.0]), x_hat=np.array([0.0]))

    assert u_before == 0.0
    assert not log_before.active
    assert u_onset == 1.5  # window is closed on the left
    assert log_onset.active
    assert u_inside == 1.5
    assert u_offset == 0.0  # window is open on the right
    assert not log_offset.active


def test_stim_window_controller_per_electrode_amplitude() -> None:
    """A per-electrode amplitude vector is emitted as an (n_u,) control inside the window."""
    controller = StimWindowController(dt=_DT, onset=0.0, offset=1.0, amplitude=[0.5, -0.5], n_u=2)
    u_inside, _ = controller.update(0.5, ref=np.array([0.0]), x_hat=np.array([0.0]))
    np.testing.assert_array_equal(u_inside, np.array([0.5, -0.5]))
    u_outside, _ = controller.update(1.5, ref=np.array([0.0]), x_hat=np.array([0.0]))
    np.testing.assert_array_equal(u_outside, np.zeros(2))


def test_stim_window_controller_rejects_mismatched_amplitude() -> None:
    """An amplitude length that is neither 1 nor n_u is rejected."""
    with pytest.raises(ValueError, match="amplitude has 3 entries but n_u is 2"):
        StimWindowController(dt=_DT, onset=0.0, offset=1.0, amplitude=[1.0, 2.0, 3.0], n_u=2)


@pytest.mark.parametrize("input_type", ["ras", "prbs", "multisine"])
@pytest.mark.parametrize("n_controls", [2, 3])
def test_input_schedule_obeys_kirchhoff_current_law(input_type: str, n_controls: int) -> None:
    """Every active row of the excitation schedule sums to zero across electrodes (Kirchhoff).

    The leading transient stays exactly zero; the persistently-exciting body must inject no net
    current, so each per-step current vector balances to ~0 regardless of ``input_type``. The
    per-row rescale must also keep every current within ``|u| <= amp`` -- for ``n_controls > 2``
    (e.g. a cathode pair plus a shared return anode) the raw zero-sum projection can overshoot.
    """
    amp = 3.0
    n_steps, transient_steps = 1100, 100  # >= 10 blocks so prbs reliably excites after the zero-sum projection
    schedule = build_input_schedule(
        input_type=input_type,
        n_steps=n_steps,
        transient_steps=transient_steps,
        n_controls=n_controls,
        amp=amp,
        hold_ms=10.0,
        dt=1e-4,
        rng=np.random.default_rng(0),
    )

    assert schedule.shape == (n_steps, n_controls)
    np.testing.assert_array_equal(schedule[:transient_steps], 0.0)  # transient is zeroed
    np.testing.assert_allclose(schedule[transient_steps:].sum(axis=1), 0.0, atol=1e-12)  # KCL
    assert np.all(np.abs(schedule) <= amp + 1e-9)  # amplitude bound respected after rescale
    assert np.any(np.abs(schedule[transient_steps:]) > 1e-6)  # ... and it actually excites


def test_stim_window_controller_from_config() -> None:
    """from_config honours dt, window bounds, amplitude and n_u."""
    controller = StimWindowController.from_config(
        {"dt": _DT, "onset": 10.0, "offset": 30.0, "amplitude": 1.5, "n_u": 1},
    )
    assert controller.dt == _DT
    assert controller.onset == 10.0
    assert controller.offset == 30.0
    assert controller.n_u == 1
    np.testing.assert_array_equal(controller.amplitude, np.array([[1.5]]))
