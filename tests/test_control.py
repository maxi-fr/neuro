"""Tests for the project control components.

Covers :class:`~neuro.control.ZeroController`, :class:`~neuro.control.StimWindowController`,
:class:`~neuro.control.AmplitudeThresholdController`, and
:func:`~neuro.control.build_input_schedule`.
"""

import numpy as np
import pytest

from neuro.control import (
    AmplitudeThresholdController,
    StimWindowController,
    ZeroController,
    build_input_schedule,
)

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


def _run_threshold_controller(
    controller: AmplitudeThresholdController, signal: np.ndarray, dt: float = _DT
) -> tuple[np.ndarray, list[bool]]:
    """Feed a scalar measurement trace through the controller; return controls and burst flags."""
    controls, active = [], []
    for step, value in enumerate(signal):
        u, log = controller.update(step * dt, ref=np.array([0.0]), x_hat=np.array([value]))
        controls.append(np.atleast_1d(u))
        active.append(log.active)
    return np.array(controls), active


def test_amplitude_threshold_controller_triggers_on_crossing() -> None:
    """Stimulation stays off on a quiet trace and switches on once peak-to-peak clears threshold.

    The measured amplitude is the peak-to-peak over the trailing window, so a burst starts on
    the first step where the window as a whole is large -- not on the first large sample.
    """
    controller = AmplitudeThresholdController(
        dt=_DT, amplitude=[-1.0, 1.0], threshold=5.0, burst_duration=1.0, window=0.5, n_u=2
    )
    quiet = np.tile([0.4, -0.4], 20)  # ptp 0.8, below threshold
    loud = np.tile([4.0, -4.0], 20)  # ptp 8.0, above threshold

    controls, active = _run_threshold_controller(controller, np.concatenate([quiet, loud]))

    assert not any(active[: len(quiet)]), "a sub-threshold trace must not trigger"
    np.testing.assert_array_equal(controls[: len(quiet)], 0.0)
    assert any(active[len(quiet) :]), "a supra-threshold trace must trigger"
    np.testing.assert_array_equal(controls[active], np.tile([-1.0, 1.0], (sum(active), 1)))


def test_amplitude_threshold_controller_holds_burst_to_completion() -> None:
    """A burst runs its full duration even after the signal falls back below threshold.

    Re-checking the threshold every step would turn the controller into a bang-bang signal at
    the update rate; the paper's protocol is one fixed tau-second stimulus per trigger.
    """
    burst, window = 1.0, 0.5
    controller = AmplitudeThresholdController(dt=_DT, amplitude=1.0, threshold=5.0, burst_duration=burst, window=window)
    # Loud long enough to fill the window and trigger, then silent for longer than the burst.
    signal = np.concatenate([np.tile([4.0, -4.0], 5), np.zeros(30)])
    _, active = _run_threshold_controller(controller, signal)

    on = [step for step, is_on in enumerate(active) if is_on]
    assert on == list(range(on[0], on[-1] + 1)), "the burst must be one contiguous block"
    assert abs(len(on) - round(burst / _DT)) <= 1, "the burst must last burst_duration"
    assert on[-1] < len(signal) - 1, "and must stop once it is over"


def test_amplitude_threshold_controller_ignores_partial_window() -> None:
    """A zero-padded start-up buffer must not read as a threshold crossing."""
    controller = AmplitudeThresholdController(dt=_DT, amplitude=1.0, threshold=5.0, burst_duration=1.0, window=1.0)
    _, log = controller.update(0.0, ref=np.array([0.0]), x_hat=np.array([50.0]))
    assert log.amplitude == 0.0
    assert not log.active


def test_amplitude_threshold_controller_from_config() -> None:
    """from_config honours the trigger settings and the per-electrode amplitude."""
    controller = AmplitudeThresholdController.from_config(
        {
            "dt": 0.01,
            "amplitude": [-0.5, -0.5, 1.0],
            "threshold": 2.0,
            "burst_duration": 0.5,
            "window": 1.0,
            "channel": 3,
            "n_u": 3,
        },
    )
    assert controller.dt == 0.01
    assert controller.threshold == 2.0
    assert controller.burst_duration == 0.5
    assert controller.channel == 3
    np.testing.assert_array_equal(controller.amplitude, np.array([-0.5, -0.5, 1.0]))


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
