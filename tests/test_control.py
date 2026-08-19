from typing import Literal

import numpy as np
import pytest

from neuro.control.threshold import AmplitudeThresholdController
from neuro.control.waveform import WaveformController, build_input_schedule
from neuro.control.zero import ZeroController

_DT = 0.1


def test_zero_controller_outputs_zero_vector() -> None:
    """ZeroController ignores its inputs and returns an (n_u,) zero control."""
    controller = ZeroController(dt=_DT, n_u=3)
    u, log = controller.update(0.0, ref=np.array([1.0]), x_hat=np.ones(5))
    np.testing.assert_array_equal(np.atleast_1d(u), np.zeros(3))
    np.testing.assert_array_equal(log.u, np.zeros(3))


def test_zero_controller_from_config() -> None:
    """from_config honours dt and n_u."""
    controller = ZeroController.from_config({"dt": _DT, "n_u": 2})
    assert controller.dt == _DT
    assert controller.n_u == 2


def test_waveform_controller_logs_the_applied_control() -> None:
    """The played-back current is logged, so ``controller.u`` lands in the excitation datasets.

    ``load_trajectory`` identifies the predictor against ``controller.u``; a log model with no
    fields is skipped wholesale by the logger, which silently drops the input channel.
    """
    schedule = np.array([[1.0, -1.0], [2.0, -2.0]])
    controller = WaveformController(dt=_DT, schedule=schedule)

    for k in range(schedule.shape[0]):
        u, log = controller.update(k * _DT, ref=np.zeros(1), x_hat=np.zeros(1))
        np.testing.assert_array_equal(np.atleast_1d(u), schedule[k])
        np.testing.assert_array_equal(log.u, schedule[k])

    # Past the end of the schedule the controller emits -- and logs -- zeros.
    u, log = controller.update(schedule.shape[0] * _DT, ref=np.zeros(1), x_hat=np.zeros(1))
    np.testing.assert_array_equal(np.atleast_1d(u), np.zeros(2))
    np.testing.assert_array_equal(log.u, np.zeros(2))


@pytest.mark.parametrize("input_type", ["ras", "prbs", "multisine"])
@pytest.mark.parametrize("n_controls", [2, 3])
def test_input_schedule_obeys_kirchhoff_current_law(
    input_type: Literal["ras", "prbs", "multisine"], n_controls: int
) -> None:
    """Every active row of the excitation schedule sums to zero across electrodes (Kirchhoff).

    The leading transient stays exactly zero; the persistently-exciting body must inject no net
    current, so each per-step current vector balances to ~0 regardless of ``input_type``. The
    per-row rescale must also keep every current within ``|u| <= amp`` -- for ``n_controls > 2``
    (e.g. a cathode pair plus a shared return anode) the raw zero-sum projection can overshoot.
    """
    amp = 3.0
    n_steps, transient_steps = 1100, 100
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
    np.testing.assert_array_equal(schedule[:transient_steps], 0.0)
    np.testing.assert_allclose(schedule[transient_steps:].sum(axis=1), 0.0, atol=1e-12)
    assert np.all(np.abs(schedule) <= amp + 1e-9)
    assert np.any(np.abs(schedule[transient_steps:]) > 1e-6)


def test_mixed_hold_schedule_spans_the_requested_block_lengths() -> None:
    """A sequence ``hold_ms`` draws each block's length from it, so the excitation is broadband.

    Run-length encode the schedule and check both that no run is shorter than the shortest
    requested hold and that the longest hold is actually used -- a single short hold leaves the
    low-frequency band the MPC commands in unexcited (see the ``build_input_schedule`` docstring).
    """
    dt, holds = 1e-4, [10.0, 50.0, 200.0]
    schedule = build_input_schedule(
        input_type="ras",
        n_steps=200_000,
        transient_steps=0,
        n_controls=3,
        amp=3.0,
        hold_ms=holds,
        dt=dt,
        rng=np.random.default_rng(0),
    )

    changes = np.flatnonzero(np.any(np.diff(schedule, axis=0) != 0.0, axis=1)) + 1
    runs = np.diff(np.concatenate([[0], changes, [len(schedule)]]))
    expected = [round(h / (dt * 1000.0)) for h in holds]
    assert runs[:-1].min() >= min(expected)
    assert set(runs[:-1].tolist()) <= set(expected)
    assert max(expected) in runs.tolist()


def test_mixed_hold_schedule_splits_time_evenly_across_holds() -> None:
    """Each entry in ``hold_ms`` occupies a roughly equal share of the schedule's *time*.

    Drawn uniformly over the values, a hold's share of time is proportional to its own length, so
    on ``[75, 200, 2000]`` the 2000 ms entry would take ~88 % of every trajectory and leave almost
    no short-hold data. The inverse-length weighting is what lets the grid span 75-2000 ms at all.
    """
    dt, holds = 1e-4, [75.0, 200.0, 2000.0]
    schedule = build_input_schedule(
        input_type="ras",
        n_steps=4_000_000,
        transient_steps=0,
        n_controls=3,
        amp=3.0,
        hold_ms=holds,
        dt=dt,
        rng=np.random.default_rng(0),
    )

    changes = np.flatnonzero(np.any(np.diff(schedule, axis=0) != 0.0, axis=1)) + 1
    runs = np.diff(np.concatenate([[0], changes, [len(schedule)]]))[:-1]
    expected = [round(h / (dt * 1000.0)) for h in holds]
    share = np.array([runs[runs == length].sum() for length in expected], dtype=np.float64)
    np.testing.assert_allclose(share / share.sum(), 1.0 / len(expected), atol=0.05)


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
    quiet = np.tile([0.4, -0.4], 20)
    loud = np.tile([4.0, -4.0], 20)

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
