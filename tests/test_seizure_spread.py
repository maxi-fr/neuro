"""Tests for the seizure-propagation measurement used to tune ``K`` and ``sigma``.

Covers the recruitment detector (threshold + persistence), the seizure state ``s(t)`` and its
time mean, the propagation summary, the two ways a recorded run is reduced to a profile (region
LFP and state trajectory), the healthy resting state that the seizure is started from, and the
chunked ``spread_profile`` integration -- which must reproduce a single long
``simulate_network`` call exactly.
"""

from dataclasses import replace

import numpy as np
import pytest

from neuro.connectome import Connectome
from neuro.jansen_rit import JansenRitDynamics, JansenRitParams, lfp, resting_state, simulate_network
from neuro.seizure import (
    A_HEALTHY,
    DT,
    EZ_REGIONS,
    PZ_REGIONS,
    SPREAD_HOP_S,
    SPREAD_WINDOW_S,
    SpreadProfile,
    build_seizure_a_gains,
    spread_profile,
    spread_profile_from_lfp,
    spread_profile_from_states,
    spread_summary,
)
from neuro.types import FloatArray

_K = 0.5357


@pytest.fixture(scope="module")
def connectome() -> Connectome:
    """Load the TVB connectome once for the module, at the canonical coupling."""
    return replace(Connectome.from_config({}), K=_K)


def _times(n_windows: int) -> FloatArray:
    return np.asarray(SPREAD_WINDOW_S / 2.0 + np.arange(n_windows) * SPREAD_HOP_S, dtype=np.float64)


def test_onset_requires_persistence() -> None:
    """A one-window amplitude spike is not a recruitment; a sustained run is."""
    times = _times(10)
    ptp = np.zeros((2, 10))
    ptp[0, 3] = 10.0  # lone burst
    ptp[1, 3:8] = 10.0  # sustained from window 3

    profile = SpreadProfile.from_ptp(times, ptp, threshold=5.0, persist_s=1.0)

    assert np.isnan(profile.onsets[0])
    assert profile.onsets[1] == pytest.approx(times[3])


def test_onset_rejects_an_envelope_shorter_than_the_persistence_window() -> None:
    """Too few windows to establish a sustained run must raise, not score every region NaN."""
    with pytest.raises(ValueError, match="shorter than the"):
        SpreadProfile.from_ptp(_times(3), np.full((2, 3), 10.0), threshold=5.0, persist_s=1.0)


def test_onset_ignores_activity_below_threshold() -> None:
    """Raising the threshold above the envelope leaves every region unrecruited."""
    times = _times(8)
    ptp = np.full((3, 8), 6.0)

    assert np.all(np.isfinite(SpreadProfile.from_ptp(times, ptp, threshold=5.0).onsets))
    assert np.all(np.isnan(SpreadProfile.from_ptp(times, ptp, threshold=7.0).onsets))


def test_n_seizing_and_recruited_by() -> None:
    """The instantaneous count follows the envelope; the recruited count follows the onsets."""
    times = _times(8)
    ptp = np.zeros((3, 8))
    ptp[0, :] = 10.0  # seizing throughout
    ptp[1, 4:] = 10.0  # recruited late

    profile = SpreadProfile.from_ptp(times, ptp, threshold=5.0, persist_s=1.0)

    assert profile.n_seizing().tolist() == [1, 1, 1, 1, 2, 2, 2, 2]
    assert profile.recruited_by(times[0]) == 1
    assert profile.recruited_by(times[-1]) == 2
    assert profile.recruited_by(times[-1], [2]) == 0


def test_burden_is_the_time_mean_seizure_state() -> None:
    """s(t) is the count over the region total, and the burden is its time mean."""
    ptp = np.zeros((4, 8))
    ptp[0] = 20.0  # seizing throughout
    ptp[1, 4:] = 20.0  # recruited halfway

    profile = SpreadProfile.from_ptp(_times(8), ptp, threshold=5.0)

    assert profile.seizure_state().tolist() == [0.25] * 4 + [0.5] * 4
    assert profile.burden() == pytest.approx((8 + 4) / (4 * 8))


def test_burden_separates_runs_the_terminal_count_ties() -> None:
    """Two runs ending at one seizing region; the one suppressed sooner scores lower."""
    early, late = np.zeros((2, 8)), np.zeros((2, 8))
    early[0], late[0] = 20.0, 20.0
    early[1, :2] = 20.0
    late[1, :6] = 20.0

    early_profile = SpreadProfile.from_ptp(_times(8), early, threshold=5.0)
    late_profile = SpreadProfile.from_ptp(_times(8), late, threshold=5.0)

    assert int(early_profile.n_seizing()[-1]) == int(late_profile.n_seizing()[-1]) == 1
    assert early_profile.burden() < late_profile.burden()


def _states(n_samples: int, ptp_mv: FloatArray) -> FloatArray:
    """A ``(n_samples, 6, n_regions)`` trajectory whose LFP ``x[1] - x[2]`` has the given amplitudes."""
    wave = np.sign(np.sin(np.linspace(0.0, 8.0 * np.pi, n_samples)))
    x_traj = np.zeros((n_samples, 6, len(ptp_mv)))
    x_traj[:, 1, :] = 0.5 * ptp_mv[None, :] * wave[:, None]
    return x_traj


def test_spread_profile_from_states_matches_the_lfp_it_reduces_to() -> None:
    """Reducing a state trajectory must agree with profiling the region LFP taken from it."""
    ptp_mv = np.array([20.0, 8.0, 1.0, 0.0])
    x_traj = _states(200, ptp_mv)

    profile = spread_profile_from_states(x_traj, dt=0.01, threshold=5.0)
    direct = spread_profile_from_lfp(lfp(np.moveaxis(x_traj, 0, -1)), dt=0.01, threshold=5.0)

    assert profile.ptp[:, -1] == pytest.approx(ptp_mv)
    assert profile.ptp == pytest.approx(direct.ptp)
    assert int(profile.n_seizing()[-1]) == 2


def test_spread_profile_from_states_rejects_a_run_shorter_than_the_window() -> None:
    """A run too short to fill one spread window raises rather than scoring nothing."""
    with pytest.raises(ValueError, match="shorter than the"):
        spread_profile_from_states(_states(50, np.array([20.0])), dt=0.01, threshold=5.0)


def test_summary_orders_ez_pz_and_hemispheres(connectome: Connectome) -> None:
    """The summary reads the schedule off the onsets: EZ first, then PZ, then the hemisphere."""
    times = _times(60)
    ptp = np.zeros((len(connectome.region_labels), 60))
    left = np.flatnonzero(~connectome.hemispheres)
    ez = [connectome.region_index[name] for name in EZ_REGIONS]
    pz = [connectome.region_index[name] for name in PZ_REGIONS]

    ptp[ez, 2:] = 10.0
    ptp[pz, 20:] = 10.0
    ptp[left, 40:] = 10.0  # the rest of the left hemisphere comes last

    summary = spread_summary(SpreadProfile.from_ptp(times, ptp, threshold=5.0), connectome)

    assert summary.t_ez == pytest.approx(times[2])
    assert summary.t_pz == pytest.approx(times[20])
    assert summary.t_left_half == pytest.approx(times[40])
    assert summary.frac_left == pytest.approx(1.0)
    assert summary.frac_right == pytest.approx(0.0)


def test_summary_marks_unrecruited_regions(connectome: Connectome) -> None:
    """A PZ that never seizes reads as NaN, and scores as if it were recruited at the end."""
    times = _times(40)
    ptp = np.zeros((len(connectome.region_labels), 40))
    ptp[[connectome.region_index[name] for name in EZ_REGIONS], 2:] = 10.0

    summary = spread_summary(SpreadProfile.from_ptp(times, ptp, threshold=5.0), connectome)

    assert np.isnan(summary.t_pz)
    assert np.isnan(summary.t_left_half)
    assert summary.frac_right == pytest.approx(0.0)
    # Nothing spreads, so the score is dominated by the late-PZ and no-extent penalties.
    assert summary.score(duration=20.0) > 3.0


def test_resting_state_is_a_fixed_point(connectome: Connectome) -> None:
    """The healthy network started at ``resting_state`` does not move (noiselessly)."""
    rest = resting_state(connectome, DT)
    dyn = JansenRitDynamics(dt=DT, params=JansenRitParams(A=A_HEALTHY, sigma=0.0), conn=connectome, initial_state=rest)
    _, x_traj = simulate_network(dyn=dyn, duration=0.5)

    assert np.abs(x_traj[:, :, -1] - rest).max() < 1e-9
    # ... and it is the physiological background, not the origin the plant defaults to.
    assert np.all(lfp(rest) > 1.0)
    assert np.all(lfp(rest) < 2.0)


def test_spread_profile_matches_one_long_run(connectome: Connectome) -> None:
    """Chunked integration reproduces a single ``simulate_network`` call step for step.

    ``spread_profile`` restarts ``simulate_network`` once per hop to bound memory, which only
    works because the plant is handed the absolute time it left off at; a chunk that restarts
    the clock at 0 silently holds its state instead of stepping.
    """
    a_gains = build_seizure_a_gains(connectome)

    def plant() -> JansenRitDynamics:
        return JansenRitDynamics(dt=DT, params=JansenRitParams(A=a_gains, sigma=500.0), conn=connectome, seed=69)

    duration = 3.0
    profile = spread_profile(plant(), duration)
    _, x_traj = simulate_network(dyn=plant(), duration=duration)
    y = lfp(x_traj)

    n_win, n_hop = round(SPREAD_WINDOW_S / DT), round(SPREAD_HOP_S / DT)
    expected = np.stack(
        [np.ptp(y[:, s + 1 : s + n_win + 1], axis=1) for s in range(0, y.shape[1] - n_win, n_hop)], axis=1
    )

    assert profile.ptp.shape == expected.shape
    assert np.abs(profile.ptp - expected).max() < 1e-9
