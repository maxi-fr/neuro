"""Stage 1 regression harness for the single-node Jansen-Rit core.

Snapshots the bifurcation behaviour every later stage relies on: a quiescent
fixed point at the healthy/PZ gains, a sustained spike-wave limit cycle at the EZ
gain, a noise-driven background near amplitude 2, and numerical stability. Values
follow Yu et al. 2024, Table 1 (``I = 90``); the seizure is a ~3 Hz spike-wave,
which is this epilepsy model's actual rhythm (not 8-12 Hz alpha).

A "deterministic" run is just ``sigma = 0`` (the integrator is always stochastic
Heun); the single node is the degenerate ``connectome=None`` network.
"""

from dataclasses import replace

import numpy as np

from neuro.jansen_rit import (
    JansenRitParams,
    heun_step,
    lfp,
    sigmoid_jit,
    simulate_network,
)
from utils.processing import compute_psd, steady_window

_A_HEALTHY = 3.25
_A_PZ = 3.4
_A_EZ = 3.6
_DURATION = 8.0
_TRANSIENT_MS = 3000.0
_SEED = 7
_DT = 1e-4


def _post_transient_output(a_gain: float, *, deterministic: bool) -> np.ndarray:
    """Run one node at gain ``a_gain`` and return its steady-state lfp ``y``."""
    sigma = 0.0 if deterministic else JansenRitParams().sigma
    params = JansenRitParams(A=a_gain, sigma=sigma)
    _, x = simulate_network(params=params, connectome=None, duration=_DURATION, dt=_DT, seed=_SEED)
    y = lfp(x)  # shape (1, n_samples) for the single node
    return steady_window(y, _DT * 1000.0, _TRANSIENT_MS)[0]


def test_sigmoid_shape_and_bounds() -> None:
    """S(v0) = e0, S is monotonic increasing, and bounded in [0, 2 e0]."""
    params = JansenRitParams()
    assert np.asarray(sigmoid_jit(params.v0, params.e0, params.v0, params.r), dtype=np.float64) == params.e0
    v = np.linspace(-20.0, 30.0, 200)
    s = np.asarray(sigmoid_jit(v, params.e0, params.v0, params.r), dtype=np.float64)
    assert np.all(np.diff(s) > 0)
    assert s.min() >= 0.0
    assert s.max() <= 2.0 * params.e0


def test_healthy_node_is_fixed_point() -> None:
    """Deterministic A = 3.25 settles to a fixed point (no self-oscillation)."""
    y = _post_transient_output(_A_HEALTHY, deterministic=True)
    assert np.ptp(y) < 0.05


def test_pz_node_isolated_is_quiescent() -> None:
    """Deterministic A = 3.4 (PZ) is also a fixed point in isolation.

    The PZ only seizes once the network drives it (Stage 2); on its own it must
    stay quiescent, mirroring the all-healthy control.
    """
    y = _post_transient_output(_A_PZ, deterministic=True)
    assert np.ptp(y) < 0.05


def test_ez_node_is_limit_cycle() -> None:
    """Deterministic A = 3.6 (EZ) is a sustained limit cycle, not a fixed point."""
    y_ez = _post_transient_output(_A_EZ, deterministic=True)
    y_healthy = _post_transient_output(_A_HEALTHY, deterministic=True)
    assert np.ptp(y_ez) > 5.0
    assert np.ptp(y_ez) > 100.0 * np.ptp(y_healthy)


def test_ez_phase_plane_is_an_orbit() -> None:
    """The EZ trajectory spans a real range in both phase-plane axes (closed orbit).

    The phase plane is the lfp ``y = x2 - x3`` against its derivative
    ``x5 - x6``; a fixed point would collapse to a point in both.
    """
    params = replace(JansenRitParams(), A=_A_EZ, sigma=0.0)
    _, x = simulate_network(params=params, connectome=None, duration=_DURATION, dt=_DT, seed=_SEED)
    n_drop = round(_TRANSIENT_MS / (_DT * 1000.0))
    y = (x[1, 0] - x[2, 0])[n_drop:]
    dy = (x[4, 0] - x[5, 0])[n_drop:]
    assert np.ptp(y) > 5.0
    assert np.ptp(dy) > 5.0


def test_ez_seizure_frequency_band() -> None:
    """The EZ limit cycle is a slow spike-wave rhythm (~3 Hz)."""
    y = _post_transient_output(_A_EZ, deterministic=True)
    freqs, pxx = compute_psd(y[np.newaxis, :], _DT * 1000.0, nperseg=8192)
    f_peak = freqs[np.argmax(pxx[0])]
    assert 1.0 < f_peak < 6.0


def test_noisy_background_amplitude_and_separation() -> None:
    """Noisy A = 3.25 background is ~2 and well below the EZ limit cycle.

    The closed-loop amplitude threshold of 5 must sit cleanly between the two.
    """
    y_bg = _post_transient_output(_A_HEALTHY, deterministic=False)
    y_ez = _post_transient_output(_A_EZ, deterministic=False)
    bg_p2p = np.ptp(y_bg)
    ez_p2p = np.ptp(y_ez)
    assert 0.5 < bg_p2p < 5.0
    assert ez_p2p > 8.0
    assert bg_p2p < 5.0 < ez_p2p


def test_all_runs_stay_finite() -> None:
    """No NaN/inf across the healthy, PZ, and EZ regimes, noisy and deterministic."""
    for a_gain in (_A_HEALTHY, _A_PZ, _A_EZ):
        for deterministic in (True, False):
            sigma = 0.0 if deterministic else JansenRitParams().sigma
            params = JansenRitParams(A=a_gain, sigma=sigma)
            _, x = simulate_network(params=params, connectome=None, duration=_DURATION, dt=_DT, seed=_SEED)
            assert np.isfinite(x).all()


def test_heun_reduces_to_deterministic_when_noiseless() -> None:
    """heun_step with sigma = 0 stays finite and still produces the EZ limit cycle.

    Confirms the noiseless integrator is well-behaved (the deterministic limit of
    the stochastic Heun scheme).
    """
    params = JansenRitParams(A=_A_EZ, sigma=0.0)
    dt = _DT
    n_steps = round(_DURATION / dt)
    x = np.zeros(6, dtype=np.float64)
    traj = np.empty((6, n_steps + 1), dtype=np.float64)
    traj[:, 0] = x
    for k in range(n_steps):
        x = heun_step(x, 0.0, params, dt, 0.0)
        traj[:, k + 1] = x
    assert np.isfinite(traj).all()
    y = steady_window((traj[1] - traj[2])[np.newaxis, :], dt * 1000.0, _TRANSIENT_MS)[0]
    assert np.ptp(y) > 5.0
