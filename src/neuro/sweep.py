"""Open-loop simulation drivers and parameter sweeps for the Jansen-Rit plant.

These helpers run the whole-brain :class:`TVBJansenRitDynamics` plant with no
control (``u = 0``) and summarise its behaviour:

* :func:`run_open_loop` / :func:`run_activity` drive the plant for a fixed
  duration and return the per-node pyramidal activity (and optionally the EEG
  projection).
* :func:`sweep_1d` produces a single-node bifurcation diagram by decoupling the
  network (``coupling_strength = 0``) and sweeping a model parameter.
* :func:`network_state_map` produces a Chouzouris-style 2-D state-space map over
  the background input ``mu`` and the global coupling strength ``sigma``.

The analysis primitives (:func:`~utils.processing.dominant_frequency`,
:func:`~utils.processing.synchronization`, :func:`~utils.processing.steady_window`)
live in :mod:`utils.processing` and are reused here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from neuro.jansen_rit_plant import TVBJansenRitDynamics, TVBJansenRitOutput
from utils.processing import dominant_frequency, steady_window, synchronization

if TYPE_CHECKING:
    from collections.abc import Sequence

FloatArray = npt.NDArray[np.float64]

_DEFAULT_CONNECTOME = "connectivity_76.zip"
_N_STATE_VARS = 6  # TVB JansenRit state variables per node (y0..y5)


def _node_activity(state_vec: FloatArray, n_nodes: int) -> FloatArray:
    """Extract per-node pyramidal activity ``y1 - y2`` from a flat state vector."""
    grid = np.asarray(state_vec, dtype=np.float64).reshape(_N_STATE_VARS, n_nodes)
    return grid[1, :] - grid[2, :]


def _drive(
    dynamics: TVBJansenRitDynamics,
    output: TVBJansenRitOutput | None,
    duration_ms: float,
) -> tuple[FloatArray, FloatArray | None]:
    """Step ``dynamics`` (and optionally ``output``) for ``duration_ms`` at ``u = 0``.

    Returns ``(activity, eeg)`` with ``activity`` shape ``(n_nodes, n_samples)``
    and ``eeg`` shape ``(n_sensors, n_samples)`` (or ``None`` if no output given).
    """
    n_nodes = dynamics.n_nodes
    n_steps = round(duration_ms / dynamics.dt)
    u = np.zeros((dynamics.b.shape[1], 1))
    act_cols: list[FloatArray] = []
    eeg_cols: list[FloatArray] = []
    t = 0.0
    for _ in range(n_steps):
        x_raw, _ = dynamics.evaluate(t, u)
        act_cols.append(_node_activity(np.asarray(x_raw, dtype=np.float64), n_nodes))
        if output is not None:
            y_raw, _ = output.evaluate(t, x_raw, u)
            eeg_cols.append(np.asarray(y_raw, dtype=np.float64).reshape(-1))
        t += dynamics.dt
    activity = np.array(act_cols, dtype=np.float64).T
    eeg = np.array(eeg_cols, dtype=np.float64).T if output is not None else None
    return activity, eeg


def run_activity(  # noqa: PLR0913
    *,
    model_params: dict[str, list[float]] | None = None,
    duration_ms: float,
    dt: float = 0.1,
    nsig: float = 0.0,
    seed: int | None = 0,
    connectome: str = _DEFAULT_CONNECTOME,
    coupling_strength: float = 1.0,
) -> FloatArray:
    """Run an open-loop simulation and return per-node activity ``(n_nodes, n_samples)``.

    Lighter than :func:`run_open_loop`: it skips the EEG lead-field projection,
    which is what the parameter sweeps need.
    """
    dynamics = TVBJansenRitDynamics(
        dt=dt,
        nsig=nsig,
        connectome=connectome,
        seed=seed,
        model_params=model_params,
        coupling_strength=coupling_strength,
    )
    activity, _ = _drive(dynamics, None, duration_ms)
    return activity


def run_open_loop(  # noqa: PLR0913
    *,
    model_params: dict[str, list[float]] | None = None,
    duration_ms: float,
    dt: float = 0.1,
    nsig: float = 0.0,
    seed: int | None = 0,
    connectome: str = _DEFAULT_CONNECTOME,
    coupling_strength: float = 1.0,
) -> tuple[FloatArray, FloatArray]:
    """Run an open-loop simulation and return ``(activity, eeg)``.

    ``activity`` has shape ``(n_nodes, n_samples)`` (per-node ``y1 - y2``) and
    ``eeg`` has shape ``(n_sensors, n_samples)`` (lead-field projection).
    """
    dynamics = TVBJansenRitDynamics(
        dt=dt,
        nsig=nsig,
        connectome=connectome,
        seed=seed,
        model_params=model_params,
        coupling_strength=coupling_strength,
    )
    output = TVBJansenRitOutput(dt=dt, n_nodes=dynamics.n_nodes)
    activity, eeg = _drive(dynamics, output, duration_ms)
    assert eeg is not None  # noqa: S101  # output was provided
    return activity, eeg


def sweep_1d(  # noqa: PLR0913
    param_name: str,
    values: Sequence[float],
    *,
    base_params: dict[str, list[float]] | None = None,
    duration_ms: float = 1500.0,
    transient_ms: float = 600.0,
    dt: float = 0.1,
    connectome: str = _DEFAULT_CONNECTOME,
    fmin: float = 1.0,
    fmax: float = 45.0,
    amplitude_eps: float = 0.1,
) -> dict[str, FloatArray]:
    """Single-node bifurcation sweep of one Jansen-Rit model parameter.

    The network is decoupled (``coupling_strength = 0``) and run deterministically
    (``nsig = 0``), so every node is an isolated column; node 0 is analysed. For
    each value the steady-state activity min/max and dominant frequency are
    recorded.

    Parameters
    ----------
    param_name
        Model parameter to sweep (e.g. ``"mu"`` or ``"A"``).
    values
        Parameter values to sweep over.
    base_params
        Baseline ``model_params`` held fixed across the sweep.
    duration_ms, transient_ms
        Total run length and leading transient to discard, in milliseconds.
    dt
        Integration step in milliseconds.
    connectome
        TVB connectivity file (only node 0 is analysed).
    fmin, fmax
        Band (Hz) for dominant-frequency detection.
    amplitude_eps
        Minimum peak-to-peak activity (mV) for a point to count as oscillating;
        below this the dominant frequency is reported as NaN (fixed point).

    Returns
    -------
    dict
        Arrays keyed ``"values"``, ``"min"``, ``"max"``, ``"freq"``.
    """
    base = dict(base_params or {})
    mins: list[float] = []
    maxs: list[float] = []
    freqs: list[float] = []
    for value in values:
        params = {**base, param_name: [float(value)]}
        activity = run_activity(
            model_params=params,
            duration_ms=duration_ms,
            dt=dt,
            nsig=0.0,
            seed=0,
            connectome=connectome,
            coupling_strength=0.0,
        )
        node0 = steady_window(activity[0:1, :], dt, transient_ms)
        lo = float(node0.min())
        hi = float(node0.max())
        mins.append(lo)
        maxs.append(hi)
        freqs.append(dominant_frequency(node0, dt, fmin=fmin, fmax=fmax) if hi - lo >= amplitude_eps else float("nan"))
    return {
        "values": np.asarray(values, dtype=np.float64),
        "min": np.asarray(mins, dtype=np.float64),
        "max": np.asarray(maxs, dtype=np.float64),
        "freq": np.asarray(freqs, dtype=np.float64),
    }


def network_state_map(  # noqa: PLR0913
    mu_values: Sequence[float],
    sigma_values: Sequence[float],
    *,
    base_params: dict[str, list[float]] | None = None,
    duration_ms: float = 1200.0,
    transient_ms: float = 600.0,
    dt: float = 0.1,
    nsig: float = 0.0,
    connectome: str = _DEFAULT_CONNECTOME,
    fmin: float = 1.0,
    fmax: float = 45.0,
) -> dict[str, FloatArray]:
    """Chouzouris-style 2-D state-space map over background input and coupling.

    For each ``(mu, sigma)`` operating point the full network is simulated and
    summarised by its synchronization, dominant frequency, and median node
    oscillation amplitude.

    Parameters
    ----------
    mu_values
        Background-input values (x-axis).
    sigma_values
        Global coupling strengths (y-axis).
    base_params
        Baseline ``model_params`` (``mu`` is overwritten per column).
    duration_ms, transient_ms
        Total run length and leading transient to discard, in milliseconds.
    dt
        Integration step in milliseconds.
    nsig
        Additive noise amplitude (0 -> deterministic).
    connectome
        TVB connectivity file.
    fmin, fmax
        Band (Hz) for dominant-frequency detection.

    Returns
    -------
    dict
        ``"mu"``, ``"sigma"`` (1-D) and ``"sync"``, ``"freq"``, ``"amplitude"``
        (each shape ``(n_sigma, n_mu)``).
    """
    base = dict(base_params or {})
    n_sigma = len(sigma_values)
    n_mu = len(mu_values)
    sync = np.zeros((n_sigma, n_mu), dtype=np.float64)
    freq = np.zeros((n_sigma, n_mu), dtype=np.float64)
    amplitude = np.zeros((n_sigma, n_mu), dtype=np.float64)
    for i, sigma in enumerate(sigma_values):
        for j, mu in enumerate(mu_values):
            params = {**base, "mu": [float(mu)]}
            activity = run_activity(
                model_params=params,
                duration_ms=duration_ms,
                dt=dt,
                nsig=nsig,
                seed=0,
                connectome=connectome,
                coupling_strength=float(sigma),
            )
            steady = steady_window(activity, dt, transient_ms)
            sync[i, j] = synchronization(steady)
            freq[i, j] = dominant_frequency(steady, dt, fmin=fmin, fmax=fmax)
            amplitude[i, j] = float(np.median(steady.max(axis=1) - steady.min(axis=1)))
    return {
        "mu": np.asarray(mu_values, dtype=np.float64),
        "sigma": np.asarray(sigma_values, dtype=np.float64),
        "sync": sync,
        "freq": freq,
        "amplitude": amplitude,
    }
