"""Tests for the open-loop drivers, sweeps, and analysis/plotting helpers.

The pure-NumPy analysis functions (dominant_frequency, synchronization,
steady_window) and the new plotting functions are tested directly. The
TVB-backed drivers (run_open_loop, run_activity, sweep_1d, network_state_map)
are exercised with short durations / tiny grids as smoke tests.
"""

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from neuro.sweep import network_state_map, run_activity, run_open_loop, sweep_1d
from utils.plotting import plot_bifurcation_1d, plot_state_map
from utils.processing import dominant_frequency, steady_window, synchronization

_DT = 0.1
_N_NODES_TVB = 76
_N_SENSORS_TVB = 65


# --- Pure analysis helpers --------------------------------------------------


def test_dominant_frequency_recovers_sine() -> None:
    fs = 1000.0
    dt_ms = 1000.0 / fs
    t = np.arange(0, 2.0, 1.0 / fs)
    signal = np.sin(2 * np.pi * 10.0 * t)[np.newaxis, :]
    assert np.isclose(dominant_frequency(signal, dt_ms), 10.0, atol=0.6)


def test_dominant_frequency_accepts_1d() -> None:
    fs = 1000.0
    t = np.arange(0, 2.0, 1.0 / fs)
    signal = np.sin(2 * np.pi * 8.0 * t)
    assert np.isclose(dominant_frequency(signal, 1000.0 / fs), 8.0, atol=0.6)


def test_synchronization_identical_vs_random() -> None:
    rng = np.random.default_rng(0)
    base = rng.standard_normal((1, 500))
    identical = np.repeat(base, 4, axis=0)
    assert synchronization(identical) > 0.99

    unrelated = rng.standard_normal((4, 500))
    assert abs(synchronization(unrelated)) < 0.3


def test_synchronization_single_channel_is_nan() -> None:
    assert np.isnan(synchronization(np.ones((1, 10))))


def test_steady_window_drops_transient() -> None:
    arr = np.arange(20.0).reshape(2, 10)
    trimmed = steady_window(arr, dt_ms=1.0, transient_ms=3.0)
    assert trimmed.shape == (2, 7)
    assert trimmed[0, 0] == 3.0


# --- Plotting ---------------------------------------------------------------


def test_plot_bifurcation_1d_returns_fig_ax() -> None:
    values = np.linspace(0.0, 0.4, 5)
    fig, ax = plot_bifurcation_1d(
        values,
        min_vals=np.zeros_like(values),
        max_vals=np.linspace(0.0, 5.0, 5),
        freqs=np.array([np.nan, np.nan, 5.0, 7.0, 9.0]),
        param_label="mu",
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_plot_state_map_returns_fig_ax() -> None:
    mu = np.linspace(0.1, 0.4, 4)
    sigma = np.linspace(0.0, 3.0, 3)
    grid = np.random.default_rng(0).random((3, 4))
    fig, ax = plot_state_map(mu, sigma, grid, metric_label="R")
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


# --- TVB-backed drivers (short smoke tests) ---------------------------------


def test_run_open_loop_shapes() -> None:
    activity, eeg = run_open_loop(duration_ms=20.0, nsig=0.0, seed=0)
    n_samples = round(20.0 / _DT)
    assert activity.shape == (_N_NODES_TVB, n_samples)
    assert eeg.shape == (_N_SENSORS_TVB, n_samples)
    assert np.isfinite(activity).all()
    assert np.isfinite(eeg).all()


def test_run_activity_shape() -> None:
    activity = run_activity(duration_ms=20.0, nsig=0.0, seed=0)
    assert activity.shape == (_N_NODES_TVB, round(20.0 / _DT))


def test_decoupled_nodes_share_envelope() -> None:
    """coupling_strength=0 -> every node is the same isolated column (same min/max)."""
    activity = run_activity(duration_ms=800.0, nsig=0.0, seed=0, coupling_strength=0.0)
    steady = steady_window(activity, _DT, 400.0)
    node_ranges = steady.max(axis=1) - steady.min(axis=1)
    # All nodes trace the same limit cycle, so their amplitudes barely differ.
    assert node_ranges.std() < 0.2


def test_sweep_1d_shapes() -> None:
    result = sweep_1d("mu", [0.2, 0.3], duration_ms=400.0, transient_ms=200.0)
    assert set(result) == {"values", "min", "max", "freq"}
    for key in result:
        assert result[key].shape == (2,)
    assert np.all(result["max"] >= result["min"])


def test_network_state_map_shapes() -> None:
    result = network_state_map([0.2], [0.0, 1.0], duration_ms=400.0, transient_ms=200.0)
    assert result["mu"].shape == (1,)
    assert result["sigma"].shape == (2,)
    for key in ("sync", "freq", "amplitude"):
        assert result[key].shape == (2, 1)
