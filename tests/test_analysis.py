import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from utils.plotting import plot_psd, plot_signals
from utils.processing import compute_psd, steady_window, synchronization


def test_compute_psd_sine() -> None:
    fs = 1000.0
    dt_ms = 1000.0 / fs
    t = np.arange(0, 1.0, 1.0 / fs)
    signal = np.sin(2 * np.pi * 10.0 * t)
    signals = np.expand_dims(signal, axis=0)

    freqs, psd = compute_psd(signals, dt_ms, nperseg=256)
    assert freqs.ndim == 1
    assert psd.shape == (1, len(freqs))
    peak_idx = np.argmax(psd[0])
    assert np.isclose(freqs[peak_idx], 10.0, atol=2.0)


def test_invalid_signals_shape() -> None:

    with pytest.raises(ValueError, match="2-D array"):
        compute_psd(np.zeros(10), 1.0)


def test_empty_signals() -> None:

    freqs, psd_val = compute_psd(np.empty((2, 0)), 1.0)
    assert freqs.size == 0
    assert psd_val.shape == (2, 0)


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
