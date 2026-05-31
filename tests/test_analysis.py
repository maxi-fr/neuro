import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from neuro.plotting import plot_dashboard, plot_fourier, plot_signals
from neuro.processing import compute_fft, compute_psd


def test_compute_fft_sine() -> None:
    # Generate a pure 10 Hz sine wave
    fs = 1000.0  # 1000 Hz sampling rate
    dt_ms = 1000.0 / fs
    duration_sec = 2.0
    t = np.arange(0, duration_sec, 1.0 / fs)
    freq = 10.0
    amplitude = 1.5
    signal = amplitude * np.sin(2 * np.pi * freq * t)
    signals = np.expand_dims(signal, axis=0)

    freqs, fft_amplitudes = compute_fft(signals, dt_ms)

    # Check output shape
    assert freqs.ndim == 1
    assert fft_amplitudes.shape == (1, len(freqs))

    # Check that peak frequency is at 10 Hz
    peak_idx = np.argmax(fft_amplitudes[0])
    assert np.isclose(freqs[peak_idx], freq, atol=0.5)

    # Check peak amplitude is close to the sine wave's amplitude
    assert np.isclose(fft_amplitudes[0, peak_idx], amplitude, rtol=0.1)


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
        compute_fft(np.zeros(10), 1.0)

    with pytest.raises(ValueError, match="2-D array"):
        compute_psd(np.zeros(10), 1.0)


def test_empty_signals() -> None:
    freqs, fft_val = compute_fft(np.empty((2, 0)), 1.0)
    assert freqs.size == 0
    assert fft_val.shape == (2, 0)

    freqs, psd_val = compute_psd(np.empty((2, 0)), 1.0)
    assert freqs.size == 0
    assert psd_val.shape == (2, 0)


def test_plotting_signals() -> None:
    rng = np.random.default_rng(0)
    signals = rng.standard_normal((3, 100))
    fig, ax = plot_signals(signals, dt_ms=1.0, stacked=True, title="Test Stacked")
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)

    fig, ax = plot_signals(signals, dt_ms=1.0, stacked=False, title="Test Overlaid")
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_plotting_fourier() -> None:
    rng = np.random.default_rng(0)
    signals = rng.standard_normal((3, 100))
    fig, ax = plot_fourier(signals, dt_ms=1.0, mode="amplitude", plot_mean=False)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)

    fig, ax = plot_fourier(signals, dt_ms=1.0, mode="power", plot_mean=True)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)

    with pytest.raises(ValueError, match="Unknown mode"):
        plot_fourier(signals, dt_ms=1.0, mode="invalid")


def test_plot_dashboard() -> None:
    rng = np.random.default_rng(0)
    activity = rng.standard_normal((5, 100))
    eeg = rng.standard_normal((8, 100))
    fig = plot_dashboard(activity, eeg, dt_ms=1.0)
    assert isinstance(fig, Figure)
    plt.close(fig)
