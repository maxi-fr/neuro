"""Signal processing functions for brain activity and EEG signals."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.signal import welch

FloatArray = npt.NDArray[np.float64]


def compute_fft(
    signals: FloatArray,
    dt_ms: float,
) -> tuple[FloatArray, FloatArray]:
    """Compute the one-sided Fast Fourier Transform (FFT) of multi-channel signals.

    Parameters
    ----------
    signals
        Input signals, shape (n_channels, n_samples).
    dt_ms
        Sampling interval (integration step) in milliseconds.

    Returns
    -------
    frequencies
        Frequency bins in Hz, shape (n_frequencies,).
    amplitudes
        FFT amplitudes (magnitude of FFT normalized by n_samples),
        shape (n_channels, n_frequencies).

    Raises
    ------
    ValueError
        If signals is not a 2-D array.
    """
    if signals.ndim != 2:  # noqa: PLR2004
        msg = f"Expected 2-D array of shape (n_channels, n_samples), got shape {signals.shape}"
        raise ValueError(msg)

    n_channels, n_samples = signals.shape
    if n_samples == 0:
        return np.empty(0, dtype=np.float64), np.empty((n_channels, 0), dtype=np.float64)

    fs = 1000.0 / dt_ms  # Sampling frequency in Hz
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    fft_vals = np.fft.rfft(signals, axis=1)

    # Calculate amplitude (magnitude / n_samples)
    amplitudes = np.abs(fft_vals) / n_samples
    if n_samples > 2:  # noqa: PLR2004
        # Multiply non-DC and non-Nyquist components by 2 to conserve energy
        if n_samples % 2 == 0:
            amplitudes[:, 1:-1] *= 2.0
        else:
            amplitudes[:, 1:] *= 2.0

    return freqs.astype(np.float64), amplitudes.astype(np.float64)


def compute_psd(
    signals: FloatArray,
    dt_ms: float,
    nperseg: int | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Compute the Power Spectral Density (PSD) using Welch's method.

    Parameters
    ----------
    signals
        Input signals, shape (n_channels, n_samples).
    dt_ms
        Sampling interval (integration step) in milliseconds.
    nperseg
        Length of each segment for Welch's method. If None, defaults to min(n_samples, 256).

    Returns
    -------
    frequencies
        Frequency bins in Hz, shape (n_frequencies,).
    psd
        Power spectral density, shape (n_channels, n_frequencies).

    Raises
    ------
    ValueError
        If signals is not a 2-D array.
    """
    if signals.ndim != 2:  # noqa: PLR2004
        msg = f"Expected 2-D array of shape (n_channels, n_samples), got shape {signals.shape}"
        raise ValueError(msg)

    n_channels, n_samples = signals.shape
    if n_samples == 0:
        return np.empty(0, dtype=np.float64), np.empty((n_channels, 0), dtype=np.float64)

    fs = 1000.0 / dt_ms
    if nperseg is None:
        nperseg = min(n_samples, 256)

    freqs, pxx = welch(signals, fs=fs, nperseg=nperseg, axis=1)
    return freqs.astype(np.float64), pxx.astype(np.float64)
