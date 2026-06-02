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


def steady_window(signals: FloatArray, dt_ms: float, transient_ms: float) -> FloatArray:
    """Drop an initial transient from multi-channel signals.

    Parameters
    ----------
    signals
        Input signals, shape (..., n_samples) with time along the last axis.
    dt_ms
        Sampling interval (integration step) in milliseconds.
    transient_ms
        Duration of the leading transient to discard, in milliseconds.

    Returns
    -------
    FloatArray
        The signals with the first ``round(transient_ms / dt_ms)`` samples removed.
    """
    n_drop = round(transient_ms / dt_ms)
    return signals[..., n_drop:]


def dominant_frequency(
    signal: FloatArray,
    dt_ms: float,
    *,
    fmin: float = 1.0,
    fmax: float = 45.0,
) -> float:
    """Estimate the dominant oscillation frequency within a band.

    Uses the full-resolution one-sided FFT (not Welch) because at small ``dt``
    the high sampling rate makes Welch's default segment length too coarse to
    resolve EEG bands. The channel power spectra are summed before peak picking,
    mirroring the network dominant frequency of Chouzouris et al. (Eq. 15).

    Parameters
    ----------
    signal
        Input signal(s), shape (n_samples,) or (n_channels, n_samples).
    dt_ms
        Sampling interval (integration step) in milliseconds.
    fmin, fmax
        Band (Hz) within which the dominant peak is searched (DC excluded).

    Returns
    -------
    float
        Frequency (Hz) of the largest spectral peak in ``[fmin, fmax]``,
        or NaN if the band is empty.
    """
    arr = np.atleast_2d(signal).astype(np.float64)
    freqs, amplitudes = compute_fft(arr, dt_ms)
    if freqs.size == 0:
        return float("nan")
    power = np.square(amplitudes).sum(axis=0)
    band = (freqs >= fmin) & (freqs <= fmax)
    if not band.any():
        return float("nan")
    band_freqs = freqs[band]
    return float(band_freqs[int(np.argmax(power[band]))])


def synchronization(activity: FloatArray) -> float:
    """Mean pairwise correlation across channels (a network synchrony index).

    This is the off-diagonal mean of the Pearson correlation matrix of the node
    activity, an analog of the network cross-correlation ``R`` of Chouzouris
    et al. (Eq. 6): ~1 for fully synchronized nodes, ~0 for unrelated ones.

    Parameters
    ----------
    activity
        Node activity, shape (n_nodes, n_samples).

    Returns
    -------
    float
        Mean of the upper-triangular (off-diagonal) correlation entries, or NaN
        if fewer than two channels are provided.
    """
    act = np.atleast_2d(activity).astype(np.float64)
    if act.shape[0] < 2:  # noqa: PLR2004
        return float("nan")
    corr = np.corrcoef(act)
    iu = np.triu_indices(corr.shape[0], k=1)
    return float(np.nanmean(corr[iu]))
