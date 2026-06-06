"""Signal processing functions for brain activity and EEG signals."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.signal import detrend, welch

FloatArray = npt.NDArray[np.float64]


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


def band_energy(
    signals: FloatArray,
    dt_ms: float,
    *,
    band: tuple[float, float] = (0.0, 50.0),
    nperseg: int | None = None,
    normalize: bool = True,
) -> FloatArray:
    """Per-channel spectral energy integrated over a frequency band.

    Detrends each channel (removing the DC/linear baseline of ``y = x2 - x3``),
    computes the Welch PSD via :func:`compute_psd`, integrates it over ``band``
    (Hz, inclusive) with the trapezoid rule, and optionally normalizes so the
    strongest channel is ``1.0`` (the paper's convention).

    Parameters
    ----------
    signals
        Input signals, shape (n_channels, n_samples).
    dt_ms
        Sampling interval (integration step) in milliseconds.
    band
        Inclusive frequency band ``(low, high)`` in Hz to integrate over.
    nperseg
        Welch segment length; passed through to :func:`compute_psd`.
    normalize
        If True, divide by the maximum so the strongest channel is 1.0.

    Returns
    -------
    FloatArray
        Per-channel energy, shape (n_channels,).

    Raises
    ------
    ValueError
        If signals is not a 2-D array.
    """
    if signals.ndim != 2:  # noqa: PLR2004
        msg = f"Expected 2-D array of shape (n_channels, n_samples), got shape {signals.shape}"
        raise ValueError(msg)

    n_channels = signals.shape[0]
    if signals.shape[1] == 0:
        return np.zeros(n_channels, dtype=np.float64)

    detrended = detrend(signals, axis=1)
    freqs, psd = compute_psd(detrended, dt_ms, nperseg=nperseg)

    low, high = band
    mask = (freqs >= low) & (freqs <= high)
    energy = np.trapezoid(psd[:, mask], freqs[mask], axis=1)

    if normalize and energy.max() > 0:
        energy = energy / energy.max()
    return energy.astype(np.float64)


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
