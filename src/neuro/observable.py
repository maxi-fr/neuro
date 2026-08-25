from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.signal.windows import hann

from neuro.config import EegMsGeometry, ObservableGeometry, StftGeometry
from neuro.spectral import LOG_FLOOR, MsEnvelope, PsdEnvelope

if TYPE_CHECKING:
    from pathlib import Path

    from neuro.types import FloatArray


def _segments(x: FloatArray, size: int, hop: int) -> FloatArray:
    """Hopped windows along the last axis: ``(..., n)`` -> ``(..., n_windows, size)``."""
    return np.lib.stride_tricks.sliding_window_view(x, size, axis=-1)[..., ::hop, :]


def _spectrogram(x: FloatArray, geometry: StftGeometry, fs: float) -> FloatArray:
    """Hopped periodograms of ``(..., n_channels, n_samples)`` -> ``(..., n_channels, n_frames, n_bins)``.

    The NumPy twin of :func:`neuro.predictor.losses.spectrogram` and of
    :func:`neuro.spectral.compute_periodograms`: one-sided, density-scaled, periodic Hann, no
    per-segment detrend.
    """
    n_segment = geometry.n_segment
    window = hann(n_segment, sym=False)
    spectrum = np.fft.rfft(_segments(x, n_segment, geometry.n_hop) * window, n=n_segment, axis=-1)
    psd = np.abs(spectrum) ** 2 / (fs * np.sum(window**2))

    # One-sided: every bin carries its negative-frequency twin, except DC and (even n) Nyquist.
    fold = np.full(psd.shape[-1], 2.0)
    fold[0] = 1.0
    if n_segment % 2 == 0:
        fold[-1] = 1.0
    return psd * fold


def pool_bins(power: FloatArray, n_bin_pool: int) -> FloatArray:
    """Mean power over consecutive groups of ``n_bin_pool`` bins, dropping the trailing remainder."""
    if n_bin_pool == 1:
        return power
    n_groups = power.shape[-1] // n_bin_pool
    grouped = power[..., : n_groups * n_bin_pool].reshape(*power.shape[:-1], n_groups, n_bin_pool)
    return grouped.mean(axis=-1)


def frame_kernel(kernel: str, width: int) -> FloatArray:
    """Normalised non-negative smoothing weights along the frame axis.

    Endpoints are kept strictly positive, so a width-``n`` taper really pools ``n`` frames -- the
    NumPy twin of :func:`neuro.predictor.losses.frame_kernel`.
    """
    if width == 1:
        return np.ones(1)
    if kernel == "boxcar":
        weights = np.ones(width)
    elif kernel == "triangular":
        weights = np.bartlett(width + 2)[1:-1]
    else:
        weights = np.hanning(width + 2)[1:-1]
    return weights / weights.sum()


def smooth_frames(power: FloatArray, weights: FloatArray) -> FloatArray:
    """Convolve power ``(..., n_frames, n_bins)`` along the frame axis, valid support only."""
    if weights.size == 1:
        return power
    windows = _segments(np.moveaxis(power, -2, -1), weights.size, 1)
    return np.moveaxis(windows @ weights, -1, -2)


def log_observable(y: FloatArray, geometry: ObservableGeometry, fs: float) -> FloatArray:
    """Reduce raw EEG ``(..., n_samples, n_channels)`` onto the Frame grid, in log units.

    Returns ``(..., n_frames, n_channels, n_values)``: the quantity the observable Predictor
    forecasts, the training Loss scores and the MPC Cost hinges -- computed by the one geometry
    object all three share.
    """
    x = np.moveaxis(np.asarray(y, dtype=np.float64), -1, -2)
    if isinstance(geometry, StftGeometry):
        bin_lo, bin_hi = geometry.bin_range(fs)
        power = _spectrogram(x, geometry, fs)[..., bin_lo:bin_hi]
        power = pool_bins(power, geometry.n_bin_pool)
        power = smooth_frames(power, frame_kernel(geometry.kernel, geometry.kernel_width))
    elif isinstance(geometry, EegMsGeometry):
        windows = _segments(x, geometry.window_steps(fs), geometry.hop_steps(fs))
        power = (windows**2).mean(axis=-1)[..., None]
    else:  # pragma: no cover -- the two kinds above are the whole of ObservableGeometry
        msg = f"unsupported observable geometry {type(geometry).__name__}"
        raise TypeError(msg)
    return np.log(np.moveaxis(power, -3, -2) + LOG_FLOOR)


def control_means(geometry: ObservableGeometry, horizon: int, fs: float) -> FloatArray:
    """Build the fixed ``(n_frames, horizon)`` operator averaging Control Currents over a Frame's support.

    Unweighted: the currents act by shifting a regional operating point rather than by cancelling
    cycles, so the mean is the first-order summary of a Segment's Stimulation Drive. The Hann taper
    belongs to the spectral estimator, not to the plant's response.
    """
    supports = geometry.frame_supports(horizon, fs)
    operator = np.zeros((len(supports), horizon), dtype=np.float64)
    for m, (start, end) in enumerate(supports):
        operator[m, start:end] = 1.0 / (end - start)
    return operator


def envelope_log_reference(envelope: PsdEnvelope | MsEnvelope, geometry: ObservableGeometry, fs: float) -> FloatArray:
    """Reduce a healthy envelope onto the Frame's value grid: ``(n_channels, n_values)`` in log units.

    Pooling happens on the reference *power*, before the log, exactly as it does on the measured
    power in :func:`log_observable`, so the hinge compares two quantities built the same way.
    """
    if isinstance(geometry, StftGeometry) and isinstance(envelope, PsdEnvelope):
        bin_lo, bin_hi = geometry.bin_range(fs)
        return np.log(pool_bins(envelope.power[:, bin_lo:bin_hi], geometry.n_bin_pool))
    if isinstance(geometry, EegMsGeometry) and isinstance(envelope, MsEnvelope):
        return np.log(envelope.power[:, None])
    msg = f"{type(envelope).__name__} is not the reference envelope for a {geometry.kind} observable."
    raise TypeError(msg)


def geometry_from_meta(meta: dict[str, Any]) -> ObservableGeometry:
    """Rebuild the recorded Observable geometry from a checkpoint's metadata block."""
    fields = dict(meta)
    kind = fields.pop("kind")
    if kind == StftGeometry.kind:
        return StftGeometry.model_validate(fields)
    if kind == EegMsGeometry.kind:
        return EegMsGeometry.model_validate(fields)
    msg = f"unsupported observable kind {kind!r}"
    raise ValueError(msg)


def geometry_meta(geometry: ObservableGeometry) -> dict[str, Any]:
    """Serialize an Observable geometry, tagged with its kind, for a checkpoint's metadata block."""
    return {"kind": geometry.kind, **geometry.model_dump(mode="json")}


def load_envelope(path: str | Path, geometry: ObservableGeometry) -> PsdEnvelope | MsEnvelope:
    """Load the healthy reference envelope of the kind ``geometry`` scores, from the shared npz."""
    if isinstance(geometry, StftGeometry):
        return PsdEnvelope.load(path)
    return MsEnvelope.load(path)


def load_log_reference(path: str | Path, geometry: ObservableGeometry, fs: float) -> FloatArray:
    """Load and reduce the healthy envelope onto the Frame's value grid, in log units."""
    return envelope_log_reference(load_envelope(path, geometry), geometry, fs)
