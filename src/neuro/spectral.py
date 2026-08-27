from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import scipy.signal as sps
from scipy.signal.windows import bartlett, hann

if TYPE_CHECKING:
    from pathlib import Path

    from neuro.config import StftGeometry
    from neuro.types import FloatArray

LOG_FLOOR = 1e-8
"""Floor added to measured power before the log; the envelope itself is strictly positive."""


def compute_periodograms(y: FloatArray, *, fs: float, window: int, hop: int) -> FloatArray:
    """Periodograms of every length-``window`` slice of ``y`` ``(n_samples, n_channels)``, hopped by ``hop``.

    One-sided, density-scaled, periodic Hann, no per-segment detrend -- the geometry
    :class:`neuro.control.costs.SpectralHingeCost` replicates with ``jnp.fft``, and the
    same convention the training loss uses, so that a segment-length change is not also a change
    of implicit high-pass.
    Returns ``(n_windows, n_channels, window // 2 + 1)`` including the DC bin (the MPC cost drops it
    at the use site); segments are never averaged over, so a hot slice cannot be cancelled by a cold
    one.
    """
    n_samples, n_channels = y.shape
    n_bins = window // 2 + 1
    if n_samples < window:
        return np.empty((0, n_channels, n_bins), dtype=np.float64)

    n_windows = (n_samples - window) // hop + 1
    w_hann = hann(window, sym=False)
    segments = np.stack([y[m * hop : m * hop + window, :] for m in range(n_windows)], axis=0)
    _, power = sps.periodogram(segments, fs=fs, window=w_hann, detrend=False, axis=1, scaling="density")
    return np.asarray(power, dtype=np.float64).transpose(0, 2, 1)


def _frame_kernel_weights(kernel: str, width: int) -> FloatArray:
    """Normalised non-negative smoothing weights along the Frame axis."""
    if width == 1:
        return np.ones(1, dtype=np.float64)
    if kernel == "boxcar":
        weights = np.ones(width, dtype=np.float64)
    elif kernel == "triangular":
        weights = bartlett(width + 2, sym=True)[1:-1]
    else:
        weights = hann(width + 2, sym=True)[1:-1]
    return np.asarray(weights / np.sum(weights), dtype=np.float64)


def compute_log_power_frames(y: FloatArray, geometry: StftGeometry, *, fs: float) -> FloatArray:
    """Reduce raw EEG trajectory to log-power Frames at the given Observable geometry.

    Parameters
    ----------
    y
        Raw EEG array of shape ``(n_samples, n_channels)``.
    geometry
        Observable STFT geometry defining Segment length, hop, band, pooling, and Frame Kernel.
    fs
        Sampling rate in Hz.

    Returns
    -------
    FloatArray
        Log-power Frames of shape ``(n_frames, n_channels, n_values)``.
    """
    n_samples, n_channels = y.shape
    n_values = geometry.n_values(fs)
    sample_support = geometry.sample_support_steps(fs)
    if n_samples < sample_support:
        return np.empty((0, n_channels, n_values), dtype=np.float64)

    n_raw_frames = (n_samples - geometry.n_segment) // geometry.n_hop + 1
    w_hann = hann(geometry.n_segment, sym=False)
    segments = np.stack(
        [y[m * geometry.n_hop : m * geometry.n_hop + geometry.n_segment, :] for m in range(n_raw_frames)],
        axis=0,
    )

    _, power = sps.periodogram(segments, fs=fs, window=w_hann, detrend=False, axis=1, scaling="density")
    power = np.asarray(power, dtype=np.float64).transpose(0, 2, 1)

    bin_lo, bin_hi = geometry.bin_range(fs)
    power = power[:, :, bin_lo:bin_hi]

    if geometry.n_bin_pool > 1:
        n_groups = power.shape[-1] // geometry.n_bin_pool
        power = (
            power[:, :, : n_groups * geometry.n_bin_pool]
            .reshape(power.shape[0], power.shape[1], n_groups, geometry.n_bin_pool)
            .mean(axis=-1)
        )

    weights = _frame_kernel_weights(geometry.kernel, geometry.kernel_width)
    n_frames = n_raw_frames - geometry.kernel_width + 1
    if geometry.kernel_width > 1:
        power = np.stack(
            [np.sum(power[i : i + geometry.kernel_width] * weights[:, None, None], axis=0) for i in range(n_frames)],
            axis=0,
        )

    return np.asarray(np.log(power + LOG_FLOOR), dtype=np.float64)


def hinge_penalty(power: FloatArray, reference: FloatArray) -> float:
    """Mean squared one-sided log excess of ``power`` over ``reference``; exactly 0 when everywhere under."""
    excess = np.log(power + LOG_FLOOR) - np.log(reference)
    return float(np.mean(np.maximum(0.0, excess) ** 2))


def windowed_mean_square(y: FloatArray, *, window: int, hop: int) -> FloatArray:
    """Mean-square power of every length-``window`` slice of ``y`` ``(n_samples, n_channels)``, hopped by ``hop``.

    The time-domain twin of :func:`compute_periodograms` on the same segment grid: what the
    ``eeg_ms`` Observable reduces a Segment to. Returns ``(n_windows, n_channels)``.
    """
    n_samples, n_channels = y.shape
    if n_samples < window:
        return np.empty((0, n_channels), dtype=np.float64)
    n_windows = (n_samples - window) // hop + 1
    return np.stack([(y[m * hop : m * hop + window, :] ** 2).mean(axis=0) for m in range(n_windows)], axis=0)


@dataclasses.dataclass(frozen=True)
class MsEnvelope:
    """Healthy per-channel mean-square power envelope plus the window geometry it was measured with.

    Stored alongside :class:`PsdEnvelope` in the same npz. It is measured in the time domain rather
    than derived from the PSD envelope by Parseval: the spectral cost leaves DC unscored while a
    time-domain mean square includes the offset, so the two are not the same quantity.
    """

    power: FloatArray
    fs: float
    window: int
    hop: int

    @classmethod
    def load(cls, path: str | Path) -> MsEnvelope:
        """Read the mean-square envelope written by ``scripts/build_healthy_psd.py``."""
        with np.load(path) as data:
            if "Pref_ms" not in data:
                msg = f"envelope at {path} carries no 'Pref_ms' array; rebuild it with scripts/build_healthy_psd.py."
                raise ValueError(msg)
            return cls(
                power=np.asarray(data["Pref_ms"], dtype=np.float64),
                fs=float(data["fs"]),
                window=int(data["L"]),
                hop=int(data["R"]),
            )


@dataclasses.dataclass(frozen=True)
class PsdEnvelope:
    """Healthy per-``(channel, bin)`` power envelope plus the window geometry it was measured with."""

    power: FloatArray
    fs: float
    window: int
    hop: int

    @classmethod
    def load(cls, path: str | Path) -> PsdEnvelope:
        """Read an envelope written by ``scripts/build_healthy_psd.py``."""
        with np.load(path) as data:
            envelope = cls(
                power=np.asarray(data["Pref"], dtype=np.float64),
                fs=float(data["fs"]),
                window=int(data["L"]),
                hop=int(data["R"]),
            )
        expected_bins = envelope.window // 2 + 1
        if envelope.power.shape[1] != expected_bins:
            msg = (
                f"envelope at {path} has {envelope.power.shape[1]} bins but its window "
                f"({envelope.window}) implies {expected_bins}; the cost must not subset bins."
            )
            raise ValueError(msg)
        return envelope


@dataclasses.dataclass(frozen=True)
class ObservableEnvelope:
    """Healthy Observable log-power Frame envelope plus the geometry it was measured with."""

    power: FloatArray
    fs: float
    geometry: StftGeometry

    @classmethod
    def load(cls, path: str | Path) -> ObservableEnvelope:
        """Read an Observable envelope written by ``scripts/build_healthy_psd.py``."""
        from neuro.config import StftGeometry  # noqa: PLC0415 -- deferred to prevent circular import with neuro.config

        with np.load(path) as data:
            if "Pref_frames" not in data:
                msg = (
                    f"envelope at {path} carries no Observable frames array; "
                    "rebuild it with scripts/build_healthy_psd.py."
                )
                raise ValueError(msg)
            power = np.asarray(data["Pref_frames"], dtype=np.float64)
            fs = float(data["fs"])

            # The writer stores an absent band as a sentinel, since npz has no None.
            band = np.asarray(data["band_hz"])
            band_hz = None if band[0] < 0 else (float(band[0]), float(band[1]))

            kernel_str = str(data["kernel"])
            if kernel_str not in ("boxcar", "triangular", "hann"):
                msg = f"envelope at {path} has unknown kernel '{kernel_str}'."
                raise ValueError(msg)

            geom = StftGeometry(
                n_segment=int(data["n_segment"]),
                n_hop=int(data["n_hop"]),
                band_hz=band_hz,
                n_bin_pool=int(data["n_bin_pool"]),
                kernel=kernel_str,
                kernel_width=int(data["kernel_width"]),
            )

            expected_values = geom.n_values(fs)
            if power.shape[1] != expected_values:
                msg = (
                    f"envelope at {path} has {power.shape[1]} values per channel but its geometry "
                    f"implies {expected_values}."
                )
                raise ValueError(msg)

            return cls(power=power, fs=fs, geometry=geom)
