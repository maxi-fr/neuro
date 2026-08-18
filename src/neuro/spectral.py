from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import scipy.signal as sps
from scipy.signal.windows import hann

if TYPE_CHECKING:
    from pathlib import Path

    from neuro.types import FloatArray

LOG_FLOOR = 1e-8
"""Floor added to measured power before the log; the envelope itself is strictly positive."""


def compute_periodograms(y: FloatArray, *, fs: float, window: int, hop: int) -> FloatArray:
    """Periodograms of every length-``window`` slice of ``y`` ``(n_samples, n_channels)``, hopped by ``hop``.

    One-sided, density-scaled, periodic Hann, ``detrend="constant"`` -- the geometry
    :meth:`neuro.control.MPCNlp.build` replicates symbolically. Returns ``(n_windows, n_channels,
    window // 2 + 1)``; segments are never averaged over, so a hot slice cannot be cancelled by a
    cold one.
    """
    n_samples, n_channels = y.shape
    n_bins = window // 2 + 1
    if n_samples < window:
        return np.empty((0, n_channels, n_bins), dtype=np.float64)

    n_windows = (n_samples - window) // hop + 1
    w_hann = hann(window, sym=False)
    segments = np.stack([y[m * hop : m * hop + window, :] for m in range(n_windows)], axis=0)
    _, power = sps.periodogram(segments, fs=fs, window=w_hann, detrend="constant", axis=1, scaling="density")
    return np.asarray(power, dtype=np.float64).transpose(0, 2, 1)


def hinge_penalty(power: FloatArray, reference: FloatArray) -> float:
    """Mean squared one-sided log excess of ``power`` over ``reference``; exactly 0 when everywhere under."""
    excess = np.log(power + LOG_FLOOR) - np.log(reference)
    return float(np.mean(np.maximum(0.0, excess) ** 2))


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
