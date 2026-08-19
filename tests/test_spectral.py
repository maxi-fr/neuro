from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import scipy.signal as sps
from scipy.signal.windows import hann

from neuro.spectral import PsdEnvelope, compute_periodograms, hinge_penalty

if TYPE_CHECKING:
    from pathlib import Path

_SEED = 11


def test_periodograms_match_scipy_per_window() -> None:
    """Each returned window equals an undetrended scipy.signal.periodogram of that slice alone."""
    rng = np.random.default_rng(_SEED)
    y = rng.standard_normal((200, 3))
    fs, window, hop = 50.0, 50, 25

    power = compute_periodograms(y, fs=fs, window=window, hop=hop)

    assert power.shape == ((200 - window) // hop + 1, 3, window // 2 + 1)
    w_hann = hann(window, sym=False)
    for m in range(power.shape[0]):
        _, expected = sps.periodogram(
            y[m * hop : m * hop + window, :], fs=fs, window=w_hann, detrend=False, axis=0, scaling="density"
        )
        np.testing.assert_allclose(power[m], expected.T, rtol=1e-12, atol=1e-15)


def test_dc_offset_is_retained_rather_than_detrended() -> None:
    """No per-segment detrend: a constant offset lands in the DC bin at its full analytic power.

    A detrended periodogram would put ~0 there. The MPC cost drops bin 0 at the use site instead,
    so the high-pass corner never moves with the segment length.
    """
    fs, window = 50.0, 50
    offset = 3.0
    y = np.full((window, 1), offset)

    power = compute_periodograms(y, fs=fs, window=window, hop=window)

    w_hann = hann(window, sym=False)
    expected_dc = (offset * w_hann.sum()) ** 2 / (fs * np.sum(w_hann**2))
    np.testing.assert_allclose(power[0, 0, 0], expected_dc, rtol=1e-12)
    assert power[0, 0, 0] > 1.0


def test_periodograms_are_not_averaged_over_windows() -> None:
    """A burst confined to one window stays confined to it, rather than being diluted across windows."""
    fs, window, hop = 50.0, 50, 25
    y = np.zeros((100, 1))
    y[:window, 0] = 5.0 * np.sin(2 * np.pi * 5.0 * np.arange(window) / fs)

    power = compute_periodograms(y, fs=fs, window=window, hop=hop)

    burst_bin = power[:, 0, 5]
    assert burst_bin[0] > 100 * burst_bin[-1]


def test_periodogram_resolves_a_known_tone() -> None:
    """A 5 Hz tone at fs=50 with a 50-sample window lands in the 5 Hz bin (df = 1 Hz)."""
    fs, window = 50.0, 50
    t = np.arange(window) / fs
    y = np.sin(2 * np.pi * 5.0 * t)[:, None]

    power = compute_periodograms(y, fs=fs, window=window, hop=window)

    assert int(np.argmax(power[0, 0])) == 5


def test_short_signal_yields_no_windows() -> None:
    """A signal shorter than one window produces an empty, correctly shaped result."""
    power = compute_periodograms(np.zeros((10, 4)), fs=50.0, window=50, hop=25)
    assert power.shape == (0, 4, 26)


def test_hinge_penalty_is_zero_under_the_envelope() -> None:
    """The one-sided hinge is exactly 0 while power stays under the reference everywhere."""
    power = np.full((3, 2, 5), 1e-3)
    assert hinge_penalty(power, np.full((2, 5), 1e3)) == 0.0


def test_hinge_penalty_ignores_quiet_bins() -> None:
    """Power far below the envelope contributes nothing, so the cost never asks for more power."""
    reference = np.full((2, 5), 1.0)
    over = np.full((1, 2, 5), np.e)
    quiet = over.copy()
    quiet[0, 0, 0] = 1e-12

    assert hinge_penalty(quiet, reference) < hinge_penalty(over, reference)
    np.testing.assert_allclose(hinge_penalty(over, reference), 1.0, rtol=1e-6)


def test_envelope_load_round_trips(tmp_path: Path) -> None:
    """PsdEnvelope.load recovers the geometry written alongside the envelope."""
    path = tmp_path / "psd.npz"
    np.savez(path, Pref=np.ones((4, 26)), freqs=np.arange(26), fs=50.0, L=50, R=25)

    envelope = PsdEnvelope.load(path)

    assert (envelope.fs, envelope.window, envelope.hop) == (50.0, 50, 25)
    assert envelope.power.shape == (4, 26)


def test_envelope_load_rejects_subset_bins(tmp_path: Path) -> None:
    """A reference whose bin count contradicts its window is rejected, never silently subset."""
    path = tmp_path / "psd.npz"
    np.savez(path, Pref=np.ones((4, 10)), freqs=np.arange(10), fs=50.0, L=50, R=25)

    with pytest.raises(ValueError, match="must not subset bins"):
        PsdEnvelope.load(path)
