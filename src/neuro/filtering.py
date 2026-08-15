from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Self

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi, sosfreqz
from simulate.estimator import Estimator

if TYPE_CHECKING:
    from neuro.types import FloatArray

_ORDER = 4


def design_lowpass_sos(fs: float, cutoff: float) -> FloatArray:
    """Design the causal Butterworth low-pass of order ``_ORDER`` at ``cutoff`` Hz.

    The one low-pass definition in the repo. :func:`design_antialias_sos` picks its cutoff from a
    decimation factor; the bandwidth sweep of
    ``docs/predictability_controllability_experiment.md`` picks it directly, and both land here so
    the sweep cannot drift into a second, silently different filter.

    Parameters
    ----------
    fs : float
        Sample rate of the signal to be filtered.
    cutoff : float
        -3 dB cutoff in Hz.

    Returns
    -------
    FloatArray
        Second-order sections, shape ``(_ORDER // 2, 6)``.
    """
    return np.asarray(butter(_ORDER, cutoff, btype="low", fs=fs, output="sos"), dtype=np.float64)


def design_bandpass_sos(fs: float, band: tuple[float, float]) -> FloatArray:
    """Design the causal Butterworth band-pass of order ``_ORDER`` over ``band`` in Hz.

    Returns
    -------
    FloatArray
        Second-order sections, shape ``(_ORDER, 6)`` -- a band-pass doubles the section count.
    """
    return np.asarray(butter(_ORDER, band, btype="band", fs=fs, output="sos"), dtype=np.float64)


def causal_filter(signals: FloatArray, sos: FloatArray) -> FloatArray:
    """Apply ``sos`` causally along the last axis, started from the steady state of sample 0.

    ``sosfilt`` from a zero initial state opens with a settling transient that is near-identical
    across rollouts branched from one parent, which would inflate short-lookahead predictability
    for a reason that has nothing to do with the plant. Seeding the state with
    :func:`scipy.signal.sosfilt_zi` scaled to the first sample removes the step-from-zero, leaving
    only the filter's genuine group delay.

    Parameters
    ----------
    signals : FloatArray
        Signal of shape ``(..., n_samples)``.
    sos : FloatArray
        Second-order sections to apply.

    Returns
    -------
    FloatArray
        Filtered signal, same shape as ``signals``.
    """
    zi = sosfilt_zi(sos)[:, None, :] * signals[..., :1]
    filtered, _ = sosfilt(sos, signals, axis=-1, zi=zi)
    return np.asarray(filtered, dtype=np.float64)


def group_delay_s(sos: FloatArray, fs: float, *, freq_hz: float = 0.0) -> float:
    """Measure the group delay of ``sos`` at ``freq_hz``, in seconds.

    Measured from the designed sections rather than quoted from a closed form: it is the latency a
    controller actually pays for the filter, and it is what turns the bandwidth sweep into a
    trade-off curve instead of a maximisation that bottoms out at DC.

    Differences the phase across the sections rather than going through
    :func:`scipy.signal.sos2tf`. A low-pass narrow against ``fs`` -- 10 Hz at 10 kHz, the far end
    of the sweep -- puts every pole within ``1e-3`` of ``z = 1``, so the assembled denominator is
    ~``1e-9`` at DC and :func:`scipy.signal.group_delay` warns about the singularity it lands on.
    Each second-order section stays well conditioned.
    """
    delta = fs * 1e-6
    freqs = np.array([max(freq_hz - delta, 0.0), freq_hz + delta])
    _, response = sosfreqz(sos, worN=freqs, fs=fs)
    phase = np.unwrap(np.angle(response))
    return float(-(phase[1] - phase[0]) / (2.0 * np.pi * (freqs[1] - freqs[0])))


def design_antialias_sos(fs: float, downsample: int) -> FloatArray:
    """Design the causal anti-alias low-pass shared by the training and closed-loop paths.

    :func:`design_lowpass_sos` cut off at the Nyquist rate implied by ``downsample``
    (``fs / (2 * downsample)``). Both the offline decimation in
    :func:`neuro.nn_training.load_trajectory` and the online :class:`AntiAliasEstimator` design
    their filter here, so the model cannot be fit on a differently-filtered signal than it sees
    in the loop.

    Parameters
    ----------
    fs : float
        Sample rate of the signal to be filtered (the plant rate, e.g. 1e4 Hz).
    downsample : int
        Decimation factor applied after filtering; must be ``>= 2``.

    Returns
    -------
    FloatArray
        Second-order sections, shape ``(_ORDER // 2, 6)``.
    """
    return design_lowpass_sos(fs, fs / (2 * downsample))


def antialias_filter(y: FloatArray, fs: float, downsample: int) -> FloatArray:
    """Apply the causal anti-alias low-pass along axis 0, starting from zero filter state.

    Uses :func:`scipy.signal.sosfilt` rather than ``sosfiltfilt``/``scipy.signal.decimate``: the
    filter must be reproducible sample-by-sample online, so it cannot be zero-phase.

    Parameters
    ----------
    y : FloatArray
        Signal of shape ``(T, C)`` sampled at ``fs``.
    fs : float
        Sample rate of ``y``.
    downsample : int
        Decimation factor the filter is designed for; ``1`` returns ``y`` unchanged.

    Returns
    -------
    FloatArray
        Filtered signal, same shape as ``y``.
    """
    if downsample == 1:
        return y
    return np.asarray(sosfilt(design_antialias_sos(fs, downsample), y, axis=0), dtype=np.float64)


@dataclasses.dataclass(frozen=True)
class AntiAliasEstimatorLog:
    """Log carrying the low-pass filtered measurement handed to the controller."""

    x_hat: np.ndarray


class AntiAliasEstimator(Estimator[AntiAliasEstimatorLog]):
    """Estimator that causally low-passes the measurement at the controller's Nyquist rate.

    Runs at the plant ``dt`` and applies exactly the filter
    :func:`neuro.nn_training.load_trajectory` applies before striding, so the controller's
    zero-order hold picking off every ``downsample``-th estimate reproduces the decimation the
    predictor was identified on. The filter state starts at zero, as it does offline.
    """

    def __init__(self, dt: float, downsample: int) -> None:
        """Initialize the filter from the plant sample time ``dt`` and the decimation factor."""
        super().__init__(dt)
        self.sos = design_antialias_sos(1.0 / dt, downsample)
        self._zi: np.ndarray | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary."""
        return cls(dt=float(config["dt"]), downsample=int(config["downsample"]))

    def update(
        self,
        t: float,  # noqa: ARG002
        y_mea: np.ndarray,
        u: np.ndarray,  # noqa: ARG002
    ) -> tuple[np.ndarray, AntiAliasEstimatorLog]:
        """
        Advance the low-pass by one sample and return the filtered measurement.

        Parameters
        ----------
        t : float
            Simulation time.
        y_mea : numpy.ndarray
            Measured output vector.
        u : numpy.ndarray
            Control input vector.

        Returns
        -------
        x_hat : numpy.ndarray
            The low-pass filtered measurement.
        log : AntiAliasEstimatorLog
            Log containing the filtered measurement.
        """
        if self._zi is None:
            self._zi = np.zeros((self.sos.shape[0], 2, np.atleast_1d(y_mea).size))

        filtered, self._zi = sosfilt(self.sos, np.atleast_1d(y_mea)[None, :], axis=0, zi=self._zi)
        x_hat = np.asarray(filtered[0]).reshape(np.shape(y_mea))
        return x_hat, AntiAliasEstimatorLog(x_hat=x_hat.copy())
