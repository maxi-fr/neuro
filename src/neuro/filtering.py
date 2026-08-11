from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Self

import numpy as np
from scipy.signal import butter, sosfilt
from simulate.estimator import Estimator

if TYPE_CHECKING:
    from neuro.types import FloatArray

_ORDER = 4


def design_antialias_sos(fs: float, downsample: int) -> FloatArray:
    """Design the causal anti-alias low-pass shared by the training and closed-loop paths.

    Butterworth of order ``_ORDER`` in second-order-section form, cut off at the Nyquist rate
    implied by ``downsample`` (``fs / (2 * downsample)``). Both the offline decimation in
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
    return np.asarray(butter(_ORDER, fs / (2 * downsample), btype="low", fs=fs, output="sos"), dtype=np.float64)


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
