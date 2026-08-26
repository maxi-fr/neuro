from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Self

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi, sosfreqz
from simulate.estimator import Estimator

from neuro.spectral import compute_log_power_frames

if TYPE_CHECKING:
    from neuro.config import StftGeometry
    from neuro.types import FloatArray

_ORDER = 4


def design_lowpass_sos(fs: float, cutoff: float) -> FloatArray:
    """Design the causal Butterworth low-pass of order ``_ORDER`` at ``cutoff`` Hz.

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


def lowpass_filter(y: FloatArray, fs: float, cutoff_hz: float) -> FloatArray:
    """Apply the causal Butterworth low-pass along axis 0, starting from zero filter state.

    Parameters
    ----------
    y : FloatArray
        Signal of shape ``(T, C)`` sampled at ``fs``.
    fs : float
        Sample rate of ``y``.
    cutoff_hz : float
        -3 dB cutoff frequency in Hz.

    Returns
    -------
    FloatArray
        Filtered signal, same shape as ``y``.
    """
    return np.asarray(sosfilt(design_lowpass_sos(fs, cutoff_hz), y, axis=0), dtype=np.float64)


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
    return lowpass_filter(y, fs, fs / (2 * downsample))


@dataclasses.dataclass(frozen=True)
class LowPassEstimatorLog:
    """Log carrying the low-pass filtered measurement handed to the controller."""

    x_hat: np.ndarray


class LowPassEstimator(Estimator[LowPassEstimatorLog]):
    """Estimator that causally low-passes the measurement at a specified cutoff frequency."""

    def __init__(self, dt: float, cutoff_hz: float) -> None:
        """Initialize the filter from the plant sample time ``dt`` and the cutoff frequency."""
        super().__init__(dt)
        self.sos = design_lowpass_sos(1.0 / dt, cutoff_hz)
        self._zi: np.ndarray | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary."""
        return cls(dt=float(config["dt"]), cutoff_hz=float(config["cutoff_hz"]))

    def update(
        self,
        t: float,  # noqa: ARG002
        y_mea: np.ndarray,
        u: np.ndarray,  # noqa: ARG002
    ) -> tuple[np.ndarray, LowPassEstimatorLog]:
        """Advance the low-pass by one sample and return the filtered measurement.

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
        log : LowPassEstimatorLog
            Log containing the filtered measurement.
        """
        if self._zi is None:
            self._zi = np.zeros((self.sos.shape[0], 2, np.atleast_1d(y_mea).size))

        filtered, self._zi = sosfilt(self.sos, np.atleast_1d(y_mea)[None, :], axis=0, zi=self._zi)
        x_hat = np.asarray(filtered[0]).reshape(np.shape(y_mea))
        return x_hat, LowPassEstimatorLog(x_hat=x_hat.copy())


class AntiAliasEstimator(LowPassEstimator):
    """Estimator that causally low-passes the measurement at the controller's Nyquist rate.

    Runs at the plant ``dt`` and applies exactly the filter
    :func:`neuro.predictor.data.load_trajectory` applies before striding, so the controller's
    zero-order hold picking off every ``downsample``-th estimate reproduces the decimation the
    predictor was identified on. The filter state starts at zero, as it does offline.
    """

    def __init__(self, dt: float, downsample: int) -> None:
        """Initialize the filter from the plant sample time ``dt`` and the decimation factor."""
        super().__init__(dt=dt, cutoff_hz=(1.0 / dt) / (2 * downsample))
        self.downsample = downsample

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary."""
        return cls(dt=float(config["dt"]), downsample=int(config["downsample"]))


@dataclasses.dataclass(frozen=True)
class ObservableEstimatorLog:
    """Log carrying the Observable log-power Frame handed to the controller."""

    x_hat: np.ndarray


class ObservableEstimator(Estimator[ObservableEstimatorLog]):
    """Estimator that causally low-passes at plant rate, decimates, buffers and emits Observable Frames."""

    def __init__(
        self,
        dt: float,
        geometry: StftGeometry,
        downsample: int = 1,
        cutoff_hz: float | None = None,
    ) -> None:
        """Initialize from plant sample time, Observable geometry, and decimation factor."""
        super().__init__(dt)
        self.downsample = int(downsample)
        self.geometry = geometry
        self.fs_decimated = 1.0 / (self.dt * self.downsample)
        self.sample_support = (geometry.kernel_width - 1) * geometry.n_hop + geometry.n_segment
        if self.downsample > 1 or cutoff_hz is not None:
            effective_cutoff = cutoff_hz if cutoff_hz is not None else (1.0 / self.dt) / (2 * self.downsample)
            self.sos: np.ndarray | None = design_lowpass_sos(1.0 / self.dt, effective_cutoff)
        else:
            self.sos = None
        self._zi: np.ndarray | None = None
        self._decimated_buffer: list[np.ndarray] = []
        self._step: int = 0
        self._decimated_count: int = 0
        self._current_frame: np.ndarray | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary."""
        from neuro.config import StftGeometry  # noqa: PLC0415

        geom_raw = config["geometry"]
        geometry = geom_raw if isinstance(geom_raw, StftGeometry) else StftGeometry.model_validate(geom_raw)
        return cls(
            dt=float(config["dt"]),
            geometry=geometry,
            downsample=int(config.get("downsample", 1)),
            cutoff_hz=float(config["cutoff_hz"]) if "cutoff_hz" in config and config["cutoff_hz"] is not None else None,
        )

    def update(
        self,
        t: float,  # noqa: ARG002 -- simulation time
        y_mea: np.ndarray,
        u: np.ndarray,  # noqa: ARG002 -- control input vector
    ) -> tuple[np.ndarray, ObservableEstimatorLog]:
        """Advance by one plant sample, decimating and emitting Observable Frames at the hop rate.

        Parameters
        ----------
        t : float
            Simulation time.
        y_mea : numpy.ndarray
            Measured output vector of shape ``(n_channels,)``.
        u : numpy.ndarray
            Control input vector.

        Returns
        -------
        x_hat : numpy.ndarray
            Observable log-power Frame of shape ``(n_channels, n_values)``.
        log : ObservableEstimatorLog
            Log containing the emitted Frame.
        """
        y_arr = np.atleast_1d(y_mea)
        if self.sos is not None:
            if self._zi is None:
                self._zi = np.zeros((self.sos.shape[0], 2, y_arr.size))
            filtered, self._zi = sosfilt(self.sos, y_arr[None, :], axis=0, zi=self._zi)
            y_filt = np.asarray(filtered[0], dtype=np.float64)
        else:
            y_filt = np.asarray(y_arr, dtype=np.float64)

        if self._step % self.downsample == 0:
            self._decimated_count += 1
            self._decimated_buffer.append(y_filt.copy())
            if len(self._decimated_buffer) > self.sample_support:
                self._decimated_buffer.pop(0)

            if (
                self._decimated_count >= self.sample_support
                and (self._decimated_count - self.sample_support) % self.geometry.n_hop == 0
            ):
                buf = np.stack(self._decimated_buffer, axis=0)
                frames = compute_log_power_frames(buf, self.geometry, fs=self.fs_decimated)
                self._current_frame = frames[0]

        self._step += 1

        if self._current_frame is None:
            n_channels = y_arr.size
            n_values = self.geometry.n_values(self.fs_decimated)
            self._current_frame = np.full((n_channels, n_values), np.nan, dtype=np.float64)

        return self._current_frame, ObservableEstimatorLog(x_hat=self._current_frame.copy())
