from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Self

import numpy as np
from pydantic import Field
from simulate.controller import Controller

from neuro.config import StrictConfig

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from neuro.types import FloatArray


@dataclasses.dataclass(frozen=True)
class AmplitudeThresholdControllerLog:
    """Dataclass for AmplitudeThresholdController logging."""

    u: FloatArray
    amplitude: float
    triggered: bool
    active: bool


class _AmplitudeThresholdControllerConfig(StrictConfig):
    """Config schema for :class:`AmplitudeThresholdController`."""

    dt: float = Field(gt=0)
    amplitude: float | list[float]
    threshold: float = Field(gt=0)
    burst_duration: float = Field(gt=0)
    window: float = Field(default=1.0, gt=0)
    channel: int | None = Field(default=None, ge=0)
    n_u: int = Field(default=1, ge=1)


class AmplitudeThresholdController(Controller[AmplitudeThresholdControllerLog]):
    """Trigger a fixed-length constant stimulus whenever EEG amplitude crosses a threshold."""

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        dt: float,
        amplitude: ArrayLike,
        threshold: float,
        burst_duration: float,
        window: float = 1.0,
        channel: int | None = None,
        n_u: int = 1,
    ) -> None:
        """Initialize the trigger.

        Parameters
        ----------
        dt
            Controller update step in seconds.
        amplitude
            tES current held during a burst: a scalar shared by every electrode or a
            length-``n_u`` per-electrode vector. Should sum to zero (Kirchhoff).
        threshold
            Peak-to-peak amplitude that starts a burst, in the measurement's units.
        burst_duration
            Length of one stimulation burst in seconds (the paper's tau).
        window
            Length of the trailing window the peak-to-peak is measured over, in seconds.
        channel
            Index of the monitored measurement channel; ``None`` (default) monitors every
            channel and triggers on the largest peak-to-peak among them.
        n_u
            Number of stimulation electrodes (control dimension).
        """
        super().__init__(dt)
        self.threshold = float(threshold)
        self.burst_duration = float(burst_duration)
        self.window = float(window)
        self.channel = channel
        self.n_u = n_u

        amp = np.atleast_1d(np.asarray(amplitude, dtype=np.float64))
        if amp.size == 1:
            amp = np.broadcast_to(amp, (n_u,))
        elif amp.size != n_u:
            msg = f"amplitude has {amp.size} entries but n_u is {n_u}"
            raise ValueError(msg)
        self.amplitude = np.ascontiguousarray(amp)

        self._n_window = max(1, round(self.window / dt))
        self._buf: FloatArray | None = None
        self._n_seen = 0
        self._until = -np.inf

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary."""
        cfg = _AmplitudeThresholdControllerConfig.model_validate(config)
        return cls(
            dt=cfg.dt,
            amplitude=cfg.amplitude,
            threshold=cfg.threshold,
            burst_duration=cfg.burst_duration,
            window=cfg.window,
            channel=cfg.channel,
            n_u=cfg.n_u,
        )

    def update(
        self,
        t: float,
        ref: FloatArray,  # noqa: ARG002
        x_hat: FloatArray,
    ) -> tuple[FloatArray, AmplitudeThresholdControllerLog]:
        """Ingest the EEG measurement and emit the burst amplitude while a burst is running."""
        y = np.atleast_1d(np.asarray(x_hat, dtype=np.float64).reshape(-1))
        if self._buf is None:
            self._buf = np.zeros((self._n_window, y.size), dtype=np.float64)
        self._buf = np.vstack([self._buf[1:], y])
        self._n_seen += 1

        filled = self._n_seen >= self._n_window
        ptp = np.ptp(self._buf, axis=0)
        amplitude = float(ptp[self.channel] if self.channel is not None else ptp.max()) if filled else 0.0

        triggered = t >= self._until and amplitude > self.threshold
        if triggered:
            self._until = t + self.burst_duration
        active = t < self._until
        u = self.amplitude if active else np.zeros(self.n_u, dtype=np.float64)
        return u, AmplitudeThresholdControllerLog(u=u.copy(), amplitude=amplitude, triggered=triggered, active=active)
