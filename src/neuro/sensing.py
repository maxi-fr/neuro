"""Sensor components for the simulate framework.

Currently only :class:`DirectSensor`, an identity EEG sensor that pins its output
to a fixed channel count (with optional Gaussian noise).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self, cast

import numpy as np
from simulate.sensor import GaussianSensorLog, Sensor

if TYPE_CHECKING:
    from neuro.plant import FloatArray


class DirectSensor(Sensor[GaussianSensorLog]):
    """Identity EEG sensor with a fixed output dimension and optional Gaussian noise.

    Unlike :class:`simulate.sensor.GaussianSensor`, this pins the measurement to
    ``n_sensors`` channels. That matters because the
    :class:`~simulate.simulation.Simulation` orchestrator seeds its loop with a
    scalar ``y_k = 0.0``: at the first step the sensor receives that scalar, at
    every later step the full EEG vector. Broadcasting the scalar seed up to
    ``n_sensors`` keeps the logged measurement -- and the identity state estimate
    derived from it -- a homogeneous array that the logger can stack into an .npz.
    """

    def __init__(self, dt: float, n_sensors: int, std_dev: float = 0.0, seed: int = 42) -> None:
        """Initialize the direct sensor for ``n_sensors`` EEG channels."""
        super().__init__(dt)
        self.n_sensors = n_sensors
        self.std_dev = std_dev
        self.rng = np.random.default_rng(seed)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary."""
        return cls(
            dt=float(config["dt"]),
            n_sensors=int(config["n_sensors"]),
            std_dev=float(config.get("std_dev", 0.0)),
            seed=int(config.get("seed", 42)),
        )

    def update(
        self,
        t: float,  # noqa: ARG002
        y: float | np.ndarray,
    ) -> tuple[float | np.ndarray, GaussianSensorLog]:
        """Measure ``y`` as ``n_sensors`` channels, broadcasting a scalar seed."""
        y_vec = self.to_col_vec(y)
        if y_vec.shape[0] != self.n_sensors:
            y_vec = np.broadcast_to(y_vec, (self.n_sensors, 1))
        noise = self.rng.normal(0, self.std_dev, size=y_vec.shape)
        return cast("FloatArray", self.from_col_vec(y_vec + noise)), GaussianSensorLog(noise=noise)
