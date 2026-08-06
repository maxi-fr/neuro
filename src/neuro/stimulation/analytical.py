from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from neuro.geometry import EXTRACEPHALIC_ELECTRODES_MM, SENSORS_FILE, sensor_positions_mm
from neuro.stimulation.base import StimulationModel, check_n_controls

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from neuro.stimulation.base import _AnalyticalConfig
    from neuro.types import FloatArray


def _coulomb_potential_fn(centres: FloatArray, sensors_file: str) -> Callable[[str, float], FloatArray]:
    """Build the per-electrode gamma of the ``analytical`` model: a Coulomb volume potential.

    ``100 / sqrt(r^2 + spread^2)`` is the potential of a point source in a homogeneous medium,
    softened at the electrode by ``spread``.
    """
    sensor_labels, sensor_positions = sensor_positions_mm(sensors_file)
    positions = {str(label).upper(): pos for label, pos in zip(sensor_labels, sensor_positions, strict=True)}
    positions |= {label: np.asarray(pos, dtype=np.float64) for label, pos in EXTRACEPHALIC_ELECTRODES_MM.items()}

    def _potential(electrode: str, spread: float) -> FloatArray:
        target = electrode.upper()
        if target not in positions:
            msg = f"Electrode {electrode} not found in sensors."
            raise ValueError(msg)
        dists = np.linalg.norm(centres - positions[target], axis=1)
        return 100.0 / np.sqrt((dists**2) + (spread**2))

    return _potential


def compute_gamma(
    centres: FloatArray,
    electrodes: Sequence[str],
    spread: float | Sequence[float] = 15.0,
    sensors_file: str = SENSORS_FILE,
) -> FloatArray:
    """Coulomb spatial projection of a tES montage, shape ``(n_electrodes, n_regions)``.

    Row ``i`` is the potential per unit current injected at ``electrodes[i]``, evaluated at each
    region centroid; ``spread`` is the softening radius in mm, shared or one value per electrode.
    """
    gamma_one = _coulomb_potential_fn(centres, sensors_file)
    spreads = [spread] * len(electrodes) if isinstance(spread, (int, float)) else list(spread)
    return np.stack([gamma_one(e, s) for e, s in zip(electrodes, spreads, strict=True)])


class AnalyticalStim(StimulationModel):
    """tES field as a Coulomb volume potential per electrode in a homogeneous head."""

    def __init__(self, cfg: _AnalyticalConfig, centres: FloatArray) -> None:
        """Build the montage's gamma rows from the region centroids ``centres``."""
        self.gamma = compute_gamma(centres, cfg.electrodes, cfg.spread)
        self.control_labels = np.asarray(cfg.electrodes, dtype=np.str_)
        self.n_controls = len(cfg.electrodes)

    def project(self, u: FloatArray) -> FloatArray:
        """Superpose the per-electrode potentials weighted by their currents."""
        check_n_controls(u, self.n_controls)
        return u @ self.gamma
