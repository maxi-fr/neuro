from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import numpy.typing as npt
from pydantic import Field

from neuro.config import StrictConfig
from neuro.geometry import SENSORS_FILE
from neuro.jansen_rit import lfp as regional_lfp

if TYPE_CHECKING:
    from neuro.types import FloatArray, StrArray

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    from tvb.datatypes.projections import ProjectionSurfaceEEG
    from tvb.datatypes.region_mapping import RegionMapping
    from tvb.datatypes.sensors import SensorsEEG

_PROJECTION_FILE = "projection_eeg_62_surface_16k.mat"
_REGION_MAPPING_FILE = "regionMapping_16k_76.txt"


def _mirror_partner_permutation(locations: FloatArray) -> npt.NDArray[np.int64]:
    """Map each sensor to the sensor at its sagittal-mirror position.

    TVB's ``eeg_unitvector_62`` sensor file and ``projection_eeg_62_surface_16k``
    lead field are in mirror-image left-right conventions: projection row ``i``
    carries the lead field of the electrode *contralateral* to ``labels[i]``, so a
    naive ``row i -> labels[i]`` pairing flips every channel to the wrong
    hemisphere. Reflecting each sensor across the montage's sagittal plane and
    matching to the nearest sensor recovers the correct row<->label pairing
    (midline sensors map to themselves).
    """

    def _reflect(axis: int) -> FloatArray:
        reflected = locations.copy()
        reflected[:, axis] *= -1.0
        return reflected

    def _match_cost(reflected: FloatArray) -> tuple[npt.NDArray[np.int64], float]:
        dist = np.linalg.norm(locations[:, None, :] - reflected[None, :, :], axis=2)
        partner = dist.argmin(axis=1).astype(np.int64)
        return partner, float(dist[np.arange(len(partner)), partner].sum())

    best_partner, best_cost = np.arange(len(locations), dtype=np.int64), np.inf
    for axis in range(locations.shape[1]):
        partner, cost = _match_cost(_reflect(axis))
        if cost < best_cost:
            best_partner, best_cost = partner, cost
    return best_partner


def build_eeg_gain() -> tuple[FloatArray, StrArray]:
    """Build the region-level EEG gain ``L`` and its channel labels.

    Collapses TVB's ``(62, 16384)`` surface projection to ``(62, 76)`` by summing
    the projection columns of the vertices mapped to each region.
    """
    sensors = SensorsEEG.from_file(SENSORS_FILE)
    channel_labels = np.asarray(sensors.labels, dtype=np.str_)
    locations = np.asarray(sensors.locations, dtype=np.float64)
    partner = _mirror_partner_permutation(locations)

    proj = ProjectionSurfaceEEG.from_file(_PROJECTION_FILE, matlab_data_name="ProjectionMatrix")
    surface_gain = np.asarray(proj.projection_data, dtype=np.float64)[partner]
    rmap = np.asarray(RegionMapping.from_file(_REGION_MAPPING_FILE).array_data, dtype=np.int64)

    n_sensors = surface_gain.shape[0]
    n_regions = int(rmap.max()) + 1
    gain = np.zeros((n_sensors, n_regions), dtype=np.float64)
    for r in range(n_regions):
        gain[:, r] = surface_gain[:, rmap == r].sum(axis=1)

    return gain, channel_labels


class _EEGMeasurementConfig(StrictConfig):
    """Config schema for :class:`EEGMeasurement`."""

    n_nodes: int | None = Field(default=None, ge=1)
    selected_channels: list[str | int] | None = None


class EEGMeasurement:
    """Map the Jansen-Rit network state to scalp EEG via the forward operator."""

    def __init__(
        self,
        n_nodes: int | None = None,
        selected_channels: list[int | str] | None = None,
    ) -> None:
        """Load the ``(n_channels, n_nodes)`` EEG forward operator."""
        gain, channel_labels = build_eeg_gain()
        if n_nodes is not None:
            gain = gain[:, : int(n_nodes)]
        self.gain = np.asarray(gain, dtype=np.float64)
        self.n_nodes = self.gain.shape[1]

        if selected_channels is None:
            self.selected_channels = None
        else:
            channel_index = {label: idx for idx, label in enumerate(channel_labels)}
            resolved = [channel_index[ch] if isinstance(ch, str) else int(ch) for ch in selected_channels]
            self.selected_channels = np.array(resolved, dtype=np.int64)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary."""
        cfg = _EEGMeasurementConfig.model_validate(config)
        return cls(n_nodes=cfg.n_nodes, selected_channels=cfg.selected_channels)

    def __call__(
        self,
        _t: float,
        x: FloatArray,
        _u: FloatArray,
    ) -> FloatArray:
        """Collapse the network state ``x`` to an EEG channel vector."""
        x_grid = x.reshape(6, self.n_nodes)
        eeg = self.gain @ regional_lfp(x_grid)
        if self.selected_channels is not None:
            eeg = eeg[self.selected_channels]
        return eeg
