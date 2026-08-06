from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from pydantic import Field

from neuro.config import StrictConfig
from neuro.geometry import SENSORS_FILE

if TYPE_CHECKING:
    from neuro.types import FloatArray, StrArray

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    from tvb.datatypes.connectivity import Connectivity
    from tvb.datatypes.projections import ProjectionSurfaceEEG
    from tvb.datatypes.region_mapping import RegionMapping
    from tvb.datatypes.sensors import SensorsEEG

_PROJECTION_FILE = "projection_eeg_62_surface_16k.mat"
_REGION_MAPPING_FILE = "regionMapping_16k_76.txt"


class _ConnectomeConfig(StrictConfig):
    """Config schema for :meth:`Connectome.from_config`."""

    speed: float = Field(default=50.0, gt=0)
    K: float = 1.0


@dataclass(frozen=True)
class Connectome:
    """Bundle of structural data the whole-brain network builds on.

    Attributes
    ----------
    K
        Global coupling constant, float
    weights
        Region-by-region connection strengths, shape ``(76, 76)``, nonnegative.
    tract_lengths
        Region-by-region fibre lengths in mm, shape ``(76, 76)``.
    centres
        Region centroid coordinates in mm, shape ``(76, 3)``.
    region_labels
        Region names, shape ``(76,)`` (e.g. ``lHC``, ``rTCI``).
    hemispheres
        Boolean mask, shape ``(76,)``; ``True`` marks the right hemisphere
        (TVB convention).
    speed
        Conduction speed in mm/ms used to derive ``delays``.
    delays
        Conduction delays in ms, ``tract_lengths / speed``, shape ``(76, 76)``.
    gain
        EEG forward operator ``L``, shape ``(62, 76)``; maps region outputs to
        sensor channels.
    channel_labels
        EEG channel names, shape ``(62,)`` (e.g. ``CP5``, ``PO3``).
    region_index
        Map from region label to its row/column index.
    channel_index
        Map from channel label to its row index in ``gain``.

    The tES projection lives on a :class:`~neuro.stimulation.base.StimulationModel`, not here.
    """

    K: float
    weights: FloatArray
    tract_lengths: FloatArray
    centres: FloatArray
    region_labels: StrArray
    hemispheres: npt.NDArray[np.bool_]
    speed: float
    delays: FloatArray
    gain: FloatArray
    channel_labels: StrArray
    region_index: dict[str, int]
    channel_index: dict[str, int]

    def __post_init__(self) -> None:
        """Validate that array shapes match the number of regions and channels."""
        n_nodes = len(self.region_labels)
        n_eeg_channels = len(self.channel_labels)

        if self.weights.shape != (n_nodes, n_nodes):
            msg = f"weights shape {self.weights.shape} does not match ({n_nodes}, {n_nodes})"
            raise ValueError(msg)
        if self.tract_lengths.shape != (n_nodes, n_nodes):
            msg = f"tract_lengths shape {self.tract_lengths.shape} does not match ({n_nodes}, {n_nodes})"
            raise ValueError(msg)
        if self.centres.shape != (n_nodes, 3):
            msg = f"centres shape {self.centres.shape} does not match ({n_nodes}, 3)"
            raise ValueError(msg)
        if self.hemispheres.shape != (n_nodes,):
            msg = f"hemispheres shape {self.hemispheres.shape} does not match ({n_nodes},)"
            raise ValueError(msg)
        if self.delays.shape != (n_nodes, n_nodes):
            msg = f"delays shape {self.delays.shape} does not match ({n_nodes}, {n_nodes})"
            raise ValueError(msg)
        if self.gain.shape != (n_eeg_channels, n_nodes):
            msg = f"gain shape {self.gain.shape} does not match ({n_eeg_channels}, {n_nodes})"
            raise ValueError(msg)

    def delay_steps(self, dt: float) -> npt.NDArray[np.int64]:
        """Conduction delays as integer step lags for integration step ``dt`` (s)."""
        return np.round(self.delays / (dt * 1000.0)).astype(np.int64)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Connectome:
        """Load the TVB structural backbone and EEG forward operator from a config dict."""
        cfg = _ConnectomeConfig.model_validate(config)

        conn = Connectivity.from_file()
        conn.speed = np.array([cfg.speed])
        conn.configure()

        weights = np.asarray(conn.weights, dtype=np.float64)
        tract_lengths = np.asarray(conn.tract_lengths, dtype=np.float64)
        centres = np.asarray(conn.centres, dtype=np.float64)
        region_labels = np.asarray(conn.region_labels, dtype=np.str_)
        hemispheres = np.asarray(conn.hemispheres, dtype=np.bool_)
        delays = tract_lengths / cfg.speed

        gain, channel_labels = _build_eeg_gain()

        region_index = {label: idx for idx, label in enumerate(region_labels)}
        channel_index = {label: idx for idx, label in enumerate(channel_labels)}

        return cls(
            K=cfg.K,
            weights=weights,
            tract_lengths=tract_lengths,
            centres=centres,
            region_labels=region_labels,
            hemispheres=hemispheres,
            speed=cfg.speed,
            delays=delays,
            gain=gain,
            channel_labels=channel_labels,
            region_index=region_index,
            channel_index=channel_index,
        )


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


def _build_eeg_gain() -> tuple[FloatArray, StrArray]:
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
