from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt
from pydantic import Field

from neuro.config import StrictConfig

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from neuro.types import FloatArray

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    from tvb.datatypes.connectivity import Connectivity
    from tvb.datatypes.projections import ProjectionSurfaceEEG
    from tvb.datatypes.region_mapping import RegionMapping
    from tvb.datatypes.sensors import SensorsEEG

StrArray = npt.NDArray[np.str_]

_SENSORS_FILE = "eeg_unitvector_62.txt.bz2"
_PROJECTION_FILE = "projection_eeg_62_surface_16k.mat"
_REGION_MAPPING_FILE = "regionMapping_16k_76.txt"

_SENSOR_TO_CONNECTOME = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
SCALP_RADIUS_MM = 90.0
EXTRACEPHALIC_ELECTRODES_MM = {"EX_NECK": (-60.0, 0.0, -180.0)}


class _ConnectomeConfig(StrictConfig):
    """Config schema for :meth:`Connectome.from_config`."""

    speed: float = Field(default=50.0, gt=0)
    K: float = 1.0
    target_electrode: str | int | list[str | int] | None = None
    gamma_spread: float | list[float] = 15.0
    gamma_model: Literal["analytical"] = "analytical"


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
    gamma
        tES spatial projection, all zeros until configured. Shape ``(76,)`` for a
        single electrode or ``(n_electrodes, 76)`` for a multi-electrode montage.
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
    gamma: FloatArray

    def __post_init__(self) -> None:
        """Validate that array shapes match the number of regions and channels."""
        n_nodes = len(self.region_labels)
        n_channels = len(self.channel_labels)

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
        if self.gain.shape != (n_channels, n_nodes):
            msg = f"gain shape {self.gain.shape} does not match ({n_channels}, {n_nodes})"
            raise ValueError(msg)
        if self.gamma.ndim == 1 and self.gamma.shape != (n_nodes,):
            msg = f"gamma shape {self.gamma.shape} does not match ({n_nodes},)"
            raise ValueError(msg)
        if self.gamma.ndim == 2 and self.gamma.shape[1] != n_nodes:  # noqa: PLR2004
            msg = f"gamma shape {self.gamma.shape} does not match (*, {n_nodes})"
            raise ValueError(msg)

    def delay_steps(self, dt: float) -> npt.NDArray[np.int64]:
        """Conduction delays as integer step lags for integration step ``dt`` (s)."""
        return np.round(self.delays / (dt * 1000.0)).astype(np.int64)

    def to_npz(self, path: str | Path) -> None:
        """Save the connectome to an NPZ file."""
        np.savez(
            path,
            K=np.array(self.K),
            weights=self.weights,
            tract_lengths=self.tract_lengths,
            centres=self.centres,
            region_labels=self.region_labels,
            hemispheres=self.hemispheres,
            speed=np.array(self.speed),
            delays=self.delays,
            gain=self.gain,
            channel_labels=self.channel_labels,
            gamma=self.gamma,
        )

    @classmethod
    def from_npz(cls, path: str | Path) -> Connectome:
        """Load the connectome from an NPZ file."""
        with np.load(path) as data:
            region_labels = data["region_labels"].astype(np.str_)
            channel_labels = data["channel_labels"].astype(np.str_)
            return cls(
                K=float(data["K"]),
                weights=data["weights"],
                tract_lengths=data["tract_lengths"],
                centres=data["centres"],
                region_labels=region_labels,
                hemispheres=data["hemispheres"],
                speed=float(data["speed"]),
                delays=data["delays"],
                gain=data["gain"],
                channel_labels=channel_labels,
                region_index={str(label): idx for idx, label in enumerate(region_labels)},
                channel_index={str(label): idx for idx, label in enumerate(channel_labels)},
                gamma=data["gamma"],
            )

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

        gamma: FloatArray = np.zeros(len(region_labels), dtype=np.float64)
        if cfg.target_electrode is not None:
            electrodes = cfg.target_electrode if isinstance(cfg.target_electrode, list) else [cfg.target_electrode]
            resolved = [str(channel_labels[el]) if isinstance(el, int) else str(el) for el in electrodes]
            gamma = _compute_gamma(
                centres,
                target_electrode=resolved if len(resolved) > 1 else resolved[0],
                spread=cfg.gamma_spread,
                model=cfg.gamma_model,
            )

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
            gamma=gamma,
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
    sensors = SensorsEEG.from_file(_SENSORS_FILE)
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


def sensor_positions_mm(
    sensors_file: str = _SENSORS_FILE, radius: float = SCALP_RADIUS_MM
) -> tuple[StrArray, FloatArray]:
    """Scalp electrode positions in the connectome's coordinate frame, in millimetres.

    TVB's sensor file gives unit vectors in a (left, posterior, superior) frame; the region
    centres are millimetres in an (anterior, left, superior) frame. Distances between the two
    are only meaningful once the sensors are rotated into the connectome frame and scaled out
    to the scalp, which is what this does.

    Returns
    -------
    labels
        Channel names, shape ``(n_sensors,)``.
    positions
        Electrode positions, shape ``(n_sensors, 3)``, on the sphere of radius ``radius``.
    """
    sensors = SensorsEEG.from_file(sensors_file)
    labels = np.asarray(sensors.labels, dtype=np.str_)
    unit = np.asarray(sensors.locations, dtype=np.float64)
    return labels, (unit @ _SENSOR_TO_CONNECTOME.T) * radius


def centres_to_mni_ras(coords: FloatArray) -> FloatArray:
    """Convert connectome-frame coordinates to MNI RAS, in millimetres.

    The connectome frame is (anterior, left, superior); MNI RAS is (right, anterior, superior).
    This is the registration seam every external head model is reached through.
    """
    coords = np.asarray(coords, dtype=np.float64)
    return np.column_stack([-coords[:, 1], coords[:, 0], coords[:, 2]])


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


def _compute_gamma(
    centres: FloatArray,
    target_electrode: str | Sequence[str] = "CP5",
    spread: float | Sequence[float] = 15.0,
    sensors_file: str = _SENSORS_FILE,
    model: Literal["analytical"] = "analytical",  # noqa: ARG001
) -> FloatArray:
    """Compute the spatial projection gamma for tES stimulation.

    Parameters
    ----------
    centres
        Region centroids coordinates in mm, shape (N_regions, 3).
    target_electrode
        Label of the stimulating electrode (e.g. 'CP5'), or a sequence of labels
        for a montage (e.g. ('TP9', 'EX_NECK')).
    spread
        Coulomb softening radius in mm, keeping the ``analytical`` kernel finite at the
        electrode; a scalar shared across electrodes, or one value per electrode.
    sensors_file
        Filename of the EEG sensors dataset.
    model
        Current-to-field model: 'analytical' (Coulomb volume potential of a point source in
        a homogeneous medium).

    Returns
    -------
    FloatArray
        Spatial projection matrix of shape ``(N_regions,)`` for a single electrode, or
        ``(n_electrodes, N_regions)`` for a montage.
    """
    _gamma_one = _coulomb_potential_fn(centres, sensors_file)

    if isinstance(target_electrode, str):
        if not isinstance(spread, (int, float)):
            msg = "spread must be a scalar when target_electrode is a single label."
            raise TypeError(msg)
        return _gamma_one(target_electrode, spread)

    electrodes = list(target_electrode)
    spreads = [spread] * len(electrodes) if isinstance(spread, (int, float)) else list(spread)
    if len(spreads) != len(electrodes):
        msg = f"spread sequence length {len(spreads)} must match {len(electrodes)} electrodes."
        raise ValueError(msg)
    return np.stack([_gamma_one(e, s) for e, s in zip(electrodes, spreads, strict=True)])
