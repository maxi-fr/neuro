"""Stage 0 -- structural data loader for the Yu et al. 2024 tES model.

Loads the network's connectivity, delays, and EEG forward operator from
The Virtual Brain (TVB) and bundles them into a :class:`Connectome`.

Two conventions are fixed here and relied on by later stages:

* **All 76 TVB regions are kept.** The paper works with 74 regions; that count is
  TVB's 76 minus ``{lCC, rCC}`` (corpus callosum, indices 37 and 75). Keeping the
  full set keeps the Stage 7 TVB cross-check apples-to-apples; any drop is deferred
  to the network stage.
* **The EEG gain ``L`` collapses surface vertices to regions by SUM.** TVB's
  region-level EEG monitor replicates each region's activity to all of its
  vertices, so the forward model is ``EEG_s = sum_r x_r * sum_{v in r} proj[s, v]``
  -- a per-region sum, not a mean. Summing makes the Stage 7 ``L``-vs-monitor check
  agree without per-region rescaling.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Sequence

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    from tvb.datatypes.connectivity import Connectivity
    from tvb.datatypes.projections import ProjectionSurfaceEEG
    from tvb.datatypes.region_mapping import RegionMapping
    from tvb.datatypes.sensors import SensorsEEG

FloatArray = npt.NDArray[np.float64]
StrArray = npt.NDArray[np.str_]

_SENSORS_FILE = "eeg_unitvector_62.txt.bz2"
_PROJECTION_FILE = "projection_eeg_62_surface_16k.mat"
_REGION_MAPPING_FILE = "regionMapping_16k_76.txt"


@dataclass(frozen=True)
class Connectome:
    """Bundle of structural data the whole-brain network builds on.

    Attributes
    ----------
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
        tES spatial projection, ``None`` until configured. Shape ``(76,)`` for a
        single electrode or ``(n_electrodes, 76)`` for a multi-electrode montage.
    """

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
    gamma: FloatArray | None = None


def _build_eeg_gain() -> tuple[FloatArray, StrArray]:
    """Build the region-level EEG gain ``L`` and its channel labels.

    Collapses TVB's ``(62, 16384)`` surface projection to ``(62, 76)`` by summing
    the projection columns of the vertices mapped to each region.
    """
    sensors = SensorsEEG.from_file(_SENSORS_FILE)
    channel_labels = np.asarray(sensors.labels, dtype=np.str_)
    proj = ProjectionSurfaceEEG.from_file(_PROJECTION_FILE, matlab_data_name="ProjectionMatrix")
    surface_gain = np.asarray(proj.projection_data, dtype=np.float64)
    rmap = np.asarray(RegionMapping.from_file(_REGION_MAPPING_FILE).array_data, dtype=np.int64)

    n_sensors = surface_gain.shape[0]
    n_regions = int(rmap.max()) + 1
    gain = np.zeros((n_sensors, n_regions), dtype=np.float64)
    for r in range(n_regions):
        gain[:, r] = surface_gain[:, rmap == r].sum(axis=1)

    return gain, channel_labels


def load_connectome(speed: float = 50.0) -> Connectome:
    """Load the TVB structural backbone and EEG forward operator.

    Parameters
    ----------
    speed
        Conduction speed in mm/ms; the paper uses 50.

    Returns
    -------
    Connectome
        All 76 TVB regions with weights, delays, centres, labels, hemispheres,
        and the ``(62, 76)`` EEG gain ``L``.
    """
    conn = Connectivity.from_file()
    conn.speed = np.array([speed])
    conn.configure()  # derives the hemisphere mask absent from the default zip

    weights = np.asarray(conn.weights, dtype=np.float64)
    tract_lengths = np.asarray(conn.tract_lengths, dtype=np.float64)
    centres = np.asarray(conn.centres, dtype=np.float64)
    region_labels = np.asarray(conn.region_labels, dtype=np.str_)
    hemispheres = np.asarray(conn.hemispheres, dtype=np.bool_)
    delays = tract_lengths / speed

    gain, channel_labels = _build_eeg_gain()

    region_index = {label: idx for idx, label in enumerate(region_labels)}
    channel_index = {label: idx for idx, label in enumerate(channel_labels)}

    return Connectome(
        weights=weights,
        tract_lengths=tract_lengths,
        centres=centres,
        region_labels=region_labels,
        hemispheres=hemispheres,
        speed=speed,
        delays=delays,
        gain=gain,
        channel_labels=channel_labels,
        region_index=region_index,
        channel_index=channel_index,
    )


def compute_gamma(
    centres: FloatArray,
    target_electrode: str | Sequence[str] = "CP5",
    sigma: float | Sequence[float] = 20.0,
    sensors_file: str = _SENSORS_FILE,
) -> FloatArray:
    """Compute the normalized spatial projection gamma for tES stimulation.

    Parameters
    ----------
    centres
        Region centroids coordinates in mm, shape (N_regions, 3).
    target_electrode
        Label of the stimulating electrode (e.g. 'CP5'), or a sequence of labels
        for a multi-electrode montage (e.g. ('CP5', 'CP6')).
    sigma
        Spatial standard deviation (spread) in mm; a scalar shared across
        electrodes, or one value per electrode.
    sensors_file
        Filename of the EEG sensors dataset.

    Returns
    -------
    FloatArray
        Normalized spatial projection; negative for cathodic stimulation (peak
        magnitude of 1.0 at the closest region centroid). Shape ``(N_regions,)``
        for a single electrode, or ``(n_electrodes, N_regions)`` for a montage,
        with each row independently normalized.
    """
    sensors = SensorsEEG.from_file(sensors_file)
    labels = [label.upper() for label in sensors.labels]

    def _gamma_one(electrode: str, spread: float) -> FloatArray:
        target = electrode.upper()
        if target not in labels:
            msg = f"Electrode {electrode} not found in sensors."
            raise ValueError(msg)
        electrode_loc = np.asarray(sensors.locations[labels.index(target)], dtype=np.float64)
        # Euclidean distance from electrode to all region centroids
        dists = np.linalg.norm(centres - electrode_loc, axis=1)
        # Gaussian falloff (negative for cathodic stimulation)
        gamma = -np.exp(-(dists**2) / (2.0 * spread**2))
        # Normalize so that the maximum absolute value is exactly 1.0
        return gamma / np.abs(gamma).max()

    if isinstance(target_electrode, str):
        if not isinstance(sigma, (int, float)):
            msg = "sigma must be a scalar when target_electrode is a single label."
            raise TypeError(msg)
        return _gamma_one(target_electrode, sigma)

    electrodes = list(target_electrode)
    sigmas = [sigma] * len(electrodes) if isinstance(sigma, (int, float)) else list(sigma)
    if len(sigmas) != len(electrodes):
        msg = f"sigma sequence length {len(sigmas)} must match {len(electrodes)} electrodes."
        raise ValueError(msg)
    return np.stack([_gamma_one(e, s) for e, s in zip(electrodes, sigmas, strict=True)])
