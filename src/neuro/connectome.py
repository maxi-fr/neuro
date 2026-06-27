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

import dataclasses
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from neuro.config import parse_array

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

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

    def to_npz(self, path: str | Path) -> None:
        """Save the connectome to an NPZ file."""
        np.savez(
            path,
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
    def from_config(cls, config: dict[str, Any]) -> Connectome:  # noqa: C901, PLR0912
        """Instantiate the connectome from a configuration dictionary.

        Loads the base TVB connectome, applies any node/region/channel subsetting,
        and computes the tES spatial projection `gamma` if configured.
        """
        speed = float(config.get("speed", 50.0))
        conn = load_connectome(speed=speed)

        selected_regions = parse_array(config.get("selected_regions"))
        region_idx = None
        if selected_regions is not None:
            region_idx = np.array(selected_regions, dtype=np.int64)

        if region_idx is not None:
            conn = dataclasses.replace(
                conn,
                weights=conn.weights[np.ix_(region_idx, region_idx)],
                tract_lengths=conn.tract_lengths[np.ix_(region_idx, region_idx)],
                centres=conn.centres[region_idx],
                region_labels=conn.region_labels[region_idx],
                hemispheres=conn.hemispheres[region_idx],
                delays=conn.delays[np.ix_(region_idx, region_idx)],
                gain=conn.gain[:, region_idx],
                gamma=conn.gamma[..., region_idx],
                region_index={str(conn.region_labels[i]): idx for idx, i in enumerate(region_idx)},
            )

        selected_channels_cfg = parse_array(config.get("selected_channels"))
        if selected_channels_cfg is not None:
            channel_idx = []
            for ch in selected_channels_cfg:
                if isinstance(ch, str):
                    channel_idx.append(conn.channel_index[ch])
                else:
                    channel_idx.append(int(ch))
            channel_idx = np.array(channel_idx, dtype=np.int64)
            from dataclasses import replace  # noqa: PLC0415

            conn = replace(
                conn,
                gain=conn.gain[channel_idx, :],
                channel_labels=conn.channel_labels[channel_idx],
                channel_index={str(conn.channel_labels[i]): idx for idx, i in enumerate(channel_idx)},
            )

        target_electrode = parse_array(config.get("target_electrode"))
        if target_electrode is not None:
            if isinstance(target_electrode, (str, int, np.integer)):
                electrodes_list = [target_electrode]
            else:
                electrodes_list = list(target_electrode)

            resolved_electrodes = []
            for el in electrodes_list:
                if isinstance(el, (int, np.integer)):
                    resolved_electrodes.append(str(conn.channel_labels[int(el)]))
                else:
                    resolved_electrodes.append(str(el))

            gamma = compute_gamma(
                conn.centres,
                target_electrode=resolved_electrodes if len(resolved_electrodes) > 1 else resolved_electrodes[0],
                spread=float(config.get("gamma_spread", 20.0)),
            )
            from dataclasses import replace  # noqa: PLC0415

            conn = replace(conn, gamma=gamma)

        overrides = {}
        for field in dataclasses.fields(cls):
            if field.name in config:
                overrides[field.name] = parse_array(config[field.name])

        if overrides:
            from dataclasses import replace  # noqa: PLC0415

            conn = replace(conn, **overrides)

        return conn


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

    # The left-right axis is the one whose reflection best maps the symmetric
    # montage onto itself (smallest total nearest-neighbour distance).
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
    the projection columns of the vertices mapped to each region. The projection
    rows are first reordered through :func:`_mirror_partner_permutation` so each
    channel label carries its own-hemisphere lead field (TVB's sensor and
    projection files disagree on left-right convention).
    """
    sensors = SensorsEEG.from_file(_SENSORS_FILE)
    channel_labels = np.asarray(sensors.labels, dtype=np.str_)
    locations = np.asarray(sensors.locations, dtype=np.float64)
    partner = _mirror_partner_permutation(locations)

    proj = ProjectionSurfaceEEG.from_file(_PROJECTION_FILE, matlab_data_name="ProjectionMatrix")
    # Row k now carries labels[k]'s own-side lead field, not its contralateral one.
    surface_gain = np.asarray(proj.projection_data, dtype=np.float64)[partner]
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
        gamma=np.zeros(len(region_labels)),
    )


def compute_gamma(
    centres: FloatArray,
    target_electrode: str | Sequence[str] = "CP5",
    spread: float | Sequence[float] = 20.0,
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
    spread
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
