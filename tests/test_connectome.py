import re

import numpy as np
import pytest

from neuro.connectome import Connectome
from neuro.geometry import centres_to_mni_ras, sensor_positions_mm

_N_REGIONS = 76
_N_CHANNELS = 62
_EZ_PZ_REGIONS = ("lHC", "lPHC", "lAMYG", "lTCI", "lTCV")
_NAMED_CHANNELS = ("CP5", "CP6", "PO3", "P1", "P3", "F3", "F5", "AF3", "O1")


@pytest.fixture(scope="module")
def connectome() -> Connectome:
    """Load the TVB connectome once for the whole module."""
    return Connectome.from_config({})


def test_structural_array_shapes(connectome: Connectome) -> None:
    """Weights/tract-lengths are square per region; centres are 3-D points."""
    assert connectome.weights.shape == (_N_REGIONS, _N_REGIONS)
    assert connectome.tract_lengths.shape == (_N_REGIONS, _N_REGIONS)
    assert connectome.centres.shape == (_N_REGIONS, 3)
    assert connectome.region_labels.shape == (_N_REGIONS,)


def test_eeg_gain_shape_and_finite(connectome: Connectome) -> None:
    """L maps 76 regions to 62 channels and contains no NaN/inf."""
    assert connectome.gain.shape == (_N_CHANNELS, _N_REGIONS)
    assert np.isfinite(connectome.gain).all()
    assert connectome.channel_labels.shape == (_N_CHANNELS,)


def test_label_indices_round_trip(connectome: Connectome) -> None:
    """The label->index maps agree with the label arrays."""
    for label, idx in connectome.region_index.items():
        assert connectome.region_labels[idx] == label
    for label, idx in connectome.channel_index.items():
        assert connectome.channel_labels[idx] == label


def test_required_regions_and_channels_present(connectome: Connectome) -> None:
    """All paper EZ/PZ regions and named EEG channels are loaded."""
    assert all(region in connectome.region_index for region in _EZ_PZ_REGIONS)
    assert all(channel in connectome.channel_index for channel in _NAMED_CHANNELS)


def _channel_side(label: str) -> str:
    """Hemisphere of a 10-20 channel from its number (odd=left, even=right)."""
    match = re.search(r"(\d+)", label)
    if match is None:
        return "z"
    return "L" if int(match.group(1)) % 2 == 1 else "R"


def test_eeg_gain_is_ipsilateral(connectome: Connectome) -> None:
    """Lateral cortical regions project most strongly to same-side EEG channels.

    Guards against the left-right mirror between TVB's sensor and projection
    files (see :func:`neuro.connectome._mirror_partner_permutation`).
    """
    for region, side in (("lTCV", "L"), ("rTCV", "R"), ("lTCI", "L"), ("rTCI", "R")):
        column = np.abs(connectome.gain[:, connectome.region_index[region]])
        top_channel = str(connectome.channel_labels[int(np.argmax(column))])
        assert _channel_side(top_channel) == side, f"{region} -> {top_channel}"


def test_delays_in_millisecond_range(connectome: Connectome) -> None:
    """Delays are nonnegative ms (not seconds), with a zero diagonal."""
    delays = connectome.delays
    np.testing.assert_array_equal(np.diag(delays), np.zeros(_N_REGIONS))
    off_diagonal = delays[~np.eye(_N_REGIONS, dtype=bool)]
    assert off_diagonal.max() < 200.0
    assert off_diagonal[off_diagonal > 0].size > 0


def test_weights_nonnegative(connectome: Connectome) -> None:
    """Connection weights are nonnegative."""
    assert (connectome.weights >= 0).all()


def test_hemispheres_split(connectome: Connectome) -> None:
    """The hemisphere mask splits the regions into two non-empty groups."""
    right = int(connectome.hemispheres.sum())
    left = int((~connectome.hemispheres).sum())
    assert right > 0
    assert left > 0
    assert right + left == _N_REGIONS


def test_sensor_positions_land_on_their_own_lead_field() -> None:
    """Placed electrodes sit near the regions their EEG lead field is strongest on.

    The EEG gain is an independent witness of where a channel is: a scalp electrode's largest
    lead-field entry must belong to a region physically close to it. This pins down the frame
    change and the scalp scaling in :func:`sensor_positions_mm` -- the raw TVB sensor file is
    unit vectors in a permuted frame, and differencing those against ``centres`` directly puts
    every electrode at the head's centre (all gamma rows then collapse onto each other, leaving
    a KCL-legal montage with no control authority at all).
    """
    conn = Connectome.from_config({})
    labels, positions = sensor_positions_mm()
    np.testing.assert_array_equal(labels, conn.channel_labels)

    dist = np.linalg.norm(conn.centres[None, :, :] - positions[:, None, :], axis=2)
    peak_region = np.abs(conn.gain).argmax(axis=1)
    to_peak = dist[np.arange(len(positions)), peak_region]
    assert to_peak.mean() < 50.0, f"electrodes are {to_peak.mean():.0f} mm from their own lead-field peak"

    left_region = np.array([str(label).startswith("l") for label in conn.region_labels])
    for channel in ("CP5", "T7", "TP9", "O1"):
        assert left_region[dist[conn.channel_index[channel]].argmin()], f"{channel} is not on the left"


def test_centres_to_mni_ras(connectome: Connectome) -> None:
    """The connectome (anterior, left, superior) frame maps to MNI RAS.

    Every external head model is reached through this one conversion, and a left-right flip
    here is invisible downstream -- the regions still exist, the fields still look plausible,
    and every number is wrong. The hemisphere check is what catches it.
    """
    ras = centres_to_mni_ras(connectome.centres)
    assert ras.shape == connectome.centres.shape

    np.testing.assert_allclose(ras[:, 0], -connectome.centres[:, 1])
    np.testing.assert_allclose(ras[:, 1], connectome.centres[:, 0])
    np.testing.assert_allclose(ras[:, 2], connectome.centres[:, 2])

    left = np.array([str(label).startswith("l") for label in connectome.region_labels])
    assert ras[left, 0].mean() < 0.0, "left-hemisphere regions must land at negative RAS x"
    assert ras[~left, 0].mean() > 0.0, "right-hemisphere regions must land at positive RAS x"
