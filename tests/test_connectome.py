"""Stage 0 regression harness for :meth:`neuro.connectome.Connectome.from_config`.

Snapshots the structural-data invariants every later stage relies on: array
shapes, the EEG gain ``L``, delay sanity, and the presence of the paper's EZ/PZ
regions and named EEG channels.
"""

import re
from pathlib import Path

import numpy as np
import pytest

from neuro.connectome import Connectome

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
    assert off_diagonal.max() < 200.0  # ms, not seconds
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


def test_npz_round_trip(connectome: Connectome, tmp_path: Path) -> None:
    """The connectome can be saved to and loaded from an NPZ file."""
    path = tmp_path / "connectome.npz"
    connectome.to_npz(path)
    loaded = Connectome.from_npz(path)

    assert loaded.K == connectome.K
    np.testing.assert_array_equal(loaded.weights, connectome.weights)
    np.testing.assert_array_equal(loaded.tract_lengths, connectome.tract_lengths)
    np.testing.assert_array_equal(loaded.centres, connectome.centres)
    np.testing.assert_array_equal(loaded.region_labels, connectome.region_labels)
    np.testing.assert_array_equal(loaded.hemispheres, connectome.hemispheres)
    assert loaded.speed == connectome.speed
    np.testing.assert_array_equal(loaded.delays, connectome.delays)
    np.testing.assert_array_equal(loaded.gain, connectome.gain)
    np.testing.assert_array_equal(loaded.channel_labels, connectome.channel_labels)
    assert loaded.region_index == connectome.region_index
    assert loaded.channel_index == connectome.channel_index
    np.testing.assert_array_equal(loaded.gamma, connectome.gamma)


def test_from_config_target_electrode_builds_gamma() -> None:
    """A ``target_electrode`` yields a normalized, positive unit-peak gamma kernel.

    The kernel is polarity-agnostic (the applied current's sign sets cathode vs anode), so it
    is strictly positive with a peak of 1.0 -- not signed.
    """
    conn = Connectome.from_config({"target_electrode": "CP5", "gamma_spread": 20.0})
    assert conn.gamma.shape == (_N_REGIONS,)
    assert np.isclose(conn.gamma.max(), 1.0)
    assert conn.gamma.min() >= 0.0


def test_from_config_default_gamma_is_zero() -> None:
    """Without a ``target_electrode`` the projection is all-zero (open loop)."""
    conn = Connectome.from_config({})
    np.testing.assert_array_equal(conn.gamma, np.zeros(_N_REGIONS))
