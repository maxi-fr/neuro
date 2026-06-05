"""Stage 0 regression harness for :func:`neuro.connectome.load_connectome`.

Snapshots the structural-data invariants every later stage relies on: array
shapes, the EEG gain ``L``, delay sanity, and the presence of the paper's EZ/PZ
regions and named EEG channels.
"""

import numpy as np
import pytest

from neuro.connectome import Connectome, load_connectome

_N_REGIONS = 76
_N_CHANNELS = 62
_EZ_PZ_REGIONS = ("lHC", "lPHC", "lAMYG", "lTCI", "lTCV")
_NAMED_CHANNELS = ("CP5", "CP6", "PO3", "P1", "P3", "F3", "F5", "AF3", "O1")


@pytest.fixture(scope="module")
def connectome() -> Connectome:
    """Load the TVB connectome once for the whole module."""
    return load_connectome()


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
