from __future__ import annotations

import numpy as np
import pytest

from neuro.connectome import Connectome
from neuro.geometry import centres_to_mni_ras

_N_REGIONS = 76
_EZ_PZ_REGIONS = ("lHC", "lPHC", "lAMYG", "lTCI", "lTCV")


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


def test_label_indices_round_trip(connectome: Connectome) -> None:
    """The label->index map agrees with the label array."""
    for label, idx in connectome.region_index.items():
        assert connectome.region_labels[idx] == label


def test_required_regions_present(connectome: Connectome) -> None:
    """All paper EZ/PZ regions are loaded."""
    assert all(region in connectome.region_index for region in _EZ_PZ_REGIONS)


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


def test_centres_to_mni_ras(connectome: Connectome) -> None:
    """The connectome (anterior, left, superior) frame maps to MNI RAS."""
    ras = centres_to_mni_ras(connectome.centres)
    assert ras.shape == connectome.centres.shape

    np.testing.assert_allclose(ras[:, 0], -connectome.centres[:, 1])
    np.testing.assert_allclose(ras[:, 1], connectome.centres[:, 0])
    np.testing.assert_allclose(ras[:, 2], connectome.centres[:, 2])

    left = np.array([str(label).startswith("l") for label in connectome.region_labels])
    assert ras[left, 0].mean() < 0.0, "left-hemisphere regions must land at negative RAS x"
    assert ras[~left, 0].mean() > 0.0, "right-hemisphere regions must land at positive RAS x"
