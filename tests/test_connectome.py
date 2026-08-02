import re
from pathlib import Path

import numpy as np
import pytest

from neuro.connectome import Connectome, sensor_positions_mm

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
    """A ``target_electrode`` yields a Coulomb kernel that decays with distance from it.

    The kernel is a potential per unit current, so it is positive and largest at the region
    nearest the electrode; the applied current's sign is what sets cathode versus anode.
    """
    conn = Connectome.from_config({"target_electrode": "CP5", "gamma_spread": 15.0})
    assert conn.gamma.shape == (_N_REGIONS,)
    assert (conn.gamma > 0.0).all()

    _, positions = sensor_positions_mm()
    dist = np.linalg.norm(conn.centres - positions[conn.channel_index["CP5"]], axis=1)
    assert conn.gamma.argmax() == dist.argmin()


def test_from_config_default_gamma_is_zero() -> None:
    """Without a ``target_electrode`` the projection is all-zero (open loop)."""
    conn = Connectome.from_config({})
    np.testing.assert_array_equal(conn.gamma, np.zeros(_N_REGIONS))


def test_from_config_reciprocal_gamma_model() -> None:
    """Option A (reciprocal) builds a *signed* current-to-field projection via Helmholtz reciprocity.

    Reciprocity equates the field a montage injects with the leadfield's recording sensitivity,
    which is signed: a region's contribution to a channel flips with the orientation of its
    source. Rectifying the row would throw that away, and with it the cathodal/anodal
    asymmetry that is the entire suppression mechanism.
    """
    conn = Connectome.from_config({"target_electrode": ["CP5", "T7", "F9"], "gamma_model": "reciprocal"})
    assert conn.gamma.shape == (3, _N_REGIONS)
    assert (conn.gamma < 0.0).any()
    assert not np.allclose(conn.gamma[0], conn.gamma[1])


def test_from_config_analytical_gamma_model() -> None:
    """Option B (analytical) builds Laplacian dipolar Coulomb projection without independent peak clipping."""
    conn = Connectome.from_config(
        {"target_electrode": ["CP5", "T7"], "gamma_model": "analytical", "gamma_spread": 15.0}
    )
    assert conn.gamma.shape == (2, _N_REGIONS)
    assert (conn.gamma > 0.0).all()


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


def test_gamma_rows_are_distinguishable_per_electrode() -> None:
    """Nearby electrodes still steer distinct fields, so a zero-sum montage keeps authority.

    ``sum(u) = 0`` (Kirchhoff) means only *differences* between gamma rows can drive the
    network, so near-collinear rows leave the controller powerless. Their pairwise cosine is
    the direct measure of that.
    """
    conn = Connectome.from_config(
        {"target_electrode": ["TP9", "CP6"], "gamma_model": "analytical", "gamma_spread": 15.0}
    )
    rows = conn.gamma / np.linalg.norm(conn.gamma, axis=1, keepdims=True)
    assert float(rows[0] @ rows[1]) < 0.95

    u = np.array([-1.0, 1.0])
    drive = u @ conn.gamma
    ez = [conn.region_index[name] for name in ("lHC", "lPHC", "lAMYG")]
    assert drive[ez].mean() < -0.1, "cathodal current must hyperpolarize the epileptogenic zone"


def test_extracephalic_return_drives_no_region_anodally() -> None:
    """An extracephalic return keeps a cathodal montage cathodal *everywhere*.

    Kirchhoff forces the injected current back out somewhere, and wherever it leaves the drive
    is anodal -- seizure-promoting. That is the real cost of ``sum(u) = 0`` with a scalp
    return, and it is what an off-head return buys out: ``EX_NECK`` is 145+ mm from every
    region, so its Coulomb potential there is small and near-uniform and no region ever
    crosses into positive drive.
    """
    conn = Connectome.from_config({"target_electrode": ["TP9", "EX_NECK"], "gamma_model": "analytical"})
    assert conn.gamma.shape == (2, _N_REGIONS)

    ex = conn.gamma[1]
    assert np.ptp(ex) / ex.mean() < 0.75, "extracephalic kernel is not near-uniform across regions"

    drive = np.array([-1.0, 1.0]) @ conn.gamma
    assert (drive < 0.0).all(), "an extracephalic return must leave no region anodally driven"

    ez = [conn.region_index[name] for name in ("lHC", "lPHC", "lAMYG")]
    assert drive[ez].mean() < drive.mean(), "the focus must be driven harder than the network mean"


def test_extracephalic_electrode_rejected_by_reciprocal() -> None:
    """``reciprocal`` reads leadfield rows, so it has nothing to say about an off-head electrode."""
    with pytest.raises(ValueError, match="analytical"):
        Connectome.from_config({"target_electrode": ["TP9", "EX_NECK"], "gamma_model": "reciprocal"})


def test_kcl_zero_sum_non_cancellation() -> None:
    """Zero-sum tES current under reciprocal or analytical model creates strong differential push-pull fields."""
    electrodes = ["CP5", "T7"]
    u = np.array([1.0, -1.0])

    conn_recip = Connectome.from_config({"target_electrode": electrodes, "gamma_model": "reciprocal"})
    conn_analyt = Connectome.from_config({"target_electrode": electrodes, "gamma_model": "analytical"})

    field_recip = u @ conn_recip.gamma
    field_analyt = u @ conn_analyt.gamma

    assert np.linalg.norm(field_recip) > 0.0
    assert np.linalg.norm(field_analyt) > 0.0
    assert np.var(field_analyt) > 0.0
