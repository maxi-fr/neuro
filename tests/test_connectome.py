import re
from pathlib import Path

import numpy as np
import pytest

from neuro.connectome import Connectome, centres_to_mni_ras, sensor_positions_mm

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


_SIMNIBS_MONTAGE = ("TP9", "CP5", "EX_NECK")


def _write_simnibs_npz(path: Path, region_labels: np.ndarray, labels: tuple[str, ...] = _SIMNIBS_MONTAGE) -> Path:
    """Write a synthetic SimNIBS leadfield NPZ, so the tests need no SimNIBS install.

    The last row is the return electrode, zero by the leadfield's reference convention.
    """
    rng = np.random.default_rng(0)
    phi = rng.normal(size=(len(labels), len(region_labels)))
    phi[-1] = 0.0
    np.savez(
        path,
        gamma_phi=phi,
        gamma_e_normal=2.0 * phi,
        electrode_labels=np.array(labels, dtype=np.str_),
        region_labels=region_labels,
    )
    return path


def test_simnibs_gamma_reads_precomputed_rows(connectome: Connectome, tmp_path: Path) -> None:
    """The ``simnibs`` model returns the file's rows in the montage's electrode order."""
    path = _write_simnibs_npz(tmp_path / "gamma.npz", connectome.region_labels)
    with np.load(path) as data:
        rows = data["gamma_phi"]

    conn = Connectome.from_config(
        {"target_electrode": list(_SIMNIBS_MONTAGE), "gamma_model": "simnibs", "leadfield_path": str(path)}
    )
    assert conn.gamma.shape == (3, _N_REGIONS)
    np.testing.assert_allclose(conn.gamma, rows)

    reordered = Connectome.from_config(
        {"target_electrode": ["CP5", "TP9"], "gamma_model": "simnibs", "leadfield_path": str(path)}
    )
    np.testing.assert_allclose(reordered.gamma, rows[[1, 0]])


def test_simnibs_return_row_is_zero(connectome: Connectome, tmp_path: Path) -> None:
    """The return electrode's row is zero, so a zero-sum montage sums only the stimulating rows.

    SimNIBS references every leadfield row against the return electrode, whose own current is
    implicit under ``sum(u) = 0``. Encoding that here pins the convention the generator writes.
    """
    path = _write_simnibs_npz(tmp_path / "gamma.npz", connectome.region_labels)
    conn = Connectome.from_config(
        {"target_electrode": list(_SIMNIBS_MONTAGE), "gamma_model": "simnibs", "leadfield_path": str(path)}
    )

    np.testing.assert_array_equal(conn.gamma[2], np.zeros(_N_REGIONS))
    drive = np.array([-0.5, -0.5, 1.0]) @ conn.gamma
    np.testing.assert_allclose(drive, -0.5 * (conn.gamma[0] + conn.gamma[1]))


def test_simnibs_quantity_selects_array(connectome: Connectome, tmp_path: Path) -> None:
    """``simnibs_quantity`` picks between the FEM potential and the cortical-normal E-field."""
    path = _write_simnibs_npz(tmp_path / "gamma.npz", connectome.region_labels)
    base = {"target_electrode": ["TP9", "EX_NECK"], "gamma_model": "simnibs", "leadfield_path": str(path)}

    phi = Connectome.from_config(base).gamma
    e_normal = Connectome.from_config({**base, "simnibs_quantity": "e_normal"}).gamma
    np.testing.assert_allclose(e_normal, 2.0 * phi)


def test_simnibs_scale_is_linear(connectome: Connectome, tmp_path: Path) -> None:
    """``simnibs_scale`` multiplies the leadfield, carrying its physical units into mV."""
    path = _write_simnibs_npz(tmp_path / "gamma.npz", connectome.region_labels)
    base = {"target_electrode": ["TP9", "EX_NECK"], "gamma_model": "simnibs", "leadfield_path": str(path)}

    unscaled = Connectome.from_config(base).gamma
    scaled = Connectome.from_config({**base, "simnibs_scale": 2.0}).gamma
    np.testing.assert_allclose(scaled, 2.0 * unscaled)


def test_simnibs_requires_leadfield_path() -> None:
    """``simnibs`` has no analytical fallback: without a precomputed file it must refuse."""
    with pytest.raises(ValueError, match="leadfield_path"):
        Connectome.from_config({"target_electrode": ["TP9"], "gamma_model": "simnibs"})


def test_simnibs_unknown_electrode_rejected(connectome: Connectome, tmp_path: Path) -> None:
    """An electrode the FEM was never run for cannot be synthesised from the file."""
    path = _write_simnibs_npz(tmp_path / "gamma.npz", connectome.region_labels)
    with pytest.raises(ValueError, match="not found"):
        Connectome.from_config(
            {"target_electrode": ["CP6", "EX_NECK"], "gamma_model": "simnibs", "leadfield_path": str(path)}
        )


def test_simnibs_region_labels_must_match(connectome: Connectome, tmp_path: Path) -> None:
    """A leadfield built against a different parcellation is rejected, not silently broadcast."""
    path = _write_simnibs_npz(tmp_path / "gamma.npz", connectome.region_labels[:-1])
    with pytest.raises(ValueError, match="do not match"):
        Connectome.from_config(
            {"target_electrode": ["TP9", "EX_NECK"], "gamma_model": "simnibs", "leadfield_path": str(path)}
        )


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


def test_kcl_zero_sum_non_cancellation() -> None:
    """Zero-sum tES current under analytical model creates strong differential push-pull fields."""
    electrodes = ["CP5", "T7"]
    u = np.array([1.0, -1.0])

    conn_analyt = Connectome.from_config({"target_electrode": electrodes, "gamma_model": "analytical"})

    field_analyt = u @ conn_analyt.gamma

    assert np.linalg.norm(field_analyt) > 0.0
    assert np.var(field_analyt) > 0.0


def test_field_gamma_peaks_near_electrodes() -> None:
    """For each of the 62 scalp electrodes, the region with the largest `|gamma|` must lie near that electrode under the field model."""
    conn = Connectome.from_config({})
    labels, positions = sensor_positions_mm()

    conn_field = Connectome.from_config(
        {"target_electrode": list(labels), "gamma_model": "field", "gamma_spread": 15.0}
    )

    peak_region = np.abs(conn_field.gamma).argmax(axis=1)
    dist = np.linalg.norm(conn.centres[None, :, :] - positions[:, None, :], axis=2)
    to_peak = dist[np.arange(len(positions)), peak_region]

    assert to_peak.mean() < 50.0, f"electrodes are {to_peak.mean():.0f} mm from their own field model peak"


def test_field_cathodal_montage_negative_drive() -> None:
    """A KCL-legal cathodal montage producing negative mean drive over the EZ under the field model."""
    conn = Connectome.from_config(
        {"target_electrode": ["TP9", "EX_NECK"], "gamma_model": "field", "gamma_spread": 15.0}
    )

    # The test injects cathodal current at TP9 and anodal at EX_NECK.
    u = np.array([-1.0, 1.0])
    drive = u @ conn.gamma
    ez = [conn.region_index[name] for name in ("lHC", "lPHC", "lAMYG")]
    assert drive[ez].mean() < 0.0, "cathodal current must produce negative drive over the EZ"


def test_signed_field_gamma_model() -> None:
    """The signed_field model (Yu 2024) computes non-zero drive and hyperpolarizes the region under a cathodal electrode."""
    conn = Connectome.from_config(
        {
            "target_electrode": ["TP9", "CP5", "EX_NECK"],
            "gamma_model": "signed_field",
            "gamma_spread": 15.0,
        }
    )
    assert conn.gamma.shape == (3, 76)
    assert np.any(conn.gamma != 0.0)

    conn_tp9 = Connectome.from_config(
        {
            "target_electrode": "TP9",
            "gamma_model": "signed_field",
            "gamma_spread": 15.0,
        }
    )
    drive = -1.0 * conn_tp9.gamma
    l_tci_idx = conn_tp9.region_index["lTCI"]
    assert drive[l_tci_idx] < 0.0, "cathodal current at TP9 must hyperpolarize region lTCI directly under electrode"
