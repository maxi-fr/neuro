from __future__ import annotations

from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError

from neuro.connectome import Connectome
from neuro.geometry import EXTRACEPHALIC_ELECTRODES_MM, sensor_positions_mm
from neuro.jansen_rit import JansenRitDynamics, JansenRitParams
from neuro.stimulation import AnalyticalStim, DynamicYuStim, NullStim, Roast3DStim, build_stimulation
from neuro.stimulation.base import StimulationConfig
from neuro.stimulation.roast_io import convert_roast_leadfield_to_npz

if TYPE_CHECKING:
    from pathlib import Path

    from neuro.types import StrArray

_N_REGIONS = 76


@pytest.fixture(scope="module")
def connectome() -> Connectome:
    """Load the TVB connectome once for the whole module."""
    return Connectome.from_config({})


_STIM_ADAPTER = TypeAdapter(StimulationConfig)


def _build(cfg: dict[str, object], conn: Connectome) -> Any:  # noqa: ANN401
    """Validate a raw stimulation config through the discriminated union and build it."""
    return build_stimulation(_STIM_ADAPTER.validate_python(cfg), conn)


# --------------------------------------------------------------------------------------
# config schema
# --------------------------------------------------------------------------------------


def test_config_rejects_foreign_model_keys(connectome: Connectome) -> None:
    """A key belonging to another model must fail loudly instead of being ignored."""
    with pytest.raises(ValidationError, match="spread"):
        _build({"model": "roast_3d", "spread": 15.0}, connectome)


def test_config_rejects_single_electrode_montage(connectome: Connectome) -> None:
    """One electrode can never satisfy Kirchhoff, so it is invalid by construction."""
    with pytest.raises(ValidationError, match="at least 2"):
        _build({"model": "analytical", "electrodes": ["CP5"]}, connectome)


def test_config_rejects_mismatched_spread_list(connectome: Connectome) -> None:
    """A per-electrode spread list must have one entry per electrode."""
    with pytest.raises(ValidationError, match="spread has 3 entries"):
        _build({"model": "analytical", "electrodes": ["CP5", "T7"], "spread": [1.0, 2.0, 3.0]}, connectome)


# --------------------------------------------------------------------------------------
# none
# --------------------------------------------------------------------------------------


def test_null_stim_drives_nothing(connectome: Connectome) -> None:
    """The default model keeps one control electrode whose drive is always zero."""
    stim = _build({"model": "none"}, connectome)
    assert isinstance(stim, NullStim)
    assert stim.n_controls == 1
    np.testing.assert_array_equal(stim.project(np.array([1.0])), np.zeros(_N_REGIONS))


def test_plant_without_stim_is_unstimulated(connectome: Connectome) -> None:
    """``stim=None`` reproduces the pre-refactor unstimulated plant, one input and no drive."""
    plant = JansenRitDynamics(dt=1e-4, params=JansenRitParams(), conn=connectome, seed=7)
    assert plant.n_controls == 1
    assert plant.n_inputs == 1
    out = plant.dynamics(0.0, plant.x, np.array([0.0]))
    assert out.shape == (6, _N_REGIONS)
    assert np.isfinite(out).all()


# --------------------------------------------------------------------------------------
# analytical
# --------------------------------------------------------------------------------------


def test_analytical_kernel_decays_from_its_electrode(connectome: Connectome) -> None:
    """Each row is a Coulomb potential per unit current: positive, and peaking at its electrode.

    The row order is load-bearing -- ``u[i]`` is the current at ``electrodes[i]`` -- so each row's
    argmax is checked against the region nearest that electrode, not just against row 0's.
    """
    stim = _build({"model": "analytical", "electrodes": ["CP5", "T7"], "spread": 15.0}, connectome)
    assert isinstance(stim, AnalyticalStim)
    assert stim.gamma.shape == (2, _N_REGIONS)
    assert (stim.gamma > 0.0).all()

    _, positions = sensor_positions_mm()
    for row, channel in enumerate(("CP5", "T7")):
        dist = np.linalg.norm(connectome.centres - positions[connectome.channel_index[channel]], axis=1)
        assert stim.gamma[row].argmax() == dist.argmin(), channel


def test_analytical_rows_are_distinguishable_per_electrode(connectome: Connectome) -> None:
    """Nearby electrodes still steer distinct fields, so a zero-sum montage keeps authority.

    ``sum(u) = 0`` (Kirchhoff) means only *differences* between gamma rows can drive the
    network, so near-collinear rows leave the controller powerless. Their pairwise cosine is
    the direct measure of that.
    """
    stim = _build({"model": "analytical", "electrodes": ["TP9", "CP6"], "spread": 15.0}, connectome)
    rows = stim.gamma / np.linalg.norm(stim.gamma, axis=1, keepdims=True)
    assert float(rows[0] @ rows[1]) < 0.95

    drive = stim.project(np.array([-1.0, 1.0]))
    ez = [connectome.region_index[name] for name in ("lHC", "lPHC", "lAMYG")]
    assert drive[ez].mean() < -0.1, "cathodal current must hyperpolarize the epileptogenic zone"


def test_extracephalic_return_drives_no_region_anodally(connectome: Connectome) -> None:
    """An extracephalic return keeps a cathodal montage cathodal *everywhere*.

    Kirchhoff forces the injected current back out somewhere, and wherever it leaves the drive
    is anodal -- seizure-promoting. That is the real cost of ``sum(u) = 0`` with a scalp
    return, and it is what an off-head return buys out: ``EX_NECK`` is 145+ mm from every
    region, so its Coulomb potential there is small and near-uniform and no region ever
    crosses into positive drive.
    """
    stim = _build({"model": "analytical", "electrodes": ["TP9", "EX_NECK"]}, connectome)
    assert stim.gamma.shape == (2, _N_REGIONS)

    ex = stim.gamma[1]
    assert np.ptp(ex) / ex.mean() < 0.75, "extracephalic kernel is not near-uniform across regions"

    drive = stim.project(np.array([-1.0, 1.0]))
    assert (drive < 0.0).all(), "an extracephalic return must leave no region anodally driven"

    ez = [connectome.region_index[name] for name in ("lHC", "lPHC", "lAMYG")]
    assert drive[ez].mean() < drive.mean(), "the focus must be driven harder than the network mean"


def test_kcl_zero_sum_non_cancellation(connectome: Connectome) -> None:
    """Zero-sum tES current under the analytical model creates a differential push-pull field."""
    stim = _build({"model": "analytical", "electrodes": ["CP5", "T7"]}, connectome)
    field = stim.project(np.array([1.0, -1.0]))

    assert np.linalg.norm(field) > 0.0
    assert np.var(field) > 0.0


def test_ex8_return_is_contralateral_to_cp5(connectome: Connectome) -> None:
    """Ex8 is the ROAST cap's return; it must sit opposite CP5, unlike the older EX_NECK."""
    stim = _build({"model": "analytical", "electrodes": ["CP5", "EX8"]}, connectome)
    assert stim.gamma.shape == (2, _N_REGIONS)

    ex8 = np.asarray(EXTRACEPHALIC_ELECTRODES_MM["EX8"])
    ex_neck = np.asarray(EXTRACEPHALIC_ELECTRODES_MM["EX_NECK"])
    # Connectome frame is (anterior, left, superior): CP5 is left, so the return must be right.
    assert ex8[1] < 0
    assert float(np.linalg.norm(ex8 - ex_neck)) > 100.0


def test_zero_current_short_circuits(connectome: Connectome) -> None:
    """Most steps of a threshold-controlled run carry no current, so zero must skip the matmul."""
    stim = _build({"model": "analytical", "electrodes": ["CP5", "T7"]}, connectome)
    np.testing.assert_array_equal(stim.project(np.zeros(2)), np.zeros(_N_REGIONS))


# --------------------------------------------------------------------------------------
# roast_3d
# --------------------------------------------------------------------------------------


def _write_leadfield_npz(
    path: Path, region_labels: StrArray, *, normals_zero: bool = False, legacy_key: bool = False
) -> None:
    """Write a synthetic ROAST leadfield NPZ over the given region ordering."""
    rng = np.random.default_rng(999)
    n_nodes = len(region_labels)
    normals = np.zeros((n_nodes, 3)) if normals_zero else rng.standard_normal((n_nodes, 3))
    if not normals_zero:
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    lf_data = rng.standard_normal((63, n_nodes, 3))
    ch_labels = np.array([f"C{i}" for i in range(62)] + ["Ex8"], dtype=str)
    r_labels = np.asarray(region_labels, dtype=str)
    if legacy_key:
        np.savez(path, leadfield_3d=lf_data, channel_labels=ch_labels, region_labels=r_labels, region_normals=normals)
    else:
        np.savez(path, leadfield_E=lf_data, channel_labels=ch_labels, region_labels=r_labels, region_normals=normals)



def test_roast_3d_reduces_along_the_cortical_normal(connectome: Connectome, tmp_path: Path) -> None:
    """The drive is ``lambda * (E . n)`` on the superposed field, checked against the raw file.

    Precomputing ``lambda * (E_k . n)`` per electrode is only legitimate because the dot product
    commutes with the superposition over electrodes; this pins the identity down against the
    3-vectors on disk rather than against the matrix the model built.
    """
    npz = tmp_path / "lf.npz"
    _write_leadfield_npz(npz, connectome.region_labels)
    stim = _build({"model": "roast_3d", "leadfield_path": str(npz)}, connectome)
    assert isinstance(stim, Roast3DStim)
    assert stim.n_controls == 63

    u = np.zeros(63)
    u[0], u[5], u[-1] = 1.0, 2.0, -3.0
    with np.load(npz) as raw:
        leadfield = raw["leadfield_E"] if "leadfield_E" in raw else raw["leadfield_3d"]
        normals = raw["region_normals"]
    e_node = np.tensordot(u, leadfield, axes=(0, 0))
    expected = Roast3DStim.polarization_length_mm * (e_node * normals).sum(axis=1)
    np.testing.assert_allclose(stim.project(u), expected)


def test_roast_3d_loads_legacy_leadfield_3d_key(connectome: Connectome, tmp_path: Path) -> None:
    """Legacy leadfield_3d NPZ files must load gracefully via the fallback path."""
    npz = tmp_path / "legacy.npz"
    _write_leadfield_npz(npz, connectome.region_labels, legacy_key=True)
    stim = _build({"model": "roast_3d", "leadfield_path": str(npz)}, connectome)
    assert isinstance(stim, Roast3DStim)
    assert stim.n_controls == 63


def test_roast_3d_projection_is_linear_and_signed(connectome: Connectome, tmp_path: Path) -> None:
    """Reversing the montage reverses the drive -- the property ``magnitude`` could not express."""
    npz = tmp_path / "lf.npz"
    _write_leadfield_npz(npz, connectome.region_labels)
    stim = _build({"model": "roast_3d", "leadfield_path": str(npz)}, connectome)

    u = np.zeros(63)
    u[0], u[-1] = 1.0, -1.0
    np.testing.assert_allclose(stim.project(-u), -stim.project(u))
    np.testing.assert_allclose(stim.project(2.0 * u), 2.0 * stim.project(u))


def test_roast_3d_electrodes_subset_holds_the_control_problem_fixed(connectome: Connectome, tmp_path: Path) -> None:
    """Selecting a montage keeps swapping the model a physics change, not a 3->63 input change."""
    npz = tmp_path / "lf.npz"
    _write_leadfield_npz(npz, connectome.region_labels)
    stim = _build({"model": "roast_3d", "leadfield_path": str(npz), "electrodes": ["C3", "C1", "Ex8"]}, connectome)

    assert stim.n_controls == 3
    assert list(stim.control_labels) == ["C3", "C1", "Ex8"]
    assert stim.project(np.array([1.0, -1.0, 0.0])).shape == (_N_REGIONS,)


def test_roast_3d_rejects_permuted_region_order(connectome: Connectome, tmp_path: Path) -> None:
    """A file-backed projection is applied positionally, so a permuted order is silent corruption."""
    npz = tmp_path / "lf.npz"
    _write_leadfield_npz(npz, connectome.region_labels[::-1])
    with pytest.raises(ValueError, match="region_labels do not match"):
        _build({"model": "roast_3d", "leadfield_path": str(npz)}, connectome)


def test_roast_3d_rejects_zero_normals(connectome: Connectome, tmp_path: Path) -> None:
    """The leadfield and its normals are one artefact from one run; there is no fallback file."""
    npz = tmp_path / "lf.npz"
    _write_leadfield_npz(npz, connectome.region_labels, normals_zero=True)
    with pytest.raises(ValueError, match="no usable region_normals"):
        _build({"model": "roast_3d", "leadfield_path": str(npz)}, connectome)


def test_roast_3d_drives_the_plant(connectome: Connectome, tmp_path: Path) -> None:
    """A 63-input leadfield plant steps to a finite state."""
    npz = tmp_path / "lf.npz"
    _write_leadfield_npz(npz, connectome.region_labels)
    stim = _build({"model": "roast_3d", "leadfield_path": str(npz)}, connectome)
    plant = JansenRitDynamics(dt=1e-3, params=JansenRitParams(), conn=connectome, stim=stim, seed=42)

    assert plant.n_controls == 63
    u = np.zeros(63)
    u[0], u[-1] = 0.001, -0.001
    out = plant.dynamics(t=0.0, x=plant.x, u=u)
    assert out.shape == (6, _N_REGIONS)
    assert not np.isnan(out).any()


# --------------------------------------------------------------------------------------
# MAT -> NPZ conversion
# --------------------------------------------------------------------------------------


def _write_cellstr(mat: h5py.File, group: h5py.Group, name: str, strings: list[str]) -> None:
    """Write a list of strings the way MATLAB v7.3 stores a cell array of char vectors."""
    refs = []
    for i, text in enumerate(strings):
        ds = mat.create_dataset(f"#refs#/{name}{i}", data=np.array([ord(c) for c in text], dtype=np.uint16))
        refs.append(ds.ref)
    group.create_dataset(name, data=np.array(refs, dtype=h5py.ref_dtype))


def _write_fake_roast_mat(
    path: Path,
    leadfield_e: np.ndarray,
    channel_labels: list[str],
    roast_labels: list[str],
    region_labels: list[str],
    region_normals: np.ndarray,
    *,
    normals_frame: str = "mni_ras",
    leadfield_v: np.ndarray | None = None,
    legacy_key: bool = False,
) -> None:
    """Write a v7.3-style MAT file mimicking generate_roast_leadfield_3d.m's output layout."""
    with h5py.File(path, "w") as mat:
        # MATLAB writes arrays transposed relative to their numpy shape.
        key = "leadfield_3d" if legacy_key else "leadfield_E"
        mat.create_dataset(key, data=leadfield_e.transpose(2, 1, 0))
        if leadfield_v is not None:
            mat.create_dataset("leadfield_V", data=leadfield_v.T)
        meta = mat.create_group("metadata")
        _write_cellstr(mat, meta, "channelLabels", channel_labels)
        _write_cellstr(mat, meta, "roastLabels", roast_labels)
        _write_cellstr(mat, meta, "regionLabels", region_labels)
        meta.create_dataset("regionNormals", data=region_normals.T)
        meta.create_dataset("normalsFrame", data=np.array([ord(c) for c in normals_frame], dtype=np.uint16))


def test_converter_recovers_v73_metadata(tmp_path: Path) -> None:
    """The .m saves -v7.3, so the HDF5 path must decode labels rather than silently dropping them."""
    rng = np.random.default_rng(7)
    leadfield_e = rng.standard_normal((5, 4, 3))
    leadfield_e[-1] = 0.0  # return row is the zero reference
    leadfield_v = rng.standard_normal((5, 4))
    leadfield_v[-1] = 0.0
    normals = rng.standard_normal((4, 3))
    channels = ["CP5", "TP9", "TP10", "Fz", "Ex8"]
    roast = ["CP5", "TPP9", "TPP10", "Fz", "Ex8"]
    regions = ["lHC", "lAMYG", "rHC", "rAMYG"]

    mat_path = tmp_path / "roast_leadfield_3d.mat"
    npz_path = tmp_path / "roast_leadfield_3d.npz"
    _write_fake_roast_mat(mat_path, leadfield_e, channels, roast, regions, normals, leadfield_v=leadfield_v)

    convert_roast_leadfield_to_npz(mat_path, npz_path)

    with np.load(npz_path) as out:
        np.testing.assert_allclose(out["leadfield_E"], leadfield_e)
        np.testing.assert_allclose(out["leadfield_V"], leadfield_v)
        np.testing.assert_allclose(out["region_normals"], normals)
        assert list(out["channel_labels"]) == channels
        assert list(out["roast_labels"]) == roast
        assert list(out["region_labels"]) == regions


def test_converter_supports_legacy_leadfield_3d_mat_key(tmp_path: Path) -> None:
    """MAT files carrying the legacy leadfield_3d key must convert properly."""
    rng = np.random.default_rng(123)
    leadfield_e = rng.standard_normal((3, 2, 3))
    leadfield_e[-1] = 0.0
    normals = rng.standard_normal((2, 3))
    mat_path = tmp_path / "legacy_roast.mat"
    npz_path = tmp_path / "legacy_roast.npz"
    _write_fake_roast_mat(
        mat_path, leadfield_e, ["C1", "C2", "Ex8"], ["C1", "C2", "Ex8"], ["r1", "r2"], normals, legacy_key=True
    )

    convert_roast_leadfield_to_npz(mat_path, npz_path)

    with np.load(npz_path) as out:
        np.testing.assert_allclose(out["leadfield_E"], leadfield_e)


def test_converter_rejects_connectome_frame_normals(tmp_path: Path) -> None:
    """Normals in the connectome frame are rotated 90 deg from the leadfield's MNI RAS E-vectors."""
    leadfield = np.zeros((2, 2, 3))
    mat_path = tmp_path / "bad_frame.mat"
    _write_fake_roast_mat(
        mat_path, leadfield, ["Fz", "Ex8"], ["Fz", "Ex8"], ["a", "b"], np.zeros((2, 3)), normals_frame="conn"
    )

    with pytest.raises(ValueError, match="normals frame"):
        convert_roast_leadfield_to_npz(mat_path, tmp_path / "out.npz")


def test_converter_rejects_nonzero_return_row(tmp_path: Path) -> None:
    """The last channel is the reference ground; a nonzero row means the basis is not differential."""
    leadfield = np.ones((2, 2, 3))
    mat_path = tmp_path / "nonzero_return.mat"
    _write_fake_roast_mat(mat_path, leadfield, ["Fz", "Ex8"], ["Fz", "Ex8"], ["a", "b"], np.zeros((2, 3)))

    with pytest.raises(ValueError, match="return row"):
        convert_roast_leadfield_to_npz(mat_path, tmp_path / "out.npz")


# --------------------------------------------------------------------------------------
# yu_dynamic
# --------------------------------------------------------------------------------------



def _write_dynamic_leadfield_npz(
    path: Path, region_labels: StrArray, *, include_v: bool = True, legacy_key: bool = False
) -> None:
    """Write a synthetic ROAST leadfield NPZ carrying leadfield_E (or leadfield_3d) and leadfield_V."""
    rng = np.random.default_rng(42)
    n_nodes = len(region_labels)
    normals = rng.standard_normal((n_nodes, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    lf_data = rng.standard_normal((63, n_nodes, 3))
    v_data = rng.standard_normal((63, n_nodes))
    ch_labels = np.array([f"C{i}" for i in range(62)] + ["Ex8"], dtype=str)
    r_labels = np.asarray(region_labels, dtype=str)

    if legacy_key:
        if include_v:
            np.savez(
                path,
                leadfield_3d=lf_data,
                leadfield_V=v_data,
                channel_labels=ch_labels,
                region_labels=r_labels,
                region_normals=normals,
            )
        else:
            np.savez(
                path,
                leadfield_3d=lf_data,
                channel_labels=ch_labels,
                region_labels=r_labels,
                region_normals=normals,
            )
    elif include_v:
        np.savez(
            path,
            leadfield_E=lf_data,
            leadfield_V=v_data,
            channel_labels=ch_labels,
            region_labels=r_labels,
            region_normals=normals,
        )
    else:
        np.savez(
            path,
            leadfield_E=lf_data,
            channel_labels=ch_labels,
            region_labels=r_labels,
            region_normals=normals,
        )



def test_dynamic_yu_stim_project(connectome: Connectome, tmp_path: Path) -> None:
    """DynamicYuStim projects currents using vector E-field magnitude and smooth voltage polarity."""
    npz = tmp_path / "lf_dynamic.npz"
    _write_dynamic_leadfield_npz(npz, connectome.region_labels)

    stim = _build({"model": "yu_dynamic", "leadfield_path": str(npz), "alpha": 4.0}, connectome)
    assert isinstance(stim, DynamicYuStim)
    assert stim.n_controls == 63
    assert stim.alpha == 4.0

    u = np.zeros(63)
    u[0], u[-1] = 1.0, -1.0
    drive = stim.project(u)
    assert drive.shape == (_N_REGIONS,)
    assert np.isfinite(drive).all()

    # Scaling input current scales field magnitude linearly
    u2 = 2.0 * u
    drive2 = stim.project(u2)
    assert not np.allclose(drive2, drive)


def test_dynamic_yu_loads_legacy_leadfield_3d_key(connectome: Connectome, tmp_path: Path) -> None:
    """DynamicYuStim must load legacy leadfield_3d NPZ files gracefully."""
    npz = tmp_path / "lf_dynamic_legacy.npz"
    _write_dynamic_leadfield_npz(npz, connectome.region_labels, legacy_key=True)

    stim = _build({"model": "yu_dynamic", "leadfield_path": str(npz)}, connectome)
    assert isinstance(stim, DynamicYuStim)


def test_dynamic_yu_rejects_missing_leadfield_v(connectome: Connectome, tmp_path: Path) -> None:
    """Loading a leadfield without leadfield_V must fail with a clear error."""
    npz = tmp_path / "lf_no_v.npz"
    _write_dynamic_leadfield_npz(npz, connectome.region_labels, include_v=False)

    with pytest.raises(ValueError, match="does not carry 'leadfield_V'"):
        _build({"model": "yu_dynamic", "leadfield_path": str(npz)}, connectome)


