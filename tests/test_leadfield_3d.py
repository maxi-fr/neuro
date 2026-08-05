"""Unit tests for 3D leadfield matrix loading, projection, and Jansen-Rit integration."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from neuro.connectome import EXTRACEPHALIC_ELECTRODES_MM, Connectome
from neuro.jansen_rit import JansenRitDynamics, JansenRitParams, project_control
from neuro.roast import convert_roast_leadfield_to_npz


def _write_cellstr(mat: h5py.File, group: h5py.Group, name: str, strings: list[str]) -> None:
    """Write a list of strings the way MATLAB v7.3 stores a cell array of char vectors."""
    refs = []
    for i, text in enumerate(strings):
        ds = mat.create_dataset(f"#refs#/{name}{i}", data=np.array([ord(c) for c in text], dtype=np.uint16))
        refs.append(ds.ref)
    group.create_dataset(name, data=np.array(refs, dtype=h5py.ref_dtype))


def _write_fake_roast_mat(
    path: Path,
    leadfield: np.ndarray,
    channel_labels: list[str],
    roast_labels: list[str],
    region_labels: list[str],
    region_normals: np.ndarray,
    normals_frame: str = "mni_ras",
) -> None:
    """Write a v7.3-style MAT file mimicking generate_roast_leadfield_3d.m's output layout."""
    with h5py.File(path, "w") as mat:
        # MATLAB writes arrays transposed relative to their numpy shape.
        mat.create_dataset("leadfield_3d", data=leadfield.transpose(2, 1, 0))
        meta = mat.create_group("metadata")
        _write_cellstr(mat, meta, "channelLabels", channel_labels)
        _write_cellstr(mat, meta, "roastLabels", roast_labels)
        _write_cellstr(mat, meta, "regionLabels", region_labels)
        meta.create_dataset("regionNormals", data=region_normals.T)
        meta.create_dataset("normalsFrame", data=np.array([ord(c) for c in normals_frame], dtype=np.uint16))


def test_project_control_3d_leadfield_cortical_normal() -> None:
    """Test 3D leadfield projection with cortical normal reduction."""
    n_controls = 63
    n_nodes = 76
    rng = np.random.default_rng(42)

    leadfield_3d = rng.standard_normal((n_controls, n_nodes, 3))
    region_normals = rng.standard_normal((n_nodes, 3))
    # Normalize region_normals
    region_normals /= np.linalg.norm(region_normals, axis=1, keepdims=True)

    # Control input u summing to 0 (62 scalp + Ex8 return)
    u = np.zeros(n_controls)
    u[0] = 1.0
    u[-1] = -1.0

    u_node = project_control(
        u,
        leadfield_3d,
        n_controls=n_controls,
        reduction_method="cortical_normal",
        region_normals=region_normals,
    )

    assert u_node.shape == (n_nodes,)
    # Expected manual dot product
    e_node = leadfield_3d[0] - leadfield_3d[-1]
    expected = (e_node * region_normals).sum(axis=1)
    np.testing.assert_allclose(u_node, expected)


def test_project_control_3d_leadfield_magnitude() -> None:
    """Test 3D leadfield projection with magnitude reduction."""
    n_controls = 63
    n_nodes = 76
    rng = np.random.default_rng(123)

    leadfield_3d = rng.standard_normal((n_controls, n_nodes, 3))
    u = np.zeros(n_controls)
    u[5] = 2.0
    u[10] = -2.0

    u_node = project_control(
        u,
        leadfield_3d,
        n_controls=n_controls,
        reduction_method="magnitude",
    )

    assert u_node.shape == (n_nodes,)
    e_node = 2.0 * leadfield_3d[5] - 2.0 * leadfield_3d[10]
    expected = np.linalg.norm(e_node, axis=1)
    np.testing.assert_allclose(u_node, expected)


def test_jansen_rit_dynamics_with_3d_leadfield(tmp_path: Path) -> None:
    """Test Jansen-Rit dynamics with 3D leadfield configured Connectome."""
    n_controls = 63
    n_nodes = 76

    rng = np.random.default_rng(999)
    leadfield_3d = rng.standard_normal((n_controls, n_nodes, 3))
    region_normals = rng.standard_normal((n_nodes, 3))
    region_normals /= np.linalg.norm(region_normals, axis=1, keepdims=True)
    channel_labels = np.array([f"C{i}" for i in range(62)] + ["Ex8"], dtype=str)
    region_labels = np.array([f"R{i}" for i in range(76)], dtype=str)

    npz_file = tmp_path / "test_roast_leadfield_3d.npz"
    np.savez(
        npz_file,
        leadfield_3d=leadfield_3d,
        channel_labels=channel_labels,
        region_labels=region_labels,
        region_normals=region_normals,
    )

    conn = Connectome.from_config(
        {
            "speed": 50.0,
            "K": 0.5,
            "gamma_model": "roast_3d",
            "leadfield_path": str(npz_file),
            "reduction_method": "cortical_normal",
        }
    )

    assert conn.leadfield_3d is not None
    assert conn.leadfield_3d.shape == (63, 76, 3)
    assert len(conn.channel_labels) == 62
    assert conn.control_channel_labels is not None
    assert len(conn.control_channel_labels) == 63

    plant = JansenRitDynamics(
        dt=0.001,
        params=JansenRitParams(),
        conn=conn,
        seed=42,
    )

    assert plant.n_controls == 63

    # Test single step execution
    x0 = plant.x
    u = np.zeros(63)
    u[0] = 0.001
    u[-1] = -0.001

    dxdt = plant.dynamics(t=0.0, x=x0, u=u)
    assert dxdt.shape == (6, 76)
    assert not np.isnan(dxdt).any()


def test_converter_recovers_v73_metadata(tmp_path: Path) -> None:
    """The .m saves -v7.3, so the HDF5 path must decode labels rather than silently dropping them."""
    rng = np.random.default_rng(7)
    leadfield = rng.standard_normal((5, 4, 3))
    leadfield[-1] = 0.0  # return row is the zero reference
    normals = rng.standard_normal((4, 3))
    channels = ["CP5", "TP9", "TP10", "Fz", "Ex8"]
    roast = ["CP5", "TPP9", "TPP10", "Fz", "Ex8"]
    regions = ["lHC", "lAMYG", "rHC", "rAMYG"]

    mat_path = tmp_path / "roast_leadfield_3d.mat"
    npz_path = tmp_path / "roast_leadfield_3d.npz"
    _write_fake_roast_mat(mat_path, leadfield, channels, roast, regions, normals)

    convert_roast_leadfield_to_npz(mat_path, npz_path)

    with np.load(npz_path) as out:
        np.testing.assert_allclose(out["leadfield_3d"], leadfield)
        np.testing.assert_allclose(out["region_normals"], normals)
        assert list(out["channel_labels"]) == channels
        assert list(out["roast_labels"]) == roast
        assert list(out["region_labels"]) == regions


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


def test_connectome_npz_roundtrip_preserves_leadfield(tmp_path: Path) -> None:
    """to_npz/from_npz must carry the 3D leadfield, or a saved connectome silently loses its gamma."""
    rng = np.random.default_rng(11)
    leadfield_3d = rng.standard_normal((63, 76, 3))
    region_normals = rng.standard_normal((76, 3))
    region_normals /= np.linalg.norm(region_normals, axis=1, keepdims=True)
    channel_labels = np.array([f"C{i}" for i in range(62)] + ["Ex8"], dtype=str)
    region_labels = np.array([f"R{i}" for i in range(76)], dtype=str)

    lf_file = tmp_path / "lf.npz"
    np.savez(
        lf_file,
        leadfield_3d=leadfield_3d,
        channel_labels=channel_labels,
        region_labels=region_labels,
        region_normals=region_normals,
    )
    conn = Connectome.from_config(
        {"gamma_model": "roast_3d", "leadfield_path": str(lf_file), "reduction_method": "magnitude"}
    )

    conn_file = tmp_path / "conn.npz"
    conn.to_npz(conn_file)
    restored = Connectome.from_npz(conn_file)

    assert restored.leadfield_3d is not None
    np.testing.assert_allclose(restored.leadfield_3d, leadfield_3d)
    assert restored.region_normals is not None
    np.testing.assert_allclose(restored.region_normals, region_normals)
    assert restored.reduction_method == "magnitude"
    assert restored.control_channel_labels is not None
    assert list(restored.control_channel_labels) == list(channel_labels)


def test_connectome_npz_roundtrip_without_leadfield(tmp_path: Path) -> None:
    """A plain analytical connectome round-trips with the optional leadfield fields absent."""
    conn = Connectome.from_config({"target_electrode": "CP5"})
    path = tmp_path / "analytical.npz"
    conn.to_npz(path)
    restored = Connectome.from_npz(path)

    assert restored.leadfield_3d is None
    assert restored.control_channel_labels is None
    np.testing.assert_allclose(restored.gamma, conn.gamma)


def test_ex8_return_is_contralateral_to_cp5() -> None:
    """Ex8 is the ROAST cap's return; it must sit opposite CP5, unlike the older EX_NECK."""
    conn = Connectome.from_config({"target_electrode": ["CP5", "EX8"], "gamma_model": "analytical"})
    assert conn.gamma.shape == (2, len(conn.region_labels))

    cp5 = np.asarray(EXTRACEPHALIC_ELECTRODES_MM["EX8"])
    ex_neck = np.asarray(EXTRACEPHALIC_ELECTRODES_MM["EX_NECK"])
    # Connectome frame is (anterior, left, superior): CP5 is left, so the return must be right.
    assert cp5[1] < 0
    assert float(np.linalg.norm(cp5 - ex_neck)) > 100.0
