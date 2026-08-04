from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import simnibs
from scipy.spatial import cKDTree
from simnibs import sim_struct

_MA = 1e-3
_V_TO_MV = 1000.0
_WITNESS_MAX_MM = 50.0
_PHI_BALL_MM = 8.0
_MIN_NODES_PER_REGION = 10
_RETURN_X_HALFWIDTH_MM = 40.0
_CAP_COLUMNS = 5  # SimNIBS EEG cap CSV: Type,X,Y,Z,Name

# The calibration anchor of docs/tes_field_geometry.md section 6: the analytical model's mean drive
# over the EZ/PZ regions for -1 mA at the first stimulating electrode against the return.
_EZ_REGIONS = ("lHC", "lPHC", "lAMYG", "lTCI", "lTCV")
_ANALYTICAL_EZ_ANCHOR_MV = 1.4681


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the SimNIBS leadfield generator."""
    parser = argparse.ArgumentParser(
        description="Compute a SimNIBS FEM tES leadfield over the TVB regions.",
        epilog=(
            "Stage 2 of the SimNIBS field model. Run under SimNIBS's own interpreter, not uv: "
            "  simnibs_python scripts/generate_simnibs_leadfield.py --geometry data/simnibs/geometry.npz "
            "--m2m external/ernie_extended/m2m_ernie_extended --out data/simnibs/gamma_ernie.npz "
            "Run scripts/export_simnibs_geometry.py first. See docs/simnibs_field_model.md."
        ),
    )
    parser.add_argument("--geometry", type=Path, required=True, help="Geometry NPZ from export_simnibs_geometry.py.")
    parser.add_argument("--m2m", type=Path, required=True, help="SimNIBS head model directory (m2m_*).")
    parser.add_argument(
        "--electrodes",
        nargs="+",
        default=["TP9", "CP5"],
        help="Stimulating electrode labels, from the head model's EEG cap.",
    )
    parser.add_argument("--return-label", default="EX_NECK", help="Label written for the return electrode's row.")
    parser.add_argument(
        "--return-centre",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Return electrode centre in subject space; default is the most inferior posterior scalp node.",
    )
    parser.add_argument("--pad-mm", type=float, nargs=2, default=[50.0, 50.0], help="Electrode pad dimensions in mm.")
    parser.add_argument("--workdir", type=Path, default=None, help="FEM scratch directory (default: next to --out).")
    parser.add_argument("--out", type=Path, required=True, help="Path for the gamma NPZ.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Register and run the witness check only; skip the FEM.",
    )
    return parser.parse_args()


def read_eeg_cap(m2m: Path) -> dict[str, np.ndarray]:
    """Read the head model's EEG cap as a label -> subject-space position map."""
    cap_path = Path(simnibs.SubjectFiles(subpath=str(m2m)).get_eeg_cap())
    positions: dict[str, np.ndarray] = {}
    with cap_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) < _CAP_COLUMNS or not row[0].startswith("Electrode"):
                continue
            positions[row[4].strip().upper()] = np.array(row[1:4], dtype=np.float64)
    return positions


def check_registration(centres_sub: np.ndarray, geometry: np.lib.npyio.NpzFile, m2m: Path) -> float:
    """Verify the MNI->subject warp against the EEG gain, and raise if it looks wrong.

    Reuses the independent witness of docs/tes_field_geometry.md section 1b: a scalp channel's
    largest lead-field entry must belong to a region physically near that channel. A mirrored or
    shifted warp leaves every downstream number plausible and wrong, so this gates the FEM.
    """
    cap = read_eeg_cap(m2m)
    gain = np.asarray(geometry["gain"], dtype=np.float64)
    channel_labels = geometry["channel_labels"].astype(str)
    region_labels = geometry["region_labels"].astype(str)

    peak_region = np.abs(gain).argmax(axis=1)
    distances = [
        float(np.linalg.norm(cap[label.upper()] - centres_sub[peak_region[idx]]))
        for idx, label in enumerate(channel_labels)
        if label.upper() in cap
    ]
    if not distances:
        msg = f"none of the {len(channel_labels)} EEG channels are present in the head model's cap"
        raise ValueError(msg)

    mean_mm = float(np.mean(distances))
    print(f"registration witness: {mean_mm:.1f} mm mean electrode-to-own-lead-field-peak ({len(distances)} channels)")
    if mean_mm > _WITNESS_MAX_MM:
        msg = f"registration witness is {mean_mm:.1f} mm (> {_WITNESS_MAX_MM} mm): the MNI->subject warp is wrong"
        raise ValueError(msg)

    left = np.array([label.startswith("l") for label in region_labels])
    left_x, right_x = centres_sub[left, 0].mean(), centres_sub[~left, 0].mean()
    print(f"hemispheres: left mean x = {left_x:.1f} mm, right mean x = {right_x:.1f} mm")
    if left_x >= 0.0 or right_x <= 0.0:
        msg = f"registration mirrored the hemispheres: left x = {left_x:.1f}, right x = {right_x:.1f}"
        raise ValueError(msg)

    return mean_mm


def find_return_centre(m2m: Path) -> np.ndarray:
    """Locate the most inferior posterior scalp node, standing in for an extracephalic pad.

    Mirrors the virtual ``EX_NECK`` of docs/tes_field_geometry.md section 9.1. On a head model
    truncated at the neck this lands on the mesh's inferior cut, which is why an extended
    head-and-shoulders model is preferred.
    """
    head = simnibs.read_msh(str(simnibs.SubjectFiles(subpath=str(m2m)).fnamehead))
    scalp = head.crop_mesh(tags=[simnibs.ElementTags.SCALP_TH_SURFACE])
    coords = np.asarray(scalp.nodes.node_coord, dtype=np.float64)

    posterior_midline = (np.abs(coords[:, 0]) < _RETURN_X_HALFWIDTH_MM) & (coords[:, 1] < 0.0)
    candidates = coords[posterior_midline] if posterior_midline.any() else coords
    centre = candidates[candidates[:, 2].argmin()]
    print(f"return electrode auto-placed at {np.round(centre, 1).tolist()} (subject space)")
    return centre


def run_basis_simulation(args: argparse.Namespace, return_centre: np.ndarray, active: int) -> Path:
    """Run one FEM solve: +1 mA at stimulating electrode ``active``, -1 mA at the return.

    Every run meshes the whole montage and only the channel currents change, so the geometry is
    identical across runs and the rows superpose exactly for any zero-sum command.
    (NOTE: SimNIBS crashes on 0.0 A channels, so we fall back to a 2-electrode montage,
    sacrificing exact superposition for solver stability.)
    """
    session = sim_struct.SESSION()
    session.subpath = str(args.m2m)
    session.pathfem = str(args.workdir / f"basis_{args.electrodes[active]}")
    session.fields = "vE"
    session.map_to_surf = True
    session.open_in_gmsh = False
    session.solver_options = "pardiso"

    tdcslist = session.add_tdcslist()
    tdcslist.currents = [_MA, -_MA]

    # Active electrode
    electrode1 = tdcslist.add_electrode()
    electrode1.channelnr = 1
    electrode1.centre = args.electrodes[active]
    electrode1.shape = "rect"
    electrode1.dimensions = list(args.pad_mm)
    electrode1.thickness = 5

    # Return electrode
    electrode2 = tdcslist.add_electrode()
    electrode2.channelnr = 2
    electrode2.centre = return_centre if isinstance(return_centre, str) else np.asarray(return_centre).tolist()
    electrode2.shape = "rect"
    electrode2.dimensions = list(args.pad_mm)
    electrode2.thickness = 5

    subid = simnibs.SubjectFiles(subpath=str(args.m2m)).subid
    msh_path = Path(session.pathfem) / f"{subid}_TDCS_1_scalar.msh"
    if msh_path.exists():
        print(f"Reusing precomputed FEM result: {msh_path}")
        return msh_path

    simnibs.run_simnibs(session)
    return msh_path


def reduce_phi(mesh_path: Path, centres_sub: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average the FEM potential over the brain nodes around each region centre.

    Returns the per-region potential in mV per mA, the node count in each ball, and the distance
    to the nearest brain node -- the last two make "is this centroid inside tissue?" auditable
    without rerunning the FEM.
    """
    brain = simnibs.read_msh(str(mesh_path)).crop_mesh(tags=[simnibs.ElementTags.GM, simnibs.ElementTags.WM])
    coords = np.asarray(brain.nodes.node_coord, dtype=np.float64)
    potential = np.asarray(brain.field["v"][:], dtype=np.float64).ravel() * _V_TO_MV

    tree = cKDTree(coords)
    nearest_mm, nearest_idx = tree.query(centres_sub)
    neighbourhoods = tree.query_ball_point(centres_sub, _PHI_BALL_MM)

    phi = np.array(
        [
            potential[nodes].mean() if nodes else potential[nearest_idx[region]]
            for region, nodes in enumerate(neighbourhoods)
        ]
    )
    counts = np.array([len(nodes) for nodes in neighbourhoods], dtype=np.int64)
    return phi, counts, np.asarray(nearest_mm, dtype=np.float64)


def reduce_e_normal(
    overlay_path: Path, surface_sub: np.ndarray, surface_region: np.ndarray, n_regions: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Area-average the cortical-normal E-field over each region's patch of the central surface.

    The normal comes from SimNIBS's middle grey-matter surface, and the region assignment from the
    nearest mapped TVB surface vertex. TVB's 76 regions are all surface patches, so every region
    has support and no normal has to be invented.
    """
    surface = simnibs.read_msh(str(overlay_path))
    coords = np.asarray(surface.nodes.node_coord, dtype=np.float64)
    e_normal = np.asarray(surface.field["E_normal"][:], dtype=np.float64).ravel()
    areas = np.asarray(surface.nodes_areas()[:], dtype=np.float64).ravel()

    vertex_mm, vertex_idx = cKDTree(surface_sub).query(coords)
    region_of_node = surface_region[vertex_idx]

    values = np.zeros(n_regions, dtype=np.float64)
    counts = np.zeros(n_regions, dtype=np.int64)
    mean_vertex_mm = np.zeros(n_regions, dtype=np.float64)
    for region in range(n_regions):
        mask = region_of_node == region
        counts[region] = int(mask.sum())
        if counts[region] == 0:
            continue
        values[region] = np.average(e_normal[mask], weights=areas[mask])
        mean_vertex_mm[region] = float(vertex_mm[mask].mean())

    if counts.min() < _MIN_NODES_PER_REGION:
        thin = int((counts < _MIN_NODES_PER_REGION).sum())
        print(f"WARNING: {thin} region(s) have fewer than {_MIN_NODES_PER_REGION} central-surface nodes")
    return values, counts, mean_vertex_mm


def recommended_scale(rows: np.ndarray, region_labels: np.ndarray) -> float:
    """Derive the config's ``simnibs_scale`` from the analytical model's EZ dose anchor."""
    ez = [int(np.flatnonzero(region_labels == name)[0]) for name in _EZ_REGIONS]
    ez_mean = float(rows[0][ez].mean())
    if ez_mean == 0.0:
        msg = "mean EZ field is zero; cannot derive calibration scale"
        raise ValueError(msg)
    drive = float((-rows[0])[ez].mean())
    if drive >= 0.0:
        print(f"WARNING: cathodal current drives the EZ by {drive:+.4g} (expected negative)")
    return _ANALYTICAL_EZ_ANCHOR_MV / abs(ez_mean)


def main() -> None:
    """Register the connectome to a SimNIBS head model and write the tES leadfield."""
    args = parse_args()
    args.workdir = args.workdir or args.out.parent / "fem"
    args.workdir.mkdir(parents=True, exist_ok=True)

    geometry = np.load(args.geometry)
    region_labels = geometry["region_labels"].astype(str)

    centres_sub = np.asarray(simnibs.mni2subject_coords(geometry["centres_mni_ras"], str(args.m2m)))
    surface_sub = np.asarray(simnibs.mni2subject_coords(geometry["surface_mni_ras"], str(args.m2m)))
    witness_mean_mm = check_registration(centres_sub, geometry, args.m2m)
    if args.dry_run:
        print("dry run: registration passed, skipping the FEM")
        return

    return_centre = (
        np.asarray(args.return_centre, dtype=np.float64) if args.return_centre else find_return_centre(args.m2m)
    )

    n_regions = len(region_labels)
    n_rows = len(args.electrodes) + 1
    gamma_phi = np.zeros((n_rows, n_regions), dtype=np.float64)
    gamma_e_normal = np.zeros((n_rows, n_regions), dtype=np.float64)
    diagnostics: dict[str, np.ndarray] = {}

    for active, label in enumerate(args.electrodes):
        print(f"--- FEM {active + 1}/{len(args.electrodes)}: +1 mA at {label} ---")
        mesh_path = run_basis_simulation(args, return_centre, active)
        overlay_path = mesh_path.parent / "subject_overlays" / f"{mesh_path.stem}_central.msh"

        gamma_phi[active], phi_n, phi_mm = reduce_phi(mesh_path, centres_sub)
        gamma_e_normal[active], en_n, en_mm = reduce_e_normal(
            overlay_path, surface_sub, surface_region=geometry["surface_region"], n_regions=n_regions
        )
        if active == 0:
            diagnostics = {"phi_n_nodes": phi_n, "phi_nearest_mm": phi_mm, "en_n_nodes": en_n, "en_vertex_mm": en_mm}

    scale_phi = recommended_scale(gamma_phi, region_labels)
    scale_e_normal = recommended_scale(gamma_e_normal, region_labels)
    print(f"recommended simnibs_scale: phi = {scale_phi:.6g}, e_normal = {scale_e_normal:.6g}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        gamma_phi=gamma_phi,
        gamma_e_normal=gamma_e_normal,
        electrode_labels=np.array([*args.electrodes, args.return_label], dtype=np.str_),
        region_labels=region_labels,
        witness_mean_mm=np.array(witness_mean_mm),
        recommended_scale_phi=np.array(scale_phi),
        recommended_scale_e_normal=np.array(scale_e_normal),
        provenance=np.array(
            json.dumps(
                {
                    "simnibs_version": simnibs.__version__,
                    "m2m": str(args.m2m),
                    "electrodes": list(args.electrodes),
                    "return_label": args.return_label,
                    "return_centre": np.asarray(return_centre).tolist(),
                    "pad_mm": list(args.pad_mm),
                    "current_mA": 1.0,
                    "argv": sys.argv,
                    "utc": datetime.now(UTC).isoformat(),
                }
            )
        ),
        **diagnostics,
    )
    print(f"Wrote {args.out}: {n_rows} electrodes x {n_regions} regions")


if __name__ == "__main__":
    main()
