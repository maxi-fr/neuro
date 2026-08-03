from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np

from neuro.connectome import Connectome, centres_to_mni_ras

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    from tvb.datatypes.region_mapping import RegionMapping
    from tvb.datatypes.surfaces import CorticalSurface

_REGION_MAPPING_FILE = "regionMapping_16k_76.txt"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the geometry export."""
    parser = argparse.ArgumentParser(
        description="Export the TVB connectome's geometry in MNI RAS for SimNIBS.",
        epilog=(
            "Stage 1 of the SimNIBS field model: runs in this project's venv (TVB, no SimNIBS) "
            "and writes the interchange file that scripts/generate_simnibs_leadfield.py reads "
            "under simnibs_python. See docs/simnibs_field_model.md."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/simnibs/geometry.npz"),
        help="Path for the geometry NPZ.",
    )
    return parser.parse_args()


def main() -> None:
    """Write region centres, the cortical surface and the EEG gain in MNI RAS millimetres."""
    args = parse_args()

    conn = Connectome.from_config({})
    surface = CorticalSurface.from_file()
    region_of_vertex = np.asarray(RegionMapping.from_file(_REGION_MAPPING_FILE).array_data, dtype=np.int64)

    vertices = np.asarray(surface.vertices, dtype=np.float64)
    if len(region_of_vertex) != len(vertices):
        msg = f"region mapping has {len(region_of_vertex)} entries but the surface has {len(vertices)} vertices"
        raise ValueError(msg)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        centres_mni_ras=centres_to_mni_ras(conn.centres),
        region_labels=conn.region_labels,
        surface_mni_ras=centres_to_mni_ras(vertices),
        surface_region=region_of_vertex,
        channel_labels=conn.channel_labels,
        gain=conn.gain,
    )
    print(f"Wrote {args.out}: {len(conn.region_labels)} regions, {len(vertices)} vertices, {conn.gain.shape} gain")


if __name__ == "__main__":
    main()
