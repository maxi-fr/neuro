from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
from scipy.io import savemat

from neuro.connectome import Connectome, centres_to_mni_ras

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    from tvb.datatypes.region_mapping import RegionMapping
    from tvb.datatypes.surfaces import CorticalSurface

_REGION_MAPPING_FILE = "regionMapping_16k_76.txt"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the TVB geometry export."""
    parser = argparse.ArgumentParser(
        description="Export the TVB connectome's geometry in MNI RAS for ROAST and field models.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/tvb_geometry.npz"),
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

    centres_ras = centres_to_mni_ras(conn.centres)
    surface_ras = centres_to_mni_ras(vertices)

    np.savez(
        args.out,
        centres_mni_ras=centres_ras,
        region_labels=conn.region_labels,
        surface_mni_ras=surface_ras,
        surface_region=region_of_vertex,
        channel_labels=conn.channel_labels,
        gain=conn.gain,
    )
    mat_out = args.out.with_suffix(".mat")
    savemat(
        mat_out,
        {
            "centres_mni_ras": centres_ras,
            "region_labels": np.asarray(conn.region_labels, dtype=object),
            "surface_mni_ras": surface_ras,
            "surface_region": region_of_vertex,
            "channel_labels": np.asarray(conn.channel_labels, dtype=object),
            "gain": conn.gain,
        },
    )
    print(
        f"Wrote {args.out} and {mat_out}: {len(conn.region_labels)} regions, {len(vertices)} vertices, {conn.gain.shape} gain"
    )


if __name__ == "__main__":
    main()
