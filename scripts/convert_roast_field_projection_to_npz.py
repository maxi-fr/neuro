from __future__ import annotations

import argparse
from pathlib import Path

from neuro.stimulation.roast_io import convert_roast_field_projection_to_npz


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the ROAST field projection conversion."""
    parser = argparse.ArgumentParser(
        description="Convert a ROAST 3D field projection MAT file into the NPZ the roast_3d stimulation model loads.",
    )
    parser.add_argument("--mat", type=Path, default=Path("data/roast_field_projection_3d.mat"))
    parser.add_argument("--out", type=Path, default=Path("data/roast_field_projection_3d.npz"))
    return parser.parse_args()


def main() -> None:
    """Convert the ROAST field projection MAT file to NPZ."""
    args = parse_args()
    convert_roast_field_projection_to_npz(args.mat, args.out)


if __name__ == "__main__":
    main()
