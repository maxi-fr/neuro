from __future__ import annotations

import argparse
from pathlib import Path

from neuro.roast import convert_roast_leadfield_to_npz


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the ROAST leadfield conversion."""
    parser = argparse.ArgumentParser(
        description="Convert a ROAST 3D leadfield MAT file into the NPZ the Connectome loads.",
    )
    parser.add_argument("--mat", type=Path, default=Path("data/roast_leadfield_3d.mat"))
    parser.add_argument("--out", type=Path, default=Path("data/roast_leadfield_3d.npz"))
    return parser.parse_args()


def main() -> None:
    """Convert the ROAST leadfield MAT file to NPZ."""
    args = parse_args()
    convert_roast_leadfield_to_npz(args.mat, args.out)


if __name__ == "__main__":
    main()
