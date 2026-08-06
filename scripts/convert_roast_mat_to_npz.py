from __future__ import annotations

import argparse
from pathlib import Path

from neuro.stimulation.roast_io import convert_roast_gamma_to_npz


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Yu gamma conversion."""
    parser = argparse.ArgumentParser(
        description="Convert a Yu signed-magnitude gamma MAT file into the NPZ the yu_signed model loads.",
    )
    parser.add_argument("--mat", type=Path, default=Path("data/roast_gamma.mat"))
    parser.add_argument("--out", type=Path, default=Path("data/roast_gamma.npz"))
    parser.add_argument("--electrodes", nargs="+", default=["TP9", "CP5"])
    return parser.parse_args()


def main() -> None:
    """Convert the Yu gamma MAT file to NPZ."""
    args = parse_args()
    convert_roast_gamma_to_npz(args.mat, args.out, tuple(args.electrodes))


if __name__ == "__main__":
    main()
