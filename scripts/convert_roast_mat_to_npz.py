"""Converts MATLAB ROAST gamma output (.mat) to Python Connectome NPZ format."""

from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat


def load_gamma_mat(mat_path: Path) -> tuple[np.ndarray, list[str]]:
    """Load gamma array and region_labels from MAT file (supports v7 and v7.3 HDF5)."""
    try:
        mat_data = loadmat(mat_path, squeeze_me=True)
        gamma = np.asarray(mat_data["gamma"], dtype=np.float64)
        metadata = mat_data.get("metadata", {})
        region_labels = list(metadata.get("regionLabels", []))
    except (NotImplementedError, ValueError):
        with h5py.File(mat_path, "r") as h5_file:
            gamma = np.asarray(h5_file["gamma"], dtype=np.float64)
            # In HDF5 v7.3, MATLAB stores matrices transposed (N_regions x N_montages)
            if gamma.ndim == 2 and gamma.shape[0] > gamma.shape[1]:  # noqa: PLR2004
                gamma = gamma.T
            region_labels = []

    if gamma.ndim == 1:
        gamma = gamma[None, :]

    return gamma, region_labels


def convert_roast_mat_to_npz(
    mat_path: str | Path = "data/roast_gamma.mat",
    npz_path: str | Path = "data/roast_gamma.npz",
    electrode_labels: tuple[str, ...] = ("TP9", "CP5"),
) -> None:
    """Read data/roast_gamma.mat and write data/roast_gamma.npz for Connectome."""
    src = Path(mat_path)
    dst = Path(npz_path)

    if not src.exists():
        msg = f"Source MAT file not found at {src}"
        raise FileNotFoundError(msg)

    gamma, region_labels = load_gamma_mat(src)

    if len(region_labels) == 0:
        # Fallback to loading standard TVB geometry region labels
        geom_path = Path("data/simnibs/geometry.npz")
        if geom_path.exists():
            with np.load(geom_path) as geom:
                region_labels = list(geom["region_labels"].astype(str))

    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        dst,
        gamma_phi=gamma,
        electrode_labels=np.asarray(electrode_labels, dtype=str),
        region_labels=np.asarray(region_labels, dtype=str),
    )
    print(f"Successfully converted {src} -> {dst}")
    print(f"  Gamma shape: {gamma.shape}")
    print(f"  Electrodes: {electrode_labels}")
    print(f"  Regions: {len(region_labels)}")


if __name__ == "__main__":
    convert_roast_mat_to_npz()
