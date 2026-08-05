from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np

if TYPE_CHECKING:
    from neuro.types import FloatArray

_NDIM_3D = 3
_EXPECTED_NORMALS_FRAME = "mni_ras"


def _decode_strings(mat: h5py.File, node: h5py.Dataset) -> list[str]:
    """Decode a MATLAB cell array of char vectors from a v7.3 MAT file."""
    return ["".join(chr(c) for c in np.asarray(mat[ref]).ravel()) for ref in np.asarray(node).ravel()]


def _decode_char(node: h5py.Dataset) -> str:
    """Decode a single MATLAB char vector from a v7.3 MAT file."""
    return "".join(chr(c) for c in np.asarray(node).ravel())


def load_leadfield_mat(mat_path: Path) -> tuple[FloatArray, list[str], list[str], list[str], FloatArray]:
    """Load the leadfield, requested/ROAST channel labels, region labels and normals from a MAT file.

    ``generate_roast_leadfield_3d.m`` writes ``-v7.3``, so the file is HDF5 and every array
    arrives transposed relative to its MATLAB shape.
    """
    with h5py.File(mat_path, "r") as mat:
        # MATLAB (63, 76, 3) is stored as (3, 76, 63).
        leadfield = np.transpose(np.asarray(mat["leadfield_3d"], dtype=np.float64), (2, 1, 0))

        meta = mat["metadata"]
        channel_labels = _decode_strings(mat, meta["channelLabels"])
        roast_labels = _decode_strings(mat, meta["roastLabels"])
        region_labels = _decode_strings(mat, meta["regionLabels"])
        region_normals = np.asarray(meta["regionNormals"], dtype=np.float64).T

        frame = _decode_char(meta["normalsFrame"]) if "normalsFrame" in meta else ""

    if frame != _EXPECTED_NORMALS_FRAME:
        msg = (
            f"MAT file declares normals frame {frame!r}, expected {_EXPECTED_NORMALS_FRAME!r}; "
            "the leadfield's E-vectors are MNI RAS and the normals must share that frame."
        )
        raise ValueError(msg)

    return leadfield, channel_labels, roast_labels, region_labels, region_normals


def _validate(
    leadfield: FloatArray,
    channel_labels: list[str],
    region_labels: list[str],
    region_normals: FloatArray,
) -> None:
    """Check the leadfield's shape against its labels and that the return row is the zero reference."""
    expected = (len(channel_labels), len(region_labels), _NDIM_3D)
    if leadfield.shape != expected:
        msg = f"leadfield_3d shape {leadfield.shape} does not match {expected}"
        raise ValueError(msg)
    if region_normals.shape != (len(region_labels), _NDIM_3D):
        msg = f"region_normals shape {region_normals.shape} does not match ({len(region_labels)}, {_NDIM_3D})"
        raise ValueError(msg)
    if np.any(leadfield[-1]):
        msg = f"the return row ({channel_labels[-1]}) is not zero; it must be the reference ground"
        raise ValueError(msg)


def convert_roast_leadfield_to_npz(
    mat_path: str | Path = "data/roast_leadfield_3d.mat",
    npz_path: str | Path = "data/roast_leadfield_3d.npz",
) -> None:
    """Read a ROAST leadfield MAT file and write the NPZ that :class:`Connectome` loads."""
    src = Path(mat_path)
    dst = Path(npz_path)

    if not src.exists():
        msg = f"Source MAT file not found at {src}"
        raise FileNotFoundError(msg)

    leadfield_3d, channel_labels, roast_labels, region_labels, region_normals = load_leadfield_mat(src)
    _validate(leadfield_3d, channel_labels, region_labels, region_normals)

    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        dst,
        leadfield_3d=leadfield_3d,
        channel_labels=np.asarray(channel_labels, dtype=str),
        roast_labels=np.asarray(roast_labels, dtype=str),
        region_labels=np.asarray(region_labels, dtype=str),
        region_normals=region_normals,
    )
    substituted = [(req, got) for req, got in zip(channel_labels, roast_labels, strict=True) if req != got]
    print(f"Successfully converted {src} -> {dst}")  # noqa: T201
    print(f"  Leadfield 3D shape: {leadfield_3d.shape}")  # noqa: T201
    print(f"  Channels ({len(channel_labels)}): {channel_labels[:5]} ... {channel_labels[-1]}")  # noqa: T201
    print(f"  Regions ({len(region_labels)}): {region_labels[:5]} ...")  # noqa: T201
    print(f"  Substituted electrodes: {substituted or 'none'}")  # noqa: T201
