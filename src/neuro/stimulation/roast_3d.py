from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from neuro.stimulation.base import (
    StimulationModel,
    assert_region_order,
    check_n_controls,
    select_rows,
)

if TYPE_CHECKING:
    from neuro.stimulation.base import _Roast3DConfig
    from neuro.types import FloatArray, StrArray

_NDIM_3D = 3


def load_roast_3d_leadfield(
    path: str | Path,
) -> tuple[FloatArray, StrArray, StrArray, FloatArray]:
    """Load a 3D ROAST electric field leadfield NPZ file: electrode current (mA) -> E-field (V/m) per region.

    Returns
    -------
    leadfield_E
        Shape ``(n_controls, n_nodes, 3)``. Row ``k`` is ``(Ex, Ey, Ez)`` in V/m sampled at each
        region centroid for +1 mA injected at channel ``k`` against the return electrode, which
        is the last row and is zero by construction (the reference). Linear in the current, so
        an arbitrary zero-sum montage is ``u @ leadfield_E``.
    channel_labels
        Control channel labels of shape ``(n_controls,)`` (e.g. 62 scalp + Ex8).
    region_labels
        Region labels of shape ``(n_nodes,)``, in the plant's node order.
    region_normals
        Outward unit region surface normals of shape ``(n_nodes, 3)``, in the leadfield's MNI
        RAS frame.
    """
    p = Path(path)
    if not p.exists():
        msg = f"3D Leadfield file not found at {p}"
        raise FileNotFoundError(msg)

    with np.load(p) as data:
        if "leadfield_E" in data:
            leadfield_E = np.asarray(data["leadfield_E"], dtype=np.float64)
        elif "leadfield_3d" in data:  # TODO: delete fallback for legacy 'leadfield_3d' key soon
            leadfield_E = np.asarray(data["leadfield_3d"], dtype=np.float64)
        else:
            msg = f"NPZ file at {p} carries neither 'leadfield_E' nor legacy 'leadfield_3d'"
            raise KeyError(msg)

        channel_labels = np.asarray(data["channel_labels"], dtype=np.str_)
        region_labels = np.asarray(data["region_labels"], dtype=np.str_)
        if "region_normals" not in data:
            msg = (
                f"{p} carries no region_normals. The leadfield and the normals it is reduced "
                "against are one artefact from one MATLAB run sharing one frame; regenerate it."
            )
            raise ValueError(msg)
        region_normals = np.asarray(data["region_normals"], dtype=np.float64)
    return leadfield_E, channel_labels, region_labels, region_normals



class Roast3DStim(StimulationModel):
    """tES field from a ROAST 3D FEM leadfield, reduced along each region's cortical normal.

    The leadfield's E-vectors and the region normals are both in MNI RAS; the file must carry
    its own normals so the two provably share the frame the MATLAB run solved in.

    ``polarization_length_mm`` is the ``lambda`` of ``docs/biophysical_tes_field_model.md``
    section 3, which turns the cortical-normal field into a somatic membrane deflection:
    ``E.n`` in V/m is mV/mm, so ``lambda * (E.n)`` is mV. It is a property of cortical tissue
    rather than of a run, so it is not a config key.

    The dot product commutes with the superposition over electrodes, so ``lambda * (E_k . n)``
    is a constant ``(n_controls, n_nodes)`` matrix -- the same object as the analytical model's
    gamma -- and the 3-vectors never have to be carried into the hot loop.
    """

    polarization_length_mm = 0.35

    def __init__(self, cfg: _Roast3DConfig, region_labels: StrArray) -> None:
        """Load the leadfield, check it against ``region_labels`` and reduce the montage rows."""
        leadfield_E, channel_labels, file_regions, normals = load_roast_3d_leadfield(cfg.leadfield_path)
        assert_region_order(file_regions, region_labels)

        n_nodes = len(region_labels)
        if leadfield_E.shape != (len(channel_labels), n_nodes, _NDIM_3D):
            msg = f"leadfield_E shape {leadfield_E.shape} does not match ({len(channel_labels)}, {n_nodes}, 3)"
            raise ValueError(msg)
        if normals.shape != (n_nodes, _NDIM_3D) or not np.any(normals):
            msg = (
                f"{cfg.leadfield_path} carries no usable region_normals (shape {normals.shape}); "
                "the leadfield must be regenerated with its own MNI RAS normals."
            )
            raise ValueError(msg)

        rows = select_rows(channel_labels, cfg.electrodes) if cfg.electrodes is not None else slice(None)
        self.gamma = self.polarization_length_mm * np.einsum("kid,id->ki", leadfield_E[rows], normals)
        self.control_labels = channel_labels[rows]
        self.n_controls = len(self.control_labels)

    def project(self, u: FloatArray) -> FloatArray:
        """Somatic deflection in mV per node, for per-electrode currents ``u`` in mA."""
        check_n_controls(u, self.n_controls)
        return u @ self.gamma
