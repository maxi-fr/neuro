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
    from neuro.stimulation.base import _DynamicYuConfig
    from neuro.types import FloatArray, StrArray

_NDIM_3D = 3
_EPSILON = 1e-12


class DynamicYuStim(StimulationModel):
    """Dynamic multi-channel Yu model: ``k * ||L_E u|| * tanh(alpha * (L_V u - V_med) / sigma_V)``.

    Calculates vector electric field superposition and electrostatic potential superposition
    dynamically during each ``project(u)`` call, preserving field superposition across
    arbitrary electrode montages while applying a smooth sigmoidal polarity transition.
    """

    def __init__(self, cfg: _DynamicYuConfig, region_labels: StrArray) -> None:
        """Load 3D electric field and potential leadfields, filter rows, and check region labels."""
        path = Path(cfg.leadfield_path)
        if not path.exists():
            msg = f"Leadfield file not found at {path}"
            raise FileNotFoundError(msg)

        with np.load(path) as data:
            if "leadfield_V" not in data:
                msg = (
                    f"Leadfield file at {path} does not carry 'leadfield_V'; regenerate it "
                    "using convert_roast_leadfield_to_npz to enable dynamic Yu stimulation."
                )
                raise ValueError(msg)

            if "leadfield_E" in data:
                leadfield_E = np.asarray(data["leadfield_E"], dtype=np.float64)
            elif "leadfield_3d" in data:  # TODO: delete fallback for legacy 'leadfield_3d' key soon
                leadfield_E = np.asarray(data["leadfield_3d"], dtype=np.float64)
            else:
                msg = f"NPZ file at {path} carries neither 'leadfield_E' nor legacy 'leadfield_3d'"
                raise KeyError(msg)

            leadfield_v = np.asarray(data["leadfield_V"], dtype=np.float64)
            channel_labels = np.asarray(data["channel_labels"], dtype=np.str_)
            file_regions = np.asarray(data["region_labels"], dtype=np.str_)

        assert_region_order(file_regions, region_labels)
        n_nodes = len(region_labels)
        if leadfield_E.shape != (len(channel_labels), n_nodes, _NDIM_3D):
            msg = f"leadfield_E shape {leadfield_E.shape} does not match ({len(channel_labels)}, {n_nodes}, 3)"
            raise ValueError(msg)
        if leadfield_v.shape != (len(channel_labels), n_nodes):
            msg = f"leadfield_V shape {leadfield_v.shape} does not match ({len(channel_labels)}, {n_nodes})"
            raise ValueError(msg)

        rows = select_rows(channel_labels, cfg.electrodes) if cfg.electrodes is not None else slice(None)
        self.leadfield_E = leadfield_E[rows]
        self.leadfield_V = leadfield_v[rows]
        self.control_labels = channel_labels[rows]
        self.n_controls = len(self.control_labels)
        self.alpha = float(cfg.alpha)
        self.scale_factor = float(cfg.scale_factor)

    def project(self, u: FloatArray) -> FloatArray:
        """Calculate dynamic E-field magnitude and smooth voltage polarity for currents ``u``."""
        check_n_controls(u, self.n_controls)

        # 1. Superpose 3D electric field vector: E_net shape (n_nodes, 3)
        e_net = np.einsum("k,kid->id", u, self.leadfield_E)
        e_mag = np.linalg.norm(e_net, axis=-1)

        # 2. Superpose electrostatic potential: V_net shape (n_nodes,)
        v_net = u @ self.leadfield_V
        v_min, v_max = np.min(v_net), np.max(v_net)
        v_med = 0.5 * (v_max + v_min)
        sigma_v = max(float(np.std(v_net)), _EPSILON)

        # 3. Smooth sigmoidal polarity transition in (-1, 1)
        polarity = np.tanh(self.alpha * (v_net - v_med) / sigma_v)

        return self.scale_factor * e_mag * polarity
