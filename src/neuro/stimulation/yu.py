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
    from neuro.stimulation.base import _YuConfig
    from neuro.types import FloatArray, StrArray

# The MATLAB montages all return at the ROAST cap's Ex8 site, which the stored rows fold into
# their drive electrode rather than carrying as a row of their own.
_RETURN_LABEL = "EX8"


class YuStim(StimulationModel):
    """Yu et al. (2024) signed field magnitude ``||E|| * sign(V - V_med)``, per electrode.

    The file stores one row per *bipolar pair* at ``-1 mA`` on the drive electrode and ``+1 mA``
    at the Ex8 return. The rows are flipped at load so each means "field per ``+1 mA`` at that
    drive electrode", and an all-zero return row is appended: for a zero-sum current vector the
    return's own coefficient is redundant, since ``-u_drive`` summed over the drive electrodes
    already equals ``u_return``.

    **Known approximation:** superposing two rows assumes linearity, but the signed magnitude
    ``||E|| * sign(V - V_med)`` does not superpose (see ``docs/biophysical_tes_field_model.md``
    section 2). This is the same approximation the multi-electrode analytical model makes, but
    here it is against a quantity documented as non-superposable.
    """

    def __init__(self, cfg: _YuConfig, region_labels: StrArray) -> None:
        """Load the montage rows, check their region order and select the requested electrodes."""
        path = Path(cfg.path)
        if not path.exists():
            msg = f"Yu gamma file not found at {path}"
            raise FileNotFoundError(msg)

        with np.load(path) as data:
            gamma_phi = np.asarray(data["gamma_phi"], dtype=np.float64)
            drive_labels = np.asarray(data["electrode_labels"], dtype=np.str_)
            assert_region_order(np.asarray(data["region_labels"], dtype=np.str_), region_labels)

        if gamma_phi.shape != (len(drive_labels), len(region_labels)):
            msg = f"gamma_phi shape {gamma_phi.shape} does not match ({len(drive_labels)}, {len(region_labels)})"
            raise ValueError(msg)

        gamma = np.vstack([-gamma_phi, np.zeros((1, len(region_labels)))])
        labels = np.append(drive_labels, _RETURN_LABEL)

        rows = select_rows(labels, cfg.electrodes) if cfg.electrodes is not None else slice(None)
        self.gamma = gamma[rows]
        self.control_labels = labels[rows]
        self.n_controls = len(self.control_labels)

    def project(self, u: FloatArray) -> FloatArray:
        """Superpose the per-electrode signed-magnitude fields weighted by their currents."""
        check_n_controls(u, self.n_controls)
        return u @ self.gamma
