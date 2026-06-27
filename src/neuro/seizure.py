"""Shared seizure-regime constants and helpers for the data-collection experiments.

The EZ/PZ seizure regime -- a seizure originating in ``lHC``/``lPHC``/``lAMYG`` (EZ) and
propagating to ``lTCI``/``lTCV`` (PZ), with all other regions healthy -- underpins the seizure
YAML configs (``configs/simulation/jansen_rit_seizure*.yaml``, whose ``A`` vectors are
:func:`build_seizure_a_gains`) and the smart UKF initialisation in
``scripts/run_ukf_feasibility.py``. Defining the regime once here keeps them in sync.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from neuro.connectome import Connectome

FloatArray = npt.NDArray[np.float64]

DT = 1e-4
K = 0.54
SPEED = 50.0
SEED = 42

EZ_REGIONS = ("lHC", "lPHC", "lAMYG")
PZ_REGIONS = ("lTCI", "lTCV")

A_HEALTHY = 3.25
A_EZ = 3.6
A_PZ = 3.4


def build_seizure_a_gains(connectome: Connectome) -> FloatArray:
    """Build the per-region excitatory-gain vector for the EZ/PZ seizure regime."""
    a_gains = np.full(len(connectome.region_labels), A_HEALTHY)
    a_gains[[connectome.region_index[name] for name in EZ_REGIONS]] = A_EZ
    a_gains[[connectome.region_index[name] for name in PZ_REGIONS]] = A_PZ
    return a_gains


def focus_indices(connectome: Connectome) -> list[int]:
    """Return the region indices of the EZ then PZ focus nodes."""
    return [connectome.region_index[name] for name in (*EZ_REGIONS, *PZ_REGIONS)]
