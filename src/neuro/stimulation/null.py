from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from neuro.stimulation.base import StimulationModel

if TYPE_CHECKING:
    from neuro.types import FloatArray


class NullStim(StimulationModel):
    """Unstimulated plant: one nominal electrode whose drive is always zero.

    A single control rather than none keeps the ``simulate`` plumbing (which sizes its input
    vector from ``n_controls``) and the Kirchhoff check working unchanged.
    """

    def __init__(self, n_nodes: int) -> None:
        """Build a zero projection over ``n_nodes`` regions."""
        self.n_controls = 1
        self.control_labels = np.array(["NONE"], dtype=np.str_)
        self.n_nodes = n_nodes

    def project(self, u: FloatArray) -> FloatArray:
        """Zero drive, whatever the current."""
        del u
        return np.zeros(self.n_nodes, dtype=np.float64)
