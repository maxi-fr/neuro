from __future__ import annotations

from typing import TYPE_CHECKING

from neuro.stimulation.analytical import AnalyticalStim
from neuro.stimulation.base import StimulationConfig, StimulationModel
from neuro.stimulation.null import NullStim
from neuro.stimulation.roast_3d import Roast3DStim
from neuro.stimulation.yu import YuStim

if TYPE_CHECKING:
    from neuro.connectome import Connectome

__all__ = [
    "AnalyticalStim",
    "NullStim",
    "Roast3DStim",
    "StimulationConfig",
    "StimulationModel",
    "YuStim",
    "build_stimulation",
]


def build_stimulation(cfg: StimulationConfig, conn: Connectome) -> StimulationModel:
    """Build the stimulation model ``cfg`` selects, on ``conn``'s node basis."""
    match cfg.model:
        case "none":
            return NullStim(len(conn.region_labels))
        case "analytical":
            return AnalyticalStim(cfg, conn.centres)
        case "roast_3d":
            return Roast3DStim(cfg, conn.region_labels)
        case "yu_signed":
            return YuStim(cfg, conn.region_labels)
