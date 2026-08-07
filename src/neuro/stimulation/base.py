from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path  # noqa: TC003  (pydantic resolves the schemas' annotations at runtime)
from typing import TYPE_CHECKING, Annotated, Literal, Self

import numpy as np
from pydantic import Field, model_validator

from neuro.config import StrictConfig

if TYPE_CHECKING:
    from neuro.types import FloatArray, StrArray


class StimulationModel(ABC):
    """Maps per-electrode tES current to a per-node drive."""

    n_controls: int
    control_labels: StrArray

    @abstractmethod
    def project(self, u: FloatArray) -> FloatArray:
        """Node-level drive, shape ``(n_nodes,)``, for currents ``u`` of shape ``(n_controls,)``."""


def check_n_controls(u: FloatArray, n_controls: int) -> None:
    """Raise if the control vector does not carry one current per control electrode."""
    if u.size != n_controls:
        msg = f"control has {u.size} electrodes but the stimulation model has {n_controls}"
        raise ValueError(msg)


def select_rows(labels: StrArray, wanted: list[str]) -> list[int]:
    """Row indices of ``wanted`` in ``labels``, matched case-insensitively and in order."""
    index = {str(label).upper(): idx for idx, label in enumerate(labels)}
    missing = [name for name in wanted if name.upper() not in index]
    if missing:
        msg = f"electrodes {missing} are not in the file's rows {sorted(index)}"
        raise ValueError(msg)
    return [index[name.upper()] for name in wanted]


def assert_region_order(file_labels: StrArray, node_labels: StrArray) -> None:
    """Raise unless a file-backed projection is stored in the plant's own node order."""
    if len(file_labels) != len(node_labels) or not np.array_equal(file_labels, node_labels):
        msg = (
            "the file's region_labels do not match the connectome's node order; the projection "
            "would be applied to the wrong regions. Regenerate the file against this connectome."
        )
        raise ValueError(msg)


class _NullConfig(StrictConfig):
    """No stimulation: a single control electrode that drives nothing."""

    model: Literal["none"] = "none"


class _AnalyticalConfig(StrictConfig):
    """Coulomb volume potential of a point source per electrode, softened by ``spread``."""

    model: Literal["analytical"]
    electrodes: list[str] = Field(min_length=2)
    spread: float | list[float] = 15.0

    @model_validator(mode="after")
    def _check_spread_length(self) -> Self:
        if isinstance(self.spread, list) and len(self.spread) != len(self.electrodes):
            msg = f"spread has {len(self.spread)} entries but there are {len(self.electrodes)} electrodes"
            raise ValueError(msg)
        return self


class _Roast3DConfig(StrictConfig):
    """ROAST 3D FEM electric-field leadfield, reduced along the cortical normal."""

    model: Literal["roast_3d"]
    leadfield_path: str | Path = "data/roast_leadfield_3d.npz"
    electrodes: list[str] | None = None


class _DynamicYuConfig(StrictConfig):
    """Dynamic Yu stimulation model: ||L_E u|| * smooth_sign(L_V u - V_med)."""

    model: Literal["yu_dynamic"]
    leadfield_path: str | Path = "data/roast_leadfield_3d.npz"
    electrodes: list[str] | None = None
    alpha: float = Field(default=3.0, gt=0.0)
    scale_factor: float = Field(default=1.0, gt=0.0)


StimulationConfig = Annotated[
    _NullConfig | _AnalyticalConfig | _Roast3DConfig | _DynamicYuConfig,
    Field(discriminator="model"),
]


