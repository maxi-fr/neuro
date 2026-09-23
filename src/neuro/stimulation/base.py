from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path  # noqa: TC003  (pydantic resolves the schemas' annotations at runtime)
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self

import numpy as np
from pydantic import Discriminator, Field, Tag, model_validator

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


_DEFAULT_ROAST_ELECTRODES: tuple[str, ...] = ("TP9", "CP5", "Ex8")


class _Roast3DConfig(StrictConfig):
    """ROAST 3D FEM electric-field projection, reduced along the cortical normal."""

    model: Literal["roast_3d"] = "roast_3d"
    field_projection_path: str | Path = "data/roast_field_projection_3d.npz"
    electrodes: list[str] | None = Field(default_factory=lambda: list(_DEFAULT_ROAST_ELECTRODES))


class _DynamicYuConfig(StrictConfig):
    """Dynamic Yu stimulation model: ||L_E u|| * smooth_sign(L_V u - V_med)."""

    model: Literal["yu_dynamic"] = "yu_dynamic"
    field_projection_path: str | Path = "data/roast_field_projection_3d.npz"
    electrodes: list[str] | None = Field(default_factory=lambda: list(_DEFAULT_ROAST_ELECTRODES))
    alpha: float = Field(default=3.0, gt=0.0)
    scale_factor: float = Field(default=1.0, gt=0.0)


def _stim_discriminator(v: Any) -> str:  # noqa: ANN401 -- pydantic discriminator receives raw input
    """Extract model discriminator tag, defaulting to roast_3d when omitted."""
    if isinstance(v, dict):
        return str(v.get("model", "roast_3d"))
    return str(getattr(v, "model", "roast_3d"))


StimulationConfig = Annotated[
    Annotated[_Roast3DConfig, Tag("roast_3d")]
    | Annotated[_NullConfig, Tag("none")]
    | Annotated[_AnalyticalConfig, Tag("analytical")]
    | Annotated[_DynamicYuConfig, Tag("yu_dynamic")],
    Discriminator(_stim_discriminator),
]
