from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import TypeVar

    import casadi as ca

    from neuro.types import FloatArray

    TMath = TypeVar("TMath", FloatArray, ca.SX, ca.MX)
else:
    TMath = Any


def zscore(x: TMath, center: FloatArray, scale: FloatArray) -> TMath:
    """Standardize ``x`` with precomputed ``center``/``scale`` (NumPy/CasADi generic)."""
    return (x - center) / scale  # ty: ignore[invalid-return-type]


def unzscore(z: TMath, center: FloatArray, scale: FloatArray) -> TMath:
    """Invert :func:`zscore`."""
    return z * scale + center  # ty:ignore[invalid-return-type]


@dataclass(frozen=True)
class Standardizer:
    """Channel-wise affine standardizer: ``(x - center) / scale``."""

    center: FloatArray
    scale: FloatArray

    @classmethod
    def fit(
        cls,
        x: FloatArray,
        *,
        kind: Literal["standard", "robust"] = "standard",
        global_scaling: bool = False,
    ) -> Standardizer:
        """Fit ``center``/``scale`` from ``(rows, C)`` data."""
        data = np.asarray(x, dtype=np.float64)
        flat = data.reshape(-1, 1) if global_scaling else data
        if kind == "robust":
            center = np.median(flat, axis=0)
            q75, q25 = np.percentile(flat, [75, 25], axis=0)
            scale = q75 - q25
        else:
            center = flat.mean(axis=0)
            scale = flat.std(axis=0)
        scale = np.where(scale == 0, 1.0, scale)
        return cls(center=np.asarray(center, dtype=np.float64), scale=np.asarray(scale, dtype=np.float64))

    def transform(self, x: TMath) -> TMath:
        """Standardize ``x``."""
        return zscore(x, self.center, self.scale)

    def inverse_transform(self, z: TMath) -> TMath:
        """Map standardized values back to raw units."""
        return unzscore(z, self.center, self.scale)

    def arrays(self, prefix: str) -> dict[str, FloatArray]:
        """Fitted parameter arrays, keyed ``<prefix>_center`` / ``<prefix>_scale``."""
        return {f"{prefix}_center": self.center, f"{prefix}_scale": self.scale}

    @classmethod
    def from_arrays(cls, a: Mapping[str, FloatArray], prefix: str) -> Standardizer:
        """Rebuild from a :meth:`arrays` mapping written under ``prefix``."""
        return cls(
            center=np.asarray(a[f"{prefix}_center"], dtype=np.float64),
            scale=np.asarray(a[f"{prefix}_scale"], dtype=np.float64),
        )
