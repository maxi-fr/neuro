from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
from sklearn.decomposition import PCA

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import TypeVar

    import casadi as ca
    import jax

    from neuro.types import FloatArray

    TMath = TypeVar("TMath", FloatArray, jax.Array, ca.SX, ca.MX)
else:
    TMath = Any


def zscore(x: TMath, center: FloatArray, scale: FloatArray) -> TMath:
    """Standardize ``x`` with precomputed ``center``/``scale`` (NumPy/JAX/CasADi generic)."""
    return (x - center) / scale  # ty: ignore[invalid-return-type]


def unzscore(z: TMath, center: FloatArray, scale: FloatArray) -> TMath:
    """Invert :func:`zscore`."""
    return z * scale + center  # ty:ignore[invalid-return-type]


@dataclass(frozen=True)
class Standardizer:
    """Channel-wise affine standardizer: ``(x - center) / scale``."""

    type_tag: ClassVar[str] = "standardizer"

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

    def arrays(self) -> dict[str, FloatArray]:
        """Fitted parameter arrays."""
        return {"center": self.center, "scale": self.scale}

    @classmethod
    def from_arrays(cls, a: Mapping[str, FloatArray]) -> Standardizer:
        """Rebuild from :meth:`arrays` output."""
        return cls(center=np.asarray(a["center"], dtype=np.float64), scale=np.asarray(a["scale"], dtype=np.float64))


@dataclass(frozen=True)
class PCAProjection:
    """Principal Component Analysis: ``z = (x - mean) @ basis.T``."""

    type_tag: ClassVar[str] = "pca"

    basis: FloatArray
    mean: FloatArray

    @classmethod
    def fit(cls, x: FloatArray, latent_dim: int) -> PCAProjection | None:
        """Fit a ``latent_dim``-component PCA on ``(rows, C)`` data."""
        data = np.asarray(x, dtype=np.float64)
        if latent_dim >= data.shape[1]:
            return None
        pca = PCA(n_components=latent_dim)
        pca.fit(data)
        return cls(basis=np.asarray(pca.components_, dtype=np.float64), mean=np.asarray(pca.mean_, dtype=np.float64))

    def transform(self, x: TMath) -> TMath:
        """Encode ``(..., C)`` onto the ``(..., k)`` latent components."""
        return (x - self.mean) @ self.basis.T  # ty: ignore[invalid-return-type]

    def inverse_transform(self, z: TMath) -> TMath:
        """Decode latent components back to channels."""
        return z @ self.basis + self.mean  # ty: ignore[invalid-return-type]

    def arrays(self) -> dict[str, FloatArray]:
        """Fitted parameter arrays."""
        return {"basis": self.basis, "mean": self.mean}

    @classmethod
    def from_arrays(cls, a: Mapping[str, FloatArray]) -> PCAProjection:
        """Rebuild from :meth:`arrays` output."""
        return cls(basis=np.asarray(a["basis"], dtype=np.float64), mean=np.asarray(a["mean"], dtype=np.float64))


Transform = Standardizer | PCAProjection
_REGISTRY: dict[str, type[Standardizer | PCAProjection]] = {
    Standardizer.type_tag: Standardizer,
    PCAProjection.type_tag: PCAProjection,
}


@dataclass(frozen=True)
class Pipeline:
    """Ordered composition of :class:`Transform` steps."""

    steps: tuple[Transform, ...]

    def transform(self, x: TMath) -> TMath:
        """Apply steps in order."""
        for step in self.steps:
            x = step.transform(x)  # ty: ignore[invalid-argument-type]
        return x

    def inverse_transform(self, z: TMath) -> TMath:
        """Apply steps' inverse in reverse order."""
        for step in reversed(self.steps):
            z = step.inverse_transform(z)  # ty: ignore[invalid-argument-type]
        return z

    @property
    def standardizer(self) -> Standardizer | None:
        """The standardizer step, if present."""
        return next((s for s in self.steps if isinstance(s, Standardizer)), None)

    @property
    def pca(self) -> PCAProjection | None:
        """The PCA projection step, if present."""
        return next((s for s in self.steps if isinstance(s, PCAProjection)), None)

    def step_tags(self) -> list[str]:
        """Ordered ``type_tag`` of each step."""
        return [s.type_tag for s in self.steps]

    def array_dict(self, prefix: str) -> dict[str, FloatArray]:
        """Flatten step parameters."""
        out: dict[str, FloatArray] = {}
        for i, step in enumerate(self.steps):
            for name, value in step.arrays().items():
                out[f"{prefix}.{i}.{name}"] = value
        return out

    @classmethod
    def from_serialized(cls, prefix: str, tags: list[str], arrays: Mapping[str, FloatArray]) -> Pipeline:
        """Load a pipeline from an npz/dict of arrays by name prefix."""
        steps: list[Transform] = []
        for i, tag in enumerate(tags):
            sub = {key.rsplit(".", 1)[-1]: arrays[key] for key in arrays if key.startswith(f"{prefix}.{i}.")}
            steps.append(_REGISTRY[tag].from_arrays(sub))
        return cls(tuple(steps))
