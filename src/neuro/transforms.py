"""Composable fit/transform/inverse-transform maps for the NN predictor.

Both the per-channel scalers and the PCA projection are per-timestep-vector affine maps:
they are fitted on training data and applied (and inverted) identically at train and
inference time. The only reason they used to live in different places is incidental -- PCA
changes the channel dimension and was fitted at a different stage. This module unifies them
behind one small :class:`Pipeline` of transform steps.

``transform`` / ``inverse_transform`` are pure arithmetic, so they run unchanged on NumPy
arrays (dataset prep, inference) and JAX arrays (the differentiable decode inside the
training loss). CasADi symbols are handled by the caller (:mod:`neuro.nn_predictor_casadi`),
which reads the fitted parameters off these objects and applies them with ``ca.mtimes``.
"""

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
    """Invert :func:`zscore`, mapping standardized values back to raw units."""
    return z * scale + center  # ty: ignore[invalid-return-type]


@dataclass(frozen=True)
class Standardizer:
    """Affine per-channel (or global) standardization ``z = (x - center) / scale``.

    ``center``/``scale`` are shaped ``(C,)`` for per-channel scaling or ``(1,)`` for a single
    global scalar, so both broadcast against a ``(rows, C)`` batch. Subsumes the previous
    ``StandardScaler``/``RobustScaler`` usage via the ``kind`` chosen at :meth:`fit`.
    """

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
        """Fit ``center``/``scale`` from ``(rows, C)`` data.

        ``kind="standard"`` uses mean/population-std (matching ``StandardScaler``);
        ``kind="robust"`` uses median/IQR (matching ``RobustScaler``). With ``global_scaling``
        the statistics are pooled over all elements into a single scalar. Zero scales are set
        to 1 (as sklearn does) so constant channels pass through unchanged.
        """
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
        """Standardize ``x`` (last axis is channels)."""
        return zscore(x, self.center, self.scale)

    def inverse_transform(self, z: TMath) -> TMath:
        """Map standardized values back to raw units."""
        return unzscore(z, self.center, self.scale)

    def arrays(self) -> dict[str, FloatArray]:
        """Fitted parameter arrays, for serialization."""
        return {"center": self.center, "scale": self.scale}

    @classmethod
    def from_arrays(cls, a: Mapping[str, FloatArray]) -> Standardizer:
        """Rebuild from :meth:`arrays` output."""
        return cls(center=np.asarray(a["center"], dtype=np.float64), scale=np.asarray(a["scale"], dtype=np.float64))


@dataclass(frozen=True)
class PCAProjection:
    """Linear projection onto ``k`` PCA components: ``z = (x - mean) @ basis.T``.

    ``basis`` has orthonormal rows, shape ``(k, C)``; ``mean`` is the per-channel training mean,
    shape ``(C,)``. The inverse ``x = z @ basis + mean`` reconstructs the in-subspace part of
    the signal (the orthogonal residual is unrepresentable, by construction).
    """

    type_tag: ClassVar[str] = "pca"

    basis: FloatArray
    mean: FloatArray

    @classmethod
    def fit(cls, x: FloatArray, latent_dim: int) -> PCAProjection | None:
        """Fit a ``latent_dim``-component PCA on ``(rows, C)`` data.

        Returns ``None`` when ``latent_dim >= C`` (the projection would be a no-op), so callers
        can treat "no reduction" and "no PCA step" identically.
        """
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
        """Decode ``(..., k)`` latent components back to ``(..., C)`` channels."""
        return z @ self.basis + self.mean  # ty: ignore[invalid-return-type]

    def arrays(self) -> dict[str, FloatArray]:
        """Fitted parameter arrays, for serialization."""
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
    """Ordered composition of :class:`Transform` steps applied along the channel axis.

    ``transform`` folds the steps forward (raw -> model space); ``inverse_transform`` folds
    them in reverse (model space -> raw). Serialization is split between a list of step tags
    (structure) and a flat ``{f"{prefix}.{i}.{param}": array}`` dict (parameters), so the whole
    pipeline round-trips through the artifact's JSON sidecar plus its ``.npz`` array bundle.
    """

    steps: tuple[Transform, ...]

    def transform(self, x: TMath) -> TMath:
        """Apply every step in order (raw channels -> model space)."""
        for step in self.steps:
            x = step.transform(x)  # ty: ignore[invalid-argument-type]
        return x

    def inverse_transform(self, z: TMath) -> TMath:
        """Apply every step's inverse in reverse order (model space -> raw channels)."""
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
        """Ordered ``type_tag`` of each step (the serialized structure)."""
        return [s.type_tag for s in self.steps]

    def array_dict(self, prefix: str) -> dict[str, FloatArray]:
        """Flatten every step's parameter arrays under ``f"{prefix}.{i}.{param}"`` keys."""
        out: dict[str, FloatArray] = {}
        for i, step in enumerate(self.steps):
            for name, value in step.arrays().items():
                out[f"{prefix}.{i}.{name}"] = value
        return out

    @classmethod
    def from_serialized(cls, prefix: str, tags: list[str], arrays: Mapping[str, FloatArray]) -> Pipeline:
        """Rebuild a pipeline from :meth:`step_tags` and :meth:`array_dict` outputs."""
        steps: list[Transform] = []
        for i, tag in enumerate(tags):
            sub = {key.rsplit(".", 1)[-1]: arrays[key] for key in arrays if key.startswith(f"{prefix}.{i}.")}
            steps.append(_REGISTRY[tag].from_arrays(sub))
        return cls(tuple(steps))
