from __future__ import annotations

from typing import Literal, Protocol, get_args, runtime_checkable

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
Float32Array = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.intp]
StrArray = npt.NDArray[np.str_]

Activation = Literal["relu", "tanh", "softplus"]
"""Names of the supported MLP activation functions."""

ACTIVATIONS: frozenset[str] = frozenset(get_args(Activation))
"""Registry validating an activation name at checkpoint-load time."""

Layers = tuple[tuple[FloatArray, FloatArray], ...]
"""Per-layer ``(weight (out, in), bias (out,))`` pairs in forward-pass order."""


@runtime_checkable
class RidgeFittable(Protocol):
    """Capability of a Predictor whose readout is linear in a feature vector it produces itself.

    A capability protocol, not a base-protocol member: it names the closed-form fit around the
    readout, so it stays independent of how the features are produced -- a depth-0 MLP and a
    depth-0 observable MLP are both linear end-to-end. ``is_linear`` would not express that, and
    a bare boolean would still leave the Ridge Trainer needing per-kind knowledge of how to
    extract features and where to write the result. The Trainer checks this capability at build
    time.
    """

    def design_normal_equations(
        self, trajectories: list[tuple[FloatArray, FloatArray]]
    ) -> tuple[FloatArray, FloatArray]:
        """Accumulate the normal equations from raw ``(u, y)`` trajectories: ``(G (f, f), P (f, c))``.

        The last feature column is the constant-1 bias, so the Ridge Trainer can leave it
        unregularized without knowing which feature it is.
        """
        ...

    def install_readout(self, A: FloatArray) -> None:
        """Write the closed-form-fitted readout ``A (c, f)``, bias column last, into the module."""
        ...
