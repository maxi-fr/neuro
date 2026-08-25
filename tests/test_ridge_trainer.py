"""Seam 3 -- the generic ridge solve and the Ridge Trainer's build-time capability check."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from neuro.predictor.module import AutoregressiveMLP
from neuro.predictor.ridge import RidgeTrainer, ridge
from neuro.types import RidgeFittable

if TYPE_CHECKING:
    from neuro.types import FloatArray


def test_ridge_leaves_the_bias_column_unregularized() -> None:
    """On a diagonal G the solve decouples: the last column sees no ridge, the others do."""
    rng = np.random.default_rng(3)
    n, c = 5, 2
    diag = rng.uniform(1.0, 10.0, n)
    diag[-1] = 100.0  # large so the ridge would be visible if it applied
    lam = 5.0
    G = np.diag(diag)
    P = rng.standard_normal((n, c))

    A = ridge(G, P, lam)
    np.testing.assert_allclose(A[:, -1], P[-1] / diag[-1], rtol=1e-10, atol=1e-12)
    for i in range(n - 1):
        np.testing.assert_allclose(A[:, i], P[i] / (diag[i] + lam), rtol=1e-10, atol=1e-12)


def _mlp(depth: int) -> AutoregressiveMLP:
    """A small depth-``depth`` waveform MLP, the natural fittable/non-fittable pair."""
    return AutoregressiveMLP(
        n_y=2,
        n_u=2,
        horizon=3,
        n_channels=2,
        n_controls=2,
        hidden_size=4,
        depth=depth,
    )


def test_ridge_fittable_is_structural_on_the_mlp() -> None:
    """Depth-0 is linear end-to-end and carries the capability; hidden layers are not fittable."""
    assert isinstance(_mlp(0), RidgeFittable)
    assert not isinstance(_mlp(1), RidgeFittable)
    assert not isinstance(_mlp(2), RidgeFittable)


class _RecordingStub:
    """A minimal Ridge-Fittable model recording the calls the Trainer makes."""

    def __init__(self) -> None:
        self.design_calls: list[list[tuple[FloatArray, FloatArray]]] = []
        self.installed: FloatArray | None = None

    def design_normal_equations(
        self, trajectories: list[tuple[FloatArray, FloatArray]]
    ) -> tuple[FloatArray, FloatArray]:
        """Record the trajectories and return a fixed diagonal problem."""
        self.design_calls.append(trajectories)
        return np.array([[2.0, 0.0], [0.0, 1.0]]), np.array([[1.0], [0.5]])

    def install_readout(self, A: FloatArray) -> None:
        """Record the fitted readout."""
        self.installed = A


def test_ridge_trainer_fits_any_ridge_fittable_model() -> None:
    """``fit`` runs design -> ridge -> install with no knowledge of the model, and returns it."""
    model = _RecordingStub()
    trajs = [(np.zeros((5, 1)), np.zeros((5, 1)))]

    out = RidgeTrainer(ridge_lambda=0.1).fit(model, trajs)

    assert out is model
    assert model.design_calls == [trajs]
    want = np.linalg.solve(np.array([[2.1, 0.0], [0.0, 1.0]]), np.array([[1.0], [0.5]])).T
    assert model.installed is not None
    np.testing.assert_allclose(model.installed, want, rtol=1e-12, atol=1e-12)


def test_ridge_trainer_rejects_a_non_fittable_model() -> None:
    """A model without the capability fails at build time, before any fit runs."""
    model = _mlp(1)
    assert not isinstance(model, RidgeFittable)
    with pytest.raises(TypeError, match="Ridge-Fittable"):
        RidgeTrainer(ridge_lambda=0.0).fit(model, [])  # ty: ignore[invalid-argument-type] -- deliberately non-fittable
