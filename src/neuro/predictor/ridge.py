from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from neuro.types import RidgeFittable

if TYPE_CHECKING:
    from neuro.types import FloatArray


def ridge(G: FloatArray, P: FloatArray, ridge_lambda: float) -> FloatArray:
    """Solve the normal-equation readout ``A = (G + lambda * I)^-1 P``, bias column last unregularized.

    ``G`` is ``(f, f)`` and ``P`` is ``(f, c)``; the last feature column of both is the constant-1
    bias, so its diagonal entry of ``G`` receives no ridge. Returns ``A (c, f)`` ready for
    :meth:`RidgeFittable.install_readout`.
    """
    g = np.asarray(G, dtype=np.float64)
    p = np.asarray(P, dtype=np.float64)
    reg = ridge_lambda * np.eye(g.shape[0], dtype=np.float64)
    reg[-1, -1] = 0.0
    return np.ascontiguousarray(np.linalg.solve(g + reg, p).T)


class RidgeTrainer:
    """Generic closed-form Trainer: fits the linear readout of any Ridge-Fittable Predictor.

    The Trainer holds no knowledge of which model it fits: ``fit`` asks the model for its normal
    equations, solves them with :func:`ridge`, and hands the readout back. A model that does not
    implement the :class:`~neuro.types.RidgeFittable` capability fails here, at build time, before
    any fit runs.
    """

    def __init__(self, ridge_lambda: float = 0.0) -> None:
        """Store the ridge regularization weight; the bias column stays unregularized."""
        self.ridge_lambda = float(ridge_lambda)

    def fit(
        self,
        model: RidgeFittable,
        trajectories: list[tuple[FloatArray, FloatArray]],
    ) -> RidgeFittable:
        """Fit ``model``'s readout: ``G, P = model.design_normal_equations(trajs); A = ridge(G, P, lambda); model.install_readout(A)``.

        Returns the fitted model. Raises ``TypeError`` when ``model`` is not Ridge-Fittable.
        """
        if not isinstance(model, RidgeFittable):
            msg = f"RidgeTrainer requires a Ridge-Fittable model, got {type(model).__name__}."
            raise TypeError(msg)
        G, P = model.design_normal_equations(trajectories)
        A = ridge(G, P, self.ridge_lambda)
        model.install_readout(A)
        return model
