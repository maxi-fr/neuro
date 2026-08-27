from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from neuro.predictor.evaluation import RolloutNMSE, free_run_stats
from neuro.types import RidgeFittable

if TYPE_CHECKING:
    from pathlib import Path

    from neuro.predictor.evaluation import LogEnergyError, ObservableFrameMSE
    from neuro.predictor.module import AutoregressiveMLP
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


@dataclass(frozen=True)
class RidgeTrainingResult:
    """Everything one closed-form Ridge training run produced; ``save`` persists it all.

    Owned by the depth-0 waveform MLP arm whose free-run scores live on the sample grid, which
    returns exactly this shape of result: a fitted Predictor, the free-run scores, and no
    training curve. The absence of ``val_loss`` in ``candidates`` is deliberate: a closed-form
    fit has no epoch loop, so the only objectives this arm can rank on are the two free-run
    scores, ``rollout_nmse`` and ``log_energy``.

    Attributes
    ----------
    predictor : AutoregressiveMLP
        The trained module holding the fitted readout, with the standardizers as buffers and the
        recorded metadata (provenance, downsample) attached.
    candidates : dict[str, float]
        Every objective the sweep seam can rank this run on: ``rollout_nmse`` and ``log_energy``,
        both lower-is-better.
    free_run : RolloutNMSE | ObservableFrameMSE
        Free-run error on ``val_trajs``, per step and pooled: rollout NMSE on the waveform kind,
        log-power Frame MSE on the observable kind.
    log_energy : LogEnergyError | None
        Free-run windowed-energy log-ratio error on ``val_trajs``. ``None`` on the observable kind.
    val_trajs : list[tuple[FloatArray, FloatArray]]
        The held-out ``(u, y)`` trajectories, kept whole so the caller can plot free runs.
    """

    predictor: AutoregressiveMLP
    candidates: dict[str, float]
    free_run: RolloutNMSE | ObservableFrameMSE
    log_energy: LogEnergyError | None
    val_trajs: list[tuple[FloatArray, FloatArray]]

    def save(self, artifact_dir: Path) -> None:
        """Write the numpy-checkpoint and ``training_stats.json`` into ``artifact_dir``."""
        self.predictor.save(artifact_dir / "model")
        stats: dict[str, object] = dict(free_run_stats(self.free_run, self.log_energy))
        if "val_loss" in self.candidates:
            stats["val_loss"] = self.candidates["val_loss"]
        (artifact_dir / "training_stats.json").write_text(json.dumps(stats, indent=2))
