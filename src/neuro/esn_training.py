from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from neuro.predictor.data import fit_standardizers

if TYPE_CHECKING:
    from neuro.config import ESNPredictorConfig
    from neuro.transforms import Standardizer
    from neuro.types import FloatArray


@dataclass(frozen=True)
class ESNTrainingData:
    """Trajectory splits and fitted standardizers shared by ESN training and sweeps."""

    train_trajs: list[tuple[FloatArray, FloatArray]]
    val_trajs: list[tuple[FloatArray, FloatArray]]
    y_std: Standardizer
    u_std: Standardizer
    in_dim: int


def prepare_training_data(cfg: ESNPredictorConfig, data_files: list[str]) -> ESNTrainingData:
    """Split ``data_files`` by trajectory, load them, and fit the y/u standardizers on the training split."""
    split = fit_standardizers(
        data_files,
        n_steps_cfg=cfg.simulation.n_steps,
        downsample=cfg.simulation.downsample,
        dt=cfg.simulation.dt,
        train_split=cfg.training.train_split,
        scaler=cfg.training.scaler,
        global_scaling=cfg.training.global_scaling,
        cutoff_hz=cfg.simulation.cutoff_hz,
    )

    return ESNTrainingData(
        train_trajs=split.train_trajs,
        val_trajs=split.val_trajs,
        y_std=split.y_std,
        u_std=split.u_std,
        in_dim=split.n_channels + split.n_controls + 1,
    )
