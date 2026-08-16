from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from neuro.nn_training import load_trajectory, split_data_files
from neuro.transforms import PCAProjection, Pipeline, Standardizer

if TYPE_CHECKING:
    from neuro.config import ESNPredictorConfig
    from neuro.types import FloatArray


@dataclass(frozen=True)
class ESNTrainingData:
    """Trajectory splits and fitted feature pipelines shared by ESN training and sweeps."""

    train_trajs: list[tuple[FloatArray, FloatArray]]
    val_trajs: list[tuple[FloatArray, FloatArray]]
    y_pipeline: Pipeline
    u_pipeline: Pipeline
    in_dim: int


def prepare_training_data(cfg: ESNPredictorConfig, data_files: list[str]) -> ESNTrainingData:
    """Split ``data_files`` by trajectory, load them, and fit the y/u pipelines on the training split."""
    train_files, val_files = split_data_files(data_files, cfg.training.train_split)

    def load(files: list[str]) -> list[tuple[FloatArray, FloatArray]]:
        return [
            load_trajectory(
                f,
                cfg.simulation.n_steps,
                cfg.simulation.downsample,
                cfg.simulation.dt,
                cutoff_hz=cfg.simulation.cutoff_hz,
            )
            for f in files
        ]

    train_trajs = load(train_files)
    val_trajs = load(val_files)

    all_y_train = np.vstack([y for _, y in train_trajs])
    all_u_train = np.vstack([u for u, _ in train_trajs])

    y_std = Standardizer.fit(all_y_train, kind=cfg.training.scaler, global_scaling=cfg.training.global_scaling)
    u_std = Standardizer.fit(all_u_train, kind=cfg.training.scaler, global_scaling=cfg.training.global_scaling)

    if cfg.model.latent_dim is not None:
        y_std_data = y_std.transform(all_y_train)
        pca = PCAProjection.fit(y_std_data, latent_dim=cfg.model.latent_dim)
        if pca is None:
            msg = f"latent_dim {cfg.model.latent_dim} must be less than feature dimension {y_std_data.shape[1]}"
            raise ValueError(msg)
        y_pipeline = Pipeline((y_std, pca))
        n_latent = cfg.model.latent_dim
    else:
        y_pipeline = Pipeline((y_std,))
        n_latent = all_y_train.shape[1]

    return ESNTrainingData(
        train_trajs=train_trajs,
        val_trajs=val_trajs,
        y_pipeline=y_pipeline,
        u_pipeline=Pipeline((u_std,)),
        in_dim=n_latent + all_u_train.shape[1] + 1,
    )
