from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from tqdm import tqdm

from neuro.artifacts import evaluate_rollouts
from neuro.predictor.data import apply_to_blocks, prepare_datasets, transform_features
from neuro.predictor.losses import curriculum_mask, predictor_loss
from neuro.predictor.module import AutoregressiveMLP
from neuro.transforms import PCAProjection, Pipeline, Standardizer

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from torch import Tensor, nn

    from neuro.artifacts import RolloutNMSE
    from neuro.config import ModelConfig, NNPredictorConfig, TrainingConfig
    from neuro.predictor.artifact import MLPArtifact
    from neuro.predictor.data import Datasets
    from neuro.types import FloatArray, IntArray

_DU_WINDOWS = 8


@dataclass(frozen=True)
class TrainingResult:
    """Everything one training run produced. Nothing is written; the caller persists and plots.

    Attributes
    ----------
    artifact : MLPArtifact
        The best-validation-loss weights frozen together with the fitted transforms.
    train_losses, val_losses : list[float]
        Per-epoch loss, one entry per epoch actually run (early stopping shortens both).
    rollout : RolloutNMSE
        Free-run rollout NMSE on ``val_trajs``, per horizon step and pooled over the horizon.
    val_trajs : list[tuple[FloatArray, FloatArray]]
        The held-out ``(u, y)`` trajectories, kept whole so the caller can plot free runs.
    du_sensitivity : float
        Mean Frobenius norm of the rollout's Jacobian with respect to the future controls. A
        value near zero means the model predicts EEG while ignoring stimulation.
    """

    artifact: MLPArtifact
    train_losses: list[float]
    val_losses: list[float]
    rollout: RolloutNMSE
    val_trajs: list[tuple[FloatArray, FloatArray]]
    du_sensitivity: float

    def save(self, artifact_dir: Path) -> None:
        """Write ``model.npz`` and ``training_stats.json`` into ``artifact_dir``."""
        self.artifact.save(artifact_dir / "model")
        stats = {
            "train_loss": self.train_losses,
            "val_loss": self.val_losses,
            "nmse_rollout": self.rollout.pooled,
            "nmse_rollout_per_step": self.rollout.per_step.tolist(),
            "du_sensitivity": self.du_sensitivity,
        }
        (artifact_dir / "training_stats.json").write_text(json.dumps(stats, indent=2))


@dataclass(frozen=True)
class _Tensors:
    """Model-space inputs and standardized-channel targets ``(samples, horizon, n_eeg_channels)``."""

    X_train: Tensor
    Y_train: Tensor
    X_val: Tensor
    Y_val: Tensor


def _tensor(a: FloatArray, device: torch.device) -> Tensor:
    """Move a NumPy array onto ``device`` as a float64 tensor."""
    return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64, device=device)


def _fit_pipelines(data: Datasets, model_cfg: ModelConfig, training_cfg: TrainingConfig) -> tuple[Pipeline, Pipeline]:
    """Fit the raw -> model-space EEG and control transforms on the training history blocks."""
    n_channels, n_controls = data.n_channels, data.n_controls
    y_len = model_cfg.n_y * n_channels

    y_past = data.X_train[:, :y_len].reshape(-1, n_channels)
    u_past = data.X_train[:, y_len : y_len + model_cfg.n_u * n_controls].reshape(-1, n_controls)

    kind, global_scaling = training_cfg.scaler, training_cfg.global_scaling
    y_standardizer = Standardizer.fit(y_past, kind=kind, global_scaling=global_scaling)
    u_standardizer = Standardizer.fit(u_past, kind=kind, global_scaling=global_scaling)

    pca = (
        None
        if model_cfg.latent_dim is None
        else PCAProjection.fit(y_standardizer.transform(y_past), model_cfg.latent_dim)
    )
    y_steps = (y_standardizer,) if pca is None else (y_standardizer, pca)
    return Pipeline(y_steps), Pipeline((u_standardizer,))


def _warm_start_linear(
    model: AutoregressiveMLP,
    X_train_m: FloatArray,
    Y_train_s: FloatArray,
    n_eeg_channels: int,
    pca: PCAProjection | None,
) -> None:
    """Overwrite a depth-0 model's single layer with the exact 1-step least-squares solution."""
    Y_step1 = Y_train_s[:, :n_eeg_channels]
    Y_latent = Y_step1 if pca is None else pca.transform(Y_step1)

    y_len = model.n_y * Y_latent.shape[1]
    m = model.n_controls
    # The rollout shifts the control window before the first MLP call, so the 1-step input pairs
    # the past-EEG block with the control window starting one step later.
    X_1step = np.hstack([X_train_m[:, :y_len], X_train_m[:, y_len + m : y_len + (model.n_u + 1) * m]])

    features = np.hstack([X_1step, np.ones((X_1step.shape[0], 1))])
    weight_bias, *_ = np.linalg.lstsq(features, Y_latent, rcond=None)

    layer = cast("nn.Linear", model.layers[0])
    with torch.no_grad():
        layer.weight.copy_(torch.as_tensor(np.ascontiguousarray(weight_bias[:-1].T)))
        layer.bias.copy_(torch.as_tensor(weight_bias[-1]))


def _shuffled_batches(n_samples: int, batch_size: int, rng: np.random.Generator) -> Iterator[IntArray]:
    """Yield index batches covering one freshly shuffled pass over the training set."""
    indices = rng.permutation(n_samples)
    for start in range(0, n_samples, batch_size):
        yield indices[start : start + batch_size]


def _batch_loss(
    model: AutoregressiveMLP, x: Tensor, y: Tensor, step_mask: Tensor, w_psd: float
) -> tuple[Tensor, dict[str, float]]:
    """Roll ``x`` out and score it against the standardized-channel targets ``y``."""
    pred_traj = model(x).reshape(x.shape[0], model.horizon, model.n_channels)
    return predictor_loss(model, pred_traj, y, step_mask, w_psd)


def _fit(
    model: AutoregressiveMLP, tensors: _Tensors, cfg: TrainingConfig, seed: int
) -> tuple[list[float], list[float]]:
    """Run the curriculum training loop, leaving ``model`` holding the best-validation weights."""
    rng = np.random.default_rng(seed)
    device = tensors.X_train.device
    horizon, n_samples = model.horizon, tensors.X_train.shape[0]

    # A warm-started linear model already solves the L = 1 problem the curriculum starts from.
    start_epoch = cfg.curriculum_start_epoch if len(model.layers) == 1 else 0
    steps_per_epoch = (n_samples + cfg.batch_size - 1) // cfg.batch_size
    total_steps = max(steps_per_epoch * (cfg.epochs - start_epoch), 1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0.0)

    val_mask = torch.ones(horizon, dtype=torch.float64, device=device)
    best_val_loss = float("inf")
    # A torch module is mutable, so the best-so-far snapshot has to be a copy, not an alias.
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    train_losses: list[float] = []
    val_losses: list[float] = []

    pbar = tqdm(range(start_epoch, cfg.epochs), desc="Training MLP")
    for epoch in pbar:
        step_mask = curriculum_mask(
            epoch, horizon, cfg.curriculum_max_steps, cfg.curriculum_start_epoch, cfg.curriculum_end_epoch
        ).to(device)

        epoch_loss, batches = 0.0, 0
        comps = {"mse": 0.0, "psd": 0.0}
        for idx in _shuffled_batches(n_samples, cfg.batch_size, rng):
            loss, parts = _batch_loss(model, tensors.X_train[idx], tensors.Y_train[idx], step_mask, cfg.w_psd)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += float(loss.detach())
            for key in comps:
                comps[key] += parts[key]
            batches += 1

        train_loss = epoch_loss / batches
        with torch.no_grad():
            val_loss = float(_batch_loss(model, tensors.X_val, tensors.Y_val, val_mask, cfg.w_psd)[0])
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if np.isnan(train_loss) or np.isnan(val_loss):
            msg = "Loss is NaN. Aborting training."
            raise ValueError(msg)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        pbar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
            mse=f"{comps['mse'] / batches:.4f}",
            psd=f"{comps['psd'] / batches:.4f}",
            L=int(step_mask.sum()),
        )

        if epochs_without_improvement >= cfg.patience:
            break

    model.load_state_dict(best_state)
    return train_losses, val_losses


def _du_sensitivity(model: AutoregressiveMLP, X_val: Tensor) -> float:
    """Mean Frobenius norm of d(rollout)/d(future controls) over a fixed subsample of windows.

    Forward mode is the cheap direction: the future-control block is far narrower than the
    ``horizon * n_channels`` rollout it drives, and one Jacobian per validation window would
    dominate the training run.
    """
    n_hist = model.n_y * model.n_channels + model.n_u * model.n_controls
    rows = X_val[:: max(X_val.shape[0] // _DU_WINDOWS, 1)][:_DU_WINDOWS]

    norms = []
    for row in rows:

        def rollout(u_future: Tensor, history: Tensor = row[:n_hist]) -> Tensor:
            return model(torch.cat([history, u_future])[None, :])[0]

        jacobian = torch.autograd.functional.jacobian(rollout, row[n_hist:], strategy="forward-mode", vectorize=True)
        norms.append(float(torch.linalg.norm(cast("Tensor", jacobian)).detach()))
    return float(np.mean(norms))


def train(cfg: NNPredictorConfig, data_files: list[str], *, seed_offset: int = 0) -> TrainingResult:
    """Train the autoregressive MLP for one config and return everything the run produced.

    Performs no I/O: the caller decides what to persist (:meth:`TrainingResult.save`) and plot.
    ``training.seed + seed_offset`` drives both the weight initialisation and the epoch shuffle,
    so a run reproduces exactly; ``seed_offset`` decorrelates otherwise-identical sweep trials.

    Parameters
    ----------
    cfg : NNPredictorConfig
        Validated configuration with any sweep overrides already applied by the caller.
    data_files : list[str]
        Paths to the ``.npz`` trajectory files, split into train/validation by trajectory.
    seed_offset : int, optional
        Added to ``training.seed``. Defaults to 0.

    Returns
    -------
    TrainingResult
        The best-validation artifact, the loss curves, the free-run rollout NMSE, the held-out
        trajectories and the control sensitivity.
    """
    sim, mdl, trn = cfg.simulation, cfg.model, cfg.training
    seed = trn.seed + seed_offset
    torch.manual_seed(seed)
    device = torch.device(trn.device)

    data = prepare_datasets(
        data_files,
        sim.n_steps,
        sim.downsample,
        mdl.n_y,
        mdl.n_u,
        mdl.horizon,
        sim.dt,
        trn.train_split,
        cutoff_hz=sim.cutoff_hz,
    )
    y_pipeline, u_pipeline = _fit_pipelines(data, mdl, trn)
    pca = y_pipeline.pca

    # Inputs go to model space; targets stay in standardized-channel space, where the loss lives.
    X_train_m = transform_features(data.X_train, y_pipeline, u_pipeline, mdl.n_y, data.n_channels, data.n_controls)
    X_val_m = transform_features(data.X_val, y_pipeline, u_pipeline, mdl.n_y, data.n_channels, data.n_controls)
    y_scale = Pipeline(y_pipeline.steps[:1])
    Y_train_s = apply_to_blocks(data.Y_train, y_scale, data.n_channels)
    Y_val_s = apply_to_blocks(data.Y_val, y_scale, data.n_channels)

    model = AutoregressiveMLP(
        n_y=mdl.n_y,
        n_u=mdl.n_u,
        horizon=mdl.horizon,
        n_channels=data.n_channels if pca is None else pca.basis.shape[0],
        n_controls=data.n_controls,
        hidden_size=mdl.hidden_size,
        depth=mdl.depth,
        activation=mdl.activation,
        decode_basis=None if pca is None else pca.basis,
        decode_mean=None if pca is None else pca.mean,
    )
    if mdl.depth == 0:
        _warm_start_linear(model, X_train_m, Y_train_s, data.n_channels, pca)

    model = model.to(device)
    target_shape = (-1, mdl.horizon, data.n_channels)
    tensors = _Tensors(
        X_train=_tensor(X_train_m, device),
        Y_train=_tensor(Y_train_s, device).reshape(target_shape),
        X_val=_tensor(X_val_m, device),
        Y_val=_tensor(Y_val_s, device).reshape(target_shape),
    )

    train_losses, val_losses = _fit(model, tensors, trn, seed)

    art = model.to_artifact(sim.dt * sim.downsample, sim.downsample, y_pipeline, u_pipeline)
    return TrainingResult(
        artifact=art,
        train_losses=train_losses,
        val_losses=val_losses,
        rollout=evaluate_rollouts(art, data.val_trajs, mdl.horizon),
        val_trajs=data.val_trajs,
        du_sensitivity=_du_sensitivity(model, tensors.X_val),
    )
