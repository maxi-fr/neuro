from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from neuro.config import ESNPredictorConfig
from neuro.predictor.data import Datasets, prepare_datasets
from neuro.predictor.esn_train import _train_esn
from neuro.predictor.evaluation import evaluate_free_run
from neuro.predictor.gradient import fit_gradient_descent, float32_tensor
from neuro.predictor.losses import LossContext, build_losses, total_loss
from neuro.predictor.module import AutoregressiveMLP
from neuro.predictor.observable_train import (
    ObservableTrainingResult,
    _train_observable,
    _train_observable_ridge,
)
from neuro.predictor.ridge import RidgeTrainer, RidgeTrainingResult
from neuro.provenance import training_provenance

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor, nn

    from neuro.config import NNPredictorConfig
    from neuro.predictor.evaluation import LogEnergyError, RolloutNMSE
    from neuro.predictor.losses import Loss
    from neuro.types import FloatArray

_DU_WINDOWS = 8


@dataclass(frozen=True)
class TrainingResult:
    """Everything one gradient-descent training run produced; ``save`` persists it all.

    Attributes
    ----------
    predictor : AutoregressiveMLP
        The trained module holding the best-validation-loss weights, with the standardizers as
        buffers and the recorded metadata (provenance, downsample) attached.
    candidates : dict[str, float]
        Every objective the sweep seam can rank this run on: ``log_energy``, ``val_loss`` and
        ``rollout_nmse``, all lower-is-better.
    train_losses, val_losses : list[float]
        Per-epoch loss, one entry per epoch actually run (early stopping shortens both).
    train_components, val_components : dict[str, list[float]]
        Per-epoch unweighted loss components and diagnostics.
    rollout : RolloutNMSE
        Free-run rollout NMSE on ``val_trajs``, per horizon step and pooled over the horizon.
    log_energy : LogEnergyError
        Free-run windowed-energy log-ratio error on ``val_trajs`` -- the error in the functional
        the MPC costs, which unlike NMSE keeps separating models past the phase horizon.
    val_trajs : list[tuple[FloatArray, FloatArray]]
        The held-out ``(u, y)`` trajectories, kept whole so the caller can plot free runs.
    du_sensitivity : float
        Mean Frobenius norm of the rollout's Jacobian with respect to the future controls. A
        value near zero means the model predicts EEG while ignoring stimulation.
    """

    predictor: AutoregressiveMLP
    candidates: dict[str, float]
    train_losses: list[float]
    val_losses: list[float]
    train_components: dict[str, list[float]]
    val_components: dict[str, list[float]]
    rollout: RolloutNMSE
    log_energy: LogEnergyError
    val_trajs: list[tuple[FloatArray, FloatArray]]
    du_sensitivity: float

    def save(self, artifact_dir: Path) -> None:
        """Write the numpy-checkpoint and ``training_stats.json`` into ``artifact_dir``."""
        self.predictor.save(artifact_dir / "model")
        stats = {
            "train_loss": self.train_losses,
            "val_loss": self.val_losses,
            "train_components": self.train_components,
            "val_components": self.val_components,
            "nmse_rollout": self.rollout.pooled,
            "nmse_rollout_per_step": self.rollout.per_step.tolist(),
            "log_energy": self.log_energy.pooled,
            "log_energy_per_position": self.log_energy.per_position.tolist(),
            "du_sensitivity": self.du_sensitivity,
        }
        (artifact_dir / "training_stats.json").write_text(json.dumps(stats, indent=2))


@dataclass(frozen=True)
class _Tensors:
    """Model-space inputs and standardized-channel targets ``(samples, horizon, n_channels)``."""

    X_train: Tensor
    Y_train: Tensor
    X_val: Tensor
    Y_val: Tensor
    y_center: Tensor
    y_scale: Tensor


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


def train(
    cfg: NNPredictorConfig | ESNPredictorConfig, data_files: list[str], *, seed_offset: int = 0
) -> TrainingResult | ObservableTrainingResult | RidgeTrainingResult:
    """Train the Predictor named by ``cfg`` for one config and return everything the run produced.

    Dispatches on the config tree first, then on ``training.fit``: an ESN config always routes to
    the ESN arm (``ridge`` only); an NN config with ``training.fit: ridge`` routes to the Ridge
    Trainer, which serves the depth-0 waveform MLP and the depth-0 observable MLP; any other NN
    config runs the generic gradient-descent fit over the waveform or observable arm. A fit the
    configured model does not support fails here at build time, before data is loaded or any fit
    runs. Every arm returns a result holding the trained Predictor, the recorded ``candidates``
    and a ``save`` that persists the numpy-checkpoint and the training stats.

    Parameters
    ----------
    cfg : NNPredictorConfig | ESNPredictorConfig
        Validated configuration with any sweep overrides already applied by the caller.
    data_files : list[str]
        Paths to the ``.npz`` trajectory files, split into train/validation by trajectory.
    seed_offset : int, optional
        Added to ``training.seed``. Defaults to 0.

    Returns
    -------
    TrainingResult | ObservableTrainingResult | RidgeTrainingResult
        The trained Predictor, the candidate objectives, the free-run scores, the held-out
        trajectories and, on the gradient-descent arms, the loss curves and control sensitivity.

    Raises
    ------
    ValueError
        If the named fit is one the configured model does not support: ``ridge`` on an MLP with
        hidden layers, or ``gradient_descent`` on the ESN.
    """
    if isinstance(cfg, ESNPredictorConfig):
        return _train_esn(cfg, data_files, seed_offset=seed_offset)
    if cfg.training.fit == "ridge":
        return _train_ridge(cfg, data_files, seed_offset=seed_offset)
    if cfg.observable is not None:
        return _train_observable(cfg, data_files, seed_offset=seed_offset)
    return _train_waveform(cfg, data_files, seed_offset=seed_offset)


def _train_ridge(
    cfg: NNPredictorConfig, data_files: list[str], *, seed_offset: int = 0
) -> RidgeTrainingResult | ObservableTrainingResult:
    """Route an NN config naming ``training.fit: ridge`` to the Ridge Trainer.

    The Ridge Trainer serves the two Ridge-Fittable NN predictors: the depth-0 waveform MLP and
    the depth-0 observable MLP (``lift_depth = transition_depth = 0``, linear end-to-end). A
    config whose model carries hidden layers is not Ridge-Fittable and fails here at build time,
    before data is loaded or any fit runs.
    """
    if cfg.observable is not None:
        if cfg.observable.lift_depth > 0 or cfg.observable.transition_depth > 0:
            msg = (
                "'training.fit: ridge' requires a depth-0 observable MLP "
                f"(lift_depth = transition_depth = 0), got lift_depth = {cfg.observable.lift_depth} "
                f"and transition_depth = {cfg.observable.transition_depth}."
            )
            raise ValueError(msg)
        return _train_observable_ridge(cfg, data_files, seed_offset=seed_offset)
    if cfg.model.depth > 0:
        msg = f"'training.fit: ridge' requires a depth-0 MLP, got model.depth = {cfg.model.depth}."
        raise ValueError(msg)
    return _train_waveform_ridge(cfg, data_files)


def _prepare_waveform(
    cfg: NNPredictorConfig, data_files: list[str], *, depth: int
) -> tuple[Datasets, AutoregressiveMLP, list[Loss]]:
    """Build the prepared datasets and the autoregressive waveform MLP for ``cfg``.

    Shared by the two waveform arms, which differ only in ``depth`` (0 on the ridge arm,
    ``cfg.model.depth`` on the gradient-descent arm); the built losses ride along for the
    gradient arm's batch scoring.
    """
    sim, mdl, trn = cfg.simulation, cfg.model, cfg.training
    if trn.losses is None:
        msg = "the waveform arm requires 'training.losses' (for the native horizon)."
        raise ValueError(msg)
    fs = cfg.fs
    losses = build_losses(trn.losses, fs)
    horizon = max(loss.span_steps for loss in losses)
    data = prepare_datasets(
        data_files,
        sim.n_steps,
        sim.downsample,
        mdl.n_y,
        mdl.n_u,
        horizon,
        sim.dt,
        trn.train_split,
        scaler=trn.scaler,
        global_scaling=trn.global_scaling,
        cutoff_hz=sim.cutoff_hz,
    )
    model = AutoregressiveMLP(
        n_y=mdl.n_y,
        n_u=mdl.n_u,
        horizon=horizon,
        n_channels=data.n_channels,
        n_controls=data.n_controls,
        hidden_size=mdl.hidden_size,
        depth=depth,
        activation=mdl.activation,
        dt=sim.dt * sim.downsample,
        y_std=data.y_std,
        u_std=data.u_std,
    )
    return data, model, losses


def _train_waveform_ridge(cfg: NNPredictorConfig, data_files: list[str]) -> RidgeTrainingResult:
    """Fit the single layer of a depth-0 waveform MLP by closed-form ridge.

    The exact 1-step least-squares fit the gradient-descent arm no longer runs: the Ridge
    Trainer folds the same features and targets into normal equations from the raw training
    trajectories and installs the result as the single layer, the only closed form left. The
    native horizon comes from ``training.losses``, as on the gradient-descent arm.
    """
    sim, trn = cfg.simulation, cfg.training
    fs = cfg.fs
    data, model, _ = _prepare_waveform(cfg, data_files, depth=0)
    RidgeTrainer(ridge_lambda=trn.ridge_lambda).fit(model, data.train_trajs)

    eval_steps = max(1, round(trn.eval_horizon_s * fs))
    rollout, log_energy = evaluate_free_run(model, data.val_trajs, eval_steps, fs)
    model.provenance = training_provenance(data_files, sim.cutoff_hz)
    model.downsample = sim.downsample
    return RidgeTrainingResult(
        predictor=model,
        candidates={
            "rollout_nmse": rollout.pooled,
            "log_energy": log_energy.pooled,
        },
        rollout=rollout,
        log_energy=log_energy,
        val_trajs=data.val_trajs,
    )


def _train_waveform(cfg: NNPredictorConfig, data_files: list[str], *, seed_offset: int = 0) -> TrainingResult:
    """Train the autoregressive waveform MLP for one config and return everything the run produced."""
    sim, trn = cfg.simulation, cfg.training
    seed = trn.seed + seed_offset
    torch.manual_seed(seed)
    device = torch.device(trn.device)
    fs = cfg.fs

    data, model, losses = _prepare_waveform(cfg, data_files, depth=cfg.model.depth)
    model = model.to(device)
    horizon = max(loss.span_steps for loss in losses)

    tensors = _Tensors(
        X_train=float32_tensor(data.X_train, device),
        Y_train=float32_tensor(data.Y_train, device).reshape(-1, horizon, data.n_channels),
        X_val=float32_tensor(data.X_val, device),
        Y_val=float32_tensor(data.Y_val, device).reshape(-1, horizon, data.n_channels),
        y_center=float32_tensor(data.y_std.center, device),
        y_scale=float32_tensor(data.y_std.scale, device),
    )

    def batch_loss(model: nn.Module, x: Tensor, y: Tensor, epoch: int | None) -> tuple[Tensor, dict[str, float]]:
        """Roll ``x`` out and score it against the standardized-channel targets ``y``."""
        ctx = LossContext(y_center=tensors.y_center, y_scale=tensors.y_scale, fs=fs, epoch=epoch)
        pred_traj = model(x).reshape(x.shape[0], horizon, data.n_channels)
        return total_loss(losses, pred_traj, y, ctx)

    train_losses, val_losses, train_comps, val_comps = fit_gradient_descent(
        model,
        tensors.X_train,
        tensors.Y_train,
        tensors.X_val,
        tensors.Y_val,
        trn,
        seed=seed,
        loss_fn=batch_loss,
        desc="Training MLP",
    )

    eval_steps = max(1, round(trn.eval_horizon_s * fs))
    du_sensitivity = _du_sensitivity(model, tensors.X_val)
    # The protocol runtime interfaces through NumPy, so the free-run evaluation runs on the CPU copy.
    model = model.cpu()
    model.provenance = training_provenance(data_files, sim.cutoff_hz)
    model.downsample = sim.downsample
    rollout, log_energy = evaluate_free_run(model, data.val_trajs, eval_steps, fs)
    return TrainingResult(
        predictor=model,
        candidates={
            "log_energy": log_energy.pooled,
            "val_loss": float(min(val_losses)),
            "rollout_nmse": rollout.pooled,
        },
        train_losses=train_losses,
        val_losses=val_losses,
        train_components=train_comps,
        val_components=val_comps,
        rollout=rollout,
        log_energy=log_energy,
        val_trajs=data.val_trajs,
        du_sensitivity=du_sensitivity,
    )
