from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from neuro.predictor.data import Datasets, prepare_datasets, prepare_observable_datasets
from neuro.predictor.evaluation import (
    LogEnergyError,
    ObservableFrameMSE,
    RolloutNMSE,
    evaluate_free_run,
    evaluate_observable_free_run,
    free_run_stats,
)
from neuro.predictor.gradient import fit_gradient_descent, float32_tensor
from neuro.predictor.inference import ObservableMLPModel, WaveformMLPModel
from neuro.predictor.losses import LossContext, build_losses, total_loss
from neuro.predictor.module import AutoregressiveMLP
from neuro.predictor.ridge import RidgeTrainer, RidgeTrainingResult
from neuro.provenance import training_provenance

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor, nn

    from neuro.config import NNPredictorConfig, StftGeometry
    from neuro.predictor.losses import Loss
    from neuro.types import FloatArray

_DU_WINDOWS = 8
_DU_PROBES = 5


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
    free_run : RolloutNMSE | ObservableFrameMSE
        Free-run error on ``val_trajs``, per step and pooled: rollout NMSE on the waveform kind,
        log-power Frame MSE on the observable kind.
    log_energy : LogEnergyError | None
        Free-run windowed-energy log-ratio error on ``val_trajs`` -- the error in the functional
        the MPC costs, which unlike NMSE keeps separating models past the phase horizon. ``None``
        on the observable kind, whose Cost reads Frames the free run already scores directly.
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
    free_run: RolloutNMSE | ObservableFrameMSE
    log_energy: LogEnergyError | None
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
            "du_sensitivity": self.du_sensitivity,
            **free_run_stats(self.free_run, self.log_energy),
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


def _du_sensitivity(model: AutoregressiveMLP, X_val: Tensor, *, n_probes: int = _DU_PROBES) -> float:
    """Mean Frobenius norm of d(rollout)/d(future controls) estimated via reverse-mode VJPs.

    Uses the Hutchinson trace estimator: for random Gaussian projections ``v ~ N(0, I)``,
    ``E[||J^T v||^2] = ||J||_F^2``. A few reverse-mode vector-Jacobian products per validation
    window estimate the full-horizon sensitivity in milliseconds without materializing the
    multi-gigabyte Jacobian tensor.
    """
    n_hist = model.n_y * model.n_outputs + model.n_u * model.n_controls
    rows = X_val[:: max(X_val.shape[0] // _DU_WINDOWS, 1)][:_DU_WINDOWS]

    norms: list[float] = []
    for row in rows:
        u_future = row[n_hist:].clone().detach().requires_grad_(requires_grad=True)
        history = row[:n_hist]
        x_in = torch.cat([history, u_future])[None, :]
        out = model(x_in)[0]

        probe_sq: list[float] = []
        for _ in range(n_probes):
            v = torch.randn_like(out)
            grad = torch.autograd.grad(out, u_future, grad_outputs=v, retain_graph=True)[0]
            probe_sq.append(float(torch.sum(grad**2).detach()))
        norms.append(float(np.sqrt(np.mean(probe_sq))))
    return float(np.mean(norms))


def train(
    cfg: NNPredictorConfig, data_files: list[str], *, seed_offset: int = 0
) -> TrainingResult | RidgeTrainingResult:
    """Train the Predictor named by ``cfg`` for one config and return everything the run produced.

    Dispatches on ``training.fit``: an NN config with ``training.fit: ridge`` routes to the Ridge
    Trainer for the depth-0 waveform MLP; any other NN config runs the generic gradient-descent
    fit. A fit the configured model does not support fails here at build time, before data is
    loaded or any fit runs. Every arm returns a result holding the trained Predictor, the recorded
    ``candidates`` and a ``save`` that persists the numpy-checkpoint and the training stats.

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
    TrainingResult | RidgeTrainingResult
        The trained Predictor, the candidate objectives, the free-run scores, the held-out
        trajectories and, on the gradient-descent arm, the loss curves and control sensitivity.

    Raises
    ------
    ValueError
        If the named fit is one the configured model does not support: ``ridge`` on an MLP with
        hidden layers.
    """
    if cfg.observable is not None:
        if cfg.training.fit == "ridge":
            return _train_observable_ridge(cfg, data_files, cfg.observable)
        return _train_observable(cfg, data_files, cfg.observable, seed_offset=seed_offset)
    if cfg.training.fit == "ridge":
        return _train_ridge(cfg, data_files)
    return _train_waveform(cfg, data_files, seed_offset=seed_offset)


def _prepare_observable(
    cfg: NNPredictorConfig, data_files: list[str], geom: StftGeometry, *, depth: int
) -> tuple[Datasets, AutoregressiveMLP, list[Loss]]:
    """Build the prepared datasets and the autoregressive observable MLP for ``cfg``."""
    sim, mdl, trn = cfg.simulation, cfg.model, cfg.training
    if trn.losses is None:
        msg = "the observable arm requires 'training.losses' (for the curriculum MSE)."
        raise ValueError(msg)
    fs = cfg.fs
    fs_frame = geom.frame_rate(fs)
    losses = build_losses(trn.losses, fs_frame)
    horizon = max(loss.span_steps for loss in losses)
    data = prepare_observable_datasets(
        data_files,
        sim.n_steps,
        sim.downsample,
        mdl.n_y,
        mdl.n_u,
        horizon,
        sim.dt,
        trn.train_split,
        geom,
        scaler=trn.scaler,
        global_scaling=trn.global_scaling,
        cutoff_hz=sim.cutoff_hz,
    )
    n_values = geom.n_values(fs)
    n_outputs = data.n_channels * n_values
    model = AutoregressiveMLP(
        n_y=mdl.n_y,
        n_u=mdl.n_u,
        horizon=horizon,
        n_channels=data.n_channels,
        n_controls=data.n_controls,
        n_outputs=n_outputs,
        hidden_size=mdl.hidden_size,
        depth=depth,
        activation=mdl.activation,
        residual=mdl.residual,
        dt=sim.dt * sim.downsample * geom.n_hop,
        y_std=data.y_std,
        u_std=data.u_std,
        geometry=geom,
    )
    return data, model, losses


def _train_observable_ridge(cfg: NNPredictorConfig, data_files: list[str], geom: StftGeometry) -> RidgeTrainingResult:
    """Fit the single layer of a depth-0 observable MLP by closed-form ridge."""
    if cfg.model.depth > 0:
        msg = f"'training.fit: ridge' requires a depth-0 MLP, got model.depth = {cfg.model.depth}."
        raise ValueError(msg)
    sim, trn = cfg.simulation, cfg.training
    fs_frame = geom.frame_rate(cfg.fs)
    data, model, _ = _prepare_observable(cfg, data_files, geom, depth=0)
    RidgeTrainer(ridge_lambda=trn.ridge_lambda).fit(model, data.train_trajs)

    model.provenance = training_provenance(data_files, sim.cutoff_hz)
    model.downsample = sim.downsample
    eval_steps = max(1, round(trn.eval_horizon_s * fs_frame))
    inference = ObservableMLPModel.from_checkpoint(*model.to_checkpoint())
    frame_mse = evaluate_observable_free_run(inference, data.val_trajs, eval_steps)

    with torch.no_grad():
        val_pred = (
            model(torch.as_tensor(data.X_val, dtype=torch.float32)).numpy().reshape(-1, model.horizon, model.n_outputs)
        )
    val_true = data.Y_val.reshape(-1, model.horizon, model.n_outputs)
    val_frame_loss = float(np.mean((val_pred - val_true) ** 2))

    return RidgeTrainingResult(
        predictor=model,
        candidates={
            "val_loss": val_frame_loss,
            "val_log_mse": frame_mse.pooled,
        },
        free_run=frame_mse,
        log_energy=None,
        val_trajs=data.val_trajs,
    )


def _train_observable(
    cfg: NNPredictorConfig, data_files: list[str], geom: StftGeometry, *, seed_offset: int = 0
) -> TrainingResult:
    """Train the autoregressive observable MLP for one config and return everything the run produced."""
    sim, trn = cfg.simulation, cfg.training
    seed = trn.seed + seed_offset
    torch.manual_seed(seed)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if trn.device == "auto"
        else torch.device(trn.device)
    )
    fs_frame = geom.frame_rate(cfg.fs)

    data, model, losses = _prepare_observable(cfg, data_files, geom, depth=cfg.model.depth)
    model = model.to(device)
    horizon = max(loss.span_steps for loss in losses)
    n_outputs = model.n_outputs

    pin = device.type == "cuda"
    cpu = torch.device("cpu")
    tensors = _Tensors(
        X_train=float32_tensor(data.X_train, cpu, pin_memory=pin),
        Y_train=float32_tensor(data.Y_train, cpu, pin_memory=pin).reshape(-1, horizon, n_outputs),
        X_val=float32_tensor(data.X_val, cpu, pin_memory=pin),
        Y_val=float32_tensor(data.Y_val, cpu, pin_memory=pin).reshape(-1, horizon, n_outputs),
        y_center=float32_tensor(data.y_std.center, device),
        y_scale=float32_tensor(data.y_std.scale, device),
    )

    def batch_loss(model: nn.Module, x: Tensor, y: Tensor, epoch: int | None) -> tuple[Tensor, dict[str, float]]:
        """Roll ``x`` out and score it against the standardized-frame targets ``y``."""
        ctx = LossContext(y_center=tensors.y_center, y_scale=tensors.y_scale, fs=fs_frame, epoch=epoch)
        pred_traj = model(x).reshape(x.shape[0], horizon, n_outputs)
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
        desc="Training Observable MLP",
    )

    eval_steps = max(1, round(trn.eval_horizon_s * fs_frame))
    model = model.cpu()
    du_sensitivity = _du_sensitivity(model, tensors.X_val)
    model.provenance = training_provenance(data_files, sim.cutoff_hz)
    model.downsample = sim.downsample
    inference = ObservableMLPModel.from_checkpoint(*model.to_checkpoint())
    frame_mse = evaluate_observable_free_run(inference, data.val_trajs, eval_steps)
    return TrainingResult(
        predictor=model,
        candidates={
            "val_loss": float(min(val_losses)),
            "val_log_mse": frame_mse.pooled,
        },
        train_losses=train_losses,
        val_losses=val_losses,
        train_components=train_comps,
        val_components=val_comps,
        free_run=frame_mse,
        log_energy=None,
        val_trajs=data.val_trajs,
        du_sensitivity=du_sensitivity,
    )


def _train_ridge(cfg: NNPredictorConfig, data_files: list[str]) -> RidgeTrainingResult:
    """Route an NN config naming ``training.fit: ridge`` to the Ridge Trainer.

    The Ridge Trainer serves the depth-0 waveform MLP. A config whose model carries hidden
    layers is not Ridge-Fittable and fails here at build time, before data is loaded or any
    fit runs.
    """
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
        n_outputs=data.n_channels,
        hidden_size=mdl.hidden_size,
        depth=depth,
        activation=mdl.activation,
        residual=mdl.residual,
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

    model.provenance = training_provenance(data_files, sim.cutoff_hz)
    model.downsample = sim.downsample
    eval_steps = max(1, round(trn.eval_horizon_s * fs))
    inference = WaveformMLPModel.from_checkpoint(*model.to_checkpoint())
    rollout, log_energy = evaluate_free_run(inference, data.val_trajs, eval_steps, fs)
    return RidgeTrainingResult(
        predictor=model,
        candidates={
            "rollout_nmse": rollout.pooled,
            "log_energy": log_energy.pooled,
        },
        free_run=rollout,
        log_energy=log_energy,
        val_trajs=data.val_trajs,
    )


def _train_waveform(cfg: NNPredictorConfig, data_files: list[str], *, seed_offset: int = 0) -> TrainingResult:
    """Train the autoregressive waveform MLP for one config and return everything the run produced."""
    sim, trn = cfg.simulation, cfg.training
    seed = trn.seed + seed_offset
    torch.manual_seed(seed)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if trn.device == "auto"
        else torch.device(trn.device)
    )
    fs = cfg.fs

    data, model, losses = _prepare_waveform(cfg, data_files, depth=cfg.model.depth)
    model = model.to(device)
    horizon = max(loss.span_steps for loss in losses)

    pin = device.type == "cuda"
    cpu = torch.device("cpu")
    tensors = _Tensors(
        X_train=float32_tensor(data.X_train, cpu, pin_memory=pin),
        Y_train=float32_tensor(data.Y_train, cpu, pin_memory=pin).reshape(-1, horizon, data.n_channels),
        X_val=float32_tensor(data.X_val, cpu, pin_memory=pin),
        Y_val=float32_tensor(data.Y_val, cpu, pin_memory=pin).reshape(-1, horizon, data.n_channels),
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
    model = model.cpu()
    du_sensitivity = _du_sensitivity(model, tensors.X_val)
    # Free-run scoring runs on the deployed jax side, built in memory from the fitted torch model.
    model.provenance = training_provenance(data_files, sim.cutoff_hz)
    model.downsample = sim.downsample
    inference = WaveformMLPModel.from_checkpoint(*model.to_checkpoint())
    rollout, log_energy = evaluate_free_run(inference, data.val_trajs, eval_steps, fs)
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
        free_run=rollout,
        log_energy=log_energy,
        val_trajs=data.val_trajs,
        du_sensitivity=du_sensitivity,
    )
