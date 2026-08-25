from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from neuro.metrics import DEFAULT_HOP_S, METRICS
from neuro.predictor.data import prepare_datasets
from neuro.predictor.evaluation import evaluate_log_energy, evaluate_rollouts
from neuro.predictor.gradient import fit_gradient_descent, float32_tensor
from neuro.predictor.losses import CurriculumMSE, LossContext, build_losses, total_loss
from neuro.predictor.module import AutoregressiveMLP
from neuro.predictor.observable_train import ObservableTrainingResult, _train_observable
from neuro.provenance import training_provenance

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor, nn

    from neuro.config import NNPredictorConfig
    from neuro.predictor.evaluation import LogEnergyError, RolloutNMSE
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


def _warm_start_linear(
    model: AutoregressiveMLP,
    X_train: FloatArray,
    Y_train: FloatArray,
    n_channels: int,
) -> None:
    """Overwrite a depth-0 model's single layer with the exact 1-step least-squares solution."""
    Y_step1 = Y_train[:, :n_channels]
    y_len = model.n_y * n_channels
    m = model.n_controls
    # The rollout shifts the control window before the first MLP call, so the 1-step input pairs
    # the past-EEG block with the control window starting one step later.
    X_1step = np.hstack([X_train[:, :y_len], X_train[:, y_len + m : y_len + (model.n_u + 1) * m]])

    X_design = np.hstack([X_1step, np.ones((X_1step.shape[0], 1))])
    weight_bias, *_ = np.linalg.lstsq(X_design, Y_step1, rcond=None)

    layer = cast("nn.Linear", model.layers[0])
    with torch.no_grad():
        layer.weight.copy_(torch.as_tensor(np.ascontiguousarray(weight_bias[:-1].T), dtype=torch.float32))
        layer.bias.copy_(torch.as_tensor(weight_bias[-1], dtype=torch.float32))


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
    cfg: NNPredictorConfig, data_files: list[str], *, seed_offset: int = 0
) -> TrainingResult | ObservableTrainingResult:
    """Train the NN predictor named by ``cfg`` for one config and return everything the run produced.

    Dispatches on the config tree: a config carrying an ``observable`` block trains the
    one-Frame-per-step observable predictor, any other trains the autoregressive waveform MLP.
    Both arms run the same generic gradient-descent fit and return a result holding the trained
    Predictor, the recorded ``candidates`` and a ``save`` that persists the numpy-checkpoint and
    the training stats.

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
    TrainingResult | ObservableTrainingResult
        The trained Predictor, the candidate objectives, the loss curves, the free-run scores, the
        held-out trajectories and the control sensitivity.
    """
    if cfg.observable is not None:
        return _train_observable(cfg, data_files, seed_offset=seed_offset)
    return _train_waveform(cfg, data_files, seed_offset=seed_offset)


def _train_waveform(cfg: NNPredictorConfig, data_files: list[str], *, seed_offset: int = 0) -> TrainingResult:
    """Train the autoregressive waveform MLP for one config and return everything the run produced."""
    sim, mdl, trn = cfg.simulation, cfg.model, cfg.training
    if trn.losses is None:
        msg = "the waveform arm requires 'training.losses'; a config with an 'observable' block is routed elsewhere."
        raise ValueError(msg)
    seed = trn.seed + seed_offset
    torch.manual_seed(seed)
    device = torch.device(trn.device)
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
        depth=mdl.depth,
        activation=mdl.activation,
        dt=sim.dt * sim.downsample,
        y_std=data.y_std,
        u_std=data.u_std,
    )
    if mdl.depth == 0:
        _warm_start_linear(model, data.X_train, data.Y_train, data.n_channels)

    model = model.to(device)

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

    # A warm-started linear model already solves the L = 1 problem the curriculum starts from.
    curr_mse = next((loss for loss in losses if isinstance(loss, CurriculumMSE)), None)
    start_epoch = curr_mse.curr_start if (len(model.layers) == 1 and curr_mse is not None) else 0
    train_losses, val_losses, train_comps, val_comps = fit_gradient_descent(
        model,
        tensors.X_train,
        tensors.Y_train,
        tensors.X_val,
        tensors.Y_val,
        trn,
        seed=seed,
        loss_fn=batch_loss,
        start_epoch=start_epoch,
        desc="Training MLP",
    )

    eval_steps = max(1, round(trn.eval_horizon_s * fs))
    # The energy course follows the metrics layer's own eeg_ms convention rather than a knob of its
    # own, clamped where the evaluation horizon is too short to hold one window.
    energy_window = min(max(1, round(METRICS["eeg_ms"].window_s * fs)), eval_steps)
    energy_hop = max(1, round(DEFAULT_HOP_S * fs))
    du_sensitivity = _du_sensitivity(model, tensors.X_val)
    # The protocol runtime interfaces through NumPy, so the free-run evaluation runs on the CPU copy.
    model = model.cpu()
    model.provenance = training_provenance(data_files, sim.cutoff_hz)
    model.downsample = sim.downsample
    rollout = evaluate_rollouts(model, data.val_trajs, eval_steps)
    log_energy = evaluate_log_energy(
        model, data.val_trajs, eval_steps, window_steps=energy_window, hop_steps=energy_hop
    )
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
