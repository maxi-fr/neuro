from __future__ import annotations

import collections
import copy
import dataclasses
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from tqdm import tqdm

from neuro.artifacts import evaluate_log_energy, evaluate_rollouts
from neuro.metrics import DEFAULT_HOP_S, METRICS
from neuro.predictor.data import prepare_datasets
from neuro.predictor.losses import CurriculumMSE, Loss, LossContext, build_losses, total_loss
from neuro.predictor.module import AutoregressiveMLP
from neuro.provenance import training_provenance

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from torch import Tensor, nn

    from neuro.artifacts import LogEnergyError, RolloutNMSE
    from neuro.config import NNPredictorConfig, TrainingConfig
    from neuro.predictor.artifact import MLPArtifact
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

    artifact: MLPArtifact
    train_losses: list[float]
    val_losses: list[float]
    train_components: dict[str, list[float]]
    val_components: dict[str, list[float]]
    rollout: RolloutNMSE
    log_energy: LogEnergyError
    val_trajs: list[tuple[FloatArray, FloatArray]]
    du_sensitivity: float

    def save(self, artifact_dir: Path) -> None:
        """Write ``model.npz`` and ``training_stats.json`` into ``artifact_dir``."""
        self.artifact.save(artifact_dir / "model")
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


def _tensor(a: FloatArray, device: torch.device) -> Tensor:
    """Move a NumPy array onto ``device`` as a float64 tensor."""
    return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64, device=device)


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
        layer.weight.copy_(torch.as_tensor(np.ascontiguousarray(weight_bias[:-1].T)))
        layer.bias.copy_(torch.as_tensor(weight_bias[-1]))


def _lr_schedule(
    optimizer: torch.optim.Optimizer, *, warmup_steps: int, total_steps: int
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear warm-up over ``warmup_steps`` batches, then cosine anneal to zero over the remainder.

    The rollout is ``max(span_steps)`` deep from the first batch, so a randomly initialised model
    backpropagates through the full horizon at epoch 0. Ramping in avoids taking that first,
    badly-conditioned gradient at the peak learning rate.
    """
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=0.0
    )
    if warmup_steps < 1:
        return cosine
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0 / warmup_steps, total_iters=warmup_steps)
    return torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])


def _shuffled_batches(n_samples: int, batch_size: int, rng: np.random.Generator) -> Iterator[IntArray]:
    """Yield index batches covering one freshly shuffled pass over the training set."""
    indices = rng.permutation(n_samples)
    for start in range(0, n_samples, batch_size):
        yield indices[start : start + batch_size]


def _batch_loss(
    model: AutoregressiveMLP, x: Tensor, y: Tensor, losses: Sequence[Loss], ctx: LossContext
) -> tuple[Tensor, dict[str, float]]:
    """Roll ``x`` out and score it against the standardized-channel targets ``y``."""
    pred_traj = model(x).reshape(x.shape[0], model.horizon, model.n_channels)
    return total_loss(losses, pred_traj, y, ctx)


def _fit(  # noqa: PLR0913, PLR0915
    model: AutoregressiveMLP,
    tensors: _Tensors,
    cfg: TrainingConfig,
    losses: Sequence[Loss],
    *,
    fs: float,
    seed: int,
) -> tuple[list[float], list[float], dict[str, list[float]], dict[str, list[float]]]:
    """Run the curriculum training loop, leaving ``model`` holding the best-validation weights."""
    rng = np.random.default_rng(seed)
    n_samples = tensors.X_train.shape[0]

    # A warm-started linear model already solves the L = 1 problem the curriculum starts from.
    curr_mse = next((loss for loss in losses if isinstance(loss, CurriculumMSE)), None)
    start_epoch = curr_mse.curr_start if (len(model.layers) == 1 and curr_mse is not None) else 0
    steps_per_epoch = (n_samples + cfg.batch_size - 1) // cfg.batch_size
    total_steps = max(steps_per_epoch * (cfg.epochs - start_epoch), 1)
    # A warm-started linear model skips ahead to curr_start, which can leave fewer epochs than the
    # configured warm-up asks for.
    warmup_steps = min(steps_per_epoch * cfg.warmup_epochs, total_steps - 1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = _lr_schedule(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)

    val_ctx = LossContext(y_center=tensors.y_center, y_scale=tensors.y_scale, fs=fs, epoch=None)
    best_val_loss = float("inf")
    # A torch module is mutable, so the best-so-far snapshot has to be a copy, not an alias.
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    train_losses: list[float] = []
    val_losses: list[float] = []
    train_components: dict[str, list[float]] = collections.defaultdict(list)
    val_components: dict[str, list[float]] = collections.defaultdict(list)

    pbar = tqdm(range(start_epoch, cfg.epochs), desc="Training MLP")
    for epoch in pbar:
        train_ctx = LossContext(y_center=tensors.y_center, y_scale=tensors.y_scale, fs=fs, epoch=epoch)

        epoch_loss, batches = 0.0, 0
        comps_sum: dict[str, float] = collections.defaultdict(float)
        for idx in _shuffled_batches(n_samples, cfg.batch_size, rng):
            loss, parts = _batch_loss(model, tensors.X_train[idx], tensors.Y_train[idx], losses, train_ctx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += float(loss.detach())
            for key, val in parts.items():
                comps_sum[key] += val
            batches += 1

        train_loss = epoch_loss / batches
        with torch.no_grad():
            val_loss_t, val_parts = _batch_loss(model, tensors.X_val, tensors.Y_val, losses, val_ctx)
            val_loss = float(val_loss_t.detach())

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        for loss_obj in losses:
            train_components[loss_obj.name].append(comps_sum[loss_obj.name] / batches)
            val_components[loss_obj.name].append(val_parts[loss_obj.name])

        if np.isnan(train_loss) or np.isnan(val_loss):
            msg = "Loss is NaN. Aborting training."
            raise ValueError(msg)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        loss_weights = {loss_obj.name: loss_obj.weight for loss_obj in losses}
        postfix: dict[str, Any] = {
            "train_loss": f"{train_loss:.4f}",
            "val_loss": f"{val_loss:.4f}",
        }
        for key, val in comps_sum.items():
            if key == "L":
                postfix["L"] = int(val / batches)
            else:
                weight = loss_weights.get(key, 1.0)
                postfix[key] = f"{(val / batches) * weight:.4f}"
        pbar.set_postfix(**postfix)

        if epochs_without_improvement >= cfg.patience:
            break

    model.load_state_dict(best_state)
    return train_losses, val_losses, dict(train_components), dict(val_components)


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
    )
    if mdl.depth == 0:
        _warm_start_linear(model, data.X_train, data.Y_train, data.n_channels)

    model = model.to(device)

    target_shape = (-1, horizon, data.n_channels)
    tensors = _Tensors(
        X_train=_tensor(data.X_train, device),
        Y_train=_tensor(data.Y_train, device).reshape(target_shape),
        X_val=_tensor(data.X_val, device),
        Y_val=_tensor(data.Y_val, device).reshape(target_shape),
        y_center=_tensor(data.y_std.center, device),
        y_scale=_tensor(data.y_std.scale, device),
    )

    train_losses, val_losses, train_comps, val_comps = _fit(model, tensors, trn, losses, fs=fs, seed=seed)

    art = dataclasses.replace(
        model.to_artifact(sim.dt * sim.downsample, sim.downsample, data.y_std, data.u_std),
        provenance=training_provenance(data_files, sim.cutoff_hz),
    )
    eval_steps = max(1, round(trn.eval_horizon_s * fs))
    # The energy course follows the metrics layer's own eeg_ms convention rather than a knob of its
    # own, clamped where the evaluation horizon is too short to hold one window.
    energy_window = min(max(1, round(METRICS["eeg_ms"].window_s * fs)), eval_steps)
    energy_hop = max(1, round(DEFAULT_HOP_S * fs))
    return TrainingResult(
        artifact=art,
        train_losses=train_losses,
        val_losses=val_losses,
        train_components=train_comps,
        val_components=val_comps,
        rollout=evaluate_rollouts(art, data.val_trajs, eval_steps),
        log_energy=evaluate_log_energy(
            art, data.val_trajs, eval_steps, window_steps=energy_window, hop_steps=energy_hop
        ),
        val_trajs=data.val_trajs,
        du_sensitivity=_du_sensitivity(model, tensors.X_val),
    )
