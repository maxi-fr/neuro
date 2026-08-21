from __future__ import annotations

import copy
import dataclasses
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from tqdm import tqdm

from neuro.observable import log_observable
from neuro.predictor.data import prepare_datasets
from neuro.predictor.observable_module import ObservableMLP
from neuro.predictor.train import float32_tensor, lr_schedule, shuffled_batches
from neuro.provenance import training_provenance
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor, nn

    from neuro.config import NNPredictorConfig, ObservableGeometry
    from neuro.observable import ObservableArtifact
    from neuro.types import FloatArray

_DU_WINDOWS = 8


@dataclass(frozen=True)
class ObservableTrainingResult:
    """Everything one observable-space training run produced. Nothing is written; the caller persists.

    Attributes
    ----------
    artifact : ObservableArtifact
        The best-validation-loss weights frozen together with the fitted transforms and geometry.
    train_losses, val_losses : list[float]
        Per-epoch MSE in standardized log-Observable space, one entry per epoch actually run.
    val_log_mse : float
        Held-out MSE in **raw** log units -- the scale the gates of decision 11 compare on, and the
        one an incumbent artifact pushed through the same geometry can be scored against.
    n_independent_samples : int
        Non-overlapping targets in the training split. Each target spans the whole Control Horizon,
        so overlapping windows are not independent evidence and the honest count is this one.
    du_sensitivity : float
        Mean Frobenius norm of ``d(l_hat) / d(u_future)``. A value near zero means the model
        forecasts log power while ignoring stimulation, which hands the solver nothing to descend.
    val_trajs : list[tuple[FloatArray, FloatArray]]
        The held-out ``(u, y)`` trajectories, kept whole so the caller can score or plot them.
    """

    artifact: ObservableArtifact
    train_losses: list[float]
    val_losses: list[float]
    val_log_mse: float
    n_independent_samples: int
    du_sensitivity: float
    val_trajs: list[tuple[FloatArray, FloatArray]]

    def save(self, artifact_dir: Path) -> None:
        """Write ``model.npz`` and ``training_stats.json`` into ``artifact_dir``."""
        self.artifact.save(artifact_dir / "model")
        stats = {
            "train_loss": self.train_losses,
            "val_loss": self.val_losses,
            "val_log_mse": self.val_log_mse,
            "n_independent_samples": self.n_independent_samples,
            "du_sensitivity": self.du_sensitivity,
        }
        (artifact_dir / "training_stats.json").write_text(json.dumps(stats, indent=2))


@dataclass(frozen=True)
class ObservableData:
    """History-plus-control inputs and their Frame targets, shared by every arm scored on this data.

    Attributes
    ----------
    x_train, x_val : FloatArray
        The incumbent history-plus-future-control rows, standardized.
    targets_train, targets_val : FloatArray
        Raw log-Observable ``(samples, n_frames, n_channels * n_values)``.
    l_std : Standardizer
        Fitted on the training targets, so every arm is optimised on one scale.
    y_std, u_std : Standardizer
        The channel and control standardizers the artifact carries.
    val_trajs : list[tuple[FloatArray, FloatArray]]
        The held-out ``(u, y)`` trajectories, kept whole.
    n_channels, n_controls : int
        Physical channel and electrode counts.
    """

    x_train: FloatArray
    targets_train: FloatArray
    x_val: FloatArray
    targets_val: FloatArray
    l_std: Standardizer
    y_std: Standardizer
    u_std: Standardizer
    val_trajs: list[tuple[FloatArray, FloatArray]]
    n_channels: int
    n_controls: int


def build_targets(  # noqa: PLR0913
    y_standardized: FloatArray,
    y_std: Standardizer,
    geometry: ObservableGeometry,
    *,
    horizon: int,
    n_channels: int,
    fs: float,
) -> FloatArray:
    """Reduce standardized future-EEG windows ``(samples, horizon * C)`` onto the Frame grid.

    Returns ``(samples, n_frames, n_channels * n_values)`` of raw log-Observable, built by the same
    geometry object the training Loss and the MPC Cost use.
    """
    raw = y_std.inverse_transform(y_standardized.reshape(-1, horizon, n_channels))
    frames = log_observable(raw, geometry, fs)
    return frames.reshape(frames.shape[0], frames.shape[1], -1)


def prepare_observable_data(cfg: NNPredictorConfig, data_files: list[str]) -> ObservableData:
    """Build the sliding windows and their Frame targets for the config's Observable geometry."""
    sim, mdl, trn, obs = cfg.simulation, cfg.model, cfg.training, cfg.observable
    if obs is None:
        msg = "an 'observable' block is required to build frame-grid targets."
        raise ValueError(msg)
    horizon, geometry = obs.horizon, obs.geometry()

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

    x_train, x_val = data.X_train, data.X_val
    if obs.control_blind:
        # The Control-blindness arm: identical architecture, schedule and seed, future controls zeroed.
        n_hist = mdl.n_y * data.n_channels + mdl.n_u * data.n_controls
        x_train, x_val = x_train.copy(), x_val.copy()
        x_train[:, n_hist:] = 0.0
        x_val[:, n_hist:] = 0.0

    targets = {
        split: build_targets(y, data.y_std, geometry, horizon=horizon, n_channels=data.n_channels, fs=cfg.fs)
        for split, y in (("train", data.Y_train), ("val", data.Y_val))
    }
    return ObservableData(
        x_train=x_train,
        targets_train=targets["train"],
        x_val=x_val,
        targets_val=targets["val"],
        l_std=Standardizer.fit(
            targets["train"].reshape(-1, targets["train"].shape[-1]),
            kind=trn.scaler,
            global_scaling=trn.global_scaling,
        ),
        y_std=data.y_std,
        u_std=data.u_std,
        val_trajs=data.val_trajs,
        n_channels=data.n_channels,
        n_controls=data.n_controls,
    )


def du_sensitivity(model: nn.Module, x_val: Tensor, n_hist: int) -> float:
    """Mean Frobenius norm of ``d(forecast) / d(future controls)`` over a fixed subsample of windows.

    Forward mode is the cheap direction: the future-control block is far narrower than the forecast
    it drives, and one Jacobian per validation window would dominate the training run.
    """
    rows = x_val[:: max(x_val.shape[0] // _DU_WINDOWS, 1)][:_DU_WINDOWS]

    norms = []
    for row in rows:

        def forecast(u_future: Tensor, history: Tensor = row[:n_hist]) -> Tensor:
            return model(torch.cat([history, u_future])[None, :])[0]

        jacobian = torch.autograd.functional.jacobian(forecast, row[n_hist:], strategy="forward-mode", vectorize=True)
        norms.append(float(torch.linalg.norm(cast("Tensor", jacobian)).detach()))
    return float(np.mean(norms))


def fit(
    model: nn.Module,
    tensors: tuple[Tensor, Tensor, Tensor, Tensor],
    cfg: NNPredictorConfig,
    *,
    seed: int,
) -> tuple[list[float], list[float]]:
    """Regress ``model`` onto the Frame targets, leaving it holding the best-validation weights.

    The Loss is a plain MSE in standardized log-Observable space over ``(frame, channel, value)``.
    The hinge stays in the controller: training one would discard every gradient from Frames already
    under the envelope. Any module mapping the input row to that target shape fits here, which is
    what lets the direct-map arm of the gate probe share this schedule exactly.
    """
    x_train, y_train, x_val, y_val = tensors
    trn = cfg.training
    rng = np.random.default_rng(seed)
    n_samples = x_train.shape[0]

    steps_per_epoch = (n_samples + trn.batch_size - 1) // trn.batch_size
    total_steps = max(steps_per_epoch * trn.epochs, 1)
    warmup_steps = min(steps_per_epoch * trn.warmup_epochs, total_steps - 1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=trn.learning_rate, weight_decay=trn.weight_decay)
    scheduler = lr_schedule(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    train_losses: list[float] = []
    val_losses: list[float] = []

    pbar = tqdm(range(trn.epochs), desc="Training observable MLP")
    for _ in pbar:
        epoch_loss, batches = 0.0, 0
        for idx in shuffled_batches(n_samples, trn.batch_size, rng):
            loss = torch.mean((model(x_train[idx]) - y_train[idx]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += float(loss.detach())
            batches += 1

        train_loss = epoch_loss / batches
        with torch.no_grad():
            val_loss = float(torch.mean((model(x_val) - y_val) ** 2).detach())

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

        pbar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")
        if epochs_without_improvement >= trn.patience:
            break

    model.load_state_dict(best_state)
    return train_losses, val_losses


def log_mse(model: nn.Module, x_val: Tensor, targets_val: FloatArray, l_std: Standardizer) -> float:
    """Held-out MSE in raw log-Observable units, the scale every gate arm is compared on."""
    with torch.no_grad():
        standardized = model(x_val).detach().cpu().numpy().astype(np.float64)
    return float(np.mean((l_std.inverse_transform(standardized) - targets_val) ** 2))


def train_observable(
    cfg: NNPredictorConfig, data_files: list[str], *, seed_offset: int = 0
) -> ObservableTrainingResult:
    """Train the observable-space predictor for one config and return everything the run produced.

    Parameters
    ----------
    cfg : NNPredictorConfig
        Validated configuration carrying an ``observable`` block.
    data_files : list[str]
        Paths to the ``.npz`` trajectory files, split into train/validation by trajectory.
    seed_offset : int, optional
        Added to ``training.seed``. Defaults to 0.

    Returns
    -------
    ObservableTrainingResult
        The best-validation artifact, the loss curves, the held-out log-space error, the honest
        sample count, the control sensitivity and the held-out trajectories.
    """
    sim, mdl, trn, obs = cfg.simulation, cfg.model, cfg.training, cfg.observable
    if obs is None:
        msg = "train_observable requires an 'observable' block; use neuro.predictor.train.train otherwise."
        raise ValueError(msg)

    seed = trn.seed + seed_offset
    torch.manual_seed(seed)
    device = torch.device(trn.device)
    data = prepare_observable_data(cfg, data_files)

    model = ObservableMLP(
        n_y=mdl.n_y,
        n_u=mdl.n_u,
        horizon=obs.horizon,
        n_channels=data.n_channels,
        n_controls=data.n_controls,
        geometry=obs.geometry(),
        fs=cfg.fs,
        z_dim=obs.z_dim,
        lift_hidden=obs.lift_hidden,
        lift_depth=obs.lift_depth,
        transition_hidden=obs.transition_hidden,
        transition_depth=obs.transition_depth,
        activation=obs.activation,
    ).to(device)

    tensors = (
        float32_tensor(data.x_train, device),
        float32_tensor(data.l_std.transform(data.targets_train), device),
        float32_tensor(data.x_val, device),
        float32_tensor(data.l_std.transform(data.targets_val), device),
    )
    train_losses, val_losses = fit(model, tensors, cfg, seed=seed)

    art = dataclasses.replace(
        model.to_artifact(sim.dt * sim.downsample, sim.downsample, data.y_std, data.u_std, data.l_std),
        provenance=training_provenance(data_files, sim.cutoff_hz),
    )
    n_hist = mdl.n_y * data.n_channels + mdl.n_u * data.n_controls
    return ObservableTrainingResult(
        artifact=art,
        train_losses=train_losses,
        val_losses=val_losses,
        val_log_mse=log_mse(model, tensors[2], data.targets_val, data.l_std),
        n_independent_samples=data.x_train.shape[0] // obs.horizon,
        du_sensitivity=du_sensitivity(model, tensors[0], n_hist),
        val_trajs=data.val_trajs,
    )
