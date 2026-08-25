from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from neuro.predictor.data import frame_targets, prepare_datasets
from neuro.predictor.gradient import fit_gradient_descent, float32_tensor
from neuro.predictor.observable_module import StepwiseObservableMLP
from neuro.predictor.ridge import RidgeTrainer
from neuro.provenance import training_provenance
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor, nn

    from neuro.config import NNPredictorConfig
    from neuro.types import FloatArray

_DU_WINDOWS = 8


@dataclass(frozen=True)
class ObservableTrainingResult:
    """Everything one observable-space training run produced; ``save`` persists it all.

    The gradient-descent arm fills the loss curves and takes ``candidates["val_loss"]`` as the
    curve minimum; the closed-form ridge arm (:func:`_train_observable_ridge`) deliberately
    carries empty ``train_losses``/``val_losses`` (no epoch loop exists) and takes
    ``candidates["val_loss"]`` as the fitted model's held-out standardized MSE instead, so the
    two arms agree on what ``val_loss`` means and on the candidate set.

    Attributes
    ----------
    predictor : StepwiseObservableMLP
        The trained one-Frame-per-step module holding the best-validation-loss weights, with the
        standardizers as buffers and the recorded metadata (geometry, provenance, downsample)
        attached.
    candidates : dict[str, float]
        Every objective the sweep seam can rank this run on: ``val_loss`` and ``val_log_mse``,
        both lower-is-better.
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

    predictor: StepwiseObservableMLP
    candidates: dict[str, float]
    train_losses: list[float]
    val_losses: list[float]
    val_log_mse: float
    n_independent_samples: int
    du_sensitivity: float
    val_trajs: list[tuple[FloatArray, FloatArray]]

    def save(self, artifact_dir: Path) -> None:
        """Write the numpy-checkpoint and ``training_stats.json`` into ``artifact_dir``."""
        self.predictor.save(artifact_dir / "model")
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
    train_trajs, val_trajs : list[tuple[FloatArray, FloatArray]]
        The training-side and held-out ``(u, y)`` trajectories, kept whole so the ridge arm can
        fit the shared readout raw-direct and free-run rollouts can be scored on them.
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
    train_trajs: list[tuple[FloatArray, FloatArray]]
    val_trajs: list[tuple[FloatArray, FloatArray]]
    n_channels: int
    n_controls: int


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
        split: frame_targets(y, geometry, horizon=horizon, n_channels=data.n_channels, fs=cfg.fs)
        for split, y in (("train", data.Y_raw_train), ("val", data.Y_raw_val))
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
        train_trajs=data.train_trajs,
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


def log_mse(model: nn.Module, x_val: Tensor, targets_val: FloatArray, l_std: Standardizer) -> float:
    """Held-out MSE in raw log-Observable units, the scale every gate arm is compared on."""
    with torch.no_grad():
        standardized = model(x_val).detach().cpu().numpy().astype(np.float64)
    return float(np.mean((l_std.inverse_transform(standardized) - targets_val) ** 2))


def _train_observable(
    cfg: NNPredictorConfig, data_files: list[str], *, seed_offset: int = 0
) -> ObservableTrainingResult:
    """Train the observable-space predictor for one config and return everything the run produced.

    The Loss is a plain MSE in standardized log-Observable space over ``(frame, channel, value)``.
    The hinge stays in the controller: training one would discard every gradient from Frames already
    under the envelope. Any module mapping the input row to that target shape fits the shared
    gradient-descent loop, which is what lets the direct-map arm of the gate probe share this
    schedule exactly.
    """
    sim, mdl, trn, obs = cfg.simulation, cfg.model, cfg.training, cfg.observable
    if obs is None:
        msg = "the observable arm requires an 'observable' block; other configs go through the waveform arm."
        raise ValueError(msg)

    seed = trn.seed + seed_offset
    torch.manual_seed(seed)
    device = torch.device(trn.device)
    data = prepare_observable_data(cfg, data_files)

    model = StepwiseObservableMLP(
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

    def mse_loss(
        model: nn.Module,
        x: Tensor,
        y: Tensor,
        epoch: int | None,  # noqa: ARG001 -- the schedule clock is epoch-independent for plain MSE
    ) -> tuple[Tensor, dict[str, float]]:
        """Standardized log-Observable MSE; the schedule clock is epoch-independent here."""
        return torch.mean((model(x) - y) ** 2), {}

    train_losses, val_losses, _, _ = fit_gradient_descent(
        model,
        *tensors,
        trn,
        seed=seed,
        loss_fn=mse_loss,
        desc="Training observable MLP",
    )

    n_hist = mdl.n_y * data.n_channels + mdl.n_u * data.n_controls
    val_log_mse = log_mse(model, tensors[2], data.targets_val, data.l_std)
    sensitivity = du_sensitivity(model, tensors[0], n_hist)
    model = model.cpu()
    model.provenance = training_provenance(data_files, sim.cutoff_hz)
    model.downsample = sim.downsample
    return ObservableTrainingResult(
        predictor=model,
        candidates={
            "val_loss": float(min(val_losses)),
            "val_log_mse": val_log_mse,
        },
        train_losses=train_losses,
        val_losses=val_losses,
        val_log_mse=val_log_mse,
        n_independent_samples=data.x_train.shape[0] // obs.horizon,
        du_sensitivity=sensitivity,
        val_trajs=data.val_trajs,
    )


def _train_observable_ridge(
    cfg: NNPredictorConfig, data_files: list[str], *, seed_offset: int = 0
) -> ObservableTrainingResult:
    """Fit the shared readout of a depth-0 observable MLP by closed-form ridge.

    The lift and transition stay at their (random) initialisation; the Ridge Trainer fits the
    shared readout on the harvested per-Frame lifted states ``z_m``, exactly the depth-0 arm
    ticket 06 exercised at the capability level. There is no epoch loop, so the result carries
    empty loss curves and ``candidates["val_loss"]`` is the fitted model's held-out MSE in
    standardized log-Observable space -- the same quantity the gradient-descent arm minimizes --
    rather than a curve minimum.
    """
    sim, mdl, trn, obs = cfg.simulation, cfg.model, cfg.training, cfg.observable
    if obs is None:
        msg = "the observable ridge arm requires an 'observable' block."
        raise ValueError(msg)

    torch.manual_seed(trn.seed + seed_offset)
    device = torch.device(trn.device)
    data = prepare_observable_data(cfg, data_files)

    model = StepwiseObservableMLP(
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
        y_std=data.y_std,
        u_std=data.u_std,
        l_std=data.l_std,
    ).to(device)
    RidgeTrainer(ridge_lambda=trn.ridge_lambda).fit(model, data.train_trajs)

    x_val = float32_tensor(data.x_val, device)
    with torch.no_grad():
        standardized = model(x_val).detach().cpu().numpy().astype(np.float64)
    val_loss = float(np.mean((standardized - data.l_std.transform(data.targets_val)) ** 2))
    val_log_mse = log_mse(model, x_val, data.targets_val, data.l_std)
    n_hist = mdl.n_y * data.n_channels + mdl.n_u * data.n_controls
    sensitivity = du_sensitivity(model, float32_tensor(data.x_train, device), n_hist)

    model = model.cpu()
    model.provenance = training_provenance(data_files, sim.cutoff_hz)
    model.downsample = sim.downsample
    return ObservableTrainingResult(
        predictor=model,
        candidates={"val_loss": val_loss, "val_log_mse": val_log_mse},
        train_losses=[],
        val_losses=[],
        val_log_mse=val_log_mse,
        n_independent_samples=data.x_train.shape[0] // obs.horizon,
        du_sensitivity=sensitivity,
        val_trajs=data.val_trajs,
    )
