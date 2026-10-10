from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from neuro.predictor.cnn import ObservableCNNModel, WaveformCNNModel
from neuro.predictor.data import (
    Datasets,
    TrajectoryWindowDataset,
    batch_iterator,
    prepare_datasets,
    prepare_observable_datasets,
)
from neuro.predictor.dmd import DmdTrainer
from neuro.predictor.evaluation import (
    LogEnergyError,
    ObservableFrameMSE,
    RolloutNMSE,
    evaluate_free_run,
    evaluate_observable_free_run,
    free_run_stats,
)
from neuro.predictor.gradient import fit_gradient_descent
from neuro.predictor.inference import inference_from_checkpoint
from neuro.predictor.losses import LossContext, build_losses, eligibility_start_epoch
from neuro.predictor.mlp import ObservableMLPModel, WaveformMLPModel
from neuro.predictor.ridge import RidgeTrainer, RidgeTrainingResult
from neuro.provenance import training_provenance

if TYPE_CHECKING:
    from pathlib import Path

    from neuro.config import NNPredictorConfig, StftGeometry
    from neuro.predictor.base import AutoregressiveModel
    from neuro.predictor.losses import Loss
    from neuro.types import FloatArray

_DU_WINDOWS = 8
_DU_PROBES = 5


@dataclass(frozen=True)
class TrainingResult:
    """Everything one gradient-descent training run produced; ``save`` persists it all.

    Attributes
    ----------
    predictor : AutoregressiveModel
        The trained module holding the best-validation-loss weights, with the standardizers as
        buffers and the recorded metadata (provenance, downsample) attached.
    candidates : dict[str, float]
        Every objective the sweep seam can rank this run on: ``log_energy``, ``val_loss`` and
        ``rollout_nmse``, all lower-is-better.
    train_losses, val_losses : list[float]
        Per-epoch loss, one entry per epoch actually run (early stopping shortens both).
    train_components, val_components : dict[str, list[float | None]]
        Per-epoch unweighted loss components and diagnostics, with None for terms not run.
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
        Mean Frobenius norm of the Rollout's Jacobian with respect to future Control Currents. A
        value near zero means the model predicts EEG while ignoring stimulation.
    eligibility_start : int
        First training epoch where every enabled scheduled Loss is active and curriculum at full Span.
    selected_epoch : int
        Epoch of the checkpoint loaded into ``predictor``, selected from the eligible phase.
    stopping_reason : str
        Reason training ended: 'early_stopping' or 'max_epochs'.
    """

    predictor: AutoregressiveModel
    candidates: dict[str, float]
    train_losses: list[float]
    val_losses: list[float]
    train_components: dict[str, list[float | None]]
    val_components: dict[str, list[float | None]]
    free_run: RolloutNMSE | ObservableFrameMSE
    log_energy: LogEnergyError | None
    val_trajs: list[tuple[FloatArray, FloatArray]]
    du_sensitivity: float
    eligibility_start: int
    selected_epoch: int
    stopping_reason: str

    def save(self, artifact_dir: Path) -> None:
        """Save the selected checkpoint, Loss curves, eligibility epoch, and stopping reason."""
        self.predictor.save(artifact_dir / "model")
        stats = {
            "train_loss": self.train_losses,
            "val_loss": self.val_losses,
            "train_components": self.train_components,
            "val_components": self.val_components,
            "du_sensitivity": self.du_sensitivity,
            "eligibility_start": self.eligibility_start,
            "selected_epoch": self.selected_epoch,
            "stopping_reason": self.stopping_reason,
            **free_run_stats(self.free_run, self.log_energy),
        }
        (artifact_dir / "training_stats.json").write_text(json.dumps(stats, indent=2))


def _du_sensitivity(
    model: AutoregressiveModel,
    val_dataset: TrajectoryWindowDataset,
    *,
    n_probes: int = _DU_PROBES,
    key: jax.Array | None = None,
) -> float:
    """Mean Frobenius norm of d(Rollout)/d(future Control Currents) estimated via reverse-mode VJPs.

    Uses the Hutchinson trace estimator: for random Gaussian projections ``v ~ N(0, I)``,
    ``E[||J^T v||^2] = ||J||_F^2``. Reverse-mode vector-Jacobian products on batched representative
    validation windows estimate the full-horizon sensitivity in milliseconds without materializing
    the multi-gigabyte Jacobian tensor.
    """
    n_total = len(val_dataset)
    if n_total == 0:
        return 0.0

    indices = np.arange(n_total) if n_total <= _DU_WINDOWS else np.linspace(0, n_total - 1, _DU_WINDOWS, dtype=int)
    y_hist, u_hist, u_future, _ = val_dataset.get_batch(indices)
    batch_size = y_hist.shape[0]

    out, vjp_fn = jax.vjp(lambda u: model.rollout(y_hist, u_hist, u), u_future)
    if key is None:
        key = jax.random.PRNGKey(0)

    probe_sq_list: list[jax.Array] = []
    for _ in range(n_probes):
        key, subkey = jax.random.split(key)
        v = jax.random.normal(subkey, out.shape, dtype=out.dtype)
        (grad,) = vjp_fn(v)
        probe_sq = jnp.sum(grad.reshape(batch_size, -1) ** 2, axis=1)
        probe_sq_list.append(probe_sq)

    mean_probe_sq = jnp.mean(jnp.stack(probe_sq_list, axis=0), axis=0)
    frob_norms = jnp.sqrt(mean_probe_sq)
    return float(jnp.mean(frob_norms))


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
    if cfg.model.architecture == "cnn" and cfg.training.fit != "gradient_descent":
        msg = "CNN predictors support only gradient_descent training."
        raise ValueError(msg)
    if cfg.observable is not None:
        if cfg.training.fit == "ridge":
            return _train_observable_ridge(cfg, data_files, cfg.observable)
        if cfg.training.fit == "dmd":
            return _train_observable_dmd(cfg, data_files, cfg.observable)
        return _train_observable(cfg, data_files, cfg.observable, seed_offset=seed_offset)
    if cfg.training.fit == "ridge":
        return _train_ridge(cfg, data_files)
    if cfg.training.fit == "dmd":
        return _train_waveform_dmd(cfg, data_files)
    return _train_waveform(cfg, data_files, seed_offset=seed_offset)


def _prepare_observable(
    cfg: NNPredictorConfig, data_files: list[str], geom: StftGeometry, *, depth: int, seed: int = 0
) -> tuple[Datasets, AutoregressiveModel, list[Loss]]:
    """Build prepared Observable datasets and the configured autoregressive Predictor."""
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
    key = jax.random.PRNGKey(seed)
    if mdl.architecture == "cnn":
        model: AutoregressiveModel = ObservableCNNModel(
            geometry=geom,
            n_y=mdl.n_y,
            n_u=mdl.n_u,
            horizon=horizon,
            n_channels=data.n_channels,
            n_controls=data.n_controls,
            n_outputs=n_outputs,
            hidden_size=mdl.hidden_size,
            depth=depth,
            kernel_size=mdl.kernel_size,
            frequency_kernel_size=mdl.frequency_kernel_size,
            activation=mdl.activation,
            residual=mdl.residual,
            dt=sim.dt * sim.downsample * geom.n_hop,
            downsample=sim.downsample,
            y_std=data.y_std,
            u_std=data.u_std,
            key=key,
        )
    else:
        model = ObservableMLPModel(
            geometry=geom,
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
            downsample=sim.downsample,
            y_std=data.y_std,
            u_std=data.u_std,
            key=key,
        )
    return data, model, losses


def _evaluate_val_mse(
    model: AutoregressiveModel,
    val_dataset: TrajectoryWindowDataset,
    batch_size: int = 256,
) -> float:
    """Score ``model`` over mini-batches of ``val_dataset`` with sample-weighted mean squared error."""
    if len(val_dataset) == 0:
        return 0.0
    val_loss_sum = 0.0
    total_samples = 0
    for y_hist, u_hist, u_future, y_target in batch_iterator(val_dataset, batch_size=batch_size, shuffle=False):
        b_size = y_hist.shape[0]
        total_samples += b_size
        pred = model.rollout(y_hist, u_hist, u_future)
        loss = jnp.mean((pred - y_target) ** 2)
        val_loss_sum += float(loss) * b_size
    if total_samples == 0:
        return 0.0
    return val_loss_sum / total_samples


def _train_observable_ridge(cfg: NNPredictorConfig, data_files: list[str], geom: StftGeometry) -> RidgeTrainingResult:
    """Fit the single layer of a depth-0 Observable MLP by closed-form ridge."""
    if cfg.model.depth > 0:
        msg = f"'training.fit: ridge' requires a depth-0 MLP, got model.depth = {cfg.model.depth}."
        raise ValueError(msg)
    sim, trn = cfg.simulation, cfg.training
    fs_frame = geom.frame_rate(cfg.fs)
    data, model, _ = _prepare_observable(cfg, data_files, geom, depth=0)
    if not isinstance(model, ObservableMLPModel):
        msg = "Observable ridge fitting requires an MLP model"
        raise TypeError(msg)
    RidgeTrainer(ridge_lambda=trn.ridge_lambda).fit(model, data.train_trajs)

    model.provenance = training_provenance(data_files, sim.cutoff_hz)
    model.downsample = sim.downsample
    eval_steps = max(1, round(trn.eval_horizon_s * fs_frame))
    inference = inference_from_checkpoint(*model.to_checkpoint())
    frame_mse = evaluate_observable_free_run(inference, data.val_trajs, eval_steps)

    val_frame_loss = _evaluate_val_mse(model, data.val_dataset, batch_size=trn.batch_size)

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


def _train_observable_dmd(cfg: NNPredictorConfig, data_files: list[str], geom: StftGeometry) -> RidgeTrainingResult:
    """Fit the single layer of a depth-0 Observable MLP by closed-form Hankel-DMDc."""
    if cfg.model.depth > 0:
        msg = f"'training.fit: dmd' requires a depth-0 MLP, got model.depth = {cfg.model.depth}."
        raise ValueError(msg)
    sim, trn = cfg.simulation, cfg.training
    fs_frame = geom.frame_rate(cfg.fs)
    data, model, _ = _prepare_observable(cfg, data_files, geom, depth=0)
    if not isinstance(model, ObservableMLPModel):
        msg = "Observable DMD fitting requires an MLP model"
        raise TypeError(msg)
    DmdTrainer(rank=trn.dmd_rank, energy=trn.dmd_energy, dmd_lambda=trn.dmd_lambda).fit(model, data.train_trajs)

    model.provenance = training_provenance(data_files, sim.cutoff_hz)
    model.downsample = sim.downsample
    eval_steps = max(1, round(trn.eval_horizon_s * fs_frame))
    inference = inference_from_checkpoint(*model.to_checkpoint())
    frame_mse = evaluate_observable_free_run(inference, data.val_trajs, eval_steps)

    val_frame_loss = _evaluate_val_mse(model, data.val_dataset, batch_size=trn.batch_size)

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
    """Train an Observable Predictor and restore its best checkpoint from the eligible phase."""
    sim, trn = cfg.simulation, cfg.training
    seed = trn.seed + seed_offset
    fs_frame = geom.frame_rate(cfg.fs)

    data, model, losses = _prepare_observable(cfg, data_files, geom, depth=cfg.model.depth, seed=seed)

    ctx = LossContext(
        y_center=jnp.asarray(data.y_std.center, dtype=jnp.float32),
        y_scale=jnp.asarray(data.y_std.scale, dtype=jnp.float32),
        fs=fs_frame,
    )

    eligibility_start = eligibility_start_epoch(losses)
    fit = fit_gradient_descent(
        model,
        data.train_dataset,
        data.val_dataset,
        trn,
        seed=seed,
        losses=losses,
        ctx=ctx,
        desc="Training Observable Predictor",
        eligibility_start=eligibility_start,
    )

    eval_steps = max(1, round(trn.eval_horizon_s * fs_frame))
    model = fit.best_model
    du_sensitivity = _du_sensitivity(model, data.val_dataset)
    model.provenance = training_provenance(data_files, sim.cutoff_hz)
    model.downsample = sim.downsample
    inference = inference_from_checkpoint(*model.to_checkpoint())
    frame_mse = evaluate_observable_free_run(inference, data.val_trajs, eval_steps)
    return TrainingResult(
        predictor=model,
        candidates={
            "val_loss": float(fit.val_losses[fit.selected_epoch]),
            "val_log_mse": frame_mse.pooled,
        },
        train_losses=fit.train_losses,
        val_losses=fit.val_losses,
        train_components=fit.train_components,
        val_components=fit.val_components,
        free_run=frame_mse,
        log_energy=None,
        val_trajs=data.val_trajs,
        du_sensitivity=du_sensitivity,
        eligibility_start=eligibility_start,
        selected_epoch=fit.selected_epoch,
        stopping_reason=fit.stopping_reason,
    )


def _train_ridge(cfg: NNPredictorConfig, data_files: list[str]) -> RidgeTrainingResult:
    """Route an NN config naming ``training.fit: ridge`` to the Ridge Trainer.

    The Ridge Trainer serves the depth-0 waveform MLP. A config whose model carries hidden
    layers is not Ridge-Fittable and fails here at build time, before data is loaded or any
    fit runs.
    """
    if cfg.model.architecture != "mlp" or cfg.model.depth > 0:
        msg = f"'training.fit: ridge' requires a depth-0 MLP, got model.depth = {cfg.model.depth}."
        raise ValueError(msg)
    return _train_waveform_ridge(cfg, data_files)


def _prepare_waveform(
    cfg: NNPredictorConfig, data_files: list[str], *, depth: int, seed: int = 0
) -> tuple[Datasets, AutoregressiveModel, list[Loss]]:
    """Build prepared datasets and the configured autoregressive waveform Predictor.

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
    key = jax.random.PRNGKey(seed)
    if mdl.architecture == "cnn":
        model: AutoregressiveModel = WaveformCNNModel(
            n_y=mdl.n_y,
            n_u=mdl.n_u,
            horizon=horizon,
            n_channels=data.n_channels,
            n_controls=data.n_controls,
            n_outputs=data.n_channels,
            hidden_size=mdl.hidden_size,
            depth=depth,
            kernel_size=mdl.kernel_size,
            activation=mdl.activation,
            residual=mdl.residual,
            dt=sim.dt * sim.downsample,
            downsample=sim.downsample,
            y_std=data.y_std,
            u_std=data.u_std,
            key=key,
        )
    else:
        model = WaveformMLPModel(
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
            downsample=sim.downsample,
            y_std=data.y_std,
            u_std=data.u_std,
            key=key,
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
    if not isinstance(model, WaveformMLPModel):
        msg = "waveform ridge fitting requires an MLP model"
        raise TypeError(msg)
    RidgeTrainer(ridge_lambda=trn.ridge_lambda).fit(model, data.train_trajs)

    model.provenance = training_provenance(data_files, sim.cutoff_hz)
    model.downsample = sim.downsample
    eval_steps = max(1, round(trn.eval_horizon_s * fs))
    inference = inference_from_checkpoint(*model.to_checkpoint())
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


def _train_waveform_dmd(cfg: NNPredictorConfig, data_files: list[str]) -> RidgeTrainingResult:
    """Fit the single layer of a depth-0 waveform MLP by closed-form Hankel-DMDc."""
    if cfg.model.architecture != "mlp":
        msg = "'training.fit: dmd' supports waveform MLP predictors only."
        raise ValueError(msg)
    if cfg.model.depth > 0:
        msg = f"'training.fit: dmd' requires a depth-0 MLP, got model.depth = {cfg.model.depth}."
        raise ValueError(msg)
    sim, trn = cfg.simulation, cfg.training
    fs = cfg.fs
    data, model, _ = _prepare_waveform(cfg, data_files, depth=0)
    if not isinstance(model, WaveformMLPModel):
        msg = "waveform DMD fitting requires an MLP model"
        raise TypeError(msg)
    DmdTrainer(rank=trn.dmd_rank, energy=trn.dmd_energy, dmd_lambda=trn.dmd_lambda).fit(model, data.train_trajs)

    model.provenance = training_provenance(data_files, sim.cutoff_hz)
    model.downsample = sim.downsample
    eval_steps = max(1, round(trn.eval_horizon_s * fs))
    inference = inference_from_checkpoint(*model.to_checkpoint())
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
    """Train a waveform Predictor and restore its best checkpoint from the eligible phase."""
    sim, trn = cfg.simulation, cfg.training
    seed = trn.seed + seed_offset
    fs = cfg.fs

    data, model, losses = _prepare_waveform(cfg, data_files, depth=cfg.model.depth, seed=seed)

    ctx = LossContext(
        y_center=jnp.asarray(data.y_std.center, dtype=jnp.float32),
        y_scale=jnp.asarray(data.y_std.scale, dtype=jnp.float32),
        fs=fs,
    )

    eligibility_start = eligibility_start_epoch(losses)
    fit = fit_gradient_descent(
        model,
        data.train_dataset,
        data.val_dataset,
        trn,
        seed=seed,
        losses=losses,
        ctx=ctx,
        desc="Training MLP" if cfg.model.architecture == "mlp" else "Training CNN",
        eligibility_start=eligibility_start,
    )

    eval_steps = max(1, round(trn.eval_horizon_s * fs))
    model = fit.best_model
    du_sensitivity = _du_sensitivity(model, data.val_dataset)
    model.provenance = training_provenance(data_files, sim.cutoff_hz)
    model.downsample = sim.downsample
    inference = inference_from_checkpoint(*model.to_checkpoint())
    rollout, log_energy = evaluate_free_run(inference, data.val_trajs, eval_steps, fs)
    return TrainingResult(
        predictor=model,
        candidates={
            "log_energy": log_energy.pooled,
            "val_loss": float(fit.val_losses[fit.selected_epoch]),
            "rollout_nmse": rollout.pooled,
        },
        train_losses=fit.train_losses,
        val_losses=fit.val_losses,
        train_components=fit.train_components,
        val_components=fit.val_components,
        free_run=rollout,
        log_energy=log_energy,
        val_trajs=data.val_trajs,
        du_sensitivity=du_sensitivity,
        eligibility_start=eligibility_start,
        selected_epoch=fit.selected_epoch,
        stopping_reason=fit.stopping_reason,
    )
