from __future__ import annotations

import collections
import copy
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from neuro.predictor.base import AutoregressiveModel
from neuro.predictor.data import TrajectoryWindowDataset, batch_iterator
from neuro.predictor.losses import CurriculumMSE, LossContext, total_loss

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from neuro.config import TrainingConfig
    from neuro.predictor.losses import Loss

_TIME_PROBE_EPOCHS = 3
_SECONDS_PER_HOUR = 3600.0


class TrainingTimeLimitError(RuntimeError):
    """A completed-epoch timing probe projects training beyond its configured wall-time budget."""

    def __init__(self, completed_epochs: int, elapsed_s: float, projected_s: float, limit_s: float) -> None:
        """Record the measured and projected duration so a sweep can persist its pruning reason."""
        self.completed_epochs = completed_epochs
        self.elapsed_s = elapsed_s
        self.projected_s = projected_s
        self.limit_s = limit_s
        super().__init__(
            f"Projected training time {projected_s / _SECONDS_PER_HOUR:.2f} h exceeds "
            f"{limit_s / _SECONDS_PER_HOUR:.2f} h after {completed_epochs} epochs."
        )


@dataclass(frozen=True)
class GradientFitResult[ModelT: eqx.Module]:
    """Outcomes and curves from fitting an Equinox model with gradient descent."""

    best_model: ModelT
    train_losses: list[float]
    val_losses: list[float]
    train_components: dict[str, list[float | None]]
    val_components: dict[str, list[float | None]]
    selected_epoch: int
    stopping_reason: str


def lr_schedule(
    *,
    warmup_steps: int,
    total_steps: int,
    learning_rate: float = 1e-3,
) -> optax.Schedule:
    """Ramp to the learning rate over warm-up batches, then cosine-anneal to zero."""
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=0.0,
    )


def fit_gradient_descent[ModelT: eqx.Module](  # noqa: PLR0913, PLR0912, PLR0915, C901 -- complex training loop with validation and loss terms
    model: ModelT,
    train_dataset: TrajectoryWindowDataset | Sequence[Any],
    val_dataset: TrajectoryWindowDataset | Sequence[Any],
    cfg: TrainingConfig,
    *,
    seed: int,
    losses: Sequence[Loss] = (),
    ctx: LossContext | None = None,
    desc: str = "Training",
    eligibility_start: int = 0,
    loss_fn: Callable[..., Any] | None = None,
) -> GradientFitResult[ModelT]:
    """Fit masked Losses and stop when completed epochs project an over-budget training duration."""
    training_started = perf_counter() if cfg.max_training_hours is not None else 0.0
    epoch_durations: list[float] = []
    if cfg.epochs <= eligibility_start:
        msg = (
            f"Training epochs ({cfg.epochs}) cannot reach schedule eligibility start ({eligibility_start}); "
            f"the epoch budget must be > {eligibility_start} to reach full training."
        )
        raise ValueError(msg)

    if ctx is None:
        ctx = LossContext(y_center=jnp.zeros(1), y_scale=jnp.ones(1), fs=1000.0)

    n_train = len(train_dataset)
    steps_per_epoch = (
        max(1, (n_train + cfg.batch_size - 1) // cfg.batch_size)
        if isinstance(train_dataset, TrajectoryWindowDataset)
        else max(1, n_train)
    )
    total_steps = max(steps_per_epoch * cfg.epochs, 1)
    warmup_steps = min(steps_per_epoch * cfg.warmup_epochs, total_steps - 1)

    schedule = lr_schedule(
        learning_rate=cfg.learning_rate,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )
    optimizer = optax.adamw(learning_rate=schedule, weight_decay=cfg.weight_decay)
    parameter_filter = jax.tree.map(eqx.is_inexact_array, model)
    if isinstance(model, AutoregressiveModel):
        parameter_filter = eqx.tree_at(
            lambda m: (m.y_center, m.y_scale, m.u_center, m.u_scale),
            parameter_filter,
            (False, False, False, False),
        )
    opt_state = optimizer.init(eqx.filter(model, parameter_filter))

    horizon = getattr(model, "horizon", 1)
    curr_mse = next((loss for loss in losses if isinstance(loss, CurriculumMSE)), None)

    def optimizer_step(
        m: ModelT,
        opt_s: optax.OptState,
        compute: Callable[[ModelT], tuple[jax.Array, dict[str, float | None]]],
    ) -> tuple[ModelT, optax.OptState, jax.Array, dict[str, float | None]]:
        """Differentiate and update trainable leaves while preserving all fixed model state."""
        params, fixed = eqx.partition(m, parameter_filter)

        def objective(p: ModelT) -> tuple[jax.Array, dict[str, float | None]]:
            """Reconstruct the model for Loss evaluation."""
            return compute(eqx.combine(p, fixed))

        (loss, parts), grads = eqx.filter_value_and_grad(objective, has_aux=True)(params)
        if any(eqx.is_array(x) for x in jax.tree.leaves(params)):
            updates, opt_s = optimizer.update(grads, opt_s, params)
            m = eqx.combine(eqx.apply_updates(params, updates), fixed)
        return m, opt_s, loss, parts

    @eqx.filter_jit
    def train_step(
        m: ModelT,
        opt_s: optax.OptState,
        batch: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        step_mask: jax.Array,
        active_mask: jax.Array,
    ) -> tuple[ModelT, optax.OptState, jax.Array, dict[str, float | None]]:
        """Train with full Loss history without retracing schedule changes."""
        y_hist, u_hist, u_future, y_target = batch

        def compute(mod: ModelT) -> tuple[jax.Array, dict[str, float | None]]:
            """Score the masked Rollout with full boundary history."""
            rollout_fn: Any = getattr(mod, "rollout", mod)
            model_history = y_hist[:, -mod.n_y :] if isinstance(mod, AutoregressiveModel) else y_hist
            pred = rollout_fn(model_history, u_hist, u_future)
            batch_ctx = eqx.tree_at(lambda context: context.y_hist, ctx, y_hist, is_leaf=lambda x: x is None)
            loss_val, parts = total_loss(
                losses, pred, y_target, batch_ctx, step_mask=step_mask, active_mask=active_mask
            )
            return loss_val, parts

        return optimizer_step(m, opt_s, compute)

    @eqx.filter_jit
    def eval_step(
        m: ModelT,
        batch: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        eval_ctx: LossContext,
    ) -> tuple[jax.Array, dict[str, float | None]]:
        """Validate full Loss Spans using boundary history and the Predictor input window."""
        y_hist, u_hist, u_future, y_target = batch
        rollout_fn: Any = getattr(m, "rollout", m)
        model_history = y_hist[:, -m.n_y :] if isinstance(m, AutoregressiveModel) else y_hist
        pred = rollout_fn(model_history, u_hist, u_future)
        batch_ctx = eqx.tree_at(lambda context: context.y_hist, eval_ctx, y_hist, is_leaf=lambda x: x is None)
        return total_loss(losses, pred, y_target, batch_ctx, step_mask=None)

    def _step_custom(
        m: ModelT,
        opt_s: optax.OptState,
        batch_arrays: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        ep: int,
    ) -> tuple[ModelT, optax.OptState, jax.Array, dict[str, float | None]]:
        """Fit one batch using the caller's Loss function."""
        b0, b1, b2, b3 = batch_arrays
        if loss_fn is None:
            msg = "Custom training step invoked without loss_fn"
            raise RuntimeError(msg)

        def compute_fn(mod: ModelT) -> tuple[jax.Array, dict[str, float | None]]:
            """Evaluate the caller's Loss at the current epoch."""
            return loss_fn(mod, b0, b1, b2, b3, ep)

        return optimizer_step(m, opt_s, compute_fn)

    best_val_loss = float("inf")
    best_model = model
    selected_epoch = eligibility_start
    epochs_without_improvement = 0
    stopping_reason = "max_epochs"

    train_losses: list[float] = []
    val_losses: list[float] = []
    train_components: dict[str, list[float | None]] = collections.defaultdict(list)
    val_components: dict[str, list[float | None]] = collections.defaultdict(list)

    rng = np.random.default_rng(seed)
    pbar = tqdm(range(cfg.epochs), desc=desc)

    ctx = LossContext(y_center=ctx.y_center, y_scale=ctx.y_scale, fs=ctx.fs, epoch=None)
    for epoch in pbar:
        epoch_started = perf_counter() if cfg.max_training_hours is not None else 0.0
        active_mask = jnp.asarray([epoch >= loss.start_epoch for loss in losses])

        L = curr_mse.trusted_length(epoch) if curr_mse is not None else horizon
        step_mask = jnp.asarray(np.arange(horizon) < L, dtype=jnp.float32)

        epoch_loss = 0.0
        comps_sum: dict[str, float] = collections.defaultdict(float)
        comps_count: dict[str, int] = collections.defaultdict(int)
        all_keys: set[str] = set()
        batches = 0

        train_iter: Any = (
            batch_iterator(train_dataset, cfg.batch_size, rng=rng, shuffle=True)
            if isinstance(train_dataset, TrajectoryWindowDataset)
            else train_dataset
        )
        for batch in train_iter:
            if loss_fn is not None:
                b_jax = (
                    jnp.asarray(batch[0]),
                    jnp.asarray(batch[1]),
                    jnp.asarray(batch[2]),
                    jnp.asarray(batch[3]),
                )
                model, opt_state, loss, parts = _step_custom(model, opt_state, b_jax, epoch)
            else:
                model, opt_state, loss, parts = train_step(model, opt_state, batch, step_mask, active_mask)
            loss_val = float(loss)
            epoch_loss += loss_val
            for key, val in parts.items():
                all_keys.add(key)
                if val is not None and not np.isnan(float(val)):
                    comps_sum[key] += float(val)
                    comps_count[key] += 1
            batches += 1

        train_loss = epoch_loss / max(batches, 1)
        train_parts = {k: (comps_sum[k] / comps_count[k] if comps_count[k] > 0 else None) for k in all_keys}

        val_ctx = LossContext(
            y_center=ctx.y_center,
            y_scale=ctx.y_scale,
            fs=ctx.fs,
            epoch=None,
        )
        val_loss_sum = 0.0
        val_comps_sum: dict[str, float] = collections.defaultdict(float)
        val_comps_count: dict[str, int] = collections.defaultdict(int)
        val_keys: set[str] = set()
        total_val_samples = 0

        val_iter: Any = (
            batch_iterator(val_dataset, cfg.batch_size, shuffle=False)
            if isinstance(val_dataset, TrajectoryWindowDataset)
            else val_dataset
        )
        for batch in val_iter:
            b_size = len(batch[0])
            if loss_fn is not None:
                b0 = jnp.asarray(batch[0])
                b1 = jnp.asarray(batch[1])
                b2 = jnp.asarray(batch[2])
                b3 = jnp.asarray(batch[3])
                val_loss_b, val_parts_b = loss_fn(model, b0, b1, b2, b3, None)
            else:
                val_loss_b, val_parts_b = eval_step(model, batch, val_ctx)
            val_loss_sum += float(val_loss_b) * b_size
            total_val_samples += b_size
            for key, val in val_parts_b.items():
                val_keys.add(key)
                if val is not None:
                    val_comps_sum[key] += float(val) * b_size
                    val_comps_count[key] += b_size

        val_loss = val_loss_sum / max(total_val_samples, 1)
        val_parts = {k: (val_comps_sum[k] / val_comps_count[k] if val_comps_count[k] > 0 else None) for k in val_keys}

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        for key, val in train_parts.items():
            train_components[key].append(val)
        for key, val in val_parts.items():
            val_components[key].append(val)

        if np.isnan(train_loss) or np.isnan(val_loss):
            msg = "Loss is NaN. Aborting training."
            raise ValueError(msg)

        if epoch >= eligibility_start:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.copy(model)
                selected_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= cfg.patience:
                    stopping_reason = "early_stopping"
                    break

        if cfg.max_training_hours is not None:
            epoch_finished = perf_counter()
            epoch_durations.append(epoch_finished - epoch_started)
            if len(epoch_durations) >= _TIME_PROBE_EPOCHS:
                elapsed_s = epoch_finished - training_started
                projected_s = elapsed_s + (cfg.epochs - epoch - 1) * float(np.mean(epoch_durations[-2:]))
                limit_s = cfg.max_training_hours * _SECONDS_PER_HOUR
                if projected_s > limit_s:
                    raise TrainingTimeLimitError(epoch + 1, elapsed_s, projected_s, limit_s)

        pbar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
            best_val=f"{best_val_loss:.4f}",
            patience=f"{epochs_without_improvement}/{cfg.patience}",
            epoch=f"{epoch + 1}/{cfg.epochs}",
            eligible="yes" if epoch >= eligibility_start else "no",
        )

    return GradientFitResult(
        best_model=best_model,
        train_losses=train_losses,
        val_losses=val_losses,
        train_components=dict(train_components),
        val_components=dict(val_components),
        selected_epoch=selected_epoch,
        stopping_reason=stopping_reason,
    )
