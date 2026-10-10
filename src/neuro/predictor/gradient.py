from __future__ import annotations

import collections
import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from neuro.predictor.data import TrajectoryWindowDataset, batch_iterator
from neuro.predictor.losses import CurriculumMSE, LossContext, total_loss

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from neuro.config import TrainingConfig
    from neuro.predictor.losses import Loss
    from neuro.types import Float32Array, FloatArray


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
    optimizer: object = None,
    *,
    warmup_steps: int,
    total_steps: int,
    learning_rate: float = 1e-3,
) -> optax.Schedule:
    """Linear warm-up over ``warmup_steps`` batches, then cosine anneal to zero over remainder."""
    del optimizer
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=0.0,
    )


def float32_tensor(a: FloatArray | Float32Array, *args: object, **kwargs: object) -> jax.Array:
    """Move a NumPy array into a float32 JAX array."""
    del args, kwargs
    return jnp.asarray(np.ascontiguousarray(a), dtype=jnp.float32)


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
    """Run gradient-descent optimization in JAX, returning outcomes and the best model."""
    if cfg.epochs <= eligibility_start:
        msg = (
            f"Training epochs ({cfg.epochs}) cannot reach schedule eligibility start ({eligibility_start}); "
            f"the epoch budget must be > {eligibility_start} to reach full training."
        )
        raise ValueError(msg)

    if ctx is None:
        ctx = LossContext(y_center=jnp.zeros(1), y_scale=jnp.ones(1), fs=1000.0)

    n_train = len(train_dataset)
    steps_per_epoch = max(1, (n_train + cfg.batch_size - 1) // cfg.batch_size)
    total_steps = max(steps_per_epoch * cfg.epochs, 1)
    warmup_steps = min(steps_per_epoch * cfg.warmup_epochs, total_steps - 1)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=0.0,
    )
    optimizer = optax.adamw(learning_rate=schedule, weight_decay=cfg.weight_decay)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    horizon = getattr(model, "horizon", 1)
    curr_mse = next((loss for loss in losses if isinstance(loss, CurriculumMSE)), None)

    @eqx.filter_jit
    def train_step(
        m: ModelT,
        opt_s: optax.OptState,
        batch: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        step_mask: jax.Array,
        step_ctx: LossContext,
    ) -> tuple[ModelT, optax.OptState, jax.Array, dict[str, float | None]]:
        y_hist, u_hist, u_future, y_target = batch

        def compute(mod: ModelT) -> tuple[jax.Array, dict[str, float | None]]:
            rollout_fn: Any = getattr(mod, "rollout", mod)
            pred = rollout_fn(y_hist, u_hist, u_future)
            loss_val, parts = total_loss(losses, pred, y_target, step_ctx, step_mask=step_mask)
            return loss_val, parts

        (loss, parts), grads = eqx.filter_value_and_grad(compute, has_aux=True)(m)
        params = eqx.filter(m, eqx.is_array)
        if any(eqx.is_array(x) for x in jax.tree.leaves(params)):
            updates, next_opt_s = optimizer.update(grads, opt_s, params)
            next_m = eqx.apply_updates(m, updates)
        else:
            next_m = m
            next_opt_s = opt_s
        return next_m, next_opt_s, loss, parts

    @eqx.filter_jit
    def eval_step(
        m: ModelT,
        batch: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        eval_ctx: LossContext,
    ) -> tuple[jax.Array, dict[str, float | None]]:
        y_hist, u_hist, u_future, y_target = batch
        rollout_fn: Any = getattr(m, "rollout", m)
        pred = rollout_fn(y_hist, u_hist, u_future)
        return total_loss(losses, pred, y_target, eval_ctx, step_mask=None)

    def _step_custom(
        m: ModelT,
        opt_s: optax.OptState,
        batch_arrays: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        ep: int,
    ) -> tuple[ModelT, optax.OptState, jax.Array, dict[str, float | None]]:
        b0, b1, b2, b3 = batch_arrays
        if loss_fn is None:
            msg = "Custom training step invoked without loss_fn"
            raise RuntimeError(msg)

        def compute_fn(mod: ModelT) -> tuple[jax.Array, dict[str, float | None]]:
            return loss_fn(mod, b0, b1, b2, b3, ep)

        (loss, parts), grads = eqx.filter_value_and_grad(compute_fn, has_aux=True)(m)
        params = eqx.filter(m, eqx.is_array)
        if any(eqx.is_array(x) for x in jax.tree.leaves(params)):
            updates, next_opt_s = optimizer.update(grads, opt_s, params)
            next_m = eqx.apply_updates(m, updates)
        else:
            next_m = m
            next_opt_s = opt_s
        return next_m, next_opt_s, loss, parts

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

    for epoch in pbar:
        epoch_ctx = LossContext(
            y_center=ctx.y_center,
            y_scale=ctx.y_scale,
            fs=ctx.fs,
            epoch=epoch,
        )

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
                model, opt_state, loss, parts = train_step(model, opt_state, batch, step_mask, epoch_ctx)
            loss_val = float(loss)
            epoch_loss += loss_val
            for key, val in parts.items():
                all_keys.add(key)
                if val is not None:
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
