from __future__ import annotations

import collections
import copy
from typing import TYPE_CHECKING

import numpy as np
import torch
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from torch import Tensor, nn

    from neuro.config import TrainingConfig
    from neuro.types import Float32Array, FloatArray, IntArray


def float32_tensor(a: FloatArray | Float32Array, device: torch.device, *, pin_memory: bool = False) -> Tensor:
    """Move a NumPy array onto ``device`` as a float32 tensor."""
    t = torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)
    return t.pin_memory() if pin_memory and device.type == "cpu" else t


def lr_schedule(
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


def shuffled_batches(n_samples: int, batch_size: int, rng: np.random.Generator) -> Iterator[IntArray]:
    """Yield index batches covering one freshly shuffled pass over the training set."""
    indices = rng.permutation(n_samples)
    for start in range(0, n_samples, batch_size):
        yield indices[start : start + batch_size]


def _evaluate_validation(
    model: nn.Module,
    x_val: Tensor,
    y_val: Tensor,
    cfg: TrainingConfig,
    loss_fn: Callable[[nn.Module, Tensor, Tensor, int | None], tuple[Tensor, dict[str, float]]],
) -> tuple[float, dict[str, float]]:
    """Score ``model`` over mini-batches of ``(x_val, y_val)`` and return the weighted loss and components."""
    n_val = x_val.shape[0]
    if n_val == 0:
        return 0.0, {}
    device = next(model.parameters()).device
    val_loss_sum = 0.0
    val_comps_sum: dict[str, float] = collections.defaultdict(float)
    with torch.no_grad():
        for start in range(0, n_val, cfg.batch_size):
            end = min(start + cfg.batch_size, n_val)
            b_size = end - start
            xb = x_val[start:end].to(device, non_blocking=True)
            yb = y_val[start:end].to(device, non_blocking=True)
            b_loss, b_parts = loss_fn(model, xb, yb, None)
            val_loss_sum += float(b_loss.detach()) * b_size
            for key, val in b_parts.items():
                val_comps_sum[key] += val * b_size
    return val_loss_sum / n_val, {k: v / n_val for k, v in val_comps_sum.items()}


def fit_gradient_descent(  # noqa: PLR0913, PLR0917 -- model, the four tensor blocks and the schedule are the loop's surface
    model: nn.Module,
    x_train: Tensor,
    y_train: Tensor,
    x_val: Tensor,
    y_val: Tensor,
    cfg: TrainingConfig,
    *,
    seed: int,
    loss_fn: Callable[[nn.Module, Tensor, Tensor, int | None], tuple[Tensor, dict[str, float]]],
    desc: str = "Training",
) -> tuple[list[float], list[float], dict[str, list[float]], dict[str, list[float]]]:
    """Run the gradient-descent training loop, leaving ``model`` holding the best-validation weights.

    Generic over any torch module: ``loss_fn`` maps ``(model, x, y, epoch)`` to that batch's loss
    and its unweighted component diagnostics. ``epoch`` is ``None`` for the validation score, so a
    curriculum schedule can trust its full span there. AdamW with the shared warmup-cosine
    schedule, a best-validation snapshot and patience-based early stopping are the same for every
    module.

    Returns ``(train_losses, val_losses, train_components, val_components)``, one entry per epoch
    actually run; the component dicts hold per-epoch unweighted means keyed by loss name.
    """
    rng = np.random.default_rng(seed)
    n_samples = x_train.shape[0]
    device = next(model.parameters()).device

    steps_per_epoch = (n_samples + cfg.batch_size - 1) // cfg.batch_size
    total_steps = max(steps_per_epoch * cfg.epochs, 1)
    warmup_steps = min(steps_per_epoch * cfg.warmup_epochs, total_steps - 1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = lr_schedule(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)

    best_val_loss = float("inf")
    # A torch module is mutable, so the best-so-far snapshot has to be a copy, not an alias.
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    train_losses: list[float] = []
    val_losses: list[float] = []
    train_components: dict[str, list[float]] = collections.defaultdict(list)
    val_components: dict[str, list[float]] = collections.defaultdict(list)

    pbar = tqdm(range(cfg.epochs), desc=desc)
    for epoch in pbar:
        epoch_loss, batches = 0.0, 0
        comps_sum: dict[str, float] = collections.defaultdict(float)
        for idx in shuffled_batches(n_samples, cfg.batch_size, rng):
            xb = x_train[idx].to(device, non_blocking=True)
            yb = y_train[idx].to(device, non_blocking=True)
            loss, parts = loss_fn(model, xb, yb, epoch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += float(loss.detach())
            for key, val in parts.items():
                comps_sum[key] += val
            batches += 1

        train_loss = epoch_loss / batches
        val_loss, val_parts = _evaluate_validation(model, x_val, y_val, cfg, loss_fn)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        for key, val in comps_sum.items():
            train_components[key].append(val / batches)
        for key, val in val_parts.items():
            val_components[key].append(val)

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
        if epochs_without_improvement >= cfg.patience:
            break

    model.load_state_dict(best_state)
    return train_losses, val_losses, dict(train_components), dict(val_components)
