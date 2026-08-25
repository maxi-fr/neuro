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
    from neuro.types import FloatArray, IntArray


def float32_tensor(a: FloatArray, device: torch.device) -> Tensor:
    """Move a NumPy array onto ``device`` as a float32 tensor."""
    return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)


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
            loss, parts = loss_fn(model, x_train[idx], y_train[idx], epoch)
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
            val_loss_t, val_parts = loss_fn(model, x_val, y_val, None)
            val_loss = float(val_loss_t.detach())

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
