from __future__ import annotations

import collections
from typing import TYPE_CHECKING

import numpy as np
import torch
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor, nn
    from torch.utils.data import DataLoader

    from neuro.config import TrainingConfig
    from neuro.types import Float32Array, FloatArray


def float32_tensor(a: FloatArray | Float32Array, device: torch.device, *, pin_memory: bool = False) -> Tensor:
    """Move a NumPy array onto ``device`` as a float32 tensor."""
    t = torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)
    if pin_memory and device.type == "cpu":
        try:
            return t.pin_memory()
        except (RuntimeError, torch.AcceleratorError):
            return t
    return t


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


def _evaluate_validation(
    model: nn.Module,
    val_loader: DataLoader[tuple[Tensor, Tensor, Tensor, Tensor]],
    loss_fn: Callable[[nn.Module, Tensor, Tensor, Tensor, Tensor, int | None], tuple[Tensor, dict[str, float]]],
) -> tuple[float, dict[str, float]]:
    """Score ``model`` over mini-batches of ``val_loader`` and return the weighted loss and components."""
    if len(val_loader) == 0:
        return 0.0, {}
    device = next(model.parameters()).device
    val_loss_sum = 0.0
    val_comps_sum: dict[str, float] = collections.defaultdict(float)
    total_samples = 0
    with torch.no_grad():
        for y_hist, u_hist, u_future, y_target in val_loader:
            b_size = y_hist.shape[0]
            total_samples += b_size
            b_loss, b_parts = loss_fn(
                model,
                y_hist.to(device, non_blocking=True),
                u_hist.to(device, non_blocking=True),
                u_future.to(device, non_blocking=True),
                y_target.to(device, non_blocking=True),
                None,
            )
            val_loss_sum += float(b_loss.detach()) * b_size
            for key, val in b_parts.items():
                val_comps_sum[key] += val * b_size
    if total_samples == 0:
        return 0.0, {}
    return val_loss_sum / total_samples, {k: v / total_samples for k, v in val_comps_sum.items()}


def fit_gradient_descent(  # noqa: PLR0913 -- model, data loaders and config
    model: nn.Module,
    train_loader: DataLoader[tuple[Tensor, Tensor, Tensor, Tensor]],
    val_loader: DataLoader[tuple[Tensor, Tensor, Tensor, Tensor]],
    cfg: TrainingConfig,
    *,
    seed: int,
    loss_fn: Callable[[nn.Module, Tensor, Tensor, Tensor, Tensor, int | None], tuple[Tensor, dict[str, float]]],
    desc: str = "Training",
) -> tuple[list[float], list[float], dict[str, list[float]], dict[str, list[float]]]:
    """Run the gradient-descent training loop, leaving ``model`` holding the best-validation weights.

    Generic over any torch module: ``loss_fn`` maps ``(model, y_hist, u_hist, u_future, y_target, epoch)``
    to that batch's loss and its unweighted component diagnostics. ``epoch`` is ``None`` for the
    validation score, so a curriculum schedule can trust its full span there. AdamW with the shared
    warmup-cosine schedule, a best-validation snapshot and patience-based early stopping are the same
    for every module.

    Returns ``(train_losses, val_losses, train_components, val_components)``, one entry per epoch
    actually run; the component dicts hold per-epoch unweighted means keyed by loss name.
    """
    torch.manual_seed(seed)
    device = next(model.parameters()).device

    steps_per_epoch = len(train_loader)
    total_steps = max(steps_per_epoch * cfg.epochs, 1)
    warmup_steps = min(steps_per_epoch * cfg.warmup_epochs, total_steps - 1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = lr_schedule(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)

    best_val_loss = float("inf")
    # A torch module is mutable, so the best-so-far snapshot has to be a copy on CPU, not an alias on GPU.
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    epochs_without_improvement = 0
    train_losses: list[float] = []
    val_losses: list[float] = []
    train_components: dict[str, list[float]] = collections.defaultdict(list)
    val_components: dict[str, list[float]] = collections.defaultdict(list)

    pbar = tqdm(range(cfg.epochs), desc=desc)
    for epoch in pbar:
        epoch_loss, batches = 0.0, 0
        comps_sum: dict[str, float] = collections.defaultdict(float)
        for y_hist, u_hist, u_future, y_target in train_loader:
            loss, parts = loss_fn(
                model,
                y_hist.to(device, non_blocking=True),
                u_hist.to(device, non_blocking=True),
                u_future.to(device, non_blocking=True),
                y_target.to(device, non_blocking=True),
                epoch,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += float(loss.detach())
            for key, val in parts.items():
                comps_sum[key] += val
            batches += 1

        train_loss = epoch_loss / max(batches, 1)
        val_loss, val_parts = _evaluate_validation(model, val_loader, loss_fn)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        for key, val in comps_sum.items():
            train_components[key].append(val / max(batches, 1))
        for key, val in val_parts.items():
            val_components[key].append(val)

        if np.isnan(train_loss) or np.isnan(val_loss):
            msg = "Loss is NaN. Aborting training."
            raise ValueError(msg)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        pbar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")
        if epochs_without_improvement >= cfg.patience:
            break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return train_losses, val_losses, dict(train_components), dict(val_components)
