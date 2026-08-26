from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from neuro.config import load_config, resolve_data_files
from neuro.observable import log_observable
from neuro.predictor.gradient import fit_gradient_descent, float32_tensor
from neuro.predictor.module import AutoregressiveMLP
from neuro.predictor.observable_module import ObservableMLP, mlp_stack
from neuro.predictor.observable_train import ObservableData, log_mse, prepare_observable_data

if TYPE_CHECKING:
    from torch import Tensor

    from neuro.config import NNPredictorConfig

_MA_SWEEP = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
# The solver only ever sees the control gradient, so it may be smaller than the state's but not
# orders of magnitude smaller.
_MIN_GRADIENT_RATIO = 1e-2


def _observable_model(cfg: NNPredictorConfig, data: ObservableData) -> ObservableMLP:
    """Build the spec's Predictor: lift, one shared Frame transition, affine log readout."""
    obs, mdl = cfg.observable, cfg.model
    assert obs is not None  # noqa: S101 -- the caller validated the config carries the block
    return ObservableMLP(
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
        residual=obs.residual,
    )


class _DirectMap(nn.Module):
    """The horizon-wide direct map ``g(x_0, u) -> l_hat`` of the exploration document's sections 1-6.

    Stage 2's arm only. It gets no checkpoint type, no symbolic bridge and no MPC path, so that
    "iterated beats direct" is measured on this data rather than assumed in either direction.
    """

    def __init__(self, n_in: int, n_frames: int, n_out: int, hidden: int, depth: int) -> None:
        """Build a ``depth``-hidden-layer MLP straight from the input row to every Frame at once."""
        super().__init__()
        self.n_frames = n_frames
        self.n_out = n_out
        self.layers = mlp_stack([n_in, *[hidden] * depth, n_frames * n_out], "softplus")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map the input row to ``(B, n_frames, n_out)`` in one shot."""
        return self.layers(x).reshape(x.shape[0], self.n_frames, self.n_out)


def _tensors(data: ObservableData, device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Standardized inputs and targets on ``device``, the four tensors the shared fit consumes."""
    return (
        float32_tensor(data.x_train, device),
        float32_tensor(data.l_std.transform(data.targets_train), device),
        float32_tensor(data.x_val, device),
        float32_tensor(data.l_std.transform(data.targets_val), device),
    )


def _mse_loss(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    epoch: int | None,  # noqa: ARG001 -- the shared loop's schedule clock; MSE is epoch-independent
) -> tuple[Tensor, dict[str, float]]:
    """Standardized log-Observable MSE over the Frame grid."""
    return torch.mean((model(x) - y) ** 2), {}


def _train_arm(model: nn.Module, cfg: NNPredictorConfig, data: ObservableData, seed: int) -> float:
    """Train one arm on the shared data and schedule; return its held-out MSE in raw log units."""
    torch.manual_seed(seed)
    device = torch.device(cfg.training.device)
    tensors = _tensors(data, device)
    fit_gradient_descent(model.to(device), *tensors, cfg.training, seed=seed, loss_fn=_mse_loss)
    return log_mse(model, tensors[2], data.targets_val, data.l_std)


def stage_control_blindness(cfg: NNPredictorConfig, data_files: list[str], seeds: list[int]) -> bool:
    """Full model against the Control-blind one at identical architecture, schedule and seed set.

    *Kill:* if the full model does not beat the Control-blind one by more than the seed-to-seed
    spread. No amount of solver speed fixes a Predictor the solver cannot push on.
    """
    print("\n== Stage 1: control-blindness gate ==", flush=True)
    obs = cfg.observable
    assert obs is not None  # noqa: S101 -- main() validated the config carries the block
    blind_cfg = cfg.model_copy(update={"observable": obs.model_copy(update={"control_blind": True})})
    full_data = prepare_observable_data(cfg, data_files)
    blind_data = prepare_observable_data(blind_cfg, data_files)

    full = np.array([_train_arm(_observable_model(cfg, full_data), cfg, full_data, s) for s in seeds])
    blind = np.array([_train_arm(_observable_model(cfg, blind_data), cfg, blind_data, s) for s in seeds])

    gap = float(blind.mean() - full.mean())
    spread = float(max(full.std(ddof=1), blind.std(ddof=1)))
    print(f"  full  log-MSE: {full.mean():.5f} +- {full.std(ddof=1):.5f}  {np.round(full, 5).tolist()}")
    print(f"  blind log-MSE: {blind.mean():.5f} +- {blind.std(ddof=1):.5f}  {np.round(blind, 5).tolist()}")
    print(f"  gap (blind - full): {gap:.5f}, seed-to-seed spread: {spread:.5f}")
    passed = gap > spread
    print(f"  {'PASS' if passed else 'KILL'}: the full model {'beats' if passed else 'does not beat'} the blind one")
    return passed


def stage_direct_vs_iterated(cfg: NNPredictorConfig, data_files: list[str], seeds: list[int]) -> None:
    """Measure the horizon-wide direct map against the recursion on this data. Reported, not gated."""
    print("\n== Stage 2: direct-vs-iterated arm ==", flush=True)
    obs = cfg.observable
    assert obs is not None  # noqa: S101
    data = prepare_observable_data(cfg, data_files)
    n_frames, n_out = data.targets_train.shape[1], data.targets_train.shape[2]

    iterated = np.array([_train_arm(_observable_model(cfg, data), cfg, data, s) for s in seeds])
    direct = np.array(
        [
            _train_arm(
                _DirectMap(data.x_train.shape[1], n_frames, n_out, obs.lift_hidden, obs.lift_depth),
                cfg,
                data,
                s,
            )
            for s in seeds
        ]
    )
    print(f"  iterated log-MSE: {iterated.mean():.5f} +- {iterated.std(ddof=1):.5f}")
    print(f"  direct   log-MSE: {direct.mean():.5f} +- {direct.std(ddof=1):.5f}")
    print(f"  {'iterated' if iterated.mean() < direct.mean() else 'direct'} wins on held-out log-MSE")


def _incumbent_log_mse(model: AutoregressiveMLP, cfg: NNPredictorConfig, data: ObservableData, stride: int) -> float:
    """Push the incumbent rollout through the identical geometry and score it on the same Frames."""
    obs = cfg.observable
    assert obs is not None  # noqa: S101
    horizon, geometry, k = obs.horizon, obs.geometry(), model.priming_steps

    squared, count = 0.0, 0
    for u, y in data.val_trajs:
        t0s = range(k, len(y) - horizon, stride)
        if not t0s:
            continue
        states = model.prime_many(np.stack([y[t0 - k : t0] for t0 in t0s]), np.stack([u[t0 - k : t0] for t0 in t0s]))
        y_pred = model.rollout_many(states, np.stack([u[t0 : t0 + horizon] for t0 in t0s]))
        y_true = np.stack([y[t0 + 1 : t0 + 1 + horizon] for t0 in t0s])
        residual = log_observable(y_pred, geometry, cfg.fs) - log_observable(y_true, geometry, cfg.fs)
        squared += float((residual**2).sum())
        count += residual.size
    if count == 0:
        msg = "no held-out trajectory is long enough to hold one Control Horizon."
        raise SystemExit(msg)
    return squared / count


def stage_incumbent_baseline(
    cfg: NNPredictorConfig, data_files: list[str], seeds: list[int], incumbent: Path, stride: int
) -> bool:
    """Score the observable Predictor against the incumbent checkpoint pushed through the same geometry.

    *Kill:* if the observable Predictor does not beat the incumbent. The baseline is the incumbent,
    not the training mean.
    """
    print("\n== Stage 3: incumbent baseline ==", flush=True)
    model = AutoregressiveMLP.load(incumbent)

    data = prepare_observable_data(cfg, data_files)
    observable = np.array([_train_arm(_observable_model(cfg, data), cfg, data, s) for s in seeds])
    baseline = _incumbent_log_mse(model, cfg, data, stride)

    print(f"  observable log-MSE: {observable.mean():.5f} +- {observable.std(ddof=1):.5f}")
    print(f"  incumbent  log-MSE: {baseline:.5f}  ({incumbent})")
    passed = bool(observable.mean() < baseline)
    print(f"  {'PASS' if passed else 'KILL'}: the observable Predictor {'beats' if passed else 'loses to'} it")
    return passed


def _electrode_index(name: str, projection: Path) -> int:
    """Resolve an electrode name to its control index through the field-projection npz."""
    with np.load(projection) as npz:
        labels = [str(label) for label in npz["channel_labels"]]
    if name not in labels:
        msg = f"electrode {name!r} is not among {labels} in {projection}"
        raise SystemExit(msg)
    return labels.index(name)


def stage_gradient_sanity(
    cfg: NNPredictorConfig,
    data_files: list[str],
    seed: int,
    electrode: str,
    projection: Path,
) -> bool:
    """Sweep a constant Control Current on one electrode and check the forecast responds to it.

    *Kill:* if forecast log power does not fall monotonically with the current, or if
    ``||grad_u l_hat||`` is orders of magnitude below ``||grad_x0 l_hat||`` -- the solver only ever
    sees the former.
    """
    print("\n== Stage 4: gradient sanity ==", flush=True)
    obs = cfg.observable
    assert obs is not None  # noqa: S101
    data = prepare_observable_data(cfg, data_files)
    model = _observable_model(cfg, data)
    _train_arm(model, cfg, data, seed)

    index = _electrode_index(electrode, projection)
    n_hist = cfg.model.n_y * data.n_channels + cfg.model.n_u * data.n_controls
    x_val = float32_tensor(data.x_val, torch.device(cfg.training.device))
    row = x_val[len(x_val) // 2]

    # Kirchhoff: the swept electrode's current returns through the others in equal shares.
    means = []
    for current in _MA_SWEEP:
        u = np.full((obs.horizon, data.n_controls), current / (data.n_controls - 1))
        u[:, index] = -current
        probe = row.clone()
        probe[n_hist:] = float32_tensor(data.u_std.transform(u).reshape(-1), row.device)
        with torch.no_grad():
            standardized = model(probe[None, :]).cpu().numpy().astype(np.float64)
        means.append(float(data.l_std.inverse_transform(standardized).mean()))

    deltas = np.diff(means)
    monotone = bool(np.all(deltas < 0))
    print(f"  mean forecast log power at {electrode} = {_MA_SWEEP.tolist()} mA: {np.round(means, 4).tolist()}")
    print(f"  step-to-step change: {np.round(deltas, 4).tolist()} ({'monotone' if monotone else 'NOT monotone'})")

    grad_u, grad_x0 = _gradient_norms(model, row, n_hist)
    ratio = grad_u / grad_x0 if grad_x0 > 0 else float("inf")
    print(f"  ||grad_u l_hat|| = {grad_u:.4e}, ||grad_x0 l_hat|| = {grad_x0:.4e}, ratio = {ratio:.4e}")
    passed = monotone and ratio > _MIN_GRADIENT_RATIO
    print(f"  {'PASS' if passed else 'KILL'}: the solver {'has' if passed else 'has no'} a gradient to descend")
    return passed


def _gradient_norms(model: nn.Module, row: torch.Tensor, n_hist: int) -> tuple[float, float]:
    """Frobenius norms of the forecast's Jacobian in the control block and in the history block."""
    probe = row.clone().requires_grad_(True)  # noqa: FBT003
    jacobian = torch.autograd.functional.jacobian(lambda x: model(x[None, :])[0], probe, vectorize=True)
    flat = jacobian.reshape(-1, row.numel())
    return float(torch.linalg.norm(flat[:, n_hist:])), float(torch.linalg.norm(flat[:, :n_hist]))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the observable-predictor gate probe."""
    parser = argparse.ArgumentParser(
        description="Run stages 1-4 of docs/observable_prediction/spec.md decision 11, with kill criteria."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/nn_predictor/observable_stft.yaml"))
    parser.add_argument("--data-path", type=str, default=None, help="Override the config's data path.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Seed offsets per arm.")
    parser.add_argument(
        "--incumbent",
        type=Path,
        default=Path("artifacts/nonlinear_mse02_psd/model"),
        help="Incumbent MLP checkpoint basename.",
    )
    parser.add_argument("--stride", type=int, default=25, help="t0 spacing when scoring the incumbent.")
    parser.add_argument("--electrode", type=str, default="TP9", help="Electrode swept in stage 4.")
    parser.add_argument("--field-projection", type=Path, default=Path("data/roast_field_projection_3d.npz"))
    return parser.parse_args()


def main() -> None:
    """Run stages 1-4 in order, stopping at the first kill criterion that trips."""
    args = parse_args()
    cfg = load_config(args.config)
    if cfg.observable is None:
        msg = f"{args.config} has no 'observable' block."
        raise SystemExit(msg)
    data_files = resolve_data_files(cfg, args.data_path)
    seeds = [cfg.training.seed + offset for offset in args.seeds]

    if not stage_control_blindness(cfg, data_files, seeds):
        msg_0 = "stage 1 kill criterion tripped; later stages do not run."
        raise SystemExit(msg_0)
    stage_direct_vs_iterated(cfg, data_files, seeds)
    if not stage_incumbent_baseline(cfg, data_files, seeds, args.incumbent, args.stride):
        msg_0 = "stage 3 kill criterion tripped; later stages do not run."
        raise SystemExit(msg_0)
    if not stage_gradient_sanity(cfg, data_files, seeds[0], args.electrode, args.field_projection):
        msg_0 = "stage 4 kill criterion tripped; stages 5 and 6 do not run."
        raise SystemExit(msg_0)

    print("\nStages 1-4 passed. Stage 5: scripts/probe_solve_time.py --artifact <observable checkpoint>.")
    print("Stage 6: run configs/simulation/observable_psd_mpc.yaml against mse02_psd_mpc_spectral.yaml.")


if __name__ == "__main__":
    main()
