from __future__ import annotations

import argparse
import itertools
import tempfile
from pathlib import Path

import numpy as np
import torch

from neuro.config import StftGeometry
from neuro.control.benchmark import (
    format_closed_loop_table,
    format_open_loop_table,
    run_observable_benchmark,
    run_waveform_benchmark,
)
from neuro.predictor.module import AutoregressiveMLP
from neuro.transforms import Standardizer


def _random_layers(rng: np.random.Generator, sizes: list[int]) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Random layer weights and biases."""
    return tuple(
        (rng.uniform(-1.0, 1.0, (out, inp)) / np.sqrt(inp), rng.uniform(-1.0, 1.0, out) / np.sqrt(inp))
        for inp, out in itertools.pairwise(sizes)
    )


def _build_synthetic_waveform_checkpoint(  # noqa: PLR0913 -- synthetic model architecture parameters
    path: Path,
    *,
    depth: int = 0,
    n_y: int = 4,
    n_u: int = 3,
    horizon: int = 10,
    n_channels: int = 2,
    n_controls: int = 2,
) -> Path:
    """Save a synthetic Waveform checkpoint."""
    rng = np.random.default_rng(42)
    in_size = n_y * n_channels + n_u * n_controls
    scalers = {
        "u_mean": rng.uniform(-1.0, 1.0, n_controls),
        "u_scale": rng.uniform(0.5, 2.0, n_controls),
        "y_mean": rng.uniform(-1.0, 1.0, n_channels),
        "y_scale": rng.uniform(0.5, 2.0, n_channels),
    }
    model = AutoregressiveMLP(
        n_y=n_y,
        n_u=n_u,
        horizon=horizon,
        n_channels=n_channels,
        n_controls=n_controls,
        n_outputs=n_channels,
        hidden_size=16 if depth > 0 else 0,
        depth=depth,
        activation="relu",
        dt=0.01,
        residual=False,
        y_std=Standardizer(center=scalers["y_mean"], scale=scalers["y_scale"]),
        u_std=Standardizer(center=scalers["u_mean"], scale=scalers["u_scale"]),
    )
    linears = [m for m in model.layers if isinstance(m, torch.nn.Linear)]
    sizes = [in_size, *([16] * depth), n_channels]
    with torch.no_grad():
        for lin, (w, b) in zip(linears, _random_layers(rng, sizes), strict=True):
            lin.weight.copy_(torch.as_tensor(w, dtype=torch.float32))
            lin.bias.copy_(torch.as_tensor(b, dtype=torch.float32))
    model.save(path)
    return path


def _build_synthetic_observable_checkpoint(  # noqa: PLR0913 -- synthetic model architecture parameters
    path: Path,
    *,
    depth: int = 0,
    n_y: int = 2,
    n_u: int = 2,
    horizon: int = 4,
    n_channels: int = 2,
    n_controls: int = 2,
) -> tuple[Path, Path]:
    """Save a synthetic Observable checkpoint and healthy envelope npz."""
    rng = np.random.default_rng(43)
    geom = StftGeometry(n_segment=20, n_hop=5, band_hz=(4.0, 16.0), n_bin_pool=2)
    fs = 50.0
    n_values = geom.n_values(fs)
    n_outputs = n_channels * n_values
    in_size = n_y * n_outputs + n_u * n_controls
    scalers = {
        "u_mean": rng.uniform(-1.0, 1.0, n_controls),
        "u_scale": rng.uniform(0.5, 2.0, n_controls),
        "y_mean": rng.uniform(-1.0, 1.0, n_outputs),
        "y_scale": rng.uniform(0.5, 2.0, n_outputs),
    }
    model = AutoregressiveMLP(
        n_y=n_y,
        n_u=n_u,
        horizon=horizon,
        n_channels=n_channels,
        n_controls=n_controls,
        hidden_size=16 if depth > 0 else 0,
        depth=depth,
        activation="relu",
        dt=geom.n_hop / fs,
        n_outputs=n_outputs,
        geometry=geom,
        residual=False,
        y_std=Standardizer(center=scalers["y_mean"], scale=scalers["y_scale"]),
        u_std=Standardizer(center=scalers["u_mean"], scale=scalers["u_scale"]),
    )
    linears = [m for m in model.layers if isinstance(m, torch.nn.Linear)]
    sizes = [in_size, *([16] * depth), n_outputs]
    with torch.no_grad():
        for lin, (w, b) in zip(linears, _random_layers(rng, sizes), strict=True):
            lin.weight.copy_(torch.as_tensor(w, dtype=torch.float32))
            lin.bias.copy_(torch.as_tensor(b, dtype=torch.float32))
    model.save(path)

    env_path = path.parent / "healthy_env.npz"
    np.savez_compressed(
        env_path,
        Pref_frames=np.full((n_channels, n_values), -2.0),
        fs=fs,
        n_segment=geom.n_segment,
        n_hop=geom.n_hop,
        band_hz=np.asarray(geom.band_hz if geom.band_hz is not None else [-1.0, -1.0]),
        n_bin_pool=geom.n_bin_pool,
        kernel=geom.kernel,
        kernel_width=geom.kernel_width,
    )
    return path, env_path


def main() -> None:
    """Run the solver benchmark suite across optimal control problems and print summary tables."""
    parser = argparse.ArgumentParser(description="Benchmark neurostimulation OCP solvers")
    parser.add_argument("--repeats", type=int, default=5, help="Number of open-loop timing repeats")
    parser.add_argument("--steps", type=int, default=10, help="Number of closed-loop MPC steps")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        print("=" * 80)
        print("1. LINEAR WAVEFORM OCP (Depth-0 / DMDc Predictor, Quadratic Tracking Cost)")
        print("=" * 80)
        lin_art = _build_synthetic_waveform_checkpoint(tmp / "lin_wf", depth=0, horizon=10)
        open_comp, closed_comp = run_waveform_benchmark(
            lin_art,
            horizon=10,
            u_max=0.5,
            w_y=1.0,
            w_u=0.1,
            kirchhoff=True,
            n_repeats=args.repeats,
            num_steps=args.steps,
            include_bipolar_boxqp=True,
        )
        print(format_open_loop_table(open_comp))
        print()
        print(format_closed_loop_table(closed_comp))
        print()

        print("=" * 80)
        print("2. NONLINEAR WAVEFORM OCP (Depth-2 MLP Predictor, Quadratic Tracking Cost)")
        print("=" * 80)
        nonlin_art = _build_synthetic_waveform_checkpoint(tmp / "nonlin_wf", depth=2, horizon=10)
        open_comp_nl, closed_comp_nl = run_waveform_benchmark(
            nonlin_art,
            horizon=10,
            u_max=0.5,
            w_y=1.0,
            w_u=0.1,
            kirchhoff=True,
            n_repeats=args.repeats,
            num_steps=args.steps,
            include_bipolar_boxqp=True,
        )
        print(format_open_loop_table(open_comp_nl))
        print()
        print(format_closed_loop_table(closed_comp_nl))
        print()

        print("=" * 80)
        print("3. OBSERVABLE OCP (Observable Predictor, Log-Power Spectral Hinge Cost)")
        print("=" * 80)
        obs_art, env_path = _build_synthetic_observable_checkpoint(tmp / "obs_model", depth=1, horizon=4)
        open_comp_obs, closed_comp_obs = run_observable_benchmark(
            obs_art,
            env_path,
            horizon=4,
            u_max=0.5,
            w_u=1.0,
            w_hinge=2.0,
            kirchhoff=True,
            n_repeats=args.repeats,
            num_steps=args.steps,
        )
        print(format_open_loop_table(open_comp_obs))
        print()
        print(format_closed_loop_table(closed_comp_obs))
        print()


if __name__ == "__main__":
    main()
