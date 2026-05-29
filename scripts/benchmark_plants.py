"""Benchmark the three FHN plant models: FHNPlant, NativeFHNPlant, TVBFHNPlant.

Measures construction time, per-step wall-clock time, simulation throughput
(simulated ms / real ms), and peak Python-heap allocation for each plant.

Note that TVBFHNPlant uses different equations, a different integrator
(HeunStochastic vs Euler), and a different connectome (76-node AAL vs 80-node
HCP), so its results are not numerically comparable — only the performance
profile is meaningful to compare directly.

Usage
-----
    uv run python scripts/benchmark_plants.py
    uv run python scripts/benchmark_plants.py --duration-ms 2000 --step-ms 20
    uv run python scripts/benchmark_plants.py --seed 0
"""

from __future__ import annotations

import argparse
import time
import tracemalloc
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from neuro.plant import FHNPlant, NativeFHNPlant, TVBFHNPlant

if TYPE_CHECKING:
    from collections.abc import Callable

_AnyPlant = FHNPlant | NativeFHNPlant | TVBFHNPlant


@dataclass
class _Result:
    name: str
    n_nodes: int
    construct_s: float
    mean_step_ms: float
    std_step_ms: float
    throughput: float  # simulated ms / real ms
    peak_mb: float


def _benchmark(name: str, factory: Callable[[], _AnyPlant], n_steps: int, step_ms: float) -> _Result:
    t0 = time.perf_counter()
    plant = factory()
    construct_s = time.perf_counter() - t0
    n_nodes: int = plant.n_nodes

    step_times = np.empty(n_steps, dtype=np.float64)
    tracemalloc.start()
    for i in range(n_steps):
        t1 = time.perf_counter()
        plant.step(step_ms)
        step_times[i] = (time.perf_counter() - t1) * 1e3  # convert to ms

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_real_ms = float(step_times.sum())
    return _Result(
        name=name,
        n_nodes=n_nodes,
        construct_s=construct_s,
        mean_step_ms=float(step_times.mean()),
        std_step_ms=float(step_times.std()),
        throughput=step_ms * n_steps / total_real_ms,
        peak_mb=peak / 1024**2,
    )


def _print_table(results: list[_Result], n_steps: int, step_ms: float) -> None:
    sim_ms = step_ms * n_steps
    print(f"\n{n_steps} steps x {step_ms} ms = {sim_ms:.0f} ms simulated\n")
    w = 20
    header = (
        f"{'Plant':<{w}}  {'Nodes':>5}  {'Construct':>10}  {'Step mean+/-std':>20}  {'Throughput':>12}  {'Peak mem':>9}"
    )
    units = f"{'':>{w}}  {'':>5}  {'(s)':>10}  {'(real ms)':>20}  {'(sim/real)':>12}  {'(MB)':>9}"
    sep = "-" * len(header)
    print(header)
    print(units)
    print(sep)
    for r in results:
        step_col = f"{r.mean_step_ms:.1f}+/-{r.std_step_ms:.1f}"
        print(
            f"{r.name:<{w}}  {r.n_nodes:>5}  {r.construct_s:>10.2f}  "
            f"{step_col:>20}  {r.throughput:>10.1f}x  {r.peak_mb:>8.1f}"
        )
    print()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the benchmark runner."""
    parser = argparse.ArgumentParser(description="Benchmark FHN plant model implementations.")
    parser.add_argument(
        "--duration-ms",
        type=float,
        default=500.0,
        help="Total simulated time per plant in milliseconds (default: 500).",
    )
    parser.add_argument(
        "--step-ms",
        type=float,
        default=10.0,
        help="Simulation chunk per step() call in milliseconds (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed passed to each plant (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the plant benchmark and print a comparison table."""
    args = parse_args()
    n_steps = round(args.duration_ms / args.step_ms)
    step_ms = args.step_ms
    seed: int = args.seed

    plants: list[tuple[str, Callable[[], _AnyPlant]]] = [
        ("FHNPlant", lambda: FHNPlant(seed=seed)),
        ("NativeFHNPlant", lambda: NativeFHNPlant(seed=seed)),
        ("TVBFHNPlant", lambda: TVBFHNPlant(seed=seed)),
    ]

    results: list[_Result] = []
    for name, factory in plants:
        print(f"Benchmarking {name}...", flush=True)
        results.append(_benchmark(name, factory, n_steps, step_ms))

    _print_table(results, n_steps, step_ms)


if __name__ == "__main__":
    main()
