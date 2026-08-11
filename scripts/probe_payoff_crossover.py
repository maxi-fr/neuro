from __future__ import annotations

import argparse
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from simulate.config import load_config as load_sim_config
from simulate.simulation import Simulation

from neuro.control import WaveformController

if TYPE_CHECKING:
    from neuro.types import FloatArray

# Settles each seed 12 s into the seizure with u = 0, then branches into two open-loop arms --
# u = 0 and a sustained [+2, 0, -2] mA hold -- and scores true EEG mean square over a range of
# lookahead windows. Both arms share a dynamics seed, and the plant draws its noise independently
# of u, so the branch is exact: the arms see the identical noise realisation.
_DESCRIPTION = "Probe where the payoff of the known-good sustained command crosses over."

SEEDS = (69, 1023, 1024, 1025, 1026)
LOOKAHEADS_S = (0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0)
SETTLE_S = 12.0
GOOD_COMMAND = (2.0, 0.0, -2.0)
PLANT_CONFIG = Path("configs/simulation/roast/nonlinear_full_mpc.yaml")


def _run_arm(base_sim_dict: dict, seed: int, command: tuple[float, float, float]) -> FloatArray:
    """Simulate one seed for ``SETTLE_S + max(LOOKAHEADS_S)``, holding ``command`` after the settle."""
    t_end = SETTLE_S + max(LOOKAHEADS_S)
    dt_u = float(base_sim_dict["controller"]["dt"])

    sim_dict = deepcopy(base_sim_dict)
    sim_dict["t_end"] = t_end
    sim_dict["dynamics"]["seed"] = seed
    # A zero controller of the right shape keeps Simulation.from_config's dt/width checks honest;
    # the fixed schedule is installed straight after, since WaveformController.from_config only
    # builds *random* excitation schedules.
    sim_dict["controller"] = {"class_path": "neuro.control.ZeroController", "dt": dt_u, "n_u": len(command)}

    sim = Simulation.from_config(sim_dict)
    schedule = np.zeros((round(t_end / dt_u), len(command)), dtype=np.float64)
    schedule[round(SETTLE_S / dt_u) :] = np.asarray(command, dtype=np.float64)
    sim.controller = WaveformController(dt=dt_u, schedule=schedule)

    # ignore_cleanup_errors: the memmaps stay open on Windows until the arrays are collected.
    with tempfile.TemporaryDirectory(prefix="payoff_probe_", ignore_cleanup_errors=True) as log_dir:
        sim.run(output_dir=log_dir, use_mmap=True)
        if sim.logger is None:
            msg = "Simulation logger is missing after run."
            raise RuntimeError(msg)
        y_mea = np.asarray(sim.logger.signal("sensor_0", "y_mea"), dtype=np.float64)
        start = round(SETTLE_S / sim.dt)
        return y_mea[start : start + round(max(LOOKAHEADS_S) / sim.dt)].copy()


def _eeg_ms(y_branch: FloatArray, lookahead_s: float, dt: float) -> float:
    """Mean square of the 62-channel EEG over the first ``lookahead_s`` of the branch."""
    return float(np.mean(y_branch[: round(lookahead_s / dt)] ** 2))


def main() -> None:
    """Run the probe over ``SEEDS`` and print the per-seed and median ratio tables."""
    parser = argparse.ArgumentParser(description=_DESCRIPTION)
    parser.add_argument("--config", type=Path, default=PLANT_CONFIG)
    args = parser.parse_args()

    base_sim_dict = load_sim_config(args.config)
    dt = float(base_sim_dict["dynamics"]["dt"])

    t0 = time.perf_counter()
    branches: dict[int, tuple[FloatArray, FloatArray]] = {}
    for seed in SEEDS:
        zero = _run_arm(base_sim_dict, seed, (0.0, 0.0, 0.0))
        good = _run_arm(base_sim_dict, seed, GOOD_COMMAND)
        branches[seed] = (zero, good)
        print(f"seed {seed} done ({time.perf_counter() - t0:.0f} s elapsed)", flush=True)

    print("\nPer-seed true EEG mean square (zero -> good, ratio good/zero)\n")
    header = "| seed | " + " | ".join(f"{w:g} s" for w in LOOKAHEADS_S) + " |"
    print(header)
    print("| --- |" + " --- |" * len(LOOKAHEADS_S))
    for seed, (zero, good) in branches.items():
        cells = []
        for w in LOOKAHEADS_S:
            z, g = _eeg_ms(zero, w, dt), _eeg_ms(good, w, dt)
            cells.append(f"{z:.1f} -> {g:.1f} ({g / z:.3f})")
        print(f"| {seed} | " + " | ".join(cells) + " |")

    print("\n| lookahead | median good/zero | improves in |")
    print("| --- | --- | --- |")
    for w in LOOKAHEADS_S:
        ratios = [_eeg_ms(good, w, dt) / _eeg_ms(zero, w, dt) for zero, good in branches.values()]
        print(f"| {w:g} s | {np.median(ratios):.3f} | {sum(r < 1.0 for r in ratios)}/{len(ratios)} |")

    print(f"\nwall time: {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
