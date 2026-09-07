from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from neuro.control.benchmark import (
    format_closed_loop_table,
    format_open_loop_table,
    run_observable_benchmark,
    run_waveform_benchmark,
)

# The deployed Predictors and horizons, not synthetic stand-ins: the transcriptions differ by how
# their decision-variable count scales with the Predictor's trailing window, so a toy checkpoint
# compares them on the one axis where they agree. Horizons are the 1 s Control Horizon each
# controller config deploys, on that Predictor's own grid.
_WAVEFORM_ARTIFACT = "artifacts/cmp_waveform_mlp_1p5s/model"  # n_y=15, n_u=10, 62 ch, 3 electrodes
_WAVEFORM_HORIZON = 50  # 50 * 0.02 s = 1 s, as configs/simulation/mse02_psd_mpc.yaml deploys
_OBSERVABLE_LINEAR = "artifacts/cmp_observable_dmd_hop5/model"  # depth-0, so the OCP is a convex QP
_OBSERVABLE_NONLINEAR = "artifacts/cmp_observable_mlp_hop5/model"
_OBSERVABLE_HORIZON = 10  # 10 * 0.1 s = 1 s on the n_hop=5 Frame grid
_OBSERVABLE_ENVELOPE = "data/healthy_psd_hop5.npz"


def _require_artifacts(*paths: str) -> None:
    """Exit with the missing checkpoints named, rather than failing deep inside a model load."""
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        listed = "\n  ".join(missing)
        print(
            f"missing benchmark inputs:\n  {listed}\n\n"
            "The suite runs the deployed Predictors, not synthetic stand-ins. Train them with "
            "scripts/run_nn_predictor.py on the matching configs/nn_predictor/*.yaml first.",
            file=sys.stderr,
        )
        raise SystemExit(1)


def main() -> None:
    """Run the solver benchmark suite across optimal control problems and print summary tables."""
    parser = argparse.ArgumentParser(description="Benchmark neurostimulation OCP solvers")
    parser.add_argument("--repeats", type=int, default=2, help="Number of open-loop timing repeats")
    parser.add_argument("--steps", type=int, default=3, help="Number of closed-loop MPC steps")
    parser.add_argument("--waveform-only", action="store_true", help="Skip the two Observable OCP experiments")
    args = parser.parse_args()
    # A solver the formulation defeats is logged and dropped, so the handler is what makes an
    # absent row legible rather than silent. cyipopt logs a line per callback at INFO, which would
    # bury the tables, so only the benchmark module is turned up.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("neuro.control.benchmark").setLevel(logging.INFO)
    logging.getLogger("cyipopt").setLevel(logging.WARNING)

    required = [_WAVEFORM_ARTIFACT]
    if not args.waveform_only:
        required += [_OBSERVABLE_LINEAR, _OBSERVABLE_NONLINEAR, _OBSERVABLE_ENVELOPE]
    _require_artifacts(*required)

    # Kirchhoff twice over, because the two formulations reach the same feasible set by different
    # routes: as a hard equality it is a constraint row the solver must drive to tolerance, and by
    # null-space reduction it holds identically, at the price of a coupled polytope on the bounds.
    experiments: tuple[tuple[str, bool, bool], ...] = (
        ("WAVEFORM OCP, Kirchhoff as hard equality", True, False),
        ("WAVEFORM OCP, Kirchhoff by null-space reduction", False, True),
    )
    for index, (title, kirchhoff, reduce_kirchhoff) in enumerate(experiments, start=1):
        print("=" * 80)
        print(f"{index}. {title} (Depth-2 MLP Predictor, 1 s Horizon)")
        print("=" * 80)
        open_comp, closed_comp = run_waveform_benchmark(
            _WAVEFORM_ARTIFACT,
            horizon=_WAVEFORM_HORIZON,
            u_max=2.0,
            w_y=1.0,
            w_u=10.0,
            kirchhoff=kirchhoff,
            reduce_kirchhoff=reduce_kirchhoff,
            n_repeats=args.repeats,
            num_steps=args.steps,
        )
        print(format_open_loop_table(open_comp))
        print()
        print(format_closed_loop_table(closed_comp))
        print(flush=True)

    if args.waveform_only:
        return

    for index, (label, artifact) in enumerate(
        (("LINEAR", _OBSERVABLE_LINEAR), ("NONLINEAR", _OBSERVABLE_NONLINEAR)), start=3
    ):
        print("=" * 80)
        print(f"{index}. {label} OBSERVABLE OCP (Log-Power Spectral Hinge Cost, 1 s Horizon)")
        print("=" * 80)
        open_comp_obs, closed_comp_obs = run_observable_benchmark(
            artifact,
            _OBSERVABLE_ENVELOPE,
            horizon=_OBSERVABLE_HORIZON,
            u_max=2.0,
            w_u=10.0,
            w_hinge=10.0,
            kirchhoff=True,
            n_repeats=args.repeats,
            num_steps=args.steps,
        )
        print(format_open_loop_table(open_comp_obs))
        print()
        print(format_closed_loop_table(closed_comp_obs))
        print(flush=True)


if __name__ == "__main__":
    main()
