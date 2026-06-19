"""System Identification feasibility test on simulated EEG data.

Usage
-----
    uv run python scripts/run_sysid.py --data results/orchestrator/log.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import casadi as ca
import numpy as np

from neuro.connectome import Connectome, load_connectome
from neuro.jansen_rit import JansenRitParams, simulate_network
from neuro.jansen_rit_casadi import JRSymbolicParams, heun_step
from neuro.seizure import DT
from neuro.sysid import SysIDSolver
from utils.channel_selection import leadfield_focus_scores

DATA_PATH = Path("results/orchestrator/log.npz")
OUTPUT_DIR = Path("simulations/sysid")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the SysID runner."""
    parser = argparse.ArgumentParser(description="SysID test on EEG.")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Path to log.npz")
    parser.add_argument("--n-select", type=int, default=10, help="Number of reduced regions.")
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps to fit.")
    parser.add_argument("--horizon", type=int, default=1000, help="Prediction horizon.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for outputs.")
    return parser.parse_args()


def select_regions_and_channels(connectome: Connectome, n_select: int) -> tuple[list[int], list[int]]:
    """Pick the reduced regions and the channels that observe them."""
    # A simplified version: just pick the top connected regions to region 0
    weights = connectome.weights
    coupling = weights[:, 0] + weights[0, :]
    regions = np.argsort(coupling)[::-1][:n_select].tolist()

    focus_scores = leadfield_focus_scores(connectome.gain, regions)
    channels = np.argsort(focus_scores)[::-1][:n_select].tolist()
    return regions, channels


def main(  # noqa: PLR0915
    *,
    data: Path = DATA_PATH,
    n_select: int = 10,
    steps: int = 1000,
    horizon: int = 1000,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Run System Identification."""
    output_dir.mkdir(parents=True, exist_ok=True)
    connectome = load_connectome(speed=50.0)

    try:
        blob = np.load(data, allow_pickle=True)
        eeg = np.asarray(blob["universal_y_mea"], dtype=np.float64).T  # (62, T)
        u_data = np.asarray(blob["u"], dtype=np.float64).T  # (N_elec, T)
    except FileNotFoundError:
        print(f"Data file not found: {data}. Creating synthetic data...")
        _, x_open = simulate_network(
            params=JansenRitParams(A=4.0), connectome=connectome, duration=10.0, dt=DT, seed=42
        )
        eeg = connectome.gain @ (x_open[1, :, :] - x_open[2, :, :])
        u_data = np.zeros((eeg.shape[1], 1))

    regions, channels = select_regions_and_channels(connectome, n_select)

    steps = min(steps, eeg.shape[1] - 1)

    chan_std = eeg[:, :steps].std(axis=1)
    chan_mean = eeg[:, :steps].mean(axis=1)
    gain_scaled = connectome.gain[np.ix_(channels, regions)] / chan_std[channels, None]
    y_meas = (eeg[channels] - chan_mean[channels, None]) / chan_std[channels, None]  # (n_channels, T)

    w_weights = connectome.weights[np.ix_(regions, regions)]
    delay_steps = np.round(connectome.delays[np.ix_(regions, regions)] / DT).astype(np.int64)
    gamma = np.eye(len(regions))[:, : u_data.shape[1]]  # Identity mapping for simplified inputs

    print(f"Identifying {steps} steps ({steps * DT:.2f} s) of a {len(regions)}-region reduced model...")

    base = JansenRitParams(A=3.25)

    solver = SysIDSolver(
        dt=DT,
        base_params=base,
        free_params=["A", "mean_input", "eeg_gain"],
        gamma=gamma,
        gain=gain_scaled,
        w_weights=w_weights,
        delay_steps=delay_steps,
        n_steps=steps,
    )

    x0_hist = [np.zeros((6, len(regions))) for _ in range(solver.max_delay + 1)]

    res = solver.solve(
        u_data=u_data[:steps],
        y_data=y_meas[:, :steps].T,
        x0_history=x0_hist,
    )

    print("\n--- Identified Parameters ---")
    print(f"Cost: {res.cost:.4f}")
    print(f"A:\n{np.array2string(np.atleast_1d(res.params.A), precision=3)}")
    print(f"mean_input:\n{np.array2string(np.atleast_1d(res.params.mean_input), precision=3)}")

    h_max = min(horizon, eeg.shape[1] - steps)
    if h_max > 0:
        print("\n--- Free-run prediction vs horizon ---")

        sym_params = JRSymbolicParams(
            A=np.atleast_1d(res.params.A),
            B=res.params.B,
            a=res.params.a,
            b=res.params.b,
            C1=res.params.C1,
            C2=res.params.C2,
            C3=res.params.C3,
            C4=res.params.C4,
            e0=res.params.e0,
            v0=res.params.v0,
            r=res.params.r,
            mean_input=np.atleast_1d(res.params.mean_input),
            sigma=res.params.sigma,
            eeg_gain=res.eeg_gain if res.eeg_gain is not None else gain_scaled,
            K=1.0,
            w_weights=w_weights,
            delay_steps=delay_steps,
        )

        x_hist = list(x0_hist)
        preds = np.empty((h_max, len(channels)))
        for h in range(h_max):
            u_h = u_data[steps + h]
            u_proj = gamma @ u_h

            x_next = heun_step(x_hist, u_proj, sym_params, DT)  # type: ignore
            x_hist.insert(0, np.array(ca.DM(x_next)))
            x_hist.pop()

            y_h = eeg(x_hist[0], sym_params.eeg_gain)  # type: ignore
            preds[h] = np.array(ca.DM(y_h)).flatten()

        future = y_meas[:, steps : steps + h_max].T
        hold = y_meas[:, steps - 1]

        for h in (10, 50, 100, 500):
            if h > h_max:
                break
            fut_h = future[:h]
            model_h = float(np.linalg.norm(preds[:h] - fut_h) / np.linalg.norm(fut_h))
            persist_h = float(np.linalg.norm(fut_h - hold) / np.linalg.norm(fut_h))
            verdict = "model wins" if model_h < persist_h else "persistence wins"
            print(f"  {h * DT * 1e3:5.0f} ms : model {model_h:.4f} / persistence {persist_h:.4f}  ({verdict})")


if __name__ == "__main__":
    cli_args = parse_args()
    main(
        data=cli_args.data,
        n_select=cli_args.n_select,
        steps=cli_args.steps,
        horizon=cli_args.horizon,
        output_dir=cli_args.output_dir,
    )
