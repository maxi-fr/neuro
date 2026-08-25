from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

from typing import TYPE_CHECKING

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from simulate.config import build_component, load_config

from neuro.checkpoint import load_mlp
from neuro.control.trajopt_mpc import WaveformMLPModel

if TYPE_CHECKING:
    from neuro.types import FloatArray

N_CHANNELS_TO_PLOT = 3


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the prediction-quality plot."""
    parser = argparse.ArgumentParser(description="Plot open-loop NN-predictor forecasts against the plant.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Simulation config; supplies the plant, the EEG montage, and controller.artifact.",
    )
    parser.add_argument(
        "--artifact",
        type=str,
        default=None,
        help="Predictor checkpoint basename; defaults to the config's controller.artifact.",
    )
    parser.add_argument("--t-end", type=float, default=None, help="Open-loop duration (s); defaults to config t_end.")
    parser.add_argument("--anchors", type=int, default=4, help="Number of forecast anchor times.")
    parser.add_argument(
        "--out", type=Path, default=None, help="Output PNG path (default: <config-stem>_predictions.png)."
    )
    return parser.parse_args()


def run_open_loop(config: dict, t_end: float) -> tuple[FloatArray, float]:
    """Run the plant open-loop (zero control); return EEG at the controller step and that step."""
    dt = float(config["dynamics"]["dt"])
    ctrl_dt = float(config["controller"]["dt"])
    downsample = round(ctrl_dt / dt)

    dynamics = build_component(config["dynamics"])
    sensor = build_component(config["sensors"])
    u = np.zeros(dynamics.n_inputs)

    eeg = []
    t = 0.0
    for k in range(round(t_end / dt)):
        if k % downsample == 0:
            eeg.append(sensor.evaluate(t, dynamics.x, u)[0])
        dynamics.evaluate(t, u)
        t += dt
    return np.asarray(eeg), ctrl_dt


def free_run(model: WaveformMLPModel, eeg: FloatArray, anchor: int, horizon: int) -> tuple[FloatArray, FloatArray]:
    """Free-run (zero control) ``horizon`` steps from ``anchor``; return (predicted, true) EEG."""
    n_y, n_controls = model.n_y, model.n_controls
    x = model.initial_state()
    for i in range(n_y):
        x = model.absorb(x, eeg[anchor - n_y + i], np.zeros(n_controls))
    preds = []
    for _ in range(horizon):
        x = np.asarray(model.discrete_dynamics(jnp.asarray(x), jnp.zeros(n_controls), 0.0, 0.0))
        preds.append(np.asarray(model.output(jnp.asarray(x))).reshape(-1))
    return np.asarray(preds), eeg[anchor : anchor + horizon]


def main() -> None:
    """Run the plant open-loop, overlay predictor forecasts on the realized EEG, and save a figure."""
    args = parse_args()
    config = load_config(args.config)
    t_end = args.t_end if args.t_end is not None else float(config["t_end"])
    artifact = args.artifact if args.artifact is not None else config["controller"]["artifact"]

    print(f"Running plant open-loop for {t_end}s from {args.config} ...", flush=True)
    eeg, dt_model = run_open_loop(config, t_end)
    model = WaveformMLPModel.from_checkpoint(artifact)
    ckpt = load_mlp(artifact)

    n_y, horizon = model.n_y, ckpt.horizon
    anchors = np.linspace(n_y + 1, len(eeg) - horizon - 1, args.anchors).astype(int)
    time = np.arange(len(eeg)) * dt_model

    fig, (ax_power, ax_chan) = plt.subplots(1, 2, figsize=(13, 4.2))

    power = (eeg**2).mean(axis=1)
    ax_power.plot(time, power, color="k", lw=0.8, label="plant EEG power")
    rmses = []
    for anchor in anchors:
        pred, true = free_run(model, eeg, anchor, horizon)
        horizon_t = (np.arange(horizon) + anchor) * dt_model
        ax_power.plot(horizon_t, (pred**2).mean(axis=1), color="C3", lw=1.5)
        rmses.append(float(np.sqrt(((pred - true) ** 2).mean())))
    ax_power.plot([], [], color="C3", lw=1.5, label=f"{horizon}-step free-run forecast")
    ax_power.set(xlabel="t (s)", ylabel="mean-square EEG", title=f"{args.config.stem}: forecasts vs plant")
    ax_power.legend(loc="upper left", fontsize=8)

    pred, true = free_run(model, eeg, anchors[len(anchors) // 2], horizon)
    horizon_t = np.arange(horizon) * dt_model
    for channel in range(min(N_CHANNELS_TO_PLOT, pred.shape[1])):
        ax_chan.plot(horizon_t, true[:, channel], color=f"C{channel}", lw=1.3)
        ax_chan.plot(horizon_t, pred[:, channel], color=f"C{channel}", lw=1.3, ls="--")
    ax_chan.plot([], [], color="k", lw=1.3, label="true")
    ax_chan.plot([], [], color="k", lw=1.3, ls="--", label="predicted")
    ax_chan.set(
        xlabel="horizon step (s)", ylabel="EEG (raw)", title=f"ch 0-{N_CHANNELS_TO_PLOT - 1}: true vs predicted"
    )
    ax_chan.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    out = args.out if args.out is not None else Path(f"{args.config.stem}_predictions.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"Mean {horizon}-step free-run RMSE over {len(anchors)} anchors: {np.mean(rmses):.4g}", flush=True)
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
