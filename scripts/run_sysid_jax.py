"""Run pure-JAX system identification of a reduced Jansen-Rit network.

The autodiff alternative to ``scripts/run_sysid.py`` (CasADi PCMS): PCA-reduce the
76-node plant to ``N`` virtual modes, then *refine* the reduced ``A`` / ``w_weights``
with Optax against a windowed N-step-ahead MSE (state evolves naturally over one
continuous rollout per recording, real recorded tES stimulation driving it; see
:func:`neuro.sysid_jax.identify`).

Usage::

    uv run python scripts/run_sysid_jax.py --config configs/sysid/sysid_jax_config.yaml
"""

from __future__ import annotations

import argparse
import datetime
import shutil
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import yaml

from neuro.config import parse_array
from neuro.connectome import compute_gamma, load_connectome
from neuro.jansen_rit import JansenRitParams
from neuro.jansen_rit_jax import eeg_jax, enable_x64, from_jansen_rit_params
from neuro.sysid_jax import IdentifyResult, PredConfig, RefineConfig, identify, reduce_via_pca, rollout_tbptt
from utils.plotting import plot_multistep_predictions


def _build_lr(lr: float | dict[str, Any]) -> float | optax.Schedule:
    """Turn a config ``lr`` (constant or ``{"name": ..., ...}`` schedule spec) into an optax LR."""
    if not isinstance(lr, dict):
        return lr
    name = lr.get("name")
    if name == "exponential_decay":
        return optax.exponential_decay(
            init_value=float(lr["init_value"]),
            transition_steps=int(lr["transition_steps"]),
            decay_rate=float(lr["decay_rate"]),
            staircase=bool(lr.get("staircase", False)),
        )
    if name == "cosine_decay":
        return optax.cosine_decay_schedule(
            init_value=float(lr["init_value"]),
            decay_steps=int(lr["decay_steps"]),
            alpha=float(lr.get("alpha", 0.0)),
        )
    if name == "linear":
        return optax.linear_schedule(
            init_value=float(lr["init_value"]),
            end_value=float(lr["end_value"]),
            transition_steps=int(lr["transition_steps"]),
        )
    msg = f"Unknown learning rate schedule name: {name!r}"
    raise ValueError(msg)


def _load_plant(sim_path: str, downsample: int, n_steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load downsampled per-region LFP, target EEG, and per-electrode tES from a plant log."""
    with np.load(sim_path) as data:
        x = np.asarray(data["x"])[::downsample][:n_steps]  # (n_steps, 6, 76)
        y_mea = np.asarray(data["y_mea"])[::downsample][:n_steps]  # (n_steps, 62)
        u = np.asarray(data["u"])[::downsample][:n_steps]  # (n_steps, n_elec)
    node_output = x[:, 1, :] - x[:, 2, :]  # (n_steps, 76)
    return node_output, y_mea.T, u  # y_data: (62, n_steps); u: (n_steps, n_elec)


def _base_params(red: Any, gamma_modes: np.ndarray, cfg: dict[str, Any]) -> Any:  # noqa: ANN401
    """Assemble the reduced-model base parameters (symmetric non-negative ``w`` init)."""
    n = red.gain.shape[1]
    train_cfg = cfg.get("training", {})
    seed = int(train_cfg.get("seed", 0))
    rng = np.random.default_rng(seed)

    model_cfg = cfg.get("model", {})
    w_init_scale = float(model_cfg.get("w_init_scale", 0.1))
    w0 = rng.uniform(0.0, w_init_scale, (n, n))
    w0 = np.triu(w0, 1)
    w0 = w0 + w0.T

    a_config = parse_array(model_cfg.get("a_init", 3.25))
    A_val = (
        np.full(n, float(a_config))
        if isinstance(a_config, (float, int, np.float64, np.int64))
        else np.asarray(a_config, dtype=np.float64)
    )

    mean_input = float(model_cfg.get("mean_input", 150.0))
    K = float(model_cfg.get("K", 1.0))

    jr = JansenRitParams(
        A=A_val,
        mean_input=mean_input,
        sigma=0.0,
        w_weights=w0,
        delay_steps=red.delay_steps,
        eeg_gain=red.gain,
        gamma=gamma_modes,
        K=K,
    )
    return from_jansen_rit_params(jr, n)


def _report(result: IdentifyResult) -> None:
    """Print the training loss (a windowed N-step MSE)."""
    print(f"   final loss        : {result.history[-1]:.6f}  (from {result.history[0]:.6f})")


def main() -> None:  # noqa: C901, PLR0915
    """Run reduced-network JAX system identification from a YAML config."""
    parser = argparse.ArgumentParser(description="Pure-JAX reduced Jansen-Rit system identification.")
    parser.add_argument(
        "--config", type=str, default="configs/sysid/sysid_jax_config.yaml", help="Path to config YAML."
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return
    cfg = yaml.safe_load(config_path.read_text())

    enable_x64()

    sim_cfg = cfg.get("simulation", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    dt_base = float(sim_cfg.get("dt", 1e-4))
    downsample = int(sim_cfg.get("downsample", 1))
    dt = dt_base * downsample  # effective reduced-model step
    n_steps = int(sim_cfg["n_steps"])
    window = int(train_cfg.get("window", 300))

    target_electrode_raw = model_cfg["target_electrode"]
    electrodes = parse_array(target_electrode_raw)
    electrodes = [str(electrodes)] if isinstance(electrodes, (str, np.str_)) else list(electrodes)
    gamma_spread = float(model_cfg["gamma_spread"])

    # Resolve simulation files
    data_path_str = sim_cfg["data_path"]
    data_path = Path(data_path_str)
    if not data_path.is_dir():
        print(f"data_path is not a valid directory: {data_path}")
        return
    sim_paths = sorted([str(p) for p in data_path.glob("*.npz")])
    n_sims = sim_cfg.get("n_sims")
    if n_sims is not None:
        sim_paths = sim_paths[: int(n_sims)]

    if not sim_paths:
        print("No simulation files specified in config.")
        return

    print(f"1. Loading {len(sim_paths)} full-plant simulation(s) and connectome...")
    loaded = [_load_plant(p, downsample, n_steps) for p in sim_paths]
    y_list = [y for _, y, _ in loaded]
    u_list_elec = [u for _, _, u in loaded]  # (n_steps, n_elec) each, per-electrode (raw)
    # PCA basis is fit on every recording's LFP so it spans all noise realisations.
    node_output = np.concatenate([no for no, _, _ in loaded], axis=0)
    conn = load_connectome()

    n_elec_data = u_list_elec[0].shape[1]
    if any(u.shape[1] != n_elec_data for u in u_list_elec):
        msg = "u electrode count differs across sims"
        raise ValueError(msg)
    if n_elec_data != len(electrodes):
        msg = f"config electrodes={electrodes} (n={len(electrodes)}) does not match u columns ({n_elec_data})"
        raise ValueError(msg)
    gamma_full = np.atleast_2d(compute_gamma(conn.centres, electrodes, gamma_spread))  # (n_elec, 76)

    n_components = int(model_cfg["n_components"])
    print(f"2. PCA-reducing to N={n_components} virtual modes...")
    red = reduce_via_pca(node_output, conn.gain, conn.delays, dt, n_components)
    print(f"   explained variance: {red.explained_variance * 100:.1f}%   gain {red.gain.shape}")

    gamma_modes = gamma_full @ red.components.T  # (n_elec, N) -- mirrors JaxSysidPredictor.load
    controls_list = [u @ gamma_modes for u in u_list_elec]  # each (n_steps, N)

    base = _base_params(red, gamma_modes, cfg)
    horizon = int(model_cfg["horizon"])
    burn_in = int(model_cfg.get("burn_in", 0))
    stride = model_cfg.get("stride")
    pred_cfg = PredConfig(horizon=horizon, burn_in=burn_in, stride=stride)

    refine_steps = int(train_cfg.get("steps", 1000))
    refine_lr = _build_lr(train_cfg.get("lr", 1e-2))
    refine_clip = float(train_cfg.get("clip", 1.0))
    refine_cfg = RefineConfig(steps=refine_steps, lr=refine_lr, clip=refine_clip)

    print("3. Refining (this can take a minute)...")
    free_params = list(model_cfg.get("free_params", ["A", "w_weights"]))
    w_max = model_cfg.get("w_max")
    if w_max is not None:
        w_max = float(w_max)

    result = identify(
        base,
        free_params,
        y_list,
        dt,
        pred_cfg=pred_cfg,
        window=window,
        controls=controls_list,
        w_max=w_max,
        refine_cfg=refine_cfg,
    )

    print("4. Results:")
    _report(result)

    # Reorganize output to use artifact directory
    artifact_dir_str = cfg.get("artifact")
    if artifact_dir_str is None:
        local_now = datetime.datetime.now(datetime.UTC).astimezone()
        artifact_dir_str = f"artifacts/sysid_jax_{local_now.strftime('%Y-%m-%d_%H-%M-%S')}"
    artifact_dir = Path(artifact_dir_str)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Copy config file
    shutil.copy2(config_path, artifact_dir / config_path.name)

    # Save results to artifact directory
    out_npz_path = artifact_dir / "sysid_jax_result.npz"
    np.savez(
        out_npz_path,
        A=np.asarray(result.params.A),
        w_weights=np.asarray(result.params.w_weights),
        K=float(result.params.K),
        mean_input=np.asarray(result.params.mean_input),
        eeg_gain=red.gain,
        components=red.components,
        delay_steps=red.delay_steps,
        history=np.asarray(result.history),
        dt=float(dt),
    )
    print(f"   saved -> {out_npz_path}")

    # Copy to legacy path for backward compatibility if configured
    legacy_out = cfg.get("out")
    if legacy_out:
        legacy_out_path = Path(legacy_out)
        legacy_out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_npz_path, legacy_out_path)
        print(f"   copied to legacy path -> {legacy_out_path}")

    # Plot and save loss curves
    loss_plot_path = artifact_dir / "loss_curve.png"
    plt.figure(figsize=(8, 5))
    plt.plot(result.history, label="Training Loss (Windowed N-Step MSE)", color="#1f77b4", linewidth=1.5)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("JAX Reduced-Model System Identification Loss")
    plt.legend()
    plt.grid(visible=True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(str(loss_plot_path), dpi=300)
    plt.close()
    print(f"   plot saved -> {loss_plot_path}")

    # Plot predictions overlay (comparison.png) using plot_multistep_predictions
    print("5. Generating prediction overlay plot...")
    try:
        # Load the first simulation's full state trajectory
        with np.load(sim_paths[0]) as data:
            x_full = np.asarray(data["x"])[::downsample][:n_steps]  # (n_steps, 6, 76)

        x_reduced = x_full @ red.components.T  # (n_steps, 6, N)
        C_y = y_list[0].shape[0]  # number of EEG channels
        n_plot_samples = min(200, n_steps - horizon)

        Y_pred_abs = []
        for t0 in range(n_plot_samples):
            x0_t = jnp.asarray(x_reduced[t0])
            u_t = jnp.asarray(controls_list[0][t0 : t0 + horizon])
            # Roll out horizon steps
            x_pred_traj = rollout_tbptt(x0_t, u_t, result.params, dt, window=horizon)
            # Project to EEG
            y_pred_overlay = np.asarray(eeg_jax(x_pred_traj, result.params.eeg_gain))  # (C_y, horizon)
            Y_pred_abs.append(y_pred_overlay.T)  # (horizon, C_y)

        Y_pred_abs = np.array(Y_pred_abs)  # (n_plot_samples, horizon, C_y)
        Y_val = y_list[0].T[: n_plot_samples + horizon]  # (n_samples, C_y)

        comparison_plot_path = artifact_dir / "comparison.png"
        fig, _ = plot_multistep_predictions(
            y_true=Y_val[:n_plot_samples],
            y_pred=Y_pred_abs[:n_plot_samples],
            dt=dt,
            channels=list(range(min(4, C_y))),
            stride=horizon,
            title=f"EEG {horizon}-Step Ahead Prediction (Fitted Reduced Model)",
        )
        fig.savefig(str(comparison_plot_path), dpi=300)
        plt.close(fig)
        print(f"   comparison plot saved -> {comparison_plot_path}")
    except Exception as e:  # noqa: BLE001
        print(f"   failed to generate comparison plot: {e}")


if __name__ == "__main__":
    main()
