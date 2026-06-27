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
from pathlib import Path
from typing import Any

import numpy as np
import optax
import yaml

from neuro.config import parse_array
from neuro.connectome import compute_gamma, load_connectome
from neuro.jansen_rit import JansenRitParams
from neuro.jansen_rit_jax import enable_x64, from_jansen_rit_params
from neuro.sysid_jax import IdentifyResult, PredConfig, RefineConfig, identify, reduce_via_pca


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
        x = np.asarray(data["universal_x"])[::downsample][:n_steps]  # (n_steps, 6, 76)
        y_mea = np.asarray(data["universal_y_mea"])[::downsample][:n_steps]  # (n_steps, 62)
        u = np.asarray(data["universal_u"])[::downsample][:n_steps]  # (n_steps, n_elec)
    node_output = x[:, 1, :] - x[:, 2, :]  # (n_steps, 76)
    return node_output, y_mea.T, u  # y_data: (62, n_steps); u: (n_steps, n_elec)


def _base_params(red: Any, gamma_modes: np.ndarray, cfg: dict[str, Any]) -> Any:  # noqa: ANN401
    """Assemble the reduced-model base parameters (symmetric non-negative ``w`` init)."""
    n = red.gain.shape[1]
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    w0 = rng.uniform(0.0, float(cfg.get("w_init_scale", 0.1)), (n, n))
    w0 = np.triu(w0, 1)
    w0 = w0 + w0.T
    a_config = parse_array(cfg.get("a_init", 3.25))
    jr = JansenRitParams(
        A=np.full(n, float(a_config)) if isinstance(a_config, (float, int)) else np.asarray(a_config, dtype=np.float64),
        mean_input=float(cfg.get("mean_input", 150.0)),
        sigma=0.0,
        w_weights=w0,
        delay_steps=red.delay_steps,
        eeg_gain=red.gain,
        gamma=gamma_modes,
        K=float(cfg.get("K", 1.0)),
    )
    return from_jansen_rit_params(jr, n)


def _report(result: IdentifyResult) -> None:
    """Print the training loss (a windowed N-step MSE)."""
    print(f"   final loss        : {result.history[-1]:.6f}  (from {result.history[0]:.6f})")


def main() -> None:
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
    dt = float(cfg.get("dt", 1e-4)) * int(cfg.get("downsample", 1))  # effective reduced-model step
    n_steps = int(cfg["n_steps"])
    window = int(cfg.get("window", 300))
    electrodes = parse_array(cfg["target_electrode"])
    electrodes = [str(electrodes)] if isinstance(electrodes, (str, np.str_)) else list(electrodes)
    gamma_spread = float(cfg["gamma_spread"])

    sim_paths = list(cfg["sims"]) if "sims" in cfg else [cfg["sim"]]
    print(f"1. Loading {len(sim_paths)} full-plant simulation(s) and connectome...")
    loaded = [_load_plant(p, int(cfg.get("downsample", 1)), n_steps) for p in sim_paths]
    y_list = [y for _, y, _ in loaded]
    u_list_elec = [u for _, _, u in loaded]  # (n_steps, n_elec) each, per-electrode (raw)
    # PCA basis is fit on every recording's LFP so it spans all noise realisations.
    node_output = np.concatenate([no for no, _, _ in loaded], axis=0)
    conn = load_connectome()

    n_elec_data = u_list_elec[0].shape[1]
    if any(u.shape[1] != n_elec_data for u in u_list_elec):
        msg = "universal_u electrode count differs across sims"
        raise ValueError(msg)
    if n_elec_data != len(electrodes):
        msg = f"config electrodes={electrodes} (n={len(electrodes)}) does not match universal_u columns ({n_elec_data})"
        raise ValueError(msg)
    gamma_full = np.atleast_2d(compute_gamma(conn.centres, electrodes, gamma_spread))  # (n_elec, 76)

    print(f"2. PCA-reducing to N={cfg['n_components']} virtual modes...")
    red = reduce_via_pca(node_output, conn.gain, conn.delays, dt, int(cfg["n_components"]))
    print(f"   explained variance: {red.explained_variance * 100:.1f}%   gain {red.gain.shape}")

    gamma_modes = gamma_full @ red.components.T  # (n_elec, N) -- mirrors JaxSysidPredictor.load
    controls_list = [u @ gamma_modes for u in u_list_elec]  # each (n_steps, N)

    base = _base_params(red, gamma_modes, cfg)
    pred_cfg = PredConfig(horizon=int(cfg["horizon"]), burn_in=int(cfg.get("burn_in", 0)), stride=cfg.get("stride"))
    refine_dict = dict(cfg.get("refine", {}))
    refine_dict["lr"] = _build_lr(refine_dict.get("lr", 1e-2))
    refine_cfg = RefineConfig(**refine_dict)

    print("3. Refining (this can take a minute)...")
    result = identify(
        base,
        list(cfg.get("free_params", ["A", "w_weights"])),
        y_list,
        dt,
        pred_cfg=pred_cfg,
        window=window,
        controls=controls_list,
        w_max=cfg.get("w_max"),
        refine_cfg=refine_cfg,
    )

    print("4. Results:")
    _report(result)

    out = cfg.get("out", "results/sysid_jax_result.npz")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
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
    print(f"   saved -> {out}")


if __name__ == "__main__":
    main()
