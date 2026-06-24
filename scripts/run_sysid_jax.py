"""Run pure-JAX system identification of a reduced Jansen-Rit network.

The autodiff alternative to ``scripts/run_sysid.py`` (CasADi PCMS): PCA-reduce the
76-node plant to ``N`` virtual modes, then *refine* the reduced ``A`` / ``w_weights``
with Optax against the uncontrolled EEG statistics.

Usage::

    uv run python scripts/run_sysid_jax.py --config configs/sysid/sysid_jax_config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import optax
import yaml

from neuro.connectome import load_connectome
from neuro.jansen_rit import JansenRitParams
from neuro.jansen_rit_jax import enable_x64, from_jansen_rit_params
from neuro.sysid_jax import (
    IdentifyResult,
    LossWeights,
    RefineConfig,
    StatConfig,
    identify,
    model_eeg,
    reduce_via_pca,
)
from utils.processing import compute_psd


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


def _load_plant(sim_path: str, downsample: int, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Load downsampled per-region LFP and EEG from a full-plant simulation log."""
    with np.load(sim_path) as data:
        x = np.asarray(data["universal_x"])[::downsample][:n_steps]  # (n_steps, 6, 76)
        y_mea = np.asarray(data["universal_y_mea"])[::downsample][:n_steps]  # (n_steps, 62)
    node_output = x[:, 1, :] - x[:, 2, :]  # (n_steps, 76)
    return node_output, y_mea.T  # y_data: (62, n_steps)


def _base_params(red: Any, cfg: dict[str, Any]) -> Any:  # noqa: ANN401
    """Assemble the reduced-model base parameters (symmetric non-negative ``w`` init)."""
    n = red.gain.shape[1]
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    w0 = rng.uniform(0.0, float(cfg.get("w_init_scale", 0.1)), (n, n))
    w0 = np.triu(w0, 1)
    w0 = (w0 + w0.T) / 2
    jr = JansenRitParams(
        A=np.full(n, float(cfg.get("a_init", 3.25)))
        if isinstance(cfg.get("a_init"), (float, int))
        else np.asarray(cfg.get("a_init")),
        mean_input=float(cfg.get("mean_input", 150.0)),
        sigma=0.0,
        w_weights=w0,
        delay_steps=red.delay_steps,
        eeg_gain=red.gain,
        gamma=np.zeros((1, n)),
        K=float(cfg.get("K", 1.0)),
    )
    return from_jansen_rit_params(jr, n)


def _report(  # noqa: PLR0913
    result: IdentifyResult, y_list: list[np.ndarray], dt: float, window: int, band: tuple[float, float], burn_in: int
) -> None:
    """Print offline fit metrics (mean over recordings): spectral-shape corr and FC error."""
    n = result.params.n_nodes

    y_fit = np.asarray(model_eeg(result.params, jnp.zeros((6, n)), jnp.zeros((y_list[0].shape[1], n)), dt, window))[
        :, burn_in:
    ]

    def _logpsd(sig: np.ndarray) -> np.ndarray:
        freqs, pxx = compute_psd(sig, dt_ms=dt * 1000.0)
        mask = (freqs >= band[0]) & (freqs <= band[1])
        return np.log(pxx[:, mask] + 1e-30)

    fit_logpsd = _logpsd(y_fit).ravel()
    fc_fit = np.corrcoef(y_fit)
    psd_corrs = [float(np.corrcoef(_logpsd(y[:, burn_in:]).ravel(), fit_logpsd)[0, 1]) for y in y_list]
    fc_errs = [float(np.mean((np.corrcoef(y[:, burn_in:]) - fc_fit) ** 2)) for y in y_list]
    print(f"   final loss        : {result.history[-1]:.5f}  (from {result.history[0]:.5f})")
    print(f"   log-PSD corr      : {np.mean(psd_corrs):.4f}  (mean over {len(y_list)} recording(s))")
    print(f"   spatial-FC MSE    : {np.mean(fc_errs):.5f}")


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
    burn_in = int(cfg.get("burn_in", 0))
    band = tuple(cfg.get("band", (4.0, 40.0)))

    sim_paths = list(cfg["sims"]) if "sims" in cfg else [cfg["sim"]]
    print(f"1. Loading {len(sim_paths)} full-plant simulation(s) and connectome...")
    loaded = [_load_plant(p, int(cfg.get("downsample", 1)), n_steps) for p in sim_paths]
    y_list = [y for _, y in loaded]
    # PCA basis is fit on every recording's LFP so it spans all noise realisations.
    node_output = np.concatenate([no for no, _ in loaded], axis=0)
    conn = load_connectome()

    print(f"2. PCA-reducing to N={cfg['n_components']} virtual modes...")
    red = reduce_via_pca(node_output, conn.gain, conn.delays, dt, int(cfg["n_components"]))
    print(f"   explained variance: {red.explained_variance * 100:.1f}%   gain {red.gain.shape}")

    base = _base_params(red, cfg)
    weights = LossWeights(**cfg.get("weights", {}))
    stat_cfg = StatConfig(nperseg=cfg.get("nperseg"), band=band)
    refine_dict = dict(cfg.get("refine", {}))
    refine_dict["lr"] = _build_lr(refine_dict.get("lr", 1e-2))
    refine_cfg = RefineConfig(**refine_dict)

    print("3. Refining (this can take a minute)...")
    result = identify(
        base,
        list(cfg.get("free_params", ["A", "w_weights"])),
        y_list,
        dt,
        window=window,
        burn_in=burn_in,
        w_max=cfg.get("w_max"),
        weights=weights,
        stat_cfg=stat_cfg,
        refine_cfg=refine_cfg,
    )

    print("4. Results:")
    _report(result, y_list, dt, window, band, burn_in)

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
        window=int(window),
        band=np.asarray(band, dtype=np.float64),
    )
    print(f"   saved -> {out}")


if __name__ == "__main__":
    main()
