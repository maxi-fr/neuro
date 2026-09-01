from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from simulate.config import load_config

from neuro.connectome import Connectome
from neuro.jansen_rit import JansenRitDynamics, JansenRitParams, lfp
from neuro.predictor.jansen_rit import JansenRitModel, lfp_jax
from neuro.predictor.oracle import JansenRitOracleEstimator
from neuro.seizure import EZ_REGIONS, PZ_REGIONS

if TYPE_CHECKING:
    from neuro.types import FloatArray

BASE_CONFIG = Path("configs/simulation/jansen_rit_oracle_mpc.yaml")
SEEDS = (69, 7, 123)


@eqx.filter_jit
def _rollout_lfp(model: JansenRitModel, z0: jax.Array, u: jax.Array, n_knots: int) -> jax.Array:
    """Roll the deterministic Predictor ``n_knots`` knots forward, returning region LFP per knot.

    Returns
    -------
    jax.Array
        Region LFP of shape ``(n_knots, n_nodes)``, one row per Control Horizon knot.
    """

    def step(z: jax.Array, _: None) -> tuple[jax.Array, jax.Array]:
        z_next = model.discrete_dynamics(z, u, 0.0, model.knot_dt)
        x_ode, _history, _k = model.unpack_state(z_next)
        return z_next, lfp_jax(x_ode)

    _, ys = jax.lax.scan(step, z0, None, length=n_knots)
    return ys


def _handover_trace(config: dict[str, Any], seed: int, substeps: int) -> tuple[FloatArray, FloatArray]:
    """Run the uncontrolled Plant once, capturing the oracle handover and the Plant LFP at each knot.

    The Estimator is stepped at the Predictor's ``dt`` exactly as the closed loop steps it, so the
    delay buffer it hands over is the one the Rollout would actually start from.

    Returns
    -------
    tuple of FloatArray
        Packed Predictor states of shape ``(n_knots, model.n)`` and the Plant's region LFP at the
        same knots, shape ``(n_knots, n_nodes)``.
    """
    plant_cfg = copy.deepcopy(config["dynamics"])
    plant_cfg.pop("class_path", None)  # the Plant is built directly, not through simulate's dispatch
    plant_cfg["seed"] = seed
    plant_cfg["log"] = "none"
    plant = JansenRitDynamics.from_config(plant_cfg)

    estimator = JansenRitOracleEstimator.from_config(config["estimator"])
    plant_dt = float(plant_cfg["dt"])
    est_stride = round(estimator.dt / plant_dt)
    knot_stride = est_stride * substeps
    n_steps = round(float(config["t_end"]) / plant_dt)
    u_zero = np.zeros(plant.stim.n_controls, dtype=np.float64)

    states, lfps = [], []
    for step in range(n_steps):
        t = step * plant_dt
        if step % est_stride == 0:
            z, _ = estimator.update(t, plant.x.reshape(-1), u_zero)
            if step % knot_stride == 0:
                states.append(np.asarray(z, dtype=np.float64))
                lfps.append(lfp(plant.x).copy())
        plant.update(t, u_zero)
    return np.asarray(states), np.asarray(lfps)


def _region_groups(connectome: Connectome) -> dict[str, list[int]]:
    """Region indices of the EZ, the PZ, and the healthy remainder."""
    ez = [connectome.region_index[name] for name in EZ_REGIONS]
    pz = [connectome.region_index[name] for name in PZ_REGIONS]
    rest = sorted(set(range(len(connectome.region_labels))) - set(ez) - set(pz))
    return {"ez": ez, "pz": pz, "healthy": rest, "all": list(range(len(connectome.region_labels)))}


def _rollouts_from_starts(
    model: JansenRitModel,
    states: FloatArray,
    lfps: FloatArray,
    *,
    horizon: int,
    start_stride: int,
) -> tuple[FloatArray, FloatArray]:
    """Roll the Predictor from every sampled start knot and pair it with what the Plant did.

    Returns
    -------
    tuple of FloatArray
        Predicted and actual region LFP, both of shape ``(n_starts, horizon, n_nodes)``.
    """
    u_zero = jnp.zeros(model.m)
    starts = range(0, len(states) - horizon, start_stride)
    predicted, actual = [], []
    for start in starts:
        predicted.append(np.asarray(_rollout_lfp(model, jnp.asarray(states[start]), u_zero, horizon)))
        actual.append(lfps[start + 1 : start + 1 + horizon])
    return np.asarray(predicted), np.asarray(actual)


def horizon_error_table(
    config: dict[str, Any],
    *,
    seeds: tuple[int, ...],
    horizon: int,
    start_stride: int,
) -> list[dict[str, Any]]:
    """Measure how the deterministic Predictor drifts from the stochastic Plant along the Control Horizon.

    With the oracle handover the initial state is exact, so everything reported here is the cost of
    the Predictor's missing noise alone: ``nrmse`` is the divergence normalised by the Plant's own
    LFP scale, and ``ptp_ratio_*`` is the amplitude the Rollout predicts over the window relative to
    the amplitude the Plant actually reaches -- the deterministic bias, left uncorrected.
    """
    problem = config["controller"]["problem"]
    substeps = int(problem["substeps"])
    connectome = Connectome.from_config(config["dynamics"]["connectome"])
    model = JansenRitModel.from_plant_components(
        JansenRitParams.from_config(config["dynamics"]["params"]),
        conn=connectome,
        dt=float(problem["dt"]),
        n_nodes=len(connectome.region_labels),
        substeps=substeps,
    )
    groups = _region_groups(connectome)

    all_predicted, all_actual, scales = [], [], []
    for seed in seeds:
        states, lfps = _handover_trace(config, seed, substeps)
        predicted, actual = _rollouts_from_starts(model, states, lfps, horizon=horizon, start_stride=start_stride)
        all_predicted.append(predicted)
        all_actual.append(actual)
        scales.append(float(np.std(lfps)))
    predicted = np.concatenate(all_predicted, axis=0)  # (n_starts, horizon, n_nodes)
    actual = np.concatenate(all_actual, axis=0)
    scale = float(np.mean(scales))

    knot_ms = 1e3 * float(problem["dt"]) * substeps
    rows = []
    for lag in range(horizon):
        rmse = np.sqrt(np.mean((predicted[:, : lag + 1] - actual[:, : lag + 1]) ** 2, axis=(1, 2)))
        row: dict[str, Any] = {
            "horizon_knots": lag + 1,
            "horizon_ms": (lag + 1) * knot_ms,
            "nrmse": float(np.mean(rmse) / scale),
            "nrmse_sd": float(np.std(rmse) / scale),
        }
        row |= amplitude_bias(predicted[:, : lag + 1], actual[:, : lag + 1], groups)
        rows.append(row)
    return rows


def amplitude_bias(predicted: FloatArray, actual: FloatArray, groups: dict[str, list[int]]) -> dict[str, float]:
    """Per-group ratio of the amplitude the Rollout predicts to the one the Plant reaches.

    Parameters
    ----------
    predicted, actual
        Region LFP over the Rollout window, both of shape ``(n_starts, n_knots, n_nodes)``.
    """
    if predicted.shape[1] < 2:  # noqa: PLR2004 -- a one-knot window has no peak-to-peak
        return dict.fromkeys((f"ptp_ratio_{name}" for name in groups), float("nan"))
    predicted_ptp = np.ptp(predicted, axis=1)
    actual_ptp = np.ptp(actual, axis=1)
    return {
        f"ptp_ratio_{name}": float(np.mean(predicted_ptp[:, idx]) / np.mean(actual_ptp[:, idx]))
        for name, idx in groups.items()
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Control Horizon error study."""
    parser = argparse.ArgumentParser(description="Measure deterministic-Predictor drift along the Control Horizon.")
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG, help="Base oracle MPC YAML config.")
    parser.add_argument("--output", type=Path, default=Path("results/horizon_error.csv"), help="Output CSV path.")
    parser.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS), help="Plant seeds to pool over.")
    parser.add_argument("--horizon", type=int, default=50, help="Longest Control Horizon in knots.")
    parser.add_argument("--start-stride", type=int, default=25, help="Knots between Rollout start points.")
    parser.add_argument("--t-end", type=float, default=None, help="Override the run length in seconds.")
    return parser.parse_args()


def main() -> None:
    """Run the horizon error study and write the per-lag table."""
    args = parse_args()
    config = load_config(args.base_config)
    if args.t_end is not None:
        config["t_end"] = args.t_end

    rows = horizon_error_table(config, seeds=tuple(args.seeds), horizon=args.horizon, start_stride=args.start_stride)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.output} ({len(rows)} lags over {len(args.seeds)} seeds)")
    for row in rows:
        if row["horizon_knots"] in {1, 5, 10, 25, args.horizon}:
            print(f"  {row['horizon_ms']:>7.0f} ms  nrmse={row['nrmse']:.3f}  ptp_ratio(ez)={row['ptp_ratio_ez']:.3f}")


if __name__ == "__main__":
    main()
