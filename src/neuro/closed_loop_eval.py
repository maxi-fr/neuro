from __future__ import annotations

import tempfile
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from simulate.config import load_config as load_sim_config
from simulate.simulation import Simulation

from neuro.jansen_rit import lfp
from neuro.seizure import SPREAD_HOP_S, SPREAD_WINDOW_S, SpreadProfile

if TYPE_CHECKING:
    from neuro.config import ClosedLoopEvalConfig


def _spread_profile_from_logs(logs: list[dict[str, Any]], dt: float, threshold: float) -> SpreadProfile:
    """Build the PTP envelope of a logged closed-loop run, one window at a time."""
    window = round(SPREAD_WINDOW_S / dt)
    hop = round(SPREAD_HOP_S / dt)
    if len(logs) < window:
        msg = f"Run is shorter than the {SPREAD_WINDOW_S} s spread window ({len(logs)} of {window} samples)."
        raise ValueError(msg)

    ptp_cols: list[np.ndarray] = []
    times: list[float] = []
    for start in range(0, len(logs) - window + 1, hop):
        # Stack one window at a time: the full state trajectory stays on disk (mmap-backed).
        y = lfp(np.stack([entry["x"] for entry in logs[start : start + window]], axis=-1))
        ptp_cols.append(np.ptp(y, axis=1))
        times.append((start + window) * dt - SPREAD_WINDOW_S / 2.0)

    return SpreadProfile.from_ptp(np.asarray(times, dtype=np.float64), np.stack(ptp_cols, axis=1), threshold=threshold)


def evaluate_closed_loop_suppression(trial_dir: Path, eval_cfg: ClosedLoopEvalConfig) -> tuple[float, dict[str, float]]:
    """Run closed-loop simulations across ``eval_cfg.seeds`` and score seizure suppression proficiency.

    Returns the Optuna score to minimize -- negative suppressed-seed count, with the mean
    stimulation amplitude (in [0, 1]) as a tie-break -- and the summary stats.
    """
    base_sim_path = Path(eval_cfg.simulation_config)
    if not base_sim_path.exists():
        msg = f"Closed-loop simulation config not found: {base_sim_path}"
        raise FileNotFoundError(msg)
    base_sim_dict = load_sim_config(base_sim_path)

    model_artifact_path = (trial_dir / "model.eqx").resolve()
    if not model_artifact_path.exists():
        msg = f"Trained predictor artifact not found: {model_artifact_path}"
        raise FileNotFoundError(msg)

    suppressed_count = 0
    amplitudes: list[float] = []
    seizing_counts: list[int] = []

    for seed in eval_cfg.seeds:
        sim_dict = deepcopy(base_sim_dict)
        sim_dict["t_end"] = eval_cfg.t_end
        sim_dict["dynamics"]["seed"] = seed
        sim_dict["controller"]["artifact"] = str(model_artifact_path)

        sim = Simulation.from_config(sim_dict)
        # Log to a scratch directory so the full state trajectory never becomes resident, and
        # is discarded with the directory rather than kept around.
        with tempfile.TemporaryDirectory(prefix="closed_loop_eval_") as log_dir:
            sim.run(output_dir=log_dir, use_mmap=True)

            if sim.logger is None:
                msg = "Simulation logger is missing after run."
                raise RuntimeError(msg)

            u_max = getattr(sim.controller, "u_max", None)
            if u_max is None:
                msg = (
                    f"Closed-loop evaluation needs a controller exposing 'u_max', got {type(sim.controller).__name__}."
                )
                raise TypeError(msg)

            logs = sim.logger.core_logs
            us = np.stack([np.atleast_1d(np.asarray(entry["u"], dtype=np.float64)) for entry in logs])
            amplitudes.append(float(np.mean(np.abs(us) / np.asarray(u_max, dtype=np.float64))))

            profile = _spread_profile_from_logs(logs, sim.dt, eval_cfg.seizure_ptp_mv)

        n_seizing = int(profile.n_seizing()[-1])
        seizing_counts.append(n_seizing)
        if n_seizing <= eval_cfg.max_seizing_regions:
            suppressed_count += 1

    mean_amplitude = float(np.mean(amplitudes))
    score = -float(suppressed_count) + mean_amplitude

    summary = {
        "score": score,
        "suppressed_seeds": float(suppressed_count),
        "total_seeds": float(len(eval_cfg.seeds)),
        "mean_amplitude": mean_amplitude,
        "mean_seizing_regions": float(np.mean(seizing_counts)),
    }
    return score, summary
