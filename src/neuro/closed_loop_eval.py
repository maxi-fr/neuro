from __future__ import annotations

import tempfile
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from simulate.config import load_config as load_sim_config
from simulate.simulation import Simulation

from neuro.jansen_rit import lfp
from neuro.seizure import SPREAD_HOP_S, SPREAD_WINDOW_S, SpreadProfile, spread_profile_from_lfp
from neuro.validation import validate_simulation_config

if TYPE_CHECKING:
    from neuro.config import ClosedLoopEvalConfig


def _spread_profile_from_trajectory(x_traj: np.ndarray, dt: float, threshold: float) -> SpreadProfile:
    """Build the PTP envelope of a logged closed-loop run, one window at a time."""
    n_samples = x_traj.shape[0]
    window = round(SPREAD_WINDOW_S / dt)
    hop = round(SPREAD_HOP_S / dt)
    if n_samples < window:
        msg = f"Run is shorter than the {SPREAD_WINDOW_S} s spread window ({n_samples} of {window} samples)."
        raise ValueError(msg)

    ptp_cols: list[np.ndarray] = []
    times: list[float] = []
    for start in range(0, n_samples - window + 1, hop):
        # Move step axis to end -> (6, N, window) so lfp returns (N, window)
        x_win = np.moveaxis(x_traj[start : start + window], 0, -1)
        y = lfp(x_win)
        ptp_cols.append(np.ptp(y, axis=1))
        times.append((start + window) * dt - SPREAD_WINDOW_S / 2.0)

    return SpreadProfile.from_ptp(np.asarray(times, dtype=np.float64), np.stack(ptp_cols, axis=1), threshold=threshold)


def _spread_profile_from_logs(logs: list[dict[str, Any]], dt: float, threshold: float) -> SpreadProfile:
    """Build the PTP envelope of a logged closed-loop run from a list of per-step dict logs."""
    if not logs:
        msg = f"Run is shorter than the {SPREAD_WINDOW_S} s spread window (0 samples)."
        raise ValueError(msg)
    x_traj = np.stack([entry["x"] for entry in logs], axis=0)
    return _spread_profile_from_trajectory(x_traj, dt, threshold)


def seizure_burden(profile: SpreadProfile) -> float:
    """Time-mean fraction of regions seizing over a run -- the continuous suppression score.

    The closed-loop reading of :func:`neuro.metrics.seizure_state`, and the run-averaged
    counterpart of ``SpreadProfile.n_seizing()[-1]``. Continuous in ``[0, 1]``, so it separates
    controllers the terminal count ties, and lower for a seizure ended sooner rather than one
    merely ended by the last window.
    """
    return float(np.mean(profile.n_seizing() / profile.ptp.shape[0]))


def evaluate_closed_loop_suppression(trial_dir: Path, eval_cfg: ClosedLoopEvalConfig) -> tuple[float, dict[str, float]]:
    """Run closed-loop simulations across ``eval_cfg.seeds`` and score seizure suppression proficiency.

    Returns the Optuna score to minimize and the summary stats. The score is the **seizure
    burden**: the fraction of regions seizing, averaged over the whole run and over the seeds,
    plus ``eval_cfg.amplitude_weight`` times the mean stimulation amplitude.

    Burden rather than a suppressed-seed count because the count is integer-valued over a handful
    of seeds, so it hands the sampler a few wide plateaus and no gradient between them. Averaging
    over the run rather than reading the terminal window also rewards suppressing early, which the
    terminal count cannot see. ``max_seizing_regions`` still defines the reported
    ``suppressed_seeds`` diagnostic; it no longer drives the score.
    """
    base_sim_path = Path(eval_cfg.simulation_config)
    if not base_sim_path.exists():
        msg = f"Closed-loop simulation config not found: {base_sim_path}"
        raise FileNotFoundError(msg)
    base_sim_dict = load_sim_config(base_sim_path)

    model_artifact_path = (trial_dir / "model.npz").resolve()
    if not model_artifact_path.exists():
        msg = f"Trained predictor artifact not found: {model_artifact_path}"
        raise FileNotFoundError(msg)

    base_sim_dict["t_end"] = eval_cfg.t_end
    base_sim_dict["controller"]["artifact"] = str(model_artifact_path)
    if base_sim_dict["dynamics"].get("log", "none") == "none":
        base_sim_dict["dynamics"]["log"] = "lfp"
    # The seeds differ only in the plant realisation, so the wiring is the same for all of them.
    validate_simulation_config(base_sim_dict)

    suppressed_count = 0
    amplitudes: list[float] = []
    seizing_counts: list[int] = []
    burdens: list[float] = []

    for seed in eval_cfg.seeds:
        sim_dict = deepcopy(base_sim_dict)
        sim_dict["dynamics"]["seed"] = seed

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

            us = sim.logger.signal("controller", "u")
            amplitudes.append(float(np.mean(np.abs(us) / np.asarray(u_max, dtype=np.float64))))

            # One of the two region-space logs is always present: the config is coerced to "lfp"
            # above unless it already asked for "state".
            if ("dynamics", "lfp") in set(sim.logger.signals()):
                y_reg = sim.logger.signal("dynamics", "lfp")
                profile = spread_profile_from_lfp(y_reg.T, sim.dt, threshold=eval_cfg.seizure_ptp_mv)
            else:
                x_traj = sim.logger.signal("dynamics", "x")
                profile = _spread_profile_from_trajectory(x_traj, sim.dt, eval_cfg.seizure_ptp_mv)

        burdens.append(seizure_burden(profile))
        n_seizing_final = int(profile.n_seizing()[-1])
        seizing_counts.append(n_seizing_final)
        if n_seizing_final <= eval_cfg.max_seizing_regions:
            suppressed_count += 1

    mean_amplitude = float(np.mean(amplitudes))
    mean_burden = float(np.mean(burdens))
    score = mean_burden + eval_cfg.amplitude_weight * mean_amplitude

    summary = {
        "score": score,
        "seizure_burden": mean_burden,
        "suppressed_seeds": float(suppressed_count),
        "total_seeds": float(len(eval_cfg.seeds)),
        "mean_amplitude": mean_amplitude,
        "mean_seizing_regions": float(np.mean(seizing_counts)),
    }
    return score, summary
