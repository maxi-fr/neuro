from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from simulate.config import load_config as load_sim_config

from neuro.comparison import control_bound, score_run
from neuro.validation import validate_simulation_config

if TYPE_CHECKING:
    from neuro.config import ClosedLoopEvalConfig


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

    The per-run reduction is :func:`neuro.comparison.score_run`, the one the comparison tables use,
    so the sweep cannot optimize a Seizure Burden that means something else than the reported one.
    ``score_run`` also forces the region-LFP log the spread metrics read, so an eval config need not
    ask for it.

    Raises
    ------
    FileNotFoundError
        If the simulation config or the trial's trained checkpoint is missing.
    """
    base_sim_path = Path(eval_cfg.simulation_config)
    if not base_sim_path.exists():
        msg = f"Closed-loop simulation config not found: {base_sim_path}"
        raise FileNotFoundError(msg)
    base_sim_dict = load_sim_config(base_sim_path)

    model_checkpoint_path = (trial_dir / "model.npz").resolve()
    if not model_checkpoint_path.exists():
        msg = f"Trained predictor checkpoint not found: {model_checkpoint_path}"
        raise FileNotFoundError(msg)

    base_sim_dict["t_end"] = eval_cfg.t_end
    base_sim_dict["controller"]["problem"]["artifact"] = str(model_checkpoint_path)
    # The seeds differ only in the plant realisation, so the wiring is the same for all of them.
    validate_simulation_config(base_sim_dict)

    # Same reader the comparison grid normalises effort by, so a sweep and a table cannot disagree
    # about what an arm's Control Budget was.
    u_max = control_bound(base_sim_dict) or 1.0

    runs = []
    for seed in eval_cfg.seeds:
        sim_dict = deepcopy(base_sim_dict)
        sim_dict["dynamics"]["seed"] = seed
        runs.append(score_run(sim_dict, u_max, threshold=eval_cfg.seizure_ptp_mv))

    def mean(key: str) -> float:
        return float(np.mean([run[key] for run in runs]))

    mean_amplitude = mean("mean_amplitude")
    mean_burden = mean("seizure_burden")
    suppressed = sum(run["n_seizing_final"] <= eval_cfg.max_seizing_regions for run in runs)

    summary = {
        "score": mean_burden + eval_cfg.amplitude_weight * mean_amplitude,
        "seizure_burden": mean_burden,
        "suppressed_seeds": float(suppressed),
        "total_seeds": float(len(eval_cfg.seeds)),
        "mean_amplitude": mean_amplitude,
        "mean_delivered_charge": mean("delivered_charge"),
        "mean_seizing_regions": mean("n_seizing_final"),
    }
    return summary["score"], summary
