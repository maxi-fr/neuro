from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from neuro.esn import generate_reservoir
from neuro.esn_training import prepare_training_data
from neuro.metrics import DEFAULT_HOP_S, METRICS
from neuro.predictor.esn_module import ESNModule
from neuro.predictor.evaluation import evaluate_log_energy, evaluate_rollouts
from neuro.predictor.ridge import RidgeTrainer, RidgeTrainingResult
from neuro.provenance import training_provenance

if TYPE_CHECKING:
    from neuro.config import ESNPredictorConfig


def _train_esn(cfg: ESNPredictorConfig, data_files: list[str], *, seed_offset: int = 0) -> RidgeTrainingResult:
    """Train the ESN predictor for one config and return everything the run produced.

    The closed-form fit is the incumbent pipeline in one place: scipy generates the reservoir, the
    torch module carries it, and the Ridge Trainer streams the normal equations over the training
    trajectories and installs ``W_out``. The free-run scores are the ESN's two candidates. The
    ESN enters the unified entry point through ``training.fit: ridge`` only; any other fit fails
    here at build time, before data is loaded or any fit runs.

    Parameters
    ----------
    cfg : ESNPredictorConfig
        Validated ESN configuration with any sweep overrides already applied by the caller.
    data_files : list[str]
        Paths to the ``.npz`` trajectory files, split into train/validation by trajectory.
    seed_offset : int, optional
        Added to ``training.seed``. Defaults to 0.

    Returns
    -------
    RidgeTrainingResult
        The fitted ESN Predictor, the ``rollout_nmse``/``log_energy`` candidates, the free-run
        scores and the held-out trajectories.

    Raises
    ------
    ValueError
        If ``training.fit`` is not ``ridge`` (gradient-descent training of the ESN is not
        implemented), or ``model.noise_sigma > 0``, which the module's
        ``design_normal_equations`` does not implement (it streams the noise-free harvest).
    """
    if cfg.training.fit != "ridge":
        msg = (
            f"the ESN supports only 'training.fit: ridge', got {cfg.training.fit!r}; "
            "gradient-descent training of the ESN is not implemented."
        )
        raise ValueError(msg)
    if cfg.model.noise_sigma > 0:
        msg = (
            f"noise_sigma = {cfg.model.noise_sigma} is not supported through the unified entry "
            "point: ESNModule.design_normal_equations streams the noise-free harvest. Use "
            "noise_sigma: 0."
        )
        raise ValueError(msg)

    seed = cfg.training.seed + seed_offset
    data = prepare_training_data(cfg, data_files)
    n_channels = data.train_trajs[0][1].shape[1]
    w_res, w_in = generate_reservoir(
        reservoir_size=cfg.model.reservoir_size,
        spectral_radius=cfg.model.spectral_radius,
        density=cfg.model.density,
        input_scaling=cfg.model.input_scaling,
        in_dim=data.in_dim,
        seed=seed,
    )
    model = ESNModule(
        w_res=w_res,
        w_in=w_in,
        w_out=np.zeros((n_channels, cfg.model.reservoir_size + 1), dtype=np.float64),
        leak_rate=cfg.model.leak_rate,
        priming_steps=cfg.model.priming_steps,
        horizon=cfg.model.horizon,
        dt=cfg.simulation.dt * cfg.simulation.downsample,
        y_std=data.y_std,
        u_std=data.u_std,
    )
    RidgeTrainer(ridge_lambda=cfg.model.ridge_lambda).fit(model, data.train_trajs)

    fs = 1.0 / (cfg.simulation.dt * cfg.simulation.downsample)
    eval_steps = cfg.model.horizon
    # The energy course follows the metrics layer's own eeg_ms convention rather than a knob of its
    # own, clamped where the evaluation horizon is too short to hold one window.
    energy_window = min(max(1, round(METRICS["eeg_ms"].window_s * fs)), eval_steps)
    energy_hop = max(1, round(DEFAULT_HOP_S * fs))
    rollout = evaluate_rollouts(model, data.val_trajs, eval_steps)
    log_energy = evaluate_log_energy(
        model, data.val_trajs, eval_steps, window_steps=energy_window, hop_steps=energy_hop
    )
    model.provenance = training_provenance(data_files, cfg.simulation.cutoff_hz)
    model.downsample = cfg.simulation.downsample
    return RidgeTrainingResult(
        predictor=model,
        candidates={
            "rollout_nmse": rollout.pooled,
            "log_energy": log_energy.pooled,
        },
        rollout=rollout,
        log_energy=log_energy,
        val_trajs=data.val_trajs,
    )
