"""Fit a depth-0 (linear) NARX predictor by least squares instead of gradient descent.

A depth-0 :class:`~neuro.prediction.AutoregressivePredictor` is an *affine* one-step map, so its
optimal one-step fit is the exact linear least-squares solution -- convex, deterministic, and
(unlike the non-convex N-step unrolled gradient training in ``run_nn_predictor.py``) it identifies
the small tES->EEG control response that closed-loop suppression depends on. See the knowledge-base
note ``mpc_performance_and_suppression.md`` sec. 9.

This fits the **one-step** map ``[y_{k-n_y+1..k}, u_{k-n_u+1..k}] -> y_{k+1}`` in model space
(standardized channels, or the PCA latents when ``latent_dim`` is set) and writes the same 3-file
:class:`~neuro.prediction.MLPArtifact` that ``run_nn_predictor.py`` produces, so the artifact is a
drop-in for the MPC controllers. Because rolling out an affine one-step map composes affine maps, the
predictor stays linear over the whole horizon -- fitting the one-step map (rather than a direct
N-step map) is deliberate: the N-step objective dilutes the already-small control response, which is
exactly the failure this script avoids.

Usage
-----
    uv run python scripts/fit_linear_predictor.py \
        --config configs/nn_predictor/meeting_seven/linear_selected.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from neuro.config import NNPredictorConfig, load_config, resolve_artifact_dir, resolve_data_files
from neuro.nn_training import apply_to_blocks, prepare_datasets
from neuro.prediction import AutoregressivePredictor, MLPArtifact
from neuro.transforms import PCAProjection, Pipeline, Standardizer

if TYPE_CHECKING:
    from neuro.types import FloatArray

jax.config.update("jax_enable_x64", val=True)


def _one_step_features(  # noqa: PLR0913
    x: FloatArray,
    y_pipeline: Pipeline,
    u_pipeline: Pipeline,
    n_y: int,
    n_u: int,
    n_channels: int,
    n_controls: int,
) -> FloatArray:
    """Map raw one-step windows to the model-space MLP input ``[y-window, u-window]``.

    ``x`` rows are ``[y_past (n_y*n_channels), u_past (n_u*n_controls), u_future (...)]`` as produced
    by :func:`~neuro.nn_training.prepare_datasets`. The one-step control window feeding ``y_{k+1}`` is
    ``u_{k-n_u+1..k}`` -- the ``u_past`` block with its oldest step dropped and the first future
    control ``u_k`` appended -- matching the window the autoregressive rollout builds on its first
    step. The y-window goes through ``y_pipeline`` (standardize, then optionally project) and the
    u-window through ``u_pipeline`` (standardize).
    """
    y_base = n_y * n_channels
    y_past = x[:, :y_base]
    u_past = x[:, y_base : y_base + n_u * n_controls]
    u_k = x[:, y_base + n_u * n_controls : y_base + n_u * n_controls + n_controls]
    u_window = np.concatenate([u_past[:, n_controls:], u_k], axis=1)

    y_past_m = apply_to_blocks(y_past, y_pipeline, n_channels)
    u_window_m = apply_to_blocks(u_window, u_pipeline, n_controls)
    return np.concatenate([y_past_m, u_window_m], axis=1)


def fit_and_save_linear_predictor(config: NNPredictorConfig, data_files: list[str], artifact_dir: Path) -> float:
    """Least-squares-fit a depth-0 predictor for one config and save the artifact.

    Mirrors :func:`~neuro.nn_training.train_and_save_predictor` (same datasets, chronological split,
    fitted transforms and artifact format) but replaces gradient training with a single closed-form
    least-squares solve of the one-step map. Returns the validation one-step MSE in raw EEG units.
    """
    sim_cfg, model_cfg, train_cfg = config.simulation, config.model, config.training
    if model_cfg.depth != 0:
        msg = (
            f"fit_linear_predictor requires a depth-0 (linear) model; got depth {model_cfg.depth}. "
            "Use scripts/run_nn_predictor.py for a nonlinear predictor."
        )
        raise ValueError(msg)

    n_y, n_u, horizon = model_cfg.n_y, model_cfg.n_u, model_cfg.horizon
    downsample = sim_cfg.downsample

    # Raw windows, split chronologically (a random split leaks between overlapping adjacent windows).
    x_full, y_full, n_channels = prepare_datasets(data_files, sim_cfg.n_steps, downsample, n_y, n_u, horizon)
    n_controls = (x_full.shape[1] - n_y * n_channels) // (n_u + horizon)
    split_idx = int(train_cfg.train_split * len(x_full))
    x_train, x_val = x_full[:split_idx], x_full[split_idx:]
    y_train, y_val = y_full[:split_idx], y_full[split_idx:]

    # Fit transforms on the training split only (identical to the gradient pipeline).
    y_past_train = x_train[:, : n_y * n_channels].reshape(-1, n_channels)
    u_past_train = x_train[:, n_y * n_channels : n_y * n_channels + n_u * n_controls].reshape(-1, n_controls)
    y_standardizer = Standardizer.fit(y_past_train, kind=train_cfg.scaler, global_scaling=train_cfg.global_scaling)
    u_standardizer = Standardizer.fit(u_past_train, kind=train_cfg.scaler, global_scaling=train_cfg.global_scaling)
    pca = (
        None
        if model_cfg.latent_dim is None
        else PCAProjection.fit(y_standardizer.transform(y_past_train), model_cfg.latent_dim)
    )
    y_pipeline = Pipeline((y_standardizer, pca)) if pca is not None else Pipeline((y_standardizer,))
    u_pipeline = Pipeline((u_standardizer,))
    latent = pca.basis.shape[0] if pca is not None else n_channels

    # One-step inputs and target (y_{k+1}) in model space, then the closed-form affine fit.
    features = _one_step_features(x_train, y_pipeline, u_pipeline, n_y, n_u, n_channels, n_controls)
    target = apply_to_blocks(y_train[:, :n_channels], y_pipeline, n_channels)
    features_aug = np.hstack([features, np.ones((len(features), 1))])
    weight_bias, *_ = np.linalg.lstsq(features_aug, target, rcond=None)
    weight, bias = weight_bias[:-1].T, weight_bias[-1]  # weight (latent, in_size), bias (latent,)

    in_size = n_y * latent + n_u * n_controls
    mlp = eqx.nn.MLP(
        in_size=in_size, out_size=latent, width_size=latent, depth=0, activation=jax.nn.relu, key=jax.random.PRNGKey(0)
    )
    mlp = eqx.tree_at(lambda m: (m.layers[0].weight, m.layers[0].bias), mlp, (jnp.asarray(weight), jnp.asarray(bias)))
    model = AutoregressivePredictor(
        model=mlp, n_y=n_y, n_u=n_u, horizon=horizon, n_channels=latent, n_controls=n_controls, activation="relu"
    )
    MLPArtifact(
        model=model,
        dt=float(sim_cfg.dt * downsample),
        downsample=int(downsample),
        y_pipeline=y_pipeline,
        u_pipeline=u_pipeline,
    ).save(artifact_dir / "model.eqx")

    # Validation one-step MSE, decoded back to raw EEG so it is comparable to run_nn_predictor's mse.
    features_val = _one_step_features(x_val, y_pipeline, u_pipeline, n_y, n_u, n_channels, n_controls)
    pred_val_raw = np.asarray(y_pipeline.inverse_transform(features_val @ weight.T + bias))
    mse = float(np.mean((y_val[:, :n_channels] - pred_val_raw) ** 2))
    spectral_radius = _one_step_spectral_radius(weight, n_y, latent)
    (artifact_dir / "training_stats.json").write_text(
        json.dumps(
            {"method": "least_squares", "one_step_val_mse": mse, "rollout_spectral_radius": spectral_radius}, indent=2
        )
    )
    return mse


def _one_step_spectral_radius(weight: FloatArray, n_y: int, latent: int) -> float:
    """Spectral radius of the affine rollout's companion state-transition (rollout stability check).

    The one-step map's dependence on the ``n_y`` past latent-output blocks forms a companion matrix
    (the newest-output row is ``weight``'s y-block; the rest shift the history). A spectral radius
    ``<= 1`` means the zero-control free-run stays bounded; ``> 1`` warns that horizon rollouts (and
    thus the MPC prediction) diverge.
    """
    y_block = weight[:, : n_y * latent]  # (latent, n_y*latent): d y_{k+1} / d [y_{k-n_y+1..k}]
    dim = n_y * latent
    companion = np.zeros((dim, dim))
    companion[:latent, :] = y_block
    companion[latent:, : dim - latent] = np.eye(dim - latent)  # shift older history down
    return float(np.max(np.abs(np.linalg.eigvals(companion))))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the least-squares linear-predictor fitter."""
    parser = argparse.ArgumentParser(description="Least-squares fit of a depth-0 (linear) NARX predictor.")
    parser.add_argument(
        "--config", type=str, default="configs/nn_predictor/nn_predictor_config.yaml", help="Path to config YAML."
    )
    parser.add_argument("--data-path", type=str, help="Override config data path.")
    return parser.parse_args()


def main() -> None:
    """Execute the least-squares linear-predictor fitting for one config."""
    args = parse_args()
    config_path = Path(args.config)
    try:
        config = load_config(config_path)
        data_files = resolve_data_files(config, args.data_path)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        return

    artifact_dir = resolve_artifact_dir(config.artifact, "nn_predictor")
    shutil.copy2(config_path, artifact_dir / config_path.name)

    mse = fit_and_save_linear_predictor(config, data_files, artifact_dir)

    print(f"Least-squares fit done (one-step val MSE {mse:.4f}).")
    print(f"Saved NN predictor artifact -> {artifact_dir / 'model.eqx'}")


if __name__ == "__main__":
    main()
