"""Core training and evaluation functions for the JAX-based Neural Network predictor.

This module implements an MLP using Equinox and Optax to predict a horizon of N steps
of EEG data from past EEG and past/future stimulation inputs. It supports training on
multiple trajectories.
"""

import json
from collections.abc import Callable, Iterator
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax.scipy.signal import welch
from tqdm import tqdm

from neuro.config import NNPredictorConfig
from neuro.prediction import AutoregressivePredictor, MLPArtifact, get_activation
from neuro.transforms import PCAProjection, Pipeline, Standardizer
from neuro.types import FloatArray
from utils.plotting import plot_multistep_predictions

jax.config.update("jax_enable_x64", val=True)


def load_trajectory(data_file: str, n_steps: int | None, downsample: int) -> tuple[FloatArray, FloatArray]:
    """Load a single simulation trajectory.

    Parameters
    ----------
    data_file : str
        Path to the `.npz` data file containing the trajectory.
    n_steps : int | None
        The total number of time steps to load, or ``None`` to load the entire trajectory.
    downsample : int
        The downsampling factor to apply.

    Returns
    -------
    u_data : FloatArray
        The stimulation input trajectory, shape ``(T, n_controls)``.
    y_data : FloatArray
        The measured output (EEG) trajectory, shape ``(T, n_channels)``.
    """
    with np.load(data_file) as data:
        max_idx = None if n_steps is None else n_steps * downsample
        try:
            y_data = data["y_mea"][:max_idx:downsample]
            u_data = data["u"][:max_idx:downsample]
        except:  # noqa: E722 # only because old experiment data used old keys
            y_data = data["universal_y_mea"][:max_idx:downsample]
            u_data = data["universal_u"][:max_idx:downsample]
    return u_data, y_data


def extract_windows_flattened(data: FloatArray, window_size: int) -> FloatArray:
    """Extract sliding windows from a 2D array and flatten the time dimension.

    Parameters
    ----------
    data : FloatArray
        Input array of shape (T, C).
    window_size : int
        Size of the sliding window.

    Returns
    -------
    FloatArray
        Flattened sliding windows of shape (T - window_size + 1, window_size * C).
    """
    _, channels = data.shape
    view = np.lib.stride_tricks.sliding_window_view(data, (window_size, channels))
    return view.reshape(-1, window_size * channels)


def apply_to_blocks(block_flat: FloatArray, pipeline: Pipeline, in_channels: int) -> FloatArray:
    """Apply a :class:`~neuro.transforms.Pipeline` to a flattened multi-step block.

    ``block_flat`` is ``(samples, steps * in_channels)`` with the per-timestep channel vectors
    laid out contiguously (as produced by :func:`extract_windows_flattened`). The pipeline is a
    per-timestep-vector map, so the block is reshaped to ``(-1, in_channels)``, transformed, and
    reshaped back -- its width becomes ``steps * out_channels`` (which differs from the input when
    a PCA step reduces the channel dimension).

    Parameters
    ----------
    block_flat : FloatArray
        Flattened block of shape ``(samples, steps * in_channels)``.
    pipeline : Pipeline
        The fitted transform to apply along the channel axis.
    in_channels : int
        Number of raw channels per timestep in ``block_flat``.

    Returns
    -------
    FloatArray
        Transformed block of shape ``(samples, steps * out_channels)``.
    """
    samples = block_flat.shape[0]
    return np.asarray(pipeline.transform(block_flat.reshape(-1, in_channels))).reshape(samples, -1)


def reshape_to_trajectory(data_flat: jax.Array | FloatArray, horizon: int, channels: int) -> jax.Array | FloatArray:
    """Reshape a flattened trajectory back to (batch, horizon, channels).

    Parameters
    ----------
    data_flat : Any
        Flattened sequence of shape (batch, horizon * channels).
    horizon : int
        Number of time steps.
    channels : int
        Number of channels per time step.

    Returns
    -------
    Any
        Reshaped sequence of shape (batch, horizon, channels).
    """
    return data_flat.reshape(-1, horizon, channels)


def build_dataset_for_trajectory(
    u_data: FloatArray, y_data: FloatArray, n_y: int, n_u: int, N: int
) -> tuple[FloatArray, FloatArray]:
    """Build the input/output pairs for the multi-step predictor.

    Parameters
    ----------
    u_data : FloatArray
        The stimulation input trajectory of shape (T, n_controls).
    y_data : FloatArray
        The measured output (EEG) trajectory of shape (T, n_channels).
    n_y : int
        Number of past output steps to include in the input feature.
    n_u : int
        Number of past input steps to include in the input feature.
    N : int
        Prediction horizon (number of future steps to predict).

    Returns
    -------
    X : FloatArray
        Input features array of shape (samples, n_y * n_channels + n_u * n_controls + N * n_controls).
    Y : FloatArray
        Target labels array of shape (samples, N * n_channels).
    """
    T_src, _ = y_data.shape

    start_idx = max(n_y - 1, n_u)
    end_idx = T_src - N
    k = np.arange(start_idx, end_idx)

    y_view = extract_windows_flattened(y_data, n_y)
    u_past_view = extract_windows_flattened(u_data, n_u)
    u_future_view = extract_windows_flattened(u_data, N)

    X = np.concatenate([y_view[k - n_y + 1], u_past_view[k - n_u], u_future_view[k]], axis=1)

    y_fut_view = extract_windows_flattened(y_data, N)
    Y = y_fut_view[k + 1]

    return X, Y


def get_dataloaders(X: FloatArray, Y: FloatArray, batch_size: int = 128) -> Iterator[tuple[jax.Array, jax.Array]]:
    """Generate batches of data.

    Parameters
    ----------
    X : FloatArray
        Input features array, shape ``(samples, n_features)``.
    Y : FloatArray
        Target labels array, shape ``(samples, n_targets)``.
    batch_size : int, optional
        Number of samples per batch. Defaults to 128.

    Yields
    ------
    batch_x : jax.Array
        A batch of input features, shape ``(batch_size, n_features)``.
    batch_y : jax.Array
        A batch of target labels, shape ``(batch_size, n_targets)``.
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    rng = np.random.default_rng()
    rng.shuffle(indices)
    for start_idx in range(0, n_samples, batch_size):
        batch_idx = indices[start_idx : start_idx + batch_size]
        yield jnp.array(X[batch_idx]), jnp.array(Y[batch_idx])


def prepare_datasets(  # noqa: PLR0913
    data_files: list[str],
    n_steps_cfg: int | None,
    downsample: int,
    n_y: int,
    n_u: int,
    horizon: int,
) -> tuple[FloatArray, FloatArray, int]:
    """Load data and build the raw (unscaled) dataset across multiple trajectories.

    Windows are built in raw EEG/control units; the standardizer and optional PCA projection
    are fitted on the training split and applied downstream (see
    :func:`train_and_save_predictor`), so no transform is applied here.

    Parameters
    ----------
    data_files : list[str]
        List of paths to data files.
    n_steps_cfg : int | None
        Number of steps to load per trajectory, or ``None`` to load the entire trajectory.
    downsample : int
        Downsampling factor.
    n_y : int
        Number of past output steps to include.
    n_u : int
        Number of past input steps to include.
    horizon : int
        Prediction horizon.

    Returns
    -------
    X_full : FloatArray
        Raw input features array, shape ``(total_samples, n_features)``.
    Y_full : FloatArray
        Raw target labels array, shape ``(total_samples, horizon * n_channels)``.
    n_channels : int
        Number of raw EEG output channels.
    """
    all_X, all_Y = [], []
    n_channels = 1
    for df in data_files:
        u, y = load_trajectory(df, n_steps_cfg, downsample)
        n_channels = y.shape[1]
        X_traj, Y_traj = build_dataset_for_trajectory(u, y, n_y, n_u, horizon)
        all_X.append(X_traj)
        all_Y.append(Y_traj)

    X_full = np.concatenate(all_X, axis=0)
    Y_full = np.concatenate(all_Y, axis=0)
    return X_full, Y_full, n_channels


def create_model(  # noqa: PLR0913
    in_size: int,
    out_size: int,
    hidden_size: int,
    depth: int,
    key: jax.Array,
    n_y: int,
    n_u: int,
    horizon: int,
    n_channels: int,
    n_controls: int,
    activation: str = "relu",
) -> AutoregressivePredictor:
    """Create the Autoregressive MLP model.

    Parameters
    ----------
    in_size : int
        Size of the input features.
    out_size : int
        Size of the output predictions.
    hidden_size : int
        Number of neurons in the hidden layers.
    depth : int
        Number of hidden layers.
    key : jax.Array
        JAX PRNG key for initialization.
    n_y : int
        Number of past output steps.
    n_u : int
        Number of past input steps.
    horizon : int
        Prediction horizon.
    n_channels : int
        Number of output channels.
    n_controls : int
        Number of input control channels.

    Returns
    -------
    eqx.Module
        The instantiated AutoregressivePredictor model.
    """
    mlp = eqx.nn.MLP(
        in_size=in_size,
        out_size=out_size,
        width_size=hidden_size,
        depth=depth,
        activation=get_activation(activation),
        key=key,
    )
    return AutoregressivePredictor(
        model=mlp,
        n_y=n_y,
        n_u=n_u,
        horizon=horizon,
        n_channels=n_channels,
        n_controls=n_controls,
        activation=activation,
    )


def transform_features(  # noqa: PLR0913
    X: FloatArray,
    y_pipeline: Pipeline,
    u_pipeline: Pipeline,
    n_y: int,
    n_channels: int,
    n_controls: int,
) -> FloatArray:
    """Map a raw feature matrix into model space.

    The past-EEG block is pushed through ``y_pipeline`` (standardize, then optionally project),
    and the past/future control blocks through ``u_pipeline`` (standardize). The past-EEG block's
    width shrinks from ``n_y * n_channels`` to ``n_y * k`` when the y-pipeline projects.

    Parameters
    ----------
    X : FloatArray
        Raw input features, shape ``(samples, n_y * n_channels + (n_u + horizon) * n_controls)``.
    y_pipeline, u_pipeline : Pipeline
        Fitted transforms for the EEG and control blocks.
    n_y : int
        History length of the EEG block.
    n_channels, n_controls : int
        Raw EEG and control channel counts.

    Returns
    -------
    FloatArray
        Model-space features, shape ``(samples, n_y * k + (n_u + horizon) * n_controls)``.
    """
    y_past = X[:, : n_y * n_channels]
    u_blocks = X[:, n_y * n_channels :]

    y_past_m = apply_to_blocks(y_past, y_pipeline, n_channels)
    u_blocks_m = apply_to_blocks(u_blocks, u_pipeline, n_controls)
    return np.concatenate([y_past_m, u_blocks_m], axis=1)


@eqx.filter_jit
def predict_batch(m: eqx.Module, x: jax.Array) -> jax.Array:
    """Run model prediction on a batch.

    Parameters
    ----------
    m : eqx.Module
        The Equinox model.
    x : jax.Array
        Batch of input features, shape ``(batch_size, n_features)``.

    Returns
    -------
    jax.Array
        Batch of predictions, shape ``(batch_size, n_targets)``.
    """
    return jax.vmap(m)(x)


def _masked_fc(traj: jax.Array, step_mask: jax.Array, eps: float) -> jax.Array:
    """Weighted channel-by-channel correlation (FC) over the first-``L`` rollout steps.

    ``traj`` is ``(batch, horizon, channels)`` and ``step_mask`` a ``(horizon,)`` prefix of ones
    selecting the ``L`` supervised steps, so FC grows with the horizon curriculum like the MSE term.
    The per-timestep weights are broadcast over the pooled ``(batch * horizon, channels)`` samples;
    with an all-ones mask this reduces to the standard Pearson correlation with a zeroed diagonal.

    Parameters
    ----------
    traj : jax.Array
        Channel-space trajectory, shape ``(batch, horizon, channels)``.
    step_mask : jax.Array
        Per-step curriculum mask, shape ``(horizon,)`` (a prefix of ``L`` ones).
    eps : float
        Small constant guarding the weight sum and the per-channel standard deviations.

    Returns
    -------
    jax.Array
        The ``(channels, channels)`` functional-connectivity matrix with a zeroed diagonal.
    """
    batch, horizon, channels = traj.shape
    weights = jnp.broadcast_to(step_mask[None, :], (batch, horizon)).reshape(-1)  # (batch * horizon,)
    pooled = traj.reshape(-1, channels)  # (batch * horizon, channels)
    weight_sum = jnp.sum(weights) + eps
    mean = jnp.sum(weights[:, None] * pooled, axis=0) / weight_sum
    centered = pooled - mean
    cov = (centered * weights[:, None]).T @ centered / weight_sum
    std = jnp.sqrt(jnp.diag(cov) + eps)
    corr = cov / (std[:, None] * std[None, :])
    return corr - jnp.diag(jnp.diag(corr))


def compute_loss(  # noqa: PLR0913
    m: eqx.Module,
    x: jax.Array,
    y: jax.Array,
    step_mask: jax.Array,
    n_channels: int,
    w_psd: jax.Array,
    w_fc: jax.Array,
    decode_basis: jax.Array | None = None,
    decode_mean: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compute the horizon-curriculum MSE loss for a batch along with PSD and FC losses.

    All terms are evaluated in standardized-channel EEG space. The model rolls out in model
    space (the ``k``-dimensional latent components when the y-pipeline projects); its output is
    decoded with the fixed, differentiable inverse PCA (``pred @ decode_basis + decode_mean``)
    before every loss term, so PSD/FC are computed over real EEG channels rather than the
    decorrelated latent components. With no projection ``decode_basis`` is ``None`` and the
    model already predicts channels (identity decode).

    ``step_mask`` is the horizon-length curriculum: a ``(horizon,)`` prefix of ones selecting the
    first ``L`` trusted rollout steps (``L=1`` is teacher forcing, ``L=horizon`` is the full
    free-running rollout). The MSE and FC are scored over the first ``L`` steps (they grow with
    ``L``); the PSD stays a batch-of-snippets estimate over the full horizon window but is gated on
    only once ``L == horizon`` (its last mask entry).

    Parameters
    ----------
    m : eqx.Module
        The Equinox model.
    x : jax.Array
        Batch of input features, shape ``(batch_size, n_features)``.
    y : jax.Array
        Batch of standardized-channel target labels, shape ``(batch_size, horizon * n_channels)``.
    step_mask : jax.Array
        Horizon-length curriculum mask, shape ``(horizon,)`` (a prefix of ``L`` ones).
    n_channels : int
        Number of EEG channels ``C`` (the target/loss dimension).
    w_psd : jax.Array
        Weight for the PSD loss.
    w_fc : jax.Array
        Weight for the FC loss.
    decode_basis : jax.Array | None
        Fixed PCA basis ``(k, C)`` mapping model-space predictions to channel space, or ``None``
        when the model already predicts channels.
    decode_mean : jax.Array | None
        Fixed PCA per-channel mean ``(C,)`` paired with ``decode_basis``.

    Returns
    -------
    tuple[jax.Array, dict[str, jax.Array]]
        The combined scalar loss and a dict of the raw ``mse``, ``psd`` and ``fc``
        loss components (before weighting) for logging.
    """
    pred_model = predict_batch(m, x)  # (batch, horizon * k)

    horizon = y.shape[1] // n_channels
    pred_traj_model = pred_model.reshape(pred_model.shape[0], horizon, -1)  # (batch, horizon, k)
    pred_traj = (
        pred_traj_model @ decode_basis + decode_mean if decode_basis is not None else pred_traj_model
    )  # (batch, horizon, C)
    true_traj = y.reshape(y.shape[0], horizon, n_channels)  # (batch, horizon, C)

    # Horizon-length curriculum: score the MSE over the first-L rollout steps selected by step_mask.
    per_step_se = jnp.mean((pred_traj - true_traj) ** 2, axis=(0, 2))  # (horizon,)
    mse_loss = jnp.sum(step_mask * per_step_se) / jnp.sum(step_mask)

    eps = 1e-8

    if horizon > 1:
        # FC grows with L (masked like the MSE): correlation over the first-L steps, batch-pooled.
        pred_fc = _masked_fc(pred_traj, step_mask, eps)
        true_fc = _masked_fc(true_traj, step_mask, eps)
        loss_fc = jnp.mean((pred_fc - true_fc) ** 2)

        # PSD is a batch-of-snippets estimate over the full horizon window, active only once the
        # rollout is trusted end-to-end (L == horizon), i.e. when the last mask entry is on.
        pred_pool = jnp.transpose(pred_traj, (2, 0, 1)).reshape(n_channels, -1)
        true_pool = jnp.transpose(true_traj, (2, 0, 1)).reshape(n_channels, -1)
        _, pred_psd = welch(pred_pool, nperseg=horizon, noverlap=0, axis=-1)
        _, true_psd = welch(true_pool, nperseg=horizon, noverlap=0, axis=-1)
        psd_gate = step_mask[-1]  # 1.0 only when L == horizon
        loss_psd = psd_gate * jnp.mean((jnp.log(pred_psd + eps) - jnp.log(true_psd + eps)) ** 2)
    else:
        loss_psd = jnp.array(0.0)
        loss_fc = jnp.array(0.0)

    total = mse_loss + w_psd * loss_psd + w_fc * loss_fc

    aux = {"mse": mse_loss, "psd": loss_psd, "fc": loss_fc}
    return total, aux


@eqx.filter_jit
def step(  # noqa: PLR0913
    m: eqx.Module,
    opt_s: optax.OptState,
    x: jax.Array,
    y: jax.Array,
    step_mask: jax.Array,
    n_channels: int,
    optimizer: optax.GradientTransformation,
    w_psd: jax.Array,
    w_fc: jax.Array,
    decode_basis: jax.Array | None = None,
    decode_mean: jax.Array | None = None,
) -> tuple[eqx.Module, optax.OptState, jax.Array, dict[str, jax.Array]]:
    """Perform one optimizer step.

    Parameters
    ----------
    m : eqx.Module
        The Equinox model.
    opt_s : optax.OptState
        The current optimizer state.
    x : jax.Array
        Batch of input features, shape ``(batch_size, n_features)``.
    y : jax.Array
        Batch of target labels, shape ``(batch_size, n_targets)``.
    step_mask : jax.Array
        Horizon-length curriculum mask, shape ``(horizon,)``. Fixed shape, so this ``filter_jit``
        does not recompile as the trusted rollout length grows.
    n_channels : int
        Number of EEG channels ``C``.
    optimizer : optax.GradientTransformation
        The Optax optimizer.
    w_psd : jax.Array
        Weight for the PSD loss.
    w_fc : jax.Array
        Weight for the FC loss.
    decode_basis, decode_mean : jax.Array | None
        Fixed PCA decode ``(k, C)`` / ``(C,)`` mapping model space to channel space (see
        :func:`compute_loss`), or ``None`` without a projection.

    Returns
    -------
    new_m : eqx.Module
        The updated Equinox model.
    new_opt_s : optax.OptState
        The updated optimizer state.
    loss_val : jax.Array
        The scalar loss value for the batch.
    aux : dict[str, jax.Array]
        Raw ``mse``, ``psd`` and ``fc`` loss components for logging.
    """
    (loss_val, aux), grads = eqx.filter_value_and_grad(compute_loss, has_aux=True)(
        m, x, y, step_mask, n_channels, w_psd, w_fc, decode_basis, decode_mean
    )
    updates, new_opt_s = optimizer.update(grads, opt_s, m)  # type: ignore
    new_m: eqx.Module = eqx.apply_updates(m, updates)
    return new_m, new_opt_s, loss_val, aux


def curriculum_state(
    epoch: int,
    horizon: int,
    start_epoch: int = 0,
    end_epoch: int = 80,
) -> FloatArray:
    """Return the horizon-length curriculum per-step loss mask for one epoch.

    Returns a prefix of ones of length ``L(epoch)``, where ``L`` grows linearly from 1 to
    ``horizon`` between ``start_epoch`` and ``end_epoch``.

    Parameters
    ----------
    epoch : int
        Current training epoch.
    horizon : int
        Full model rollout horizon.
    start_epoch : int, optional
        Epoch to start growing the horizon, by default 0.
    end_epoch : int, optional
        Epoch to reach the full horizon, by default 80.

    Returns
    -------
    FloatArray
        Binary mask of shape ``(horizon,)``.
    """
    span = max(end_epoch - start_epoch, 1)
    frac = min(max((epoch - start_epoch) / span, 0.0), 1.0)
    length = round(1 + (horizon - 1) * frac)
    length = max(1, min(length, horizon))
    mask = np.zeros(horizon, dtype=np.float64)
    mask[:length] = 1.0
    return mask


def _step_mask_selector(
    horizon: int, curriculum_max_steps: int | None, start_epoch: int, end_epoch: int
) -> Callable[[int], FloatArray]:
    """Return the ``epoch -> loss mask`` map for a curriculum ramping ``1 -> curriculum_max_steps``.

    ``curriculum_max_steps`` is the rollout length the ramp *approaches*, defaulting to the full
    model ``horizon``. Capping it below the horizon keeps the multi-step structure that the
    rollout needs while never entering the long-unroll regime that destabilises this fit; setting
    it to ``1`` degenerates to plain one-step training (a ramp from 1 to 1 never grows). Steps
    beyond the cap stay masked out, so the returned mask is always ``horizon``-wide.
    """
    target = min(curriculum_max_steps or horizon, horizon)

    def select(epoch: int) -> FloatArray:
        """Loss mask for ``epoch``: a prefix of ones, zero-padded out to the full horizon."""
        mask = np.zeros(horizon, dtype=np.float64)
        mask[:target] = curriculum_state(epoch, target, start_epoch=start_epoch, end_epoch=end_epoch)
        return mask

    return select


def _warm_start_linear_model(  # noqa: PLR0913
    model: AutoregressivePredictor,
    X_train_s: FloatArray,
    Y_train_s: FloatArray,
    n_channels: int,
    decode_basis: FloatArray | None,
    decode_mean: FloatArray | None,
) -> eqx.Module:
    """Warm-start a depth-0 linear model using the exact 1-step least squares solution."""
    n_y = model.n_y
    n_u = model.n_u
    n_controls = model.n_controls

    Y_step1 = Y_train_s[:, :n_channels]
    if decode_basis is not None:
        assert decode_mean is not None  # noqa: S101
        Y_latent = (Y_step1 - decode_mean) @ decode_basis.T
    else:
        Y_latent = Y_step1

    latent = Y_latent.shape[1]
    y_len = n_y * latent
    y_past = X_train_s[:, :y_len]
    u_window = X_train_s[:, y_len + n_controls : y_len + n_controls + n_u * n_controls]
    X_1step = np.hstack([y_past, u_window])

    features_aug = np.hstack([X_1step, np.ones((X_1step.shape[0], 1))])
    weight_bias, *_ = np.linalg.lstsq(features_aug, Y_latent, rcond=None)
    weight, bias = weight_bias[:-1].T, weight_bias[-1]

    mlp = eqx.tree_at(
        lambda m: (m.layers[0].weight, m.layers[0].bias),
        model.model,
        (jnp.asarray(weight), jnp.asarray(bias)),
    )
    return eqx.tree_at(lambda m: m.model, model, mlp)


def train_model(  # noqa: PLR0913, PLR0915
    model: eqx.Module,
    X_train_s: FloatArray,
    Y_train_s: FloatArray,
    X_val_s: FloatArray,
    Y_val_s: FloatArray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    n_channels: int,
    curriculum_start_epoch: int = 0,
    curriculum_end_epoch: int = 80,
    curriculum_max_steps: int | None = None,
    w_psd: float = 0.0,
    w_fc: float = 0.0,
    patience: int = 50,
    decode_basis: FloatArray | None = None,
    decode_mean: FloatArray | None = None,
) -> tuple[eqx.Module, list[float], list[float]]:
    """Train the MLP model using Optax under the horizon-length curriculum.

    Parameters
    ----------
    model : eqx.Module
        The Equinox MLP model to train.
    X_train_s : FloatArray
        Scaled training inputs, shape ``(samples_train, n_features)``.
    Y_train_s : FloatArray
        Scaled training targets, shape ``(samples_train, horizon * n_channels)``.
    X_val_s : FloatArray
        Scaled validation inputs, shape ``(samples_val, n_features)``.
    Y_val_s : FloatArray
        Scaled validation targets, shape ``(samples_val, horizon * n_channels)``.
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size.
    learning_rate : float
        Learning rate.
    weight_decay : float
        Weight decay.
    n_channels : int
        Number of output channels.
    curriculum_start_epoch : int
        Epoch at which the trusted rollout length ``L`` starts growing (``L = 1``, pure teacher
        forcing, before it).
    curriculum_max_steps : int | None
        Rollout length the curriculum ramps *toward*; defaults to the full model ``horizon``.
        Capping it below the horizon keeps multi-step training while avoiding the long-unroll
        regime that destabilises this fit, and ``1`` degenerates to plain one-step training.
        Validation stays at the full horizon regardless, so model selection still targets the
        horizon the MPC rolls out over.
    curriculum_end_epoch : int
        Epoch at which ``L`` reaches the model ``horizon`` (the full free-running rollout); it holds
        there afterwards. Growing the rollout length keeps multi-step structure throughout while
        ramping gently, instead of annealing a teacher-forcing weight to a pure N-step loss (see
        :func:`curriculum_state`).
    w_psd : float
        Weight for the PSD loss.
    w_fc : float
        Weight for the FC loss.
    patience : int
        Number of epochs with no improvement after which training will be stopped.
    decode_basis, decode_mean : FloatArray | None
        Fixed PCA decode ``(k, C)`` / ``(C,)`` passed to :func:`compute_loss` so the losses are
        scored in channel space; ``None`` without a projection.

    Returns
    -------
    best_model : eqx.Module
        The trained Equinox model with the best validation loss.
    train_losses : list[float]
        Training loss per epoch.
    val_losses : list[float]
        Validation loss per epoch.
    """
    horizon = Y_train_s.shape[1] // n_channels

    do_lstsq = isinstance(model, AutoregressivePredictor) and model.model.depth == 0

    if do_lstsq:
        model = _warm_start_linear_model(model, X_train_s, Y_train_s, n_channels, decode_basis, decode_mean)

    start_epoch = curriculum_start_epoch if do_lstsq else 0
    steps_per_epoch = (X_train_s.shape[0] + batch_size - 1) // batch_size
    total_steps = steps_per_epoch * (epochs - start_epoch)

    lr_schedule = optax.cosine_decay_schedule(init_value=learning_rate, decay_steps=total_steps, alpha=0.0)

    optimizer = optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    best_val_loss = float("inf")
    best_model = model

    X_val_jnp = jnp.array(X_val_s)
    Y_val_jnp = jnp.array(Y_val_s)

    pbar = tqdm(range(start_epoch, epochs), desc="Training MLP")
    train_losses = []
    val_losses = []

    w_psd_jax = jnp.array(w_psd, dtype=jnp.float32)
    w_fc_jax = jnp.array(w_fc, dtype=jnp.float32)

    val_mask_jax = jnp.ones(horizon)

    decode_basis_jax = None if decode_basis is None else jnp.asarray(decode_basis)
    decode_mean_jax = None if decode_mean is None else jnp.asarray(decode_mean)

    epochs_without_improvement = 0

    step_mask_for = _step_mask_selector(horizon, curriculum_max_steps, curriculum_start_epoch, curriculum_end_epoch)

    for epoch in pbar:
        step_mask = step_mask_for(epoch)
        step_mask_jax = jnp.asarray(step_mask)

        epoch_loss = 0.0
        comp_sums = {"mse": 0.0, "psd": 0.0, "fc": 0.0}
        batches = 0
        for batch_x, batch_y in get_dataloaders(X_train_s, Y_train_s, batch_size):
            model, opt_state, loss, aux = step(
                model,
                opt_state,
                batch_x,
                batch_y,
                step_mask_jax,
                n_channels,
                optimizer,
                w_psd_jax,
                w_fc_jax,
                decode_basis_jax,
                decode_mean_jax,
            )
            epoch_loss += loss.item()
            for key in comp_sums:
                comp_sums[key] += float(aux[key])
            batches += 1

        avg_train_loss = float(epoch_loss / batches)
        train_losses.append(avg_train_loss)
        avg_comps = {key: val / batches for key, val in comp_sums.items()}

        last_val_loss = float(
            compute_loss(
                model,
                X_val_jnp,
                Y_val_jnp,
                val_mask_jax,
                n_channels,
                w_psd_jax,
                w_fc_jax,
                decode_basis_jax,
                decode_mean_jax,
            )[0].item()
        )
        val_losses.append(last_val_loss)

        if np.isnan(avg_train_loss) or np.isnan(last_val_loss):
            msg = "Loss is NaN. Aborting training."
            raise ValueError(msg)

        if last_val_loss < best_val_loss:
            best_val_loss = last_val_loss
            best_model = model
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        pbar.set_postfix(
            train_loss=f"{avg_train_loss:.4f}",
            val_loss=f"{last_val_loss:.4f}",
            mse=f"{avg_comps['mse']:.4f}",
            psd=f"{avg_comps['psd']:.4f}",
            fc=f"{avg_comps['fc']:.4f}",
            L=int(step_mask.sum()),
        )

        if epochs_without_improvement >= patience:
            break

    return best_model, train_losses, val_losses


def plot_training_curves(train_losses: list[float], val_losses: list[float], plot_path: str) -> None:
    """Plot training and validation loss curves."""
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss (Scaled MSE)", linewidth=1.5)
    plt.plot(val_losses, label="Val Loss (Scaled MSE)", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Scaled MSE")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(visible=True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)


def evaluate_model(
    model: eqx.Module,
    y_pipeline: Pipeline,
    X_val_m: FloatArray,
    Y_val: FloatArray,
    latent_dim: int,
) -> tuple[FloatArray, float]:
    """Evaluate the trained model in raw EEG units.

    The model predicts in model space (``k``-dimensional latent components under a projection);
    predictions are decoded back to raw EEG channels via ``y_pipeline.inverse_transform`` and
    scored against the raw EEG targets, so the reported MSE is comparable across ``latent_dim``.

    Parameters
    ----------
    model : eqx.Module
        The trained Equinox MLP model.
    y_pipeline : Pipeline
        The fitted raw-EEG -> model-space transform (used here for its inverse).
    X_val_m : FloatArray
        Model-space validation inputs, shape ``(samples_val, n_features)``.
    Y_val : FloatArray
        Raw EEG targets for the validation set, shape ``(samples_val, horizon * n_eeg)``.
    latent_dim : int
        Model output channel count ``k`` (latent dimension, else the raw EEG channel count).

    Returns
    -------
    Y_pred_raw : FloatArray
        Decoded raw-EEG predictions flattened per step/channel, shape ``(samples_val, horizon * n_eeg)``.
    mse : float
        Mean squared error of the raw-EEG predictions.
    """
    Y_pred_m = np.array(predict_batch(model, jnp.array(X_val_m)))  # (samples, horizon * k)
    Y_pred_raw = np.asarray(y_pipeline.inverse_transform(Y_pred_m.reshape(-1, latent_dim))).reshape(
        Y_pred_m.shape[0], -1
    )
    mse = np.mean((Y_val - Y_pred_raw) ** 2)
    return Y_pred_raw, float(mse)


def train_and_save_predictor(  # noqa: PLR0915
    config: NNPredictorConfig,
    data_files: list[str],
    artifact_dir: Path,
    *,
    seed_offset: int = 0,
) -> float:
    """Run the full NN-predictor pipeline for one config and save all artifacts.

    Prepares datasets from ``data_files``, fits the EEG/control transforms (a channel
    standardizer plus an optional PCA projection), builds and trains the autoregressive MLP,
    then writes ``model.eqx``, ``loss_curve.png``, ``training_stats.json`` and ``comparison.png``
    into ``artifact_dir``.

    Parameters
    ----------
    config : NNPredictorConfig
        Typed, validated configuration with ``simulation``, ``model`` and ``training``
        sections (with any sweep overrides already applied by the caller).
    data_files : list[str]
        Paths to the ``.npz`` trajectory files to train on.
    artifact_dir : Path
        Existing directory the artifacts are written into.
    seed_offset : int, optional
        Added to ``training.seed`` to decorrelate otherwise-identical runs (used to
        vary the seed across sweep trials). Defaults to 0.

    Returns
    -------
    float
        *Normalized* mean squared error on the validation set: the raw-EEG MSE divided by the
        variance of the validation targets, i.e. the fraction of target variance the forecast
        leaves unexplained (``0`` perfect, ``1`` no better than predicting the mean). Normalizing
        makes it comparable across trials whose window offsets -- and therefore target variance --
        differ; the un-normalized MSE is still recorded in ``training_stats.json``.
    """
    sim_cfg = config.simulation
    downsample = sim_cfg.downsample
    n_steps_cfg = sim_cfg.n_steps
    dt_real = sim_cfg.dt * downsample

    model_cfg = config.model
    n_y = model_cfg.n_y
    n_u = model_cfg.n_u
    horizon = model_cfg.horizon
    hidden_size = model_cfg.hidden_size
    depth = model_cfg.depth
    activation = model_cfg.activation
    latent_dim = model_cfg.latent_dim

    train_cfg = config.training
    epochs = train_cfg.epochs
    batch_size = train_cfg.batch_size
    learning_rate = train_cfg.learning_rate
    weight_decay = train_cfg.weight_decay
    train_split = train_cfg.train_split
    curriculum_start_epoch = train_cfg.curriculum_start_epoch
    curriculum_end_epoch = train_cfg.curriculum_end_epoch
    curriculum_max_steps = train_cfg.curriculum_max_steps
    seed = train_cfg.seed + seed_offset
    w_psd = train_cfg.w_psd
    w_fc = train_cfg.w_fc
    patience = train_cfg.patience
    scaler_type = train_cfg.scaler
    global_scaling = train_cfg.global_scaling

    # Build raw (unscaled) windows, then split chronologically (a random split would leak between
    # the heavily-overlapping adjacent windows).
    X_full, Y_full, n_channels = prepare_datasets(data_files, n_steps_cfg, downsample, n_y, n_u, horizon)
    n_controls = (X_full.shape[1] - n_y * n_channels) // (n_u + horizon)

    split_idx = int(train_split * len(X_full))
    X_train, X_val = X_full[:split_idx], X_full[split_idx:]
    Y_train, Y_val = Y_full[:split_idx], Y_full[split_idx:]

    # Fit the transforms on the training split only. The y-pipeline standardizes the raw EEG
    # channels, then (optionally) projects the standardized channels onto k PCA components, so
    # "model space" is standardize-then-project. Controls get their own standardizer.
    y_past_train = X_train[:, : n_y * n_channels].reshape(-1, n_channels)
    u_past_train = X_train[:, n_y * n_channels : n_y * n_channels + n_u * n_controls].reshape(-1, n_controls)

    y_standardizer = Standardizer.fit(y_past_train, kind=scaler_type, global_scaling=global_scaling)
    u_standardizer = Standardizer.fit(u_past_train, kind=scaler_type, global_scaling=global_scaling)

    pca = None if latent_dim is None else PCAProjection.fit(y_standardizer.transform(y_past_train), latent_dim)
    y_pipeline = Pipeline((y_standardizer, pca)) if pca is not None else Pipeline((y_standardizer,))
    u_pipeline = Pipeline((u_standardizer,))

    # Model output dimension: latent k with a projection, else the raw channel count.
    latent = pca.basis.shape[0] if pca is not None else n_channels

    # Inputs -> model space; targets stay in standardized-channel (loss) space.
    X_train_m = transform_features(X_train, y_pipeline, u_pipeline, n_y, n_channels, n_controls)
    X_val_m = transform_features(X_val, y_pipeline, u_pipeline, n_y, n_channels, n_controls)
    Y_train_s = apply_to_blocks(Y_train, Pipeline((y_standardizer,)), n_channels)
    Y_val_s = apply_to_blocks(Y_val, Pipeline((y_standardizer,)), n_channels)

    key = jax.random.PRNGKey(seed)
    in_size = n_y * latent + n_u * n_controls
    out_size = latent

    model = create_model(
        in_size, out_size, hidden_size, depth, key, n_y, n_u, horizon, latent, n_controls, activation=activation
    )

    decode_basis = pca.basis if pca is not None else None
    decode_mean = pca.mean if pca is not None else None

    model, train_losses, val_losses = train_model(
        model,
        X_train_m,
        Y_train_s,
        X_val_m,
        Y_val_s,
        epochs,
        batch_size,
        learning_rate,
        weight_decay,
        n_channels,
        curriculum_start_epoch=curriculum_start_epoch,
        curriculum_end_epoch=curriculum_end_epoch,
        curriculum_max_steps=curriculum_max_steps,
        w_psd=w_psd,
        w_fc=w_fc,
        patience=patience,
        decode_basis=decode_basis,
        decode_mean=decode_mean,
    )

    plot_training_curves(train_losses, val_losses, str(artifact_dir / "loss_curve.png"))

    if not isinstance(model, AutoregressivePredictor):
        msg = f"expected train_model to return an AutoregressivePredictor, got {type(model)}"
        raise TypeError(msg)
    MLPArtifact(
        model=model,
        dt=float(dt_real),
        downsample=int(downsample),
        y_pipeline=y_pipeline,
        u_pipeline=u_pipeline,
    ).save(artifact_dir / "model.eqx")

    Y_pred_raw_flat, mse = evaluate_model(model, y_pipeline, X_val_m, Y_val, latent)
    nmse = float(mse) / float(np.var(Y_val))

    stats = {"train_loss": train_losses, "val_loss": val_losses, "mse": float(mse), "nmse": nmse}
    (artifact_dir / "training_stats.json").write_text(json.dumps(stats, indent=2))

    Y_pred_unflat = np.asarray(reshape_to_trajectory(Y_pred_raw_flat, horizon, n_channels))
    n_plot_samples = min(200, len(Y_val))
    y_true_anchors = X_val[:n_plot_samples, (n_y - 1) * n_channels : n_y * n_channels]
    fig, _ = plot_multistep_predictions(
        y_true=y_true_anchors,
        y_pred=Y_pred_unflat[:n_plot_samples],
        dt=dt_real,
        channels=list(range(min(4, n_channels))),
        stride=horizon,
        title=f"EEG {horizon}-Step Ahead Prediction",
    )
    fig.savefig(str(artifact_dir / "comparison.png"), dpi=300)
    plt.close(fig)

    return nmse
