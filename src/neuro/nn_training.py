"""Core training and evaluation functions for the JAX-based Neural Network predictor.

This module implements an MLP using Equinox and Optax to predict a horizon of N steps
of EEG data from past EEG and past/future stimulation inputs. It supports training on
multiple trajectories.
"""

import json
from collections.abc import Iterator
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler
from tqdm import tqdm
from tvboptim.observations.observation import compute_fc

from neuro.config import NNPredictorConfig
from neuro.prediction import AutoregressivePredictor, MLPArtifact, get_activation
from neuro.types import FloatArray
from utils.plotting import plot_multistep_predictions

jax.config.update("jax_enable_x64", val=True)


def load_trajectory(data_file: str, n_steps: int, downsample: int) -> tuple[FloatArray, FloatArray]:
    """Load a single simulation trajectory.

    Parameters
    ----------
    data_file : str
        Path to the `.npz` data file containing the trajectory.
    n_steps : int
        The total number of time steps to load.
    downsample : int
        The downsampling factor to apply.

    Returns
    -------
    u_data : FloatArray
        The stimulation input trajectory, shape ``(T, C_u)``.
    y_data : FloatArray
        The measured output (EEG) trajectory, shape ``(T, C_y)``.
    """
    with np.load(data_file) as data:
        max_idx = n_steps * downsample
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


def scale_flat_sequence(
    data_flat: FloatArray,
    scaler: StandardScaler | RobustScaler,
    channels: int,
    global_scaling: bool,  # noqa: FBT001
) -> FloatArray:
    """Scale a flattened sequence per-channel.

    Parameters
    ----------
    data_flat : FloatArray
        Flattened sequence of shape (samples, time_steps * channels).
    scaler : StandardScaler | RobustScaler
        Fitted scaler to apply.
    channels : int
        Number of channels per time step.

    Returns
    -------
    FloatArray
        Scaled flattened sequence of the same shape.
    """
    samples = data_flat.shape[0]
    if global_scaling:
        data_scaled = scaler.transform(data_flat.reshape(-1, 1))
    else:
        data_scaled = scaler.transform(data_flat.reshape(-1, channels))
    return data_scaled.reshape(samples, -1)


def unscale_flat_sequence(
    data_flat: FloatArray,
    scaler: StandardScaler | RobustScaler,
    channels: int,
    global_scaling: bool,  # noqa: FBT001
) -> FloatArray:
    """Inverse-scale a flattened sequence per-channel.

    Parameters
    ----------
    data_flat : FloatArray
        Flattened sequence of shape (samples, time_steps * channels).
    scaler : StandardScaler | RobustScaler
        Fitted scaler to apply.
    channels : int
        Number of channels per time step.

    Returns
    -------
    FloatArray
        Unscaled flattened sequence of the same shape.
    """
    samples = data_flat.shape[0]
    if global_scaling:
        data_unscaled = scaler.inverse_transform(data_flat.reshape(-1, 1))
    else:
        data_unscaled = scaler.inverse_transform(data_flat.reshape(-1, channels))
    return data_unscaled.reshape(samples, -1)


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


def reshape_flat_to_channels(data_flat: FloatArray, channels: int) -> FloatArray:
    """Reshape a flattened sequence to have channels as the last dimension.

    Parameters
    ----------
    data_flat : FloatArray
        Flattened sequence of shape (samples, time_steps * channels).
    channels : int
        Number of channels.

    Returns
    -------
    FloatArray
        Reshaped sequence of shape (-1, channels).
    """
    return data_flat.reshape(-1, channels)


def build_dataset_for_trajectory(
    u_data: FloatArray, y_data: FloatArray, n_y: int, n_u: int, N: int
) -> tuple[FloatArray, FloatArray]:
    """Build the input/output pairs for the multi-step predictor.

    Parameters
    ----------
    u_data : FloatArray
        The stimulation input trajectory of shape (T, C_u).
    y_data : FloatArray
        The measured output (EEG) trajectory of shape (T, C_y).
    n_y : int
        Number of past output steps to include in the input feature.
    n_u : int
        Number of past input steps to include in the input feature.
    N : int
        Prediction horizon (number of future steps to predict).

    Returns
    -------
    X : FloatArray
        Input features array of shape (samples, n_y * C_y + n_u * C_u + N * C_u).
    Y : FloatArray
        Target labels array of shape (samples, N * C_y).
    """
    T_src, _C_y = y_data.shape
    _, _C_u = u_data.shape

    start_idx = max(n_y - 1, n_u)
    end_idx = T_src - N
    k = np.arange(start_idx, end_idx)

    y_view = extract_windows_flattened(y_data, n_y)
    u_past_view = extract_windows_flattened(u_data, n_u)
    u_fut_view = extract_windows_flattened(u_data, N)

    X = np.concatenate([y_view[k - n_y + 1], u_past_view[k - n_u], u_fut_view[k]], axis=1)

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
    n_steps_cfg: int,
    downsample: int,
    n_y: int,
    n_u: int,
    horizon: int,
    projection: tuple[FloatArray, FloatArray] | None = None,
) -> tuple[FloatArray, FloatArray, int]:
    """Load data and build dataset across multiple trajectories.

    Parameters
    ----------
    data_files : list[str]
        List of paths to data files.
    n_steps_cfg : int
        Number of steps to load per trajectory.
    downsample : int
        Downsampling factor.
    n_y : int
        Number of past output steps to include.
    n_u : int
        Number of past input steps to include.
    horizon : int
        Prediction horizon.
    projection : tuple[FloatArray, FloatArray] | None
        Optional ``(E, mean)`` latent projection from :func:`fit_latent_projection`.
        When given, each trajectory's EEG is encoded ``y -> (y - mean) @ E.T`` before
        windowing, so the returned ``C_y`` is the latent dimension ``k`` instead of the
        raw EEG channel count.

    Returns
    -------
    X_full : FloatArray
        Input features array, shape ``(total_samples, n_features)``.
    Y_full : FloatArray
        Target labels array, shape ``(total_samples, n_targets)``.
    C_y : int
        Number of output channels (latent dimension ``k`` when ``projection`` is set).
    """
    all_X, all_Y = [], []
    C_y = 1
    for df in data_files:
        u, y = load_trajectory(df, n_steps_cfg, downsample)
        if projection is not None:
            basis, mean = projection
            y = (y - mean) @ basis.T
        C_y = y.shape[1]
        X_traj, Y_traj = build_dataset_for_trajectory(u, y, n_y, n_u, horizon)
        all_X.append(X_traj)
        all_Y.append(Y_traj)

    X_full = np.concatenate(all_X, axis=0)
    Y_full = np.concatenate(all_Y, axis=0)
    return X_full, Y_full, C_y


def fit_latent_projection(
    data_files: list[str],
    n_steps_cfg: int,
    downsample: int,
    latent_dim: int | None = None,
    explained_variance: float | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Fit a fixed orthonormal PCA basis on the concatenated training EEG.

    The basis maps the full EEG channel space onto ``k`` latent components so the
    predictor can train and run in the reduced space. The projection is
    ``z = (y - mean) @ E.T`` with inverse ``y = z @ E + mean``; the mean is part of the
    map (PCA centres the data) and must be applied on both encode and decode.

    Parameters
    ----------
    data_files : list[str]
        Paths to the trajectory ``.npz`` files to fit the basis on.
    n_steps_cfg : int
        Number of steps to load per trajectory (matches :func:`prepare_datasets`).
    downsample : int
        Downsampling factor (matches :func:`prepare_datasets`).
    latent_dim : int | None
        Number of latent components ``k``. Takes precedence when set.
    explained_variance : float | None
        If ``latent_dim`` is ``None``, the smallest ``k`` reaching this cumulative
        explained-variance fraction (in ``(0, 1)``) is chosen automatically.

    Returns
    -------
    E : FloatArray
        Orthonormal PCA basis, shape ``(k, C_y)``.
    mean : FloatArray
        Per-channel training mean, shape ``(C_y,)``.
    """
    if latent_dim is None and explained_variance is None:
        msg = "fit_latent_projection requires either latent_dim or explained_variance"
        raise ValueError(msg)
    y_all = np.concatenate([load_trajectory(df, n_steps_cfg, downsample)[1] for df in data_files], axis=0)
    pca = PCA(n_components=latent_dim if latent_dim is not None else explained_variance)
    pca.fit(np.asarray(y_all, dtype=np.float64))
    return np.asarray(pca.components_, dtype=np.float64), np.asarray(pca.mean_, dtype=np.float64)


def create_model(  # noqa: PLR0913
    in_size: int,
    out_size: int,
    hidden_size: int,
    depth: int,
    key: jax.Array,
    n_y: int,
    n_u: int,
    horizon: int,
    C_y: int,
    C_u: int,
    activation: str = "relu",
) -> eqx.Module:
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
    C_y : int
        Number of output channels.
    C_u : int
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
        model=mlp, n_y=n_y, n_u=n_u, horizon=horizon, C_y=C_y, C_u=C_u, activation=activation
    )


def scale_dataset(  # noqa: PLR0913
    X: FloatArray,
    Y: FloatArray,
    scaler_y: StandardScaler | RobustScaler,
    scaler_u: StandardScaler | RobustScaler,
    n_y: int,
    n_u: int,
    C_y: int,
    C_u: int,
    global_scaling: bool,  # noqa: FBT001
) -> tuple[FloatArray, FloatArray]:
    """Scale datasets per-channel.

    Parameters
    ----------
    X : FloatArray
        Input features array, shape ``(samples, n_features)``.
    Y : FloatArray
        Target labels array, shape ``(samples, n_targets)``.
    scaler_y : StandardScaler | RobustScaler
        Fitted scaler for the outputs.
    scaler_u : StandardScaler | RobustScaler
        Fitted scaler for the inputs.
    n_y : int
        Number of past output steps.
    n_u : int
        Number of past input steps.
    C_y : int
        Number of output channels.
    C_u : int
        Number of input control channels.

    Returns
    -------
    X_s : FloatArray
        Scaled input features array, shape ``(samples, n_features)``.
    Y_s : FloatArray
        Scaled target labels array, shape ``(samples, n_targets)``.
    """
    y_past_flat = X[:, : n_y * C_y]
    u_past_flat = X[:, n_y * C_y : n_y * C_y + n_u * C_u]
    u_fut_flat = X[:, n_y * C_y + n_u * C_u :]

    y_past_s = scale_flat_sequence(y_past_flat, scaler_y, C_y, global_scaling)
    u_past_s = scale_flat_sequence(u_past_flat, scaler_u, C_u, global_scaling)
    u_fut_s = scale_flat_sequence(u_fut_flat, scaler_u, C_u, global_scaling)

    X_s = np.concatenate([y_past_s, u_past_s, u_fut_s], axis=1)
    Y_s = scale_flat_sequence(Y, scaler_y, C_y, global_scaling)

    return X_s, Y_s


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


def compute_loss(  # noqa: PLR0913
    m: eqx.Module,
    x: jax.Array,
    y: jax.Array,
    alpha: jax.Array,
    C_y: int,
    w_psd: jax.Array,
    w_fc: jax.Array,
) -> jax.Array:
    """Compute Mixed MSE loss for a batch along with PSD and FC losses.

    alpha: 1.0 = pure 1-step Teacher Forcing loss.
           0.0 = pure N-step unrolled loss.

    Parameters
    ----------
    m : eqx.Module
        The Equinox model.
    x : jax.Array
        Batch of input features, shape ``(batch_size, n_features)``.
    y : jax.Array
        Batch of target labels, shape ``(batch_size, horizon * C_y)``.
    alpha : jax.Array
        Curriculum learning parameter balancing 1-step and N-step loss.
    C_y : int
        Number of output channels.
    w_psd : jax.Array
        Weight for the PSD loss.
    w_fc : jax.Array
        Weight for the FC loss.

    Returns
    -------
    jax.Array
        The computed scalar loss value.
    """
    pred_y = predict_batch(m, x)

    # The first C_y elements correspond to the first step of the rollout,
    # which relies purely on the ground-truth past context (1-step prediction).
    loss_1step = jnp.mean((pred_y[:, :C_y] - y[:, :C_y]) ** 2)

    # The full N-step unrolled loss
    loss_Nstep = jnp.mean((pred_y - y) ** 2)

    mse_loss = alpha * loss_1step + (1.0 - alpha) * loss_Nstep

    horizon = y.shape[1] // C_y

    if horizon > 1:
        # Reshape to (batch_size, horizon, C_y) to compute statistics over the trajectory
        pred_traj = reshape_to_trajectory(pred_y, horizon, C_y)
        true_traj = reshape_to_trajectory(y, horizon, C_y)

        # Simple FFT-based PSD loss across the horizon dimension (axis=1)
        pred_psd = jnp.abs(jnp.fft.rfft(pred_traj, axis=1)) ** 2
        true_psd = jnp.abs(jnp.fft.rfft(true_traj, axis=1)) ** 2
        eps = 1e-8
        loss_psd = jnp.mean((jnp.log(pred_psd + eps) - jnp.log(true_psd + eps)) ** 2)

        # Functional Connectivity loss across the horizon dimension
        # compute_fc expects 2D (time, regions) or handles batched?
        # The tvboptim compute_fc uses _as_2d which doesn't automatically batch.
        # So we use jax.vmap to apply it over the batch dimension!
        batched_compute_fc = jax.vmap(compute_fc)
        pred_fc = batched_compute_fc(pred_traj)
        true_fc = batched_compute_fc(true_traj)
        loss_fc = jnp.mean((pred_fc - true_fc) ** 2)
    else:
        loss_psd = jnp.array(0.0)
        loss_fc = jnp.array(0.0)

    return mse_loss + w_psd * loss_psd + w_fc * loss_fc


@eqx.filter_jit
def step(  # noqa: PLR0913
    m: eqx.Module,
    opt_s: optax.OptState,
    x: jax.Array,
    y: jax.Array,
    alpha: jax.Array,
    C_y: int,
    optimizer: optax.GradientTransformation,
    w_psd: jax.Array,
    w_fc: jax.Array,
) -> tuple[eqx.Module, optax.OptState, jax.Array]:
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
    alpha : jax.Array
        Curriculum learning parameter.
    C_y : int
        Number of output channels.
    optimizer : optax.GradientTransformation
        The Optax optimizer.
    w_psd : jax.Array
        Weight for the PSD loss.
    w_fc : jax.Array
        Weight for the FC loss.

    Returns
    -------
    new_m : eqx.Module
        The updated Equinox model.
    new_opt_s : optax.OptState
        The updated optimizer state.
    loss_val : jax.Array
        The scalar loss value for the batch.
    """
    loss_val, grads = eqx.filter_value_and_grad(compute_loss)(m, x, y, alpha, C_y, w_psd, w_fc)
    updates, new_opt_s = optimizer.update(grads, opt_s, m)  # type: ignore
    new_m: eqx.Module = eqx.apply_updates(m, updates)
    return new_m, new_opt_s, loss_val


def train_model(  # noqa: PLR0913
    model: eqx.Module,
    X_train_s: FloatArray,
    Y_train_s: FloatArray,
    X_val_s: FloatArray,
    Y_val_s: FloatArray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    curriculum_decay_fraction: float,
    C_y: int,
    w_psd: float = 0.0,
    w_fc: float = 0.0,
    patience: int = 50,
) -> tuple[eqx.Module, list[float], list[float]]:
    """Train the MLP model using Optax.

    Parameters
    ----------
    model : eqx.Module
        The Equinox MLP model to train.
    X_train_s : FloatArray
        Scaled training inputs, shape ``(samples_train, n_features)``.
    Y_train_s : FloatArray
        Scaled training targets, shape ``(samples_train, n_targets)``.
    X_val_s : FloatArray
        Scaled validation inputs, shape ``(samples_val, n_features)``.
    Y_val_s : FloatArray
        Scaled validation targets, shape ``(samples_val, n_targets)``.
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size.
    learning_rate : float
        Learning rate.
    weight_decay : float
        Weight decay.
    curriculum_decay_fraction : float
        Fraction of epochs to decay curriculum alpha.
    C_y : int
        Number of output channels.
    w_psd : float
        Weight for the PSD loss.
    w_fc : float
        Weight for the FC loss.
    patience : int
        Number of epochs with no improvement after which training will be stopped.

    Returns
    -------
    best_model : eqx.Module
        The trained Equinox model with the best validation loss.
    train_losses : list[float]
        Training loss per epoch.
    val_losses : list[float]
        Validation loss per epoch.
    """
    optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    best_val_loss = float("inf")
    best_model = model

    X_val_jnp = jnp.array(X_val_s)
    Y_val_jnp = jnp.array(Y_val_s)

    pbar = tqdm(range(epochs), desc="Training MLP")
    train_losses = []
    val_losses = []

    decay_epochs = int(epochs * curriculum_decay_fraction)

    w_psd_jax = jnp.array(w_psd, dtype=jnp.float32)
    w_fc_jax = jnp.array(w_fc, dtype=jnp.float32)

    epochs_without_improvement = 0

    for epoch in pbar:
        alpha = 1.0 - epoch / decay_epochs if epoch < decay_epochs else 0.0

        alpha_jax = jnp.array(alpha, dtype=jnp.float32)

        epoch_loss = 0.0
        batches = 0
        for batch_x, batch_y in get_dataloaders(X_train_s, Y_train_s, batch_size):
            model, opt_state, loss = step(
                model, opt_state, batch_x, batch_y, alpha_jax, C_y, optimizer, w_psd_jax, w_fc_jax
            )
            epoch_loss += loss.item()
            batches += 1

        avg_train_loss = float(epoch_loss / batches)
        train_losses.append(avg_train_loss)

        last_val_loss = float(
            compute_loss(model, X_val_jnp, Y_val_jnp, jnp.array(0.0), C_y, w_psd_jax, w_fc_jax).item()
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

        pbar.set_postfix(train_loss=f"{avg_train_loss:.4f}", val_loss=f"{last_val_loss:.4f}", alpha=f"{alpha:.2f}")

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


def evaluate_model(  # noqa: PLR0913
    model: eqx.Module,
    scaler_y: StandardScaler | RobustScaler,
    X_val_s: FloatArray,
    Y_val: FloatArray,
    C_y: int,
    global_scaling: bool,  # noqa: FBT001
) -> tuple[FloatArray, float]:
    """Evaluate the trained model.

    Parameters
    ----------
    model : eqx.Module
        The trained Equinox MLP model.
    scaler_y : StandardScaler | RobustScaler
        Fitted scaler for the targets.
    X_val_s : FloatArray
        Scaled validation inputs, shape ``(samples_val, n_features)``.
    Y_val : FloatArray
        Target values for the validation set, shape ``(samples_val, n_targets)``.
    C_y : int
        Number of output channels.

    Returns
    -------
    Y_pred_unscaled : FloatArray
        Absolute predictions flattened per step/channel, shape ``(samples_val, n_targets)``.
    mse : float
        Mean squared error of the absolute predictions.
    """
    Y_pred_s = np.array(predict_batch(model, jnp.array(X_val_s)))
    Y_pred_unscaled = unscale_flat_sequence(Y_pred_s, scaler_y, C_y, global_scaling)

    mse = np.mean((Y_val - Y_pred_unscaled) ** 2)
    return Y_pred_unscaled, float(mse)


def train_and_save_predictor(  # noqa: PLR0915
    config: NNPredictorConfig,
    data_files: list[str],
    artifact_dir: Path,
    *,
    seed_offset: int = 0,
) -> float:
    """Run the full NN-predictor pipeline for one config and save all artifacts.

    Prepares datasets from ``data_files``, fits the input/output scalers, builds and
    trains the autoregressive MLP, then writes ``model.eqx``, ``loss_curve.png``,
    ``training_stats.json`` and ``comparison.png`` into ``artifact_dir``.

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
        Mean squared error of the absolute predictions on the validation set.
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
    projection_cfg = model_cfg.projection
    enable_projection = projection_cfg.enable
    latent_dim = projection_cfg.latent_dim
    explained_variance = projection_cfg.explained_variance

    train_cfg = config.training
    epochs = train_cfg.epochs
    batch_size = train_cfg.batch_size
    learning_rate = train_cfg.learning_rate
    weight_decay = train_cfg.weight_decay
    train_split = train_cfg.train_split
    curriculum_decay_fraction = train_cfg.curriculum_decay_fraction
    seed = train_cfg.seed + seed_offset
    w_psd = train_cfg.w_psd
    w_fc = train_cfg.w_fc
    patience = train_cfg.patience
    scaler_type = train_cfg.scaler
    global_scaling = train_cfg.global_scaling

    projection: tuple[FloatArray, FloatArray] | None = None
    if enable_projection:
        if w_psd > 0 or w_fc > 0:
            msg = "Latent projection is incompatible with the per-channel PSD/FC losses (set w_psd=w_fc=0)."
            raise ValueError(msg)
        projection = fit_latent_projection(data_files, n_steps_cfg, downsample, latent_dim, explained_variance)

    X_full, Y_full, C_y = prepare_datasets(data_files, n_steps_cfg, downsample, n_y, n_u, horizon, projection)
    C_u = (X_full.shape[1] - n_y * C_y) // (n_u + horizon)

    split_idx = int(train_split * len(X_full))
    X_train, X_val = X_full[:split_idx], X_full[split_idx:]
    Y_train, Y_val = Y_full[:split_idx], Y_full[split_idx:]

    y_past_train = reshape_flat_to_channels(X_train[:, : n_y * C_y], C_y)
    u_past_train = reshape_flat_to_channels(X_train[:, n_y * C_y : n_y * C_y + n_u * C_u], C_u)

    if scaler_type == "robust":
        scaler_y = RobustScaler()
        scaler_u = RobustScaler()
    else:
        scaler_y = StandardScaler()
        scaler_u = StandardScaler()

    if global_scaling:
        scaler_y.fit(y_past_train.reshape(-1, 1))
        scaler_u.fit(u_past_train.reshape(-1, 1))
    else:
        scaler_y.fit(y_past_train)
        scaler_u.fit(u_past_train)

    X_train_s, Y_train_s = scale_dataset(X_train, Y_train, scaler_y, scaler_u, n_y, n_u, C_y, C_u, global_scaling)
    X_val_s, Y_val_s = scale_dataset(X_val, Y_val, scaler_y, scaler_u, n_y, n_u, C_y, C_u, global_scaling)

    key = jax.random.PRNGKey(seed)
    in_size = n_y * C_y + n_u * C_u
    out_size = C_y

    model = create_model(in_size, out_size, hidden_size, depth, key, n_y, n_u, horizon, C_y, C_u, activation=activation)

    model, train_losses, val_losses = train_model(
        model,
        X_train_s,
        Y_train_s,
        X_val_s,
        Y_val_s,
        epochs,
        batch_size,
        learning_rate,
        weight_decay,
        curriculum_decay_fraction,
        C_y,
        w_psd=w_psd,
        w_fc=w_fc,
        patience=patience,
    )

    plot_training_curves(train_losses, val_losses, str(artifact_dir / "loss_curve.png"))

    if not isinstance(model, AutoregressivePredictor):
        msg = f"expected train_model to return an AutoregressivePredictor, got {type(model)}"
        raise TypeError(msg)
    u_mean = getattr(scaler_u, "mean_", getattr(scaler_u, "center_", None))
    u_scale = getattr(scaler_u, "scale_", None)
    y_mean = getattr(scaler_y, "mean_", getattr(scaler_y, "center_", None))
    y_scale = getattr(scaler_y, "scale_", None)
    if u_mean is None or u_scale is None or y_mean is None or y_scale is None:
        msg = "Scalers must be fitted before saving the artifact."
        raise ValueError(msg)
    MLPArtifact(
        model=model,
        dt=float(dt_real),
        downsample=int(downsample),
        u_mean=np.asarray(u_mean, dtype=np.float64),
        u_scale=np.asarray(u_scale, dtype=np.float64),
        y_mean=np.asarray(y_mean, dtype=np.float64),
        y_scale=np.asarray(y_scale, dtype=np.float64),
        latent_basis=projection[0] if projection is not None else None,
        latent_mean=projection[1] if projection is not None else None,
    ).save(artifact_dir / "model.eqx")

    Y_pred_abs_flat, mse = evaluate_model(model, scaler_y, X_val_s, Y_val, C_y, global_scaling)

    stats = {"train_loss": train_losses, "val_loss": val_losses, "mse": float(mse)}
    (artifact_dir / "training_stats.json").write_text(json.dumps(stats, indent=2))

    Y_pred_unflat = np.asarray(reshape_to_trajectory(Y_pred_abs_flat, horizon, C_y))
    n_plot_samples = min(200, len(Y_val))
    y_true_anchors = X_val[:n_plot_samples, (n_y - 1) * C_y : n_y * C_y]
    fig, _ = plot_multistep_predictions(
        y_true=y_true_anchors,
        y_pred=Y_pred_unflat[:n_plot_samples],
        dt=dt_real,
        channels=list(range(min(4, C_y))),
        stride=horizon,
        title=f"EEG {horizon}-Step Ahead Prediction",
    )
    fig.savefig(str(artifact_dir / "comparison.png"), dpi=300)
    plt.close(fig)

    return mse
