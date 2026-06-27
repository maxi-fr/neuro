"""Script to train a JAX-based Neural Network to predict EEG signals (Multi-step predictor).

This script implements an MLP using Equinox and Optax to predict a horizon of N steps
of EEG data from past EEG and past/future stimulation inputs. It supports training on
multiple trajectories.
"""

import argparse
import datetime
import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import yaml
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tvboptim.observations.observation import compute_fc

from neuro.jansen_rit_jax import enable_x64
from neuro.prediction import AutoregressivePredictor
from utils.plotting import plot_multistep_predictions


def save_artifact(
    artifact: Path, model: eqx.Module, scaler_u: StandardScaler, scaler_y: StandardScaler, meta: dict[str, object]
) -> None:
    """Persist the trained MLP (eqx leaves) plus a JSON sidecar and the scaler arrays.

    The comparison notebook's ``neuro.prediction.NNPredictor.load`` rebuilds an MLP
    skeleton from ``meta`` (architecture sizes), deserialises the leaves, and
    inverse-scales with the saved ``StandardScaler`` statistics. Three files share the
    artifact stem: ``<stem>.eqx`` (weights), ``<stem>.json`` (meta), and
    ``<stem>.scalers.npz`` (scaler mean/scale arrays).

    Parameters
    ----------
    artifact : Path
        Path to the artifact file.
    model : eqx.Module
        The trained Equinox module (MLP).
    scaler_u : StandardScaler
        Scikit-learn StandardScaler fitted to the input controls u.
    scaler_y : StandardScaler
        Scikit-learn StandardScaler fitted to the output targets y.
    meta : dict[str, object]
        Metadata dictionary to be saved alongside the model.
    """
    artifact.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(str(artifact), model)
    artifact.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    u_mean, u_scale = scaler_u.mean_, scaler_u.scale_
    y_mean, y_scale = scaler_y.mean_, scaler_y.scale_
    if u_mean is None or u_scale is None or y_mean is None or y_scale is None:
        msg = "Scalers must be fitted before saving the artifact."
        raise ValueError(msg)
    np.savez(artifact.with_suffix(".scalers.npz"), u_mean=u_mean, u_scale=u_scale, y_mean=y_mean, y_scale=y_scale)


def load_trajectory(data_file: str, n_steps: int, downsample: int) -> tuple[np.ndarray, np.ndarray]:
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
    u_data : np.ndarray
        The stimulation input trajectory, shape ``(T, C_u)``.
    y_data : np.ndarray
        The measured output (EEG) trajectory, shape ``(T, C_y)``.
    """
    print(f"Loading data from {data_file}...")
    with np.load(data_file) as data:
        max_idx = n_steps * downsample
        y_data = data["universal_y_mea"][:max_idx:downsample]
        u_data = data["universal_u"][:max_idx:downsample]
    return u_data, y_data


def extract_windows_flattened(data: np.ndarray, window_size: int) -> np.ndarray:
    """Extract sliding windows from a 2D array and flatten the time dimension.

    Parameters
    ----------
    data : np.ndarray
        Input array of shape (T, C).
    window_size : int
        Size of the sliding window.

    Returns
    -------
    np.ndarray
        Flattened sliding windows of shape (T - window_size + 1, window_size * C).
    """
    _, channels = data.shape
    view = np.lib.stride_tricks.sliding_window_view(data, (window_size, channels))
    return view.reshape(-1, window_size * channels)


def scale_flat_sequence(data_flat: np.ndarray, scaler: StandardScaler, channels: int) -> np.ndarray:
    """Scale a flattened sequence per-channel.

    Parameters
    ----------
    data_flat : np.ndarray
        Flattened sequence of shape (samples, time_steps * channels).
    scaler : StandardScaler
        Fitted scaler to apply.
    channels : int
        Number of channels per time step.

    Returns
    -------
    np.ndarray
        Scaled flattened sequence of the same shape.
    """
    samples = data_flat.shape[0]
    data_scaled = scaler.transform(data_flat.reshape(-1, channels))
    return data_scaled.reshape(samples, -1)


def unscale_flat_sequence(data_flat: np.ndarray, scaler: StandardScaler, channels: int) -> np.ndarray:
    """Inverse-scale a flattened sequence per-channel.

    Parameters
    ----------
    data_flat : np.ndarray
        Flattened sequence of shape (samples, time_steps * channels).
    scaler : StandardScaler
        Fitted scaler to apply.
    channels : int
        Number of channels per time step.

    Returns
    -------
    np.ndarray
        Unscaled flattened sequence of the same shape.
    """
    samples = data_flat.shape[0]
    data_unscaled = scaler.inverse_transform(data_flat.reshape(-1, channels))
    return data_unscaled.reshape(samples, -1)


def reshape_to_trajectory(data_flat: jax.Array | np.ndarray, horizon: int, channels: int) -> jax.Array | np.ndarray:
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


def reshape_flat_to_channels(data_flat: np.ndarray, channels: int) -> np.ndarray:
    """Reshape a flattened sequence to have channels as the last dimension.

    Parameters
    ----------
    data_flat : np.ndarray
        Flattened sequence of shape (samples, time_steps * channels).
    channels : int
        Number of channels.

    Returns
    -------
    np.ndarray
        Reshaped sequence of shape (-1, channels).
    """
    return data_flat.reshape(-1, channels)


def build_dataset_for_trajectory(
    u_data: np.ndarray, y_data: np.ndarray, n_y: int, n_u: int, N: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build the input/output pairs for the multi-step predictor.

    Parameters
    ----------
    u_data : np.ndarray
        The stimulation input trajectory of shape (T, C_u).
    y_data : np.ndarray
        The measured output (EEG) trajectory of shape (T, C_y).
    n_y : int
        Number of past output steps to include in the input feature.
    n_u : int
        Number of past input steps to include in the input feature.
    N : int
        Prediction horizon (number of future steps to predict).

    Returns
    -------
    X : np.ndarray
        Input features array of shape (samples, n_y * C_y + n_u * C_u + N * C_u).
    Y : np.ndarray
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


def get_dataloaders(X: np.ndarray, Y: np.ndarray, batch_size: int = 128) -> Iterator[tuple[jax.Array, jax.Array]]:
    """Generate batches of data.

    Parameters
    ----------
    X : np.ndarray
        Input features array, shape ``(samples, n_features)``.
    Y : np.ndarray
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
) -> tuple[np.ndarray, np.ndarray, int]:
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

    Returns
    -------
    X_full : np.ndarray
        Input features array, shape ``(total_samples, n_features)``.
    Y_full : np.ndarray
        Target labels array, shape ``(total_samples, n_targets)``.
    C_y : int
        Number of output channels.
    """
    print("\n1. Building dataset from multiple trajectories...")
    all_X, all_Y = [], []
    C_y = 1
    for df in data_files:
        u, y = load_trajectory(df, n_steps_cfg, downsample)
        C_y = y.shape[1]
        X_traj, Y_traj = build_dataset_for_trajectory(u, y, n_y, n_u, horizon)
        all_X.append(X_traj)
        all_Y.append(Y_traj)

    X_full = np.concatenate(all_X, axis=0)
    Y_full = np.concatenate(all_Y, axis=0)
    print(f"Total Dataset X shape: {X_full.shape}, Y shape: {Y_full.shape}")
    return X_full, Y_full, C_y


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
        activation=jax.nn.relu,
        key=key,
    )
    return AutoregressivePredictor(model=mlp, n_y=n_y, n_u=n_u, horizon=horizon, C_y=C_y, C_u=C_u)


def scale_dataset(  # noqa: PLR0913
    X: np.ndarray,
    Y: np.ndarray,
    scaler_y: StandardScaler,
    scaler_u: StandardScaler,
    n_y: int,
    n_u: int,
    C_y: int,
    C_u: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale datasets per-channel.

    Parameters
    ----------
    X : np.ndarray
        Input features array, shape ``(samples, n_features)``.
    Y : np.ndarray
        Target labels array, shape ``(samples, n_targets)``.
    scaler_y : StandardScaler
        Fitted scaler for the outputs.
    scaler_u : StandardScaler
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
    X_s : np.ndarray
        Scaled input features array, shape ``(samples, n_features)``.
    Y_s : np.ndarray
        Scaled target labels array, shape ``(samples, n_targets)``.
    """
    y_past_flat = X[:, : n_y * C_y]
    u_past_flat = X[:, n_y * C_y : n_y * C_y + n_u * C_u]
    u_fut_flat = X[:, n_y * C_y + n_u * C_u :]

    y_past_s = scale_flat_sequence(y_past_flat, scaler_y, C_y)
    u_past_s = scale_flat_sequence(u_past_flat, scaler_u, C_u)
    u_fut_s = scale_flat_sequence(u_fut_flat, scaler_u, C_u)

    X_s = np.concatenate([y_past_s, u_past_s, u_fut_s], axis=1)
    Y_s = scale_flat_sequence(Y, scaler_y, C_y)

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
    X_train_s: np.ndarray,
    Y_train_s: np.ndarray,
    X_val_s: np.ndarray,
    Y_val_s: np.ndarray,
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
    X_train_s : np.ndarray
        Scaled training inputs, shape ``(samples_train, n_features)``.
    Y_train_s : np.ndarray
        Scaled training targets, shape ``(samples_train, n_targets)``.
    X_val_s : np.ndarray
        Scaled validation inputs, shape ``(samples_val, n_features)``.
    Y_val_s : np.ndarray
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

    print("\n2. Training JAX MLP with Optax (with Weight Decay and Curriculum Learning)...")
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

        if last_val_loss < best_val_loss:
            best_val_loss = last_val_loss
            best_model = model
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        pbar.set_postfix(train_loss=f"{avg_train_loss:.4f}", val_loss=f"{last_val_loss:.4f}", alpha=f"{alpha:.2f}")

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch}. No improvement for {patience} epochs.")
            break

    print(f"\nTraining completed. Best Val Loss (N-Step MSE): {best_val_loss:.6f}")
    return best_model, train_losses, val_losses


def plot_training_curves(train_losses: list[float], val_losses: list[float], plot_path: str) -> None:
    """Plot training and validation loss curves."""
    print("\nPlotting training curves...")
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
    print(f"Training curves saved to {plot_path}")


def evaluate_model(
    model: eqx.Module,
    scaler_y: StandardScaler,
    X_val_s: np.ndarray,
    Y_val: np.ndarray,
    C_y: int,
) -> tuple[np.ndarray, float]:
    """Evaluate the trained model.

    Parameters
    ----------
    model : eqx.Module
        The trained Equinox MLP model.
    scaler_y : StandardScaler
        Fitted scaler for the targets.
    X_val_s : np.ndarray
        Scaled validation inputs, shape ``(samples_val, n_features)``.
    Y_val : np.ndarray
        Target values for the validation set, shape ``(samples_val, n_targets)``.
    C_y : int
        Number of output channels.

    Returns
    -------
    Y_pred_unscaled : np.ndarray
        Absolute predictions flattened per step/channel, shape ``(samples_val, n_targets)``.
    mse : float
        Mean squared error of the absolute predictions.
    """
    Y_pred_s = np.array(predict_batch(model, jnp.array(X_val_s)))
    Y_pred_unscaled = unscale_flat_sequence(Y_pred_s, scaler_y, C_y)

    mse = np.mean((Y_val - Y_pred_unscaled) ** 2)
    print(f"\nValidation MSE (Absolute Error): {mse:.4f}")
    return Y_pred_unscaled, float(mse)


def main() -> None:  # noqa: PLR0915
    """Execute the main script."""
    enable_x64()
    parser = argparse.ArgumentParser(description="Run JAX NN Predictor on multiple trajectories.")
    parser.add_argument(
        "--config", type=str, default="configs/nn_predictor/nn_predictor_config.yaml", help="Path to config YAML."
    )
    parser.add_argument("--data-path", type=str, help="Override config data path.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return

    with config_path.open() as f:
        config: dict[str, Any] = yaml.safe_load(f)

    sim_cfg = config.get("simulation", {})
    downsample = sim_cfg.get("downsample", 1)
    n_steps_cfg = sim_cfg.get("n_steps", 2000)
    dt_base = sim_cfg.get("dt", 1e-4)
    dt_real = dt_base * downsample

    model_cfg = config.get("model", {})
    n_y = model_cfg.get("n_y", 5)
    n_u = model_cfg.get("n_u", 5)
    horizon = model_cfg.get("horizon", 5)
    hidden_size = int(model_cfg.get("hidden_size", 128))
    depth = model_cfg.get("depth", 2)

    train_cfg = config.get("training", {})
    epochs = train_cfg.get("epochs", 100)
    batch_size = train_cfg.get("batch_size", 128)
    learning_rate = float(train_cfg.get("learning_rate", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    train_split = float(train_cfg.get("train_split", 0.8))
    curriculum_decay_fraction = float(train_cfg.get("curriculum_decay_fraction", 0.8))
    seed = int(train_cfg.get("seed", 69))
    w_psd = float(train_cfg.get("w_psd", 0.0))
    w_fc = float(train_cfg.get("w_fc", 0.0))
    patience = int(train_cfg.get("patience", 50))

    artifact_dir = config.get("artifact")
    if artifact_dir is None:
        local_now = datetime.datetime.now(datetime.UTC).astimezone()
        artifact_dir = f"artifacts/nn_predictor_{local_now.strftime('%Y-%m-%d_%H-%M-%S')}"
    artifact_dir: Path = Path(artifact_dir)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, artifact_dir / config_path.name)
    artifact = artifact_dir / "model.eqx"

    data_path_str = args.data_path or sim_cfg.get("data_path")
    if not data_path_str:
        print("data_path not specified in config or arguments.")
        return

    data_path = Path(data_path_str)
    if not data_path.is_dir():
        print(f"data_path is not a valid directory: {data_path}")
        return

    data_files = sorted([str(p) for p in data_path.glob("*.npz")])

    if not data_files:
        print(f"No .npz data files found in: {data_path}")
        return

    X_full, Y_full, C_y = prepare_datasets(data_files, n_steps_cfg, downsample, n_y, n_u, horizon)
    C_u = (X_full.shape[1] - n_y * C_y) // (n_u + horizon)

    split_idx = int(train_split * len(X_full))
    X_train, X_val = X_full[:split_idx], X_full[split_idx:]
    Y_train, Y_val = Y_full[:split_idx], Y_full[split_idx:]

    y_past_train = reshape_flat_to_channels(X_train[:, : n_y * C_y], C_y)
    u_past_train = reshape_flat_to_channels(X_train[:, n_y * C_y : n_y * C_y + n_u * C_u], C_u)

    scaler_y = StandardScaler()
    scaler_y.fit(y_past_train)

    scaler_u = StandardScaler()
    scaler_u.fit(u_past_train)

    X_train_s, Y_train_s = scale_dataset(X_train, Y_train, scaler_y, scaler_u, n_y, n_u, C_y, C_u)
    X_val_s, Y_val_s = scale_dataset(X_val, Y_val, scaler_y, scaler_u, n_y, n_u, C_y, C_u)

    key = jax.random.PRNGKey(seed)
    in_size = n_y * C_y + n_u * C_u
    out_size = C_y

    model = create_model(in_size, out_size, hidden_size, depth, key, n_y, n_u, horizon, C_y, C_u)

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

    loss_plot_path = artifact_dir / "loss_curve.png"
    plot_training_curves(train_losses, val_losses, str(loss_plot_path))

    save_artifact(
        artifact,
        model,
        scaler_u,
        scaler_y,
        {
            "in_size": int(in_size),
            "out_size": int(out_size),
            "hidden_size": int(hidden_size),
            "depth": int(depth),
            "n_y": int(n_y),
            "n_u": int(n_u),
            "horizon": int(horizon),
            "downsample": int(downsample),
            "dt": float(dt_real),
            "n_channels": int(C_y),
            "n_controls": int(C_u),
        },
    )
    print(f"Saved NN predictor artifact -> {artifact}")

    Y_pred_abs_flat, _mse = evaluate_model(model, scaler_y, X_val_s, Y_val, C_y)

    stats_path = artifact_dir / "training_stats.json"
    stats = {
        "train_loss": train_losses,
        "val_loss": val_losses,
    }
    stats_path.write_text(json.dumps(stats, indent=2))

    plot_path = artifact_dir / "comparison.png"
    Y_pred_unflat = np.asarray(reshape_to_trajectory(Y_pred_abs_flat, horizon, C_y))

    n_plot_samples = min(200, len(Y_val))
    split_idx * dt_real
    fig, _ = plot_multistep_predictions(
        y_true=Y_val[:n_plot_samples],
        y_pred=Y_pred_unflat[:n_plot_samples],
        dt=dt_real,
        channels=list(range(min(4, C_y))),
        stride=horizon,
        title=f"EEG {horizon}-Step Ahead Prediction",
    )
    fig.savefig(str(plot_path), dpi=300)
    plt.close(fig)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
