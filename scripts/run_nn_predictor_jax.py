"""Script to train a JAX-based Neural Network to predict EEG signals (Multi-step predictor).

This script implements an MLP using Equinox and Optax to predict a horizon of N steps
of EEG data from past EEG and past/future stimulation inputs. It supports training on
multiple trajectories.
"""

import argparse
import datetime
import json
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

from neuro.jansen_rit_jax import enable_x64


def save_artifact(
    artifact: Path, model: eqx.Module, scaler_x: StandardScaler, scaler_y: StandardScaler, meta: dict[str, object]
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
    scaler_x : StandardScaler
        Scikit-learn StandardScaler fitted to the input features.
    scaler_y : StandardScaler
        Scikit-learn StandardScaler fitted to the output targets.
    meta : dict[str, object]
        Metadata dictionary to be saved alongside the model.
    """
    artifact.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(str(artifact), model)
    artifact.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    x_mean, x_scale = scaler_x.mean_, scaler_x.scale_
    y_mean, y_scale = scaler_y.mean_, scaler_y.scale_
    if x_mean is None or x_scale is None or y_mean is None or y_scale is None:
        msg = "Scalers must be fitted before saving the artifact."
        raise ValueError(msg)
    np.savez(artifact.with_suffix(".scalers.npz"), x_mean=x_mean, x_scale=x_scale, y_mean=y_mean, y_scale=y_scale)


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
        The stimulation input trajectory.
    y_data : np.ndarray
        The measured output (EEG) trajectory.
    """
    print(f"Loading data from {data_file}...")
    with np.load(data_file) as data:
        max_idx = n_steps * downsample
        y_data = data["universal_y_mea"][:max_idx:downsample]
        u_data = data["universal_u"][:max_idx:downsample]
    return u_data, y_data


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
    T_src, C_y = y_data.shape
    _, C_u = u_data.shape

    start_idx = max(n_y - 1, n_u)
    end_idx = T_src - N
    k = np.arange(start_idx, end_idx)

    y_view = np.lib.stride_tricks.sliding_window_view(y_data, (n_y, C_y)).reshape(-1, n_y * C_y)
    u_past_view = np.lib.stride_tricks.sliding_window_view(u_data, (n_u, C_u)).reshape(-1, n_u * C_u)
    u_fut_view = np.lib.stride_tricks.sliding_window_view(u_data, (N, C_u)).reshape(-1, N * C_u)

    X = np.concatenate([y_view[k - n_y + 1], u_past_view[k - n_u], u_fut_view[k]], axis=1)

    y_fut_view = np.lib.stride_tricks.sliding_window_view(y_data, (N, C_y)).reshape(-1, N * C_y)
    Y = y_fut_view[k + 1]

    return X, Y


def get_dataloaders(X: np.ndarray, Y: np.ndarray, batch_size: int = 128) -> Iterator[tuple[jax.Array, jax.Array]]:
    """Generate batches of data.

    Parameters
    ----------
    X : np.ndarray
        Input features array.
    Y : np.ndarray
        Target labels array.
    batch_size : int, optional
        Number of samples per batch. Defaults to 128.

    Yields
    ------
    batch_x : jax.Array
        A batch of input features.
    batch_y : jax.Array
        A batch of target labels.
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
        Input features array.
    Y_full : np.ndarray
        Target labels array.

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


def create_model(
    in_size: int,
    out_size: int,
    hidden_size: int | list[int],
    depth: int,
    key: jax.Array,
) -> eqx.Module:
    """Create the MLP model, supporting either a single width or a list of widths."""
    if isinstance(hidden_size, int):
        return eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=hidden_size,
            depth=depth,
            activation=jax.nn.relu,
            key=key,
        )

    keys = jax.random.split(key, len(hidden_size) + 1)
    layers = []
    last_size = in_size
    for size, k in zip(hidden_size, keys[:-1], strict=True):
        layers.append(eqx.nn.Linear(last_size, size, key=k))
        layers.append(eqx.nn.Lambda(jax.nn.relu))
        last_size = size
    layers.append(eqx.nn.Linear(last_size, out_size, key=keys[-1]))
    return eqx.nn.Sequential(layers)


@eqx.filter_jit
def predict_batch(m: eqx.Module, x: jax.Array) -> jax.Array:
    """Run model prediction on a batch."""
    return jax.vmap(m)(x)


def compute_loss(m: eqx.Module, x: jax.Array, y: jax.Array) -> jax.Array:
    """Compute MSE loss for a batch."""
    pred_y = predict_batch(m, x)
    return jnp.mean((pred_y - y) ** 2)


@eqx.filter_jit
def step(
    m: eqx.Module, opt_s: optax.OptState, x: jax.Array, y: jax.Array, optimizer: optax.GradientTransformation
) -> tuple[eqx.Module, optax.OptState, jax.Array]:
    """Perform one optimizer step."""
    loss_val, grads = eqx.filter_value_and_grad(compute_loss)(m, x, y)
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
) -> tuple[eqx.Module, list[float], list[float]]:
    """Train the MLP model using Optax.

    Parameters
    ----------
    model : eqx.Module
        The Equinox MLP model to train.
    X_train_s : np.ndarray
        Scaled training inputs.
    Y_train_s : np.ndarray
        Scaled training targets.
    X_val_s : np.ndarray
        Scaled validation inputs.
    Y_val_s : np.ndarray
        Scaled validation targets.
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size.
    learning_rate : float
        Learning rate.
    weight_decay : float
        Weight decay.

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

    print("\n2. Training JAX MLP with Optax (with Weight Decay and Early Stopping)...")
    best_val_loss = float("inf")
    best_model = model
    last_val_loss = float("nan")

    X_val_jnp = jnp.array(X_val_s)
    Y_val_jnp = jnp.array(Y_val_s)

    pbar = tqdm(range(epochs), desc="Training MLP")
    train_losses = []
    val_losses = []

    for _epoch in pbar:
        epoch_loss = 0.0
        batches = 0
        for batch_x, batch_y in get_dataloaders(X_train_s, Y_train_s, batch_size):
            model, opt_state, loss = step(model, opt_state, batch_x, batch_y, optimizer)
            epoch_loss += loss.item()
            batches += 1

        avg_train_loss = float(epoch_loss / batches)
        train_losses.append(avg_train_loss)

        last_val_loss = float(compute_loss(model, X_val_jnp, Y_val_jnp).item())
        val_losses.append(last_val_loss)

        if last_val_loss < best_val_loss:
            best_val_loss = last_val_loss
            best_model = model

        pbar.set_postfix(train_loss=f"{avg_train_loss:.6f}", val_loss=f"{last_val_loss:.6f}")

    print(f"\nTraining completed. Best Val Loss (Scaled MSE): {best_val_loss:.6f}")
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
    scaler_Y: StandardScaler,
    X_val_s: np.ndarray,
    Y_val: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Evaluate the trained model.

    Parameters
    ----------
    model : eqx.Module
        The trained Equinox MLP model.
    scaler_Y : StandardScaler
        Fitted scaler for the targets.
    X_val_s : np.ndarray
        Scaled validation inputs.
    Y_val : np.ndarray
        Target values for the validation set.

    Returns
    -------
    Y_pred_unscaled : np.ndarray
        Absolute predictions flattened per step/channel.
    mse : float
        Mean squared error of the absolute predictions.
    """
    Y_pred_s = np.array(predict_batch(model, jnp.array(X_val_s)))
    Y_pred_unscaled = scaler_Y.inverse_transform(Y_pred_s)

    mse = np.mean((Y_val - Y_pred_unscaled) ** 2)
    print(f"\nValidation MSE (Absolute Error): {mse:.4f}")
    return Y_pred_unscaled, float(mse)


def plot_predictions(  # noqa: PLR0913
    Y_val: np.ndarray,
    Y_pred: np.ndarray,
    split_idx: int,
    dt_real: float,
    horizon: int,
    C_y: int,
    plot_path: str = "nn_predictor_jax_comparison.png",
) -> None:
    """Plot N-step predictions vs true continuous data.

    Parameters
    ----------
    Y_val : np.ndarray
        Target values for the validation set.
    Y_pred : np.ndarray
        Absolute predictions shaped as (samples, horizon, C_y).
    split_idx : int
        Index where the validation split starts.
    dt_real : float
        Real time delta per step.
    horizon : int
        Prediction horizon.
    C_y : int
        Number of output channels.
    plot_path : str, optional
        Path to save the plot. Defaults to "nn_predictor_jax_comparison.png".
    """
    print("\n3. Plotting N-step horizon predictions...")

    num_channels_plot = min(4, C_y)
    _fig, axes = plt.subplots(num_channels_plot, 1, figsize=(10, 8), sharex=True)
    if num_channels_plot == 1:
        axes = [axes]

    # Plot a subset of the validation dataset continuous ground truth
    n_plot_samples = min(200, len(Y_val))
    val_start_time = split_idx * dt_real
    time_axis = val_start_time + np.arange(n_plot_samples) * dt_real

    for ch in range(num_channels_plot):
        # Extract the 1-step truth for a continuous line
        true_continuous = Y_val[:n_plot_samples, ch]
        axes[ch].plot(time_axis, true_continuous, label="True Data", color="black", linewidth=1.5)

        # Now overlay a few N-step predictions to explicitly show it predicts N steps ahead
        stride = horizon
        for idx in range(0, n_plot_samples - horizon, stride):
            n_step_pred = Y_pred[idx, :, ch]

            pred_time_axis = val_start_time + (idx + np.arange(horizon)) * dt_real

            label = "N-Step Prediction" if idx == 0 else ""
            axes[ch].plot(
                pred_time_axis, n_step_pred, label=label, color="red", linestyle="--", marker="o", markersize=3
            )

        axes[ch].set_ylabel(f"Ch {ch}")
        if ch == 0:
            axes[ch].legend()
            axes[ch].set_title(f"EEG {horizon}-Step Ahead Prediction (First {num_channels_plot} Channels)")

    axes[-1].set_xlabel("Time (seconds)")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to {plot_path}")


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

    artifact_dir = config.get("artifact")
    if artifact_dir is None:
        local_now = datetime.datetime.now(datetime.UTC).astimezone()
        artifact_dir = f"artifacts/nn_predictor_{local_now.strftime('%Y-%m-%d_%H-%M-%S')}"
    artifact_dir: Path = Path(artifact_dir)

    artifact_dir.mkdir(parents=True, exist_ok=True)
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

    split_idx = int(0.8 * len(X_full))
    X_train, X_val = X_full[:split_idx], X_full[split_idx:]
    Y_train, Y_val = Y_full[:split_idx], Y_full[split_idx:]

    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_val_s = scaler_X.transform(X_val)

    scaler_Y = StandardScaler()
    Y_train_s = scaler_Y.fit_transform(Y_train)
    Y_val_s = scaler_Y.transform(Y_val)

    key = jax.random.PRNGKey(42)
    in_size = X_train_s.shape[1]
    out_size = Y_train_s.shape[1]

    model = create_model(in_size, out_size, hidden_size, depth, key)

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
    )

    loss_plot_path = artifact_dir / "loss_curve.png"
    plot_training_curves(train_losses, val_losses, str(loss_plot_path))

    save_artifact(
        artifact,
        model,
        scaler_X,
        scaler_Y,
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
        },
    )
    print(f"Saved NN predictor artifact -> {artifact}")

    Y_pred_abs_flat, _mse = evaluate_model(model, scaler_Y, X_val_s, Y_val)

    plot_path = artifact_dir / "comparison.png"
    Y_pred_unflat = Y_pred_abs_flat.reshape(-1, horizon, C_y)
    plot_predictions(Y_val, Y_pred_unflat, split_idx, dt_real, horizon, C_y, str(plot_path))


if __name__ == "__main__":
    main()
