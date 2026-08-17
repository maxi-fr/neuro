from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from neuro.filtering import antialias_filter, lowpass_filter

if TYPE_CHECKING:
    from neuro.transforms import Pipeline
    from neuro.types import FloatArray


def load_trajectory(
    data_file: str,
    n_steps: int | None,
    downsample: int,
    dt: float,
    cutoff_hz: float | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Load a single simulation trajectory and decimate it.

    The EEG is causally low-passed (at ``cutoff_hz`` if specified, or at the decimated Nyquist
    rate) before striding. The control is strided unfiltered.

    Parameters
    ----------
    data_file : str
        Path to the `.npz` data file containing the trajectory.
    n_steps : int | None
        The total number of time steps to load, or ``None`` to load the entire trajectory.
    downsample : int
        The downsampling factor to apply.
    dt : float
        Sample time of the stored trajectory, used to design the low-pass filter.
    cutoff_hz : float | None, optional
        Explicit -3 dB cutoff frequency in Hz. If ``None``, defaults to the decimated Nyquist rate.

    Returns
    -------
    u_data : FloatArray
        The stimulation input trajectory, shape ``(T, n_controls)``.
    y_data : FloatArray
        The measured output (EEG) trajectory, shape ``(T, n_channels)``.
    """
    with np.load(data_file) as data:
        max_idx = None if n_steps is None else n_steps * downsample
        y_full = np.asarray(data["sensor_0.y_mea"][:max_idx], dtype=np.float64)
        u_data = data["controller.u"][:max_idx:downsample]

    if cutoff_hz is not None:
        y_filtered = lowpass_filter(y_full, 1.0 / dt, cutoff_hz)
    else:
        y_filtered = antialias_filter(y_full, 1.0 / dt, downsample)
    y_data = y_filtered[::downsample]

    return u_data, y_data


def split_data_files(data_files: list[str], train_split: float) -> tuple[list[str], list[str]]:
    """Split trajectory files into train/validation lists, keeping at least one file on each side.

    Splitting by trajectory rather than by window keeps the validation set free of windows that
    overlap training windows, so free-run rollouts started there are genuinely held out.
    """
    n_train = min(max(int(len(data_files) * train_split), 1), len(data_files) - 1)
    train_files, val_files = data_files[:n_train], data_files[n_train:]
    if not train_files or not val_files:
        msg = f"need at least 2 trajectory files to hold one out for validation, got {len(data_files)}"
        raise ValueError(msg)
    return train_files, val_files


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
    """Apply a Pipeline to a flattened multi-step block.

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


def transform_features(  # noqa: PLR0913, PLR0917
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


@dataclass(frozen=True)
class Datasets:
    """Raw (unscaled) train/validation windows plus the validation trajectories they came from.

    Attributes
    ----------
    X_train, X_val : FloatArray
        Raw input features, shape ``(samples, n_y * n_channels + (n_u + horizon) * n_controls)``.
    Y_train, Y_val : FloatArray
        Raw target labels, shape ``(samples, horizon * n_channels)``.
    val_trajs : list[tuple[FloatArray, FloatArray]]
        The held-out ``(u, y)`` trajectories, kept whole so free-run rollouts can be scored on them.
    n_channels : int
        Number of **raw** EEG output channels (the windows carry no projection yet).
    n_controls : int
        Number of control input channels.
    """

    X_train: FloatArray
    Y_train: FloatArray
    X_val: FloatArray
    Y_val: FloatArray
    val_trajs: list[tuple[FloatArray, FloatArray]]
    n_channels: int
    n_controls: int


def prepare_datasets(  # noqa: PLR0913, PLR0917
    data_files: list[str],
    n_steps_cfg: int | None,
    downsample: int,
    n_y: int,
    n_u: int,
    horizon: int,
    dt: float,
    train_split: float,
    cutoff_hz: float | None = None,
) -> Datasets:
    """Split ``data_files`` by trajectory and build the raw (unscaled) windows on both sides.

    Windows are built in raw EEG/control units; the standardizer and optional PCA projection
    are fitted on the training split and applied downstream, so no transform is applied here.

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
    dt : float
        Sample time of the stored trajectories (the filter is designed at ``1 / dt``).
    train_split : float
        Fraction of ``data_files`` held for training; the tail is validation.
    cutoff_hz : float | None, optional
        Explicit -3 dB cutoff frequency in Hz. If ``None``, defaults to the decimated Nyquist rate.

    Returns
    -------
    Datasets
        Raw train/validation windows, the held-out trajectories, and the raw channel counts.
    """
    train_files, val_files = split_data_files(data_files, train_split)

    def windows(files: list[str]) -> tuple[list[tuple[FloatArray, FloatArray]], FloatArray, FloatArray]:
        trajs = [load_trajectory(f, n_steps_cfg, downsample, dt, cutoff_hz=cutoff_hz) for f in files]
        pairs = [build_dataset_for_trajectory(u, y, n_y, n_u, horizon) for u, y in trajs]
        return trajs, np.concatenate([x for x, _ in pairs], axis=0), np.concatenate([y for _, y in pairs], axis=0)

    train_trajs, X_train, Y_train = windows(train_files)
    val_trajs, X_val, Y_val = windows(val_files)

    n_channels = train_trajs[0][1].shape[1]
    n_controls = (X_train.shape[1] - n_y * n_channels) // (n_u + horizon)

    return Datasets(
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        val_trajs=val_trajs,
        n_channels=n_channels,
        n_controls=n_controls,
    )
