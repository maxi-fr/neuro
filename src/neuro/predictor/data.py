from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from neuro.filtering import antialias_filter, lowpass_filter
from neuro.transforms import Standardizer

if TYPE_CHECKING:
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


@dataclass(frozen=True)
class TrajectorySplit:
    """Loaded train/validation trajectories in raw units, plus the standardizers fitted on the training split.

    Attributes
    ----------
    train_trajs, val_trajs : list[tuple[FloatArray, FloatArray]]
        The ``(u, y)`` trajectories on each side of the split, kept whole.
    y_std, u_std : Standardizer
        Channel and control standardizers, fitted on the concatenated training trajectories.
    """

    train_trajs: list[tuple[FloatArray, FloatArray]]
    val_trajs: list[tuple[FloatArray, FloatArray]]
    y_std: Standardizer
    u_std: Standardizer

    @property
    def n_channels(self) -> int:
        """Number of raw EEG channels."""
        return self.train_trajs[0][1].shape[1]

    @property
    def n_controls(self) -> int:
        """Number of control input channels."""
        return self.train_trajs[0][0].shape[1]


def fit_standardizers(  # noqa: PLR0913
    data_files: list[str],
    *,
    n_steps_cfg: int | None,
    downsample: int,
    dt: float,
    train_split: float,
    scaler: Literal["standard", "robust"],
    global_scaling: bool,
    cutoff_hz: float | None = None,
) -> TrajectorySplit:
    """Split ``data_files`` by trajectory, load both sides, and fit the y/u standardizers on the training split."""
    train_files, val_files = split_data_files(data_files, train_split)

    def load(files: list[str]) -> list[tuple[FloatArray, FloatArray]]:
        return [load_trajectory(f, n_steps_cfg, downsample, dt, cutoff_hz=cutoff_hz) for f in files]

    train_trajs = load(train_files)
    all_y_train = np.concatenate([y for _, y in train_trajs], axis=0)
    all_u_train = np.concatenate([u for u, _ in train_trajs], axis=0)

    return TrajectorySplit(
        train_trajs=train_trajs,
        val_trajs=load(val_files),
        y_std=Standardizer.fit(all_y_train, kind=scaler, global_scaling=global_scaling),
        u_std=Standardizer.fit(all_u_train, kind=scaler, global_scaling=global_scaling),
    )


@dataclass(frozen=True)
class Datasets:
    """Standardized train/validation windows, fitted standardizers, and raw validation trajectories.

    Attributes
    ----------
    X_train, X_val : FloatArray
        Standardized input features, shape ``(samples, n_y * n_channels + (n_u + horizon) * n_controls)``.
    Y_train, Y_val : FloatArray
        Standardized target labels, shape ``(samples, horizon * n_channels)``.
    y_std, u_std : Standardizer
        Fitted channel and control standardizers.
    val_trajs : list[tuple[FloatArray, FloatArray]]
        The held-out ``(u, y)`` trajectories in raw units, kept whole so free-run rollouts can be scored on them.
    n_channels : int
        Number of raw EEG output channels.
    n_controls : int
        Number of control input channels.
    u_max : float
        Largest absolute current the training split holds, per electrode.
    """

    X_train: FloatArray
    Y_train: FloatArray
    X_val: FloatArray
    Y_val: FloatArray
    y_std: Standardizer
    u_std: Standardizer
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
    *,
    scaler: Literal["standard", "robust"],
    global_scaling: bool,
    cutoff_hz: float | None = None,
) -> Datasets:
    """Split ``data_files`` by trajectory, standardize, and build sliding windows."""
    split = fit_standardizers(
        data_files,
        n_steps_cfg=n_steps_cfg,
        downsample=downsample,
        dt=dt,
        train_split=train_split,
        scaler=scaler,
        global_scaling=global_scaling,
        cutoff_hz=cutoff_hz,
    )
    y_std, u_std = split.y_std, split.u_std

    def windows(trajs: list[tuple[FloatArray, FloatArray]]) -> tuple[FloatArray, FloatArray]:
        pairs = [
            build_dataset_for_trajectory(u_std.transform(u), y_std.transform(y), n_y, n_u, horizon) for u, y in trajs
        ]
        return np.concatenate([x for x, _ in pairs], axis=0), np.concatenate([y for _, y in pairs], axis=0)

    X_train, Y_train = windows(split.train_trajs)
    X_val, Y_val = windows(split.val_trajs)

    n_channels = split.n_channels
    n_controls = (X_train.shape[1] - n_y * n_channels) // (n_u + horizon)

    return Datasets(
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        y_std=y_std,
        u_std=u_std,
        val_trajs=split.val_trajs,
        n_channels=n_channels,
        n_controls=n_controls,
    )
