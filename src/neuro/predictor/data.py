from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from torch import Tensor

from neuro.filtering import antialias_filter, lowpass_filter
from neuro.spectral import compute_log_power_frames
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from neuro.config import StftGeometry
    from neuro.types import Float32Array, FloatArray, IntArray


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
        The control current trajectory, shape ``(T, n_controls)``.
    y_data : FloatArray
        The measured output (EEG) trajectory, shape ``(T, n_channels)``.
    """
    with np.load(data_file) as data:
        max_idx = None if n_steps is None else n_steps * downsample
        y_full = np.asarray(data["sensor_0.y_mea"][:max_idx], dtype=np.float64)
        u_data = np.asarray(data["controller.u"][:max_idx:downsample], dtype=np.float64)

    if cutoff_hz is not None:
        y_filtered = lowpass_filter(y_full, 1.0 / dt, cutoff_hz)
    else:
        y_filtered = antialias_filter(y_full, 1.0 / dt, downsample)
    y_data = np.asarray(y_filtered[::downsample], dtype=np.float64)

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


def _window_starts(T_src: int, n_y: int, n_u: int, N: int) -> IntArray:
    """Sample indices at which a length-``N`` future window may start: history ahead, horizon fits."""
    start_idx = max(n_y - 1, n_u)
    end_idx = T_src - N
    return np.arange(start_idx, end_idx)


def build_dataset_for_trajectory(
    u_data: FloatArray, y_data: FloatArray, n_y: int, n_u: int, N: int
) -> tuple[Float32Array, Float32Array]:
    """Build the input/output pairs for the multi-step predictor.

    Parameters
    ----------
    u_data : FloatArray
        The Control Current trajectory of shape (T, n_controls).
    y_data : FloatArray
        The measured output (EEG) trajectory of shape (T, n_channels).
    n_y : int
        Number of past output steps to include in the input regression window.
    n_u : int
        Number of past input steps to include in the input regression window.
    N : int
        Prediction horizon (number of future steps to predict).

    Returns
    -------
    X : Float32Array
        Input features array of shape (samples, n_y * n_channels + n_u * n_controls + N * n_controls).
    Y : Float32Array
        Target labels array of shape (samples, N * n_channels).
    """
    k = _window_starts(y_data.shape[0], n_y, n_u, N)

    y_view = extract_windows_flattened(y_data, n_y)
    u_past_view = extract_windows_flattened(u_data, n_u)
    u_future_view = extract_windows_flattened(u_data, N)

    X = np.ascontiguousarray(
        np.concatenate([y_view[k - n_y + 1], u_past_view[k - n_u], u_future_view[k]], axis=1),
        dtype=np.float32,
    )

    y_fut_view = extract_windows_flattened(y_data, N)
    Y = np.ascontiguousarray(y_fut_view[k + 1], dtype=np.float32)

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


class TrajectoryWindowDataset(torch.utils.data.Dataset[tuple[Tensor, Tensor, Tensor, Tensor]]):
    """On-the-fly sliding window dataset over standardized continuous trajectories.

    Parameters
    ----------
    trajectories : list[tuple[FloatArray, FloatArray]]
        List of standardized (u, y) continuous trajectory pairs.
    n_y : int
        Number of past output steps in the history window.
    n_u : int
        Number of past control steps in the history window.
    horizon : int
        Prediction horizon (number of future steps).
    """

    def __init__(
        self,
        trajectories: list[tuple[FloatArray, FloatArray]],
        *,
        n_y: int,
        n_u: int,
        horizon: int,
    ) -> None:
        self.n_y = n_y
        self.n_u = n_u
        self.horizon = horizon
        self.trajectories: list[tuple[Tensor, Tensor]] = [
            (
                torch.as_tensor(np.ascontiguousarray(u), dtype=torch.float32),
                torch.as_tensor(np.ascontiguousarray(y), dtype=torch.float32),
            )
            for u, y in trajectories
        ]

        index_pairs: list[tuple[int, int]] = []
        start_offset = max(n_y - 1, n_u)
        for traj_idx, (_, y_t) in enumerate(self.trajectories):
            t_len = y_t.shape[0]
            end_offset = t_len - horizon
            if end_offset > start_offset:
                index_pairs.extend((traj_idx, k) for k in range(start_offset, end_offset))

        if index_pairs:
            self._index_map = np.asarray(index_pairs, dtype=np.int32)
        else:
            self._index_map = np.empty((0, 2), dtype=np.int32)

    def __len__(self) -> int:
        """Total number of valid sliding windows across all trajectories."""
        return len(self._index_map)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Extract (y_hist, u_hist, u_future, y_future) for the window indexed by index.

        Returns
        -------
        y_hist : Tensor
            Past outputs of shape ``(n_y, n_outputs)``.
        u_hist : Tensor
            Past controls of shape ``(n_u, n_controls)``.
        u_future : Tensor
            Future controls of shape ``(horizon, n_controls)``.
        y_future : Tensor
            Future output targets of shape ``(horizon, n_outputs)``.
        """
        traj_idx, k = self._index_map[index]
        u, y = self.trajectories[traj_idx]
        y_hist = y[k - self.n_y + 1 : k + 1]
        u_hist = u[k - self.n_u : k]
        u_future = u[k : k + self.horizon]
        y_future = y[k + 1 : k + 1 + self.horizon]
        return y_hist, u_hist, u_future, y_future


@dataclass(frozen=True)
class Datasets:
    """Standardized train/validation datasets, fitted standardizers, and raw trajectories.

    Attributes
    ----------
    train_dataset : TrajectoryWindowDataset
        On-the-fly PyTorch Dataset of standardized training windows.
    val_dataset : TrajectoryWindowDataset
        On-the-fly PyTorch Dataset of standardized validation windows.
    y_std, u_std : Standardizer
        Fitted channel and control standardizers.
    train_trajs, val_trajs : list[tuple[FloatArray, FloatArray]]
        The training-side and held-out ``(u, y)`` trajectories in raw units, kept whole so the
        Ridge arms can fit readouts raw-direct and free-run rollouts can be scored on them.
    n_channels : int
        Number of raw EEG output channels.
    n_controls : int
        Number of control input channels.
    """

    train_dataset: TrajectoryWindowDataset
    val_dataset: TrajectoryWindowDataset
    y_std: Standardizer
    u_std: Standardizer
    train_trajs: list[tuple[FloatArray, FloatArray]]
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
    """Split ``data_files`` by trajectory, standardize, and build on-the-fly window datasets."""
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

    train_dataset = TrajectoryWindowDataset(
        [(u_std.transform(u), y_std.transform(y)) for u, y in split.train_trajs],
        n_y=n_y,
        n_u=n_u,
        horizon=horizon,
    )
    val_dataset = TrajectoryWindowDataset(
        [(u_std.transform(u), y_std.transform(y)) for u, y in split.val_trajs],
        n_y=n_y,
        n_u=n_u,
        horizon=horizon,
    )

    return Datasets(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        y_std=y_std,
        u_std=u_std,
        train_trajs=split.train_trajs,
        val_trajs=split.val_trajs,
        n_channels=split.n_channels,
        n_controls=split.n_controls,
    )


def reduce_trajectory_to_frames(y: FloatArray, geometry: StftGeometry, fs: float, offset: int = 0) -> FloatArray:
    """Reduce Raw EEG trajectory to flattened log-power Frames with optional sub-hop sample offset.

    Parameters
    ----------
    y : FloatArray
        Raw EEG array of shape ``(n_samples, n_channels)``.
    geometry : StftGeometry
        Observable STFT geometry defining Segment length, hop, band, pooling, and Frame Kernel.
    fs : float
        Sampling frequency in Hz.
    offset : int, default=0
        Sub-hop sample offset into the trajectory.

    Returns
    -------
    FloatArray
        Flattened log-power Frames of shape ``(n_frames, n_channels * n_values)``.
    """
    frames = compute_log_power_frames(y[offset:], geometry, fs=fs)
    n_frames, n_channels, n_values = frames.shape
    return np.asarray(frames.reshape(n_frames, n_channels * n_values), dtype=np.float64)


def frame_aligned_controls(u: FloatArray, geometry: StftGeometry, *, fs: float, offset: int = 0) -> FloatArray:
    """Pick the control held when each Frame is emitted with optional sub-hop offset.

    Parameters
    ----------
    u : FloatArray
        Decimated control of shape ``(n_steps, n_controls)``.
    geometry : StftGeometry
        Observable STFT geometry defining Segment length, hop, band, pooling, and Frame Kernel.
    fs : float
        Sampling frequency in Hz.
    offset : int, default=0
        Sub-hop sample offset into the trajectory.

    Returns
    -------
    FloatArray
        One control per Frame, shape ``(n_frames, n_controls)``.
    """
    return np.asarray(u[offset + geometry.sample_support_steps(fs) - 1 :: geometry.n_hop], dtype=np.float64)


def prepare_observable_datasets(  # noqa: PLR0913, PLR0917 -- one flat call from train(); a params object would only relay
    data_files: list[str],
    n_steps_cfg: int | None,
    downsample: int,
    n_y: int,
    n_u: int,
    horizon: int,
    dt: float,
    train_split: float,
    geometry: StftGeometry,
    *,
    scaler: Literal["standard", "robust"],
    global_scaling: bool,
    cutoff_hz: float | None = None,
    subhop_offsets: bool = True,
) -> Datasets:
    """Split ``data_files`` by trajectory, reduce to Frames, standardize, and build sliding windows on the Frame grid.

    Parameters
    ----------
    data_files : list[str]
        Paths to the ``.npz`` trajectory files.
    n_steps_cfg : int | None
        Number of steps to load per trajectory.
    downsample : int
        Decimation factor applied to raw simulation trajectories.
    n_y : int
        Number of past Frames in history window.
    n_u : int
        Number of past Control Currents in history window.
    horizon : int
        Number of future Frames to predict.
    dt : float
        Plant sampling step in seconds.
    train_split : float
        Fraction of trajectories held for training.
    geometry : StftGeometry
        Observable STFT geometry defining the Frame reduction.
    scaler : Literal["standard", "robust"]
        Standardizer scaling algorithm.
    global_scaling : bool
        Whether scale is shared across outputs.
    cutoff_hz : float | None, optional
        Explicit lowpass filter cutoff frequency in Hz.
    subhop_offsets : bool, default=True
        Whether to expand training trajectories across all sub-hop offsets.

    Returns
    -------
    Datasets
        Standardized train/validation Frame datasets and fitted standardizers.
    """
    train_files, val_files = split_data_files(data_files, train_split)
    fs = 1.0 / (dt * downsample)
    train_offsets = list(range(geometry.n_hop)) if subhop_offsets else [0]

    def load_frames(files: list[str], offsets: list[int]) -> list[tuple[FloatArray, FloatArray]]:
        trajs: list[tuple[FloatArray, FloatArray]] = []
        for f in files:
            u_raw, y_raw = load_trajectory(f, n_steps_cfg, downsample, dt, cutoff_hz=cutoff_hz)
            for offset in offsets:
                y_frames = reduce_trajectory_to_frames(y_raw, geometry, fs, offset=offset)
                u_frames = frame_aligned_controls(u_raw, geometry, fs=fs, offset=offset)[: y_frames.shape[0]]
                min_len = min(y_frames.shape[0], u_frames.shape[0])
                trajs.append((u_frames[:min_len], y_frames[:min_len]))
        return trajs

    train_trajs = load_frames(train_files, train_offsets)
    val_trajs = load_frames(val_files, [0])

    all_y_train = np.concatenate([y for _, y in train_trajs], axis=0)
    all_u_train = np.concatenate([u for u, _ in train_trajs], axis=0)

    y_std = Standardizer.fit(all_y_train, kind=scaler, global_scaling=global_scaling)
    u_std = Standardizer.fit(all_u_train, kind=scaler, global_scaling=global_scaling)

    train_dataset = TrajectoryWindowDataset(
        [(u_std.transform(u), y_std.transform(y)) for u, y in train_trajs],
        n_y=n_y,
        n_u=n_u,
        horizon=horizon,
    )
    val_dataset = TrajectoryWindowDataset(
        [(u_std.transform(u), y_std.transform(y)) for u, y in val_trajs],
        n_y=n_y,
        n_u=n_u,
        horizon=horizon,
    )

    with np.load(data_files[0]) as data:
        n_channels = int(data["sensor_0.y_mea"].shape[1])
    n_controls = all_u_train.shape[1]

    return Datasets(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        y_std=y_std,
        u_std=u_std,
        train_trajs=train_trajs,
        val_trajs=val_trajs,
        n_channels=n_channels,
        n_controls=n_controls,
    )
