"""Plotting functions for brain activity and EEG signals."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from .processing import compute_psd

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

FloatArray = npt.NDArray[np.float64]


def _filter_channels(
    signals: FloatArray,
    channel_names: list[str] | None,
    channels_to_plot: list[int] | None,
) -> tuple[FloatArray, list[str]]:
    """Select a subset of channels and their corresponding names."""
    n_channels = signals.shape[0]
    indices = channels_to_plot if channels_to_plot is not None else list(range(n_channels))
    names = channel_names if channel_names is not None else [f"Ch {i}" for i in range(n_channels)]
    selected_signals = signals[indices, :]
    selected_names = [names[i] for i in indices]
    return selected_signals, selected_names


def plot_signals(  # noqa: PLR0913
    signals: FloatArray,
    dt_ms: float,
    *,
    channel_names: list[str] | None = None,
    channels_to_plot: list[int] | None = None,
    stacked: bool = True,
    offset_scale: float = 1.5,
    title: str = "Signals over Time",
    color: str | list[str] = "#1f77b4",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot multi-channel signals (EEG or activity) over time.

    Parameters
    ----------
    signals
        Input signals, shape (n_channels, n_samples).
    dt_ms
        Sampling interval (integration step) in milliseconds.
    channel_names
        Optional list of channel names (length must match n_channels).
    channels_to_plot
        Optional list of indices of channels to plot. If None, plots all channels.
    stacked
        If True, plots channels vertically stacked (waterfall style).
        If False, plots all channels overlaid on the same axis.
    offset_scale
        Spacing multiplier between stacked channels. Ignored if stacked is False.
    title
        Title of the plot.
    color
        Color or list of colors for the channels.
    ax
        Optional matplotlib Axes to plot into. If None, a new figure is created.

    Returns
    -------
    fig
        The matplotlib Figure object.
    ax
        The matplotlib Axes object.

    Raises
    ------
    ValueError
        If signals is not a 2-D array.
    """
    if signals.ndim != 2:  # noqa: PLR2004
        msg = f"Expected 2-D array of shape (n_channels, n_samples), got shape {signals.shape}"
        raise ValueError(msg)

    n_samples = signals.shape[1]
    time_ms = np.arange(n_samples) * dt_ms
    time_sec = time_ms / 1000.0  # Convert to seconds for plotting

    selected_signals, selected_names = _filter_channels(signals, channel_names, channels_to_plot)
    num_to_plot = len(selected_names)

    if ax is None:
        # Determine figure height based on the number of plotted channels
        fig_height = max(4.0, num_to_plot * 0.5 if stacked else 5.0)
        fig, ax = plt.subplots(figsize=(10, fig_height), layout="constrained")
    else:
        fig = cast("Figure", ax.figure)

    # Handle colors
    if isinstance(color, str):
        colors = [color] * num_to_plot
    else:
        colors = list(color)
        while len(colors) < num_to_plot:
            colors.append("#1f77b4")

    if stacked:
        # Stacked / waterfall plot
        p2p = np.ptp(selected_signals, axis=1)
        mean_p2p = np.mean(p2p) if np.mean(p2p) > 0 else 1.0
        offset = mean_p2p * offset_scale

        yticks = []
        for idx, (sig, col) in enumerate(zip(selected_signals, colors, strict=False)):
            y_offset = (num_to_plot - 1 - idx) * offset
            ax.plot(time_sec, sig + y_offset, color=col, alpha=0.85, linewidth=1.2)
            yticks.append(y_offset)

        ax.set_yticks(yticks)
        ax.set_yticklabels(selected_names)
        ax.set_ylim(-offset, num_to_plot * offset)
    else:
        # Overlaid plot
        for sig, name, col in zip(selected_signals, selected_names, colors, strict=False):
            ax.plot(time_sec, sig, label=name, color=col, alpha=0.75, linewidth=1.2)
        ax.legend(loc="upper right", framealpha=0.9)

    ax.set_xlabel("Time (s)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Amplitude", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    # Styling cleanups
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(visible=True, linestyle="--", alpha=0.3)

    return fig, ax


def plot_psd(  # noqa: PLR0913
    signals: FloatArray,
    dt_ms: float,
    *,
    channel_names: list[str] | None = None,
    channels_to_plot: list[int] | None = None,
    plot_mean: bool = False,
    max_freq: float | None = None,
    normalize: bool = False,
    nperseg: int | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot the spectrum of multi-channel signals.

    Parameters
    ----------
    signals
        Input signals, shape (n_channels, n_samples).
    dt_ms
        Sampling interval (integration step) in milliseconds.
    channel_names
        Optional list of channel names (length must match n_channels).
    channels_to_plot
        Optional list of indices of channels to plot. If None, plots all channels.
    plot_mean
        If True, plots the mean and standard deviation (shaded area) across the
        selected channels, instead of plotting individual lines.
    max_freq
        Maximum frequency (Hz) to show on the x-axis.
    normalize
        If True, divide each channel's spectrum by its own total so the curves
        compare spectral *shape* rather than absolute magnitude. Useful when
        signals differ in overall amplitude (e.g. different plants, noise levels,
        or lead-field scales).
    nperseg
        Welch segment length (``mode="power"`` only). Larger values give finer
        frequency resolution; needed at small ``dt`` to resolve EEG bands.
        Ignored for ``mode="amplitude"``.
    ax
        Optional matplotlib Axes to plot into. If None, a new figure is created.

    Returns
    -------
    fig
        The matplotlib Figure object.
    ax
        The matplotlib Axes object.

    Raises
    ------
    ValueError
        If signals is not a 2-D array or if mode is unknown.
    """
    freqs, vals = compute_psd(signals, dt_ms, nperseg=nperseg)
    ylabel = "Power (V²/Hz)"
    title = "Power Spectral Density (Welch)"

    if normalize and vals.size:
        totals = vals.sum(axis=1, keepdims=True)
        vals = np.divide(vals, totals, out=np.zeros_like(vals), where=totals > 0)
        ylabel = f"Normalized {ylabel}"
        title = f"Normalized {title}"

    if freqs.size == 0:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        else:
            fig = cast("Figure", ax.figure)
        ax.set_title("No data to plot")
        return fig, ax

    selected_vals, selected_names = _filter_channels(vals, channel_names, channels_to_plot)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    else:
        fig = cast("Figure", ax.figure)

    if plot_mean:
        # Plot mean and std shading
        mean_val = np.mean(selected_vals, axis=0)
        std_val = np.std(selected_vals, axis=0)
        ax.plot(freqs, mean_val, color="#2ca02c", label="Mean", linewidth=2.0)
        ax.fill_between(
            freqs,
            np.maximum(0.0, mean_val - std_val),
            mean_val + std_val,
            color="#2ca02c",
            alpha=0.2,
            label="± 1 SD",
        )
        ax.legend(loc="upper right")
    else:
        # Plot individual channels
        for val, name in zip(selected_vals, selected_names, strict=False):
            ax.plot(freqs, val, label=name, alpha=0.6, linewidth=1.0)
        if len(selected_names) <= 10:  # noqa: PLR2004
            ax.legend(loc="upper right", framealpha=0.9)

    ax.set_xlabel("Frequency (Hz)", fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    if max_freq is not None:
        ax.set_xlim(0, max_freq)
    else:
        ax.set_xlim(0, freqs[-1])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(visible=True, linestyle="--", alpha=0.3)

    return fig, ax
