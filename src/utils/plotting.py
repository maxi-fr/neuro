"""Plotting functions for brain activity and EEG signals."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from .processing import compute_fft, compute_psd

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


def plot_fourier(  # noqa: C901, PLR0913, PLR0912
    signals: FloatArray,
    dt_ms: float,
    *,
    mode: str = "amplitude",
    channel_names: list[str] | None = None,
    channels_to_plot: list[int] | None = None,
    plot_mean: bool = False,
    max_freq: float | None = None,
    normalize: bool = False,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot the Fourier decomposition / spectrum of multi-channel signals.

    Parameters
    ----------
    signals
        Input signals, shape (n_channels, n_samples).
    dt_ms
        Sampling interval (integration step) in milliseconds.
    mode
        Type of spectrum to plot: "amplitude" (FFT amplitude) or "power" (Welch PSD).
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
    if mode == "amplitude":
        freqs, vals = compute_fft(signals, dt_ms)
        ylabel = "Amplitude"
        title = "FFT Amplitude Spectrum"
    elif mode == "power":
        freqs, vals = compute_psd(signals, dt_ms)
        ylabel = "Power (V²/Hz)"
        title = "Power Spectral Density (Welch)"
    else:
        msg = f"Unknown mode: {mode!r}. Expected 'amplitude' or 'power'"
        raise ValueError(msg)

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


def plot_dashboard(  # noqa: PLR0913
    activity: FloatArray,
    eeg: FloatArray,
    dt_ms: float,
    *,
    nodes_to_plot: list[int] | None = None,
    sensors_to_plot: list[int] | None = None,
    max_freq: float = 50.0,
    title: str = "Whole-Brain Simulation Dashboard",
) -> Figure:
    """Create a unified dashboard showing activity and EEG signals in time & frequency domains.

    Parameters
    ----------
    activity
        Brain region activity, shape (n_nodes, n_samples).
    eeg
        EEG sensor projection, shape (n_sensors, n_samples).
    dt_ms
        Integration step (sampling interval) in milliseconds.
    nodes_to_plot
        Optional list of node indices to plot in time domain. Defaults to first 5 nodes.
    sensors_to_plot
        Optional list of EEG sensor indices to plot. Defaults to first 5 sensors.
    max_freq
        Maximum frequency (Hz) to plot in the spectrum panel.
    title
        Overall title of the dashboard.

    Returns
    -------
    fig
        The generated matplotlib Figure.
    """
    if nodes_to_plot is None:
        nodes_to_plot = list(range(min(5, activity.shape[0])))

    if sensors_to_plot is None:
        sensors_to_plot = list(range(min(5, eeg.shape[0])))

    # Create figure with 2x2 layout
    fig = plt.figure(figsize=(14, 9), layout="constrained")
    gs = fig.add_gridspec(2, 2)

    ax_act_time = fig.add_subplot(gs[0, 0])
    ax_eeg_time = fig.add_subplot(gs[0, 1])
    ax_act_freq = fig.add_subplot(gs[1, 0])
    ax_eeg_freq = fig.add_subplot(gs[1, 1])

    # Plot Time-Domain Activity
    plot_signals(
        activity,
        dt_ms=dt_ms,
        channels_to_plot=nodes_to_plot,
        channel_names=[f"Node {i}" for i in range(activity.shape[0])],
        stacked=True,
        title="Brain Region Activity (V)",
        color="#1f77b4",
        ax=ax_act_time,
    )

    # Plot Time-Domain EEG
    plot_signals(
        eeg,
        dt_ms=dt_ms,
        channels_to_plot=sensors_to_plot,
        channel_names=[f"EEG {i}" for i in range(eeg.shape[0])],
        stacked=True,
        title="EEG Projection (Leadfield @ V)",
        color="#ff7f0e",
        ax=ax_eeg_time,
    )

    # Plot Frequency-Domain Activity (Power Spectral Density)
    plot_fourier(
        activity,
        dt_ms=dt_ms,
        mode="power",
        channels_to_plot=nodes_to_plot,
        channel_names=[f"Node {i}" for i in range(activity.shape[0])],
        plot_mean=False,
        max_freq=max_freq,
        ax=ax_act_freq,
    )
    ax_act_freq.set_title("Brain Activity Power Spectrum (Welch)")

    # Plot Frequency-Domain EEG (Power Spectral Density)
    plot_fourier(
        eeg,
        dt_ms=dt_ms,
        mode="power",
        channels_to_plot=sensors_to_plot,
        channel_names=[f"EEG {i}" for i in range(eeg.shape[0])],
        plot_mean=False,
        max_freq=max_freq,
        ax=ax_eeg_freq,
    )
    ax_eeg_freq.set_title("EEG Power Spectrum (Welch)")

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    return fig
