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
    nperseg: int | None = None,
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
    if mode == "amplitude":
        freqs, vals = compute_fft(signals, dt_ms)
        ylabel = "Amplitude"
        title = "FFT Amplitude Spectrum"
    elif mode == "power":
        freqs, vals = compute_psd(signals, dt_ms, nperseg=nperseg)
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


def plot_bifurcation_1d(  # noqa: PLR0913
    values: FloatArray,
    min_vals: FloatArray,
    max_vals: FloatArray,
    freqs: FloatArray,
    *,
    param_label: str = "Parameter",
    title: str = "Bifurcation diagram",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot a single-node bifurcation diagram (activity envelope + frequency).

    Mirrors Fig. 1 of Chouzouris et al. / Grimbert-Faugeras: the shaded band is
    the min-max range of the steady-state activity as the swept parameter varies
    (a single line where the node is at a fixed point, a widening band once it
    oscillates), and the right axis shows the dominant oscillation frequency.

    Parameters
    ----------
    values
        Swept parameter values, shape (n_points,).
    min_vals, max_vals
        Minimum and maximum steady-state activity at each value, shape (n_points,).
    freqs
        Dominant frequency (Hz) at each value, shape (n_points,). NaNs (no
        oscillation) are simply not drawn.
    param_label
        Axis label for the swept parameter.
    title
        Plot title.
    ax
        Optional matplotlib Axes for the activity (left) axis. If None, a new
        figure is created.

    Returns
    -------
    fig
        The matplotlib Figure object.
    ax
        The activity (left) Axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    else:
        fig = cast("Figure", ax.figure)

    ax.fill_between(values, min_vals, max_vals, color="#1f77b4", alpha=0.25, label="activity range")
    ax.plot(values, max_vals, color="#1f77b4", linewidth=1.4)
    ax.plot(values, min_vals, color="#1f77b4", linewidth=1.4)
    ax.set_xlabel(param_label, fontsize=11, fontweight="bold")
    ax.set_ylabel("Activity $y_1 - y_2$ (mV)", color="#1f77b4", fontsize=11, fontweight="bold")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.spines["top"].set_visible(False)
    ax.grid(visible=True, linestyle="--", alpha=0.3)

    ax_freq = ax.twinx()
    ax_freq.plot(values, freqs, color="#d62728", linewidth=1.6, marker=".", label="dominant freq")
    ax_freq.set_ylabel("Dominant frequency (Hz)", color="#d62728", fontsize=11, fontweight="bold")
    ax_freq.tick_params(axis="y", labelcolor="#d62728")
    ax_freq.spines["top"].set_visible(False)

    return fig, ax


def plot_state_map(  # noqa: PLR0913
    mu_values: FloatArray,
    sigma_values: FloatArray,
    metric_grid: FloatArray,
    *,
    metric_label: str = "Synchronization R",
    title: str = "Network state-space map",
    cmap: str = "viridis",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot a 2-D network state-space map over background input and coupling.

    Mirrors Fig. 3(a) of Chouzouris et al.: each pixel summarises the network's
    behaviour (e.g. synchronization or dominant frequency) at one operating point
    in the (background input mu, coupling strength sigma) plane.

    Parameters
    ----------
    mu_values
        Background-input values along the x-axis, shape (n_mu,).
    sigma_values
        Coupling-strength values along the y-axis, shape (n_sigma,).
    metric_grid
        Metric to colour by, shape (n_sigma, n_mu) (rows indexed by sigma).
    metric_label
        Colourbar label.
    title
        Plot title.
    cmap
        Matplotlib colormap name.
    ax
        Optional matplotlib Axes. If None, a new figure is created.

    Returns
    -------
    fig
        The matplotlib Figure object.
    ax
        The matplotlib Axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 6), layout="constrained")
    else:
        fig = cast("Figure", ax.figure)

    mesh = ax.pcolormesh(mu_values, sigma_values, metric_grid, cmap=cmap, shading="auto")
    fig.colorbar(mesh, ax=ax, label=metric_label)
    ax.set_xlabel(r"Background input $\mu$", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"Coupling strength $\sigma$", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    return fig, ax


def plot_heatmap(
    signals: FloatArray,
    dt_ms: float,
    *,
    title: str = "Signals Heatmap",
    cmap: str = "RdBu_r",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot multi-channel signals as a heatmap over time.

    Parameters
    ----------
    signals
        Input signals, shape (n_channels, n_samples).
    dt_ms
        Sampling interval (integration step) in milliseconds.
    title
        Title of the plot.
    cmap
        Colormap name.
    ax
        Optional matplotlib Axes to plot into. If None, a new figure is created.

    Returns
    -------
    fig
        The matplotlib Figure object.
    ax
        The matplotlib Axes object.
    """
    if signals.ndim != 2:  # noqa: PLR2004
        msg = f"Expected 2-D array of shape (n_channels, n_samples), got shape {signals.shape}"
        raise ValueError(msg)

    n_channels, n_samples = signals.shape
    time_sec = np.arange(n_samples) * dt_ms / 1000.0

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    else:
        fig = cast("Figure", ax.figure)

    vlim = float(np.max(np.abs(signals)))
    if vlim == 0:
        vlim = 1.0
    image = ax.imshow(
        signals,
        aspect="auto",
        origin="lower",
        extent=(float(time_sec[0]), float(time_sec[-1]), 0.0, float(n_channels)),
        cmap=cmap,
        vmin=-vlim,
        vmax=vlim,
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Time (s)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Channel", fontsize=11, fontweight="bold")
    fig.colorbar(image, ax=ax, label="Amplitude")

    return fig, ax
