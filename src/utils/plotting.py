"""Plotting functions for brain activity and EEG signals."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import matplotlib.pyplot as plt
import numpy as np

from .processing import compute_psd

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from neuro.types import FloatArray, IntArray


_DEFAULT_N_FANS = 10


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
    signals : FloatArray
        Input signals, shape ``(n_channels, n_samples)``.
    dt_ms : float
        Sampling interval (integration step) in milliseconds.
    channel_names : list of str, optional
        Optional list of channel names (length must match n_channels).
    channels_to_plot : list of int, optional
        Optional list of indices of channels to plot. If None, plots all channels.
    stacked : bool, default=True
        If True, plots channels vertically stacked (waterfall style).
        If False, plots all channels overlaid on the same axis.
    offset_scale : float, default=1.5
        Spacing multiplier between stacked channels. Ignored if stacked is False.
    title : str, default="Signals over Time"
        Title of the plot.
    color : str or list of str, default="#1f77b4"
        Color or list of colors for the channels.
    ax : matplotlib.axes.Axes, optional
        Optional matplotlib Axes to plot into. If None, a new figure is created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object.
    ax : matplotlib.axes.Axes
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
            ax.plot(time_sec, sig, label=name, color=col, alpha=0.54, linewidth=1.2)
        ax.legend(loc="upper right", framealpha=0.9)

    ax.set_xlabel("Time (s)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Amplitude", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    # Styling cleanups
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(visible=True, linestyle="--", alpha=0.3)

    return fig, ax


def _resolve_anchors(  # noqa: PLR0913
    n_samples: int,
    dt: float,
    stride: int | None,
    anchors: Sequence[float] | None,
    *,
    anchors_in_seconds: bool,
    t_start: float,
    t_end: float,
) -> IntArray:
    """Resolve the sample indices at which a prediction fan is drawn.

    Explicit ``anchors`` take precedence over ``stride``. Anchors are clipped to
    the valid sample range and to the visible window ``[t_start, t_end]`` (compared
    against each anchor's own time ``i * dt``).
    """
    if anchors is not None:
        arr = np.asarray(anchors, dtype=np.float64)
        idx = np.rint(arr / dt).astype(np.intp) if anchors_in_seconds else arr.astype(np.intp)
    else:
        step = stride if stride is not None else max(1, n_samples // _DEFAULT_N_FANS)
        idx = np.arange(0, n_samples, step, dtype=np.intp)

    idx = idx[(idx >= 0) & (idx < n_samples)]
    times = idx * dt
    return idx[(times >= t_start) & (times <= t_end)]


def _draw_fans(  # noqa: PLR0913
    ax: Axes,
    y_true_c: FloatArray,
    y_pred_c: FloatArray,
    anchor_idx: IntArray,
    dt: float,
    *,
    connect_to_truth: bool,
    pred_style: dict[str, object],
    cmap_name: str | None,
    marker_style: dict[str, object] | None,
    add_labels: bool,
) -> None:
    """Draw every prediction fan for a single channel onto ``ax``.

    ``y_pred_c`` has shape ``(n_samples, horizon)``. Fan ``i`` spans times
    ``(i + 1) * dt ... (i + horizon) * dt`` (predictions start one step ahead).
    """
    horizon = y_pred_c.shape[1]
    cmap = plt.get_cmap(cmap_name) if cmap_name is not None else None
    denom = max(len(anchor_idx) - 1, 1)

    for k, i in enumerate(anchor_idx):
        pred_t = (np.arange(horizon) + i + 1) * dt
        pred_y = y_pred_c[i]
        if connect_to_truth:
            pred_t = np.concatenate(([i * dt], pred_t))
            pred_y = np.concatenate(([y_true_c[i]], pred_y))

        style: dict[str, Any] = dict(pred_style)
        if cmap is not None:
            style["color"] = cmap(k / denom)
        style["label"] = style.get("label", "Prediction") if (add_labels and k == 0) else None
        ax.plot(pred_t, pred_y, **style)

        if marker_style is not None:
            marker: dict[str, Any] = dict(marker_style)
            ax.plot(i * dt, y_true_c[i], **marker)


def plot_multistep_predictions(  # noqa: PLR0913
    y_true: FloatArray,
    y_pred: FloatArray,
    dt: float,
    *,
    channels: list[int] | None = None,
    channel_names: list[str] | None = None,
    stride: int | None = None,
    anchors: Sequence[float] | None = None,
    anchors_in_seconds: bool = False,
    t_start: float | None = None,
    t_end: float | None = None,
    connect_to_truth: bool = True,
    show_anchor_markers: bool = True,
    cmap: str | None = None,
    title: str = "Multistep predictions",
    ylabel: str = "Amplitude",
    true_kwargs: dict[str, object] | None = None,
    pred_kwargs: dict[str, object] | None = None,
    figsize: tuple[float, float] | None = None,
    axes: Sequence[Axes] | None = None,
) -> tuple[Figure, list[Axes]]:
    """Plot receding-horizon multistep predictions against the true signal.

    For each anchor time the model's ``horizon``-step forecast is drawn as a fan
    branching forward from the true trajectory (classic MPC-style overlay), with
    one vertically stacked subplot per channel.

    Parameters
    ----------
    y_true : FloatArray
        Ground-truth signal, shape ``(n_samples, n_channels)``.
    y_pred : FloatArray
        Multistep predictions, shape ``(n_samples, horizon, n_channels)``.
        ``y_pred[i, h, c]`` is the ``(h + 1)``-step-ahead prediction made at
        sample ``i``, i.e. it is plotted at time ``(i + 1 + h) * dt``.
    dt : float
        Sampling interval in **seconds** (the x-axis is shown in seconds).
    channels : list of int, optional
        Indices of the channels to plot. If None, all channels are plotted.
    channel_names : list of str, optional
        Names for *all* channels (length ``n_channels``); used as subplot labels.
    stride : int, optional
        Draw a prediction fan every ``stride`` samples. If None (and ``anchors``
        is not given), a stride giving ~``_DEFAULT_N_FANS`` fans is chosen.
    anchors : sequence of float, optional
        Explicit anchor positions overriding ``stride``. Interpreted as sample
        indices, or as seconds when ``anchors_in_seconds`` is True.
    anchors_in_seconds : bool, default=False
        If True, ``anchors`` are given in seconds rather than sample indices.
    t_start : float, optional
        Visible time window in seconds. Defaults to the full true trace.
    t_end : float, optional
        Visible time window in seconds. Defaults to the full true trace. Anchors
        outside the window are dropped and the x-axis is limited to it.
    connect_to_truth : bool, default=True
        If True, each fan starts from the true value at its anchor so it visibly
        branches off the trajectory.
    show_anchor_markers : bool, default=True
        If True, mark each anchor point on the true signal.
    cmap : str, optional
        Optional matplotlib colormap name; if given, fans are coloured along it by
        anchor order (otherwise all fans share ``pred_kwargs`` colour).
    title : str, default="Multistep predictions"
        Figure suptitle.
    ylabel : str, default="Amplitude"
        Y-axis label applied to every subplot.
    true_kwargs : dict, optional
        Style overrides forwarded to ``Axes.plot`` for the true line.
    pred_kwargs : dict, optional
        Style overrides forwarded to ``Axes.plot`` for the prediction fans.
    figsize : tuple of (float, float), optional
        Figure size. Defaults to a height that scales with the channel count.
    axes : sequence of Axes, optional
        Optional pre-created axes (one per selected channel) to draw into. If
        None, a new stacked figure is created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object.
    axes : list of matplotlib.axes.Axes
        List of per-channel Axes objects.

    Raises
    ------
    ValueError
        If array dimensions/shapes are inconsistent or ``dt <= 0``.
    """
    if y_true.ndim != 2:  # noqa: PLR2004
        msg = f"y_true must be 2-D (n_samples, n_channels), got shape {y_true.shape}"
        raise ValueError(msg)
    if y_pred.ndim != 3:  # noqa: PLR2004
        msg = f"y_pred must be 3-D (n_samples, horizon, n_channels), got shape {y_pred.shape}"
        raise ValueError(msg)
    if y_pred.shape[0] != y_true.shape[0] or y_pred.shape[2] != y_true.shape[1]:
        msg = (
            "y_true and y_pred shapes are inconsistent: "
            f"{y_true.shape} vs {y_pred.shape} (expected (S, C) and (S, H, C))"
        )
        raise ValueError(msg)
    if dt <= 0:
        msg = f"dt must be positive, got {dt}"
        raise ValueError(msg)

    n_samples, n_channels = y_true.shape
    indices = channels if channels is not None else list(range(n_channels))
    names = channel_names if channel_names is not None else [f"Ch {i}" for i in range(n_channels)]

    win_start = 0.0 if t_start is None else t_start
    win_end = (n_samples - 1) * dt if t_end is None else t_end
    anchor_idx = _resolve_anchors(
        n_samples,
        dt,
        stride,
        anchors,
        anchors_in_seconds=anchors_in_seconds,
        t_start=win_start,
        t_end=win_end,
    )

    true_style: dict[str, Any] = {"color": "#1f1f1f", "linewidth": 1.5, "alpha": 0.9, "zorder": 3}
    true_style.update(true_kwargs or {})
    pred_style: dict[str, Any] = {
        "color": "#d62728",
        "linewidth": 1.0,
        "alpha": 0.7,
        "linestyle": "--",
        "zorder": 2,
    }
    pred_style.update(pred_kwargs or {})
    marker_style: dict[str, Any] | None = (
        {"marker": "o", "markersize": 3.5, "color": "#333333", "linestyle": "none", "zorder": 4}
        if show_anchor_markers
        else None
    )

    n_plot = len(indices)
    if axes is None:
        height = figsize[1] if figsize is not None else max(2.2 * n_plot, 3.0)
        width = figsize[0] if figsize is not None else 11.0
        fig, axes_arr = plt.subplots(
            n_plot, 1, sharex=True, figsize=(width, height), squeeze=False, layout="constrained"
        )
        ax_list = list(axes_arr[:, 0])
    else:
        ax_list = list(axes)
        if len(ax_list) != n_plot:
            msg = f"axes has length {len(ax_list)} but {n_plot} channels were selected"
            raise ValueError(msg)
        fig = cast("Figure", ax_list[0].figure)

    time = np.arange(n_samples) * dt
    for j, (ax, c) in enumerate(zip(ax_list, indices, strict=True)):
        ax.plot(time, y_true[:, c], label="True" if j == 0 else None, **true_style)  # type: ignore[arg-type]
        _draw_fans(
            ax,
            y_true[:, c],
            y_pred[:, :, c],
            anchor_idx,
            dt,
            connect_to_truth=connect_to_truth,
            pred_style=pred_style,
            cmap_name=cmap,
            marker_style=marker_style,
            add_labels=(j == 0),
        )
        ax.set_ylabel(names[c], fontsize=10, fontweight="bold")
        ax.set_xlim(win_start, win_end)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(visible=True, linestyle="--", alpha=0.3)

    ax_list[-1].set_xlabel("Time (s)", fontsize=11, fontweight="bold")
    ax_list[0].legend(loc="upper right", framealpha=0.9)
    fig.supylabel(ylabel, fontsize=11, fontweight="bold")
    fig.suptitle(title, fontsize=13, fontweight="bold")

    return fig, ax_list


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
    signals : FloatArray
        Input signals, shape ``(n_channels, n_samples)``.
    dt_ms : float
        Sampling interval (integration step) in milliseconds.
    channel_names : list of str, optional
        Optional list of channel names (length must match n_channels).
    channels_to_plot : list of int, optional
        Optional list of indices of channels to plot. If None, plots all channels.
    plot_mean : bool, default=False
        If True, plots the mean and standard deviation (shaded area) across the
        selected channels, instead of plotting individual lines.
    max_freq : float, optional
        Maximum frequency (Hz) to show on the x-axis.
    normalize : bool, default=False
        If True, divide each channel's spectrum by its own total so the curves
        compare spectral *shape* rather than absolute magnitude. Useful when
        signals differ in overall amplitude (e.g. different plants, noise levels,
        or lead-field scales).
    nperseg : int, optional
        Welch segment length (``mode="power"`` only). Larger values give finer
        frequency resolution; needed at small ``dt`` to resolve EEG bands.
        Ignored for ``mode="amplitude"``.
    ax : matplotlib.axes.Axes, optional
        Optional matplotlib Axes to plot into. If None, a new figure is created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object.
    ax : matplotlib.axes.Axes
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
