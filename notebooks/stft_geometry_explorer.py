import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full", app_title="STFT geometry explorer")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from matplotlib import pyplot as plt
    from pydantic import ValidationError

    from neuro.config import StftGeometry
    from neuro.connectome import Connectome
    from neuro.eeg import build_eeg_leadfield, focal_channels
    from neuro.filtering import antialias_filter
    from neuro.geometry import sensor_positions_mm
    from neuro.predictor.data import load_trajectory
    from neuro.seizure import EZ_REGIONS, spread_profile_from_lfp
    from neuro.spectral import compute_log_power_frames, compute_segment_window

    return (
        Connectome,
        EZ_REGIONS,
        StftGeometry,
        ValidationError,
        antialias_filter,
        build_eeg_leadfield,
        compute_log_power_frames,
        compute_segment_window,
        focal_channels,
        load_trajectory,
        mo,
        np,
        plt,
        sensor_positions_mm,
        spread_profile_from_lfp,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # STFT geometry explorer

    Inspect the exact log-power Frames emitted by `StftGeometry` for uncontrolled seizure
    simulations. Change the geometry, then look for the smoothing that makes the signal stable
    without hiding seizure onset or fast changes after it begins. This notebook does not load or
    run a predictor.
    """)
    return


@app.cell
def _(
    Connectome,
    EZ_REGIONS,
    antialias_filter,
    build_eeg_leadfield,
    focal_channels,
    load_trajectory,
    np,
    sensor_positions_mm,
    spread_profile_from_lfp,
):
    FS = 50.0
    DT = 1e-4
    DOWNSAMPLE = round(1.0 / (FS * DT))
    trajectory_paths = {"uncontrolled": "data/uncontrolled_2/log.npz"}
    leadfield, labels_array = build_eeg_leadfield()
    channel_labels = [str(label) for label in labels_array]
    sensor_labels, sensor_positions = sensor_positions_mm()
    position_by_label = {str(label): position for label, position in zip(sensor_labels, sensor_positions, strict=True)}
    left_eeg_labels = [label for label in channel_labels if position_by_label[label][1] > 0.0]
    connectome = Connectome.from_config({"speed": 50.0, "K": 0.60})
    focal_indices = focal_channels(leadfield, connectome.region_index[EZ_REGIONS[0]])
    focal_labels = [channel_labels[index] for index in focal_indices]
    region_labels = [str(label) for label in connectome.region_labels]
    focal_regions = [*EZ_REGIONS, "lTCI", "lTCV"]

    trajectories = {name: load_trajectory(path, None, DOWNSAMPLE, DT)[1] for name, path in trajectory_paths.items()}

    lfp_trajectories = {}
    seizure_profiles = {}
    for name, path in trajectory_paths.items():
        with np.load(path) as archive:
            lfp_raw = np.asarray(archive["dynamics.lfp"], dtype=np.float64)
        lfp_trajectories[name] = np.asarray(antialias_filter(lfp_raw, 1.0 / DT, DOWNSAMPLE)[::DOWNSAMPLE])
        seizure_profiles[name] = spread_profile_from_lfp(lfp_raw.T, DT)
    duration_s = {name: y.shape[0] / FS for name, y in trajectories.items()}
    return (
        FS,
        channel_labels,
        duration_s,
        focal_labels,
        focal_regions,
        left_eeg_labels,
        lfp_trajectories,
        region_labels,
        seizure_profiles,
        trajectories,
        trajectory_paths,
    )


@app.cell
def _(
    FS,
    channel_labels,
    duration_s,
    focal_labels,
    focal_regions,
    mo,
    region_labels,
    trajectory_paths,
):
    trajectory = mo.ui.dropdown(options=list(trajectory_paths), value="uncontrolled", label="Trajectory")
    start_s = mo.ui.slider(
        0.0,
        max(duration_s.values()) - 1.0 / FS,
        1.0 / FS,
        value=0.0,
        label="View start (s)",
    )
    duration = mo.ui.slider(1.0, min(20.0, max(duration_s.values())), 0.5, value=12.0, label="View duration (s)")
    channel_picker = mo.ui.multiselect(options=channel_labels, value=focal_labels, label="Channels")
    select_all = mo.ui.button(label="Select all")
    select_left = mo.ui.button(label="Select left EEG")
    select_none = mo.ui.button(label="Select none")
    lfp_picker = mo.ui.multiselect(options=region_labels, value=focal_regions, label="LFP regions")
    vertical_lines = mo.ui.text(value="", label="Vertical lines (s, comma-separated)")

    mo.vstack(
        [
            mo.hstack([trajectory, start_s, duration], justify="start", gap=2),
            mo.hstack([channel_picker, select_all, select_left, select_none], justify="start", gap=2),
            lfp_picker,
            vertical_lines,
        ]
    )
    return (
        channel_picker,
        duration,
        lfp_picker,
        select_all,
        select_none,
        start_s,
        trajectory,
        vertical_lines,
    )


@app.cell
def _(
    channel_labels,
    channel_picker,
    left_eeg_labels,
    select_all,
    select_none,
):
    def selected_channels() -> list[str]:
        """Resolve the multiselect value, with the latest all-or-none action taking precedence."""
        all_clicks = int(select_all.value or 0)
        left_clicks = 1000  # int(select_left.value or 0) # TODO: not working
        none_clicks = int(select_none.value or 0)
        if left_clicks > max(all_clicks, none_clicks):
            return left_eeg_labels
        if all_clicks > none_clicks:
            return channel_labels
        if none_clicks > all_clicks:
            return []
        return list(channel_picker.value)

    selected = left_eeg_labels  # selected_channels()
    return (selected,)


@app.cell
def _(mo, vertical_lines):
    marker_times: tuple[float, ...] = ()
    marker_error = None
    try:
        marker_times = tuple(float(value.strip()) for value in vertical_lines.value.split(",") if value.strip())
    except ValueError:
        marker_error = "Vertical lines must be comma-separated times in seconds."
    mo.stop(marker_error is not None, mo.callout(marker_error or "", kind="danger"))
    marker_lines = tuple((time, f"C{index % 10}") for index, time in enumerate(marker_times))
    return (marker_lines,)


@app.cell
def _(mo):
    segment = mo.ui.slider(0.02, 3.0, 0.02, value=1.0, label="`n_segment` (s)")
    hop = mo.ui.slider(0.02, 3.0, 0.02, value=0.10, label="`n_hop` (s)")
    window = mo.ui.dropdown(
        options=["hann", "hann_poisson"],
        value="hann",
        label="`window`",
    )
    asymmetric_window = mo.ui.checkbox(value=False, label="`asymmetric_window`")
    use_band = mo.ui.checkbox(value=True, label="Apply `band_hz`")
    band = mo.ui.range_slider(0.0, 25.0, 0.5, value=[1.0, 25.0], label="`band_hz` (Hz)")
    bin_pool = mo.ui.slider(1, 8, 1, value=1, label="`n_bin_pool`")
    kernel = mo.ui.dropdown(
        options=["boxcar", "triangular", "hann", "exponential", "linear"], value="boxcar", label="`kernel`"
    )

    kernel_width = mo.ui.slider(1, 15, 1, value=1, label="`kernel_width`")

    mo.hstack(
        [
            mo.vstack([mo.md("#### Segment grid"), segment, hop, window, asymmetric_window]),
            mo.vstack([mo.md("#### Frequency bins"), use_band, band, bin_pool]),
            mo.vstack([mo.md("#### Frame smoothing"), kernel, kernel_width]),
        ],
        justify="space-between",
        gap=3,
    )
    return (
        asymmetric_window,
        band,
        bin_pool,
        hop,
        kernel,
        kernel_width,
        segment,
        use_band,
        window,
    )


@app.cell
def _(
    FS,
    StftGeometry,
    ValidationError,
    asymmetric_window,
    band,
    bin_pool,
    hop,
    kernel,
    kernel_width,
    mo,
    segment,
    use_band,
    window,
):
    geometry, geometry_error = None, None

    try:
        geometry = StftGeometry(
            n_segment=round(float(segment.value) * FS),
            n_hop=round(float(hop.value) * FS),
            band_hz=(float(band.value[0]), float(band.value[1])) if use_band.value else None,
            n_bin_pool=int(bin_pool.value),
            kernel=kernel.value,
            kernel_width=int(kernel_width.value),
            window=window.value,
            asymmetric_window=asymmetric_window.value,
        )
        _bin_lo, _bin_hi = geometry.bin_range(FS)
        if _bin_hi <= _bin_lo:
            geometry_error = "The selected frequency band contains no STFT bins."
        elif geometry.n_bin_pool > _bin_hi - _bin_lo:
            geometry_error = "`n_bin_pool` exceeds the number of selected frequency bins."
    except ValidationError as exc:
        geometry_error = str(exc)

    mo.stop(
        geometry_error is not None,
        mo.callout(mo.md(f"This geometry is invalid:\n\n```text\n{geometry_error}\n```"), kind="danger"),
    )
    return (geometry,)


@app.cell
def _(FS, compute_segment_window, geometry, mo, np, plt):
    _w = compute_segment_window(
        geometry.n_segment,
        window=geometry.window,
        asymmetric_window=geometry.asymmetric_window,
    )
    _n = geometry.n_segment
    _t_ms = np.arange(_n) / FS * 1000.0
    _w_sq = _w**2
    _sum_sq = float(np.sum(_w_sq))
    _center_idx = float(np.sum(np.arange(_n) * _w_sq) / _sum_sq) if _sum_sq > 0 else (_n - 1) / 2.0
    _center_ms = _center_idx / FS * 1000.0
    _group_delay_ms = (_n - 1.0 - _center_idx) / FS * 1000.0

    _fig, _ax = plt.subplots(figsize=(8, 2.4), layout="constrained")
    _ax.plot(_t_ms, _w, color="tab:blue", lw=1.8, label=f"{geometry.window}")
    _ax.axvline(
        _center_ms,
        color="tab:orange",
        ls="--",
        lw=1.2,
        label=f"Center of energy ({_center_ms:.1f} ms | delay {_group_delay_ms:.1f} ms)",
    )
    _ax.axvline(_t_ms[-1], color="tab:gray", ls=":", lw=1.0, label="Newest sample (t = 0 ms)")
    _ax.set(
        xlabel="Segment elapsed time (ms)",
        ylabel="Taper weight",
        title=f"Segment window function (effective group delay: {_group_delay_ms:.1f} ms)",
        ylim=(-0.05, 1.05),
    )
    _ax.legend(fontsize=8, frameon=False, loc="upper left")
    _ax.spines[["top", "right"]].set_visible(False)
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(FS, duration, lfp_trajectories, np, start_s, trajectories, trajectory):
    def view_samples(y: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Clip the requested time view to a decimated trajectory."""
        start = min(float(start_s.value), y.shape[0] / FS - 1.0 / FS)
        stop = min(start + float(duration.value), y.shape[0] / FS)
        return y[round(start * FS) : round(stop * FS)], start, stop

    y_view, view_start, view_stop = view_samples(trajectories[trajectory.value])
    lfp_view, _, _ = view_samples(lfp_trajectories[trajectory.value])
    return lfp_view, view_start, view_stop, y_view


@app.cell
def _(FS, compute_log_power_frames, geometry, np, view_start, y_view):
    frames = compute_log_power_frames(y_view, geometry, fs=FS)
    freqs = np.fft.rfftfreq(geometry.n_segment, d=1.0 / FS)
    _bin_lo, _bin_hi = geometry.bin_range(FS)
    freqs = freqs[_bin_lo:_bin_hi]
    n_values = freqs.size // geometry.n_bin_pool
    freqs = freqs[: n_values * geometry.n_bin_pool].reshape(n_values, geometry.n_bin_pool).mean(axis=1)
    frame_times = view_start + (np.arange(frames.shape[0]) * geometry.n_hop + geometry.sample_support_steps(FS)) / FS
    return frame_times, frames, freqs


@app.cell
def _(mo, view_start, view_stop):
    frame_time = mo.ui.slider(
        view_start,
        view_stop,
        0.02,
        value=view_start,
        label="Frame time for spectral slice (s)",
    )
    frame_time
    return (frame_time,)


@app.cell
def _(
    FS,
    lfp_picker,
    lfp_view,
    marker_lines,
    mo,
    np,
    plt,
    region_labels,
    view_start,
    view_stop,
):
    _selected_regions = list(lfp_picker.value)
    mo.stop(not _selected_regions, mo.callout("Select at least one LFP region to draw its traces.", kind="warn"))
    _region_indices = [region_labels.index(region) for region in _selected_regions]
    _figure, _axes = plt.subplots(
        len(_region_indices),
        1,
        figsize=(12, max(2.5, 1.7 * len(_region_indices))),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    _axes = np.atleast_1d(_axes)
    _time = view_start + np.arange(lfp_view.shape[0]) / FS
    for _axis, _region, _index in zip(_axes, _selected_regions, _region_indices, strict=True):
        _axis.plot(_time, lfp_view[:, _index], color="tab:red" if _region.startswith("l") else "tab:blue", lw=0.8)
        for _marker_time, _marker_color in marker_lines:
            _axis.axvline(_marker_time, color=_marker_color, lw=1.0)
        _axis.set(ylabel=f"{_region}\n(mV)")
        _axis.spines[["top", "right"]].set_visible(False)
    _axes[-1].set(xlim=(view_start, view_stop), xlabel="time (s)")
    _figure.suptitle("Regional LFP traces")
    mo.mpl.interactive(_figure)
    return


@app.cell
def _(
    marker_lines,
    mo,
    plt,
    seizure_profiles,
    trajectory,
    view_start,
    view_stop,
):
    _profile = seizure_profiles[trajectory.value]
    _figure, _axis = plt.subplots(figsize=(12, 2.8), layout="constrained")
    _axis.step(_profile.times, _profile.n_seizing(), where="mid", color="black", lw=1.4)
    for _marker_time, _marker_color in marker_lines:
        _axis.axvline(_marker_time, color=_marker_color, lw=1.0)
    _axis.set(
        xlim=(view_start, view_stop),
        ylim=(0, len(_profile.onsets)),
        xlabel="time (s)",
        ylabel="seizing regions",
        title="Regional seizure count",
    )
    _axis.spines[["top", "right"]].set_visible(False)
    mo.mpl.interactive(_figure)
    return


@app.cell
def _(
    channel_labels,
    frame_time,
    frame_times,
    frames,
    freqs,
    marker_lines,
    mo,
    np,
    plt,
    selected,
    view_start,
    view_stop,
    y_view,
):
    mo.stop(not selected, mo.callout("Select at least one channel to draw the plots.", kind="warn"))
    indices = [channel_labels.index(label) for label in selected]
    selected_frames = frames[:, indices, :]
    no_frames = not len(frame_times)
    mo.stop(no_frames, mo.callout("This view is shorter than the selected Frame support.", kind="warn"))

    selected_time = float(frame_time.value)
    frame_index = int(np.argmin(np.abs(frame_times - selected_time)))
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, len(indices)))
    vmin, vmax = float(selected_frames.min()), float(selected_frames.max())

    trace_fig, trace_axes = plt.subplots(
        len(indices), 1, figsize=(12, max(2.5, 1.7 * len(indices))), sharex=True, sharey=True, layout="constrained"
    )
    trace_axes = np.atleast_1d(trace_axes)
    time = view_start + np.arange(y_view.shape[0]) / 50.0
    for axis, label, index, color in zip(trace_axes, selected, indices, colors, strict=True):
        axis.plot(time, y_view[:, index], color=color, lw=0.8)
        for marker_time, marker_color in marker_lines:
            axis.axvline(marker_time, color=marker_color, lw=1.0)
        axis.axvline(frame_times[frame_index], color="black", lw=0.8, ls="--")
        axis.set(ylabel=f"{label}\n(mV)")
        axis.spines[["top", "right"]].set_visible(False)
    trace_axes[-1].set(xlim=(view_start, view_stop), xlabel="time (s)")
    trace_fig.suptitle("Selected EEG traces")

    ncols = min(3, len(indices))
    nrows = int(np.ceil(len(indices) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 3.2 * nrows), squeeze=False, sharex=True, sharey=True, layout="constrained"
    )
    extent = (frame_times[0], frame_times[-1], freqs[0], freqs[-1])
    for axis, label, values in zip(
        axes.ravel()[: len(selected)], selected, selected_frames.transpose(1, 0, 2), strict=True
    ):
        image = axis.imshow(
            values.T,
            aspect="auto",
            origin="lower",
            extent=extent,
            interpolation="nearest",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        axis.axvline(frame_times[frame_index], color="white", lw=1.0)
        axis.set(title=label, xlabel="Frame emission time (s)", ylabel="Hz")
    for axis in axes.flat[len(indices) :]:
        axis.set_visible(False)
    fig.colorbar(image, ax=axes.flat[: len(indices)], label="log power (nats)")
    fig.suptitle("Selected-channel Observable Frames")

    _line_fig, line_axis = plt.subplots(figsize=(11, 4), layout="constrained")
    for label, values, color in zip(selected, selected_frames[frame_index], colors, strict=True):
        line_axis.plot(freqs, values, marker="o", ms=3, lw=1.2, color=color, label=label)
    line_axis.set(
        xlabel="Hz",
        ylabel="log power (nats)",
        title=f"Observable Frame emitted at {frame_times[frame_index]:.2f} s",
    )
    line_axis.legend(ncols=min(4, len(indices)), frameon=False)
    line_axis.spines[["top", "right"]].set_visible(False)
    mo.vstack([mo.mpl.interactive(trace_fig), mo.mpl.interactive(fig), mo.mpl.interactive(_line_fig)])
    return


@app.cell
def _(geometry, mo):
    mo.md(f"""
    ```python
    StftGeometry(
        n_segment={geometry.n_segment},
        n_hop={geometry.n_hop},
        band_hz={geometry.band_hz!r},
        n_bin_pool={geometry.n_bin_pool},
        kernel={geometry.kernel!r},
        kernel_width={geometry.kernel_width},
        window={geometry.window!r},
        asymmetric_window={geometry.asymmetric_window!r},
    )
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
