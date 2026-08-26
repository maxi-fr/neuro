import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium", app_title="STFT Loss Walkthrough")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import torch
    from matplotlib import pyplot as plt
    from pydantic import ValidationError

    from neuro.config import StftSpec
    from neuro.eeg import build_eeg_leadfield
    from neuro.predictor.data import load_trajectory
    from neuro.predictor.inference import WaveformMLPModel
    from neuro.predictor.losses import frame_kernel, pool_bins, smooth_frames, spectrogram
    from neuro.spectral import LOG_FLOOR

    return (
        LOG_FLOOR,
        StftSpec,
        ValidationError,
        WaveformMLPModel,
        build_eeg_leadfield,
        frame_kernel,
        load_trajectory,
        mo,
        np,
        plt,
        pool_bins,
        smooth_frames,
        spectrogram,
        torch,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # The `stft` training loss, one channel at a time

    Every step of §2.2 of [`docs/spectral_objectives.md`](../docs/spectral_objectives.md), plotted
    for a single channel of a single rollout. `pred` is a free run of the trained predictor
    `artifacts/nonlinear_mse02_psd`; `true` is the held-out trajectory it was primed on. The
    sliders are the nine `StftSpec` fields of §2.1, so an invalid combination fails here the same
    way it fails at config load.

    | symbol | field |
    | :--- | :--- |
    | $S$ | `n_span` |
    | $W$ | `n_segment` |
    | $H$ | `n_hop` |
    | $K_f$ | `n_bin_pool` |
    | $w_m$, $N_w$ | `kernel`, `kernel_width` |
    """)
    return


@app.cell
def _(WaveformMLPModel, build_eeg_leadfield, load_trajectory):
    MAX_SPAN = 150

    art = WaveformMLPModel.load("artifacts/nonlinear_mse02_psd/model")
    fs = 1.0 / art.dt
    channel_labels = [str(label) for label in build_eeg_leadfield()[1]]

    trajectories = [
        load_trajectory(
            f"data/experiment_excited_roast/test/sim_{i:03d}.npz", None, art.downsample, art.dt / art.downsample
        )
        for i in range(4)
    ]
    n_samples = trajectories[0][1].shape[0]
    return MAX_SPAN, art, channel_labels, fs, n_samples, trajectories


@app.cell
def _(MAX_SPAN, art, channel_labels, fs, mo, n_samples, trajectories):
    traj_slider = mo.ui.slider(0, len(trajectories) - 1, 1, value=0, label="Trajectory")
    t0_slider = mo.ui.slider(
        art.priming_steps,
        n_samples - MAX_SPAN,
        1,
        value=art.priming_steps + 100,
        label="Rollout start $t_0$ (samples)",
    )
    channel_dropdown = mo.ui.dropdown(options=channel_labels, value="CP5", label="Channel")

    mo.hstack(
        [mo.md(f"#### Rollout ({fs:g} Hz)"), traj_slider, t0_slider, channel_dropdown],
        justify="start",
        gap=2,
    )
    return channel_dropdown, t0_slider, traj_slider


@app.cell
def _(
    MAX_SPAN,
    art,
    channel_dropdown,
    channel_labels,
    np,
    t0_slider,
    traj_slider,
    trajectories,
):
    u_traj, y_traj = trajectories[traj_slider.value]
    t0 = int(t0_slider.value)
    k = art.priming_steps

    pred_full = np.asarray(
        art.free_run(y_traj[t0 - k : t0][None], u_traj[t0 - k : t0][None], u_traj[t0 : t0 + MAX_SPAN][None])
    )[0]
    true_full = y_traj[t0 : t0 + MAX_SPAN]
    channel = channel_labels.index(channel_dropdown.value)
    return channel, pred_full, true_full


@app.cell
def _(MAX_SPAN, mo):
    span_slider = mo.ui.slider(10, MAX_SPAN, 5, value=50, label="`n_span` $S$")
    segment_slider = mo.ui.slider(4, MAX_SPAN, 1, value=25, label="`n_segment` $W$")
    hop_slider = mo.ui.slider(1, MAX_SPAN, 1, value=12, label="`n_hop` $H$")
    band_checkbox = mo.ui.checkbox(value=True, label="Apply `band_hz`")
    band_slider = mo.ui.range_slider(0.5, 25.0, 0.5, value=[3.0, 12.0], label="`band_hz` (Hz)")
    pool_slider = mo.ui.slider(1, 8, 1, value=1, label="`n_bin_pool` $K_f$")
    kernel_dropdown = mo.ui.dropdown(options=["boxcar", "triangular", "hann"], value="boxcar", label="`kernel` $w_m$")
    width_slider = mo.ui.slider(1, 9, 1, value=1, label="`kernel_width` $N_w$")

    mo.hstack(
        [
            mo.vstack([mo.md("#### Geometry"), span_slider, segment_slider, hop_slider]),
            mo.vstack([mo.md("#### Band"), band_checkbox, band_slider]),
            mo.vstack([mo.md("#### Pre-log pooling"), pool_slider, kernel_dropdown, width_slider]),
        ],
        justify="space-between",
        gap=3,
    )
    return (
        band_checkbox,
        band_slider,
        hop_slider,
        kernel_dropdown,
        pool_slider,
        segment_slider,
        span_slider,
        width_slider,
    )


@app.cell
def _(
    StftSpec,
    ValidationError,
    band_checkbox,
    band_slider,
    fs,
    hop_slider,
    kernel_dropdown,
    mo,
    pool_slider,
    segment_slider,
    span_slider,
    width_slider,
):
    spec, spec_error = None, None
    try:
        spec = StftSpec(
            weight=1.0,
            n_span=int(span_slider.value),
            n_segment=int(segment_slider.value),
            n_hop=int(hop_slider.value),
            band_hz=(float(band_slider.value[0]), float(band_slider.value[1])) if band_checkbox.value else None,
            n_bin_pool=int(pool_slider.value),
            kernel=kernel_dropdown.value,
            kernel_width=int(width_slider.value),
        )
    except ValidationError as exc:
        spec_error = str(exc)

    # The band and pooling checks need the rate, so they sit on NNPredictorConfig
    # (_validate_losses_and_horizon) rather than on the spec. Repeated here at the notebook's fs.
    if spec is not None:
        _lo, _hi = spec.bin_range(fs)
        if _hi - _lo < 1:
            spec_error = f"stft leaves no frequency bins at fs={fs} Hz for band_hz={spec.band_hz}."
        elif spec.n_bin_pool > _hi - _lo:
            spec_error = f"stft.n_bin_pool ({spec.n_bin_pool}) exceeds the {_hi - _lo} in-band bin(s) at fs={fs} Hz."

    mo.stop(
        spec_error is not None,
        mo.callout(mo.md(f"This geometry is rejected at config load:\n\n```text\n{spec_error}\n```"), kind="danger"),
    )
    return (spec,)


@app.cell
def _(
    LOG_FLOOR,
    frame_kernel,
    fs,
    np,
    pool_bins,
    smooth_frames,
    spec,
    spectrogram,
    torch,
):
    def stages(y):
        """Run one ``(span, channels)`` rollout through steps 1-7, keeping every intermediate."""
        raw = torch.as_tensor(y[None, : spec.n_span], dtype=torch.float64).transpose(1, 2)
        segments = raw.unfold(dimension=-1, size=spec.n_segment, step=spec.n_hop)
        power = spectrogram(raw, spec.n_segment, spec.n_hop, fs=fs)
        bin_lo, bin_hi = spec.bin_range(fs)
        banded = power[..., bin_lo:bin_hi]
        pooled = pool_bins(banded, spec.n_bin_pool)
        weights = frame_kernel(spec.kernel, spec.kernel_width, pooled)
        smoothed = smooth_frames(pooled, weights)
        return {
            "segments": np.asarray(segments[0]),
            "power": np.asarray(power[0]),
            "banded": np.asarray(banded[0]),
            "pooled": np.asarray(pooled[0]),
            "smoothed": np.asarray(smoothed[0]),
            "log": np.asarray(torch.log(smoothed[0] + LOG_FLOOR)),
            "weights": np.asarray(weights),
        }

    taper = np.asarray(torch.hann_window(spec.n_segment, periodic=True, dtype=torch.float64))
    return stages, taper


@app.cell
def _(fs, np, pred_full, spec, stages, true_full):
    stage_pred = stages(pred_full)
    stage_true = stages(true_full)

    residual = stage_pred["log"] - stage_true["log"]
    loss = float(np.mean(residual**2))

    bin_lo, bin_hi = spec.bin_range(fs)
    freqs_all = np.arange(spec.n_segment // 2 + 1) * fs / spec.n_segment
    freqs_band = freqs_all[bin_lo:bin_hi]
    n_pooled = freqs_band.size // spec.n_bin_pool
    freqs_pooled = freqs_band[: n_pooled * spec.n_bin_pool].reshape(n_pooled, spec.n_bin_pool).mean(axis=1)

    m_in = spec.n_segment_frames(spec.n_span)
    m_out = stage_pred["log"].shape[1]
    frame_times = (np.arange(m_in) * spec.n_hop + spec.n_segment / 2) / fs
    out_times = frame_times[(spec.kernel_width - 1) // 2 :][:m_out]
    k_eff = float(stage_pred["weights"].sum() ** 2 / (stage_pred["weights"] ** 2).sum())
    return (
        bin_hi,
        bin_lo,
        freqs_all,
        freqs_pooled,
        k_eff,
        loss,
        m_in,
        m_out,
        out_times,
        residual,
        stage_pred,
        stage_true,
    )


@app.cell
def _(bin_hi, bin_lo, freqs_pooled, fs, k_eff, loss, m_in, m_out, mo, spec):
    mo.md(f"""
    | quantity | value |
    | :--- | ---: |
    | $\\Delta f = f_s / W$ | {fs / spec.n_segment:.3g} Hz |
    | $M$ (frames cut) | {m_in} |
    | $F$ (in-band bins, DC dropped) | {bin_hi - bin_lo} |
    | $F'$ (after $K_f$) | {freqs_pooled.size} |
    | $M_\\text{{out}}$ (after the kernel) | {m_out} |
    | $K_\\text{{eff}}$ | {k_eff:.3g} |
    | $\\mathcal{{L}}_\\text{{STFT}}$ | **{loss:.4g}** nats² |
    """)
    return


@app.cell
def _(m_out, mo):
    frame_slider = mo.ui.slider(0, max(m_out - 1, 0), 1, value=0, label="Output frame $m$ to inspect")
    frame_slider
    return (frame_slider,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Steps 1-2: the span, cut into $M$ segments

    Segments hop by $H$ and are $W$ long, so they overlap whenever $H < W$ and leave gaps whenever
    $H > W$. The trailing $S - ((M-1)H + W)$ samples fall off the grid entirely and are never
    scored. The bars under the traces are the $M$ segment supports, staggered over three rows so
    overlap stays readable; the orange one is the input frame at the centre of the selected output
    frame's kernel. The lower panel is that segment after the periodic Hann taper of step 3; there
    is no per-segment detrend, so whatever mean the segment carries goes into the DC bin, and the
    DC bin is then dropped.
    """)
    return


@app.cell
def _(
    channel,
    channel_dropdown,
    frame_slider,
    fs,
    m_in,
    np,
    plt,
    pred_full,
    spec,
    stage_pred,
    stage_true,
    taper,
    true_full,
):
    _fig, _axes = plt.subplots(2, 1, figsize=(11, 6), layout="constrained")

    _t = np.arange(spec.n_span) / fs
    _axes[0].plot(_t, true_full[: spec.n_span, channel], color="black", lw=1.2, label="true")
    _axes[0].plot(_t, pred_full[: spec.n_span, channel], color="crimson", lw=1.2, label="pred")

    _m_in = min(frame_slider.value + (spec.kernel_width - 1) // 2, m_in - 1)
    _span = float(np.ptp(true_full[: spec.n_span, channel]))
    _floor = float(true_full[: spec.n_span, channel].min()) - 0.15 * _span

    # A ruler of segment supports under the traces, staggered so overlapping frames stay legible.
    for _m in range(m_in):
        _start = _m * spec.n_hop / fs
        _axes[0].hlines(
            _floor - 0.06 * _span * (_m % 3),
            _start,
            _start + spec.n_segment / fs,
            color="tab:orange" if _m == _m_in else "tab:blue",
            lw=4,
            alpha=0.9 if _m == _m_in else 0.35,
        )
    _axes[0].axvspan(
        _m_in * spec.n_hop / fs, (_m_in * spec.n_hop + spec.n_segment) / fs, color="tab:orange", alpha=0.12
    )
    _axes[0].set(
        xlabel="time from $t_0$ (s)",
        ylabel=f"{channel_dropdown.value} (mV)",
        title=f"$S$ = {spec.n_span} samples, $W$ = {spec.n_segment}, $H$ = {spec.n_hop}, $M$ = {m_in} frames",
    )
    _axes[0].legend(loc="upper right", frameon=False)

    _tw = (_m_in * spec.n_hop + np.arange(spec.n_segment)) / fs
    _axes[1].plot(_tw, stage_true["segments"][channel, _m_in] * taper, color="black", lw=1.2, label="true, tapered")
    _axes[1].plot(_tw, stage_pred["segments"][channel, _m_in] * taper, color="crimson", lw=1.2, label="pred, tapered")
    _axes[1].plot(_tw, stage_true["segments"][channel, _m_in], color="black", lw=0.8, alpha=0.3, label="true, raw")
    _axes[1].set(xlabel="time from $t_0$ (s)", ylabel="mV", title=f"segment $m$ = {_m_in} after the Hann taper")
    _axes[1].legend(loc="upper right", frameon=False, ncols=3)

    for _ax in _axes:
        _ax.spines[["top", "right"]].set_visible(False)
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Steps 3-7: one frame, from periodogram to pooled log power

    DC is dropped before anything else, so the left panel's grey region is discarded whatever
    `band_hz` says. Bin pooling averages *power*, before the log, which is the only reason the
    curve moves toward $\log \mathbb{E}[P]$ rather than $\mathbb{E}[\log P]$ (§2.4). The frame
    kernel then averages the same power across neighbouring frames, so the bottom-left curve
    differs from the top-right one only when $N_w > 1$.
    """)
    return


@app.cell
def _(
    bin_hi,
    bin_lo,
    channel,
    channel_dropdown,
    frame_slider,
    freqs_all,
    freqs_pooled,
    m_in,
    np,
    plt,
    residual,
    spec,
    stage_pred,
    stage_true,
):
    _fig, _axes = plt.subplots(2, 2, figsize=(11, 6.5), layout="constrained")
    _m = frame_slider.value
    _m_raw = min(_m + (spec.kernel_width - 1) // 2, m_in - 1)

    _ax = _axes[0, 0]
    _ax.semilogy(freqs_all, stage_true["power"][channel, _m_raw], color="black", lw=1.1, label="true")
    _ax.semilogy(freqs_all, stage_pred["power"][channel, _m_raw], color="crimson", lw=1.1, label="pred")
    _ax.axvspan(freqs_all[0] - 0.5, freqs_all[bin_lo] - 1e-9, color="grey", alpha=0.18)
    _ax.axvspan(freqs_all[bin_hi - 1] + 1e-9, freqs_all[-1] + 0.5, color="grey", alpha=0.18)
    _ax.set(
        xlabel="Hz", ylabel="power (mV² / Hz)", title=f"step 3-4: periodogram of frame {_m_raw}, kept band unshaded"
    )
    _ax.legend(loc="upper right", frameon=False)

    _ax = _axes[0, 1]
    _ax.semilogy(freqs_pooled, stage_true["pooled"][channel, _m_raw], "o-", color="black", lw=1.1, ms=3, label="true")
    _ax.semilogy(freqs_pooled, stage_pred["pooled"][channel, _m_raw], "o-", color="crimson", lw=1.1, ms=3, label="pred")
    _ax.set(xlabel="Hz", ylabel="power (mV² / Hz)", title=f"step 5: pooled over $K_f$ = {spec.n_bin_pool} bins")

    _ax = _axes[1, 0]
    _ax.plot(freqs_pooled, stage_true["log"][channel, _m], "o-", color="black", lw=1.1, ms=3, label="true")
    _ax.plot(freqs_pooled, stage_pred["log"][channel, _m], "o-", color="crimson", lw=1.1, ms=3, label="pred")
    _ax.set(
        xlabel="Hz",
        ylabel="log power (nats)",
        title=f"steps 6-7: {spec.kernel} kernel of width {spec.kernel_width}, then log",
    )

    _ax = _axes[1, 1]
    _ax.bar(
        freqs_pooled,
        residual[channel, _m] ** 2,
        width=0.8 * np.diff(freqs_pooled, prepend=freqs_pooled[0] - 1).min(),
        color="tab:purple",
    )
    _ax.axhline(float(np.mean(residual[channel, _m] ** 2)), color="black", ls="--", lw=1.0, label="frame mean")
    _ax.set(xlabel="Hz", ylabel="nats²", title="step 8: squared log residual, this frame and channel")
    _ax.legend(loc="upper right", frameon=False)

    for _ax in _axes.flat:
        _ax.spines[["top", "right"]].set_visible(False)
    _fig.suptitle(f"channel {channel_dropdown.value}, output frame $m$ = {_m}")
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Steps 6-9: the whole $(m, f)$ plane

    The residual panel is what step 9 averages, here over one channel only. The kernel panel is
    $w_m$ itself: `boxcar` puts every frame on equal footing and maximises $K_\text{eff}$, `hann`
    concentrates on the centre frame and keeps time localisation.
    """)
    return


@app.cell
def _(
    channel,
    channel_dropdown,
    freqs_pooled,
    fs,
    k_eff,
    loss,
    m_out,
    np,
    out_times,
    plt,
    residual,
    spec,
    stage_pred,
    stage_true,
):
    _fig, _axes = plt.subplots(2, 2, figsize=(11, 6.5), layout="constrained")
    _half_bin = 0.5 * spec.n_bin_pool * fs / spec.n_segment
    _extent = (
        out_times[0] - spec.n_hop / (2 * fs),
        out_times[-1] + spec.n_hop / (2 * fs),
        freqs_pooled[0] - _half_bin,
        freqs_pooled[-1] + _half_bin,
    )
    _kw = {"aspect": "auto", "origin": "lower", "extent": _extent, "interpolation": "nearest"}

    _true = stage_true["log"][channel].T
    _pred = stage_pred["log"][channel].T
    _vmin, _vmax = float(min(_true.min(), _pred.min())), float(max(_true.max(), _pred.max()))

    for _ax, _data, _title in (
        (_axes[0, 0], _true, "step 7: $\\log P$ true"),
        (_axes[0, 1], _pred, "step 7: $\\log \\hat{P}$ pred"),
    ):
        _im = _ax.imshow(_data, vmin=_vmin, vmax=_vmax, cmap="viridis", **_kw)
        _ax.set(xlabel="frame centre (s from $t_0$)", ylabel="Hz", title=_title)
        _fig.colorbar(_im, ax=_ax, label="nats")

    _ax = _axes[1, 0]
    _im = _ax.imshow(residual[channel].T ** 2, cmap="magma", **_kw)
    _ax.set(
        xlabel="frame centre (s from $t_0$)", ylabel="Hz", title=f"step 8: squared residual, {channel_dropdown.value}"
    )
    _fig.colorbar(_im, ax=_ax, label="nats²")

    _ax = _axes[1, 1]
    _ax.stem(np.arange(spec.kernel_width), stage_pred["weights"], basefmt=" ")
    _ax.set(
        xlabel="frame offset",
        ylabel="$w_m$",
        xlim=(-0.5, spec.kernel_width - 0.5),
        ylim=(0.0, 1.15 * float(stage_pred["weights"].max())),
        title=f"{spec.kernel} kernel, $K_\\text{{eff}}$ = {k_eff:.2f}, $M_\\text{{out}}$ = {m_out}",
    )
    _ax.spines[["top", "right"]].set_visible(False)

    _fig.suptitle(f"$\\mathcal{{L}}_\\text{{STFT}}$ over all channels and frames = {loss:.4g} nats²")
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Things worth driving the sliders at

    1. **Push $W$ to $S$.** One frame, the Welch endpoint the live config sits at. The $(m, f)$
       plane collapses to a column and every trace of time resolution is gone.
    2. **Drop $W$ to 15 with $H = 8$.** Frames get cheap, $\Delta f$ goes to 3.3 Hz, and the
       3-12 Hz band holds three bins. Time resolution is bought straight out of frequency
       resolution.
    3. **Raise $K_f$ or $N_w$.** Both smooth the residual plane, and both do it *before* the log,
       which is what makes them legitimate. The squared residual drops because the estimator got
       quieter, not because the prediction got better.
    4. **Turn `band_hz` off.** The bins above 12 Hz carry little power and a lot of $\chi^2$
       scatter, so they dominate the residual plane while saying least about the seizure.
    """)
    return


if __name__ == "__main__":
    app.run()
