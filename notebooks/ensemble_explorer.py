import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium", app_title="Ensemble Explorer")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import numpy as np
    from matplotlib import pyplot as plt

    from neuro.ensembles import cached_scalp, load_manifest, region_path
    from neuro.geometry import sensor_positions_mm
    from neuro.seizure import SEIZURE_PTP_MV, SPREAD_WINDOW_S, spread_profile_from_lfp
    from utils.processing import compute_psd, synchronization

    return (
        Path,
        SEIZURE_PTP_MV,
        SPREAD_WINDOW_S,
        cached_scalp,
        compute_psd,
        load_manifest,
        mo,
        np,
        plt,
        region_path,
        sensor_positions_mm,
        spread_profile_from_lfp,
        synchronization,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # 🧠 Ensemble Explorer — the visual tier

    Looks at the ensemble `scripts/run_predictability_experiment.py` generates, from the
    descriptive side: **trajectories, PSD, functional connectivity, topoplots and spread
    rasters**. The quantitative side — R², $d_{ctrl}$, separability — lives in
    `metric_scoring.py`. The split is by *purpose*, not by experiment, so the headline figure
    stays in one file.

    Everything here reads a **1 kHz cache** built on first run. The full-rate 10 kHz stores are
    ~19 GB and will not sit in a kernel; nothing plotted below resolves above ~50 Hz anyway.
    Scoring does *not* use this cache — see `metric_scoring.py`.

    See `docs/predictability_controllability_experiment.md` for the design.
    """)
    return


@app.cell
def _(mo):
    dir_input = mo.ui.text(
        value="data/predictability_ensemble",
        label="Ensemble directory",
        full_width=True,
    )
    mo.md(f"**Ensemble directory:** {dir_input}")
    return (dir_input,)


@app.cell
def _(Path, dir_input, load_manifest, mo):
    ens_dir = Path(dir_input.value)
    mo.stop(
        not (ens_dir / "manifest.json").exists(),
        mo.md(
            f"⚠️ No `manifest.json` under `{ens_dir}`. Generate one first:\n\n"
            f"```\nuv run python scripts/run_predictability_experiment.py --out {ens_dir}\n```"
        ),
    )

    manifest = load_manifest(ens_dir)
    branch_names = [b["name"] for b in manifest["branches"]]
    # arms are allocated per branch, so only these pairs have a store on disk
    branch_arms = {b["name"]: list(b["arms"]) for b in manifest["branches"]}
    n_rollouts = manifest["n_parents"] * manifest["n_children"]
    return branch_arms, branch_names, ens_dir, manifest, n_rollouts


@app.cell
def _(manifest, mo, n_rollouts):
    _seizing = [p["n_seizing_final"] for p in manifest["parents"]]
    mo.md(f"""
    | | |
    | --- | --- |
    | branches | {", ".join(f"`{b['name']}`@{b['t_branch']:g}s" for b in manifest["branches"])} |
    | arms | {", ".join(f"`{a['name']}`={a['current']}" for a in manifest["arms"])} |
    | parents × children | {manifest["n_parents"]} × {manifest["n_children"]} = {n_rollouts} rollouts per branch × arm |
    | rollout | {manifest["rollout_s"]:g} s at {manifest["fs"] / 1000:g} kHz (region stored at {manifest["region_fs"] / 1000:g} kHz) |
    | channel sets | {", ".join(f"`{k}` ({len(v)} ch)" for k, v in manifest["channel_sets"].items())} |
    | parent basins | {_seizing} seizing regions at the end of each parent run |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Parent basins

    The plant is **bimodal across seeds** — `tes_field_geometry.md` §1.3 — so the later branches
    contain parents that never seized. Averaging over that hides a mixture rather than
    summarising one, which is why the basin label is carried in the manifest and shown here.
    """)
    return


@app.cell
def _(manifest, mo, np, plt):
    _seizing = np.array([p["n_seizing_final"] for p in manifest["parents"]])
    _plants = [p["plant"] for p in manifest["parents"]]

    _fig, _ax = plt.subplots(figsize=(7, 3))
    for _kind, _colour in (("healthy", "tab:green"), ("seizure", "tab:red")):
        _mask = np.array([p == _kind for p in _plants])
        if _mask.any():
            _ax.scatter(np.flatnonzero(_mask), _seizing[_mask], label=_kind, color=_colour, s=60)
    _ax.set_xlabel("parent")
    _ax.set_ylabel("seizing regions at end of run")
    _ax.set_title("Parent basins — each parent's final spread")
    _ax.legend(frameon=False)
    _ax.spines[["top", "right"]].set_visible(False)
    _ax.grid(visible=True, axis="y", linestyle="--", alpha=0.3)
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(branch_names, mo, n_rollouts):
    branch_select = mo.ui.dropdown(options=branch_names, value=branch_names[0], label="Branch")
    rollout_slider = mo.ui.slider(0, n_rollouts - 1, value=0, label="Rollout", show_value=True)
    return branch_select, rollout_slider


@app.cell
def _(branch_arms, branch_select, mo, rollout_slider):
    # rebuilt whenever the branch changes, so the arm offered is always one this branch carries
    _arms = branch_arms[branch_select.value]
    arm_select = mo.ui.dropdown(options=_arms, value=_arms[0], label="Arm")
    mo.hstack([branch_select, arm_select, rollout_slider], justify="start", gap=2)
    return (arm_select,)


@app.cell
def _(
    arm_select,
    branch_select,
    cached_scalp,
    ens_dir,
    manifest,
    mo,
    np,
    region_path,
):
    with mo.status.spinner(title="Building the 1 kHz scalp cache (first run only)…"):
        scalp = cached_scalp(ens_dir, branch_select.value, arm_select.value, manifest)
    region = np.load(region_path(ens_dir, branch_select.value, arm_select.value), mmap_mode="r")

    scalp_fs = 1000.0
    region_fs = manifest["region_fs"]
    return region, region_fs, scalp, scalp_fs


@app.cell
def _(mo):
    mo.md(r"""
    ## Trajectories

    One rollout, scalp and region side by side. The scalp is the **primary space** — a metric the
    controller cannot measure is disqualified whatever else it scores — and the region LFP is
    only ever the ground-truth generator.
    """)
    return


@app.cell
def _(
    manifest,
    mo,
    np,
    plt,
    region,
    region_fs,
    rollout_slider,
    scalp,
    scalp_fs,
):
    _eeg = np.asarray(scalp[rollout_slider.value], dtype=np.float64)
    _reg = np.asarray(region[rollout_slider.value], dtype=np.float64)
    _focal = manifest["channel_sets"][next(k for k in manifest["channel_sets"] if k != "all62")]

    _fig, _axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    _t = np.arange(_eeg.shape[1]) / scalp_fs
    for _c in _focal:
        _axes[0].plot(_t, _eeg[_c], lw=0.7, label=manifest["channel_labels"][_c])
    _axes[0].set_ylabel("scalp EEG [mV]")
    _axes[0].legend(frameon=False, ncol=len(_focal), fontsize=8)
    _axes[0].set_title(f"rollout {rollout_slider.value} — focal channels and the seizure focus")

    _t_reg = np.arange(_reg.shape[1]) / region_fs
    for _name in ("lHC", "lTCI", "rTCI"):
        if _name in manifest["region_labels"]:
            _axes[1].plot(_t_reg, _reg[manifest["region_labels"].index(_name)], lw=0.7, label=_name)
    _axes[1].set_ylabel("region LFP [mV]")
    _axes[1].set_xlabel("lookahead h since branch [s]")
    _axes[1].legend(frameon=False, ncol=3, fontsize=8)

    for _ax in _axes:
        _ax.spines[["top", "right"]].set_visible(False)
        _ax.grid(visible=True, linestyle="--", alpha=0.3)
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Ensemble spread — how fast the future stops being determined

    Every child of every parent, one line each. This is what the R² score in `metric_scoring.py`
    measures: the ensemble fans out from a shared branch state at a rate that no predictor can
    beat.
    """)
    return


@app.cell
def _(manifest, mo, np, plt, scalp, scalp_fs):
    _ch = manifest["channel_labels"].index("TP9")
    _n_children = manifest["n_children"]
    _t = np.arange(scalp.shape[2]) / scalp_fs

    _fig, _ax = plt.subplots(figsize=(10, 4))
    for _row in range(scalp.shape[0]):
        _ax.plot(_t, np.asarray(scalp[_row, _ch]), lw=0.4, alpha=0.6, color=f"C{_row // _n_children}")
    _ax.set_xlabel("lookahead h since branch [s]")
    _ax.set_ylabel("TP9 [mV]")
    _ax.set_title("Ensemble fan-out at TP9 — one colour per parent")
    _ax.spines[["top", "right"]].set_visible(False)
    _ax.grid(visible=True, linestyle="--", alpha=0.3)
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Spectrum — PSD Comparison

    Compare power spectral density across branches and channels for the selected arm and rollout.
    The seizure rhythm sits at 3–4 Hz with harmonics, while the healthy branch is broad and
    unpeaked. This is what the `band_power` (3–12 Hz) and `spectral_centroid` metrics reduce to a
    scalar.
    """)
    return


@app.cell
def _(branch_names, manifest, mo):
    _default_branches = [b for b in ("healthy", "pre_onset", "saturated") if b in branch_names] or branch_names
    psd_branches = mo.ui.multiselect(
        options=branch_names,
        value=_default_branches,
        label="Branches to compare",
    )
    psd_channels = mo.ui.multiselect(
        options=list(manifest["channel_labels"]),
        value=["TP9"],
        label="Channels",
    )
    psd_show_mean = mo.ui.checkbox(value=False, label="Plot 62-ch mean")
    mo.hstack([psd_branches, psd_channels, psd_show_mean], justify="start", gap=2)
    return psd_branches, psd_channels, psd_show_mean


@app.cell
def _(
    arm_select,
    cached_scalp,
    compute_psd,
    ens_dir,
    manifest,
    mo,
    np,
    plt,
    psd_branches,
    psd_channels,
    psd_show_mean,
    rollout_slider,
    scalp_fs,
):
    mo.stop(
        not psd_branches.value,
        mo.md("⚠️ Select at least one branch to compare PSD curves."),
    )
    mo.stop(
        not psd_channels.value and not psd_show_mean.value,
        mo.md("⚠️ Select at least one channel or enable the 62-channel mean."),
    )

    _branch_colors = {
        "healthy": "tab:green",
        "pre_onset": "tab:orange",
        "ez_ignited": "tab:purple",
        "mid_spread": "tab:blue",
        "saturated": "tab:red",
    }
    _styles = ["-", "--", ":", "-."]

    _fig, _ax = plt.subplots(figsize=(10, 5))

    for _b_idx, _branch in enumerate(psd_branches.value):
        _color = _branch_colors.get(_branch, f"C{_b_idx}")
        _store = cached_scalp(ens_dir, _branch, arm_select.value, manifest)
        _eeg = np.asarray(_store[rollout_slider.value], dtype=np.float64)
        _freqs, _pxx = compute_psd(_eeg, dt_ms=1000.0 / scalp_fs)

        if psd_show_mean.value:
            _mean_pxx = np.mean(_pxx, axis=0)
            _ax.plot(
                _freqs,
                _mean_pxx,
                color=_color,
                lw=2.0 if len(psd_branches.value) == 1 else 1.6,
                linestyle="-",
                label=f"{_branch} (mean)",
            )
            if len(psd_branches.value) == 1 and not psd_channels.value:
                _std_pxx = np.std(_pxx, axis=0)
                _ax.fill_between(
                    _freqs,
                    np.maximum(1e-6, _mean_pxx - _std_pxx),
                    _mean_pxx + _std_pxx,
                    color=_color,
                    alpha=0.15,
                    label="± 1 SD",
                )

        for _c_idx, _ch_name in enumerate(psd_channels.value):
            _ch_idx = manifest["channel_labels"].index(_ch_name)
            _ls = _styles[_c_idx % len(_styles)] if (len(psd_channels.value) > 1 or psd_show_mean.value) else "-"
            _ax.plot(
                _freqs,
                _pxx[_ch_idx],
                color=_color,
                lw=1.3,
                linestyle=_ls,
                alpha=0.85,
                label=f"{_branch} ({_ch_name})",
            )

    _ax.set_yscale("log")
    _ax.set_ylim(bottom=1e-4)
    _ax.set_xlim(0, 50.0)
    _ax.set_xlabel("Frequency [Hz]")
    _ax.set_ylabel("PSD [mV²/Hz]")
    _ax.set_title(f"PSD comparison — rollout {rollout_slider.value}, arm `{arm_select.value}`")
    _ax.axvspan(3.0, 12.0, color="tab:red", alpha=0.08, label="3–12 Hz band")
    _n_curves = len(psd_branches.value) * (len(psd_channels.value) + int(psd_show_mean.value))
    _ncol = max(1, min(4, (_n_curves + 3) // 4))
    _ax.legend(frameon=False, fontsize=8, ncol=_ncol)
    _ax.spines[["top", "right"]].set_visible(False)
    _ax.grid(visible=True, linestyle="--", alpha=0.3)
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Functional connectivity — descriptive only

    Shown, never predicted. A 62×62 correlation estimated over a window of $W$ samples has rank
    $\leq W-1$, so at a control-relevant window it is rank-deficient and cannot express
    $\partial FC/\partial u$. `_masked_fc` in `nn_training.py` is the existing instance of that
    problem. Its scalar reduction — the off-diagonal mean, the `synchronization` metric — is what
    bridges into the predicted tier.
    """)
    return


@app.cell
def _(branch_names, manifest, mo):
    _default_branches = [b for b in ("healthy", "pre_onset", "saturated") if b in branch_names] or branch_names
    fc_branches = mo.ui.multiselect(
        options=branch_names,
        value=_default_branches,
        label="Branches to compare",
    )
    _channel_set_options = list(manifest["channel_sets"].keys())
    fc_channel_set = mo.ui.dropdown(
        options=_channel_set_options,
        value="all62" if "all62" in _channel_set_options else _channel_set_options[0],
        label="Channel set",
    )
    mo.hstack([fc_branches, fc_channel_set], justify="start", gap=2)
    return fc_branches, fc_channel_set


@app.cell
def _(
    arm_select,
    cached_scalp,
    ens_dir,
    fc_branches,
    fc_channel_set,
    manifest,
    mo,
    np,
    plt,
    rollout_slider,
    synchronization,
):
    mo.stop(
        not fc_branches.value,
        mo.md("⚠️ Select at least one branch to compare FC matrices."),
    )

    _n_branches = len(fc_branches.value)
    _ch_indices = manifest["channel_sets"][fc_channel_set.value]
    _ch_labels = [manifest["channel_labels"][i] for i in _ch_indices]
    _is_focal = len(_ch_indices) <= 10

    _fig, _axes = plt.subplots(
        1,
        _n_branches,
        figsize=(max(4.0 * _n_branches, 5.0), 4.2),
        squeeze=False,
        sharey=True,
        layout="constrained",
    )

    _last_im = None
    for _idx, _branch in enumerate(fc_branches.value):
        _ax = _axes[0, _idx]
        _store = cached_scalp(ens_dir, _branch, arm_select.value, manifest)
        _eeg = np.asarray(_store[rollout_slider.value], dtype=np.float64)[_ch_indices]
        _fc = np.corrcoef(_eeg)
        _sync = synchronization(_eeg)

        _last_im = _ax.imshow(_fc, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
        _ax.set_title(f"{_branch}\nmean R = {_sync:.3f}", fontsize=10)
        _ax.set_xlabel("channel")
        if _idx == 0:
            _ax.set_ylabel("channel")

        if _is_focal:
            _ax.set_xticks(range(len(_ch_labels)))
            _ax.set_xticklabels(_ch_labels, fontsize=8, rotation=45)
            if _idx == 0:
                _ax.set_yticks(range(len(_ch_labels)))
                _ax.set_yticklabels(_ch_labels, fontsize=8)

    if _last_im is not None:
        _fig.colorbar(_last_im, ax=_axes.ravel().tolist(), shrink=0.8, pad=0.03, label="Pearson R")
    _fig.suptitle(
        f"Scalp Functional Connectivity ({fc_channel_set.value}) — arm `{arm_select.value}`, rollout {rollout_slider.value}",
        fontsize=11,
    )
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Topography

    Where the amplitude sits on the scalp. The one controller in this repo that works —
    `AmplitudeThresholdController`, 6/7 seeds suppressed — triggers on **TP9 alone**, which is
    also the channel loading hardest on `lTCI`. That is the whole reason the focal channel set
    is scored alongside the 62-channel pool.
    """)
    return


@app.cell
def _(manifest, mo, np, plt, rollout_slider, scalp, sensor_positions_mm):
    _labels, _pos = sensor_positions_mm()
    _order = [list(_labels).index(_c) for _c in manifest["channel_labels"]]
    _xy = _pos[_order][:, [1, 0]]

    _eeg = np.asarray(scalp[rollout_slider.value], dtype=np.float64)
    _amp = np.ptp(_eeg, axis=1)
    _focal = set(manifest["channel_sets"][next(k for k in manifest["channel_sets"] if k != "all62")])

    _fig, _ax = plt.subplots(figsize=(5.5, 5))
    _sc = _ax.scatter(_xy[:, 0], _xy[:, 1], c=_amp, cmap="magma", s=170, edgecolors="none")
    for _i, _name in enumerate(manifest["channel_labels"]):
        _ax.annotate(
            _name,
            _xy[_i],
            fontsize=6,
            ha="center",
            va="center",
            color="white" if _i in _focal else "0.6",
            weight="bold" if _i in _focal else "normal",
        )
    _ax.set_aspect("equal")
    _ax.set_title("peak-to-peak amplitude [mV] — focal set in white")
    _ax.set_axis_off()
    _fig.colorbar(_sc, ax=_ax, shrink=0.8)
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Spread raster

    Region space, from the `SpreadProfile` criterion used unchanged everywhere in this repo:
    **5 mV peak-to-peak, 1 s window, 1 s persistence**. This is the label generator, never the
    controller's input.
    """)
    return


@app.cell
def _(
    SEIZURE_PTP_MV,
    SPREAD_WINDOW_S,
    manifest,
    mo,
    np,
    plt,
    region,
    region_fs,
    rollout_slider,
    spread_profile_from_lfp,
):
    _reg = np.asarray(region[rollout_slider.value], dtype=np.float64)
    _too_short = _reg.shape[1] < SPREAD_WINDOW_S * region_fs

    mo.stop(
        _too_short,
        mo.md(
            f"⚠️ The rollout is {_reg.shape[1] / region_fs:.2f} s, shorter than the "
            f"{SPREAD_WINDOW_S:g} s spread window, so no `SpreadProfile` is defined for it."
        ),
    )

    _profile = spread_profile_from_lfp(_reg, 1.0 / region_fs, threshold=SEIZURE_PTP_MV)
    _left = [_i for _i, _n in enumerate(manifest["region_labels"]) if _n.startswith("l")]

    _fig, _ax = plt.subplots(figsize=(9, 5))
    # the store leads the branch by SPREAD_WINDOW_S, so shift the profile onto branch-referenced time
    _im = _ax.pcolormesh(_profile.times - SPREAD_WINDOW_S, np.arange(len(_left)), _profile.ptp[_left], cmap="magma")
    _ax.set_yticks(np.arange(len(_left)) + 0.5)
    _ax.set_yticklabels([manifest["region_labels"][_i] for _i in _left], fontsize=5)
    _ax.set_xlabel("lookahead h since branch [s]")
    _ax.set_title(f"left-hemisphere PTP [mV] — {SEIZURE_PTP_MV:g} mV is the seizing threshold")
    _fig.colorbar(_im, ax=_ax, shrink=0.85)
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Healthy against seizure

    The separability reference. Both branches, same channel, same scale — this contrast is what
    Cohen's d in `metric_scoring.py` puts a number on, and what fixes the **sign** of
    $d_{ctrl}$: "toward healthy" is measured here, not assumed.
    """)
    return


@app.cell
def _(arm_select, branch_names, cached_scalp, ens_dir, manifest, mo, np, plt):
    _ch = manifest["channel_labels"].index("TP9")
    _pair = [_b for _b in ("healthy", "saturated") if _b in branch_names]

    mo.stop(
        len(_pair) < 2,
        mo.md("⚠️ This ensemble has no `healthy`/`saturated` pair, so the contrast is undefined."),
    )

    _fig, _axes = plt.subplots(len(_pair), 1, figsize=(10, 5), sharex=True, sharey=True)
    for _ax, _branch in zip(_axes, _pair, strict=True):
        _store = cached_scalp(ens_dir, _branch, arm_select.value, manifest)
        _t = np.arange(_store.shape[2]) / 1000.0
        for _row in range(min(8, _store.shape[0])):
            _ax.plot(_t, np.asarray(_store[_row, _ch]), lw=0.5, alpha=0.7)
        _ax.set_ylabel(f"{_branch}\nTP9 [mV]")
        _ax.spines[["top", "right"]].set_visible(False)
        _ax.grid(visible=True, linestyle="--", alpha=0.3)
    _axes[-1].set_xlabel("lookahead h since branch [s]")
    _axes[0].set_title(f"healthy against saturated at TP9 — arm `{arm_select.value}`")
    mo.mpl.interactive(_fig)
    return


if __name__ == "__main__":
    app.run()
