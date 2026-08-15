import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium", app_title="Metric Scoring")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt

    from neuro.ensembles import RAW, REGION_SET, SWEPT_METRICS, load_manifest, score_ensemble_dir
    from neuro.filtering import design_lowpass_sos, group_delay_s
    from neuro.metrics import (
        METRICS,
        controllability,
        predictability_r2,
        scalp_region_correlation,
        separability,
        sigma_ens,
        spread_reference,
    )

    return (
        METRICS,
        Path,
        RAW,
        REGION_SET,
        SWEPT_METRICS,
        controllability,
        design_lowpass_sos,
        group_delay_s,
        load_manifest,
        mo,
        np,
        pd,
        plt,
        predictability_r2,
        scalp_region_correlation,
        score_ensemble_dir,
        separability,
        sigma_ens,
        spread_reference,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # 📈 Metric Scoring — the quantitative tier

    Scores every candidate control observable on the four axes of
    `docs/predictability_controllability_experiment.md`:

    | axis | score | what it answers |
    | --- | --- | --- |
    | **predictability** | $R^2(h) = 1 - \mathrm{Var}_{within} / \mathrm{Var}_{total}$ | does fixing $x_0$ determine the metric's future? |
    | **controllability** | $d_{ctrl}(h) = \mathrm{dir}\cdot\bar\delta(h) / \sigma_{ens}(h)$ | does the actuator move it, *toward healthy*? |
    | **separability** | Cohen's $d$, healthy vs saturated | does it tell the two states apart at all? |
    | **observability** | scalp↔region correlation | does it survive volume conduction? |
    | **feasibility** | minimum window length | is it usable at a control rate? |

    Predictability and controllability are **independent axes**, not nested. `tes_field_geometry.md`
    §2.3 is the reason: a predictor that fit cleanly and drove the objective 57 % below baseline
    left the seizure at **31/34 regions, 0/7 seeds suppressed**. Predicted-and-uncontrollable is an
    expected quadrant here, not a failure.

    Thresholds are **not pre-registered** — that was a deliberate call. Full curves over $h$ are
    shown so that any threshold is applied in the open, against the whole curve rather than a
    cherry-picked point.
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
def _(Path, dir_input, load_manifest, mo, score_ensemble_dir):
    ens_dir = Path(dir_input.value)
    mo.stop(
        not (ens_dir / "manifest.json").exists(),
        mo.md(
            f"⚠️ No `manifest.json` under `{ens_dir}`. Generate one first:\n\n"
            f"```\nuv run python scripts/run_predictability_experiment.py --out {ens_dir}\n```"
        ),
    )

    manifest = load_manifest(ens_dir)
    with mo.status.spinner(title="Scoring the stores (first run only, ~40 min at full scale)…"):
        archive = score_ensemble_dir(ens_dir)

    branch_names = [b["name"] for b in manifest["branches"]]
    arm_names = [a["name"] for a in manifest["arms"]]
    stim_arms = [a for a in arm_names if a != "zero"]
    return archive, branch_names, manifest, stim_arms


@app.cell
def _(archive, mo, stim_arms):
    h_slider = mo.ui.slider(
        0.1,
        3.0,
        step=0.05,
        value=1.0,
        label="h_eval [s]",
        show_value=True,
    )
    arm_select = mo.ui.dropdown(options=stim_arms, value=stim_arms[0], label="Stimulation arm")
    set_select = mo.ui.multiselect(
        options=archive.channel_sets,
        value=archive.channel_sets,
        label="Channel sets",
    )
    mo.hstack([h_slider, arm_select, set_select], justify="start", gap=2)
    return arm_select, h_slider, set_select


@app.cell
def _(mo):
    mo.md(r"""
    $h_{eval}$ defaults to **1.0 s**. That is a chosen working lookahead, not one this design
    derives: `probe_payoff_crossover.py` measured the payoff sign flip between 0.8 s and 1.0 s and
    characterised 0.2–0.8 s as "a plateau, not a ramp". Move the slider — every number below
    follows it.
    """)
    return


@app.cell
def _(
    METRICS,
    REGION_SET,
    archive,
    arm_select,
    branch_names,
    controllability,
    h_slider,
    np,
    pd,
    predictability_r2,
    scalp_region_correlation,
    separability,
    set_select,
    sigma_ens,
    spread_reference,
):
    def _at(metric, h):
        return int(np.argmin(np.abs(archive.times[metric] - h)))

    def _score_rows(h):
        rows = []
        for name, metric in METRICS.items():
            idx = _at(name, h)
            valid = archive.times[name][idx] >= metric.window_s
            for channel_set in set_select.value:
                sep = separability(
                    archive.ensemble("scalp", "healthy", "zero", name, channel_set),
                    archive.ensemble("scalp", "saturated", "zero", name, channel_set),
                )
                for branch in branch_names:
                    zero = archive.ensemble("scalp", branch, "zero", name, channel_set)
                    stim = archive.ensemble("scalp", branch, arm_select.value, name, channel_set)
                    region = archive.ensemble("region", branch, "zero", name, REGION_SET)
                    ctrl = controllability(zero, stim, direction=sep.direction, gap=sep.gap)
                    low, high = spread_reference(zero)
                    rows.append(
                        {
                            "metric": name,
                            "channels": channel_set,
                            "branch": branch,
                            "R2": predictability_r2(zero)[idx],
                            "sigma_ens": sigma_ens(zero)[idx],
                            "p5_p95": high[idx] - low[idx],
                            "d_ctrl": ctrl.d_ctrl[idx],
                            "delta_bar": ctrl.delta_bar[idx],
                            "paired_sd": ctrl.paired_sd[idx],
                            "rel_gap": ctrl.relative[idx],
                            "cohens_d": sep.cohens_d[idx],
                            "observability": float(np.nanmedian(scalp_region_correlation(zero, region))),
                            "window_s": metric.window_s,
                            "valid": valid,
                        }
                    )
        return pd.DataFrame(rows)

    scores = _score_rows(h_slider.value)
    return (scores,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Scores at $h_{eval}$

    `sigma_ens` is in the metric's native units; `p5_p95` is the ensemble's 5th-to-95th percentile
    range as a scale reference for it (percentiles, not min–max, which grows with sample size and
    would make the ranking shift as seeds are added).

    `d_ctrl` is signed — **positive is toward healthy**, and that direction is *measured* from the
    healthy-against-saturated contrast, not assumed. A metric stimulation drives confidently in a
    direction that is not recovery is controllable in the linear-systems sense and useless.

    Rows where $h < $ the metric's own window are flagged `valid=False`: the trailing window still
    overlaps pre-branch history every child shares, so `sigma_ens` there is artificially near zero.
    """)
    return


@app.cell
def _(mo, scores):
    mo.ui.table(
        scores.round(
            {
                "R2": 3,
                "sigma_ens": 4,
                "p5_p95": 4,
                "d_ctrl": 3,
                "delta_bar": 4,
                "paired_sd": 4,
                "rel_gap": 3,
                "cohens_d": 2,
                "observability": 2,
            }
        ),
        page_size=15,
        selection=None,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The headline figure

    $R^2$ against signed $d_{ctrl}$ at $h_{eval}$, one panel per phase, one point per metric ×
    channel set, marker size scaled by Cohen's $d$. Both axes are dimensionless SNRs, so they are
    commensurate.

    The **top-right** quadrant is what a controller wants: forecastable *and* steerable toward
    healthy. **Top-left** — predictable but driven the wrong way — is the documented failure mode
    this whole experiment exists to detect.
    """)
    return


@app.cell
def _(branch_names, mo, np, plt, scores, set_select):
    _markers = {"all62": "o", **dict(zip(set_select.value, "os^Dv", strict=False))}
    _fig, _axes = plt.subplots(1, len(branch_names), figsize=(3.2 * len(branch_names), 3.6), sharex=True, sharey=True)
    _axes = np.atleast_1d(_axes)

    for _ax, _branch in zip(_axes, branch_names, strict=True):
        _sub = scores[scores["branch"] == _branch]
        for _i, _name in enumerate(sorted(_sub["metric"].unique())):
            for _cs in set_select.value:
                _row = _sub[(_sub["metric"] == _name) & (_sub["channels"] == _cs)]
                if _row.empty:
                    continue
                _size = 30.0 + 8.0 * np.clip(np.abs(_row["cohens_d"].to_numpy()), 0, 30)
                _ax.scatter(
                    _row["d_ctrl"],
                    _row["R2"],
                    s=_size,
                    marker=_markers.get(_cs, "o"),
                    color=f"C{_i}",
                    alpha=0.55 if _cs != "all62" else 0.9,
                    edgecolors="k",
                    linewidths=0.4,
                    label=_name if (_branch == branch_names[0] and _cs == set_select.value[0]) else None,
                )
        _ax.axvline(0.0, color="0.5", lw=0.8, ls="--")
        _ax.set_title(_branch, fontsize=10)
        _ax.set_xlabel(r"$d_{ctrl}$  (+ = toward healthy)")
        _ax.spines[["top", "right"]].set_visible(False)
        _ax.grid(visible=True, linestyle="--", alpha=0.3)

    _axes[0].set_ylabel(r"$R^2$  (predictability)")
    _fig.legend(loc="lower center", ncol=6, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.12))
    _fig.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Curves over $h$

    The summary point above is one slice of these. The shaded band is $h <$ the metric's window,
    where scores are **not valid** — shown rather than hidden, so the artefact is visible instead
    of being silently cropped out.
    """)
    return


@app.cell
def _(archive, branch_names, mo, set_select):
    curve_branch = mo.ui.dropdown(options=branch_names, value=branch_names[0], label="Branch")
    curve_set = mo.ui.dropdown(options=archive.channel_sets, value=set_select.value[0], label="Channel set")
    mo.hstack([curve_branch, curve_set], justify="start", gap=2)
    return curve_branch, curve_set


@app.cell
def _(
    METRICS,
    archive,
    arm_select,
    controllability,
    curve_branch,
    curve_set,
    h_slider,
    mo,
    plt,
    predictability_r2,
    separability,
):
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4))

    for _i, (_name, _metric) in enumerate(METRICS.items()):
        _t = archive.times[_name]
        _zero = archive.ensemble("scalp", curve_branch.value, "zero", _name, curve_set.value)
        _stim = archive.ensemble("scalp", curve_branch.value, arm_select.value, _name, curve_set.value)
        _sep = separability(
            archive.ensemble("scalp", "healthy", "zero", _name, curve_set.value),
            archive.ensemble("scalp", "saturated", "zero", _name, curve_set.value),
        )
        _ctrl = controllability(_zero, _stim, direction=_sep.direction, gap=_sep.gap)

        _axes[0].plot(_t, predictability_r2(_zero), color=f"C{_i}", lw=1.4, label=_name)
        _axes[1].plot(_t, _ctrl.d_ctrl, color=f"C{_i}", lw=1.4, label=_name)

    _windows = max(_m.window_s for _m in METRICS.values())
    for _ax, _label in zip(_axes, [r"$R^2$", r"$d_{ctrl}$"], strict=True):
        _ax.axvspan(0.0, _windows, color="0.5", alpha=0.12)
        _ax.axvline(h_slider.value, color="k", lw=1.0, ls=":")
        _ax.axhline(0.0, color="0.5", lw=0.8)
        _ax.set_xlabel("lookahead h [s]")
        _ax.set_ylabel(_label)
        _ax.spines[["top", "right"]].set_visible(False)
        _ax.grid(visible=True, linestyle="--", alpha=0.3)

    _axes[0].set_title(f"predictability — {curve_branch.value} / {curve_set.value}")
    _axes[1].set_title(f"controllability — arm `{arm_select.value}`")
    _axes[1].legend(frameon=False, fontsize=8, ncol=2)
    _fig.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The rungs — what the metrics are ranked *against*

    The scores above rank the metrics against each other and never against the signal the
    predictor pipeline actually forecasts. These are that missing reference, on the same
    estimator and the same variance decomposition, averaged **across channels as $R^2$** rather
    than across channels as signal — a channel-mean of a signed waveform cancels.

    | rung | bounds a predictor that forecasts |
    | --- | --- |
    | **waveform** | $y(t)$ sample-wise — what the MLP/ESN pipeline does today |
    | **envelope** | 3–12 Hz amplitude, phase discarded (causal detector, *not* Hilbert) |
    | `eeg_ms` | broadband power over a 100 ms window |

    The gap between the first two is **the cost of phase divergence**, quantified. It is the
    number that says whether moving the MPC objective off the waveform buys anything at all. The
    $h$ axis is logarithmic because waveform predictability is expected to collapse inside a few
    hundred ms — which is why the baselines carry a 5 ms grid of their own rather than the
    metrics' 50 ms one.
    """)
    return


@app.cell
def _(
    METRICS,
    RAW,
    archive,
    curve_branch,
    curve_set,
    h_slider,
    mo,
    plt,
    predictability_r2,
):
    _fig, _ax = plt.subplots(figsize=(8, 4.5))

    for _i, _name in enumerate(METRICS):
        _ens = archive.ensemble("scalp", curve_branch.value, "zero", _name, curve_set.value)
        _ax.plot(archive.times[_name], predictability_r2(_ens), color=f"C{_i}", lw=1.1, alpha=0.65, label=_name)

    for _name, _style in (("waveform", "-"), ("envelope", "--")):
        _ax.plot(
            archive.baseline_times,
            archive.baseline(curve_branch.value, _name, curve_set.value, RAW),
            color="k",
            lw=1.2,
            ls=_style,
            label=_name,
        )

    _ax.axvline(h_slider.value, color="k", lw=1.0, ls=":")
    _ax.axhline(0.0, color="0.5", lw=0.8)
    # _ax.set_xscale("log")  # noqa: ERA001
    _ax.set_xlabel("lookahead h [s]")
    _ax.set_ylabel(r"$R^2$")
    _ax.set_title(f"predictability rungs — {curve_branch.value} / {curve_set.value}")
    _ax.legend(frameon=False, fontsize=8, ncol=2)
    _ax.spines[["top", "right"]].set_visible(False)
    _ax.grid(visible=True, which="both", linestyle="--", alpha=0.3)
    _fig.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Bandwidth — and why $R^2$ alone cannot rank it

    Each metric is re-scored on a causally low-passed signal, **at the plant's own 10 kHz rate**
    so that cutoff and sampling rate stay separable. Only the broadband metrics are swept:
    `band_power` and `spectral_centroid` are *redefined* by a low-pass rather than denoised by
    one, so their sweep would not be one metric measured at several bandwidths.

    Filtering strips the fast, child-specific component while leaving the slow, parent-determined
    one, so $\sigma_{ens}$ falls faster than the total variance and **$R^2$ rises monotonically**
    — to a degenerate optimum at DC, where the observable is perfectly predictable and perfectly
    useless. Two things make this a trade-off instead of a maximisation:

    - every axis is recomputed per cutoff, not just $R^2$, so a metric whose separability
      collapses as its predictability climbs is visible as such;
    - the x-axis is **effective latency** $=$ window $+$ group delay, measured at DC from the
      designed sections. Filtering is not free, and this is its price.

    Hollow markers are points where $h_{eval}$ is below that latency — the metric is not yet
    available, so the score there is an artefact of shared pre-branch history.
    """)
    return


@app.cell
def _(
    METRICS,
    RAW,
    SWEPT_METRICS,
    archive,
    arm_select,
    controllability,
    curve_branch,
    curve_set,
    design_lowpass_sos,
    group_delay_s,
    h_slider,
    manifest,
    np,
    pd,
    predictability_r2,
    separability,
):
    def _latency(name, cutoff):
        fs = manifest["fs"]
        delay = 0.0 if cutoff == RAW else group_delay_s(design_lowpass_sos(fs, float(cutoff)), fs)
        return METRICS[name].window_s + delay

    def _sweep_rows(h):
        rows = []
        for cutoff in archive.cutoffs:
            for name in SWEPT_METRICS:
                idx = int(np.argmin(np.abs(archive.times[name] - h)))
                sep = separability(
                    archive.ensemble("scalp", "healthy", "zero", name, curve_set.value, cutoff),
                    archive.ensemble("scalp", "saturated", "zero", name, curve_set.value, cutoff),
                )
                zero = archive.ensemble("scalp", curve_branch.value, "zero", name, curve_set.value, cutoff)
                stim = archive.ensemble("scalp", curve_branch.value, arm_select.value, name, curve_set.value, cutoff)
                ctrl = controllability(zero, stim, direction=sep.direction, gap=sep.gap)
                latency = _latency(name, cutoff)
                rows.append(
                    {
                        "metric": name,
                        "cutoff_hz": cutoff,
                        "latency_s": latency,
                        "R2": predictability_r2(zero)[idx],
                        "d_ctrl": ctrl.d_ctrl[idx],
                        "cohens_d": sep.cohens_d[idx],
                        "valid": archive.times[name][idx] >= latency,
                    }
                )
        return pd.DataFrame(rows)

    sweep = _sweep_rows(h_slider.value)
    return (sweep,)


@app.cell
def _(mo, sweep):
    mo.ui.table(sweep.round({"latency_s": 4, "R2": 3, "d_ctrl": 3, "cohens_d": 2}), page_size=12, selection=None)
    return


@app.cell
def _(
    SWEPT_METRICS,
    archive,
    arm_select,
    curve_branch,
    curve_set,
    h_slider,
    mo,
    np,
    plt,
    sweep,
):
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)

    for _i, _name in enumerate(SWEPT_METRICS):
        _sub = sweep[sweep["metric"] == _name].sort_values("latency_s")
        _ok = _sub["valid"].to_numpy()
        for _ax, _column in zip(_axes, ["R2", "d_ctrl"], strict=True):
            _ax.plot(_sub["latency_s"], _sub[_column], color=f"C{_i}", lw=1.3, alpha=0.8, label=_name)
            _ax.scatter(
                _sub["latency_s"],
                _sub[_column],
                s=42,
                color=np.where(_ok, f"C{_i}", "none"),
                edgecolors=f"C{_i}",
                zorder=3,
            )

    _wave = archive.baseline(curve_branch.value, "waveform", curve_set.value)
    _at_h = int(np.argmin(np.abs(archive.baseline_times - h_slider.value)))
    _axes[0].axhline(_wave[_at_h], color="k", lw=1.6, ls="--", label="waveform baseline")

    for _ax, _label in zip(_axes, [r"$R^2$", r"$d_{ctrl}$"], strict=True):
        _ax.axhline(0.0, color="0.5", lw=0.8)
        _ax.set_xscale("log")
        _ax.set_xlabel("effective latency = window + group delay [s]")
        _ax.set_ylabel(_label)
        _ax.spines[["top", "right"]].set_visible(False)
        _ax.grid(visible=True, which="both", linestyle="--", alpha=0.3)

    _axes[0].set_title(f"bandwidth vs predictability — {curve_branch.value} / {curve_set.value}")
    _axes[1].set_title(f"bandwidth vs controllability — arm `{arm_select.value}`")
    _axes[0].legend(frameon=False, fontsize=8)
    _fig.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Is the pairing actually buying power?

    The two arms share their child seed, so the plant sees the identical noise realisation and the
    difference $M_{stim} - M_{zero}$ is paired. The design expected `paired_sd` $\ll$ `sigma_ens`,
    making it a sensitive significance check that distinguishes a true null from an underpowered
    one.

    **Check that here rather than assuming it.** The plant is chaotic: shared noise keeps the arms
    coupled only while the trajectories have not yet separated, so the advantage decays with $h$
    and can invert once stimulation changes which cycle the trajectory settles into. Where the
    ratio is $\geq 1$, the paired column is *not* the more sensitive test.

    Either way the headline $d_{ctrl}$ is unaffected — it uses the **unpaired** `sigma_ens` on
    purpose, because a causal controller cannot know the noise realisation.
    """)
    return


@app.cell
def _(
    METRICS,
    archive,
    arm_select,
    controllability,
    curve_branch,
    curve_set,
    mo,
    np,
    plt,
    separability,
    sigma_ens,
):
    _fig, _ax = plt.subplots(figsize=(7, 4))

    for _i, _name in enumerate(METRICS):
        _zero = archive.ensemble("scalp", curve_branch.value, "zero", _name, curve_set.value)
        _stim = archive.ensemble("scalp", curve_branch.value, arm_select.value, _name, curve_set.value)
        _sep = separability(
            archive.ensemble("scalp", "healthy", "zero", _name, curve_set.value),
            archive.ensemble("scalp", "saturated", "zero", _name, curve_set.value),
        )
        _ctrl = controllability(_zero, _stim, direction=_sep.direction, gap=_sep.gap)
        _ax.plot(archive.times[_name], _ctrl.paired_sd / sigma_ens(_zero), color=f"C{_i}", lw=1.3, label=_name)

    _ax.axhline(1.0, color="k", lw=1.0, ls="--")
    _ax.set_yscale("log")
    _ax.set_xlabel("lookahead h [s]")
    _ax.set_ylabel(r"paired_sd / $\sigma_{ens}$")
    _ax.set_title("below 1 = pairing helps; at or above 1 = the coupling has decayed")
    _ax.legend(frameon=False, fontsize=8, ncol=2)
    _ax.spines[["top", "right"]].set_visible(False)
    _ax.grid(visible=True, which="both", linestyle="--", alpha=0.3)
    np.seterr(all="ignore")
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Feasibility and observability — the hard filters

    Three independent filters, not one blended number. **Feasibility** is the metric's own window:
    one needing a 1 s window is unusable at a 20 ms control rate however well it scores.
    **Observability** is the scalp↔region correlation, which catches metrics volume conduction has
    smeared away by the time they reach the scalp. **Separability** is the healthy-vs-saturated
    Cohen's $d$ — a metric that cannot tell the two states apart has nothing to control toward.
    """)
    return


@app.cell
def _(mo, scores):
    _filters = (
        scores.groupby(["metric", "channels"])
        .agg(
            window_s=("window_s", "first"),
            cohens_d=("cohens_d", "first"),
            observability=("observability", "median"),
            best_d_ctrl=("d_ctrl", "max"),
            best_R2=("R2", "max"),
        )
        .reset_index()
        .sort_values("best_d_ctrl", ascending=False)
    )
    mo.ui.table(_filters.round(3), page_size=15, selection=None)
    return


@app.cell
def _(manifest, mo, np):
    _seizing = np.array([p["n_seizing_final"] for p in manifest["parents"]])
    mo.md(f"""
    ## Basin split

    The plant is bimodal across seeds (`tes_field_geometry.md` §1.3): runs settle into ~4–5 seizing
    regions or ~27–35. `Var_across_parents` at the later branches is therefore partly *which basin
    the parent fell into* rather than within-phase variation. That is still the correct
    denominator — it genuinely is the uncertainty before knowing $x_0$ — but it is bimodal, and
    {manifest["n_parents"]} parents estimate it coarsely.

    Final seizing-region count per parent: **{list(_seizing)}**.

    Read the scores above with that spread in mind: where it is wide, the `mid_spread` and
    `saturated` ensembles are mixtures rather than one population.
    """)
    return


if __name__ == "__main__":
    app.run()
