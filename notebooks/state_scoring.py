import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium", app_title="Seizure-State Scoring")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy.signal import welch

    from neuro.ensembles import (
        RAW,
        SWEPT_METRICS,
        cached_scalp,
        eeg_path,
        load_manifest,
        region_path,
        score_ensemble_dir,
    )
    from neuro.filtering import design_lowpass_sos, group_delay_s
    from neuro.metrics import (
        METRICS,
        RAW_SERIES,
        Ensemble,
        controllability,
        coupling,
        latency_s,
        separability,
        sigma_ens,
        state_predictability_r2,
        state_predictability_snr_db,
        state_readout_r2,
        variance_ratio,
    )
    from neuro.seizure import EZ_REGIONS, SEIZURE_PTP_MV, SPREAD_WINDOW_S

    return (
        EZ_REGIONS,
        Ensemble,
        METRICS,
        Path,
        RAW,
        RAW_SERIES,
        SEIZURE_PTP_MV,
        SPREAD_WINDOW_S,
        SWEPT_METRICS,
        cached_scalp,
        controllability,
        coupling,
        design_lowpass_sos,
        eeg_path,
        group_delay_s,
        latency_s,
        load_manifest,
        mo,
        np,
        pd,
        plt,
        region_path,
        score_ensemble_dir,
        separability,
        sigma_ens,
        state_predictability_r2,
        state_predictability_snr_db,
        state_readout_r2,
        variance_ratio,
        welch,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # 🧠 Seizure-state scoring

    Which observable should the MPC minimise? This notebook answers it against a **measured
    ground truth** — the fraction of brain regions actually seizing — rather than against the
    labels attached to the ensemble or against the metrics themselves.

    Every candidate is scored on five axes, all of them variances of the same quantity:

    | axis | question |
    | --- | --- |
    | $R^2_{\text{read}}$ | does the observable see the seizure state? |
    | $R^2_{\text{pred}}$ / $\text{SNR}_{\text{dB}}$ | can it be forecast to the precision control needs? |
    | $d_{\text{ctrl}}$ | does the command move the observable? |
    | $d_{\text{state}}$ | does the command move the seizure? |
    | $\rho$ | does moving the observable move the seizure? |

    The candidates include the two signals the predictor pipeline is trained on today — the raw
    **waveform** and its band **envelope** — so "should the objective change at all?" is answered
    on the same axes as "which metric should it change to?".
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 Notation

    The generated ensemble is a four-level design. Naming the levels precisely is most of the
    work, because every score below is a variance taken over exactly one of them.

    | symbol | range | meaning |
    | --- | --- | --- |
    | $i$ | $1 \dots I$, $I = 16$ | **trajectory** — one independent realisation of the disease course, i.e. one plant seed |
    | $b$ | 5 values | **branch** — a time along that trajectory at which it was frozen |
    | $x_0^{\,i,b}$ | — | the **state**: trajectory $i$ at branch $b$, comprising $x$, the delay history, and the step counter |
    | $j$ | $1 \dots J$, $J = 8$ | **replicate** — one noise realisation rolled out from that state |
    | $a$ | $\{0,\; d_1,\; {-}d_1\}$ | **arm** — unstimulated, or the sustained hold $u = \pm[{+}2, 0, {-}2]$ mA |
    | $h$ | $0 \dots 3$ s | **lookahead** since the branch |
    | $c$ | $1 \dots C$, $C = 62$ | **channel** |
    | $n$ | $1 \dots N$, $N = 76$ | **region** |

    Replicates share their seed across arms, so $M^{0}_{ij}$ and $M^{d_1}_{ij}$ are the *same*
    noise realisation under two commands and their difference is paired.

    **Arms are allocated per branch**, not globally: `healthy` and `saturated` carry only the zero
    arm, `mid_spread` adds $d_1$, and only `pre_onset` and `ez_ignited` also carry $-d_1$. Every
    arm contrast below is therefore defined on a subset of the branches; the readout and
    predictability scores, which read the zero arm alone, are defined on all of them.

    Two observables are read off each rollout:

    $$M^{a}_{ij}(h) \in \mathbb{R}^{C} \qquad\text{the candidate metric, per channel}$$
    $$s^{a}_{ij}(h) \in [0,1] \qquad\text{the seizure state, one scalar for the network}$$

    ---

    **On "trajectory / state / replicate".** The generation code calls these parent and child.
    That names the *operation* — a run is snapshotted and branched — but not the *statistics*,
    which is what every formula below is about: a replicate is a repeat draw at fixed $x_0$, and a
    trajectory is an independent draw of $x_0$ itself. `Ensemble.by_state` and `n_replicates` use
    this vocabulary; `EnsembleConfig` still uses the generation one.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 The ground truth: $s(t)$

    A region counts as seizing when its local field potential swings more than $\theta = 5$ mV
    peak-to-peak within a trailing window of $W$ = `neuro.seizure.SPREAD_WINDOW_S` — the criterion
    `neuro.seizure.SpreadProfile` already uses to label spread, unchanged. The network state is the
    fraction of regions meeting it:

    $$s(t) \;=\; \frac{1}{N}\sum_{n=1}^{N} \mathbb{1}\!\left[\;\mathrm{PTP}_{n}\big(t-W,\,t\big) > \theta\;\right]$$

    So $s = 0$ is a healthy network and $s \approx 0.4$ is ~30 of 76 regions recruited.

    **Why one number for the network, and not one per channel.** The obvious refinement is to give
    each channel its own seizure state by weighting regions with that channel's row of the EEG
    forward operator, $s_c = \sum_n w_{cn} z_n / \sum_n w_{cn}$. Measured on this ensemble, $s_c$
    correlates with the single network scalar at a **median 0.996 (`pre_onset`), 0.995
    (`ez_ignited`), 0.982 (`mid_spread`)** across all 62 channels. Volume conduction is broad
    enough that every channel sees nearly the same weighted fraction. The channel axis carries no
    target information at the branches where stimulation does anything, so it is spent on the
    *metric* side instead, where §7 shows it does carry something.

    **$s$ is causal, so it needs history.** $W$ is inherited from the threshold's calibration — a
    shorter window measures less peak-to-peak and would silently re-tune $\theta$. Each region
    rollout is therefore stored with $W$ seconds of its parent's pre-branch history in front of it,
    so the first window closes exactly at the branch and $s$ is defined from $h = 0$. What that
    history cannot supply is **contrast**: it is one array per trajectory and branch, shared by
    every replicate *and every arm*, so any arm difference in $s$ is diluted below $h = W$. §6
    marks that band.
    """)
    return


@app.cell
def _(METRICS, Path, RAW_SERIES, load_manifest, mo, score_ensemble_dir):
    ensemble_dir = Path("data/predictability_ensemble")
    manifest = load_manifest(ensemble_dir)
    archive = score_ensemble_dir(ensemble_dir)

    fs = float(manifest["fs"])
    branch_names = [b["name"] for b in manifest["branches"]]
    seizure_branches = [b["name"] for b in manifest["branches"] if b["plant"] == "seizure"]
    channel_labels = manifest["channel_labels"]
    n_regions = len(manifest["region_labels"])

    # the windowed metrics and the two window-free series, scored on the same axes throughout
    OBSERVABLES = (*METRICS, *RAW_SERIES)

    def arm_branches(arm: str) -> list[str]:
        """Branches carrying this arm — arms are allocated per branch, not globally."""
        return [b["name"] for b in manifest["branches"] if arm in b["arms"]]

    mo.md(
        f"Loaded **{manifest['n_parents']} trajectories × {manifest['n_children']} replicates** over "
        f"{sum(len(b['arms']) for b in manifest['branches'])} (branch, arm) stores across "
        f"{len(branch_names)} branches, "
        f"{manifest['rollout_s']} s rollouts. "
        f"{len(OBSERVABLES)} observables: {', '.join(f'`{name}`' for name in OBSERVABLES)}. "
        f"Seizure state on a {len(archive.state_times)}-point grid, "
        f"{archive.state_times[0]:.2f}–{archive.state_times[-1]:.2f} s."
    )
    return (
        OBSERVABLES,
        archive,
        arm_branches,
        branch_names,
        channel_labels,
        ensemble_dir,
        fs,
        manifest,
        n_regions,
        seizure_branches,
    )


@app.cell
def _(OBSERVABLES, mo):
    metric_pick = mo.ui.multiselect(options=list(OBSERVABLES), value=list(OBSERVABLES), label="observables shown")
    channel_metric = mo.ui.dropdown(options=list(OBSERVABLES), value="eeg_ms", label="per-channel map of")
    pred_unit = mo.ui.radio(options=["R²", "SNR (dB)"], value="R²", label="predictability unit")
    h_slider = mo.ui.slider(start=0.1, stop=3.0, step=0.05, value=1.5, label="$h_{eval}$ (s)", show_value=True)
    n_bins = mo.ui.slider(start=4, stop=20, step=1, value=10, label="state bins", show_value=True)
    cut = mo.ui.slider(start=2, stop=30, step=1, value=12, label="recruited above (regions)", show_value=True)

    mo.vstack([metric_pick, mo.hstack([channel_metric, pred_unit, h_slider, n_bins, cut])])
    return channel_metric, cut, h_slider, metric_pick, n_bins, pred_unit


@app.cell
def _(OBSERVABLES, metric_pick):
    # everything is scored for every observable; the pick only decides what is drawn
    shown = tuple(name for name in OBSERVABLES if name in set(metric_pick.value))
    return (shown,)


@app.cell
def _(mo):
    mo.md(r"""
    ### What a branch actually is

    Before any score, the signals themselves. Each row below is one branch of **one** trajectory,
    left the scalp channel a controller would read and right the two focus regions the scalp is a
    blurred sum of. The rows are the same disease course frozen later and later — except `healthy`,
    which is the other plant and has no counterpart in the seizure rows.

    The point of the figure is the mismatch between the two columns. In region space the seizure is
    unmistakable: `lHC` crosses the 5 mV peak-to-peak criterion early and `lTCI` follows, which is
    exactly what $s(t)$ counts. On the scalp the same event is a change in amplitude and rhythmicity
    of one continuous trace, mixed with 60 other regions that are not seizing and buried in a noise
    floor no threshold cleanly separates. That gap — obvious in region space, arguable on the scalp
    — is what the rest of the notebook measures, one observable at a time.

    Move the trajectory slider: the plant is bimodal across seeds, and some trajectories never
    recruit anything at all, at any branch.
    """)
    return


@app.cell
def _(manifest, mo):
    traj = mo.ui.slider(
        start=0, stop=int(manifest["n_parents"]) - 1, step=1, value=0, label="trajectory $i$", show_value=True
    )
    traj
    return (traj,)


@app.cell
def _(
    EZ_REGIONS,
    SEIZURE_PTP_MV,
    SPREAD_WINDOW_S,
    branch_names,
    cached_scalp,
    ensemble_dir,
    manifest,
    np,
    plt,
    region_path,
    traj,
):
    # the store's focal channel set, and the EZ/PZ pair the scalp channel is loaded by
    _focal_set = next(name for name in manifest["channel_sets"] if name != "all62")
    _channel = manifest["channel_sets"][_focal_set][0]
    _regions = [manifest["region_labels"].index(name) for name in (EZ_REGIONS[0], _focal_set)]
    _row = traj.value * int(manifest["n_children"])

    _fig, _axes = plt.subplots(
        len(branch_names), 2, figsize=(11, 1.5 * len(branch_names)), sharex="col", sharey="col", squeeze=False
    )

    for _i, _b in enumerate(branch_names):
        _eeg = np.asarray(cached_scalp(ensemble_dir, _b, "zero", manifest)[_row, _channel], dtype=np.float64)
        _axes[_i, 0].plot(np.arange(len(_eeg)) / 1000.0, _eeg, lw=0.5, color="k")

        _region = np.load(region_path(ensemble_dir, _b, "zero"), mmap_mode="r")[_row][_regions]
        _t = np.arange(_region.shape[1]) / float(manifest["region_fs"]) - SPREAD_WINDOW_S
        for _j, _name in enumerate((EZ_REGIONS[0], _focal_set)):
            _axes[_i, 1].plot(_t, _region[_j], lw=0.6, color=f"C{_j}", label=_name if _i == 0 else None)

        # the pre-branch history the seizure state's first window is read off
        _axes[_i, 1].axvspan(-SPREAD_WINDOW_S, 0.0, color="0.92", zorder=0)
        for _ax in _axes[_i]:
            _ax.set_ylabel(_b, fontsize=8)
            _ax.spines[["top", "right"]].set_visible(False)

    # the criterion is a swing, not a level, so it is a scale bar in the margin and not a band
    _axes[0, 1].set_xlim(-SPREAD_WINDOW_S - 0.35, float(manifest["rollout_s"]))
    _lo, _hi = _axes[0, 1].get_ylim()
    for _ax in _axes[:, 1]:
        _base = _lo + 0.08 * (_hi - _lo)
        _ax.plot([-SPREAD_WINDOW_S - 0.2] * 2, [_base, _base + SEIZURE_PTP_MV], color="tab:red", lw=3)

    _axes[0, 0].set_title(f"scalp `{manifest['channel_labels'][_channel]}` (mV) — what the metrics see", fontsize=9)
    _axes[0, 1].set_title(
        f"region LFP (mV) — what $s$ counts; red bar is the {SEIZURE_PTP_MV:g} mV PTP criterion", fontsize=9
    )
    _axes[0, 1].legend(fontsize=7, loc="upper left")
    _axes[-1, 0].set_xlabel("lookahead $h$ (s)")
    _axes[-1, 1].set_xlabel("lookahead $h$ (s) — shaded: the shared pre-branch history")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(archive, branch_names, np, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.6))

    for _b in branch_names:
        _s = archive.state(_b, "zero")
        _axes[0].plot(archive.state_times, _s.values.mean(axis=0), label=_b, lw=2)
        _axes[0].fill_between(
            archive.state_times,
            np.percentile(_s.values, 25, axis=0),
            np.percentile(_s.values, 75, axis=0),
            alpha=0.15,
        )

    _axes[0].set_xlabel("lookahead $h$ (s)")
    _axes[0].set_ylabel("$s(h)$ — fraction of regions seizing")
    _axes[0].set_title("Seizure state per branch (zero arm, IQR band)")
    _axes[0].legend(fontsize=8)

    for _b in branch_names:
        _s = archive.state(_b, "zero").values[:, -1]
        _axes[1].scatter(np.full_like(_s, branch_names.index(_b)) + 0.06 * np.random.randn(len(_s)), _s, s=4, alpha=0.4)

    _axes[1].set_xticks(range(len(branch_names)))
    _axes[1].set_xticklabels(branch_names, rotation=30, ha="right", fontsize=8)
    _axes[1].set_ylabel("$s$ at $h = 3$ s")
    _axes[1].set_title("Per-rollout spread — branches overlap heavily")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(archive, branch_names, mo, n_regions):
    _rows = []
    for _b in branch_names:
        _v = archive.state(_b, "zero").values[:, -1]
        _rows.append(f"| `{_b}` | {_v.mean() * n_regions:.1f} | {_v.std() * n_regions:.1f} |")

    mo.md(
        "### The branch label is a weak proxy for the state\n\n"
        "Regions seizing at $h = 3$ s, mean ± sd across all rollouts:\n\n"
        "| branch | mean | sd |\n| --- | --- | --- |\n" + "\n".join(_rows) + "\n\n"
        "The within-branch spread is comparable to the between-branch separation, and `pre_onset` "
        "reaches double digits during its own rollout. **This is the case for scoring against "
        "$s$ rather than against $b$**: conditioning on the branch does not hold the seizure "
        "state fixed, so a per-branch statistic mixes states that a controller would treat "
        "completely differently."
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 2.1 Grouping by the state instead of by the branch

    So the branch is dropped as a grouping variable. Everything below that needs a group uses the
    **seizure state at the branch itself**:

    $$s_0^{\,i,b} \;=\; s^{a}_{ij}(0) \qquad\text{regions seizing when the rollout starts}$$

    Three properties make this the right label rather than a cosmetic relabelling:

    - It is a property of $x_0$ **exactly**. $s(0)$ is read off the shared pre-branch window, so
      every replicate of a trajectory and every arm of it carry the identical value — measured, not
      assumed. Cutting on it therefore moves whole trajectories, leaving the replicate grouping and
      the paired arm contrast intact.
    - It is what a controller knows. The state at $h = 0$ is the information the MPC conditions on;
      grouping by anything later would score the observable against a future it cannot see.
    - It splits the **basin mixture** rather than averaging over it. The plant is bimodal across
      seeds: some trajectories ignite and recruit 25–35 regions, others sit at 3–8 indefinitely,
      and `mid_spread` and `saturated` each contain both. Under the branch label that mixture is
      pooled into one mean; under $s_0$ it is the grouping variable.

    The distribution of $s_0$ below is bimodal with a wide empty gap, so one cut is enough and its
    placement inside the gap is not delicate.
    """)
    return


@app.cell
def _(Ensemble, RAW, archive, cut, n_regions, np):
    STRATA = ("focal", "recruited")

    def branch_s0(branch: str):
        """Regions seizing at the branch instant, one value per trajectory."""
        return archive.state(branch, "zero").by_state[:, 0, 0] * n_regions

    def stratum_of(branch: str):
        """Stratum index per trajectory of one branch, from the seizure state at $h = 0$."""
        return (branch_s0(branch) > cut.value).astype(int)

    def restrict(ens: Ensemble, keep) -> Ensemble:
        """Keep whole trajectories, so the replicate grouping and the arm pairing survive the cut."""
        return Ensemble(ens.times, ens.by_state[keep].reshape(-1, *ens.values.shape[1:]), ens.n_replicates)

    def stack(parts: list[Ensemble]) -> Ensemble:
        """Join ensembles sharing a grid — states from several branches, replicates intact."""
        return Ensemble(parts[0].times, np.concatenate([p.values for p in parts]), parts[0].n_replicates)

    def stratum_metric(stratum, arm, metric, channel_set, branches, cutoff=RAW, *, pool=True) -> Ensemble:
        """One observable over every trajectory in a stratum, pooled across the branches it spans."""
        return stack(
            [
                restrict(archive.ensemble(b, arm, metric, channel_set, cutoff, pool=pool), stratum_of(b) == stratum)
                for b in branches
            ]
        )

    def stratum_state(stratum, arm, branches) -> Ensemble:
        """The seizure state over every trajectory in a stratum, row-aligned with `stratum_metric`."""
        return stack([restrict(archive.state(b, arm), stratum_of(b) == stratum) for b in branches])

    def populated(branches) -> list[int]:
        """Strata carrying at least one trajectory among `branches`."""
        return [s for s in range(len(STRATA)) if sum(int((stratum_of(b) == s).sum()) for b in branches) > 0]

    return (
        STRATA,
        branch_s0,
        populated,
        stratum_metric,
        stratum_of,
        stratum_state,
    )


@app.cell
def _(STRATA, branch_s0, cut, np, plt, seizure_branches, stratum_of):
    _fig, _ax = plt.subplots(figsize=(7.5, 3.4))

    for _i, _b in enumerate(seizure_branches):
        _s0 = branch_s0(_b)
        _colour = np.where(stratum_of(_b) == 1, "tab:red", "tab:blue")
        _ax.scatter(_s0, np.full_like(_s0, _i) + 0.08 * np.random.randn(len(_s0)), c=_colour, s=22, alpha=0.8)

    _ax.axvline(cut.value, color="k", lw=1.0, ls="--")
    _ax.set_yticks(range(len(seizure_branches)))
    _ax.set_yticklabels(seizure_branches, fontsize=8)
    _ax.set_xlabel("$s_0$ — regions seizing at the branch ($h = 0$)")
    _ax.set_title(f"one trajectory per point; dashed line is the cut at {cut.value} regions", fontsize=9)
    _ax.spines[["top", "right"]].set_visible(False)
    _fig.suptitle(f"blue = {STRATA[0]}, red = {STRATA[1]}", fontsize=9)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(STRATA, arm_branches, mo, seizure_branches, stratum_of):
    _header = "| stratum | " + " | ".join(f"`{b}`" for b in seizure_branches) + " | total | probed |\n"
    _header += "| --- | " + " | ".join(["---"] * (len(seizure_branches) + 2)) + " |\n"
    _rows = []
    for _s, _name in enumerate(STRATA):
        _per_branch = [int((stratum_of(b) == _s).sum()) for b in seizure_branches]
        _probed = sum(int((stratum_of(b) == _s).sum()) for b in arm_branches("d1"))
        _rows.append(
            f"| **{_name}** | " + " | ".join(str(n) for n in _per_branch) + f" | {sum(_per_branch)} | {_probed} |"
        )

    mo.md(
        "Trajectories per stratum, and how many of them sit at a branch that carries the $d_1$ probe:\n\n"
        + _header
        + "\n".join(_rows)
        + "\n\nThe last column is what makes this worth doing for the control axes. `saturated` "
        "carries no stimulation arm, so under the branch grouping the heavily-recruited states "
        "were unprobed by construction. Grouped by $s_0$ they are not: `ez_ignited` and "
        "`mid_spread` between them contribute recruited trajectories that **do** carry both a "
        "$d_1$ and a zero rollout, so the recruited condition gets a real arm contrast."
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 The variance decomposition

    Everything below is one of three variances of the same metric. Write $\mathrm{Var}$ for the
    variance over all unstimulated rollouts and all times, pooled.

    **Split 1 — by seizure state.** Conditioning on $s$ and applying the law of total variance:

    $$\mathrm{Var}\big[M\big] \;=\; \underbrace{\mathrm{Var}_s\!\big(\mathbb{E}[M \mid s]\big)}_{V_{\text{state}}}
    \;+\; \underbrace{\mathbb{E}_s\!\big[\mathrm{Var}(M \mid s)\big]}_{V_{\text{resid}}}$$

    $V_{\text{state}}$ is the part of the metric that *is* the seizure — the signal a controller
    wants. $V_{\text{resid}}$ is everything the metric does that the seizure state does not
    explain. Estimated by binning $s$ into deciles, so no functional form is imposed and a
    non-monotone metric is not penalised for being non-monotone.

    **Split 2 — by branch state.** Conditioning instead on $x_0$, at each lookahead:

    $$V_{\text{rep}}(h) \;=\; \frac{1}{\#\{i,b\}}\sum_{i,b}\;\mathrm{Var}_{j}\!\big[M^{0}_{ij}(h)\big]$$

    This is what remains unknown *after* fixing the full plant state, so it is the irreducible
    forecast error — a ceiling no predictor can beat, and model-free, so it does not depend on
    the state of the predictor pipeline. It grows with $h$ as trajectories diverge.

    ---

    The two splits answer the two questions. Question 1 (*what correlates with seizure state?*)
    is $V_{\text{state}}$ against the total. Question 2 (*what can we predict?*) is
    $V_{\text{rep}}(h)$ against $V_{\text{state}}$ — **not** against the total.

    That choice of denominator is what makes the score comparable across groups. The spread of
    whatever states a group's trajectories happen to occupy is an accident of the draw: in the
    focal stratum they are nearly identical and it is tiny, in the recruited one they are spread
    and it is large, so a score referred to it cannot be read across the two. $V_{\text{state}}$ is
    one number for the whole ensemble, and it is the quantity a controller has to resolve.
    """)
    return


@app.cell
def _(archive, np):
    def align(metric: str):
        """Index the metric grid at each seizure-state time, so the two line up sample for sample."""
        return np.array([int(np.argmin(np.abs(archive.times[metric] - t))) for t in archive.state_times])

    def observations(metric: str, branches, channel_set: str = "all62", arm: str = "zero"):
        """Stack (metric, state) pairs over branches, rollouts and times into a flat design.

        Returns ``(M, s)`` with ``M`` of shape ``(n_obs, n_channels)`` and ``s`` of ``(n_obs,)``.
        Rollout-major so the two orders match exactly.
        """
        idx = align(metric)
        metric_rows, state_rows = [], []
        for branch in branches:
            values = archive.ensemble(branch, arm, metric, channel_set, pool=False).values[:, :, idx]
            state = archive.state(branch, arm).values
            metric_rows.append(values.transpose(0, 2, 1).reshape(-1, values.shape[1]))
            state_rows.append(state.reshape(-1))
        return np.concatenate(metric_rows), np.concatenate(state_rows)

    return align, observations


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 Axis 1 — does the observable read the seizure state?

    $$R^2_{\text{read}} \;=\; \frac{V_{\text{state}}}{\mathrm{Var}[M]} \;=\; 1 - \frac{\mathbb{E}_s[\mathrm{Var}(M\mid s)]}{\mathrm{Var}[M]}$$

    Pooled over the four seizure branches and the whole time grid, on the **zero arm only** — the
    question is what the natural history of the observable looks like, and including the stimulated
    arm would mix control-induced variance into the state range.

    Pooling across branches is deliberate here and would be wrong for §5. For a *readout*, the
    swing between operating points is the signal; for *predictability*, it is a confound. The two
    questions want opposite pooling, which is why they are separate cells rather than one number.

    `separability`, the standardised healthy-to-saturated contrast, is not scored as an axis of its
    own — it is the same question asked with two branches instead of the whole range. It stays in
    use for one thing this score cannot give: the **sign**, i.e. which direction along the metric
    is recovery, which §6 needs before $d_{\text{ctrl}}$ means anything.

    **On the two window-free series.** `waveform` is signed, so its channel mean partly cancels;
    read its `best single channel` column, not its pooled one. That column scores surprisingly
    well, and the next cell shows it is not what it looks like.
    """)
    return


@app.cell
def _(
    fs,
    latency_s,
    n_bins,
    np,
    observations,
    pd,
    seizure_branches,
    shown,
    state_readout_r2,
):
    def readout_table(names, bins: int) -> pd.DataFrame:
        rows = []
        for name in names:
            m_all, s_all = observations(name, seizure_branches, "all62")
            m_foc, _ = observations(name, seizure_branches, "lTCI")
            pooled = state_readout_r2(m_all.mean(axis=1), s_all, n_bins=bins)
            focal = state_readout_r2(m_foc.mean(axis=1), s_all, n_bins=bins)
            per_channel = state_readout_r2(m_all, s_all, n_bins=bins)
            rows.append(
                {
                    "metric": name,
                    "latency_s": latency_s(name, fs),
                    "R2_read all62": pooled.r2,
                    "R2_read lTCI": focal.r2,
                    "best single channel": float(np.nanmax(per_channel.r2)),
                    "V_state (all62)": pooled.explained_var,
                }
            )
        return pd.DataFrame(rows).sort_values("R2_read all62", ascending=False)

    readout = readout_table(shown, n_bins.value)
    readout
    return (readout_table,)


@app.cell
def _(
    align,
    archive,
    mo,
    n_bins,
    np,
    observations,
    seizure_branches,
    state_readout_r2,
):
    def dc_free(metric: str, branches):
        """The same observations with each rollout's own level removed, per channel.

        What any real amplifier's high-pass does to the signal before the controller ever sees it.
        """
        idx = align(metric)
        rows = []
        for branch in branches:
            values = archive.ensemble(branch, "zero", metric, "all62", pool=False).values[:, :, idx]
            values = values - values.mean(axis=2, keepdims=True)
            rows.append(values.transpose(0, 2, 1).reshape(-1, values.shape[1]))
        return np.concatenate(rows)

    _m, _s = observations("waveform", seizure_branches, "all62")
    _raw = state_readout_r2(_m, _s, n_bins=n_bins.value)
    _levelled = state_readout_r2(dc_free("waveform", seizure_branches), _s, n_bins=n_bins.value)
    _best = int(np.nanargmax(_raw.r2))
    _swing = _raw.bin_metric[:, _best]

    mo.md(
        "### The `waveform` row is a level shift, not a reading\n\n"
        f"On its best channel (`{archive.manifest['channel_labels'][_best]}`) the raw signal scores "
        f"$R^2_{{read}} = {np.nanmax(_raw.r2):.3f}$, which for a zero-mean oscillation should be "
        "impossible. It is not the oscillation. $\\mathbb{E}[M \\mid s]$ on that channel runs from "
        f"**{_swing[0]:+.1f} mV** at the lowest state bin to **{_swing[-1]:+.1f} mV** at the "
        "highest: the operating point of the local field drifts as regions are recruited, and the "
        "forward operator passes that drift to the scalp as a slow level change.\n\n"
        "Remove each rollout's own level — one subtraction, and less than any real amplifier's "
        f"high-pass does — and the same score falls to **{np.nanmax(_levelled.r2):.3f}**. So the "
        "incumbent objective's apparent readout is a DC component no recording chain would deliver, "
        "and `waveform` should be read as scoring near zero on this axis.\n\n"
        "**This does not carry over to the windowed metrics.** `block_ptp` and `line_length` are "
        "blind to a constant offset by construction — a peak-to-peak and a mean absolute difference "
        "both annihilate DC — and they still score in the mid-0.8s. The seizure state really is "
        "readable from the oscillation. `eeg_ms` is the one to keep an eye on: a mean square "
        "includes the offset squared, so part of its score is the same level shift, and the two "
        "DC-blind metrics are its honest comparison."
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### How to read the two panels below

    **Left — the calibration curve $\mathbb{E}[M \mid s]$.** The $x$ axis is the mean seizure state
    inside each quantile bin of $s$; the $y$ axis is the mean of the observable over the rollouts
    in that bin. Each curve is min–max scaled onto $[0, 1]$ so that observables in mV², in Hz and
    dimensionless can share one axis — the scaling destroys the magnitudes on purpose and keeps
    only the **shape**, which is the thing being compared.

    A curve that rises steeply and monotonically is an observable that separates seizure states.
    A flat stretch is a range of $s$ the observable cannot tell apart, whatever it does elsewhere;
    a fold-back is an observable that maps two different states onto the same value, which is worse
    than flat for a controller because the gradient points the wrong way there. $R^2_{\text{read}}$
    is essentially the vertical spread of this curve measured against the scatter around it, so the
    table and this panel say the same thing — the panel says *where* along the range it is earned.

    **Right — where on the scalp the readout lives.** One bar per EEG channel, the same
    $R^2_{\text{read}}$ computed on that channel alone, sorted best to worst. A flat profile means
    every electrode sees the seizure equally and montage choice is free; a steep one means the
    readout is carried by a handful of channels and the focal set is worth using. Compare its
    height to the `all62` row in the table above: where the best channel beats the channel mean,
    pooling is *costing* information rather than averaging noise away.
    """)
    return


@app.cell
def _(
    channel_labels,
    channel_metric,
    n_bins,
    np,
    observations,
    plt,
    seizure_branches,
    shown,
    state_readout_r2,
):
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.8))

    for _name in shown:
        _m, _s = observations(_name, seizure_branches, "all62")
        _sc = state_readout_r2(_m.mean(axis=1), _s, n_bins=n_bins.value)
        _span = np.ptp(_sc.bin_metric)
        _norm = (_sc.bin_metric - _sc.bin_metric.min()) / _span if _span > 0 else np.zeros_like(_sc.bin_metric)
        _axes[0].plot(_sc.bin_state, _norm, "o-", label=_name, ms=4)

    _axes[0].set_xlabel("seizure state $s$")
    _axes[0].set_ylabel("observable (min–max scaled)")
    _axes[0].set_title("Calibration: $\\mathbb{E}[M \\mid s]$")
    _axes[0].legend(fontsize=7)

    _m, _s = observations(channel_metric.value, seizure_branches, "all62")
    _per = state_readout_r2(_m, _s, n_bins=n_bins.value)
    _order = np.argsort(_per.r2)[::-1]
    _axes[1].bar(range(len(_order)), _per.r2[_order], color="tab:blue")
    _axes[1].set_xticks(range(0, len(_order), 4))
    _axes[1].set_xticklabels([channel_labels[c] for c in _order[::4]], rotation=90, fontsize=6)
    _axes[1].set_ylabel("$R^2_{read}$")
    _axes[1].set_title(f"`{channel_metric.value}` per channel — which channels see the seizure", fontsize=9)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 Axis 2 — can it be predicted, to the precision control needs?

    $$R^2_{\text{pred}}(h) \;=\; 1 - \frac{V_{\text{rep}}(h)}{V_{\text{state}}}, \qquad \text{SNR}_{\text{dB}}(h) \;=\; 10 \log_{10}\!\left(\frac{V_{\text{state}}}{V_{\text{rep}}(h)}\right) \;=\; -10 \log_{10}\!\big(1 - R^2_{\text{pred}}(h)\big)$$

    The numerator is the spread that survives fixing $x_0$ — the irreducible forecast error —
    against a **stated** denominator instead of an incidental one. It reads as: *is my forecast
    error small compared to the state difference I am trying to steer between?*

    It is **unbounded below**, and that is not a defect. $R^2_{\text{pred}} < 0$ (or equivalently
    $\text{SNR} < 0\text{ dB}$) says the observable's irreducible noise at lookahead $h$ is wider than the span it
    traverses across the seizure states — so at that horizon it cannot distinguish the states the
    controller exists to move between, however smooth its curve looks. $0\text{ dB}$ ($R^2_{\text{pred}} = 0$)
    marks the break-even threshold where forecast noise equals the state range; $+10\text{ dB}$ corresponds to
    $R^2_{\text{pred}} = 0.90$.

    One curve per stratum, because $V_{\text{rep}}$ is a within-state quantity and the two strata
    are different plants in every respect but the parameter vector: a focal trajectory has 3–8
    regions oscillating and a recruited one has 25–35, and their replicate spreads have no reason
    to match.

    **`healthy` is out of the picture entirely**, here and in §4. It is a different plant —
    $A = A_{\text{healthy}}$ rather than the EZ/PZ gain vector — so pooling it in would let an
    observable draw denominator from a parameter change no electrode can produce, and flatter every
    score by a span that is not reachable. The strata are cut over the seizure plant alone for the
    same reason: a stratum is a range of *states*, and `healthy` is not a state of this plant.
    `pre_onset` already anchors $s \approx 0$ under the plant the controller actually faces.

    Curves run over the whole grid. Within-state variance genuinely collapses as $h \to 0$ —
    every replicate branches from one $x_0$ and has not had time to diverge — so
    $R^2_{\text{pred}} \to 1$ ($\text{SNR} \to +\infty$) there is the plant being deterministic at
    short range, not an artefact. Only each observable's own latency is masked.
    """)
    return


@app.cell
def _(
    STRATA,
    archive,
    fs,
    latency_s,
    n_bins,
    np,
    observations,
    plt,
    populated,
    pred_unit,
    seizure_branches,
    shown,
    state_predictability_r2,
    state_predictability_snr_db,
    state_readout_r2,
    stratum_metric,
):
    _is_db = pred_unit.value == "SNR (dB)"
    _cols = min(3, max(1, len(shown)))
    _rows = int(np.ceil(len(shown) / _cols))
    _fig, _axes = plt.subplots(_rows, _cols, figsize=(4 * _cols, 3 * _rows), sharex=True, squeeze=False)

    for _ax, _name in zip(_axes.ravel(), shown, strict=False):
        _m, _s = observations(_name, seizure_branches, "all62")
        _v_state = state_readout_r2(_m.mean(axis=1), _s, n_bins=n_bins.value).explained_var
        _t = archive.times[_name]
        _valid = _t >= latency_s(_name, fs)

        for _stratum in populated(seizure_branches):
            _ens = stratum_metric(_stratum, "zero", _name, "all62", seizure_branches)
            _score = state_predictability_snr_db(_ens, _v_state) if _is_db else state_predictability_r2(_ens, _v_state)
            _ax.plot(_t[_valid], _score[_valid], lw=2, label=STRATA[_stratum])

        _ax.axhline(0.0, color="k", lw=0.6)
        if _is_db:
            _ax.set_ylim(-15.0, 25.0)
        else:
            _ax.set_ylim(-1.5, 1.05)
        _ax.set_title(f"{_name}  (latency {latency_s(_name, fs):.3g} s)", fontsize=9)
        _ax.set_xlabel("$h$ (s)")

    for _ax in _axes.ravel()[len(shown) :]:
        _ax.set_visible(False)
    for _ax in _axes[:, 0]:
        _ax.set_ylabel("SNR (dB)" if _is_db else "$R^2$")
    _axes[0, 0].legend(fontsize=7)
    _fig.suptitle(f"{'SNR (dB)' if _is_db else '$R^2_{pred}$'} against $V_{{state}}$, per stratum", fontsize=10)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 Axes 3–5 — the three control questions

    One control question is not enough; the repo's history says these three come apart.

    $$\text{(3) does }u\text{ move the observable?}\qquad
    d_{\text{ctrl}}(h) = \mathrm{dir}\cdot\frac{\bar\delta_M(h)}{\sigma_{\text{rep}}(h)},
    \quad \delta_{M,ij} = M^{d_1}_{ij} - M^{0}_{ij}$$

    $$\text{(4) does }u\text{ move the seizure?}\qquad
    d_{\text{state}}(h) = -\frac{\bar\delta_s(h)}{\sigma_{\text{rep}}\!\big(s^{0}\big)},
    \quad \delta_{s,ij} = s^{d_1}_{ij} - s^{0}_{ij}$$

    $$\text{(5) does moving the observable move the seizure?}\qquad
    \rho(h) = \mathrm{corr}_{ij}\!\big(\delta_{M,ij}(h),\; \delta_{s,ij}(h)\big)$$

    ### Every symbol in those three lines

    | symbol | what it is | why it is that and not something else |
    | --- | --- | --- |
    | $\delta_{M,ij}$ | **paired** difference in the observable: replicate $j$ of trajectory $i$ under $d_1$ minus the *same* replicate under the zero arm | the two arms share a noise seed, so subtracting removes the noise realisation and leaves the command's effect |
    | $\delta_{s,ij}$ | the same difference in the seizure state, one scalar per rollout | pairs one-for-one with $\delta_{M,ij}$, which is what makes $\rho$ a within-rollout quantity |
    | $\bar\delta$ | mean over all rollouts $(i,j)$ in the group | the average effect of the command |
    | $\sigma_{\text{rep}}$ | **unpaired** within-state spread: $\sqrt{\mathbb{E}_i\,\mathrm{Var}_j[\,\cdot\,]}$ over replicates of a fixed $x_0$, `neuro.metrics.sigma_ens` | the noise a controller is actually looking through — see the note below |
    | $\mathrm{dir}$ | $\pm1$ from `separability`: the sign that makes "toward healthy" positive | measured from the healthy-to-saturated contrast, not assumed, because a metric can be driven confidently in the direction that makes things worse |
    | $d_{\text{ctrl}}$ | effect on the observable in units of $\sigma_{\text{rep}}$ | dimensionless, so mV² and Hz can be ranked together |
    | $d_{\text{state}}$ | effect on the seizure in units of the seizure's own $\sigma_{\text{rep}}$ | negated, so positive means *fewer* regions seizing |
    | $\rho$ | correlation across rollouts of the two $\delta$'s | both $\delta$'s are centred first, so $\rho$ is blind to the mean effect that $d_{\text{ctrl}}$ and $d_{\text{state}}$ already report |

    **Why both $d$'s divide by the unpaired spread.** A controller cannot know which noise
    realisation it is in, so the effect it must steer against is buried in $\sigma_{\text{rep}}$.
    The paired $\mathrm{sd}(\delta)$ is far smaller — the arms share their noise — and dividing by
    it would license calling an observable controllable when the effect is invisible to any causal
    controller. The paired spread is the right *significance* test and the wrong *controllability*
    score; the table below reports a $t$ statistic for the first job.

    **$d_{\text{state}}$ is observable-independent — it is a property of the probe.** Read it
    first. Where the command does not move the seizure, every $d_{\text{ctrl}}$ in that group is
    scoring the response to a null actuator, and no ranking of observables there means anything.

    **Why $\rho$ is scored at all.** `tes_field_geometry.md` §2.3 is the case that forces it: the
    objective was driven 57 % in the intended direction while the regional seizure count stayed at
    31/34. $d_{\text{ctrl}}$ was large, $d_{\text{state}}$ was zero, and no combination of the
    other four axes would have caught it.

    **$d_{\text{ctrl}}$ does not apply to `waveform`.** It needs $\mathrm{dir}$, and $\mathrm{dir}$
    comes from the healthy-to-saturated *level* difference — which a zero-mean oscillation does not
    have. Its gap is a rounding residue and its sign is arbitrary, so read `waveform`'s
    $d_{\text{ctrl}}$ as undefined rather than as a measurement, and use $\rho$ (which needs no
    direction) for that row. The envelope is signed and non-negative, so it has a real direction.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Two masks, and they are not the same mask

    §5 plots $R^2_{\text{pred}}$ over the full $h$ range on purpose. Its numerator really does
    vanish as $h \to 0$, because every replicate branches from one $x_0$ and $s$ is continuous —
    that is the plant being deterministic at short range, a measurement and not an artefact.

    The two $\delta_s$ scores are the opposite case. The pre-branch history $s$ is read through is
    **one array per trajectory and branch**, shared by every replicate and every arm, so below
    $h = W$ the trailing window is part bit-identical samples and $\delta_s$ is diluted in
    proportion to the overlap — exactly zero at $h = 0$, half strength at $h = W/2$. That is a
    smooth ramp up from zero that reads exactly like a stimulation latency and is pure window
    overlap, so $d_{\text{state}}$ and $\rho$ are **marked** below $W$, not deleted.

    $d_{\text{ctrl}}$ is **not** in that band. The scalp store carries no pre-branch history at
    all, so an observable's window is post-branch from its first sample and its own latency already
    covers it. Its small values at short $h$ are latency, and real.
    """)
    return


@app.cell
def _(
    STRATA,
    arm_branches,
    n_regions,
    np,
    pd,
    populated,
    sigma_ens,
    stratum_state,
):
    def state_response() -> pd.DataFrame:
        """One row per stratum and probe arm at the last lookahead; empty combinations drop out."""
        rows = []
        for arm in ("d1", "-d1"):
            branches = arm_branches(arm)
            for stratum in populated(branches):
                zero = stratum_state(stratum, "zero", branches)
                spread = sigma_ens(zero)
                delta = stratum_state(stratum, arm, branches).values - zero.values
                # per-trajectory means: replicates from one state are not independent draws
                per_traj = delta.reshape(zero.n_states, zero.n_replicates, -1).mean(axis=1)
                # n = 1 has no t, and at h = 0 the shared pre-branch window makes every delta_s
                # identically zero, so the standard error vanishes there too
                mean = per_traj.mean(axis=0)
                se = per_traj.std(axis=0, ddof=1) / np.sqrt(zero.n_states) if zero.n_states > 1 else np.zeros_like(mean)
                t_stat = np.divide(mean, se, out=np.full_like(mean, np.nan), where=se > 0)
                rows.append(
                    {
                        "stratum": STRATA[stratum],
                        "arm": arm,
                        "trajectories": zero.n_states,
                        "s (zero)": zero.values[:, -1].mean(),
                        "delta regions": delta[:, -1].mean() * n_regions,
                        "d_state": -delta[:, -1].mean() / spread[-1],
                        "t": t_stat[-1],
                        "trajectories suppressed": f"{int((per_traj[:, -1] < 0).sum())}/{zero.n_states}",
                    }
                )
        return pd.DataFrame(rows)

    d_state = state_response()
    d_state
    return (d_state,)


@app.cell
def _(d_state, mo):
    _worst = d_state.loc[d_state["t"].idxmin()]
    mo.md(
        "### Read this table before any ranking of observables\n\n"
        f"The probe's strongest effect is at **{_worst['stratum']} / `{_worst['arm']}`**: "
        f"{_worst['delta regions']:+.1f} regions, $t_{{({int(_worst['trajectories']) - 1})}} = {_worst['t']:.2f}$, "
        f"{_worst['trajectories suppressed']} trajectories suppressed. Where $|t|$ is small the "
        "actuator is doing nothing to the seizure, and $d_{ctrl}$ and $\\rho$ in that group are "
        "describing an ensemble in which there is no effect to attribute.\n\n"
        "Read the two arms of a stratum as a pair: a command that steers the seizure should "
        "reverse $d_{state}$ when it reverses, and one that moves $s$ the same way under both is "
        "not steering it. That is the reading the reversed arm was bought for. The $-d_1$ rows "
        "cover fewer trajectories than the $d_1$ rows, because only `pre_onset` and `ez_ignited` "
        "carry that arm."
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### The same two scores over the horizon

    The table reads $d_{\text{state}}$ at $h = 3$ s and §7 reads $\rho$ at the slider; both are
    curves, and the band below $W$ is shaded rather than clipped so the dilution ramp stays
    visible as the artefact it is. $\rho$ is shown for the $d_1$ arm at the stratum whose
    $d_{\text{state}}$ is largest outside that band, since everywhere else both $\delta$'s are
    noise.
    """)
    return


@app.cell
def _(
    Ensemble,
    SPREAD_WINDOW_S,
    STRATA,
    align,
    archive,
    arm_branches,
    coupling,
    np,
    plt,
    populated,
    shown,
    sigma_ens,
    stratum_metric,
    stratum_state,
):
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.6), sharex=True)
    _undiluted = archive.state_times >= SPREAD_WINDOW_S
    _curves = {}

    for _arm, _style in (("d1", "-"), ("-d1", "--")):
        _branches = arm_branches(_arm)
        for _stratum in populated(_branches):
            _s_zero = stratum_state(_stratum, "zero", _branches)
            _delta = (stratum_state(_stratum, _arm, _branches).values - _s_zero.values).mean(axis=0)
            _spread = sigma_ens(_s_zero)
            _curve = -np.divide(_delta, _spread, out=np.full_like(_delta, np.nan), where=_spread > 0)
            _curves[_stratum, _arm] = _curve
            _axes[0].plot(
                archive.state_times,
                _curve,
                lw=2,
                ls=_style,
                color=f"C{_stratum}",
                label=f"{STRATA[_stratum]} / {_arm}",
            )

    _best = max(
        (key for key in _curves if key[1] == "d1"),
        key=lambda key: np.nanmean(_curves[key][_undiluted]),
    )
    _rho_stratum, _rho_branches = _best[0], arm_branches("d1")
    _s_zero = stratum_state(_rho_stratum, "zero", _rho_branches)
    _s_stim = stratum_state(_rho_stratum, "d1", _rho_branches)

    for _name in shown:
        _zero = stratum_metric(_rho_stratum, "zero", _name, "all62", _rho_branches)
        _stim = stratum_metric(_rho_stratum, "d1", _name, "all62", _rho_branches)
        _on_state = align(_name)
        _axes[1].plot(
            archive.state_times,
            coupling(
                Ensemble(archive.state_times, _zero.values[:, _on_state], _zero.n_replicates),
                Ensemble(archive.state_times, _stim.values[:, _on_state], _stim.n_replicates),
                _s_zero,
                _s_stim,
            ),
            lw=1.3,
            label=_name,
        )

    for _ax, _label in zip(_axes, ["$d_{state}$", r"$\rho$"], strict=True):
        _ax.axvspan(0.0, SPREAD_WINDOW_S, color="0.9", zorder=0)
        _ax.axhline(0.0, color="k", lw=0.6)
        _ax.set_xlabel("lookahead $h$ (s)")
        _ax.set_ylabel(_label)
        _ax.legend(fontsize=7)

    # both deltas vanish at h = 0, so the view is set by the undiluted band rather than by a 0/0
    _limit = 1.2 * max(np.nanmax(np.abs(_c[_undiluted])) for _c in _curves.values())
    _axes[0].set_ylim(-_limit, _limit)
    _axes[1].set_ylim(-1.05, 1.05)
    _axes[0].set_title("does $u$ move the seizure? — per stratum and arm", fontsize=9)
    _axes[1].set_title(f"does moving $M$ move $s$? — {STRATA[_rho_stratum]} / d1", fontsize=9)
    _fig.suptitle(
        f"shaded: $h < W = {SPREAD_WINDOW_S:g}$ s, where the shared pre-branch window dilutes "
        r"$\delta_s$",
        fontsize=9,
    )
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(
    Ensemble,
    SPREAD_WINDOW_S,
    STRATA,
    align,
    archive,
    arm_branches,
    controllability,
    coupling,
    fs,
    h_slider,
    latency_s,
    n_bins,
    np,
    observations,
    pd,
    populated,
    readout_table,
    seizure_branches,
    separability,
    shown,
    state_predictability_r2,
    state_predictability_snr_db,
    state_readout_r2,
    stratum_metric,
    stratum_state,
):
    def score_table(names, h: float, bins: int) -> pd.DataFrame:
        """One row per observable and stratum; the arm contrasts are ``NaN`` where no probe was run."""
        read = readout_table(names, bins).set_index("metric")
        probed = arm_branches("d1")
        rows = []
        for name in names:
            grid = archive.times[name]
            idx = int(np.argmin(np.abs(grid - h)))
            s_idx = int(np.argmin(np.abs(archive.state_times - h)))
            m_all, s_all = observations(name, seizure_branches, "all62")
            v_state = state_readout_r2(m_all.mean(axis=1), s_all, n_bins=bins).explained_var
            sep = separability(
                archive.ensemble("healthy", "zero", name, "all62"),
                archive.ensemble("saturated", "zero", name, "all62"),
            )
            for stratum in populated(seizure_branches):
                zero = stratum_metric(stratum, "zero", name, "all62", seizure_branches)
                # the readout scores are zero-arm quantities, so they stand in every stratum
                row = {
                    "metric": name,
                    "stratum": STRATA[stratum],
                    "R2_read": read.loc[name, "R2_read all62"],
                    "R2_pred": state_predictability_r2(zero, v_state)[idx],
                    "SNR_pred_dB": state_predictability_snr_db(zero, v_state)[idx],
                    "d_ctrl": np.nan,
                    "rho": np.nan,
                    # the observable's own latency: what R2_pred and d_ctrl need
                    "valid": grid[idx] >= latency_s(name, fs),
                    # the shared pre-branch window, which dilutes delta_s: what rho needs
                    "delta_s_valid": archive.state_times[s_idx] >= SPREAD_WINDOW_S,
                }
                if stratum in populated(probed):
                    on_probed = stratum_metric(stratum, "zero", name, "all62", probed)
                    stim = stratum_metric(stratum, "d1", name, "all62", probed)
                    s_zero = stratum_state(stratum, "zero", probed)
                    s_stim = stratum_state(stratum, "d1", probed)
                    # align the observable onto the coarser state grid before pairing the two deltas
                    on_state = align(name)
                    rho = coupling(
                        Ensemble(archive.state_times, on_probed.values[:, on_state], on_probed.n_replicates),
                        Ensemble(archive.state_times, stim.values[:, on_state], stim.n_replicates),
                        s_zero,
                        s_stim,
                    )
                    row["d_ctrl"] = controllability(on_probed, stim, direction=sep.direction, gap=sep.gap).d_ctrl[idx]
                    row["rho"] = rho[s_idx]
                rows.append(row)
        return pd.DataFrame(rows)

    scores = score_table(shown, h_slider.value, n_bins.value)
    scores
    return (scores,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 The headline

    $R^2_{\text{read}}$ against $\rho$, one point per observable and probed stratum. This is the
    pair that has to be non-zero for an objective to be worth minimising at all: the observable has
    to *see* the seizure, and steering it has to *move* the seizure.

    - **top right** — the observable reads the state and driving it drives the seizure. What we want.
    - **top left** — reads the state, but pushing it pushes the seizure the *wrong* way.
    - **bottom band** ($\rho \approx 0$) — the §2.3 quadrant: whatever $d_{\text{ctrl}}$ says, the
      control effect on this observable is uninformative about the seizure.
    """)
    return


@app.cell
def _(STRATA, arm_branches, np, plt, populated, scores):
    _valid = scores[scores["valid"]]
    # rho is the x axis here, so a diluted delta_s shades the whole panel rather than a band of it
    _diluted = not scores["delta_s_valid"].all()
    _panels = [STRATA[s] for s in populated(arm_branches("d1"))]
    _fig, _axes = plt.subplots(
        1, len(_panels), figsize=(3.4 * len(_panels), 3.6), sharex=True, sharey=True, squeeze=False
    )

    for _ax, _name in zip(_axes[0], _panels, strict=True):
        if _diluted:
            _ax.set_facecolor("0.9")
        _sub = _valid[_valid["stratum"] == _name]
        _sizes = 30 + 300 * np.abs(_sub["d_ctrl"]) / (np.abs(_valid["d_ctrl"]).max() + 1e-12)
        _ax.scatter(_sub["rho"], _sub["R2_read"], s=_sizes, alpha=0.65)
        for _, _r in _sub.iterrows():
            _ax.annotate(
                _r["metric"], (_r["rho"], _r["R2_read"]), fontsize=6, xytext=(3, 3), textcoords="offset points"
            )
        _ax.axvline(0.0, color="k", lw=0.6)
        _ax.set_title(_name, fontsize=9)
        _ax.set_xlabel(r"$\rho$  (does moving $M$ move $s$?)")

    _axes[0, 0].set_ylabel("$R^2_{read}$")
    _shaded = r"  —  shaded: $h$ is inside the shared pre-branch window, so $\rho$ is diluted"
    _fig.suptitle(f"marker size = $|d_{{ctrl}}|${_shaded if _diluted else ''}", fontsize=9)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8 Does a reduction buy anything over the signal it reduces?

    A metric is a *reduction* of the scalp signal, and a reduction is only worth making if what
    survives it is easier to forecast than what went in. Two rungs sit below every windowed metric,
    and both are now scored on every axis above rather than only here:

    - **waveform** — the sample values themselves, which is what the MLP predictor is trained
      to forecast today. It needs the phase, which diverges fastest.
    - **envelope** — the causal band amplitude. It is *already* a reduction of the waveform: the
      band-pass and rectifier throw the phase away and keep the amplitude, at the cost of nothing
      but its two filters' group delay. So it, not the waveform, is the bar a windowed metric has
      to clear — a metric that only beats the waveform has beaten it by discarding phase, which the
      envelope does for free.

    ### Why this panel's numbers do not compare to §5's

    Both are "$1 - {}$ forecast error over something", and the something differs — in two ways, on
    purpose.

    **Different denominator.** Here it is $V_{\text{total}}$, the observable's whole variance within
    the group: the score is $1 - V_{\text{rep}}/V_{\text{total}}$, bounded in $[0, 1]$, and it
    answers *how deterministic is the plant, as seen through this observable*. §5 divides the same
    numerator by $V_{\text{state}}$ and answers *is the forecast error small against the seizure
    states I must tell apart*, which is unbounded below and is the question a controller asks. §5's
    denominator is deliberately not this one, because $V_{\text{total}}$ is set by whichever states
    the trajectories happened to occupy. Here that objection does not apply: the race is between
    quantities read off the *same* rollouts, so the accidental denominator is common to all of them
    and cancels out of the comparison.

    **Different pooling.** Here every observable is scored per channel and the scores are then
    averaged over the channel set; §4–§7 score the channel *mean*. A channel-mean waveform cancels,
    so pooling first would enter the incumbent objective into its own race as an artefact. Averaging
    the score instead is also what the predictor's loss actually is — a per-channel error summed
    over channels.

    The two rungs carry a 5 ms grid of their own: waveform predictability collapses inside a few
    hundred ms, which the metrics' 50 ms hop would miss.
    """)
    return


@app.cell
def _(STRATA, archive, mo, populated, seizure_branches):
    rung_stratum = mo.ui.dropdown(
        options=[STRATA[s] for s in populated(seizure_branches)], value=STRATA[0], label="Stratum"
    )
    rung_set = mo.ui.dropdown(options=archive.channel_sets, value="all62", label="Channel set")
    mo.hstack([rung_stratum, rung_set])
    return rung_set, rung_stratum


@app.cell
def _(
    RAW_SERIES,
    STRATA,
    archive,
    plt,
    rung_set,
    rung_stratum,
    seizure_branches,
    shown,
    stratum_metric,
    variance_ratio,
):
    _fig, _ax = plt.subplots(figsize=(8, 4.2))
    _stratum = STRATA.index(rung_stratum.value)

    for _i, _name in enumerate(shown):
        _ens = stratum_metric(_stratum, "zero", _name, rung_set.value, seizure_branches, pool=False)
        _per_channel = variance_ratio(_ens.values, _ens.n_replicates)
        _is_rung = _name in RAW_SERIES
        _ax.plot(
            archive.times[_name],
            _per_channel.mean(axis=0),
            color="k" if _is_rung else f"C{_i}",
            lw=1.4 if _is_rung else 1.1,
            ls="-" if _name == "waveform" else ("--" if _is_rung else "-"),
            alpha=1.0 if _is_rung else 0.7,
            label=_name,
        )

    _ax.axhline(0.0, color="0.5", lw=0.8)
    _ax.set_xlabel("lookahead $h$ (s)")
    _ax.set_ylabel(r"$1 - V_{rep}/V_{total}$  (per channel, then averaged)")
    _ax.set_title(f"predictability rungs — {rung_stratum.value} / {rung_set.value}", fontsize=10)
    _ax.legend(frameon=False, fontsize=8, ncol=2)
    _ax.spines[["top", "right"]].set_visible(False)
    _ax.grid(visible=True, which="both", linestyle="--", alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9 What does bandwidth cost the readout?

    Every low-pass buys noise rejection and pays group delay. The $x$ axis is the **cutoff**,
    because that is the knob being turned; the price it charges is the *effective latency* — the
    metric's own window plus the filter's group delay — which the table carries as a column and
    which sets the hollow-marker rule below, since a point evaluated at an $h$ shorter than the
    latency is reading a value the controller could not yet have had.

    (Latency, not cutoff, is the axis to think in when comparing *across* metrics at equal cost;
    cutoff is the axis to think in when reading what the filter does to *one* metric. The table
    gives the first, this plot the second.)

    Asked here against $s$: the $y$ axis is $R^2_{\text{read}}$, so the question is *how much of
    the metric's ability to see the seizure survives the filter*, alongside $d_{\text{ctrl}}$ for
    how much of its drivability does.

    Only the four amplitude-like metrics are swept. `band_power` and `spectral_centroid` are
    excluded because a low-pass **redefines** them rather than denoising them — it mutilates a
    3–12 Hz integral and moves a centroid down mechanically — so their sweep would not be one
    metric measured at several bandwidths. The two window-free series are excluded for the same
    reason on the envelope's side, and because a filtered waveform is the question this sweep asks
    of the metrics that survive it.

    ### Why the grid stops at 50 Hz

    The spectrum below sets the range, and it is the reason the sweep no longer carries the 100 Hz
    and 500 Hz points it started with. Those two were noise-side anchors: they asked *how much of a
    metric is the plant's white $x_5'$ tail*, which is a real question, but one the unfiltered point
    already prices in a single number.

    The panel says something stronger than "there is little power up there". **The five branches
    separate only below about 20 Hz.** Above ~25 Hz all of them collapse onto one identical
    $f^{-4}$-ish slope running to Nyquist — the noise shaped by the synaptic kernels, the same in a
    healthy network and a saturated one. So a cutoff above 25 Hz cannot be discarding *state*
    information, only common-mode noise, and where exactly it lands between 25 Hz and Nyquist barely
    matters. Everything that distinguishes a seizing network from a healthy one sits in the 3–12 Hz
    band and its immediate shoulder, which is why the grid is now spent between **10 and 50 Hz** and
    is finer at the low end — that is where a cutoff stops trimming noise and starts cutting into
    the signal.

    The one metric with a real stake in the tail is `line_length`: it is a differencer, so its
    integrand is the PSD weighted by $f^2$, and on this ensemble ~40 % of the healthy signal's line
    length comes from content above 50 Hz against ~6 % of `ez_ignited`'s. Its dotted unfiltered
    reference line is where that shows up, and that is the right place for it — the bandwidth a
    controller would actually choose is not 500 Hz.

    The dotted lines double as the sweep's own consistency check. If there is really nothing above
    50 Hz, the 50 Hz point has to land on the unfiltered reference for every metric on both panels,
    since the two are then the same signal. It does.
    """)
    return


@app.cell
def _(branch_names, eeg_path, ensemble_dir, fs, np, plt, welch):
    # a spectrum needs a handful of rollouts, and the full-rate store is 19 GB
    _n_rollouts = 4
    _fig, _ax = plt.subplots(figsize=(8, 3.8))

    for _b in branch_names:
        _x = np.asarray(np.load(eeg_path(ensemble_dir, _b, "zero"), mmap_mode="r")[:_n_rollouts], dtype=np.float64)
        _f, _psd = welch(_x, fs=fs, nperseg=8192, axis=-1)
        _ax.loglog(_f[1:], _psd.mean(axis=(0, 1))[1:], lw=1.2, label=_b)

    _ax.axvspan(3.0, 12.0, color="tab:red", alpha=0.10, zorder=0)
    _ax.axvspan(10.0, 50.0, color="tab:green", alpha=0.10, zorder=0)
    _ax.set_xlabel("frequency (Hz) — red: the 3–12 Hz seizure band, green: the swept cutoffs")
    _ax.set_ylabel(r"PSD (mV$^2$/Hz)")
    _ax.set_title(f"scalp spectrum, all 62 channels of {_n_rollouts} rollouts per branch", fontsize=10)
    _ax.legend(frameon=False, fontsize=8)
    _ax.spines[["top", "right"]].set_visible(False)
    _ax.grid(visible=True, which="both", linestyle="--", alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(
    RAW,
    SWEPT_METRICS,
    align,
    archive,
    arm_branches,
    controllability,
    design_lowpass_sos,
    fs,
    group_delay_s,
    latency_s,
    n_bins,
    np,
    pd,
    populated,
    rung_set,
    seizure_branches,
    separability,
    state_readout_r2,
    stratum_metric,
):
    def _latency(name: str, cutoff: str) -> float:
        delay = 0.0 if cutoff == RAW else group_delay_s(design_lowpass_sos(fs, float(cutoff)), fs)
        return latency_s(name, fs) + delay

    def _pooled_observations(metric: str, cutoff: str):
        """Stack (M, s) pairs over the seizure branches for a pooled, possibly filtered series."""
        idx = align(metric)
        m_rows, s_rows = [], []
        for branch in seizure_branches:
            m_rows.append(archive.ensemble(branch, "zero", metric, rung_set.value, cutoff).values[:, idx].reshape(-1))
            s_rows.append(archive.state(branch, "zero").values.reshape(-1))
        return np.concatenate(m_rows), np.concatenate(s_rows)

    def _sweep_rows(h: float, bins: int) -> pd.DataFrame:
        probed = arm_branches("d1")
        rows = []
        for cutoff in archive.cutoffs:
            for name in SWEPT_METRICS:
                grid = archive.times[name]
                idx = int(np.argmin(np.abs(grid - h)))
                m, s = _pooled_observations(name, cutoff)
                sep = separability(
                    archive.ensemble("healthy", "zero", name, rung_set.value, cutoff),
                    archive.ensemble("saturated", "zero", name, rung_set.value, cutoff),
                )
                latency = _latency(name, cutoff)
                per_stratum = []
                for stratum in populated(probed):
                    # the cutoff has to reach the arm contrast too, or the panel is the raw signal
                    zero = stratum_metric(stratum, "zero", name, rung_set.value, probed, cutoff)
                    stim = stratum_metric(stratum, "d1", name, rung_set.value, probed, cutoff)
                    per_stratum.append(controllability(zero, stim, direction=sep.direction, gap=sep.gap).d_ctrl[idx])
                rows.append(
                    {
                        "metric": name,
                        "cutoff_hz": cutoff,
                        "latency_s": latency,
                        "R2_read": float(state_readout_r2(m, s, n_bins=bins).r2),
                        "d_ctrl": float(np.mean(per_stratum)),
                        "valid": grid[idx] >= latency,
                    }
                )
        return pd.DataFrame(rows)

    sweep = _sweep_rows(1.5, n_bins.value)
    return (sweep,)


@app.cell
def _(mo, sweep):
    mo.ui.table(sweep.round({"latency_s": 4, "R2_read": 3, "d_ctrl": 3}), page_size=12, selection=None)
    return


@app.cell
def _(RAW, SWEPT_METRICS, np, plt, sweep):
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    # the unfiltered signal has no place on a cutoff axis, so it is drawn as the level to beat
    _swept = sweep[sweep["cutoff_hz"] != RAW].astype({"cutoff_hz": float})
    _unfiltered = sweep[sweep["cutoff_hz"] == RAW].set_index("metric")

    for _i, _name in enumerate(SWEPT_METRICS):
        _sub = _swept[_swept["metric"] == _name].sort_values("cutoff_hz")
        _ok = _sub["valid"].to_numpy()
        for _ax, _column in zip(_axes, ["R2_read", "d_ctrl"], strict=True):
            _ax.plot(_sub["cutoff_hz"], _sub[_column], color=f"C{_i}", lw=1.3, alpha=0.8, label=_name)
            _ax.scatter(
                _sub["cutoff_hz"],
                _sub[_column],
                s=42,
                color=np.where(_ok, f"C{_i}", "none"),
                edgecolors=f"C{_i}",
                zorder=3,
            )
            _ax.axhline(_unfiltered.loc[_name, _column], color=f"C{_i}", lw=0.9, ls=":", alpha=0.9)

    for _ax, _label in zip(_axes, [r"$R^2_{read}$", r"$d_{ctrl}$"], strict=True):
        _ax.axhline(0.0, color="0.5", lw=0.8)
        _ax.set_xlabel("low-pass cutoff (Hz)")
        _ax.set_ylabel(_label)
        _ax.spines[["top", "right"]].set_visible(False)
        _ax.grid(visible=True, which="both", linestyle="--", alpha=0.3)

    _axes[0].set_title("bandwidth vs seizure readout", fontsize=10)
    _axes[1].set_title(r"bandwidth vs drivability (mean over the probed strata)", fontsize=10)
    _axes[0].legend(frameon=False, fontsize=8)
    _fig.suptitle(
        "dotted: the same metric unfiltered  —  hollow markers: $h$ shorter than the effective latency",
        fontsize=9,
    )
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10 Caveats that belong with the numbers

    **The basin mixture is now the grouping variable, not a confound.** The plant is bimodal across
    seeds — some trajectories ignite and recruit 25–35 regions, others settle at 3–8 — and §2.1
    cuts on exactly that. What remains is that the two strata are unbalanced and the recruited one
    is the smaller: its arm contrasts rest on fewer trajectories than the focal one's, and the
    table in §2.1 gives the count for every row that depends on it. $V_{\text{state}}$ is still
    estimated across all $I = 16$ trajectories of every seizure branch, pooled, which is what makes
    it one number the strata can be compared against.

    **$\rho$ near zero is ambiguous where $d_{\text{state}}$ is null.** Where the probe barely
    moves the seizure, both $\delta$'s are mostly noise and their correlation is estimating a ratio
    of two small numbers. Read $\rho$ there as *undetermined*, not as *no coupling*.

    **The observable grid and the state grid do not start together.** Windowed metrics begin at
    their own window and the window-free series on a 5 ms grid, $s$ at $h = 0$, so the pairing in §6
    snaps the observable to the nearest state time and the first sample is reused for the state
    times below it. What the pre-branch history cannot fix is the shared-window dilution of
    $\delta_s$ below $h = W$, which is why the slider reaches into that band but the §7 panels are
    shaded there.

    **One probe direction, except on $d_{\text{state}}$.** `pre_onset` and `ez_ignited` carry
    $-d_1$, and the $d_{\text{state}}$ table reads it, so the trajectories drawn from those two
    branches test more than a null along one direction. $d_{\text{ctrl}}$ and $\rho$ are still
    conditional on $u = d_1$ everywhere. Under KCL the `roast_3d` leadfield's three electrodes give
    an admissible input space of exactly two dimensions, so a matched-norm orthogonal $d_2$ is what
    remains before first-order controllability is characterised completely.

    **No thresholds are pre-registered.** Curves over $h$ rather than a single summary point are
    the mitigation: any threshold gets applied in the open, against the whole curve.
    """)
    return


if __name__ == "__main__":
    app.run()
