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

    from neuro.ensembles import RAW, SWEPT_METRICS, load_manifest, score_ensemble_dir
    from neuro.filtering import design_lowpass_sos, group_delay_s
    from neuro.metrics import (
        METRICS,
        Ensemble,
        controllability,
        coupling,
        separability,
        sigma_ens,
        state_predictability_r2,
        state_readout_r2,
        variance_ratio,
    )
    from neuro.seizure import SPREAD_WINDOW_S

    return (
        Ensemble,
        METRICS,
        Path,
        RAW,
        SPREAD_WINDOW_S,
        SWEPT_METRICS,
        controllability,
        coupling,
        design_lowpass_sos,
        group_delay_s,
        load_manifest,
        mo,
        np,
        pd,
        plt,
        score_ensemble_dir,
        separability,
        sigma_ens,
        state_predictability_r2,
        state_readout_r2,
        variance_ratio,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # 🧠 Seizure-state scoring

    Which observable should the MPC minimise? This notebook answers it against a **measured
    ground truth** — the fraction of brain regions actually seizing — rather than against the
    branch labels or against the metrics themselves.

    It replaces `metric_scoring.py`, which scored the same ensemble on the four axes of
    `docs/predictability_controllability_experiment.md` and has been removed. Three things
    changed, each because a measurement said so; the sections below give the evidence in place.

    | | old | here |
    | --- | --- | --- |
    | ground truth | branch label (5 categories) | $s(t)$, a continuous seizing fraction |
    | $R^2$ denominator | spread of whatever states the seeds produced | the metric's span over $s$ |
    | control axis | does $u$ move the **metric**? | …and does moving the metric move the **seizure**? |
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
def _(Path, load_manifest, mo, score_ensemble_dir):
    ensemble_dir = Path("data/predictability_ensemble")
    manifest = load_manifest(ensemble_dir)
    archive = score_ensemble_dir(ensemble_dir)

    branch_names = [b["name"] for b in manifest["branches"]]
    seizure_branches = [b["name"] for b in manifest["branches"] if b["plant"] == "seizure"]
    # arms are allocated per branch, so not every seizure branch carries the probe
    stim_branches = [b["name"] for b in manifest["branches"] if "d1" in b["arms"]]
    channel_labels = manifest["channel_labels"]
    n_regions = len(manifest["region_labels"])

    mo.md(
        f"Loaded **{manifest['n_parents']} trajectories × {manifest['n_children']} replicates** over "
        f"{sum(len(b['arms']) for b in manifest['branches'])} (branch, arm) stores across "
        f"{len(branch_names)} branches, "
        f"{manifest['rollout_s']} s rollouts. "
        f"Seizure state on a {len(archive.state_times)}-point grid, "
        f"{archive.state_times[0]:.2f}–{archive.state_times[-1]:.2f} s."
    )
    return (
        archive,
        branch_names,
        channel_labels,
        manifest,
        n_regions,
        seizure_branches,
        stim_branches,
    )


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

    $$V_{\text{rep}}(h) \;=\; \frac{1}{I \cdot B}\sum_{i,b}\;\mathrm{Var}_{j}\!\big[M^{0}_{ij}(h)\big]$$

    This is what remains unknown *after* fixing the full plant state, so it is the irreducible
    forecast error — a ceiling no predictor can beat, and model-free, so it does not depend on
    the state of the predictor pipeline. It grows with $h$ as trajectories diverge.

    ---

    The two splits answer the two questions. Question 1 (*what correlates with seizure state?*)
    is $V_{\text{state}}$ against the total. Question 2 (*what can we predict?*) is
    $V_{\text{rep}}(h)$ against $V_{\text{state}}$ — **not** against the total, and that is the
    substantive change from the superseded four-axis scoring, which used the spread of whatever
    states the sixteen seeds happened to produce at that branch. That denominator is an accident of the draw: at
    `pre_onset` the trajectories are nearly identical and it is tiny, at `saturated` they are
    bimodal and it is large, so the old $R^2$ is not comparable across branches. $V_{\text{state}}$
    is the same number everywhere and it is the quantity a controller has to resolve.
    """)
    return


@app.cell
def _(mo):
    h_slider = mo.ui.slider(start=0.1, stop=3.0, step=0.05, value=1.5, label="$h_{eval}$ (s)", show_value=True)
    n_bins = mo.ui.slider(start=4, stop=20, step=1, value=10, label="state bins", show_value=True)
    mo.hstack([h_slider, n_bins])
    return h_slider, n_bins


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
    ## 4 Axis 1 — does the metric read the seizure state?

    $$R^2_{\text{read}} \;=\; \frac{V_{\text{state}}}{\mathrm{Var}[M]} \;=\; 1 - \frac{\mathbb{E}_s[\mathrm{Var}(M\mid s)]}{\mathrm{Var}[M]}$$

    Pooled over the four seizure branches and the whole time grid, on the **zero arm only** — the
    question is what the natural history of the metric looks like, and including the stimulated
    arm would mix control-induced variance into the state range.

    Pooling across branches is deliberate here and would be wrong for §5. For a *readout*, the
    between-branch swing is the signal; for *predictability*, it is a confound. The two questions
    want opposite pooling, which is why they are separate cells rather than one number.

    This axis replaces both **separability** (Cohen's $d$ between two extreme branches, $n = 16$)
    and **observability** (scalp-vs-region correlation of the same metric). Both were
    approximating this with less of the data, and both are now gone from `metrics.py`.
    """)
    return


@app.cell
def _(
    METRICS,
    n_bins,
    np,
    observations,
    pd,
    seizure_branches,
    state_readout_r2,
):
    def readout_table(bins: int) -> pd.DataFrame:
        rows = []
        for name in METRICS:
            m_all, s_all = observations(name, seizure_branches, "all62")
            m_foc, _ = observations(name, seizure_branches, "lTCI")
            pooled = state_readout_r2(m_all.mean(axis=1), s_all, n_bins=bins)
            focal = state_readout_r2(m_foc.mean(axis=1), s_all, n_bins=bins)
            per_channel = state_readout_r2(m_all, s_all, n_bins=bins)
            rows.append(
                {
                    "metric": name,
                    "window_s": METRICS[name].window_s,
                    "R2_read all62": pooled.r2,
                    "R2_read lTCI": focal.r2,
                    "best single channel": float(np.nanmax(per_channel.r2)),
                    "V_state (all62)": pooled.explained_var,
                }
            )
        return pd.DataFrame(rows).sort_values("R2_read all62", ascending=False)

    readout = readout_table(n_bins.value)
    readout
    return (readout_table,)


@app.cell
def _(
    METRICS,
    channel_labels,
    n_bins,
    np,
    observations,
    plt,
    seizure_branches,
    state_readout_r2,
):
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.8))

    for _name in METRICS:
        _m, _s = observations(_name, seizure_branches, "all62")
        _sc = state_readout_r2(_m.mean(axis=1), _s, n_bins=n_bins.value)
        _norm = (_sc.bin_metric - _sc.bin_metric.min()) / np.ptp(_sc.bin_metric)
        _axes[0].plot(_sc.bin_state, _norm, "o-", label=_name, ms=4)

    _axes[0].set_xlabel("seizure state $s$")
    _axes[0].set_ylabel("metric (min–max scaled)")
    _axes[0].set_title("Calibration: $\\mathbb{E}[M \\mid s]$")
    _axes[0].legend(fontsize=7)

    _m, _s = observations("eeg_ms", seizure_branches, "all62")
    _per = state_readout_r2(_m, _s, n_bins=n_bins.value)
    _order = np.argsort(_per.r2)[::-1]
    _axes[1].bar(range(len(_order)), _per.r2[_order], color="tab:blue")
    _axes[1].set_xticks(range(0, len(_order), 4))
    _axes[1].set_xticklabels([channel_labels[c] for c in _order[::4]], rotation=90, fontsize=6)
    _axes[1].set_ylabel("$R^2_{read}$")
    _axes[1].set_title("`eeg_ms` per channel — which channels see the seizure")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 Axis 2 — can it be predicted, to the precision control needs?

    $$R^2_{\text{pred}}(h) \;=\; 1 - \frac{V_{\text{rep}}(h)}{V_{\text{state}}}$$

    The numerator is the spread that survives fixing $x_0$ — the irreducible forecast error —
    against a **stated** denominator instead of an incidental one. It reads as: *is my forecast
    error small compared to the state difference I am trying to steer between?*

    It is **unbounded below**, and that is not a defect. $R^2_{\text{pred}} < 0$ says the metric's
    irreducible noise at lookahead $h$ is wider than the span it traverses across the seizure
    branches — so at that horizon the observable cannot distinguish the states the controller
    exists to move between, however smooth its curve looks.

    **`healthy` is deliberately out of $V_{\text{state}}$**, here and in §4. It is a different
    plant — $A = A_{\text{healthy}}$ rather than the EZ/PZ gain vector — so pooling it in would let
    a metric draw denominator from a parameter change no electrode can produce, and flatter every
    score by a span that is not reachable. `pre_onset` already anchors $s \approx 0$ under the
    plant the controller actually faces, and with per-branch arms `healthy` carries only the zero
    arm, so it can contribute nothing to axes 3–5 either. It stays where it earns its keep: fixing
    the sign and the native-units gap in `separability`.

    Curves run over the whole grid. Within-state variance genuinely collapses as $h \to 0$ —
    every replicate branches from one $x_0$ and has not had time to diverge — so
    $R^2_{\text{pred}} \to 1$ there is the plant being deterministic at short range, not an
    artefact. Only the metric's own window is masked.
    """)
    return


@app.cell
def _(
    METRICS,
    archive,
    n_bins,
    observations,
    plt,
    seizure_branches,
    state_predictability_r2,
    state_readout_r2,
):
    _fig, _axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)

    for _ax, _name in zip(_axes.ravel(), METRICS, strict=False):
        _m, _s = observations(_name, seizure_branches, "all62")
        _v_state = state_readout_r2(_m.mean(axis=1), _s, n_bins=n_bins.value).explained_var
        _t = archive.times[_name]
        _valid = _t >= METRICS[_name].window_s

        for _b in seizure_branches:
            _ens = archive.ensemble(_b, "zero", _name, "all62")
            _ax.plot(_t[_valid], state_predictability_r2(_ens, _v_state)[_valid], lw=2, label=_b)

        _ax.axhline(0.0, color="k", lw=0.6)
        _ax.set_ylim(-1.5, 1.05)
        _ax.set_title(f"{_name}  (window {METRICS[_name].window_s:g} s)", fontsize=9)
        _ax.set_xlabel("$h$ (s)")

    _axes[0, 0].set_ylabel("$R^2$")
    _axes[1, 0].set_ylabel("$R^2$")
    _axes[0, 0].legend(fontsize=6)
    _fig.suptitle("$R^2_{pred}$ against $V_{state}$, per branch", fontsize=10)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 Axes 3–5 — the three control questions

    The old notebook asked one control question. There are three, and the repo's history says
    they come apart:

    $$\text{(3) does }u\text{ move the metric?}\qquad
    d_{\text{ctrl}}(h) = \mathrm{dir}\cdot\frac{\bar\delta_M(h)}{\sigma_{\text{rep}}(h)},
    \quad \delta_{M,ij} = M^{d_1}_{ij} - M^{0}_{ij}$$

    $$\text{(4) does }u\text{ move the seizure?}\qquad
    d_{\text{state}}(h) = -\frac{\bar\delta_s(h)}{\sigma_{\text{rep}}\!\big(s^{0}\big)},
    \quad \delta_{s,ij} = s^{d_1}_{ij} - s^{0}_{ij}$$

    $$\text{(5) does moving the metric move the seizure?}\qquad
    \rho(h) = \mathrm{corr}_{ij}\!\big(\delta_{M,ij}(h),\; \delta_{s,ij}(h)\big)$$

    Both $d$'s divide by the **unpaired** replicate spread, not the paired $\mathrm{sd}(\delta)$.
    A controller cannot know which noise realisation it is in, so the effect it must steer against
    is buried in $\sigma_{\text{rep}}$; scoring on the paired spread would license calling a metric
    controllable when the effect is invisible to any causal controller.

    **$d_{\text{state}}$ is metric-independent — it is a property of the probe.** It should be read
    first. If the command does not move the seizure at a branch, every $d_{\text{ctrl}}$ at that
    branch is scoring response to a null actuator, and no ranking of metrics there means anything.

    **$\rho$ is the axis the original design has no slot for**, and it is the one
    `tes_field_geometry.md` §2.3 turned on: there the objective was driven 57 % in the intended
    direction while regional seizure count stayed at 31/34. $d_{\text{ctrl}}$ was large,
    $d_{\text{state}}$ was zero, and nothing in the four axes would have predicted it.
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
    all, so a metric window is post-branch from its first sample and `METRICS[name].window_s`
    already covers it. Its small values at short $h$ are latency, and real.
    """)
    return


@app.cell
def _(archive, manifest, n_regions, np, pd, sigma_ens):
    def state_response() -> pd.DataFrame:
        """One row per branch and probe arm at the last lookahead; branches with no probe drop out."""
        rows = []
        # the manifest's own arm allocation, so a branch contributes exactly the arms it carries
        for branch in manifest["branches"]:
            zero = archive.state(branch["name"], "zero")
            spread = sigma_ens(zero)
            for arm in (a for a in branch["arms"] if a != "zero"):
                delta = archive.state(branch["name"], arm).values - zero.values
                # per-trajectory means: replicates from one state are not independent draws
                per_traj = delta.reshape(zero.n_states, zero.n_replicates, -1).mean(axis=1)
                t_stat = per_traj.mean(axis=0) / (per_traj.std(axis=0, ddof=1) / np.sqrt(zero.n_states))
                rows.append(
                    {
                        "branch": branch["name"],
                        "arm": arm,
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
def _(d_state, manifest, mo):
    _worst = d_state.loc[d_state["t"].idxmin()]
    _df = manifest["n_parents"] - 1
    mo.md(
        "### Read this table before any metric ranking\n\n"
        f"The probe's strongest effect is at **`{_worst['branch']}` / `{_worst['arm']}`**: "
        f"{_worst['delta regions']:+.1f} regions, $t_{{({_df})}} = {_worst['t']:.2f}$, "
        f"{_worst['trajectories suppressed']} trajectories suppressed. Where $|t|$ is small the "
        "actuator is doing nothing to the seizure, and $d_{ctrl}$ and $\\rho$ at that branch are "
        "describing an ensemble in which there is no effect to attribute.\n\n"
        "Where a branch lists both $d_1$ and $-d_1$, read the pair: a command that steers the "
        "seizure should reverse $d_{state}$ when it reverses, and one that moves $s$ the same way "
        "under both is not steering it. That is the reading the reversed arm was bought for."
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### The same two scores over the horizon

    The table reads $d_{\text{state}}$ at $h = 3$ s and §7 reads $\rho$ at the slider; both are
    curves, and the band below $W$ is shaded rather than clipped so the dilution ramp stays
    visible as the artefact it is. $\rho$ is shown at the branch whose $d_{\text{state}}$ is
    largest outside that band, since everywhere else both $\delta$'s are noise.
    """)
    return


@app.cell
def _(
    Ensemble,
    METRICS,
    SPREAD_WINDOW_S,
    align,
    archive,
    coupling,
    np,
    plt,
    sigma_ens,
    stim_branches,
):
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.6), sharex=True)
    _undiluted = archive.state_times >= SPREAD_WINDOW_S
    _curves = {}

    for _b in stim_branches:
        _s_zero, _s_stim = archive.state(_b, "zero"), archive.state(_b, "d1")
        _delta = (_s_stim.values - _s_zero.values).mean(axis=0)
        _spread = sigma_ens(_s_zero)
        _curves[_b] = -np.divide(_delta, _spread, out=np.full_like(_delta, np.nan), where=_spread > 0)
        _axes[0].plot(archive.state_times, _curves[_b], lw=2, label=_b)

    _rho_branch = max(_curves, key=lambda b: np.nanmean(_curves[b][_undiluted]))
    _s_zero, _s_stim = archive.state(_rho_branch, "zero"), archive.state(_rho_branch, "d1")

    for _name in METRICS:
        _zero = archive.ensemble(_rho_branch, "zero", _name, "all62")
        _stim = archive.ensemble(_rho_branch, "d1", _name, "all62")
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
    _axes[0].set_title("does $u$ move the seizure? — per branch", fontsize=9)
    _axes[1].set_title(f"does moving $M$ move $s$? — {_rho_branch}", fontsize=9)
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
    METRICS,
    SPREAD_WINDOW_S,
    align,
    archive,
    controllability,
    coupling,
    h_slider,
    n_bins,
    np,
    observations,
    pd,
    readout_table,
    seizure_branches,
    separability,
    state_predictability_r2,
    state_readout_r2,
    stim_branches,
):
    def score_table(h: float, bins: int) -> pd.DataFrame:
        """One row per metric and seizure branch; the arm contrasts are ``NaN`` where no probe was run."""
        read = readout_table(bins).set_index("metric")
        rows = []
        for name in METRICS:
            grid = archive.times[name]
            idx = int(np.argmin(np.abs(grid - h)))
            s_idx = int(np.argmin(np.abs(archive.state_times - h)))
            m_all, s_all = observations(name, seizure_branches, "all62")
            v_state = state_readout_r2(m_all.mean(axis=1), s_all, n_bins=bins).explained_var
            sep = separability(
                archive.ensemble("healthy", "zero", name, "all62"),
                archive.ensemble("saturated", "zero", name, "all62"),
            )
            for branch in seizure_branches:
                zero = archive.ensemble(branch, "zero", name, "all62")
                # the readout scores are zero-arm quantities, so they stand at every seizure branch
                row = {
                    "metric": name,
                    "branch": branch,
                    "R2_read": read.loc[name, "R2_read all62"],
                    "R2_pred": state_predictability_r2(zero, v_state)[idx],
                    "d_ctrl": np.nan,
                    "rho": np.nan,
                    # the metric's own window: what R2_pred and d_ctrl need
                    "valid": grid[idx] >= METRICS[name].window_s,
                    # the shared pre-branch window, which dilutes delta_s: what rho needs
                    "delta_s_valid": archive.state_times[s_idx] >= SPREAD_WINDOW_S,
                }
                if branch in stim_branches:
                    stim = archive.ensemble(branch, "d1", name, "all62")
                    s_zero, s_stim = archive.state(branch, "zero"), archive.state(branch, "d1")
                    # align the metric onto the coarser state grid before pairing the two deltas
                    on_state = align(name)
                    rho = coupling(
                        Ensemble(archive.state_times, zero.values[:, on_state], zero.n_replicates),
                        Ensemble(archive.state_times, stim.values[:, on_state], stim.n_replicates),
                        s_zero,
                        s_stim,
                    )
                    row["d_ctrl"] = controllability(zero, stim, direction=sep.direction, gap=sep.gap).d_ctrl[idx]
                    row["rho"] = rho[s_idx]
                rows.append(row)
        return pd.DataFrame(rows)

    scores = score_table(h_slider.value, n_bins.value)
    scores
    return (scores,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 The headline

    $R^2_{\text{read}}$ against $\rho$, one point per metric and probed branch — $\rho$ is an arm
    contrast, so only the branches carrying a $d_1$ arm get a panel. The old headline plotted
    predictability against $d_{\text{ctrl}}$; this plots *sees the seizure* against *steering it
    moves the seizure*, which is the pair that has to be non-zero for the objective to be worth
    minimising at all.

    - **top right** — the metric reads the state and driving it drives the seizure. What we want.
    - **top left** — reads the state, but pushing it pushes the seizure the *wrong* way.
    - **bottom band** ($\rho \approx 0$) — the §2.3 quadrant: whatever $d_{\text{ctrl}}$ says, the
      control effect on this observable is uninformative about the seizure.
    """)
    return


@app.cell
def _(np, plt, scores, stim_branches):
    _valid = scores[scores["valid"]]
    # rho is the x axis here, so a diluted delta_s shades the whole panel rather than a band of it
    _diluted = not scores["delta_s_valid"].all()
    _fig, _axes = plt.subplots(1, len(stim_branches), figsize=(3.2 * len(stim_branches), 3.6), sharex=True, sharey=True)

    for _ax, _b in zip(_axes, stim_branches, strict=False):
        if _diluted:
            _ax.set_facecolor("0.9")
        _sub = _valid[_valid["branch"] == _b]
        _sizes = 30 + 300 * np.abs(_sub["d_ctrl"]) / (np.abs(_valid["d_ctrl"]).max() + 1e-12)
        _ax.scatter(_sub["rho"], _sub["R2_read"], s=_sizes, alpha=0.65)
        for _, _r in _sub.iterrows():
            _ax.annotate(
                _r["metric"], (_r["rho"], _r["R2_read"]), fontsize=6, xytext=(3, 3), textcoords="offset points"
            )
        _ax.axvline(0.0, color="k", lw=0.6)
        _ax.set_title(_b, fontsize=9)
        _ax.set_xlabel(r"$\rho$  (does moving $M$ move $s$?)")

    _axes[0].set_ylabel("$R^2_{read}$")
    _shaded = r"  —  shaded: $h$ is inside the shared pre-branch window, so $\rho$ is diluted"
    _fig.suptitle(f"marker size = $|d_{{ctrl}}|${_shaded if _diluted else ''}", fontsize=9)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8 Does a metric beat the raw signal?

    A metric is a *reduction* of the scalp signal, and a reduction is only worth making if what
    survives it is easier to forecast than what went in. The two rungs are the raw signal itself:

    - **waveform** — predict the sample values. Needs the phase, which diverges fastest.
    - **envelope** — predict the analytic amplitude, phase discarded. The rung a metric has to clear.

    **These curves are on a different scale from §5 and must not be read across to it.** Both the
    rungs and the metric curves here are $1 - V_{\text{rep}}/V_{\text{total}}$ — variance explained
    by $x_0$ within one branch, bounded in $[0,1]$, which is a statement about *how deterministic
    the plant is*, not about resolving seizure states. That is exactly the denominator §5 rejects
    as an accident of the trajectory draw. It is the right denominator here only because the
    question is a like-for-like race between quantities read off the same rollouts.

    The rungs carry a 5 ms grid of their own: waveform predictability collapses inside a few
    hundred ms, which the metrics' 50 ms hop would miss.
    """)
    return


@app.cell
def _(archive, branch_names, mo):
    rung_branch = mo.ui.dropdown(options=branch_names, value="ez_ignited", label="Branch")
    rung_set = mo.ui.dropdown(options=archive.channel_sets, value="all62", label="Channel set")
    mo.hstack([rung_branch, rung_set])
    return rung_branch, rung_set


@app.cell
def _(METRICS, RAW, archive, plt, rung_branch, rung_set, variance_ratio):
    _fig, _ax = plt.subplots(figsize=(8, 4.2))

    for _i, _name in enumerate(METRICS):
        _ens = archive.ensemble(rung_branch.value, "zero", _name, rung_set.value)
        _ax.plot(
            archive.times[_name],
            variance_ratio(_ens.values, _ens.n_replicates),
            color=f"C{_i}",
            lw=1.1,
            alpha=0.7,
            label=_name,
        )

    for _name, _style in (("waveform", "-"), ("envelope", "--")):
        _ax.plot(
            archive.baseline_times,
            archive.baseline(rung_branch.value, _name, rung_set.value, RAW),
            color="k",
            lw=1.4,
            ls=_style,
            label=_name,
        )

    _ax.axhline(0.0, color="0.5", lw=0.8)
    _ax.set_xlabel("lookahead $h$ (s)")
    _ax.set_ylabel(r"$1 - V_{rep}/V_{total}$")
    _ax.set_title(f"predictability rungs — {rung_branch.value} / {rung_set.value}", fontsize=10)
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

    Every low-pass buys noise rejection and pays group delay. The cost is not the cutoff but the
    **effective latency** — the metric's own window plus the filter's group delay — because that is
    what the controller actually waits for before it can act on a sample.

    Asked here against $s$, not against the old per-branch $R^2$: the $y$ axis is
    $R^2_{\text{read}}$, so the question is *how much of the metric's ability to see the seizure
    survives the filter*, alongside $d_{\text{ctrl}}$ for how much of its drivability does.

    Only the four amplitude-like metrics are swept. `band_power` and `spectral_centroid` are
    excluded because a low-pass **redefines** them rather than denoising them — it mutilates a
    3–12 Hz integral and moves a centroid down mechanically — so their sweep would not be one
    metric measured at several bandwidths.
    """)
    return


@app.cell
def _(
    METRICS,
    RAW,
    SWEPT_METRICS,
    align,
    archive,
    controllability,
    design_lowpass_sos,
    group_delay_s,
    manifest,
    n_bins,
    np,
    pd,
    rung_set,
    seizure_branches,
    separability,
    state_readout_r2,
    stim_branches,
):
    def _latency(name: str, cutoff: str) -> float:
        fs = manifest["fs"]
        delay = 0.0 if cutoff == RAW else group_delay_s(design_lowpass_sos(fs, float(cutoff)), fs)
        return METRICS[name].window_s + delay

    def _pooled_observations(metric: str, cutoff: str):
        """Stack (M, s) pairs over the seizure branches for a pooled, possibly filtered series."""
        idx = align(metric)
        m_rows, s_rows = [], []
        for branch in seizure_branches:
            m_rows.append(archive.ensemble(branch, "zero", metric, rung_set.value, cutoff).values[:, idx].reshape(-1))
            s_rows.append(archive.state(branch, "zero").values.reshape(-1))
        return np.concatenate(m_rows), np.concatenate(s_rows)

    def _sweep_rows(h: float, bins: int) -> pd.DataFrame:
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
                per_branch = []
                for branch in stim_branches:
                    zero = archive.ensemble(branch, "zero", name, rung_set.value, cutoff)
                    stim = archive.ensemble(branch, "d1", name, rung_set.value, cutoff)
                    per_branch.append(controllability(zero, stim, direction=sep.direction, gap=sep.gap).d_ctrl[idx])
                rows.append(
                    {
                        "metric": name,
                        "cutoff_hz": cutoff,
                        "latency_s": latency,
                        "R2_read": float(state_readout_r2(m, s, n_bins=bins).r2),
                        "d_ctrl": float(np.mean(per_branch)),
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
def _(SWEPT_METRICS, np, plt, sweep):
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)

    for _i, _name in enumerate(SWEPT_METRICS):
        _sub = sweep[sweep["metric"] == _name].sort_values("latency_s")
        _ok = _sub["valid"].to_numpy()
        for _ax, _column in zip(_axes, ["R2_read", "d_ctrl"], strict=True):
            _ax.plot(_sub["latency_s"], _sub[_column], color=f"C{_i}", lw=1.3, alpha=0.8, label=_name)
            _ax.scatter(
                _sub["latency_s"],
                _sub[_column],
                s=42,
                color=np.where(_ok, f"C{_i}", "none"),
                edgecolors=f"C{_i}",
                zorder=3,
            )

    for _ax, _label in zip(_axes, [r"$R^2_{read}$", r"$d_{ctrl}$"], strict=True):
        _ax.axhline(0.0, color="0.5", lw=0.8)
        _ax.set_xscale("log")
        _ax.set_xlabel("effective latency = window + group delay (s)")
        _ax.set_ylabel(_label)
        _ax.spines[["top", "right"]].set_visible(False)
        _ax.grid(visible=True, which="both", linestyle="--", alpha=0.3)

    _axes[0].set_title("bandwidth vs seizure readout", fontsize=10)
    _axes[1].set_title(r"bandwidth vs drivability (mean over the $d_1$ branches)", fontsize=10)
    _axes[0].legend(frameon=False, fontsize=8)
    _fig.suptitle("hollow markers: $h$ shorter than the effective latency", fontsize=9)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10 Caveats that belong with the numbers

    **The basin mixture.** The plant is bimodal across seeds — in the eight-trajectory draft
    ensemble, seed 1076 settled at 4 seizing regions where the others reached 25–35. It is a
    mixture, not an outlier to drop, and $V_{\text{state}}$ and every branch mean are estimated
    across $I = 16$ trajectories that need not share a basin.

    **$\rho$ near zero is ambiguous where $d_{\text{state}}$ is null.** At `mid_spread` the probe
    barely moves the seizure, so both $\delta$'s are mostly noise and their correlation is
    estimating a ratio of two small numbers. Read $\rho$ there as *undetermined*, not as *no
    coupling*. `healthy` and `saturated` carry no probe at all, so axes 3–5 are absent there rather
    than null, and the table leaves them empty.

    **The metric grid and the state grid do not start together.** Metrics begin at their own
    window, $s$ at $h = 0$, so the pairing in §6 snaps the metric to the nearest state time and the
    first metric sample is reused for the state times below it. What the pre-branch history cannot
    fix is the shared-window dilution of $\delta_s$ below $h = W$, which is why the slider reaches
    into that band but the §7 panels are shaded there.

    **One probe direction, except on $d_{\text{state}}$.** `pre_onset` and `ez_ignited` carry
    $-d_1$, and the $d_{\text{state}}$ table reads it, so at those two branches a null probe effect
    is not merely a null along one direction. $d_{\text{ctrl}}$ and $\rho$ are still conditional on
    $u = d_1$ everywhere, and `mid_spread` has $d_1$ alone. Under KCL the `roast_3d` leadfield's
    three electrodes give an admissible input space of exactly two dimensions, so a matched-norm
    orthogonal $d_2$ is what remains before first-order controllability is characterised completely.

    **No thresholds are pre-registered.** Curves over $h$ rather than a single summary point are
    the mitigation: any threshold gets applied in the open, against the whole curve.
    """)
    return


if __name__ == "__main__":
    app.run()
