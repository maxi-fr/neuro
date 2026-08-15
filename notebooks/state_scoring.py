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

    from neuro.ensembles import load_manifest, score_ensemble_dir
    from neuro.metrics import (
        METRICS,
        Ensemble,
        controllability,
        coupling,
        predictability_r2,
        separability,
        sigma_ens,
        state_readout_r2,
    )

    return (
        Ensemble,
        METRICS,
        Path,
        controllability,
        coupling,
        load_manifest,
        mo,
        np,
        pd,
        plt,
        predictability_r2,
        score_ensemble_dir,
        separability,
        sigma_ens,
        state_readout_r2,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # 🧠 Seizure-state scoring

    Which observable should the MPC minimise? This notebook answers it against a **measured
    ground truth** — the fraction of brain regions actually seizing — rather than against the
    branch labels or against the metrics themselves.

    It is the successor to `metric_scoring.py`, which scores the same ensemble on the four axes of
    `docs/predictability_controllability_experiment.md`. Three things changed, each because a
    measurement said so; the sections below give the evidence in place.

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
    | $i$ | $1 \dots I$, $I = 8$ | **trajectory** — one independent realisation of the disease course, i.e. one plant seed |
    | $b$ | 5 values | **branch** — a time along that trajectory at which it was frozen |
    | $x_0^{\,i,b}$ | — | the **state**: trajectory $i$ at branch $b$, comprising $x$, the delay history, and the step counter |
    | $j$ | $1 \dots J$, $J = 16$ | **replicate** — one noise realisation rolled out from that state |
    | $a$ | $\{0,\; d_1\}$ | **arm** — unstimulated, or the sustained hold $u = [{+}2, 0, {-}2]$ mA |
    | $h$ | $0 \dots 3$ s | **lookahead** since the branch |
    | $c$ | $1 \dots C$, $C = 62$ | **channel** |
    | $n$ | $1 \dots N$, $N = 76$ | **region** |

    Replicates share their seed across arms, so $M^{0}_{ij}$ and $M^{d_1}_{ij}$ are the *same*
    noise realisation under two commands and their difference is paired.

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
    peak-to-peak within a trailing 1 s window — the criterion `neuro.seizure.SpreadProfile`
    already uses to label spread, unchanged. The network state is the fraction of regions meeting
    it:

    $$s(t) \;=\; \frac{1}{N}\sum_{n=1}^{N} \mathbb{1}\!\left[\;\mathrm{PTP}_{n}\big(t-1\mathrm{s},\,t\big) > \theta\;\right]$$

    So $s = 0$ is a healthy network and $s \approx 0.4$ is ~30 of 76 regions recruited.

    **Why one number for the network, and not one per channel.** The obvious refinement is to give
    each channel its own seizure state by weighting regions with that channel's row of the EEG
    forward operator, $s_c = \sum_n w_{cn} z_n / \sum_n w_{cn}$. Measured on this ensemble, $s_c$
    correlates with the single network scalar at a **median 0.996 (`pre_onset`), 0.995
    (`ez_ignited`), 0.982 (`mid_spread`)** across all 62 channels. Volume conduction is broad
    enough that every channel sees nearly the same weighted fraction. The channel axis carries no
    target information at the branches where stimulation does anything, so it is spent on the
    *metric* side instead, where §7 shows it does carry something.

    **$s$ is causal but slow.** Its 1 s window is inherited from the threshold's calibration —
    a shorter window measures less peak-to-peak and would silently re-tune $\theta$. So $s$ is
    only defined for $h \geq 1$ s, and every state-referenced score below starts there. That is a
    property of the ground truth, not of the metrics.
    """)
    return


@app.cell
def _(Path, load_manifest, mo, score_ensemble_dir):
    ensemble_dir = Path("data/predictability_ensemble")
    manifest = load_manifest(ensemble_dir)
    archive = score_ensemble_dir(ensemble_dir)

    branch_names = [b["name"] for b in manifest["branches"]]
    seizure_branches = [b["name"] for b in manifest["branches"] if b["plant"] == "seizure"]
    channel_labels = manifest["channel_labels"]
    n_regions = len(manifest["region_labels"])

    mo.md(
        f"Loaded **{manifest['n_parents']} trajectories × {manifest['n_children']} replicates × "
        f"{len(manifest['arms'])} arms × {len(branch_names)} branches**, "
        f"{manifest['rollout_s']} s rollouts. "
        f"Seizure state on a {len(archive.state_times)}-point grid, "
        f"{archive.state_times[0]:.2f}–{archive.state_times[-1]:.2f} s."
    )
    return archive, branch_names, channel_labels, n_regions, seizure_branches


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
    substantive change from the old notebook, which used the spread of whatever states the eight
    seeds happened to produce at that branch. That denominator is an accident of the draw: at
    `pre_onset` the trajectories are nearly identical and it is tiny, at `saturated` they are
    bimodal and it is large, so the old $R^2$ is not comparable across branches. $V_{\text{state}}$
    is the same number everywhere and it is the quantity a controller has to resolve.
    """)
    return


@app.cell
def _(mo):
    h_slider = mo.ui.slider(start=1.0, stop=3.0, step=0.05, value=1.5, label="$h_{eval}$ (s)", show_value=True)
    n_bins = mo.ui.slider(start=4, stop=20, step=1, value=10, label="state bins", show_value=True)
    mo.hstack([h_slider, n_bins])
    return h_slider, n_bins


@app.cell
def _(archive, np):
    def align(metric: str):
        """Index the metric grid at each seizure-state time, so the two line up sample for sample."""
        idx = np.array([int(np.argmin(np.abs(archive.times[metric] - t))) for t in archive.state_times])
        return archive.state_times, idx

    def observations(metric: str, branches, channel_set: str = "all62", arm: str = "zero"):
        """Stack (metric, state) pairs over branches, rollouts and times into a flat design.

        Returns ``(M, s)`` with ``M`` of shape ``(n_obs, n_channels)`` and ``s`` of ``(n_obs,)``.
        Rollout-major so the two orders match exactly.
        """
        _, idx = align(metric)
        metric_rows, state_rows = [], []
        for branch in branches:
            values = archive.channel_ensemble(branch, arm, metric, channel_set).values[:, :, idx]
            state = archive.state(branch, arm).values
            metric_rows.append(values.transpose(0, 2, 1).reshape(-1, values.shape[1]))
            state_rows.append(state.reshape(-1))
        return np.concatenate(metric_rows), np.concatenate(state_rows)

    return (observations,)


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

    This axis replaces both **separability** (Cohen's $d$ between two extreme branches, $n = 8$)
    and **observability** (scalp-vs-region correlation of the same metric) from the old notebook.
    Both were approximating this with less of the data.
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

    Same numerator as the old score — the spread that survives fixing $x_0$ — against a **stated**
    denominator instead of an incidental one. It reads as: *is my forecast error small compared to
    the state difference I am trying to steer between?*

    It is **unbounded below**, and that is not a defect. $R^2_{\text{pred}} < 0$ says the metric's
    irreducible noise at lookahead $h$ is wider than the whole span it traverses from healthy to
    saturated — so at that horizon the observable cannot distinguish the states the controller
    exists to move between, however smooth its curve looks.

    The dashed lines are the old per-branch $R^2$ for comparison. Where they disagree, the
    denominator is the reason, not the metric.
    """)
    return


@app.cell
def _(
    METRICS,
    archive,
    n_bins,
    observations,
    plt,
    predictability_r2,
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
            _ens = archive.ensemble("scalp", _b, "zero", _name, "all62")
            _ax.plot(_t[_valid], state_predictability_r2(_ens, _v_state)[_valid], lw=2, label=_b)
            _ax.plot(_t[_valid], predictability_r2(_ens)[_valid], lw=0.8, ls="--", alpha=0.5)

        _ax.axhline(0.0, color="k", lw=0.6)
        _ax.set_ylim(-1.5, 1.05)
        _ax.set_title(f"{_name}  (window {METRICS[_name].window_s:g} s)", fontsize=9)
        _ax.set_xlabel("$h$ (s)")

    _axes[0, 0].set_ylabel("$R^2$")
    _axes[1, 0].set_ylabel("$R^2$")
    _axes[0, 0].legend(fontsize=6)
    _fig.suptitle("solid: $R^2_{pred}$ against $V_{state}$   ·   dashed: old per-branch $R^2$", fontsize=10)
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
def _(archive, n_regions, np, pd, seizure_branches, sigma_ens):
    def state_response() -> pd.DataFrame:
        rows = []
        for branch in seizure_branches:
            zero, stim = archive.state(branch, "zero"), archive.state(branch, "d1")
            delta = stim.values - zero.values
            spread = sigma_ens(zero)
            # per-trajectory means: replicates from one state are not independent draws
            per_traj = delta.reshape(zero.n_states, zero.n_replicates, -1).mean(axis=1)
            t_stat = per_traj.mean(axis=0) / (per_traj.std(axis=0, ddof=1) / np.sqrt(zero.n_states))
            rows.append(
                {
                    "branch": branch,
                    "s (zero)": zero.values[:, -1].mean(),
                    "delta regions": delta[:, -1].mean() * n_regions,
                    "d_state": -delta[:, -1].mean() / spread[-1],
                    "t (7 df)": t_stat[-1],
                    "trajectories suppressed": f"{int((per_traj[:, -1] < 0).sum())}/{zero.n_states}",
                }
            )
        return pd.DataFrame(rows)

    d_state = state_response()
    d_state
    return (d_state,)


@app.cell
def _(d_state, mo):
    _worst = d_state.loc[d_state["t (7 df)"].idxmin()]
    mo.md(
        "### Read this table before any metric ranking\n\n"
        f"The probe's strongest effect is at **`{_worst['branch']}`**: "
        f"{_worst['delta regions']:+.1f} regions, $t_{{(7)}} = {_worst['t (7 df)']:.2f}$, "
        f"{_worst['trajectories suppressed']} trajectories suppressed. Where $|t|$ is small the "
        "actuator is doing nothing to the seizure, and $d_{ctrl}$ and $\\rho$ at that branch are "
        "describing an ensemble in which there is no effect to attribute."
    )
    return


@app.cell
def _(
    Ensemble,
    METRICS,
    archive,
    controllability,
    coupling,
    h_slider,
    n_bins,
    np,
    observations,
    pd,
    predictability_r2,
    readout_table,
    seizure_branches,
    separability,
    state_predictability_r2,
    state_readout_r2,
):
    def score_table(h: float, bins: int) -> pd.DataFrame:
        read = readout_table(bins).set_index("metric")
        rows = []
        for name in METRICS:
            grid = archive.times[name]
            idx = int(np.argmin(np.abs(grid - h)))
            s_idx = int(np.argmin(np.abs(archive.state_times - h)))
            m_all, s_all = observations(name, seizure_branches, "all62")
            v_state = state_readout_r2(m_all.mean(axis=1), s_all, n_bins=bins).explained_var
            sep = separability(
                archive.ensemble("scalp", "healthy", "zero", name, "all62"),
                archive.ensemble("scalp", "saturated", "zero", name, "all62"),
            )
            for branch in seizure_branches:
                zero = archive.ensemble("scalp", branch, "zero", name, "all62")
                stim = archive.ensemble("scalp", branch, "d1", name, "all62")
                s_zero, s_stim = archive.state(branch, "zero"), archive.state(branch, "d1")
                ctrl = controllability(zero, stim, direction=sep.direction, gap=sep.gap)
                # align the metric onto the coarser state grid before pairing the two deltas
                on_state = np.array([int(np.argmin(np.abs(grid - t))) for t in archive.state_times])
                rho = coupling(
                    Ensemble(archive.state_times, zero.values[:, on_state], zero.n_replicates),
                    Ensemble(archive.state_times, stim.values[:, on_state], stim.n_replicates),
                    s_zero,
                    s_stim,
                )
                rows.append(
                    {
                        "metric": name,
                        "branch": branch,
                        "R2_read": read.loc[name, "R2_read all62"],
                        "R2_pred": state_predictability_r2(zero, v_state)[idx],
                        "R2_old": predictability_r2(zero)[idx],
                        "d_ctrl": ctrl.d_ctrl[idx],
                        "rho": rho[s_idx],
                        "valid": grid[idx] >= METRICS[name].window_s,
                    }
                )
        return pd.DataFrame(rows)

    scores = score_table(h_slider.value, n_bins.value)
    scores
    return (scores,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 The headline

    $R^2_{\text{read}}$ against $\rho$, one point per metric and branch. The old headline plotted
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
def _(np, plt, scores, seizure_branches):
    _valid = scores[scores["valid"]]
    _fig, _axes = plt.subplots(1, len(seizure_branches), figsize=(14, 3.6), sharex=True, sharey=True)

    for _ax, _b in zip(_axes, seizure_branches, strict=False):
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
    _fig.suptitle("marker size = $|d_{ctrl}|$", fontsize=9)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8 Caveats that belong with the numbers

    **The basin mixture.** One of eight trajectories (seed 1076) settles at 4 seizing regions where
    the others reach 25–35. It is a mixture, not an outlier to drop — but $V_{\text{state}}$ and
    every branch mean are estimated from $I = 8$ with one of them in a different basin.

    **$\rho$ near zero is ambiguous where $d_{\text{state}}$ is null.** At `mid_spread` and
    `saturated` the probe barely moves the seizure, so both $\delta$'s are mostly noise and their
    correlation is estimating a ratio of two small numbers. Read $\rho$ at those branches as
    *undetermined*, not as *no coupling*.

    **The metric grid is coarser than the state grid at short $h$.** Metrics are read on 0.1 s or
    0.5 s windows and $s$ on a 1 s one, so the pairing in §6 snaps the metric to the nearest state
    time. At $h < 1$ s there is no state to pair with at all, which is why the slider starts at 1 s.

    **One probe direction.** Everything on axes 3–5 is conditional on $u = d_1$. Under KCL the
    `roast_3d` leadfield's three electrodes give an admissible input space of exactly two
    dimensions, so adding $-d_1$ and a matched-norm orthogonal $d_2$ would characterise
    first-order controllability completely. A null $\rho$ here means "not along this direction",
    not "not steerable".

    **No thresholds are pre-registered.** Curves over $h$ rather than a single summary point are
    the mitigation: any threshold gets applied in the open, against the whole curve.
    """)
    return


if __name__ == "__main__":
    app.run()
