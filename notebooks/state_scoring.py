import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full", app_title="EEG Feature Evaluation")


@app.cell
def _():
    import os
    from pathlib import Path

    import marimo as mo
    import numpy as np
    from matplotlib import pyplot as plt

    from neuro.plant.seizure_calibration import SEIZURE_PTP_MV, SPREAD_WINDOW_S
    from observable_predictability.archive import EVALUATION_SCORES_FILE, load_evaluation_bundle
    from observable_predictability.features import REPRESENTATIONS
    from observable_predictability.plots import (
        REPRESENTATION_LABELS,
        REPRESENTATION_PENALTY_LABELS,
        representation_shared_history_s,
    )

    return (
        EVALUATION_SCORES_FILE,
        Path,
        REPRESENTATIONS,
        REPRESENTATION_LABELS,
        REPRESENTATION_PENALTY_LABELS,
        SEIZURE_PTP_MV,
        SPREAD_WINDOW_S,
        load_evaluation_bundle,
        mo,
        np,
        os,
        plt,
        representation_shared_history_s,
    )


@app.cell
def _(SEIZURE_PTP_MV, SPREAD_WINDOW_S, mo):
    mo.md(rf"""
    # Evaluating EEG features

    1. How closely does a feature's suppression penalty track the fraction of regions seizing?
    2. How much of the feature's future variation is predictable from the full starting state?

    Independent healthy EEG supplies the penalty references. The analysis uses only
    zero-current continuations of the seizure Plant. At each snapshot, the continuations
    share the full Plant state and pre-branch history, with independent future noise.
    Synchronized scalp EEG supplies the features, and regional LFP supplies Seizure State.

    **Seizure State** is the fraction of regions whose trailing {SPREAD_WINDOW_S:g} s
    LFP exceeds {SEIZURE_PTP_MV:g} mV peak to peak. We use its continuous range throughout.
    """)
    return


@app.cell
def _(EVALUATION_SCORES_FILE, Path, load_evaluation_bundle, mo, os):
    _root = Path(__file__).resolve().parents[1] / "artifacts"
    _expanded = _root / "observable_predictability_expanded"
    _latest = _root / "observable_predictability_two_criteria"
    _override = os.environ.get("NEURO_PREDICTABILITY_ARCHIVE")
    _default = _latest
    if (_expanded / EVALUATION_SCORES_FILE).is_file() and (_expanded / "waveform_split.npz").is_file():
        _default = _expanded
    _archive = Path(_override) if _override else _default
    bundle = load_evaluation_bundle(_archive)
    mo.md(
        f"Archive **{_archive.name}** · **{bundle.manifest['n_parents']} evaluation Plant seeds** "
        f"· **{bundle.manifest['n_children']} noise replicates per snapshot**."
    )
    waveform_split_path = _archive / "waveform_split.npz"
    if _archive.resolve() == _latest.resolve():
        waveform_split_path = _root / "observable_phase_diagnostic" / "waveform_split.npz"
    return bundle, waveform_split_path


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Penalty versus Seizure State

    Pair the measured suppression penalty $p(z(t))$ with Seizure State $s(t)$ at the
    same time. Waveform sample penalties are averaged onto the common Frame grid.

    $$\rho_{\mathrm{state}} = \operatorname{PearsonCorr}\!\big(p(z(t)),s(t)\big).$$

    Waveform uses squared deviation from healthy channel means, spectral uses squared excess
    above the healthy log-power envelope, and connectivity uses squared distance outside the
    healthy intervals. Each penalty averages over its components.

    Each point is one measured pair from a zero-current seizure-Plant replay.
    Each panel reports one Pearson correlation across all the points shown.
    """)
    return


@app.cell
def _(REPRESENTATIONS, REPRESENTATION_PENALTY_LABELS, bundle, np, plt):
    _fig, _axes = plt.subplots(1, len(REPRESENTATIONS), sharex=True, figsize=(12, 3.6), layout="constrained")
    for _rep, _ax in zip(REPRESENTATIONS, _axes, strict=True):
        _result = bundle.seizure_relevance[_rep]
        _points = _result.finite_pairs()
        _ax.scatter(_points[:, 0], _points[:, 1], s=5, alpha=0.15, linewidths=0)
        _summary = "Pearson r undefined"
        _r = _result.pearson(_points)
        if np.isfinite(_r):
            _summary = f"Pearson r = {_r:.3f}"
        _ax.set_title(f"{REPRESENTATION_PENALTY_LABELS[_rep]}\n{_summary}")
        _ax.set_xlabel("Seizure State (fraction of regions)")
    _axes[0].set_ylabel("Suppression penalty (native units)")
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Potential predictability versus lookahead

    At each lookahead and for each component, let $y_{ir}$ be the value from snapshot $i$
    and replay $r$. With $n$ snapshots and $m$ replays per snapshot, estimate the
    within-snapshot noise variance $V_N$ and subtract the noise remaining in each replay
    mean from the between-snapshot variance.

    $$\widehat V_N = \frac1n\sum_i\operatorname{Var}_r(y_{ir}), \qquad
      \widehat V_S = \operatorname{Var}_i(\bar y_i) - \widehat V_N/m.$$

    The plot shows the signal-to-total variance ratio
    $\sum_k\widehat V_{S,k}/\sum_k(\widehat V_{S,k}+\widehat V_{N,k})$.
    The ratio can be negative in a finite sample. A zero estimated total leaves it undefined.
    Each curve uses components that have complete replays at each lookahead.

    The ratio is relative to the variation among the sampled seizure snapshots. It estimates
    potential squared-error skill with access to the full Plant snapshot and history. The ratio
    is dimensionless, so the curves share an axis. Dashed lines mark when the corresponding
    feature Frames stop including shared pre-branch EEG.

    [Finite-ensemble variance estimator, Matsumura et al. (2010), equations 1–4](https://eprints.lib.hokudai.ac.jp/repo/huscap/all/43127/JMSJ88-1_43-62.pdf)
    """)
    return


@app.cell
def _(
    REPRESENTATIONS,
    REPRESENTATION_LABELS,
    bundle,
    mo,
    plt,
    representation_shared_history_s,
):
    mo.stop(
        not bundle.potential_predictability,
        mo.md(
            "**The older score archive lacks between-snapshot variance.** Regenerate and score "
            "a zero-current ensemble with the updated runner to calculate the ratio."
        ),
    )
    _fig, _ax = plt.subplots(figsize=(8, 4), layout="constrained")
    _ax.axhline(0, color="0.65", linewidth=0.7)
    for _rep in REPRESENTATIONS:
        _result = bundle.potential_predictability[_rep]
        (_line,) = _ax.plot(_result.times, _result.signal_fraction, label=REPRESENTATION_LABELS[_rep])
        _window = representation_shared_history_s(bundle, _rep)
        if _window > 0:
            _ax.axvline(_window, color=_line.get_color(), linestyle="--", alpha=0.6)
    _ax.set(xlabel="Lookahead from snapshot (s)", ylabel="Signal / total variance")
    _ax.legend()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Waveform predictability with and without the trailing mean

    At each Frame endpoint, take the last 25 samples (0.50 s at 50 Hz) of each EEG
    channel. The trailing mean $m_c$ averages those samples with a half-Hann taper,
    giving the newest samples the most weight. With weights $w_j$ ordered from oldest
    to newest, the two components are

    $$m_c(t)=\frac{\sum_{j=0}^{24}w_j x_c(t-(24-j)/50)}{\sum_{j=0}^{24}w_j},
      \qquad r_c(t)=x_c(t)-m_c(t).$$

    The mean captures slow variation. The residual is the waveform sample at the same
    endpoint minus that mean. The dashed line marks the 0.50 s support of the mean.

    The two Seizure State groups use the fraction of regions seizing **at the snapshot**,
    before the replays diverge. The 25% cutoff was chosen from a gap in the original
    ensemble's sampled states. Each curve normalizes by its target's total variance
    within that group.
    """)
    return


@app.cell
def _(mo, np, plt, waveform_split_path):
    mo.stop(
        not waveform_split_path.is_file(),
        mo.md("The selected archive has no waveform component curves."),
    )
    with np.load(waveform_split_path) as _data:
        _times = _data["times"]
        _ceilings = _data["ceilings"]
        _counts = _data["counts"]
        _ranges = _data["state_ranges"]
        _threshold = float(_data["threshold"])
        _support_s = float(_data["support_s"])
    _labels = ("Waveform", "Trailing weighted mean", "Waveform minus mean")
    _titles = ("All snapshots", f"Seizure State < {_threshold:.0%}", f"Seizure State ≥ {_threshold:.0%}")
    _fig, _axes = plt.subplots(1, 3, sharey=True, figsize=(12, 3.8), layout="constrained")
    for _ax, _curves, _title, _count, _range in zip(_axes, _ceilings, _titles, _counts, _ranges, strict=True):
        for _label, _curve in zip(_labels, _curves, strict=True):
            _ax.plot(_times, _curve, label=_label)
        _ax.axhline(0, color="0.65", linewidth=0.7)
        _ax.axvline(_support_s, color="0.55", linestyle="--", linewidth=0.8)
        _ax.set(title=f"{_title}\nn={_count} · observed {_range[0]:.0%} to {_range[1]:.0%}", xlabel="Lookahead (s)")
    _axes[0].set_ylabel("Signal / total variance")
    _fig.legend(*_axes[0].get_legend_handles_labels(), loc="outside upper center", ncols=3)
    _fig
    return


if __name__ == "__main__":
    app.run()
