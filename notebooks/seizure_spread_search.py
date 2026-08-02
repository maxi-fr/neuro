import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium", app_title="Seizure Spread: K / sigma Search")


@app.cell
def imports():
    """imports definition."""
    from dataclasses import asdict, replace
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    from matplotlib import patheffects as path_effects
    from matplotlib import pyplot as plt

    from neuro.connectome import Connectome
    from neuro.jansen_rit import JansenRitDynamics, JansenRitParams, resting_state
    from neuro.seizure import (
        DT,
        EZ_REGIONS,
        PZ_REGIONS,
        SpreadProfile,
        build_seizure_a_gains,
        spread_profile,
        spread_summary,
    )

    artifact_base = Path(__file__).parent.parent / "artifacts"
    sweep_dirs = sorted(d.name for d in artifact_base.iterdir() if d.is_dir() and (d / "sweep.npz").exists())

    def _grid_size(name: str) -> int:
        """_grid_size definition."""
        with np.load(artifact_base / name / "sweep.npz") as data:
            return int(data["k_vals"].size * data["sigma_vals"].size)

    default_sweep = max(sweep_dirs, key=_grid_size) if sweep_dirs else None
    return (
        Connectome,
        DT,
        EZ_REGIONS,
        JansenRitDynamics,
        JansenRitParams,
        PZ_REGIONS,
        SpreadProfile,
        artifact_base,
        asdict,
        build_seizure_a_gains,
        default_sweep,
        mo,
        np,
        path_effects,
        pd,
        plt,
        replace,
        resting_state,
        spread_profile,
        spread_summary,
        sweep_dirs,
    )


@app.cell(hide_code=True)
def intro(mo):
    """intro definition."""
    mo.md(r"""
    # 🌊 Seizure spread — searching $K$ and $\sigma$

    The EZ/PZ regime fixes the excitability map ($A = 3.6$ at `lHC`/`lPHC`/`lAMYG`, $A = 3.4$ at
    `lTCI`/`lTCV`, $A = 3.25$ elsewhere). What that map does *not* fix is how fast the seizure
    leaves the focus. Two knobs set the drive that carries it:

    * **$K$** — the global coupling gain multiplying $\sum_j w_{ij}\,S(y_j)$. It is the
      deterministic, structure-following part of the drive: raise it and a seizing region
      pushes its neighbours over their own threshold.
    * **$\sigma$** — the std of the white noise on $\dot x_5$. It is the stochastic part:
      a subcritical region escapes to the ictal limit cycle only when a noise excursion
      carries it there, so $\sigma$ sets *how long that takes*.

    **Goal:** the seizure ignites in the EZ, reaches the PZ a few seconds later, and takes
    roughly 10 s to work through the left hemisphere, leaving the right hemisphere healthy.

    **Recruitment detector.** A region is seizing when the peak-to-peak swing of its
    $y = x_2 - x_3$ over a 1 s window exceeds 5 mV — comfortably above the noise-driven
    background ($\le 2$ mV) and below the ~14 mV ictal limit cycle. Its *onset* is the first
    window that begins 1 s of uninterrupted supra-threshold activity; the persistence rule is
    what separates a real recruitment from a single noise burst.
    """)
    return


@app.cell
def ui_sweep(default_sweep, mo, sweep_dirs):
    """ui_sweep definition."""
    sweep_dropdown = mo.ui.dropdown(options=sweep_dirs, value=default_sweep, label="Sweep directory")
    sweep_dropdown
    return (sweep_dropdown,)


@app.cell
def load_sweep(artifact_base, mo, np, sweep_dropdown):
    """load_sweep definition."""
    mo.stop(sweep_dropdown.value is None, "No sweep found — run `scripts/sweep_seizure_spread.py` first.")
    with np.load(artifact_base / sweep_dropdown.value / "sweep.npz") as _data:
        k_vals = _data["k_vals"]
        sigma_vals = _data["sigma_vals"]
        seeds = _data["seeds"]
        duration = float(_data["duration"])
        times = _data["times"]
        ptp = _data["ptp"]
        from_rest = bool(_data["from_rest"])
        region_labels = _data["region_labels"]
        hemispheres = _data["hemispheres"]

    mo.md(
        f"**{ptp.shape[0]} × {ptp.shape[1]} grid**, {len(seeds)} seeds, {duration:.0f} s per run, "
        f"initial state: **{'healthy resting state' if from_rest else 'all-zeros'}**  \n"
        f"$K \\in [{k_vals.min():.3f}, {k_vals.max():.3f}]$, "
        f"$\\sigma \\in [{sigma_vals.min():.0f}, {sigma_vals.max():.0f}]$"
    )
    return (
        duration,
        hemispheres,
        k_vals,
        ptp,
        region_labels,
        seeds,
        sigma_vals,
        times,
    )


@app.cell
def connectome_cell(Connectome, hemispheres, np, region_labels):
    """connectome_cell definition."""
    conn = Connectome.from_config({"speed": 50.0})
    if not np.array_equal(conn.region_labels, region_labels):
        msg = "the sweep ran on a different parcellation than the connectome loaded here"
        raise ValueError(msg)
    left_mask = ~hemispheres
    return conn, left_mask


@app.cell
def ui_threshold(mo):
    """ui_threshold definition."""
    thr_slider = mo.ui.slider(2.0, 10.0, 0.5, value=5.0, label="Seizing threshold (mV peak-to-peak)")
    thr_slider
    return (thr_slider,)


@app.cell
def summaries(
    SpreadProfile,
    asdict,
    conn,
    duration,
    k_vals,
    pd,
    ptp,
    sigma_vals,
    spread_summary,
    thr_slider,
    times,
):
    """summaries definition."""
    _rows = []
    for _i, _k in enumerate(k_vals):
        for _j, _s in enumerate(sigma_vals):
            for _n in range(ptp.shape[2]):
                _profile = SpreadProfile.from_ptp(times, ptp[_i, _j, _n], threshold=thr_slider.value)
                _summary = spread_summary(_profile, conn)
                _rows.append(
                    {"K": _k, "sigma": _s, "seed_idx": _n, "score": _summary.score(duration), **asdict(_summary)}
                )

    trials = pd.DataFrame(_rows)

    cells = (
        trials.fillna(duration).groupby(["K", "sigma"], as_index=False).mean(numeric_only=True).drop(columns="seed_idx")
    )
    spread_maps = {
        col: cells.pivot_table(index="K", columns="sigma", values=col).to_numpy()
        for col in ("score", "t_ez", "t_pz", "t_left_half", "frac_left", "frac_right")
    }
    return cells, spread_maps, trials


@app.cell(hide_code=True)
def heatmaps(k_vals, mo, path_effects, plt, sigma_vals, spread_maps):
    """heatmaps definition."""
    _panels = [
        ("score", "Score (lower = closer to target)", "viridis_r", None),
        ("t_pz", "PZ recruited (s)", "magma", None),
        ("t_left_half", "Half the left hemisphere recruited (s)", "magma", None),
        ("frac_left", "Left hemisphere recruited (fraction)", "Blues", (0, 1)),
        ("frac_right", "Right hemisphere recruited (fraction)", "Reds", (0, 1)),
        ("t_ez", "EZ recruited (s)", "magma", None),
    ]

    _fig, _axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for _ax, (_key, _title, _cmap, _lim) in zip(_axes.ravel(), _panels, strict=True):
        _m = spread_maps[_key]
        _im = _ax.imshow(
            _m,
            origin="lower",
            aspect="auto",
            cmap=_cmap,
            vmin=None if _lim is None else _lim[0],
            vmax=None if _lim is None else _lim[1],
        )
        _ax.set_xticks(range(len(sigma_vals)), [f"{_v:.0f}" for _v in sigma_vals])
        _ax.set_yticks(range(len(k_vals)), [f"{_v:.3f}" for _v in k_vals])
        _ax.set_xlabel("σ", fontweight="bold")
        _ax.set_ylabel("K", fontweight="bold")
        _ax.set_title(_title, fontsize=11, fontweight="bold")
        _fig.colorbar(_im, ax=_ax)
        for _r in range(_m.shape[0]):
            for _c in range(_m.shape[1]):
                _ax.text(
                    _c,
                    _r,
                    f"{_m[_r, _c]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="w",
                    path_effects=[path_effects.withStroke(linewidth=1.8, foreground="0.15")],
                )

    mo.md(f"""
    ## The $K \\times \\sigma$ landscape

    Seed-averaged. The two knobs trade off along a ridge: what matters is the *total* drive
    out of the focus, so a lower $K$ can be bought back with a higher $\\sigma$ — but only the
    stochastic part buys a slow, staged recruitment, because raising $K$ past the point where
    coupling alone is supercritical recruits everything at once, right hemisphere included.

    {mo.as_html(_fig)}
    """)
    return


@app.cell(hide_code=True)
def ranked(cells, mo):
    """ranked definition."""
    mo.md(f"""
    ## Ranked grid cells

    {mo.as_html(cells.sort_values("score").head(12).round(3))}
    """)
    return


@app.cell
def ui_cell(cells, k_vals, mo, sigma_vals):
    """ui_cell definition."""
    _best = cells.loc[cells.score.idxmin()]
    k_dropdown = mo.ui.dropdown(
        options={f"{_v:.4f}": _i for _i, _v in enumerate(k_vals)}, value=f"{_best.K:.4f}", label="K"
    )
    sigma_dropdown = mo.ui.dropdown(
        options={f"{_v:.0f}": _i for _i, _v in enumerate(sigma_vals)}, value=f"{_best.sigma:.0f}", label="σ"
    )
    mo.hstack([k_dropdown, sigma_dropdown], justify="start", gap=2)
    return k_dropdown, sigma_dropdown


@app.cell(hide_code=True)
def cell_detail(
    EZ_REGIONS,
    PZ_REGIONS,
    SpreadProfile,
    conn,
    k_dropdown,
    k_vals,
    left_mask,
    mo,
    np,
    plt,
    ptp,
    seeds,
    sigma_dropdown,
    sigma_vals,
    thr_slider,
    times,
):
    """cell_detail definition."""
    _i, _j = k_dropdown.value, sigma_dropdown.value
    _profiles = [SpreadProfile.from_ptp(times, ptp[_i, _j, _n], threshold=thr_slider.value) for _n in range(len(seeds))]

    _ez = [conn.region_index[_r] for _r in EZ_REGIONS]
    _pz = [conn.region_index[_r] for _r in PZ_REGIONS]
    _other_left = [_n for _n in np.flatnonzero(left_mask) if _n not in _ez + _pz]

    _fig, (_ax0, _ax1) = plt.subplots(2, 1, figsize=(13, 13), height_ratios=[1, 2.2], layout="constrained")

    for _grp, _nodes, _col in (
        ("EZ", _ez, "#d62728"),
        ("PZ", _pz, "#ff7f0e"),
        ("left, other", _other_left, "#1f77b4"),
        ("right", np.flatnonzero(~left_mask), "#7f7f7f"),
    ):
        _curves = np.asarray([np.mean(_p.onsets[_nodes, None] <= times, axis=0) for _p in _profiles])
        _ax0.plot(times, _curves.mean(0), color=_col, lw=2, label=_grp)
        _ax0.fill_between(times, _curves.min(0), _curves.max(0), color=_col, alpha=0.18)
    _ax0.set_xlabel("Time (s)", fontweight="bold")
    _ax0.set_ylabel("Fraction of group recruited", fontweight="bold")
    _ax0.set_title(f"Recruitment, K={k_vals[_i]:.4f}, σ={sigma_vals[_j]:.0f} (band = seed spread)", fontweight="bold")
    _ax0.legend(loc="lower right")
    _ax0.grid(visible=True, ls="--", alpha=0.3)
    _ax0.spines[["top", "right"]].set_visible(False)

    _onsets = _profiles[0].onsets
    _order = np.argsort(np.nan_to_num(_onsets, nan=np.inf))
    for _row, _n in enumerate(_order):
        _col = "#d62728" if _n in _ez else "#ff7f0e" if _n in _pz else "#1f77b4" if left_mask[_n] else "#7f7f7f"
        if np.isfinite(_onsets[_n]):
            _ax1.barh(_row, _onsets[_n], color=_col, height=0.75)
        else:
            _ax1.plot([times[0], times[-1]], [_row, _row], color=_col, ls=":", lw=0.8, alpha=0.5)
    _ax1.set_yticks(range(len(_order)), [conn.region_labels[_n] for _n in _order], fontsize=6)
    _ax1.set_ylim(-1, len(_order))
    _ax1.set_xlim(0, times[-1])
    _ax1.set_xlabel("Onset (s)", fontweight="bold")
    _ax1.set_title(f"Onset order, seed {seeds[0]} (dotted = never recruited)", fontweight="bold")
    _ax1.grid(visible=True, axis="x", ls="--", alpha=0.3)
    _ax1.spines[["top", "right"]].set_visible(False)

    mo.md(f"""
    ## One grid cell up close

    {mo.as_html(_fig)}
    """)
    return


@app.cell(hide_code=True)
def seed_spread(duration, k_dropdown, mo, np, plt, sigma_dropdown, trials):
    """seed_spread definition."""
    _sel = trials[
        (trials.K.unique()[k_dropdown.value] == trials.K)
        & (trials.sigma == trials.sigma.unique()[sigma_dropdown.value])
    ]

    _fig, _ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for _key, _col, _label in (
        ("t_ez", "#d62728", "EZ"),
        ("t_pz", "#ff7f0e", "PZ"),
        ("t_left_half", "#1f77b4", "half the left hemisphere"),
    ):
        _never = _sel[_key].isna()
        _ax.plot(_sel.seed_idx, _sel[_key].fillna(duration), "o-", color=_col, label=_label)
        _ax.plot(_sel.seed_idx[_never], np.full(_never.sum(), duration), "o", mfc="w", mec=_col, mew=1.8)
    _ax.axhline(duration, color="0.4", ls=":", lw=1)
    _ax.text(_sel.seed_idx.min(), duration, " never recruited", va="bottom", fontsize=8, color="0.4")
    _ax.set_xlabel("Seed index", fontweight="bold")
    _ax.set_ylabel("Onset (s)", fontweight="bold")
    _ax.set_title("Seed-to-seed variability of the schedule", fontweight="bold")
    _ax.legend(loc="center right")
    _ax.grid(visible=True, ls="--", alpha=0.3)
    _ax.spines[["top", "right"]].set_visible(False)

    mo.md(f"""
    ### How reproducible is the schedule?

    Recruitment is a noise-driven escape, so its timing is a random variable — the closer the
    cell sits to the coupling-only threshold, the slower *and* the more variable it is. A gap
    between seeds of a few seconds is inherent, not a bug; a cell whose PZ recruits on some
    seeds and not others is being run below threshold.

    {mo.as_html(_fig)}
    """)
    return


@app.cell(hide_code=True)
def ui_verify(mo):
    """ui_verify definition."""
    verify_button = mo.ui.run_button(label="Re-simulate the selected cell (~40 s)")
    verify_duration = mo.ui.slider(10.0, 40.0, 5.0, value=20.0, label="Duration (s)")
    verify_seed = mo.ui.number(0, 10_000, 1, value=69, label="Seed")
    mo.vstack(
        [
            mo.md("## Verification — the raw traces"),
            mo.hstack([verify_duration, verify_seed, verify_button], justify="start", gap=2),
        ]
    )
    return verify_button, verify_duration, verify_seed


@app.cell
def verify(
    DT,
    EZ_REGIONS,
    JansenRitDynamics,
    JansenRitParams,
    PZ_REGIONS,
    build_seizure_a_gains,
    conn,
    k_dropdown,
    k_vals,
    left_mask,
    mo,
    np,
    plt,
    replace,
    resting_state,
    sigma_dropdown,
    sigma_vals,
    spread_profile,
    verify_button,
    verify_duration,
    verify_seed,
):
    """verify definition."""
    mo.stop(not verify_button.value, mo.md("*Press the button to run the plant at the selected $(K, \\sigma)$.*"))

    _conn = replace(conn, K=float(k_vals[k_dropdown.value]))
    _params = JansenRitParams(A=build_seizure_a_gains(_conn), sigma=float(sigma_vals[sigma_dropdown.value]))
    _dyn = JansenRitDynamics(
        dt=DT, params=_params, conn=_conn, seed=int(verify_seed.value), initial_state=resting_state(_conn, DT)
    )
    _prof = spread_profile(_dyn, verify_duration.value)

    _ez = [conn.region_index[_r] for _r in EZ_REGIONS]
    _pz = [conn.region_index[_r] for _r in PZ_REGIONS]
    _late = [_n for _n in np.argsort(np.nan_to_num(_prof.onsets, nan=np.inf)) if left_mask[_n] and _n not in _ez + _pz][
        :3
    ]
    _right = [_n for _n in np.argsort(np.nan_to_num(_prof.onsets, nan=np.inf)) if not left_mask[_n]][:2]

    _rows = [
        (_ez, "#d62728", "EZ"),
        (_pz, "#ff7f0e", "PZ"),
        (_late, "#1f77b4", "left, recruited"),
        (_right, "#7f7f7f", "right"),
    ]
    _fig, _ax = plt.subplots(figsize=(13, 6), layout="constrained")
    _offset = 0.0
    _yticks, _ylabels = [], []
    for _nodes, _col, _grp in _rows:
        for _n in _nodes:
            _ax.plot(_prof.times, _prof.ptp[_n] + _offset, color=_col, lw=1.4)
            _ax.axhline(_offset + _prof.threshold, color="k", ls=":", lw=0.6, alpha=0.5)
            if np.isfinite(_prof.onsets[_n]):
                _ax.plot(_prof.onsets[_n], _offset + _prof.threshold, "v", color=_col, ms=8)
            _yticks.append(_offset)
            _ylabels.append(f"{conn.region_labels[_n]} ({_grp})")
            _offset += 18.0
    _ax.set_yticks(_yticks, _ylabels, fontsize=8)
    _ax.set_xlabel("Time (s)", fontweight="bold")
    _ax.set_ylabel("Peak-to-peak amplitude, stacked (mV)", fontweight="bold")
    _ax.set_title(
        f"K={k_vals[k_dropdown.value]:.4f}, σ={sigma_vals[sigma_dropdown.value]:.0f}: "
        "amplitude envelopes, ▼ = onset, dotted = threshold",
        fontweight="bold",
    )
    _ax.grid(visible=True, axis="x", ls="--", alpha=0.3)
    _ax.spines[["top", "right"]].set_visible(False)
    _fig
    return


if __name__ == "__main__":
    app.run()
