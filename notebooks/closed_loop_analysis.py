import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium", app_title="Stage 8 Closed Loop MPC Analysis")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import yaml

    from neuro.jansen_rit import lfp
    from neuro.metrics import seizure_state
    from neuro.seizure import SEIZURE_PTP_MV, SPREAD_WINDOW_S, spread_profile_from_lfp
    from neuro.stimulation.base import select_rows
    from utils.plotting import plot_signals
    from utils.processing import steady_window

    return (
        Path,
        SEIZURE_PTP_MV,
        SPREAD_WINDOW_S,
        lfp,
        mo,
        np,
        plot_signals,
        plt,
        seizure_state,
        select_rows,
        spread_profile_from_lfp,
        steady_window,
        yaml,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # 🧠 Closed Loop MPC Analysis

    Analyze **one** closed-loop simulation at a time. Point the input below at a single run
    directory — the output of `scripts/run_simulation.py`, containing a log archive
    (`log.npz` or `logs.npz`) and its config (`config.yaml` or any `*.yaml`).

    Views:
    - **Run summary**: MPC weights + aggregate metrics (EEG energy, control $L_1$/$L_2$, cost).
    - **Time-series**: EEG output and control inputs.
    - **Seizure state $s(t)$**: the fraction of regions seizing, optionally against an
      uncontrolled run of the same plant.
    - **L1 stimulation** (`w_u_l1` test): per-electrode traces, sparsity (off-fraction +
      active-electrode count), per-electrode $L_1$, Kirchhoff residual, and suppression.
    """)
    return


@app.cell
def _(mo):
    run_dir_input = mo.ui.text(value="data/mpc_mse02_eeg_ms", label="Simulation Run Directory", full_width=True)
    baseline_dir_input = mo.ui.text(
        value="data/uncontrolled", label="Uncontrolled Reference (optional)", full_width=True
    )
    mo.md(f"**Run directory:** {run_dir_input}\n\n**Uncontrolled reference:** {baseline_dir_input}")
    return baseline_dir_input, run_dir_input


@app.cell
def _(Path, lfp, np, select_rows, yaml):
    def find_run(directory):
        """The log archive path and parsed config of a run directory, or ``None`` if it holds neither."""

        def hits(names, patterns):
            found = [directory / name for name in names if (directory / name).exists()]
            for pattern in patterns:
                found += sorted(directory.glob(pattern))
            return found

        npz_hits = hits(["log.npz", "logs.npz"], ["*.npz"])
        config_hits = hits(["config.yaml"], ["*.yaml", "*.yml"])
        if not npz_hits or not config_hits:
            return None
        with config_hits[0].open() as f:
            return npz_hits[0], yaml.safe_load(f)

    def region_lfp(data):
        """Region LFP ``(n_regions, n_samples)``, or ``None`` if the run logged no region signal."""
        if "dynamics.lfp" in data.files:
            return data["dynamics.lfp"].T
        if "dynamics.x" in data.files:
            return lfp(np.moveaxis(data["dynamics.x"], 0, -1))
        return None

    def electrode_labels(config, n_u):
        """Montage labels for an ``n_u``-wide control, read off the field projection the config points at."""
        stim = config.get("dynamics", {}).get("stimulation", {})
        path = stim.get("field_projection_path")
        if path is not None and Path(path).exists():
            with np.load(path) as projection:
                labels = projection["channel_labels"]
            wanted = stim.get("electrodes")
            rows = select_rows(labels, wanted) if wanted else slice(None)
            selected = [str(label) for label in labels[rows]]
            if len(selected) == n_u:
                return selected
        return [f"E{i}" for i in range(n_u)]

    def plant_block(config):
        """The config entries that define the plant, so two runs can be checked for comparability."""
        dynamics = config.get("dynamics", {})
        return {key: dynamics.get(key) for key in ("dt", "seed", "initial_state", "connectome", "params")}

    return electrode_labels, find_run, plant_block, region_lfp


@app.cell
def _(Path, electrode_labels, find_run, mo, np, region_lfp, run_dir_input):
    _dir = Path(run_dir_input.value)
    _found = find_run(_dir)
    mo.stop(
        _found is None,
        mo.md(f"⚠️ **No simulation found in** `{_dir}` — expected a `log.npz`/`logs.npz` and a `*.yaml`."),
    )
    _npz_path, _config = _found

    with np.load(_npz_path) as data:
        _y_mea = data["sensor_0.y_mea"]
        _u = data["controller.u"]
        _u = _u.reshape(_u.shape[0], -1)
        _nan = np.full(_u.shape[0], np.nan)
        _cost = data["controller.cost"] if "controller.cost" in data.files else _nan
        _y_reg = region_lfp(data)

    _ctrl = _config.get("controller", {})
    _near_zero = np.abs(_u) < 1e-6
    run = {
        "dir": _dir,
        "label": f"w_u_l1={float(_ctrl.get('w_u_l1', 0.0)):g}",
        "config": _config,
        "controller": _ctrl,
        "electrodes": electrode_labels(_config, _u.shape[1]),
        "dt": float(_config.get("dynamics", {}).get("dt", 1e-4)),
        "y_mea": _y_mea,
        "y_reg": _y_reg,
        "u": _u,
        "cost": _cost,
        "eeg_energy": float(np.mean(_y_mea**2)),
        "control_energy": float(np.sum(_u**2)),
        "control_l1": float(np.sum(np.abs(_u))),
        "control_l1_per_electrode": np.abs(_u).sum(axis=0),
        "max_u": float(np.max(np.abs(_u))),
        "frac_off": _near_zero.mean(axis=0),
        "mean_active": float((~_near_zero).sum(axis=1).mean()),
        "kcl": _u.sum(axis=1),
    }
    return (run,)


@app.cell
def _(Path, baseline_dir_input, find_run, np, region_lfp):
    _value = baseline_dir_input.value.strip()
    _found = find_run(Path(_value)) if _value else None

    baseline = None
    if _found is not None:
        _npz_path, _config = _found
        with np.load(_npz_path) as _data:
            _y_reg = region_lfp(_data)
        baseline = {
            "dir": Path(_value),
            "config": _config,
            "dt": float(_config.get("dynamics", {}).get("dt", 1e-4)),
            "y_reg": _y_reg,
        }

    # Distinguishes "left blank" from "pointed somewhere that holds no run", which needs saying.
    baseline_missing = bool(_value) and _found is None
    return baseline, baseline_missing


@app.cell
def _(mo):
    mo.md("""
    ## 📊 Run Summary
    """)
    return


@app.cell
def _(mo, np, run):
    _ctrl = run["controller"]
    _mean_cost = float(np.nanmean(run["cost"])) if not np.isnan(run["cost"]).all() else None
    _rows = [
        {"Metric": "Run directory", "Value": str(run["dir"])},
        {"Metric": "w_y (EEG-power weight)", "Value": _ctrl.get("w_y", 1.0)},
        {"Metric": "w_u (quadratic effort)", "Value": _ctrl.get("w_u", 0.0)},
        {"Metric": "w_u_l1 (L1 sparsity)", "Value": _ctrl.get("w_u_l1", 0.0)},
        {"Metric": "horizon", "Value": _ctrl.get("horizon", "—")},
        {"Metric": "u_max", "Value": _ctrl.get("u_max", "—")},
        {"Metric": "EEG energy  mean(y²)  (lower = better suppression)", "Value": round(run["eeg_energy"], 4)},
        {"Metric": "Control energy  Σu²", "Value": round(run["control_energy"], 4)},
        {"Metric": "Control L1  Σ|u|", "Value": round(run["control_l1"], 4)},
        {"Metric": "Max control amplitude  max|u|", "Value": round(run["max_u"], 4)},
        {"Metric": "Mean active electrodes / step", "Value": round(run["mean_active"], 3)},
        {"Metric": "Kirchhoff residual  max|Σu|", "Value": f"{np.abs(run['kcl']).max():.2e}"},
        {"Metric": "Mean MPC cost", "Value": round(_mean_cost, 4) if _mean_cost is not None else "N/A"},
    ]
    mo.ui.table(_rows, selection=None)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 📉 EEG Output & Control Currents
    """)
    return


@app.cell
def _(np, plt, run):
    _y = run["y_mea"]
    _u = run["u"]
    _dt = run["dt"]
    _t_y = np.arange(_y.shape[0]) * _dt
    _t_u = np.arange(_u.shape[0]) * _dt

    _fig_ts, _axes_ts = plt.subplots(2, 1, figsize=(11, 8), sharex=True, layout="constrained")

    _n_channels_to_plot = min(8, _y.shape[1])
    _axes_ts[0].plot(_t_y, _y[:, :_n_channels_to_plot], linewidth=0.8)
    _axes_ts[0].set_title(f"EEG Output (first {_n_channels_to_plot} of {_y.shape[1]} channels) — {run['label']}")
    _axes_ts[0].set_ylabel("Voltage")
    _axes_ts[0].grid(visible=True, linestyle="--", alpha=0.5)
    _axes_ts[0].spines[["top", "right"]].set_visible(False)

    _axes_ts[1].step(_t_u, _u, where="post", linewidth=1.2)
    _axes_ts[1].legend(run["electrodes"], loc="upper right")
    _axes_ts[1].set_title("Control Currents (per electrode)")
    _axes_ts[1].set_ylabel("Current")
    _axes_ts[1].set_xlabel("Time (s)")
    _axes_ts[1].grid(visible=True, linestyle="--", alpha=0.5)
    _axes_ts[1].spines[["top", "right"]].set_visible(False)
    _fig_ts
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 🌊 Seizure State $s(t)$

    $s(t)$ is the fraction of the 76 regions whose peak-to-peak LFP over a
    $W = 1$ s window exceeds the calibrated $5$ mV seizing threshold — the region-space ground
    truth the EEG objective is only a proxy for. Its time mean is the **seizure burden**,
    the score `sweep_nn_predictor` minimises.

    Each run is drawn twice, from the same criterion on two window grids:

    - **solid** — `SpreadProfile`, stamped at the window *centre* (hop $0.25$ s);
    - **dashed** — `metrics.seizure_state`, stamped at the window *end* (hop $0.05$ s), the
      earliest instant a real-time controller could know the value.

    The offset between them is the $W/2 = 0.5$ s lag causality costs.
    """)
    return


@app.cell
def _(
    SEIZURE_PTP_MV,
    SPREAD_WINDOW_S,
    baseline,
    run,
    seizure_state,
    spread_profile_from_lfp,
):
    def _curves(y_reg, dt):
        """The centred profile and the causal reading of s(t), both on the run's own clock."""
        profile = spread_profile_from_lfp(y_reg, dt, threshold=SEIZURE_PTP_MV)
        times, state = seizure_state(y_reg, 1.0 / dt, threshold=SEIZURE_PTP_MV)
        return {"profile": profile, "t_causal": times + SPREAD_WINDOW_S, "s_causal": state}

    seizure = [
        {"name": _name, "colour": _colour, "dir": _entry["dir"], **_curves(_entry["y_reg"], _entry["dt"])}
        for _name, _entry, _colour in (("controlled", run, "#1f77b4"), ("uncontrolled", baseline, "#d62728"))
        if _entry is not None and _entry["y_reg"] is not None
    ]
    return (seizure,)


@app.cell
def _(baseline, baseline_missing, mo, plant_block, run, seizure):
    _warnings = []
    if not seizure:
        _warnings.append(
            "⚠️ This run logged no region signal, so $s(t)$ is undefined. Re-run it with "
            "`dynamics.log: lfp` (or `state`)."
        )
    if baseline_missing:
        _warnings.append("⚠️ **No simulation found** in the uncontrolled reference directory.")
    if baseline is not None and plant_block(baseline["config"]) != plant_block(run["config"]):
        _warnings.append(
            "⚠️ The two runs do **not** share a plant (`dt`, `seed`, `initial_state`, `connectome` "
            "or `params` differ), so the difference between them is not attributable to the stimulation."
        )
    mo.md("\n\n".join(_warnings)) if _warnings else None
    return


@app.cell
def _(plt, seizure):
    _fig_s, _ax_s = plt.subplots(figsize=(11, 4.2), layout="constrained")
    for _c in seizure:
        _ax_s.plot(
            _c["profile"].times,
            _c["profile"].seizure_state(),
            color=_c["colour"],
            linewidth=1.6,
            label=f"{_c['name']} — centred",
        )
        _ax_s.plot(
            _c["t_causal"],
            _c["s_causal"],
            color=_c["colour"],
            linewidth=1.0,
            linestyle="--",
            alpha=0.8,
            label=f"{_c['name']} — causal",
        )
    _ax_s.set_xlabel("Time (s)")
    _ax_s.set_ylabel(r"$s(t)$   fraction of regions seizing")
    _ax_s.set_ylim(0, 1)
    _ax_s.set_title(r"Seizure state $s(t)$ — solid = window-centred, dashed = causal")
    _ax_s.grid(visible=True, linestyle="--", alpha=0.5)
    _ax_s.spines[["top", "right"]].set_visible(False)
    if seizure:
        _ax_s.legend(loc="upper left", fontsize=8, ncols=len(seizure))
    _fig_s
    return


@app.cell
def _(mo, np, seizure):
    def _stats(profile):
        """The four onset/burden numbers a run is compared on; NaN where nothing was recruited."""
        onsets = profile.onsets
        recruited = onsets[np.isfinite(onsets)]
        return {
            "Seizure burden  mean s(t)  (lower = better)": profile.burden(),
            "Final seizure state": float(profile.seizure_state()[-1]),
            "First onset (s)": float(recruited.min()) if recruited.size else float("nan"),
            "Median onset of recruited regions (s)": float(np.median(recruited)) if recruited.size else float("nan"),
        }

    _by_name = {_c["name"]: _stats(_c["profile"]) for _c in seizure}
    _controlled = _by_name.get("controlled", {})
    _uncontrolled = _by_name.get("uncontrolled")

    _rows = []
    for _metric, _value in _controlled.items():
        _row = {"Metric": _metric, "controlled": round(_value, 4)}
        if _uncontrolled is not None:
            _row["uncontrolled"] = round(_uncontrolled[_metric], 4)
            _row["Δ"] = round(_value - _uncontrolled[_metric], 4)
        _rows.append(_row)

    mo.ui.table(_rows, selection=None) if _rows else mo.md("_No seizure statistics for this run._")
    return


@app.cell
def _(mo, run):
    mo.md(rf"""
    ## 🔌 L1 Sparsity & Stimulation

    Testing the `w_u_l1` term for this run (**{run["label"]}**): the L1 penalty
    $w_{{u,l_1}}\sum_k\lVert u_k\rVert_1$ soft-thresholds electrode currents toward exact zero
    (a sparser montage) while obeying Kirchhoff's law $\sum_\text{{electrodes}} u = 0$.
    """)
    return


@app.cell
def _(plot_signals, plt, run):
    _colors = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]
    _n_ctrl = run["u"].shape[1]
    _fig_tr, _ax_tr = plt.subplots(figsize=(11, 4.2), layout="constrained")
    plot_signals(
        run["u"].T,
        dt_ms=run["dt"] * 1000.0,
        channel_names=run["electrodes"],
        channels_to_plot=list(range(_n_ctrl)),
        stacked=False,
        color=_colors[:_n_ctrl],
        title=f"Per-electrode stimulation  ({run['label']}, mean active = {run['mean_active']:.2f})",
        ax=_ax_tr,
    )
    _ax_tr.set_ylabel("Current")
    _fig_tr
    return


@app.cell
def _(np, plt, run):
    _elec = run["electrodes"]
    _frac = run["frac_off"]
    _x = np.arange(len(_elec))

    _fig_sp, _ax_sp = plt.subplots(figsize=(8, 4.2), layout="constrained")
    _bars = _ax_sp.bar(_x, _frac, color="#1f77b4")
    _ax_sp.bar_label(_bars, fmt="%.3f", padding=3)
    _ax_sp.set_xticks(_x)
    _ax_sp.set_xticklabels(_elec)
    _ax_sp.set_ylabel("Fraction of time |current| < 1e-6")
    _ax_sp.set_ylim(0, 1.05)
    _ax_sp.set_title(f"L1 sparsity: electrode off-fraction (mean active = {run['mean_active']:.2f} / {len(_elec)})")
    _ax_sp.grid(visible=True, axis="y", linestyle="--", alpha=0.5)
    _ax_sp.spines[["top", "right"]].set_visible(False)
    _fig_sp
    return


@app.cell
def _(np, plt, run):
    _elec = run["electrodes"]
    _x = np.arange(len(_elec))
    _fig_mag, _ax_mag = plt.subplots(figsize=(8, 4.2), layout="constrained")
    _bars = _ax_mag.bar(_x, run["control_l1_per_electrode"], color="#ff7f0e")
    _ax_mag.bar_label(_bars, fmt="%.0f", padding=3)
    _ax_mag.set_xticks(_x)
    _ax_mag.set_xticklabels(_elec)
    _ax_mag.set_ylabel(r"$\sum_t |u|$")
    _ax_mag.set_title(r"Per-electrode control $L_1$ (total absolute current)")
    _ax_mag.grid(visible=True, axis="y", linestyle="--", alpha=0.5)
    _ax_mag.spines[["top", "right"]].set_visible(False)
    _fig_mag
    return


@app.cell
def _(np, plt, run):
    _kcl = run["kcl"]
    _t = np.arange(_kcl.shape[0]) * run["dt"]
    _fig_k, _ax_k = plt.subplots(figsize=(11, 3.6), layout="constrained")
    _ax_k.plot(_t, _kcl, linewidth=0.7, color="#2ca02c")
    _ax_k.set_xlabel("Time (s)")
    _ax_k.set_ylabel(r"$\sum_\mathrm{electrodes} u$")
    _ax_k.set_title(f"Kirchhoff current-law residual (max |Σu| = {np.abs(_kcl).max():.1e}, should be ~0)")
    _ax_k.grid(visible=True, linestyle="--", alpha=0.5)
    _ax_k.spines[["top", "right"]].set_visible(False)
    _fig_k
    return


@app.cell
def _(mo):
    transient_ms = mo.ui.number(value=1000.0, start=0.0, stop=100000.0, step=100.0, label="Transient to drop (ms)")
    transient_ms
    return (transient_ms,)


@app.cell
def _(np, plt, run, steady_window, transient_ms):
    _y = run["y_mea"]
    _dt_ms = run["dt"] * 1000.0
    _t = np.arange(_y.shape[0]) * run["dt"]
    _power = np.mean(_y**2, axis=1)

    _steady = steady_window(_y.T, _dt_ms, float(transient_ms.value))
    _steady_ms = float(np.mean(_steady**2))
    _transient_s = float(transient_ms.value) / 1000.0

    _fig_su, _ax_su = plt.subplots(figsize=(11, 3.8), layout="constrained")
    _ax_su.plot(_t, _power, linewidth=0.7, color="#d62728")
    if _transient_s > 0:
        _ax_su.axvspan(0, _transient_s, color="grey", alpha=0.15, label="dropped transient")
        _ax_su.legend(loc="upper right")
    _ax_su.set_xlabel("Time (s)")
    _ax_su.set_ylabel(r"EEG power  $\langle y^2\rangle$")
    _ax_su.set_title(f"Suppression — steady-window mean-square EEG = {_steady_ms:.3f} (lower = better)")
    _ax_su.grid(visible=True, linestyle="--", alpha=0.5)
    _ax_su.spines[["top", "right"]].set_visible(False)
    _fig_su
    return


if __name__ == "__main__":
    app.run()
