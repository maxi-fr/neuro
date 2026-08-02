import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium", app_title="Stage 8 Closed Loop MPC Analysis")


@app.cell
def _():
    """Marimo cell."""
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import yaml

    from utils.plotting import plot_signals
    from utils.processing import steady_window

    return Path, mo, np, plot_signals, plt, steady_window, yaml


@app.cell
def _(mo):
    """Marimo cell."""
    mo.md(r"""
    # 🧠 Closed Loop MPC Analysis

    Analyze **one** closed-loop simulation at a time. Point the input below at a single run
    directory — the output of `scripts/run_simulation.py`, containing a log archive
    (`log.npz` or `logs.npz`) and its config (`config.yaml` or any `*.yaml`).

    Views:
    - **Run summary**: MPC weights + aggregate metrics (EEG energy, control $L_1$/$L_2$, cost).
    - **Time-series**: EEG output and control inputs.
    - **L1 stimulation** (`w_u_l1` test): per-electrode traces, sparsity (off-fraction +
      active-electrode count), per-electrode $L_1$, Kirchhoff residual, and suppression.
    """)
    return


@app.cell
def _(mo):
    """Marimo cell."""
    run_dir_input = mo.ui.text(
        value="artifacts/l1_before_after/after", label="Simulation Run Directory", full_width=True
    )
    mo.md(f"**Run directory:** {run_dir_input}")
    return (run_dir_input,)


@app.cell
def _(Path, mo, np, run_dir_input, yaml):
    """Marimo cell."""

    def _find(directory, names, patterns):
        """Files in *directory* matching an exact name, then any glob pattern (a possibly-empty list)."""
        _hits = [directory / _n for _n in names if (directory / _n).exists()]
        for _pat in patterns:
            _hits += sorted(directory.glob(_pat))
        return _hits

    _dir = Path(run_dir_input.value)
    _npz_hits = _find(_dir, ["log.npz", "logs.npz"], ["*.npz"])
    _config_hits = _find(_dir, ["config.yaml"], ["*.yaml", "*.yml"])
    mo.stop(
        not _npz_hits or not _config_hits,
        mo.md(f"⚠️ **No simulation found in** `{_dir}` — expected a `log.npz`/`logs.npz` and a `*.yaml`."),
    )
    _npz_path, _config_path = _npz_hits[0], _config_hits[0]

    with _config_path.open() as f:
        _config = yaml.safe_load(f)

    with np.load(_npz_path) as data:
        _y_mea = data["y_mea"]
        _u = data["u"]
        _u = _u.reshape(_u.shape[0], -1)
        _nan = np.full(_u.shape[0], np.nan)
        _cost = data.get("controller_cost", _nan)

    _ctrl = _config.get("controller", {})
    _electrodes = _config.get("dynamics", {}).get("connectome", {}).get("target_electrode")
    if not (isinstance(_electrodes, list) and len(_electrodes) == _u.shape[1]):
        _electrodes = [f"E{_i}" for _i in range(_u.shape[1])]

    _near_zero = np.abs(_u) < 1e-6
    run = {
        "dir": _dir,
        "label": f"w_u_l1={float(_ctrl.get('w_u_l1', 0.0)):g}",
        "config": _config,
        "controller": _ctrl,
        "electrodes": [str(_e) for _e in _electrodes],
        "dt": float(_config.get("dynamics", {}).get("dt", 1e-4)),
        "y_mea": _y_mea,
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
def _(mo):
    """Marimo cell."""
    mo.md("""
    ## 📊 Run Summary
    """)
    return


@app.cell
def _(mo, np, run):
    """Marimo cell."""
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
    """Marimo cell."""
    mo.md("""
    ## 📉 EEG Output & Control Inputs
    """)
    return


@app.cell
def _(np, plt, run):
    """Marimo cell."""
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
    _axes_ts[1].set_title("Control Inputs (per electrode)")
    _axes_ts[1].set_ylabel("Current")
    _axes_ts[1].set_xlabel("Time (s)")
    _axes_ts[1].grid(visible=True, linestyle="--", alpha=0.5)
    _axes_ts[1].spines[["top", "right"]].set_visible(False)
    _fig_ts
    return


@app.cell
def _(mo, run):
    """Marimo cell."""
    mo.md(rf"""
    ## 🔌 L1 Sparsity & Stimulation

    Testing the `w_u_l1` term for this run (**{run["label"]}**): the L1 penalty
    $w_{{u,l_1}}\sum_k\lVert u_k\rVert_1$ soft-thresholds electrode currents toward exact zero
    (a sparser montage) while obeying Kirchhoff's law $\sum_\text{{electrodes}} u = 0$.
    """)
    return


@app.cell
def _(plot_signals, plt, run):
    """Marimo cell."""
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
    """Marimo cell."""
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
    """Marimo cell."""
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
    """Marimo cell."""
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
    """Marimo cell."""
    transient_ms = mo.ui.number(value=1000.0, start=0.0, stop=100000.0, step=100.0, label="Transient to drop (ms)")
    transient_ms
    return (transient_ms,)


@app.cell
def _(np, plt, run, steady_window, transient_ms):
    """Marimo cell."""
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
