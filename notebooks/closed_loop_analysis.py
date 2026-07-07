import marimo

__generated_with = "0.23.13"
app = marimo.App(
    width="medium",
    app_title="Stage 8 Closed Loop MPC Comparison",
)


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import yaml

    return Path, mo, np, plt, yaml


@app.cell
def _(mo):
    mo.md(r"""
    # 🧠 Closed Loop MPC Comparison

    This notebook analyzes closed-loop simulation outputs for various MPC variants.
    It expects a directory structure where each subdirectory corresponds to a different controller variation and contains a `logs.npz` and `config.yaml`.

    Metrics extracted:
    - **Seizure Suppression (EEG Energy)**: Measured as the mean square of the EEG channels.
    - **Control Energy**: Total squared control input $\sum u^2$.
    - **Max Control Amplitude**: $\max |u|$.
    - **Computation Time**: Solve time per step (if available).
    - **Max Solver Iterations**: Max iterations per step (if available).
    """)
    return


@app.cell
def _(mo):
    data_dir_input = mo.ui.text(value="artifacts/mpc_comparison", label="Simulation Results Directory", full_width=True)
    mo.md(f"**Data directory:** {data_dir_input}")
    return (data_dir_input,)


@app.cell
def _(Path, data_dir_input, np, yaml):
    # Load all simulation results
    _results = {}
    _base_dir = Path(data_dir_input.value)

    if _base_dir.exists() and _base_dir.is_dir():
        for _subdir in _base_dir.iterdir():
            if _subdir.is_dir() and (_subdir / "logs.npz").exists() and (_subdir / "config.yaml").exists():
                _npz_path = _subdir / "logs.npz"
                _config_path = _subdir / "config.yaml"

                with _config_path.open() as f:
                    _config = yaml.safe_load(f)

                with np.load(_npz_path) as data:
                    _y_mea = data["y_mea"]  # Shape typically (n_samples, n_sensors)
                    _u = data["u"]  # Shape typically (n_samples, n_controls)

                    # Some metrics might not be logged yet, gracefully handle their absence
                    _solve_times = data.get("solve_times", np.zeros(_u.shape[0]) * np.nan)
                    _iterations = data.get("iterations", np.zeros(_u.shape[0]) * np.nan)

                    # Calculate aggregate metrics
                    _eeg_energy = np.mean(_y_mea**2)
                    _control_energy = np.sum(_u**2)
                    _max_u = np.max(np.abs(_u))

                    # Extract a better variant name from the artifact
                    _variant_name = _config.get("controller", {}).get("artifact", _subdir.name).split("/")[-1]

                    _results[_subdir.name] = {
                        "name": _variant_name,
                        "y_mea": _y_mea,
                        "u": _u,
                        "solve_times": _solve_times,
                        "iterations": _iterations,
                        "eeg_energy": _eeg_energy,
                        "control_energy": _control_energy,
                        "max_u": _max_u,
                        "config": _config,
                    }

    # Sort results by name
    results = {k: _results[k] for k in sorted(_results.keys())}
    return (results,)


@app.cell
def _(mo, results):
    mo.stop(
        not results,
        mo.md(
            "⚠️ **No simulation results found in the specified directory.** Please run the helper script to generate data first."
        ),
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 📊 Summary Metrics Table
    """)
    return


@app.cell
def _(mo, np, results):
    # Create a summary table
    _table_data = [
        {
            "Variant": r["name"],
            "EEG Energy (Lower = Better Suppression)": round(r["eeg_energy"], 4),
            "Control Energy": round(r["control_energy"], 4),
            "Max Control Amplitude": round(r["max_u"], 4),
            "Avg Solve Time (ms)": round(float(np.nanmean(r["solve_times"])), 4)
            if not np.isnan(r["solve_times"]).all()
            else "N/A",
            "Max Iterations": int(np.nanmax(r["iterations"])) if not np.isnan(r["iterations"]).all() else "N/A",
        }
        for r in results.values()
    ]
    mo.ui.table(_table_data)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 📈 Pareto Front: Control Energy vs Seizure Suppression
    """)
    return


@app.cell
def _(plt, results):
    _fig, _ax = plt.subplots(figsize=(8, 6))
    for _name, _data in results.items():
        _ax.scatter(_data["eeg_energy"], _data["control_energy"], s=100, label=_name, alpha=0.8)
        _ax.annotate(
            _name, (_data["eeg_energy"], _data["control_energy"]), xytext=(5, 5), textcoords="offset points", fontsize=9
        )

    _ax.set_title("Pareto Front: Control Effort vs EEG Energy")
    _ax.set_xlabel("EEG Energy (Suppression Error)")
    _ax.set_ylabel("Total Control Energy")
    _ax.grid(visible=True, linestyle="--", alpha=0.5)
    _ax.spines[["top", "right"]].set_visible(False)
    _ax.legend(loc="best")
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## ⏱️ Computation Time Analysis
    """)
    return


@app.cell
def _(mo, np, plt, results):
    _valid_times = {
        name: data["solve_times"][~np.isnan(data["solve_times"])]
        for name, data in results.items()
        if not np.isnan(data["solve_times"]).all()
    }

    if not _valid_times:
        _time_plot = mo.md("*Solve times were not logged in these results. Skipping box plot.*")
    else:
        _fig_time, _ax_time = plt.subplots(figsize=(10, 5))
        _labels = list(_valid_times.keys())
        _data_to_plot = list(_valid_times.values())

        _ax_time.boxplot(_data_to_plot, labels=_labels, patch_artist=True)
        _ax_time.set_title("Computation Time per Step across Variants")
        _ax_time.set_ylabel("Solve Time (ms)")
        _ax_time.tick_params(axis="x", rotation=45)
        _ax_time.grid(visible=True, axis="y", linestyle="--", alpha=0.5)
        _ax_time.spines[["top", "right"]].set_visible(False)
        _time_plot = _fig_time

    _time_plot
    return


@app.cell
def _(mo):
    mo.md("""
    ## 📉 Time-Series Trajectories
    """)
    return


@app.cell
def _(mo, results):
    variant_selector = mo.ui.dropdown(
        options=list(results.keys()), value=next(iter(results.keys())), label="Select Variant to Plot:"
    )
    variant_selector
    return (variant_selector,)


@app.cell
def _(np, plt, results, variant_selector):
    _selected = variant_selector.value
    _data = results[_selected]
    _y = _data["y_mea"]
    _u = _data["u"]
    _dt = float(_data["config"].get("dynamics", {}).get("dt", 0.0001))

    _t_y = np.arange(_y.shape[0]) * _dt
    _t_u = np.arange(_u.shape[0]) * _dt  # assuming same dt or scaled correctly

    _fig_ts, _axes_ts = plt.subplots(2, 1, figsize=(10, 8), sharex=True, layout="constrained")

    # Plot EEG
    _n_channels_to_plot = min(8, _y.shape[1]) if _y.ndim > 1 else 1
    if _y.ndim > 1:
        _axes_ts[0].plot(_t_y, _y[:, :_n_channels_to_plot], linewidth=0.8)
    else:
        _axes_ts[0].plot(_t_y, _y, linewidth=0.8)
    _axes_ts[0].set_title(f"EEG Output (First {_n_channels_to_plot} Channels) - {_selected}")
    _axes_ts[0].set_ylabel("Voltage")
    _axes_ts[0].grid(visible=True, linestyle="--", alpha=0.5)
    _axes_ts[0].spines[["top", "right"]].set_visible(False)

    # Plot Control Inputs
    _n_controls = _u.shape[1] if _u.ndim > 1 else 1
    if _u.ndim > 1:
        _axes_ts[1].step(_t_u, _u[:, :_n_controls], where="post", linewidth=1.5)
    else:
        _axes_ts[1].step(_t_u, _u, where="post", linewidth=1.5)
    _axes_ts[1].set_title(f"Control Inputs - {_selected}")
    _axes_ts[1].set_ylabel("Current")
    _axes_ts[1].set_xlabel("Time (s)")
    _axes_ts[1].grid(visible=True, linestyle="--", alpha=0.5)
    _axes_ts[1].spines[["top", "right"]].set_visible(False)

    _fig_ts
    return


if __name__ == "__main__":
    app.run()
