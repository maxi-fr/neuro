import marimo

__generated_with = "0.23.10"
app = marimo.App(
    width="medium",
    app_title="Interactive Simulation & EEG Plotter",
)


@app.cell
def _():
    """Marimo cell."""
    from dataclasses import replace

    import marimo as mo
    import numpy as np
    from matplotlib import pyplot as plt

    from neuro.connectome import Connectome
    from neuro.jansen_rit import JansenRitParams, lfp, simulate_network

    return (
        Connectome,
        JansenRitParams,
        lfp,
        mo,
        np,
        plt,
        replace,
        simulate_network,
    )


@app.cell
def _(Connectome):
    """Marimo cell."""
    connectome = Connectome.from_config({})
    return (connectome,)


@app.cell
def _(connectome):
    """Marimo cell."""
    ez_names = ("lHC", "lPHC", "lAMYG")
    pz_names = ("lTCI", "lTCV")

    def get_region_display_name(name: str) -> str:
        """get_region_display_name definition."""
        if name in ez_names:
            return f"{name} (EZ)"
        if name in pz_names:
            return f"{name} (PZ)"
        return f"{name} (Healthy)"

    sorted_regions = sorted(connectome.region_labels)
    region_options = {get_region_display_name(name): name for name in sorted_regions}
    channel_options = sorted(map(str, connectome.channel_labels))

    default_regions = [
        get_region_display_name("lHC"),
        get_region_display_name("lTCI"),
        get_region_display_name("rHC"),
    ]
    default_channels = ["F3", "P3", "CP5"]
    return (
        channel_options,
        default_channels,
        default_regions,
        ez_names,
        pz_names,
        region_options,
    )


@app.cell
def _(mo):
    """Marimo cell."""
    mo.md(r"""
    # 🧠 Interactive Whole-Brain Jansen-Rit Simulator & Plotter

    Configure network parameters, run the Jansen-Rit mass model simulation, and interactively select which brain regions and EEG channels to plot.
    """)
    return


@app.cell
def _(mo):
    """Marimo cell."""
    k_slider = mo.ui.slider(0.0, 2.0, 0.05, value=0.54, label="Global Coupling Strength K")
    speed_slider = mo.ui.slider(5.0, 100.0, 5.0, value=50.0, label="Conduction speed (mm/ms)")
    duration_slider = mo.ui.slider(1.0, 10.0, 0.5, value=4.0, label="Simulation duration (s)")
    deterministic_toggle = mo.ui.checkbox(value=False, label="Deterministic (RK4, no noise)")
    seed_slider = mo.ui.slider(0, 100, 1, value=69, label="RNG Seed")

    mo.hstack(
        [
            mo.vstack([mo.md("### 🧠 Coupling & Speed"), k_slider, speed_slider]),
            mo.vstack([mo.md("### ⏱️ Time & Noise"), duration_slider, deterministic_toggle, seed_slider]),
        ],
        justify="space-between",
        gap=4,
    )
    return (
        deterministic_toggle,
        duration_slider,
        k_slider,
        seed_slider,
        speed_slider,
    )


@app.cell
def _(
    JansenRitDynamics,
    JansenRitParams,
    conn,
    connectome,
    deterministic_toggle,
    duration_slider,
    ez_names,
    k_slider,
    lfp,
    np,
    pz_names,
    replace,
    seed_slider,
    simulate_network,
    speed_slider,
):
    """Marimo cell."""
    _conn = replace(connectome, speed=speed_slider.value, delays=connectome.tract_lengths / speed_slider.value)

    _n_nodes = len(connectome.region_labels)
    _a_gains = np.full(_n_nodes, 3.25)

    ez_idxs = [connectome.region_index[name] for name in ez_names]
    pz_idxs = [connectome.region_index[name] for name in pz_names]
    _a_gains[ez_idxs] = 3.6
    _a_gains[pz_idxs] = 3.4

    _noise_sigma = 0.0 if deterministic_toggle.value else JansenRitParams().sigma

    _params = JansenRitParams.from_config({"A": _a_gains, "sigma": _noise_sigma})
    dyn__params = JansenRitDynamics(
        dt=0.0001, params=_params, conn=replace(conn, K=k_slider.value), seed=int(seed_slider.value)
    )
    (t, _x_traj) = simulate_network(dyn=dyn__params, duration=float(duration_slider.value))
    y = lfp(_x_traj)

    eeg = _conn.gain @ y
    return eeg, t, y


@app.cell
def _(mo):
    """Marimo cell."""
    mo.md("""
    ## 📊 Plot Configuration
    """)
    return


@app.cell
def _(channel_options, default_channels, default_regions, mo, region_options):
    """Marimo cell."""
    regions_multiselect = mo.ui.multiselect(
        options=region_options,
        value=default_regions,
        label="Select Brain Regions to Plot (Node Activity):",
    )
    eeg_multiselect = mo.ui.multiselect(
        options=channel_options,
        value=default_channels,
        label="Select EEG Channels to Plot:",
    )
    stacked_toggle = mo.ui.checkbox(value=True, label="Stack signals vertically (waterfall style)")
    offset_scale_slider = mo.ui.slider(0.5, 3.0, 0.1, value=1.5, label="Waterfall Offset Scale")

    mo.hstack(
        [
            mo.vstack([regions_multiselect, eeg_multiselect]),
            mo.vstack([stacked_toggle, offset_scale_slider]),
        ],
        justify="space-between",
        gap=4,
    )
    return (
        eeg_multiselect,
        offset_scale_slider,
        regions_multiselect,
        stacked_toggle,
    )


@app.cell
def _(
    connectome,
    mo,
    np,
    offset_scale_slider,
    plt,
    regions_multiselect,
    stacked_toggle,
    t,
    y,
):
    """Marimo cell."""
    if not regions_multiselect.value:
        fig_node_out = mo.md("⚠️ *Select at least one brain region to plot.*")
    else:
        _selected_regions = regions_multiselect.value
        _num_to_plot = len(_selected_regions)
        _time_sec = t

        fig_node_out, _ax = plt.subplots(
            figsize=(10, max(4.0, _num_to_plot * 0.6 if stacked_toggle.value else 5.0)),
            layout="constrained",
        )

        _cmap = plt.get_cmap("tab10")
        _colors = [_cmap(_i % 10) for _i in range(_num_to_plot)]

        if stacked_toggle.value:
            _selected_sigs = y[[connectome.region_index[_r] for _r in _selected_regions], :]
            _p2p = np.ptp(_selected_sigs, axis=1)
            _mean_p2p = np.mean(_p2p) if np.mean(_p2p) > 0 else 1.0
            _offset = _mean_p2p * offset_scale_slider.value

            _yticks = []
            for _idx, _r_name in enumerate(_selected_regions):
                _r_idx = connectome.region_index[_r_name]
                _sig = y[_r_idx, :]
                _y_offset = (_num_to_plot - 1 - _idx) * _offset
                _col = _colors[_idx]
                _ax.plot(_time_sec, _sig + _y_offset, color=_col, alpha=0.85, linewidth=1.2)
                _yticks.append(_y_offset)

            _ax.set_yticks(_yticks)
            _ax.set_yticklabels(
                [
                    f"{_r} (EZ)"
                    if _r in ("lHC", "lPHC", "lAMYG")
                    else (f"{_r} (PZ)" if _r in ("lTCI", "lTCV") else f"{_r} (Healthy)")
                    for _r in _selected_regions
                ]
            )
            _ax.set_ylim(-_offset, _num_to_plot * _offset)
        else:
            for _idx, _r_name in enumerate(_selected_regions):
                _r_idx = connectome.region_index[_r_name]
                _sig = y[_r_idx, :]
                _col = _colors[_idx]
                _display_lbl = (
                    f"{_r_name} (EZ)"
                    if _r_name in ("lHC", "lPHC", "lAMYG")
                    else (f"{_r_name} (PZ)" if _r_name in ("lTCI", "lTCV") else f"{_r_name} (Healthy)")
                )
                _ax.plot(_time_sec, _sig, color=_col, alpha=0.54, linewidth=1.2, label=_display_lbl)
            _ax.legend(loc="upper right", framealpha=0.9)

        _ax.set_xlabel("Time (s)", fontsize=11, fontweight="bold")
        _ax.set_ylabel("Amplitude (mV)", fontsize=11, fontweight="bold")
        _ax.set_title("Brain Region Node Activity ($y_i$)", fontsize=13, fontweight="bold", pad=12)
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)
        _ax.grid(visible=True, linestyle="--", alpha=0.3)
        plt.close(fig_node_out)

    fig_node_out
    return (fig_node_out,)


@app.cell
def _(
    connectome,
    eeg,
    eeg_multiselect,
    mo,
    np,
    offset_scale_slider,
    plt,
    stacked_toggle,
    t,
):
    """Marimo cell."""
    if not eeg_multiselect.value:
        fig_eeg_out = mo.md("⚠️ *Select at least one EEG channel to plot.*")
    else:
        _selected_channels = eeg_multiselect.value
        _num_to_plot = len(_selected_channels)
        _time_sec = t

        fig_eeg_out, _ax = plt.subplots(
            figsize=(10, max(4.0, _num_to_plot * 0.6 if stacked_toggle.value else 5.0)),
            layout="constrained",
        )

        _cmap = plt.get_cmap("Set2")
        _colors = [_cmap(_i % 8) for _i in range(_num_to_plot)]

        if stacked_toggle.value:
            _selected_sigs = eeg[[connectome.channel_index[_c] for _c in _selected_channels], :]
            _p2p = np.ptp(_selected_sigs, axis=1)
            _mean_p2p = np.mean(_p2p) if np.mean(_p2p) > 0 else 1.0
            _offset = _mean_p2p * offset_scale_slider.value

            _yticks = []
            for _idx, _ch_name in enumerate(_selected_channels):
                _ch_idx = connectome.channel_index[_ch_name]
                _sig = eeg[_ch_idx, :]
                _y_offset = (_num_to_plot - 1 - _idx) * _offset
                _col = _colors[_idx]
                _ax.plot(_time_sec, _sig + _y_offset, color=_col, alpha=0.85, linewidth=1.2)
                _yticks.append(_y_offset)

            _ax.set_yticks(_yticks)
            _ax.set_yticklabels(_selected_channels)
            _ax.set_ylim(-_offset, _num_to_plot * _offset)
        else:
            for _idx, _ch_name in enumerate(_selected_channels):
                _ch_idx = connectome.channel_index[_ch_name]
                _sig = eeg[_ch_idx, :]
                _col = _colors[_idx]
                _ax.plot(_time_sec, _sig, color=_col, alpha=0.54, linewidth=1.2, label=_ch_name)
            _ax.legend(loc="upper right", framealpha=0.9)

        _ax.set_xlabel("Time (s)", fontsize=11, fontweight="bold")
        _ax.set_ylabel("EEG Potential (a.u.)", fontsize=11, fontweight="bold")
        _ax.set_title("Projected EEG Signals", fontsize=13, fontweight="bold", pad=12)
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)
        _ax.grid(visible=True, linestyle="--", alpha=0.3)
        plt.close(fig_eeg_out)

    fig_eeg_out
    return (fig_eeg_out,)


@app.cell
def _(mo):
    """Marimo cell."""
    mo.md("""
    ## 💾 Export Options
    """)
    return


@app.cell
def _(mo):
    """Marimo cell."""
    save_dir_input = mo.ui.text(value="artifacts", label="Save Directory:")
    save_button = mo.ui.button(label="💾 Save Plots to Disk", tooltip="Saves both plots as PNG files")

    mo.hstack([save_dir_input, save_button], gap=2)
    return save_button, save_dir_input


@app.cell
def _(fig_eeg_out, fig_node_out, mo, save_button, save_dir_input):
    """Marimo cell."""
    mo.stop(not save_button.value)

    from pathlib import Path

    _out_dir = Path(__file__).parent.parent / save_dir_input.value
    _out_dir.mkdir(parents=True, exist_ok=True)

    _activity_path = _out_dir / "activity_plot.png"
    _eeg_path = _out_dir / "eeg_plot.png"

    _success_msg = ""
    try:
        if hasattr(fig_node_out, "savefig"):
            fig_node_out.savefig(_activity_path, dpi=200, bbox_inches="tight")
            _success_msg += (
                f"* **Node Activity Plot**: [activity_plot.png](file:///{_activity_path.resolve().as_posix()})\n"
            )
        if hasattr(fig_eeg_out, "savefig"):
            fig_eeg_out.savefig(_eeg_path, dpi=200, bbox_inches="tight")
            _success_msg += f"* **EEG Signals Plot**: [eeg_plot.png](file:///{_eeg_path.resolve().as_posix()})\n"

        if _success_msg:
            _ = mo.md(f"✅ **Plots saved successfully to `{_out_dir}`!**\n\n" + _success_msg)
        else:
            _ = mo.md("⚠️ *No valid plots selected to save.*")
    except Exception as e:  # noqa: BLE001
        _ = mo.md(f"❌ **Error saving plots:** {e!s}")
    return


if __name__ == "__main__":
    app.run()
