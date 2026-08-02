import marimo

__generated_with = "0.23.10"
app = marimo.App(
    width="medium",
    app_title="Stage 2 Network: Jansen-Rit Whole-Brain",
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
def _(mo):
    """Marimo cell."""
    mo.md("""
    # 🧠 Stage 2 — Network: Instantaneous Coupling, then Delays

    This notebook explores the coupled whole-brain network of Jansen-Rit neural masses.
    We couple $N$ nodes using the structural connectivity weights and delays from **The Virtual Brain**.

    ### Key Objectives
    1. **Seizure Propagation:** Set higher gains $A$ in the Epileptogenic Zone (EZ) and Propagation Zone (PZ) to observe seizure recruitment of healthy nodes.
    2. **Network Dynamics:** Study how coupling strength $K$, conduction speed, and delays alter seizure propagation timing and spread.
    3. **Coupling Correctness:** Isolate a node by zeroing its incoming weights to prove that recruitment is network-driven (it stays quiescent while neighbors seize).
    """)
    return


@app.cell
def _(Connectome):
    """Marimo cell."""
    connectome = Connectome.from_config({})
    region_options = ["None", *list(connectome.region_labels)]
    return connectome, region_options


@app.cell
def _(mo, region_options):
    """Marimo cell."""
    k_slider = mo.ui.slider(0.0, 2.0, 0.05, value=0.54, label="Global Coupling Strength K")
    speed_slider = mo.ui.slider(5.0, 100.0, 5.0, value=50.0, label="Conduction speed (mm/ms)")
    duration_slider = mo.ui.slider(1.0, 10.0, 0.5, value=4.0, label="Simulation duration (s)")
    deterministic_toggle = mo.ui.checkbox(value=False, label="Deterministic (RK4, no noise)")
    use_delays_toggle = mo.ui.checkbox(value=True, label="Use conduction delays")
    isolated_node_dropdown = mo.ui.dropdown(
        options=region_options, value="None", label="Isolate a region (zero incoming weights)"
    )
    seed_slider = mo.ui.slider(0, 100, 1, value=69, label="RNG Seed")

    mo.hstack(
        [
            mo.vstack(
                [
                    k_slider,
                    speed_slider,
                    duration_slider,
                ]
            ),
            mo.vstack(
                [
                    deterministic_toggle,
                    use_delays_toggle,
                    isolated_node_dropdown,
                    seed_slider,
                ]
            ),
        ],
        justify="space-between",
    )
    return (
        deterministic_toggle,
        duration_slider,
        isolated_node_dropdown,
        k_slider,
        seed_slider,
        speed_slider,
        use_delays_toggle,
    )


@app.cell
def _(
    JansenRitDynamics,
    JansenRitParams,
    conn,
    connectome,
    deterministic_toggle,
    duration_slider,
    isolated_node_dropdown,
    k_slider,
    lfp,
    np,
    replace,
    seed_slider,
    simulate_network,
    speed_slider,
    use_delays_toggle,
):
    """Marimo cell."""
    n_nodes = len(connectome.region_labels)

    conn_scaled = replace(connectome, speed=speed_slider.value, delays=connectome.tract_lengths / speed_slider.value)

    if isolated_node_dropdown.value != "None":
        isolated_idx = connectome.region_index[isolated_node_dropdown.value]
        custom_weights = conn_scaled.weights.copy()
        custom_weights[isolated_idx, :] = 0.0
        conn_scaled = replace(conn_scaled, weights=custom_weights)

    ez_names = ("lHC", "lPHC", "lAMYG")
    pz_names = ("lTCI", "lTCV")
    ez_idxs = [connectome.region_index[name] for name in ez_names]
    pz_idxs = [connectome.region_index[name] for name in pz_names]

    a_gains = np.full(n_nodes, 3.25)
    a_gains[ez_idxs] = 3.6
    a_gains[pz_idxs] = 3.4
    sigma = 0.0 if deterministic_toggle.value else JansenRitParams().sigma

    if not use_delays_toggle.value:
        conn_scaled = replace(conn_scaled, delays=np.zeros_like(conn_scaled.delays))

    params = JansenRitParams.from_config({"A": a_gains, "sigma": sigma})
    dyn = JansenRitDynamics(dt=0.0001, params=params, conn=replace(conn, K=k_slider.value), seed=int(seed_slider.value))
    (t, x_traj) = simulate_network(dyn=dyn, duration=float(duration_slider.value))
    y = lfp(x_traj)
    return ez_idxs, pz_idxs, t, y


@app.cell
def _(connectome, ez_idxs, mo, np, pz_idxs, t, y):
    """Marimo cell."""
    y_steady = y[:, int(1.0 / (t[1] - t[0])) :]
    ptps = np.ptp(y_steady, axis=1)

    seizure_threshold = 5.0
    recruited_idxs = np.where(ptps > seizure_threshold)[0]
    n_recruited = len(recruited_idxs)

    def _onset_time(signal, times, threshold=seizure_threshold):
        """_onset_time definition."""
        idx = np.where(np.abs(signal) > threshold)[0]
        if len(idx) == 0:
            return "quiescent"
        return f"{times[idx[0]]:.3f} s"

    onset_summaries = []
    for name, idxs in (("Epileptogenic Zone (EZ)", ez_idxs), ("Propagation Zone (PZ)", pz_idxs)):
        details = []
        for i in idxs:
            lbl = connectome.region_labels[i]
            on = _onset_time(y[i], t)
            details.append(f"{lbl}: {on}")
        onset_summaries.append(f"**{name}**: {', '.join(details)}")

    mo.md(f"""
    ### 📊 Simulation Metrics
    * **Recruited Regions (PTP > 5.0):** {n_recruited} of {len(connectome.region_labels)} regions
    * {onset_summaries[0]}
    * {onset_summaries[1]}
    """)
    return


@app.cell
def _(connectome, plt, t, y):
    """Marimo cell."""
    fig = plt.figure(figsize=(12, 8), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1])

    ax_heat = fig.add_subplot(gs[:, 0])
    im = ax_heat.imshow(
        y,
        cmap="RdBu_r",
        aspect="auto",
        extent=[t[0], t[-1], 0, len(connectome.region_labels)],
        vmin=-10,
        vmax=10,
    )
    ax_heat.set_title("Spatiotemporal Raster (N-node membrane potentials y)")
    ax_heat.set_xlabel("time (s)")
    ax_heat.set_ylabel("brain region index")
    fig.colorbar(im, ax=ax_heat, label="y = x2 - x3 (mV)")

    ax_trace_ez = fig.add_subplot(gs[0, 1])
    lhc_idx = connectome.region_index["lHC"]
    ax_trace_ez.plot(t, y[lhc_idx], color="#d62728", label="lHC (EZ)", linewidth=0.8)
    ax_trace_ez.set_title("Representative Traces")
    ax_trace_ez.set_ylabel("y (mV)")
    ax_trace_ez.legend(loc="upper right")
    ax_trace_ez.spines[["top", "right"]].set_visible(False)
    ax_trace_ez.grid(visible=True, linestyle="--", alpha=0.3)

    ax_trace_pz = fig.add_subplot(gs[1, 1])
    ltci_idx = connectome.region_index["lTCI"]
    rhc_idx = connectome.region_index["rHC"]
    ax_trace_pz.plot(t, y[ltci_idx], color="#ff7f0e", label="lTCI (PZ)", linewidth=0.8)
    ax_trace_pz.plot(t, y[rhc_idx], color="#1f77b4", label="rHC (Healthy)", linewidth=0.8)
    ax_trace_pz.set_xlabel("time (s)")
    ax_trace_pz.set_ylabel("y (mV)")
    ax_trace_pz.legend(loc="upper right")
    ax_trace_pz.spines[["top", "right"]].set_visible(False)
    ax_trace_pz.grid(visible=True, linestyle="--", alpha=0.3)

    fig
    return


if __name__ == "__main__":
    app.run()
