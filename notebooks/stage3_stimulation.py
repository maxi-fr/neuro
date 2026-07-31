import marimo

__generated_with = "0.23.13"
app = marimo.App(
    width="medium",
    app_title="Stage 3 Stimulation: Immediate tES",
)


@app.cell
def _():
    from dataclasses import replace

    import marimo as mo
    import numpy as np
    from matplotlib import pyplot as plt

    from neuro.connectome import EXTRACEPHALIC_ELECTRODES_MM, Connectome
    from neuro.connectome import _compute_gamma as compute_gamma
    from neuro.jansen_rit import JansenRitDynamics, JansenRitParams, lfp, simulate_network

    return (
        Connectome,
        EXTRACEPHALIC_ELECTRODES_MM,
        JansenRitDynamics,
        JansenRitParams,
        compute_gamma,
        lfp,
        mo,
        np,
        plt,
        replace,
        simulate_network,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # 🧠 Stage 3 — Immediate tES via the $\gamma$ Vector
    Explore the immediate suppressive effects of Transcranial Electrical Stimulation (tES) on epileptic seizure propagation in a whole-brain Jansen-Rit model.

    ### Introduction & Stimulation Model

    Transcranial Electrical Stimulation (tES) is modeled as an electrical polarization $U_{\text{tES}}$ entering the firing-rate sigmoid function of the pyramidal populations:

    $$\dot{x}_{i,4} = A_i a \cdot S(x_{i,2} - x_{i,3} + U_{\text{tES}, i}) - 2a x_{i,4} - a^2 x_{i,1}$$

    The regional perturbation is calculated as:

    $$U_{\text{tES}, i}(t) = u(t) \cdot \gamma_i$$

    where:
    * $u(t)$ is the time-varying stimulation current (equal to $\hat{u}_{\text{tES}}$ during the stimulation window, else $0$). Its **sign sets the polarity**: $u < 0$ is a cathode (inhibitory), $u > 0$ an anode (excitatory).
    * $\gamma_i \ge 0$ is the spatial projection factor from the stimulating electrode (e.g., **TP9**) to region $i$: the Coulomb potential $100/\sqrt{r_i^2 + \sigma_\text{stim}^2}$ of a point source in a homogeneous head, so a montage's fields simply superpose. The same kernel serves cathodes and anodes; the current's sign decides which.
    * **EX_NECK** is a virtual extracephalic return, 145+ mm from every region. Kirchhoff's law forces the injected current back out somewhere, and wherever it leaves the drive is anodal; an off-head return makes that contribution small and near-uniform, so no region is driven anodally.

    In this stage, **we assume no synaptic plasticity ($Z$ dynamics are off)**, so the stimulation only exerts an immediate, transient effect. Once the stimulation turns off, the seizure is expected to re-expand across the network.
    """)
    return


@app.cell
def _(Connectome, EXTRACEPHALIC_ELECTRODES_MM):
    connectome = Connectome.from_config({})
    electrode_options = sorted(connectome.channel_labels) + sorted(EXTRACEPHALIC_ELECTRODES_MM)
    return connectome, electrode_options


@app.cell
def _(electrode_options, mo):
    # Interactive UI controls
    electrode_multiselect = mo.ui.multiselect(
        options=electrode_options,
        value=["TP9", "EX_NECK"],
        label="Stimulation montage (select ≥1; current sign sets polarity)",
    )
    spread_slider = mo.ui.slider(5.0, 100.0, 5.0, value=15.0, label="Coulomb softening radius σ_stim (mm)")
    u_intensity_slider = mo.ui.slider(
        -3.0, 3.0, 0.25, value=-1.5, label="Total tES Intensity u_hat_tES (negative = cathodic/inhibitory)"
    )

    onset_slider = mo.ui.slider(0.0, 10.0, 0.5, value=0.0, label="Stimulation Onset (s)")
    offset_slider = mo.ui.slider(0.0, 10.0, 0.5, value=2.5, label="Stimulation Offset (s)")

    k_slider = mo.ui.slider(0.0, 2.0, 0.05, value=0.54, label="Global Coupling K")
    speed_slider = mo.ui.slider(5.0, 100.0, 5.0, value=50.0, label="Conduction Speed (mm/ms)")
    duration_slider = mo.ui.slider(1.0, 50.0, 0.5, value=5.0, label="Simulation Duration (s)")
    seed_slider = mo.ui.slider(0, 100, 1, value=69, label="RNG Seed")
    deterministic_toggle = mo.ui.checkbox(value=False, label="Deterministic (RK4, no noise)")

    mo.hstack(
        [
            mo.vstack(
                [
                    mo.md("#### 🎛️ Stimulation Settings"),
                    mo.md(
                        "_The spatial kernel γ is positive; the **sign of each electrode's current sets its polarity** (cathode < 0, anode > 0). Per Kirchhoff's current law the montage currents must sum to zero, so cathodes need a return anode. Yu (2024) returns through the extracephalic Ex7/Ex8; the equivalent here is **EX_NECK**, which — unlike a scalp return — drives no region anodally. E.g. TP9 cathode with an EX_NECK return._"
                    ),
                    electrode_multiselect,
                    spread_slider,
                    u_intensity_slider,
                    onset_slider,
                    offset_slider,
                ]
            ),
            mo.vstack(
                [
                    mo.md("#### 🧠 Network & Integration Settings"),
                    k_slider,
                    speed_slider,
                    duration_slider,
                    seed_slider,
                    deterministic_toggle,
                ]
            ),
        ],
        justify="space-between",
        gap=4,
    )
    return (
        deterministic_toggle,
        duration_slider,
        electrode_multiselect,
        k_slider,
        offset_slider,
        onset_slider,
        seed_slider,
        speed_slider,
        spread_slider,
        u_intensity_slider,
    )


@app.cell
def _(
    JansenRitDynamics,
    JansenRitParams,
    compute_gamma,
    connectome,
    deterministic_toggle,
    duration_slider,
    electrode_multiselect,
    k_slider,
    lfp,
    np,
    offset_slider,
    onset_slider,
    replace,
    seed_slider,
    simulate_network,
    speed_slider,
    spread_slider,
    u_intensity_slider,
):
    # 1. Compute the gamma steering matrix (n_controls, 76) for the selected montage.
    selected_electrodes = list(electrode_multiselect.value) or ["CP5"]
    n_controls = len(selected_electrodes)
    gamma = compute_gamma(connectome.centres, target_electrode=selected_electrodes, spread=spread_slider.value)
    conn_scaled = replace(
        connectome,
        speed=speed_slider.value,
        delays=connectome.tract_lengths / speed_slider.value,
        K=k_slider.value,
        gamma=gamma,
    )

    # Kirchhoff: the intensity is split over the leading electrodes and returns through the
    # last one, so the montage sums to zero. Put EX_NECK last to keep every region cathodal.
    u_amp = np.full(n_controls, float(u_intensity_slider.value) / max(n_controls - 1, 1))
    if n_controls > 1:
        u_amp[-1] = -u_amp[:-1].sum()
    gamma_field = u_amp @ gamma  # combined per-node U_tES field for visualization

    # 2. Configure EZ and PZ gains
    n_nodes = len(connectome.region_labels)
    ez_names = ("lHC", "lPHC", "lAMYG")
    pz_names = ("lTCI", "lTCV")
    ez_idxs = [connectome.region_index[name] for name in ez_names]
    pz_idxs = [connectome.region_index[name] for name in pz_names]

    a_gains = np.full(n_nodes, 3.25)
    a_gains[ez_idxs] = 3.6
    a_gains[pz_idxs] = 3.4
    noise_sigma = 0.0 if deterministic_toggle.value else JansenRitParams().sigma
    # 3. Stimulation window
    stim_window = (float(onset_slider.value), float(offset_slider.value))

    # 4. Run the network with the tES current applied over the stim window.
    params = JansenRitParams.from_config({"A": a_gains, "sigma": noise_sigma})
    dyn = JansenRitDynamics(
        dt=0.0001,
        params=params,
        conn=conn_scaled,
        seed=int(seed_slider.value),
        enforce_zero_sum_current=False,  # a one-electrode selection stays monopolar (pedagogical)
    )
    (t, x_traj) = simulate_network(
        dyn=dyn,
        duration=float(duration_slider.value),
        u_hat_tES=u_amp,
        stim_window=stim_window,
    )
    y = lfp(x_traj)
    return conn_scaled, ez_idxs, gamma_field, pz_idxs, stim_window, t, y


@app.cell
def _(conn_scaled, gamma_field, np, plt):
    # Visualizing the top 15 regions by combined montage field |U_tES|
    sorted_idxs = np.argsort(np.abs(gamma_field))[::-1]
    top_idxs = sorted_idxs[:15]
    top_labels = conn_scaled.region_labels[top_idxs]
    top_gammas = gamma_field[top_idxs]

    fig_gamma, ax_g = plt.subplots(figsize=(6, 3.5), layout="constrained")
    colors = ["#f50057" if g < 0 else "#2979ff" for g in top_gammas]
    bars = ax_g.barh(top_labels[::-1], top_gammas[::-1], color=colors[::-1], height=0.6, alpha=0.85)
    ax_g.axvline(0.0, color="gray", linestyle="-", linewidth=0.8)
    ax_g.set_title("Combined Montage Field U_tES (Top 15 regions)", fontsize=11, fontweight="bold")
    ax_g.set_xlabel("U_tES = u·γ (combined)")
    ax_g.spines[["top", "right"]].set_visible(False)
    ax_g.grid(visible=True, axis="x", linestyle="--", alpha=0.3)

    # Add labels to bars
    for bar in bars:
        width = bar.get_width()
        ax_g.text(
            width - 0.05 if width < 0 else width + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}",
            va="center",
            ha="right" if width < 0 else "left",
            fontsize=8,
            fontweight="bold",
            color="black",
        )

    plt.close(fig_gamma)
    return (fig_gamma,)


@app.cell
def _(conn_scaled, ez_idxs, np, pz_idxs, stim_window, t, y):
    # Seizure suppression metrics
    seizure_threshold = 5.0
    dt = t[1] - t[0]
    win_len = int(0.2 / dt)  # 200 ms rolling window

    # Count how many regions are active (oscillating) over time
    n_samples = y.shape[1]
    active_regions = np.zeros(n_samples)
    for k in range(n_samples):
        start = max(0, k - win_len // 2)
        end = min(n_samples, k + win_len // 2)
        ptp_window = np.ptp(y[:, start:end], axis=1)
        active_regions[k] = np.sum(ptp_window > seizure_threshold)

    # Let's count how many healthy regions are recruited before, during, and after stimulation
    _t_onset, _t_offset = stim_window
    _healthy_idxs = [i for i in range(len(conn_scaled.region_labels)) if i not in ez_idxs and i not in pz_idxs]

    # Calculate PTP in different epochs
    during_ptps = np.zeros(len(_healthy_idxs))
    after_ptps = np.zeros(len(_healthy_idxs))

    # During: look at [_t_onset + 0.5, _t_offset - 0.1]
    if _t_offset - _t_onset > 1.0:
        idx_during = np.where((t >= _t_onset + 0.5) & (t < _t_offset - 0.1))[0]
        if len(idx_during) > 0:
            during_ptps = np.ptp(y[_healthy_idxs, :][:, idx_during], axis=1)

    # After: look at [_t_offset + 1.0, t[-1]]
    if t[-1] - _t_offset >= 1.5:
        idx_after = np.where(t >= _t_offset + 1.0)[0]
        if len(idx_after) > 0:
            after_ptps = np.ptp(y[_healthy_idxs, :][:, idx_after], axis=1)

    n_recruited_during = np.sum(during_ptps > seizure_threshold)
    n_recruited_after = np.sum(after_ptps > seizure_threshold)
    return active_regions, n_recruited_after, n_recruited_during


@app.cell
def _(
    conn_scaled,
    ez_idxs,
    mo,
    n_recruited_after,
    n_recruited_during,
    pz_idxs,
    stim_window,
):
    _t_onset, _t_offset = stim_window

    metric_md = f"""
    ### 📊 Simulation Metrics
    * **Stimulation Window:** {_t_onset:.1f} s to {_t_offset:.1f} s
    * **Recruited Healthy Regions (PTP > 5.0):**
      * **During Stimulation:** **{n_recruited_during}** of {len(conn_scaled.region_labels) - len(ez_idxs) - len(pz_idxs)} regions
      * **After Stimulation:** **{n_recruited_after}** of {len(conn_scaled.region_labels) - len(ez_idxs) - len(pz_idxs)} regions
    """

    # Verify success criteria
    suppressed = n_recruited_during == 0
    reexpanded = n_recruited_after > 0

    if suppressed and reexpanded:
        status_md = """
        <div style="background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; padding: 12px; border-radius: 5px; margin-top: 10px; font-family: sans-serif; font-size: 14px;">
            ✅ <strong>Stage 3 Verification Passed:</strong> Immediate suppression is active (0 healthy nodes recruited during stimulation) and seizure re-expansion is observed after stimulation offsets. This proves the transient nature of the immediate tES effect!
        </div>
        """
    elif suppressed:
        status_md = """
        <div style="background-color: #fff3cd; border: 1px solid #ffeeba; color: #856404; padding: 12px; border-radius: 5px; margin-top: 10px; font-family: sans-serif; font-size: 14px;">
            ⚠️ <strong>Partial Verification:</strong> Seizures are suppressed during stimulation, but did not re-expand afterward. Ensure the simulation duration is long enough after offset.
        </div>
        """
    else:
        status_md = """
        <div style="background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; padding: 12px; border-radius: 5px; margin-top: 10px; font-family: sans-serif; font-size: 14px;">
            ❌ <strong>Verification Failed:</strong> Stimulation intensity u_hat_tES or spatial spread is insufficient to suppress propagation to healthy regions. Increase intensity or adjust the spread.
        </div>
        """

    mo.md(metric_md + status_md)
    return


@app.cell
def _(active_regions, conn_scaled, plt, stim_window, t, y):
    # Plotting raster, traces, and count
    fig = plt.figure(figsize=(12, 9), layout="constrained")
    gs = fig.add_gridspec(3, 2, width_ratios=[1.8, 1], height_ratios=[1.5, 1, 0.8])

    # 1. Spatiotemporal raster (Heatmap)
    ax_raster = fig.add_subplot(gs[0:2, 0])
    im = ax_raster.imshow(
        y,
        cmap="RdBu_r",
        aspect="auto",
        extent=[t[0], t[-1], 0, len(conn_scaled.region_labels)],
        vmin=-10,
        vmax=10,
    )
    ax_raster.set_title("Whole-Brain Membrane Potentials (y) with tES", fontsize=12, fontweight="bold")
    ax_raster.set_ylabel("Brain Region Index")
    fig.colorbar(im, ax=ax_raster, label="y = x2 - x3 (mV)")

    # Highlight stimulation window
    _t_onset, _t_offset = stim_window
    ax_raster.axvspan(_t_onset, _t_offset, color="magenta", alpha=0.1, label="tES Active")
    ax_raster.legend(loc="upper right")

    # 2. Representative traces
    ax_trace_ez = fig.add_subplot(gs[0, 1])
    lhc_idx = conn_scaled.region_index["lHC"]
    ax_trace_ez.plot(t, y[lhc_idx], color="#d62728", label="lHC (EZ)", linewidth=0.8)
    ax_trace_ez.axvspan(_t_onset, _t_offset, color="magenta", alpha=0.08)
    ax_trace_ez.set_title("Representative Traces", fontsize=11, fontweight="bold")
    ax_trace_ez.set_ylabel("y (mV)")
    ax_trace_ez.legend(loc="upper right")
    ax_trace_ez.spines[["top", "right"]].set_visible(False)
    ax_trace_ez.grid(visible=True, linestyle="--", alpha=0.3)

    ax_trace_pz = fig.add_subplot(gs[1, 1])
    ltci_idx = conn_scaled.region_index["lTCI"]
    rhc_idx = conn_scaled.region_index["rHC"]
    ax_trace_pz.plot(t, y[ltci_idx], color="#ff7f0e", label="lTCI (PZ)", linewidth=0.8)
    ax_trace_pz.plot(t, y[rhc_idx], color="#1f77b4", label="rHC (Healthy)", linewidth=0.8)
    ax_trace_pz.axvspan(_t_onset, _t_offset, color="magenta", alpha=0.08)
    ax_trace_pz.set_ylabel("y (mV)")
    ax_trace_pz.legend(loc="upper right")
    ax_trace_pz.spines[["top", "right"]].set_visible(False)
    ax_trace_pz.grid(visible=True, linestyle="--", alpha=0.3)

    # 3. Seizure Count over time
    ax_count = fig.add_subplot(gs[2, 0])
    ax_count.plot(t, active_regions, color="#8e24aa", linewidth=1.5, label="Oscillating Regions (PTP > 5.0)")
    ax_count.axvspan(_t_onset, _t_offset, color="magenta", alpha=0.1)
    ax_count.set_title("Suppression Effect: Number of Active Seizure Regions", fontsize=11, fontweight="bold")
    ax_count.set_xlabel("Time (s)")
    ax_count.set_ylabel("Count")
    ax_count.set_ylim(-1, len(conn_scaled.region_labels) + 2)
    ax_count.spines[["top", "right"]].set_visible(False)
    ax_count.grid(visible=True, linestyle="--", alpha=0.3)

    # Empty right grid for balance
    ax_empty = fig.add_subplot(gs[2, 1])
    ax_empty.axis("off")

    plt.close(fig)
    return (fig,)


@app.cell
def _(dashboard):
    dashboard
    return


@app.cell
def _(fig, fig_gamma, mo):
    # Construct layout
    dashboard = mo.vstack(
        [
            mo.hstack(
                [
                    mo.vstack([mo.md("### 📊 Spatial Configuration"), fig_gamma]),
                    mo.md(" "),
                ],
                widths=[6, 6],
            ),
            mo.md("---"),
            mo.md("### 📊 Network Dynamics under Stimulation"),
            fig,
        ]
    )
    return (dashboard,)


if __name__ == "__main__":
    app.run()
