import marimo

__generated_with = "0.23.16"
app = marimo.App(
    width="wide",
    app_title="Optimal Stimulation Montage Explorer",
)


@app.cell
def _():

    import marimo as mo
    import matplotlib.pyplot as plt

    from neuro.connectome import Connectome
    from neuro.jansen_rit import JansenRitDynamics, JansenRitParams, lfp, simulate_network
    from neuro.stimulation.montage_optimization import (
        SelectedMontageStim,
        build_target_field,
        load_field_projection_matrix,
        sweep_sparse_montages,
    )

    return (
        Connectome,
        JansenRitDynamics,
        JansenRitParams,
        SelectedMontageStim,
        build_target_field,
        lfp,
        load_field_projection_matrix,
        mo,
        plt,
        simulate_network,
        sweep_sparse_montages,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # ⚡ Optimal Stimulation Montage Explorer

    Select a sparse transcranial electrical stimulation (tES) electrode montage ($2 \le k \le 5$ active electrodes)
    that targets hyperpolarizing current to the Epileptogenic Zone (EZ) while satisfying Kirchhoff's Current Law (KCL).

    $$\min_{\mathbf{u} \in \mathbb{R}^P} \quad \frac{1}{2} \Vert{}\mathbf{L}_\mathrm{stim} \mathbf{u} - \mathbf{s}^*\Vert{}_2^2 + \lambda \Vert{}\mathbf{u}\Vert{}_1 \quad \text{subject to} \quad \sum_{p=1}^P u_p = 0, \quad \vert{}u_p\vert{} \le I_\mathrm{max}$$

    1. **$\ell_1$-Regularized Screening**: Sweeps $\lambda$ using `osqp` to identify distinct sparse active sets.
    2. **Unregularized Refitting**: Re-optimizes the QP on the active subset without $\ell_1$ penalty to remove shrinkage bias and restore full current authority.
    """)
    return


@app.cell
def _(Connectome, load_field_projection_matrix):
    connectome = Connectome.from_config({})
    l_stim, channel_labels, is_fallback, model_desc = load_field_projection_matrix(
        "data/roast_field_projection_3d.npz",
        connectome,
    )
    return channel_labels, connectome, is_fallback, l_stim, model_desc


@app.cell
def _(is_fallback, mo, model_desc):
    _status_badge = "⚠️ **Development Fallback Active**" if is_fallback else "✅ **High-Fidelity ROAST FEM Loaded**"
    mo.callout(
        mo.md(f"""
        {_status_badge}
        - **Forward Model**: {model_desc}
        """),
        kind="warn" if is_fallback else "success",
    )
    return


@app.cell
def _(mo):
    ez_checkbox = mo.ui.checkbox(value=True, label="Target EZ (lHC, lPHC, lAMYG)")
    pz_checkbox = mo.ui.checkbox(value=False, label="Include PZ (lTCI, lTCV)")
    s0_slider = mo.ui.slider(0.1, 2.5, step=0.1, value=0.8, label="Target Shift s_0 (mV)")
    imax_slider = mo.ui.slider(0.5, 5.0, step=0.5, value=2.0, label="Max Electrode Current I_max (mA)")
    min_k_slider = mo.ui.slider(2, 4, step=1, value=2, label="Min Electrodes")
    max_k_slider = mo.ui.slider(3, 6, step=1, value=5, label="Max Electrodes")
    n_pts_slider = mo.ui.slider(10, 40, step=5, value=25, label="Lambda Sweep Points")

    mo.hstack(
        [
            mo.vstack([mo.md("### 🎯 Targeting"), ez_checkbox, pz_checkbox, s0_slider]),
            mo.vstack([mo.md("### ⚙️ Constraints & Budget"), imax_slider, min_k_slider, max_k_slider]),
            mo.vstack([mo.md("### 🔍 Sweep Resolution"), n_pts_slider]),
        ],
        justify="space-between",
        gap=4,
    )
    return (
        ez_checkbox,
        imax_slider,
        max_k_slider,
        min_k_slider,
        n_pts_slider,
        pz_checkbox,
        s0_slider,
    )


@app.cell
def _(
    build_target_field,
    channel_labels,
    connectome,
    ez_checkbox,
    imax_slider,
    l_stim,
    max_k_slider,
    min_k_slider,
    n_pts_slider,
    pz_checkbox,
    s0_slider,
    sweep_sparse_montages,
):
    _target_regions: list[str] = []
    if ez_checkbox.value:
        _target_regions.extend(["lHC", "lPHC", "lAMYG"])
    if pz_checkbox.value:
        _target_regions.extend(["lTCI", "lTCV"])

    s_target = build_target_field(connectome, _target_regions, s0=s0_slider.value)

    candidates = sweep_sparse_montages(
        l_stim,
        s_target,
        channel_labels=channel_labels,
        min_electrodes=min_k_slider.value,
        max_electrodes=max_k_slider.value,
        i_max=imax_slider.value,
        n_lambdas=n_pts_slider.value,
    )
    return (candidates,)


@app.cell
def _(candidates, mo, plt):
    if not candidates:
        mo.md("⚠️ No candidates found within the specified electrode budget. Try broadening the budget or sweep range.")
        _fig = None
    else:
        _ks = [c.n_active for c in candidates]
        _mse_l1 = [c.mse_l1 for c in candidates]
        _mse_refit = [c.mse_refit for c in candidates]
        _lams = [c.lam for c in candidates]

        _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 4))

        _ax1.scatter(_lams, _ks, color="royalblue", s=60, edgecolors="k", zorder=3)
        _ax1.set_xscale("log")
        _ax1.set_xlabel(r"Regularization Parameter $\lambda$")
        _ax1.set_ylabel("Active Electrodes ($k$)")
        _ax1.set_title("Sparsity vs. Regularization")
        _ax1.grid(visible=True, linestyle="--", alpha=0.5)

        _ax2.plot(_ks, _mse_l1, "o--", label="L1 (with shrinkage)", color="gray", alpha=0.7)
        _ax2.plot(_ks, _mse_refit, "s-", label="Refit QP (debiased)", color="forestgreen", lw=2)
        _ax2.set_xlabel("Active Electrodes ($k$)")
        _ax2.set_ylabel("Targeting MSE")
        _ax2.set_title("Targeting Error vs. Electrode Budget")
        _ax2.legend()
        _ax2.grid(visible=True, linestyle="--", alpha=0.5)

        _fig.tight_layout()

    _fig
    return


@app.cell
def _(candidates, mo):
    if not candidates:
        montage_table = None
    else:
        _table_data = []
        for idx, _cand in enumerate(candidates):
            _table_data.append(
                {
                    "ID": idx,
                    "Active Channels": ", ".join(_cand.active_labels),
                    "Count": _cand.n_active,
                    "Refit MSE": f"{_cand.mse_refit:.5f}",
                    "L1 MSE": f"{_cand.mse_l1:.5f}",
                    "Max |u| (mA)": f"{float(abs(_cand.u_refit).max()):.2f}",
                    "KCL Residual": f"{_cand.kcl_residual:.1e}",
                }
            )
        montage_table = mo.ui.table(_table_data, selection="single", page_size=10)

    mo.vstack(
        [
            mo.md("### 📋 Discovered Sparse Montages (Select one to inspect)"),
            montage_table if montage_table is not None else mo.md("_No candidates available_"),
        ]
    )
    return (montage_table,)


@app.cell
def _(candidates, connectome, montage_table, plt):
    if not candidates or montage_table is None:
        _fig_detail = None
    else:
        _selected_idx = 0
        if montage_table.value and len(montage_table.value) > 0:
            _selected_idx = int(montage_table.value[0]["ID"])

        _cand = candidates[_selected_idx]

        _fig_detail, (_ax_u, _ax_s) = plt.subplots(1, 2, figsize=(13, 5))

        # 1. Currents bar chart
        _colors = ["firebrick" if u > 0 else "royalblue" for u in _cand.u_refit]
        _ax_u.bar(_cand.active_labels, _cand.u_refit, color=_colors, edgecolor="black")
        _ax_u.axhline(0, color="k", linewidth=0.8, linestyle="--")
        _ax_u.set_ylabel("Current u (mA)")
        _ax_u.set_title(
            f"Montage #{_selected_idx}: {_cand.n_active} Electrodes (KCL $\\sum u = {_cand.kcl_residual:.1e}$)"
        )
        _ax_u.grid(axis="y", linestyle="--", alpha=0.5)

        # 2. Regional Drive Profile
        _ez_set = {"lHC", "lPHC", "lAMYG"}
        _pz_set = {"lTCI", "lTCV"}
        _bar_colors = []
        for reg in connectome.region_labels:
            if reg in _ez_set:
                _bar_colors.append("dodgerblue")
            elif reg in _pz_set:
                _bar_colors.append("gold")
            elif _cand.s_refit[connectome.region_index[reg]] > 0.05:
                _bar_colors.append("salmon")
            else:
                _bar_colors.append("lightgray")

        _ax_s.bar(range(len(connectome.region_labels)), _cand.s_refit, color=_bar_colors)
        _ax_s.axhline(0, color="k", linewidth=0.8, linestyle="--")
        _ax_s.set_xlabel("Region Index (76 TVB Regions)")
        _ax_s.set_ylabel("Somatic Drive s (mV)")
        _ax_s.set_title("Regional Somatic Drive Profile (Blue=EZ, Gold=PZ, Red=Anodal)")
        _ax_s.grid(axis="y", linestyle="--", alpha=0.5)

        _fig_detail.tight_layout()

    _fig_detail
    return


@app.cell
def _(mo):
    sim_button = mo.ui.button(label="🚀 Run 4s Jansen-Rit Dynamic Simulation with Selected Montage")
    mo.vstack(
        [
            mo.md("### 🔬 Dynamic Plant Verification"),
            mo.md(
                "Simulate the whole-brain Jansen-Rit network under uncontrolled vs. constant cathodal burst using the selected montage."
            ),
            sim_button,
        ]
    )
    return (sim_button,)


@app.cell
def _(
    JansenRitDynamics,
    JansenRitParams,
    SelectedMontageStim,
    candidates,
    connectome,
    l_stim,
    lfp,
    montage_table,
    plt,
    sim_button,
    simulate_network,
):
    if not sim_button.value or not candidates or montage_table is None:
        _fig_sim = None
    else:
        _selected_idx = 0
        if montage_table.value and len(montage_table.value) > 0:
            _selected_idx = int(montage_table.value[0]["ID"])
        _cand = candidates[_selected_idx]

        _dt = 0.001
        _t_end = 3.0

        _params = JansenRitParams.from_config({"A": "seizure"})
        _dyn_uncontrolled = JansenRitDynamics(dt=_dt, params=_params, conn=connectome, seed=42)
        _t_vec, _x_uncontrolled = simulate_network(dyn=_dyn_uncontrolled, duration=_t_end)
        _lfp_uncontrolled = lfp(_x_uncontrolled)

        # Drive with constant selected montage
        _stim_model = SelectedMontageStim(l_stim[:, _cand.active_indices], _cand.active_labels)
        _dyn_controlled = JansenRitDynamics(dt=_dt, params=_params, conn=connectome, stim=_stim_model, seed=42)
        _, _x_controlled = simulate_network(
            dyn=_dyn_controlled,
            duration=_t_end,
            control_current=_cand.u_refit,
            stim_window=(0.0, _t_end),
        )
        _lfp_controlled = lfp(_x_controlled)

        _ez_idx = connectome.region_index["lHC"]

        _fig_sim, _ax = plt.subplots(figsize=(10, 4))
        _ax.plot(
            _t_vec,
            _lfp_uncontrolled[_ez_idx],
            label="Uncontrolled Seizure",
            color="crimson",
            alpha=0.7,
        )
        _ax.plot(
            _t_vec,
            _lfp_controlled[_ez_idx],
            label=f"Controlled ({_cand.n_active}-electrode montage)",
            color="teal",
            lw=1.5,
        )
        _ax.set_xlabel("Time (s)")
        _ax.set_ylabel("lHC LFP (mV)")
        _ax.set_title(f"Seizure Suppression Verification: lHC LFP under Montage #{_selected_idx}")
        _ax.legend(loc="upper right")
        _ax.grid(visible=True, linestyle="--", alpha=0.5)
        _fig_sim.tight_layout()

    _fig_sim
    return


if __name__ == "__main__":
    app.run()
