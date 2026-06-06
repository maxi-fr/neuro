import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium", app_title="Validation Suite")


@app.cell
def _():
    from dataclasses import replace

    import marimo as mo
    import numpy as np
    from matplotlib import pyplot as plt

    from neuro.connectome import load_connectome
    from neuro.jansen_rit import JansenRitParams, output, simulate_network, simulate_node
    from utils.processing import compute_psd

    return (
        JansenRitParams,
        compute_psd,
        load_connectome,
        mo,
        np,
        output,
        plt,
        replace,
        simulate_network,
        simulate_node,
    )


@app.cell
def _(mo):
    mo.md("""
    <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.15); margin-bottom: 20px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
        <h1 style="margin-top: 0; margin-bottom: 8px; font-weight: 600; font-size: 28px; letter-spacing: -0.5px;">🧠 Jansen-Rit Epilepsy Model: Validation Suite</h1>
        <p style="margin-bottom: 0; opacity: 0.9; font-size: 15px; line-height: 1.5;">
            An interactive validation dashboard for verifying model dynamics, single-node bifurcation, tract delays,
            and whole-brain network recruitment against standard TVB references.
        </p>
    </div>
    """)
    return


@app.cell
def _(mo):
    test_case_selector = mo.ui.dropdown(
        options={
            "Stage 0: Connectome Invariants & EEG Mapping": "stage0",
            "Stage 1: Single-Node Bifurcation & Seizure Rhythm": "stage1",
            "Stage 2: Healthy Control Quiescence": "stage2_healthy",
            "Stage 2: EZ/PZ Seizure Recruitment": "stage2_recruitment",
            "Stage 2: Coupling Correctness (Node Disconnect)": "stage2_isolation",
            "Stage 2: Delays vs. Instantaneous Propagation": "stage2_delays",
            "Stage 3: Immediate tES Suppression & Re-expansion": "stage3_stimulation",
        },
        value="stage1",
        label="Select Validation Case",
    )
    test_case_selector
    return (test_case_selector,)


@app.cell
def _(case, connectome, mo):
    region_options = ["None", *list(connectome.region_labels)]
    channel_options = sorted(connectome.channel_labels)

    # 1. Single Node params
    a_slider = mo.ui.slider(3.0, 4.0, 0.05, value=3.6, label="A (Synaptic Gain)")
    sigma_slider = mo.ui.slider(0.0, 1000.0, 50.0, value=550.0, label="Noise sigma")

    # 2. Network params
    k_slider = mo.ui.slider(0.0, 2.0, 0.05, value=0.75, label="Coupling Strength K")
    divisor_slider = mo.ui.slider(1.0, 3.0, 0.05, value=1.40, label="Weights Divisor")
    speed_slider = mo.ui.slider(5.0, 100.0, 5.0, value=50.0, label="Conduction Speed (mm/ms)")
    duration_slider = mo.ui.slider(1.0, 10.0, 0.5, value=4.0, label="Duration (s)")
    seed_slider = mo.ui.slider(0, 100, 1, value=42, label="RNG Seed")
    deterministic_toggle = mo.ui.checkbox(value=False, label="Deterministic (RK4)")

    # Isolate dropdown
    isolate_dropdown = mo.ui.dropdown(options=region_options, value="lTCI", label="Isolate Region")

    # 3. Stimulation params
    electrode_dropdown = mo.ui.dropdown(options=channel_options, value="CP5", label="Target Electrode (Cathode)")
    sigma_stim_slider = mo.ui.slider(5.0, 100.0, 5.0, value=20.0, label="tES Spread σ_stim (mm)")
    u_intensity_slider = mo.ui.slider(-3.0, 3.0, 0.25, value=1.5, label="tES Intensity u_hat_tES")
    onset_slider = mo.ui.slider(0.0, 10.0, 0.5, value=0.0, label="tES Onset (s)")
    offset_slider = mo.ui.slider(0.0, 10.0, 0.5, value=2.5, label="tES Offset (s)")

    # Render panels
    controls = None
    if case == "stage0":
        controls = mo.md("*No simulation parameters needed for Stage 0 invariants.*")
    elif case == "stage1":
        controls = mo.vstack(
            [
                mo.md("#### 🎛️ Single Node Settings"),
                a_slider,
                sigma_slider,
                deterministic_toggle,
                duration_slider,
                seed_slider,
            ]
        )
    elif case in ("stage2_healthy", "stage2_recruitment", "stage2_delays"):
        controls = mo.vstack(
            [
                mo.md("#### 🎛️ Network Settings"),
                k_slider,
                divisor_slider,
                speed_slider,
                duration_slider,
                deterministic_toggle,
                seed_slider,
            ]
        )
    elif case == "stage2_isolation":
        controls = mo.vstack(
            [
                mo.md("#### 🎛️ Disconnect / Isolation Settings"),
                isolate_dropdown,
                k_slider,
                divisor_slider,
                speed_slider,
                duration_slider,
                deterministic_toggle,
                seed_slider,
            ]
        )
    elif case == "stage3_stimulation":
        controls = mo.vstack(
            [
                mo.md("#### 🎛️ Stimulation Settings"),
                electrode_dropdown,
                sigma_stim_slider,
                u_intensity_slider,
                onset_slider,
                offset_slider,
                mo.md("---"),
                mo.md("#### 🎛️ Network Settings"),
                k_slider,
                divisor_slider,
                speed_slider,
                duration_slider,
                deterministic_toggle,
                seed_slider,
            ]
        )
    return (
        a_slider,
        controls,
        deterministic_toggle,
        divisor_slider,
        duration_slider,
        electrode_dropdown,
        isolate_dropdown,
        k_slider,
        offset_slider,
        onset_slider,
        seed_slider,
        sigma_slider,
        sigma_stim_slider,
        speed_slider,
        u_intensity_slider,
    )


@app.cell
def _(mo, test_case_selector):
    case = test_case_selector.value
    desc = ""
    if case == "stage0":
        desc = """
        ### Stage 0: Connectome & EEG Forward Operator Invariants
        **Goal:** Validate structural-data invariants from **The Virtual Brain (TVB)**.
        - **Success Criteria:**
          1. Connectome weights and tract lengths are $76 \\times 76$ square matrices.
          2. Conduction delays are in the millisecond range ($0 < D_{\\text{max}} < 200\\text{ ms}$).
          3. EEG gain matrix $L$ collapses $16k$ vertices to $62$ channels and $76$ regions.
          4. EZ/PZ regions (`lHC`, `lPHC`, `lAMYG`, `lTCI`, `lTCV`) are present.
        """
    elif case == "stage1":
        desc = """
        ### Stage 1: Single-Node Bifurcation & Seizure Rhythm
        **Goal:** Verify that a single isolated node reproduces the fixed-point $\\leftrightarrow$ limit-cycle transition.
        - **Success Criteria:**
          1. **Quiescence:** At healthy gain $A=3.25$, the node is quiescent. Under noise, output amplitude $PTP < 5.0$ (standard background $\\approx 2.0$).
          2. **Limit Cycle:** At EZ gain $A=3.6$, the node is a sustained limit cycle ($PTP > 5.0$).
          3. **Seizure Rhythm:** The dominant frequency of the EZ limit cycle is a slow spike-wave rhythm in the $1\\text{--}6\\text{ Hz}$ band (typically $\\sim 3\\text{ Hz}$), matching the seizure characteristics.
        """
    elif case == "stage2_healthy":
        desc = """
        ### Stage 2: All-Healthy Control Quiescence
        **Goal:** Confirm that coupling alone doesn't trigger spontaneous seizures in a healthy network.
        - **Success Criteria:**
          1. With all nodes set to healthy gain $A = 3.25$ and global coupling $K = 0.75$, the network remains quiescent.
          2. Under noise, the peak-to-peak amplitude of all nodes remains in the background range ($PTP < 5.0$).
        """
    elif case == "stage2_recruitment":
        desc = """
        ### Stage 2: EZ/PZ Seizure Recruitment
        **Goal:** Observe seizure initiation in the EZ and propagation to PZ and other nodes.
        - **Success Criteria:**
          1. EZ nodes (`lHC`, `lPHC`, `lAMYG` at $A=3.6$) oscillate immediately (onset time $< 1.0\\text{ s}$, $PTP > 5.0$).
          2. Downstream PZ nodes (`lTCI`, `lTCV` at $A=3.4$) get recruited ($PTP > 5.0$).
          3. Recruitment order is preserved: EZ onsets first, then PZ, then others.
        """
    elif case == "stage2_isolation":
        desc = """
        ### Stage 2: Coupling Correctness (Node Disconnect)
        **Goal:** Prove recruitment is network-driven by disconnecting a node and verifying it stops seizing.
        - **Success Criteria:**
          1. Zeroing incoming weights to a PZ node (e.g. `lTCI`) prevents its recruitment ($PTP < 5.0$), while neighbors still seize.
          2. Zeroing incoming weights to an EZ node (e.g. `lHC`) does *not* prevent it from seizing ($PTP > 5.0$), confirming it is intrinsically epileptogenic.
        """
    elif case == "stage2_delays":
        desc = """
        ### Stage 2: Delays vs. Instantaneous Propagation
        **Goal:** Verify conduction delays slow down recruitment propagation but do not prevent seizure spread.
        - **Success Criteria:**
          1. Both delayed and instantaneous coupling result in successful recruitment.
          2. Conduction delays strictly increase the onset time of recruitment for downstream regions compared to instantaneous coupling.
        """
    elif case == "stage3_stimulation":
        desc = """
        ### Stage 3: Immediate tES Suppression & Re-expansion
        **Goal:** Verify cathodic tES suppresses seizure propagation immediately, with quick re-expansion after offset.
        - **Success Criteria:**
          1. **Immediate Suppression:** With cathodic tES active (intensity $\\hat{u}_{\\text{tES}} = 1.5$, target CP5), the number of recruited healthy regions during stimulation is $0$ (PTP of healthy nodes $< 5.0$).
          2. **Re-expansion:** After stimulation stops, the seizure propagation re-expands and recruits healthy regions (PTP of healthy nodes $> 5.0$ in the post-stimulation window).
          3. **Polarity Sanity:** Anodic stimulation (flipping tES sign to negative intensity, e.g. $\\hat{u}_{\\text{tES}} = -1.5$) fails to suppress recruitment.
        """
    mo.md(desc)
    return (case,)


@app.cell
def _(load_connectome):
    connectome = load_connectome()
    return (connectome,)


@app.cell
def _(
    JansenRitParams,
    a_slider,
    case,
    connectome,
    deterministic_toggle,
    divisor_slider,
    duration_slider,
    electrode_dropdown,
    isolate_dropdown,
    k_slider,
    np,
    offset_slider,
    onset_slider,
    output,
    replace,
    seed_slider,
    sigma_slider,
    sigma_stim_slider,
    simulate_network,
    simulate_node,
    speed_slider,
    u_intensity_slider,
):
    # Reactive simulation execution based on parameters
    sim_data = {}

    if case == "stage0":
        # No simulation needed, invariants check
        sim_data["checked"] = True
    elif case == "stage1":
        # Single node simulation
        base_params = JansenRitParams()
        params = replace(base_params, A=a_slider.value, sigma=sigma_slider.value)
        _t, _x = simulate_node(
            params=params,
            duration=float(duration_slider.value),
            seed=int(seed_slider.value),
            deterministic=deterministic_toggle.value,
        )
        _y = output(_x)
        sim_data = {"t": _t, "x": _x, "y": _y, "A": a_slider.value, "sigma": sigma_slider.value}
    elif case == "stage2_healthy":
        # All-healthy network
        conn_scaled = replace(
            connectome, speed=speed_slider.value, delays=connectome.tract_lengths / speed_slider.value
        )
        conn_scaled = replace(conn_scaled, weights=conn_scaled.weights / divisor_slider.value)

        _t, _x = simulate_network(
            params=JansenRitParams(A=3.25),
            connectome=conn_scaled,
            K=k_slider.value,
            duration=float(duration_slider.value),
            seed=int(seed_slider.value),
            deterministic=deterministic_toggle.value,
            use_delays=True,
        )
        _y = output(_x)
        sim_data = {"t": _t, "x": _x, "y": _y}
    elif case == "stage2_recruitment":
        # EZ/PZ recruitment
        conn_scaled = replace(
            connectome, speed=speed_slider.value, delays=connectome.tract_lengths / speed_slider.value
        )
        conn_scaled = replace(conn_scaled, weights=conn_scaled.weights / divisor_slider.value)

        n_nodes = len(connectome.region_labels)
        ez_names = ("lHC", "lPHC", "lAMYG")
        pz_names = ("lTCI", "lTCV")
        _ez_idxs = [connectome.region_index[name] for name in ez_names]
        _pz_idxs = [connectome.region_index[name] for name in pz_names]

        a_gains = np.full(n_nodes, 3.25)
        a_gains[_ez_idxs] = 3.6
        a_gains[_pz_idxs] = 3.4

        _t, _x = simulate_network(
            params=JansenRitParams(A=a_gains),
            connectome=conn_scaled,
            K=k_slider.value,
            duration=float(duration_slider.value),
            seed=int(seed_slider.value),
            deterministic=deterministic_toggle.value,
            use_delays=True,
        )
        _y = output(_x)
        sim_data = {"t": _t, "x": _x, "y": _y, "ez_idxs": _ez_idxs, "pz_idxs": _pz_idxs}
    elif case == "stage2_isolation":
        # Isolate a node
        conn_scaled = replace(
            connectome, speed=speed_slider.value, delays=connectome.tract_lengths / speed_slider.value
        )
        conn_scaled = replace(conn_scaled, weights=conn_scaled.weights / divisor_slider.value)

        n_nodes = len(connectome.region_labels)
        ez_names = ("lHC", "lPHC", "lAMYG")
        pz_names = ("lTCI", "lTCV")
        _ez_idxs = [connectome.region_index[name] for name in ez_names]
        _pz_idxs = [connectome.region_index[name] for name in pz_names]

        a_gains = np.full(n_nodes, 3.25)
        a_gains[_ez_idxs] = 3.6
        a_gains[_pz_idxs] = 3.4

        _isolated_idx = None
        if isolate_dropdown.value != "None":
            _isolated_idx = connectome.region_index[isolate_dropdown.value]
            custom_weights = conn_scaled.weights.copy()
            custom_weights[_isolated_idx, :] = 0.0
            conn_scaled = replace(conn_scaled, weights=custom_weights)

        _t, _x = simulate_network(
            params=JansenRitParams(A=a_gains),
            connectome=conn_scaled,
            K=k_slider.value,
            duration=float(duration_slider.value),
            seed=int(seed_slider.value),
            deterministic=deterministic_toggle.value,
            use_delays=True,
        )
        _y = output(_x)
        sim_data = {"t": _t, "x": _x, "y": _y, "ez_idxs": _ez_idxs, "pz_idxs": _pz_idxs, "isolated_idx": _isolated_idx}
    elif case == "stage2_delays":
        # Delays vs instantaneous
        conn_scaled = replace(
            connectome, speed=speed_slider.value, delays=connectome.tract_lengths / speed_slider.value
        )
        conn_scaled = replace(conn_scaled, weights=conn_scaled.weights / divisor_slider.value)

        n_nodes = len(connectome.region_labels)
        ez_names = ("lHC", "lPHC", "lAMYG")
        pz_names = ("lTCI", "lTCV")
        _ez_idxs = [connectome.region_index[name] for name in ez_names]
        _pz_idxs = [connectome.region_index[name] for name in pz_names]

        a_gains = np.full(n_nodes, 3.25)
        a_gains[_ez_idxs] = 3.6
        a_gains[_pz_idxs] = 3.4

        _t, _x_delayed = simulate_network(
            params=JansenRitParams(A=a_gains),
            connectome=conn_scaled,
            K=k_slider.value,
            duration=float(duration_slider.value),
            seed=int(seed_slider.value),
            deterministic=deterministic_toggle.value,
            use_delays=True,
        )
        _y_delayed = output(_x_delayed)

        _t_inst, _x_inst = simulate_network(
            params=JansenRitParams(A=a_gains),
            connectome=conn_scaled,
            K=k_slider.value,
            duration=float(duration_slider.value),
            seed=int(seed_slider.value),
            deterministic=deterministic_toggle.value,
            use_delays=False,
        )
        _y_inst = output(_x_inst)

        sim_data = {
            "t": _t,
            "y_delayed": _y_delayed,
            "y_inst": _y_inst,
            "ez_idxs": _ez_idxs,
            "pz_idxs": _pz_idxs,
        }
    elif case == "stage3_stimulation":
        # Stage 3 tES
        conn_scaled = replace(
            connectome, speed=speed_slider.value, delays=connectome.tract_lengths / speed_slider.value
        )
        conn_scaled = replace(conn_scaled, weights=conn_scaled.weights / divisor_slider.value)

        # Compute gamma
        from neuro.connectome import compute_gamma

        gamma = compute_gamma(
            connectome.centres,
            target_electrode=electrode_dropdown.value,
            sigma=sigma_stim_slider.value,
        )
        conn_scaled = replace(conn_scaled, gamma=gamma)

        n_nodes = len(connectome.region_labels)
        ez_names = ("lHC", "lPHC", "lAMYG")
        pz_names = ("lTCI", "lTCV")
        _ez_idxs = [connectome.region_index[name] for name in ez_names]
        _pz_idxs = [connectome.region_index[name] for name in pz_names]
        _healthy_idxs = [i for i in range(n_nodes) if i not in _ez_idxs and i not in _pz_idxs]

        a_gains = np.full(n_nodes, 3.25)
        a_gains[_ez_idxs] = 3.6
        a_gains[_pz_idxs] = 3.4
        params = JansenRitParams(A=a_gains)

        stim_window = (float(onset_slider.value), float(offset_slider.value))

        _t, _x = simulate_network(
            params=params,
            connectome=conn_scaled,
            K=k_slider.value,
            duration=float(duration_slider.value),
            seed=int(seed_slider.value),
            deterministic=deterministic_toggle.value,
            u_hat_tES=float(u_intensity_slider.value),
            stim_window=stim_window,
        )
        _y = output(_x)
        sim_data = {
            "t": _t,
            "x": _x,
            "y": _y,
            "ez_idxs": _ez_idxs,
            "pz_idxs": _pz_idxs,
            "healthy_idxs": _healthy_idxs,
            "stim_window": stim_window,
        }
    return (sim_data,)


@app.cell
def _(case, compute_psd, connectome, mo, np, sim_data):
    # Perform verification check on sim_data
    results = []
    passed_all = True

    def _badge(ok, label):
        return mo.md(f"<div>{'✅' if ok else '❌'} {label}</div>")

    if case == "stage0":
        # 1. Structural dimensions
        w_ok = connectome.weights.shape == (76, 76)
        t_ok = connectome.tract_lengths.shape == (76, 76)
        results.append(
            _badge(w_ok and t_ok, f"Weights & Tract Lengths shape: {connectome.weights.shape} (Expected: 76x76)")
        )

        # 2. Delays ms range
        delays = connectome.delays
        off_diag = delays[~np.eye(76, dtype=bool)]
        d_ok = (np.diag(delays) == 0).all() and 0 < off_diag.max() < 200.0
        results.append(_badge(d_ok, f"Delays in ms: max={off_diag.max():.2f} ms, zero diagonal"))

        # 3. Gain L matrix
        g_ok = connectome.gain.shape == (62, 76) and np.isfinite(connectome.gain).all()
        results.append(_badge(g_ok, f"EEG gain matrix L: {connectome.gain.shape} (Expected: 62x76, finite)"))

        # 4. Regions & Channels
        ez_pz_names = ("lHC", "lPHC", "lAMYG", "lTCI", "lTCV")
        r_ok = all(r in connectome.region_index for r in ez_pz_names)
        results.append(_badge(r_ok, "EZ/PZ regions loaded correctly"))

        passed_all = w_ok and t_ok and d_ok and g_ok and r_ok

    elif case == "stage1":
        _t = sim_data["t"]
        _y = sim_data["y"]
        a_val = sim_data["A"]

        # Calculate PTP on steady portion (after 1s)
        n_steady = int(1.0 / (_t[1] - _t[0]))
        y_steady = _y[n_steady:]
        ptp = float(np.ptp(y_steady))

        if a_val == 3.25:
            # Healthy should be quiescent: ptp < 5.0
            ok = ptp < 5.0
            results.append(_badge(ok, f"Healthy Node Quiescence: PTP={ptp:.2f} mV (Expected: < 5.0 mV)"))
            passed_all = ok
        elif a_val == 3.6:
            # EZ should oscillate: ptp > 5.0
            ok_ptp = ptp > 5.0
            results.append(_badge(ok_ptp, f"EZ Node Seizure Limit Cycle: PTP={ptp:.2f} mV (Expected: > 5.0 mV)"))

            # Dominant frequency check: peak frequency should be 1-6 Hz
            _freqs, _pxx = compute_psd(_y[np.newaxis, :], (_t[1] - _t[0]) * 1000.0, nperseg=min(len(_y), 8192))
            f_peak = _freqs[np.argmax(_pxx[0])]
            ok_freq = 1.0 < f_peak < 6.0
            results.append(_badge(ok_freq, f"EZ Peak Frequency: {f_peak:.2f} Hz (Expected: 1.0 - 6.0 Hz)"))
            passed_all = ok_ptp and ok_freq
        else:
            # Other values
            results.append(mo.md(f"ℹ️ Custom A={a_val:.2f} simulation. PTP = {ptp:.2f} mV."))
            passed_all = True

    elif case == "stage2_healthy":
        _t = sim_data["t"]
        _y = sim_data["y"]
        # Calculate steady PTP for all regions
        n_steady = int(1.0 / (_t[1] - _t[0]))
        ptps = np.ptp(_y[:, n_steady:], axis=1)

        all_quiescent = np.all(ptps < 5.0)
        max_ptp = np.max(ptps)
        results.append(_badge(all_quiescent, f"All-Healthy Control: Max PTP = {max_ptp:.2f} mV (Expected: < 5.0 mV)"))
        passed_all = all_quiescent

    elif case == "stage2_recruitment":
        _t = sim_data["t"]
        _y = sim_data["y"]
        _ez_idxs = sim_data["ez_idxs"]
        _pz_idxs = sim_data["pz_idxs"]

        # Calculate PTP on steady portion
        n_steady = int(1.0 / (_t[1] - _t[0]))
        ptps = np.ptp(_y[:, n_steady:], axis=1)

        # EZ nodes must oscillate
        ez_ok = np.all(ptps[_ez_idxs] > 5.0)
        results.append(
            _badge(ez_ok, f"EZ Regions Oscillating: min PTP = {np.min(ptps[_ez_idxs]):.2f} mV (Expected: > 5.0 mV)")
        )

        # At least one PZ node must recruit
        pz_ok = np.any(ptps[_pz_idxs] > 5.0)
        results.append(
            _badge(pz_ok, f"PZ Regions Recruited: max PZ PTP = {np.max(ptps[_pz_idxs]):.2f} mV (Expected: > 5.0 mV)")
        )

        # Onset times check
        def _get_onset(signal, times, threshold=5.0):
            idx = np.where(np.abs(signal) > threshold)[0]
            return times[idx[0]] if len(idx) > 0 else float("inf")

        onsets = np.array([_get_onset(_y[i], _t) for i in range(len(ptps))])

        # EZ onsets must be small
        ez_onset_ok = np.all(onsets[_ez_idxs] < 1.0)
        results.append(_badge(ez_onset_ok, f"EZ Onset Times: max={np.max(onsets[_ez_idxs]):.3f} s (Expected: < 1.0 s)"))

        # PZ onsets must be finite
        pz_onset_ok = np.all(onsets[_pz_idxs] < float("inf"))
        results.append(
            _badge(pz_onset_ok, f"PZ Onset Times: max={np.max(onsets[_pz_idxs]):.3f} s (Expected: recruited)")
        )

        # PZ onset strictly after EZ onset
        order_ok = np.all(onsets[_pz_idxs] > np.min(onsets[_ez_idxs]))
        results.append(_badge(order_ok, "Recruitment Order: EZ starts before PZ"))

        passed_all = ez_ok and pz_ok and ez_onset_ok and pz_onset_ok and order_ok

    elif case == "stage2_isolation":
        _t = sim_data["t"]
        _y = sim_data["y"]
        _ez_idxs = sim_data["ez_idxs"]
        _pz_idxs = sim_data["pz_idxs"]
        _isolated_idx = sim_data["isolated_idx"]

        n_steady = int(1.0 / (_t[1] - _t[0]))
        ptps = np.ptp(_y[:, n_steady:], axis=1)

        if _isolated_idx is None:
            results.append(mo.md("ℹ️ No region isolated. Standard recruitment is active."))
            passed_all = True
        else:
            isolated_name = connectome.region_labels[_isolated_idx]
            is_ez = _isolated_idx in _ez_idxs
            is_pz = _isolated_idx in _pz_idxs

            isolated_ptp = ptps[_isolated_idx]

            if is_pz:
                # Isolated PZ should not seize
                ok_pz = isolated_ptp < 5.0
                results.append(
                    _badge(
                        ok_pz,
                        f"Isolated PZ '{isolated_name}' Quiescent: PTP={isolated_ptp:.2f} mV (Expected: < 5.0 mV)",
                    )
                )

                # Others should still seize
                other_pz = [i for i in _pz_idxs if i != _isolated_idx]
                ok_other = np.any(ptps[other_pz] > 5.0) if other_pz else True
                results.append(_badge(ok_other, "Other non-isolated PZ regions still recruit"))
                passed_all = ok_pz and ok_other
            elif is_ez:
                # Isolated EZ should still seize because it is intrinsically unstable (A=3.6)
                ok_ez = isolated_ptp > 5.0
                results.append(
                    _badge(
                        ok_ez,
                        f"Isolated EZ '{isolated_name}' Still Seizes: PTP={isolated_ptp:.2f} mV (Expected: > 5.0 mV)",
                    )
                )
                passed_all = ok_ez
            else:
                results.append(mo.md(f"ℹ️ Isolated healthy node '{isolated_name}'. PTP={isolated_ptp:.2f} mV."))
                passed_all = True

    elif case == "stage2_delays":
        _t = sim_data["t"]
        _y_delayed = sim_data["y_delayed"]
        _y_inst = sim_data["y_inst"]
        _ez_idxs = sim_data["ez_idxs"]
        _pz_idxs = sim_data["pz_idxs"]

        # Steady PTP
        n_steady = int(1.0 / (_t[1] - _t[0]))
        ptps_delayed = np.ptp(_y_delayed[:, n_steady:], axis=1)
        ptps_inst = np.ptp(_y_inst[:, n_steady:], axis=1)

        # Both must seize in EZ
        ez_ok_del = np.all(ptps_delayed[_ez_idxs] > 5.0)
        ez_ok_inst = np.all(ptps_inst[_ez_idxs] > 5.0)
        results.append(_badge(ez_ok_del and ez_ok_inst, "EZ Oscillations active in both simulations"))

        # Onset times for PZ nodes
        def _get_onset(signal, times, threshold=5.0):
            idx = np.where(np.abs(signal) > threshold)[0]
            return times[idx[0]] if len(idx) > 0 else float("inf")

        onsets_delayed = np.array([_get_onset(_y_delayed[i], _t) for i in _pz_idxs])
        onsets_inst = np.array([_get_onset(_y_inst[i], _t) for i in _pz_idxs])

        # Delays should slow down recruitment
        slower_ok = np.all(onsets_delayed > onsets_inst)
        results.append(
            _badge(
                slower_ok,
                f"Delayed Propagation is slower: delayed PZ onsets = {onsets_delayed} s vs. instantaneous = {onsets_inst} s",
            )
        )
        passed_all = ez_ok_del and ez_ok_inst and slower_ok
    elif case == "stage3_stimulation":
        _t = sim_data["t"]
        _y = sim_data["y"]
        _ez_idxs = sim_data["ez_idxs"]
        _pz_idxs = sim_data["pz_idxs"]
        _healthy_idxs = sim_data["healthy_idxs"]
        _onset, _offset = sim_data["stim_window"]

        # Calculate PTP in and out of stim window (allowing some transient delay)
        supp_ok = False
        reexp_ok = False

        if _offset - _onset > 0.6:
            idx_during = np.where((_t >= _onset + 0.5) & (_t < _offset - 0.1))[0]
            if len(idx_during) > 0:
                ptps_during = np.ptp(_y[_healthy_idxs, :][:, idx_during], axis=1)
                n_during = np.sum(ptps_during > 5.0)
                supp_ok = n_during == 0
                results.append(
                    _badge(
                        supp_ok,
                        f"Immediate Suppression: {n_during} healthy regions recruited during stimulation (Expected: 0)",
                    )
                )
            else:
                results.append(_badge(ok=False, label="No time indices found during stimulation window"))
        else:
            results.append(_badge(ok=False, label="Stimulation window too short to evaluate suppression"))

        if _t[-1] - _offset >= 1.0:
            idx_after = np.where(_t >= _offset + 0.8)[0]
            if len(idx_after) > 0:
                ptps_after = np.ptp(_y[_healthy_idxs, :][:, idx_after], axis=1)
                n_after = np.sum(ptps_after > 5.0)
                reexp_ok = n_after > 0
                results.append(
                    _badge(
                        reexp_ok,
                        f"Seizure Re-expansion: {n_after} healthy regions recruited after stimulation stops (Expected: > 0)",
                    )
                )
            else:
                results.append(_badge(ok=False, label="No time indices found after stimulation window"))
        else:
            results.append(_badge(ok=False, label="Post-stimulation duration too short to evaluate re-expansion"))

        passed_all = supp_ok and reexp_ok

    # Layout result banner
    banner_color = "#d4edda" if passed_all else "#f8d7da"
    banner_border = "#c3e6cb" if passed_all else "#f5c6cb"
    banner_text = "#155724" if passed_all else "#721c24"
    status_label = "✅ VALIDATION PASSED" if passed_all else "❌ VALIDATION FAILED"

    validation_panel = mo.vstack(
        [
            mo.md(f"""
        <div style="background-color: {banner_color}; border: 1px solid {banner_border}; color: {banner_text}; padding: 15px; border-radius: 5px; margin: 10px 0; font-family: sans-serif;">
            <h4 style="margin: 0 0 5px 0; font-weight: bold;">{status_label}</h4>
            Review individual invariants and metrics below.
        </div>
        """),
            mo.vstack(results),
        ]
    )
    return (validation_panel,)


@app.cell
def _(
    case,
    compute_psd,
    connectome,
    controls,
    mo,
    np,
    plt,
    sim_data,
    test_case_selector,
    validation_panel,
):
    fig = None

    if case == "stage0":
        # Render Connectome weights & delays matrix
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
        im0 = axes[0].imshow(connectome.weights, cmap="magma", aspect="auto")
        axes[0].set_title("Connection Weights (76x76)")
        axes[0].set_xlabel("Source Region")
        axes[0].set_ylabel("Target Region")
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(connectome.delays, cmap="viridis", aspect="auto")
        axes[1].set_title("Conduction Delays (ms)")
        axes[1].set_xlabel("Source Region")
        axes[1].set_ylabel("Target Region")
        fig.colorbar(im1, ax=axes[1], label="ms")

    elif case == "stage1":
        # Render time series and PSD/phase plane for single node
        _t = sim_data["t"]
        _y = sim_data["y"]
        _x = sim_data["x"]

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")

        # 1. Time Series
        axes[0].plot(_t, _y, color="#9467bd", linewidth=1.0)
        axes[0].set_title(f"Membrane Potential y (A = {sim_data['A']:.2f})")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("y = x2 - x3 (mV)")
        axes[0].spines[["top", "right"]].set_visible(False)
        axes[0].grid(visible=True, linestyle="--", alpha=0.3)

        # 2. Phase Plane or PSD depending on A
        if sim_data["A"] == 3.6:
            # Seizure - plot PSD
            dt_ms = (_t[1] - _t[0]) * 1000.0
            _freqs, _pxx = compute_psd(_y[np.newaxis, :], dt_ms, nperseg=min(len(_y), 8192))
            axes[1].plot(_freqs, _pxx[0], color="#2ca02c", linewidth=1.5)
            axes[1].set_title("Power Spectral Density")
            axes[1].set_xlabel("Frequency (Hz)")
            axes[1].set_ylabel("Power")
            axes[1].set_xlim(0, 20.0)
            axes[1].spines[["top", "right"]].set_visible(False)
            axes[1].grid(visible=True, linestyle="--", alpha=0.3)
        else:
            # Phase Plane
            # y vs y_dot (x5 - x6)
            dy = _x[4] - _x[5]
            axes[1].plot(_y, dy, color="#1f77b4", linewidth=0.8)
            axes[1].set_title("Phase Plane (y vs dy/dt)")
            axes[1].set_xlabel("y = x2 - x3")
            axes[1].set_ylabel("dy/dt = x5 - x6")
            axes[1].spines[["top", "right"]].set_visible(False)
            axes[1].grid(visible=True, linestyle="--", alpha=0.3)

    elif case in ("stage2_healthy", "stage2_recruitment", "stage2_isolation"):
        _t = sim_data["t"]
        _y = sim_data["y"]

        fig = plt.figure(figsize=(11, 7), layout="constrained")
        gs = fig.add_gridspec(2, 2, width_ratios=[1.8, 1])

        # 1. Spatiotemporal raster
        ax_raster = fig.add_subplot(gs[:, 0])
        im = ax_raster.imshow(
            _y,
            cmap="RdBu_r",
            aspect="auto",
            extent=[_t[0], _t[-1], 0, len(connectome.region_labels)],
            vmin=-10,
            vmax=10,
        )
        ax_raster.set_title("Whole-Brain Membrane Potentials (y)")
        ax_raster.set_xlabel("Time (s)")
        ax_raster.set_ylabel("Brain Region Index")
        fig.colorbar(im, ax=ax_raster, label="y = x2 - x3 (mV)")

        # 2. Representative traces
        ax_trace_ez = fig.add_subplot(gs[0, 1])
        lhc_idx = connectome.region_index["lHC"]
        ax_trace_ez.plot(_t, _y[lhc_idx], color="#d62728", label="lHC (EZ)", linewidth=0.8)
        ax_trace_ez.set_title("Representative Traces")
        ax_trace_ez.set_ylabel("y (mV)")
        ax_trace_ez.legend(loc="upper right")
        ax_trace_ez.spines[["top", "right"]].set_visible(False)
        ax_trace_ez.grid(visible=True, linestyle="--", alpha=0.3)

        ax_trace_pz = fig.add_subplot(gs[1, 1])
        ltci_idx = connectome.region_index["lTCI"]
        rhc_idx = connectome.region_index["rHC"]
        ax_trace_pz.plot(_t, _y[ltci_idx], color="#ff7f0e", label="lTCI (PZ)", linewidth=0.8)
        ax_trace_pz.plot(_t, _y[rhc_idx], color="#1f77b4", label="rHC (Healthy)", linewidth=0.8)
        ax_trace_pz.set_xlabel("Time (s)")
        ax_trace_pz.set_ylabel("y (mV)")
        ax_trace_pz.legend(loc="upper right")
        ax_trace_pz.spines[["top", "right"]].set_visible(False)
        ax_trace_pz.grid(visible=True, linestyle="--", alpha=0.3)

    elif case == "stage2_delays":
        _t = sim_data["t"]
        _y_delayed = sim_data["y_delayed"]
        _y_inst = sim_data["y_inst"]

        # Compare Delayed vs Instantaneous on representative PZ region (lTCI)
        ltci_idx = connectome.region_index["lTCI"]
        lhc_idx = connectome.region_index["lHC"]

        fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, layout="constrained")

        # Delayed
        axes[0].plot(_t, _y_delayed[lhc_idx], color="#d62728", alpha=0.4, label="lHC (EZ)", linewidth=0.8)
        axes[0].plot(_t, _y_delayed[ltci_idx], color="#ff7f0e", label="lTCI (PZ) - Delayed Coupling", linewidth=1.0)
        axes[0].set_title("Delayed Coupling Simulation")
        axes[0].set_ylabel("y (mV)")
        axes[0].legend(loc="upper right")
        axes[0].spines[["top", "right"]].set_visible(False)
        axes[0].grid(visible=True, linestyle="--", alpha=0.3)

        # Instantaneous
        axes[1].plot(_t, _y_inst[lhc_idx], color="#d62728", alpha=0.4, label="lHC (EZ)", linewidth=0.8)
        axes[1].plot(_t, _y_inst[ltci_idx], color="#1f77b4", label="lTCI (PZ) - Instantaneous Coupling", linewidth=1.0)
        axes[1].set_title("Instantaneous Coupling Simulation")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("y (mV)")
        axes[1].legend(loc="upper right")
        axes[1].spines[["top", "right"]].set_visible(False)
        axes[1].grid(visible=True, linestyle="--", alpha=0.3)

    elif case == "stage3_stimulation":
        _t = sim_data["t"]
        _y = sim_data["y"]
        _onset, _offset = sim_data["stim_window"]

        fig = plt.figure(figsize=(11, 7), layout="constrained")
        gs = fig.add_gridspec(2, 2, width_ratios=[1.8, 1])

        # 1. Spatiotemporal raster
        ax_raster = fig.add_subplot(gs[:, 0])
        im = ax_raster.imshow(
            _y,
            cmap="RdBu_r",
            aspect="auto",
            extent=[_t[0], _t[-1], 0, len(connectome.region_labels)],
            vmin=-10,
            vmax=10,
        )
        ax_raster.set_title("Whole-Brain Membrane Potentials (y) with tES")
        ax_raster.set_xlabel("Time (s)")
        ax_raster.set_ylabel("Brain Region Index")
        ax_raster.axvspan(_onset, _offset, color="magenta", alpha=0.1, label="tES Active")
        ax_raster.legend(loc="upper right")
        fig.colorbar(im, ax=ax_raster, label="y = x2 - x3 (mV)")

        # 2. Representative traces
        ax_trace_ez = fig.add_subplot(gs[0, 1])
        _lhc_idx = connectome.region_index["lHC"]
        ax_trace_ez.plot(_t, _y[_lhc_idx], color="#d62728", label="lHC (EZ)", linewidth=0.8)
        ax_trace_ez.axvspan(_onset, _offset, color="magenta", alpha=0.08)
        ax_trace_ez.set_title("Representative Traces")
        ax_trace_ez.set_ylabel("y (mV)")
        ax_trace_ez.legend(loc="upper right")
        ax_trace_ez.spines[["top", "right"]].set_visible(False)
        ax_trace_ez.grid(visible=True, linestyle="--", alpha=0.3)

        ax_trace_pz = fig.add_subplot(gs[1, 1])
        _ltci_idx = connectome.region_index["lTCI"]
        _rhc_idx = connectome.region_index["rHC"]
        ax_trace_pz.plot(_t, _y[_ltci_idx], color="#ff7f0e", label="lTCI (PZ)", linewidth=0.8)
        ax_trace_pz.plot(_t, _y[_rhc_idx], color="#1f77b4", label="rHC (Healthy)", linewidth=0.8)
        ax_trace_pz.axvspan(_onset, _offset, color="magenta", alpha=0.08)
        ax_trace_pz.set_xlabel("Time (s)")
        ax_trace_pz.set_ylabel("y (mV)")
        ax_trace_pz.legend(loc="upper right")
        ax_trace_pz.spines[["top", "right"]].set_visible(False)
        ax_trace_pz.grid(visible=True, linestyle="--", alpha=0.3)

    plt.close(fig)
    dashboard = mo.hstack(
        [
            mo.vstack([mo.md("### ⚙️ Selection"), test_case_selector, mo.md("---"), controls], width=3),
            mo.vstack(
                [
                    mo.md("### 📋 Validation Results"),
                    validation_panel,
                    mo.md("### 📊 Visualization"),
                    fig if fig is not None else mo.md("*No visualization generated.*"),
                ],
                width=9,
            ),
        ],
        justify="start",
        gap=3,
    )

    dashboard
    return


if __name__ == "__main__":
    app.run()
