import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium", app_title="Stage 7: Hand-Rolled Network")


@app.cell
def _():

    import marimo as mo
    import numpy as np
    from matplotlib import pyplot as plt

    from neuro.connectome import load_connectome
    from neuro.jansen_rit import JansenRitParams, lfp, simulate_network
    from utils.processing import band_energy, compute_psd, steady_window

    return (
        JansenRitParams,
        band_energy,
        compute_psd,
        lfp,
        load_connectome,
        mo,
        np,
        plt,
        simulate_network,
        steady_window,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # 🧠 Stage 7 — Hand-Rolled Network Simulation

    Simulate the **uncontrolled** EZ/PZ network using the hand-rolled backbone — Stages 1 (single node), 2 (coupled +
    delayed network) and 5 (EEG forward operator $L$).
    """)
    return


@app.cell
def _(load_connectome):
    conn = load_connectome()
    return (conn,)


@app.cell
def _(mo):
    k_slider = mo.ui.slider(0.0, 2.0, 0.01, value=0.61, label="Global Coupling K")
    duration_slider = mo.ui.slider(2.0, 10.0, 1.0, value=5.0, label="Duration (s)")
    seed_slider = mo.ui.slider(0, 100, 1, value=23, label="RNG Seed")
    sigma_slider = mo.ui.slider(0.0, 500.0, 25.0, value=225.0, label="Hand-rolled noise sigma")
    deterministic_toggle = mo.ui.checkbox(value=False, label="Deterministic (no noise)")

    mo.hstack(
        [
            mo.vstack([mo.md("#### 🧠 Network"), k_slider]),
            mo.vstack([mo.md("#### ⏱️ Window"), duration_slider, seed_slider]),
            mo.vstack([mo.md("#### 🎲 Noise"), sigma_slider, deterministic_toggle]),
        ],
        justify="space-between",
        gap=3,
    )
    return (
        deterministic_toggle,
        duration_slider,
        k_slider,
        seed_slider,
        sigma_slider,
    )


@app.cell
def _(conn, np):
    ez = ("lHC", "lPHC", "lAMYG")
    pz = ("lTCI", "lTCV")
    a_gains = np.full(len(conn.region_labels), 3.25)
    for _nm in ez:
        a_gains[conn.region_index[_nm]] = 3.6
    for _nm in pz:
        a_gains[conn.region_index[_nm]] = 3.4
    return (a_gains,)


@app.cell
def _(
    JansenRitParams,
    a_gains,
    conn,
    deterministic_toggle,
    duration_slider,
    k_slider,
    lfp,
    seed_slider,
    sigma_slider,
    simulate_network,
):
    _det = deterministic_toggle.value
    _sigma = 0.0 if _det else sigma_slider.value

    # Hand-rolled run, same network and gains.
    hr_params = JansenRitParams.from_config(
        {
            "connectome": conn,
            "dt": 1e-4,
            "params": {
                "A": a_gains,
                "sigma": _sigma,
                "K": k_slider.value,
                "initial_bounds": [
                    [-1.0, 1.0],
                    [-500.0, 500.0],
                    [-50.0, 50.0],
                    [-6.0, 6.0],
                    [-20.0, 20.0],
                    [-500.0, 500.0],
                ],
            },
        }
    )
    hr_t, _x = simulate_network(
        params=hr_params,
        duration=float(duration_slider.value),
        dt=1e-4,
        seed=int(seed_slider.value),
    )
    hr_y = lfp(_x)
    hr_eeg = conn.gain @ hr_y
    return hr_eeg, hr_t, hr_y


@app.cell
def _(conn, hr_t, hr_y, mo, np):
    _thr = 5.0

    def _seizing(y, t):
        ptp = np.ptp(y[:, t >= 1.0], axis=1)
        return {str(conn.region_labels[i]) for i in np.where(ptp > _thr)[0]}

    hr_seiz = _seizing(hr_y, hr_t)

    def _lat(seiz):
        if not seiz:
            return "0 L / 0 R"
        idxs = [conn.region_index[s] for s in seiz]
        left = int(np.sum(~conn.hemispheres[idxs]))
        return f"{left} L / {len(idxs) - left} R"

    mo.md(f"""
    ### 📊 Seizing-node comparison (PTP > {_thr})

    * **Hand-rolled:** {len(hr_seiz)} regions ({_lat(hr_seiz)})
    """)
    return


@app.cell
def _(hr_t, hr_y, plt):
    # Spatiotemporal raster.
    _fig, _ax = plt.subplots(figsize=(8, 5), layout="constrained")
    _im = _ax.imshow(
        hr_y,
        cmap="RdBu_r",
        aspect="auto",
        extent=[hr_t[0], hr_t[-1], 0, hr_y.shape[0]],
        vmin=-10,
        vmax=10,
    )
    _ax.set_title("Hand-rolled Simulation Raster")
    _ax.set_xlabel("time (s)")
    _ax.set_ylabel("brain region index")
    _fig.colorbar(_im, ax=_ax, label="y = x2 - x3 (a.u.)")
    _fig
    return


@app.cell
def _(compute_psd, conn, hr_t, hr_y, plt):
    # EZ trace + PSD overlay (lHC).
    _lhc = conn.region_index["lHC"]

    _fig, _axes = plt.subplots(1, 2, figsize=(13, 4), layout="constrained")
    _axes[0].plot(hr_t, hr_y[_lhc], color="#1f77b4", lw=0.7, label="hand-rolled")
    _axes[0].set_title("lHC (EZ) trace")
    _axes[0].set_xlabel("time (s)")
    _axes[0].set_ylabel("y (a.u.)")
    _axes[0].legend(loc="upper right")

    _mask = hr_t >= 1.0
    _f, _p = compute_psd(hr_y[_lhc][_mask][None, :], 0.1)
    _band = _f <= 40.0
    _axes[1].semilogy(_f[_band], _p[0][_band], color="#1f77b4", label="hand-rolled")
    _axes[1].set_title("lHC PSD")
    _axes[1].set_xlabel("frequency (Hz)")
    _axes[1].set_ylabel("PSD (a.u.²/Hz)")
    _axes[1].legend(loc="upper right")
    _fig
    return


@app.cell
def _(band_energy, conn, hr_eeg, np, plt, steady_window):
    # EEG band energy (0-50 Hz), top channels.
    _hr_e = band_energy(steady_window(hr_eeg, 0.1, 1000.0), 0.1, band=(0.0, 50.0))

    _order = np.argsort(_hr_e)[::-1][:12]
    _labels = [str(c) for c in conn.channel_labels[_order]]
    _x = np.arange(len(_order))

    _fig, _ax = plt.subplots(figsize=(12, 4), layout="constrained")
    _ax.bar(_x, _hr_e[_order], 0.6, color="#1f77b4", label="hand-rolled")
    _ax.set_xticks(_x)
    _ax.set_xticklabels(_labels, rotation=45)
    _ax.set_title("EEG 0–50 Hz energy (top hand-rolled channels) — expect left temporo-parietal")
    _ax.set_ylabel("normalized energy")
    _ax.legend()
    _fig
    return


if __name__ == "__main__":
    app.run()
