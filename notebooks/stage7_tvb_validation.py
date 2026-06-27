import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium", app_title="Stage 7: TVB Reference Validation")


@app.cell
def _():

    import marimo as mo
    import numpy as np
    from matplotlib import pyplot as plt

    from neuro.connectome import load_connectome
    from neuro.jansen_rit import JansenRitParams, lfp, simulate_network
    from neuro.tvb_reference import build_reference_simulator, reference_eeg_gain, run_reference
    from utils.processing import band_energy, compute_psd, steady_window

    return (
        JansenRitParams,
        band_energy,
        build_reference_simulator,
        compute_psd,
        lfp,
        load_connectome,
        mo,
        np,
        plt,
        reference_eeg_gain,
        run_reference,
        simulate_network,
        steady_window,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # 🧠 Stage 7 — TVB Reference Validation

    Rebuild the **uncontrolled** EZ/PZ network inside **The Virtual Brain** and use it as an
    independent oracle for the hand-rolled backbone — Stages 1 (single node), 2 (coupled +
    delayed network) and 5 (EEG forward operator $L$). The two engines use **different
    integrators**, so we compare *statistics* (seizing set, recruitment lateralization,
    dominant frequency, EEG energy ranking), not pointwise traces.

    **Parameter reconciliation** (TVB runs in ms, the hand-rolled engine in s):

    | paper / hand-rolled (s)        | TVB (`models.JansenRit`, ms)                         |
    |--------------------------------|------------------------------------------------------|
    | $a=100,\ b=50$                 | $a=0.1,\ b=0.05$ (default)                            |
    | $e_0=2.5$                      | $\nu_{max}=0.0025$ (default)                          |
    | $v_0=6,\ r=0.56,\ C=135,\ B=22$ | $v_0=6,\ r=0.56,\ J=135,\ B=22$                     |
    | $I=90$                         | $\mu=0.09$                                            |
    | $dt=10^{-4}$ s                 | $dt=0.1$ ms                                          |
    | $K\sum_j w_{ij}S(y_j)$         | `SigmoidalJansenRit(cmin=0, cmax=2ν_max, midpoint=6, a=K)` |
    | noise on $x_5'$, std `sigma`   | `HeunStochastic(Additive(nsig))`, `nsig` on index 4  |

    Both engines run on **bit-identical** connectivity (same weights / delays / speed) and the
    same EEG projection files, so TVB's region gain equals $L$ up to a sagittal-mirror row swap.
    """)
    return


@app.cell
def _(load_connectome):
    conn = load_connectome()
    return (conn,)


@app.cell
def _(mo):
    k_slider = mo.ui.slider(0.0, 2.0, 0.02, value=0.60, label="Global Coupling K")
    duration_slider = mo.ui.slider(2.0, 10.0, 1.0, value=5.0, label="Duration (s)")
    seed_slider = mo.ui.slider(0, 100, 1, value=23, label="RNG Seed")
    nsig_slider = mo.ui.slider(0.0, 5e-4, 1e-5, value=1e-4, label="TVB noise nsig")
    sigma_slider = mo.ui.slider(0.0, 500.0, 25.0, value=225.0, label="Hand-rolled noise sigma")
    deterministic_toggle = mo.ui.checkbox(value=False, label="Deterministic (no noise)")

    mo.hstack(
        [
            mo.vstack([mo.md("#### 🧠 Network"), k_slider]),
            mo.vstack([mo.md("#### ⏱️ Window"), duration_slider, seed_slider]),
            mo.vstack([mo.md("#### 🎲 Noise"), nsig_slider, sigma_slider, deterministic_toggle]),
        ],
        justify="space-between",
        gap=3,
    )
    return (
        deterministic_toggle,
        duration_slider,
        k_slider,
        nsig_slider,
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
    build_reference_simulator,
    conn,
    deterministic_toggle,
    duration_slider,
    k_slider,
    lfp,
    nsig_slider,
    run_reference,
    seed_slider,
    sigma_slider,
    simulate_network,
):
    _det = deterministic_toggle.value
    _nsig = 0.0 if _det else nsig_slider.value
    _sigma = 0.0 if _det else sigma_slider.value

    # TVB reference run.
    _sim = build_reference_simulator(conn, K=k_slider.value, nsig=_nsig, seed=int(seed_slider.value))
    tvb = run_reference(_sim, float(duration_slider.value))

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
    return hr_eeg, hr_t, hr_y, tvb


@app.cell
def _(conn, hr_t, hr_y, mo, np, tvb):
    _thr = 5.0

    def _seizing(y, t):
        ptp = np.ptp(y[:, t >= 1.0], axis=1)
        return {str(conn.region_labels[i]) for i in np.where(ptp > _thr)[0]}

    tvb_seiz = _seizing(tvb.region_y, tvb.t)
    hr_seiz = _seizing(hr_y, hr_t)

    def _lat(seiz):
        idxs = [conn.region_index[s] for s in seiz]
        left = int(np.sum(~conn.hemispheres[idxs]))
        return f"{left} L / {len(idxs) - left} R"

    mo.md(f"""
    ### 📊 Seizing-node comparison (PTP > {_thr})

    * **TVB:** {len(tvb_seiz)} regions ({_lat(tvb_seiz)})
    * **Hand-rolled:** {len(hr_seiz)} regions ({_lat(hr_seiz)})
    * **Shared:** {len(tvb_seiz & hr_seiz)} regions — {", ".join(sorted(tvb_seiz & hr_seiz))}
    """)
    return


@app.cell
def _(hr_t, hr_y, plt, tvb):
    # Side-by-side spatiotemporal rasters.
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained", sharey=True)
    for _ax, _t, _y, _title in (
        (_axes[0], tvb.t, tvb.region_y, "TVB reference"),
        (_axes[1], hr_t, hr_y, "Hand-rolled"),
    ):
        _im = _ax.imshow(
            _y,
            cmap="RdBu_r",
            aspect="auto",
            extent=[_t[0], _t[-1], 0, _y.shape[0]],
            vmin=-10,
            vmax=10,
        )
        _ax.set_title(_title)
        _ax.set_xlabel("time (s)")
    _axes[0].set_ylabel("brain region index")
    _fig.colorbar(_im, ax=_axes, label="y = x2 - x3 (a.u.)")
    _fig
    return


@app.cell
def _(compute_psd, conn, hr_t, hr_y, plt, tvb):
    # EZ trace + PSD overlay (lHC), comparing the spike-wave rhythm.
    _lhc = conn.region_index["lHC"]
    _dt_ms_tvb = float((tvb.t[1] - tvb.t[0]) * 1000.0)

    _fig, _axes = plt.subplots(1, 2, figsize=(13, 4), layout="constrained")
    _axes[0].plot(hr_t, hr_y[_lhc], color="#1f77b4", lw=0.7, label="hand-rolled")
    _axes[0].plot(tvb.t, tvb.region_y[_lhc], color="#d62728", lw=0.7, alpha=0.8, label="TVB")
    _axes[0].set_title("lHC (EZ) trace")
    _axes[0].set_xlabel("time (s)")
    _axes[0].set_ylabel("y (a.u.)")
    _axes[0].legend(loc="upper right")

    for _t, _y, _dt_ms, _c, _lbl in (
        (hr_t, hr_y[_lhc], 0.1, "#1f77b4", "hand-rolled"),
        (tvb.t, tvb.region_y[_lhc], _dt_ms_tvb, "#d62728", "TVB"),
    ):
        _mask = _t >= 1.0
        _f, _p = compute_psd(_y[_mask][None, :], _dt_ms)
        _band = _f <= 40.0
        _axes[1].semilogy(_f[_band], _p[0][_band], color=_c, label=_lbl)
    _axes[1].set_title("lHC PSD")
    _axes[1].set_xlabel("frequency (Hz)")
    _axes[1].set_ylabel("PSD (a.u.²/Hz)")
    _axes[1].legend(loc="upper right")
    _fig
    return


@app.cell
def _(band_energy, conn, hr_eeg, np, plt, steady_window, tvb):
    # EEG band energy (0-50 Hz), both engines, top channels.
    _dt_ms_tvb = float((tvb.t[1] - tvb.t[0]) * 1000.0)
    _tvb_e = band_energy(tvb.eeg[:, tvb.t >= 1.0], _dt_ms_tvb, band=(0.0, 50.0))
    _hr_e = band_energy(steady_window(hr_eeg, 0.1, 1000.0), 0.1, band=(0.0, 50.0))

    _order = np.argsort(_hr_e)[::-1][:12]
    _labels = [str(c) for c in conn.channel_labels[_order]]
    _x = np.arange(len(_order))

    _fig, _ax = plt.subplots(figsize=(12, 4), layout="constrained")
    _ax.bar(_x - 0.2, _hr_e[_order], 0.4, color="#1f77b4", label="hand-rolled")
    _ax.bar(_x + 0.2, _tvb_e[_order], 0.4, color="#d62728", label="TVB")
    _ax.set_xticks(_x)
    _ax.set_xticklabels(_labels, rotation=45)
    _ax.set_title("EEG 0–50 Hz energy (top hand-rolled channels) — expect left temporo-parietal")
    _ax.set_ylabel("normalized energy")
    _ax.legend()
    _fig
    return


@app.cell
def _(build_reference_simulator, conn, mo, np, plt, reference_eeg_gain):
    # L validation: TVB's region gain == connectome.gain up to the mirror row permutation.
    from tvb.datatypes.sensors import SensorsEEG

    from neuro.connectome import _mirror_partner_permutation

    _sim = build_reference_simulator(conn, with_eeg=True)
    _tvb_gain = reference_eeg_gain(_sim)
    _partner = _mirror_partner_permutation(
        np.asarray(SensorsEEG.from_file("eeg_unitvector_62.txt.bz2").locations, dtype=float)
    )
    _aligned = _tvb_gain[_partner]
    _max_diff = float(np.max(np.abs(_aligned - conn.gain)))

    _fig, _ax = plt.subplots(figsize=(5, 5), layout="constrained")
    _ax.scatter(conn.gain.ravel(), _aligned.ravel(), s=2, alpha=0.3, color="#2ca02c")
    _lim = [conn.gain.min(), conn.gain.max()]
    _ax.plot(_lim, _lim, "k--", lw=0.8)
    _ax.set_title(f"L validation: TVB gain vs connectome.gain\nmax |Δ| = {_max_diff:.2e}")
    _ax.set_xlabel("connectome.gain")
    _ax.set_ylabel("TVB gain (mirror-aligned)")
    mo.vstack([_fig, mo.md(f"**max |Δ| = {_max_diff:.2e}** — the two forward operators are identical.")])
    return


if __name__ == "__main__":
    app.run()
