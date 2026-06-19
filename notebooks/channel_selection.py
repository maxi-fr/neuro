import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium", app_title="EEG Channel Selection")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from matplotlib import pyplot as plt

    from neuro.connectome import load_connectome
    from neuro.jansen_rit import JansenRitParams, lfp, simulate_network
    from utils.channel_selection import compute_channel_gramians, pca_loading_scores, select_channels
    from utils.plotting import plot_psd, plot_signals
    from utils.processing import steady_window

    return (
        JansenRitParams,
        compute_channel_gramians,
        lfp,
        load_connectome,
        mo,
        np,
        pca_loading_scores,
        plot_psd,
        plot_signals,
        plt,
        select_channels,
        simulate_network,
        steady_window,
    )


@app.cell
def _(load_connectome):
    sim_dt = 1e-4
    sim_k = 0.75
    sim_seed = 42

    connectome = load_connectome(speed=50.0)
    return connectome, sim_dt, sim_k, sim_seed


@app.cell
def _(connectome, np):
    ez_names = ("lHC", "lPHC", "lAMYG")
    pz_names = ("lTCI", "lTCV")

    n_nodes = len(connectome.region_labels)
    a_gains = np.full(n_nodes, 3.25)
    a_gains[[connectome.region_index[name] for name in ez_names]] = 3.6
    a_gains[[connectome.region_index[name] for name in pz_names]] = 3.4
    return (a_gains,)


@app.cell
def _(mo):
    mo.md(r"""
    # 🧠 EEG Channel Selection

    Runs an EZ/PZ seizure simulation, then ranks the 62 EEG channels by **PCA loading
    scores** and by the **empirical observability Gramian** (see
    `knowledge-base/Notes/model_reduction.md`).
    """)
    return


@app.cell
def _(mo):
    duration_slider = mo.ui.slider(2.0, 30.0, 1.0, value=10.0, label="Seizure duration (s)")
    transient_slider = mo.ui.slider(0.0, 3000.0, 250.0, value=500.0, label="Transient discarded (ms)")
    n_select_slider = mo.ui.slider(1, 20, 1, value=8, label="Number of channels to select (M)")
    variance_slider = mo.ui.slider(0.5, 0.999, 0.01, value=0.95, label="PCA variance threshold")

    mo.hstack(
        [
            mo.vstack([mo.md("### ⏱️ Seizure Data"), duration_slider, transient_slider]),
            mo.vstack([mo.md("### 📊 Selection"), n_select_slider, variance_slider]),
        ],
        justify="space-between",
        gap=4,
    )
    return duration_slider, n_select_slider, transient_slider, variance_slider


@app.cell
def _(
    JansenRitParams,
    a_gains,
    connectome,
    duration_slider,
    lfp,
    sim_dt,
    sim_k,
    sim_seed,
    simulate_network,
):
    _, x_traj = simulate_network(
        params=JansenRitParams.from_config(
            {"connectome": connectome, "dt": sim_dt, "params": {"A": a_gains, "K": sim_k}}
        ),
        duration=float(duration_slider.value),
        dt=sim_dt,
        seed=sim_seed,
    )
    eeg = connectome.gain @ lfp(x_traj)
    x0 = x_traj[:, :, -1]
    return eeg, x0


@app.cell
def _(eeg, sim_dt, steady_window, transient_slider):
    eeg_view = steady_window(eeg, sim_dt * 1000.0, float(transient_slider.value))
    return (eeg_view,)


@app.cell
def _(eeg_view, n_select_slider, np, pca_loading_scores, variance_slider):
    pca_scores, pca_explained_variance_ratio = pca_loading_scores(
        eeg_view, variance_threshold=float(variance_slider.value)
    )
    pca_ranking = np.argsort(pca_scores)[::-1]
    pca_top = pca_ranking[: n_select_slider.value]
    return pca_explained_variance_ratio, pca_ranking, pca_scores, pca_top


@app.cell
def _(connectome, mo, np, pca_ranking, pca_scores, pca_top, plt):
    _fig, _ax = plt.subplots(figsize=(12, 4), layout="constrained")

    _labels = connectome.channel_labels[pca_ranking]
    _scores = pca_scores[pca_ranking]
    _top_set = set(pca_top.tolist())
    _colors = ["#d62728" if i in _top_set else "#1f77b4" for i in pca_ranking]

    _ax.bar(np.arange(len(_labels)), _scores, color=_colors)
    _ax.set_xticks(np.arange(len(_labels)))
    _ax.set_xticklabels(_labels, rotation=90, fontsize=7)
    _ax.set_ylabel("PCA loading score")
    _ax.set_title("PCA loading scores per EEG channel (red = selected)")
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)

    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo, np, pca_explained_variance_ratio, plt, variance_slider):
    _cum_var = np.cumsum(pca_explained_variance_ratio)
    _threshold = float(variance_slider.value)
    _n_comps = min(
        int(np.searchsorted(_cum_var, _threshold)) + 1,
        pca_explained_variance_ratio.size,
    )

    _fig, _ax1 = plt.subplots(figsize=(12, 3), layout="constrained")

    _color_ind = "#1f77b4"
    _ax1.bar(
        np.arange(len(pca_explained_variance_ratio)) + 1,
        pca_explained_variance_ratio,
        color=_color_ind,
        alpha=0.6,
        label="Individual",
    )
    _ax1.set_xlabel("Principal Component")
    _ax1.set_ylabel("Explained Variance Ratio", color=_color_ind)
    _ax1.tick_params(axis="y", labelcolor=_color_ind)

    _ax2 = _ax1.twinx()
    _color_cum = "#d62728"
    _ax2.plot(
        np.arange(len(_cum_var)) + 1,
        _cum_var,
        color=_color_cum,
        marker="o",
        markersize=3,
        label="Cumulative",
    )
    _ax2.axhline(
        y=_threshold,
        color="gray",
        linestyle="--",
        alpha=0.7,
        label=f"Threshold ({_threshold * 100:.0f}%)",
    )
    _ax2.axvline(
        x=_n_comps,
        color="green",
        linestyle=":",
        alpha=0.7,
        label=f"Selected components ({_n_comps})",
    )
    _ax2.set_ylabel("Cumulative Explained Variance", color=_color_cum)
    _ax2.tick_params(axis="y", labelcolor=_color_cum)
    _ax2.set_ylim(0, 1.05)

    _ax1.set_title(f"PCA Scree Plot: {_n_comps} components explain >= {_threshold * 100:.1f}% variance")
    _ax1.spines["top"].set_visible(False)
    _ax2.spines["top"].set_visible(False)

    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo):
    mo.md("""
    ## EEG traces and spectra for the PCA-selected channels
    """)
    return


@app.cell
def _(connectome, eeg_view, mo, pca_top, plot_signals, sim_dt):
    _fig, _ = plot_signals(
        eeg_view,
        dt_ms=sim_dt * 1000.0,
        channels_to_plot=pca_top.tolist(),
        channel_names=list(connectome.channel_labels),
        title="EEG traces of PCA-selected channels",
        color="#d62728",
    )
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(connectome, eeg_view, mo, pca_top, plot_psd, sim_dt):
    _nperseg = min(8192, eeg_view.shape[1])
    _fig, _ = plot_psd(
        eeg_view,
        dt_ms=sim_dt * 1000.0,
        channels_to_plot=pca_top.tolist(),
        channel_names=list(connectome.channel_labels),
        max_freq=50.0,
        nperseg=_nperseg,
    )
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Empirical Observability Gramian

    Computing the Gramian runs `6 * 76 + 1 = 457` short, deterministic simulations
    (one baseline plus one per perturbed state dimension), so it is **not**
    recomputed automatically. Adjust the settings below and click "Compute" to run it.
    """)
    return


@app.cell
def _(mo):
    gramian_duration_slider = mo.ui.slider(0.05, 0.5, 0.05, value=0.1, label="Gramian sim duration (s)")
    epsilon_input = mo.ui.number(value=1e-6, label="Perturbation epsilon")
    compute_button = mo.ui.run_button(label="Compute Gramians")

    mo.hstack([gramian_duration_slider, epsilon_input, compute_button], gap=4)
    return compute_button, epsilon_input, gramian_duration_slider


@app.cell
def _(
    JansenRitParams,
    a_gains,
    compute_button,
    compute_channel_gramians,
    connectome,
    epsilon_input,
    gramian_duration_slider,
    mo,
    n_select_slider,
    select_channels,
    sim_dt,
    sim_k,
    x0,
):
    mo.stop(not compute_button.value, mo.md("*Click 'Compute Gramians' to run the sensitivity simulations.*"))

    gramians = compute_channel_gramians(
        x0,
        JansenRitParams(A=a_gains),
        connectome,
        K=sim_k,
        dt=sim_dt,
        duration=float(gramian_duration_slider.value),
        epsilon=float(epsilon_input.value),
    )

    det_selection = select_channels(gramians, n_select_slider.value, criterion="det")
    min_eig_selection = select_channels(gramians, n_select_slider.value, criterion="min_eig")
    return det_selection, gramians, min_eig_selection


@app.cell
def _(connectome, det_selection, gramians, min_eig_selection, mo, np, plt):
    _traces = np.trace(gramians, axis1=1, axis2=2)
    _order = np.argsort(_traces)[::-1]

    _det_set = set(det_selection)
    _min_eig_set = set(min_eig_selection)

    def _color(idx: int) -> str:
        if idx in _det_set and idx in _min_eig_set:
            return "#9467bd"  # both
        if idx in _det_set:
            return "#1f77b4"  # det only
        if idx in _min_eig_set:
            return "#ff7f0e"  # min_eig only
        return "#cccccc"

    _fig, _ax = plt.subplots(figsize=(12, 4), layout="constrained")
    _ax.bar(np.arange(len(_order)), _traces[_order], color=[_color(i) for i in _order])
    _ax.set_xticks(np.arange(len(_order)))
    _ax.set_xticklabels(connectome.channel_labels[_order], rotation=90, fontsize=7)
    _ax.set_ylabel("trace(Wo_i)")
    _ax.set_title("Per-channel observability Gramian trace (purple=det & min_eig, blue=det, orange=min_eig)")
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)

    mo.mpl.interactive(_fig)
    return


@app.cell
def _(connectome, det_selection, min_eig_selection, mo, pca_top):
    _pca_labels = ", ".join(connectome.channel_labels[i] for i in pca_top)
    _det_labels = ", ".join(connectome.channel_labels[i] for i in det_selection)
    _min_eig_labels = ", ".join(connectome.channel_labels[i] for i in min_eig_selection)

    mo.md(f"""
    **Selected channels**

    * PCA: {_pca_labels}
    * Gramian (det): {_det_labels}
    * Gramian (min eig): {_min_eig_labels}
    """)
    return


@app.cell
def _(
    connectome,
    det_selection,
    min_eig_selection,
    mo,
    np,
    pca_ranking,
    pca_top,
    plt,
):
    _criteria = ["PCA", "Gramian (det)", "Gramian (min eig)"]
    _selections = [set(pca_top.tolist()), set(det_selection), set(min_eig_selection)]
    _order = pca_ranking  # show channels sorted by PCA score for a consistent x-axis

    _matrix = np.array([[1.0 if idx in sel else 0.0 for idx in _order] for sel in _selections])

    _fig, _ax = plt.subplots(figsize=(12, 2.5), layout="constrained")
    _ax.imshow(_matrix, aspect="auto", cmap="Greys", vmin=0, vmax=1)
    _ax.set_yticks(range(len(_criteria)))
    _ax.set_yticklabels(_criteria)
    _ax.set_xticks(range(len(_order)))
    _ax.set_xticklabels(connectome.channel_labels[_order], rotation=90, fontsize=7)
    _ax.set_title("Channel selection overlap across criteria (channels sorted by PCA score)")

    mo.mpl.interactive(_fig)
    return


if __name__ == "__main__":
    app.run()
