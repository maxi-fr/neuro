import marimo

__generated_with = "0.23.10"
app = marimo.App(
    width="medium",
    app_title="Reduced sysid model vs ground truth",
)


@app.cell
def _():
    from pathlib import Path

    import jax.numpy as jnp
    import marimo as mo
    import numpy as np
    import yaml
    from matplotlib import pyplot as plt

    from neuro.connectome import load_connectome
    from neuro.jansen_rit import JansenRitParams
    from neuro.jansen_rit_jax import eeg_jax, enable_x64, from_jansen_rit_params
    from neuro.sysid_jax import rollout_tbptt
    from utils.plotting import plot_multistep_predictions, plot_signals
    from utils.processing import compute_psd

    enable_x64()  # float64 parity; must run before any jnp array is created
    return (
        JansenRitParams,
        Path,
        compute_psd,
        eeg_jax,
        from_jansen_rit_params,
        jnp,
        load_connectome,
        mo,
        np,
        plot_multistep_predictions,
        plot_signals,
        plt,
        rollout_tbptt,
        yaml,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Reduced system-identification result vs ground truth

    `scripts/run_sysid_jax.py` PCA-reduces the 76-node Jansen-Rit plant to `N` virtual
    modes and fits the reduced `A` / `w_weights` against the uncontrolled EEG
    statistics, saving the learned parameters to `results/sysid_jax_result.npz`.

    This notebook rebuilds that reduced model, runs it forward, and compares its EEG
    against the ground-truth plant simulation.

    **The ground truth is a stochastic simulation while the reduced free-run is
    deterministic** — so the comparison is *statistical* (spectral shape, spatial
    functional connectivity, latent-mode character), not sample-by-sample. The
    time-domain traces below are illustrative only.
    """)
    return


@app.cell
def _(Path, yaml):
    config_path = Path(__file__).parent.parent / "./configs/sysid/sysid_jax_config.yaml"
    cfg = yaml.safe_load(config_path.read_text())

    sim_cfg = cfg.get("simulation", {})
    train_cfg = cfg.get("training", {})

    downsample = int(sim_cfg.get("downsample", 1))
    n_steps = int(sim_cfg["n_steps"])
    window = int(train_cfg.get("window", 300))
    dt = float(sim_cfg.get("dt", 1e-4)) * downsample  # effective reduced-model step (s)
    dt_ms = dt * 1000.0
    band = tuple(cfg.get("band", (4.0, 40.0)))

    # Determine simulation path
    data_path_str = sim_cfg["data_path"]
    data_path = Path(__file__).parent.parent / data_path_str
    sim_paths = sorted([str(p) for p in data_path.glob("*.npz")])
    sim_path = Path(sim_paths[0])

    # Determine output path
    legacy_out = cfg.get("out")
    if legacy_out:
        out_path = Path(__file__).parent.parent / legacy_out
    else:
        # Search for the latest artifact directory
        artifacts_root = Path(__file__).parent.parent / "artifacts"
        sysid_dirs = sorted(artifacts_root.glob("sysid_jax_*"))
        if sysid_dirs:
            out_path = sysid_dirs[-1] / "sysid_jax_result.npz"
        else:
            out_path = Path(__file__).parent.parent / "results/sysid_jax_result.npz"

    print(f"sim        : {sim_path}")
    print(f"result     : {out_path}")
    print(f"dt={dt:g}s  dt_ms={dt_ms:g}  n_steps={n_steps}  window={window}  band={band}")
    return band, downsample, dt, dt_ms, n_steps, out_path, sim_path, window


@app.cell
def _(downsample, n_steps, np, sim_path):
    # Ground truth from the full-plant simulation log (mirrors run_sysid_jax._load_plant).
    with np.load(sim_path) as _data:
        x_data = np.asarray(_data["x"])[::downsample][:n_steps]  # (n_steps, 6, 76)
        _y_mea = np.asarray(_data["y_mea"])[::downsample][:n_steps]  # (n_steps, 62)
    node_output = x_data[:, 1, :] - x_data[:, 2, :]  # (n_steps, 76) per-region LFP
    y_data = _y_mea.T  # (62, n_steps) ground-truth EEG
    print(f"node_output {node_output.shape}   y_data {y_data.shape}")
    return node_output, x_data, y_data


@app.cell
def _(load_connectome, np, out_path):
    # Identified reduced-model parameters.
    with np.load(out_path) as _res:
        a_syn = _res["A"]  # excitatory gain A, shape (N,)
        w_weights = _res["w_weights"]
        k_global = float(_res["K"])  # global coupling K
        mean_input = _res["mean_input"]
        eeg_gain = _res["eeg_gain"]  # coarse leadfield (62, N)
        components = _res["components"]  # PCA modes -> regions (N, 76)
        delay_steps = _res["delay_steps"]
        history = _res["history"]

    n = int(a_syn.shape[0])
    connectome = load_connectome()
    channel_names = list(connectome.channel_labels)
    print(f"N modes = {n}   eeg_gain {eeg_gain.shape}   components {components.shape}")
    return (
        a_syn,
        channel_names,
        components,
        delay_steps,
        eeg_gain,
        history,
        k_global,
        mean_input,
        n,
        w_weights,
    )


@app.cell
def _(
    JansenRitParams,
    a_syn,
    delay_steps,
    dt,
    eeg_gain,
    eeg_jax,
    from_jansen_rit_params,
    jnp,
    k_global,
    mean_input,
    n,
    np,
    rollout_tbptt,
    w_weights,
    window,
    y_data,
):
    # Rebuild the reduced JRParamsJax (defaults supply the fixed JR constants) and run it
    # forward, exactly as run_sysid_jax._base_params / _report do.
    _jr = JansenRitParams(
        A=a_syn,
        mean_input=mean_input,
        sigma=0.0,
        w_weights=w_weights,
        delay_steps=delay_steps,
        eeg_gain=eeg_gain,
        gamma=np.zeros((1, n)),
        K=k_global,
    )
    params = from_jansen_rit_params(_jr, n)

    n_t = y_data.shape[1]
    x_traj = rollout_tbptt(jnp.zeros((6, n)), jnp.zeros((n_t, n)), params, dt, window)
    y_fit = np.asarray(eeg_jax(x_traj, params.eeg_gain))  # (62, n_steps)
    lfp_reduced = np.asarray(x_traj[1] - x_traj[2]).T  # (n_steps, N)
    print(f"y_fit {y_fit.shape}   lfp_reduced {lfp_reduced.shape}")
    return lfp_reduced, params, y_fit


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. EEG time series

    A handful of channels, ground truth (blue) vs reduced model (red). Note these will
    not line up sample-for-sample — only their *character* (rhythm, amplitude regime)
    is comparable.
    """)
    return


@app.cell
def _(channel_names, dt_ms, plot_signals, plt, y_data, y_fit):
    _sel = [0, 15, 30, 45, 60]
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained", sharex=True)
    plot_signals(
        y_data,
        dt_ms,
        channel_names=channel_names,
        channels_to_plot=_sel,
        title="Ground truth EEG",
        color="#1f77b4",
        ax=_axes[0],
    )
    plot_signals(
        y_fit,
        dt_ms,
        channel_names=channel_names,
        channels_to_plot=_sel,
        title="Reduced-model EEG",
        color="#d62728",
        ax=_axes[1],
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Power spectral density

    Channel-mean Welch PSD (log scale). The shaded region is the band the loss was
    fitted on — that is where the curves should agree best.
    """)
    return


@app.cell
def _(band, compute_psd, dt_ms, plt, y_data, y_fit):
    _f_gt, _p_gt = compute_psd(y_data, dt_ms)
    _f_fit, _p_fit = compute_psd(y_fit, dt_ms)

    _fig, _ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    _ax.axvspan(band[0], band[1], color="grey", alpha=0.12, label=f"fit band {band[0]:g}-{band[1]:g} Hz")
    _ax.plot(_f_gt, _p_gt.mean(axis=0), color="#1f77b4", linewidth=2.0, label="ground truth")
    _ax.plot(_f_fit, _p_fit.mean(axis=0), color="#d62728", linewidth=2.0, alpha=0.85, label="reduced model")
    _ax.set_xlim(0, min(60.0, float(_f_gt[-1])))
    _ax.set_xlabel("Frequency (Hz)", fontsize=11, fontweight="bold")
    _ax.set_ylabel("Power (a.u.)", fontsize=11, fontweight="bold")
    _ax.set_title("Channel-mean PSD: ground truth vs reduced", fontsize=13, fontweight="bold", pad=12)
    _ax.spines[["top", "right"]].set_visible(False)
    _ax.grid(visible=True, linestyle="--", alpha=0.3)
    _ax.legend(loc="upper right", framealpha=0.9)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Spatial functional connectivity

    Channel-by-channel correlation of the EEG. The reduced model should reproduce the
    spatial structure (the `spatial` term in the loss); the right panel is the
    signed difference.
    """)
    return


@app.cell
def _(np, plt, y_data, y_fit):
    _fc_gt = np.corrcoef(y_data)
    _fc_fit = np.corrcoef(y_fit)
    _diff = _fc_gt - _fc_fit

    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    for _ax, _m, _ttl, _cmap, _vmax in (
        (_axes[0], _fc_gt, "Ground truth FC", "RdBu_r", 1.0),
        (_axes[1], _fc_fit, "Reduced-model FC", "RdBu_r", 1.0),
        (_axes[2], _diff, "Difference (GT - reduced)", "RdBu_r", float(np.abs(_diff).max())),
    ):
        _im = _ax.imshow(_m, cmap=_cmap, aspect="auto", vmin=-_vmax, vmax=_vmax)
        _ax.set_title(_ttl, fontsize=12, fontweight="bold")
        _ax.set_xlabel("channel")
        _ax.set_ylabel("channel")
        _fig.colorbar(_im, ax=_ax, fraction=0.046, pad=0.04)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Latent modes & training loss

    The reduced model lives in the PCA mode space. Left/centre: the first few modes of
    the ground truth (projected through the saved PCA components) vs the reduced
    model's own latent LFP. Right: the optimisation loss history.
    """)
    return


@app.cell
def _(components, dt_ms, history, lfp_reduced, node_output, plot_signals, plt):
    gt_latent = node_output @ components.T  # (n_steps, N)
    _sel = list(range(min(4, gt_latent.shape[1])))
    _mode_names = [f"mode {i}" for i in range(gt_latent.shape[1])]

    _fig, _axes = plt.subplots(1, 3, figsize=(16, 4.5), layout="constrained")
    plot_signals(
        gt_latent.T,
        dt_ms,
        channel_names=_mode_names,
        channels_to_plot=_sel,
        title="Ground-truth latent modes",
        color="#1f77b4",
        ax=_axes[0],
    )
    plot_signals(
        lfp_reduced.T,
        dt_ms,
        channel_names=_mode_names,
        channels_to_plot=_sel,
        title="Reduced-model latent modes",
        color="#d62728",
        ax=_axes[1],
    )
    _axes[2].plot(history, color="#2ca02c", linewidth=1.5)
    _axes[2].set_title("Refinement loss", fontsize=12, fontweight="bold")
    _axes[2].set_xlabel("optimisation step")
    _axes[2].set_ylabel("loss")
    _axes[2].spines[["top", "right"]].set_visible(False)
    _axes[2].grid(visible=True, linestyle="--", alpha=0.3)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Fit metrics
    """)
    return


@app.cell
def _(band, compute_psd, dt_ms, history, mo, np, y_data, y_fit):
    def _logpsd(sig):
        _f, _p = compute_psd(sig, dt_ms=dt_ms)
        _mask = (_f >= band[0]) & (_f <= band[1])
        return np.log(_p[:, _mask] + 1e-30)

    psd_corr = float(np.corrcoef(_logpsd(y_data).ravel(), _logpsd(y_fit).ravel())[0, 1])
    fc_err = float(np.mean((np.corrcoef(y_data) - np.corrcoef(y_fit)) ** 2))

    mo.md(
        f"""
        | metric | value |
        | --- | --- |
        | log-PSD correlation (band {band[0]:g}-{band[1]:g} Hz) | **{psd_corr:.4f}** |
        | spatial-FC MSE | **{fc_err:.5f}** |
        | final refinement loss | **{history[-1]:.5f}** (from {history[0]:.5f}) |

        These mirror the scalars `run_sysid_jax.py` prints, recomputed here from the
        rebuilt model.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Short-Horizon Predictions

    Evaluate the identified model's ability to predict the ground-truth plant over relatively short prediction horizons. We project the full-plant states to the latent modes to initialize the reduced model, then simulate forward and compare the predicted EEG.
    """)
    return


@app.cell
def _(
    components,
    dt,
    eeg_gain,
    eeg_jax,
    jnp,
    np,
    params,
    plt,
    rollout_tbptt,
    x_data,
    y_data,
):
    # Project ground truth full states into reduced latent space
    x_reduced = x_data @ components.T  # (n_steps, 6, N)

    horizons = [10, 20, 50, 100, 1000]
    n_t_pred = y_data.shape[1]
    n_trials = 50  # evaluate from multiple random start points

    # pick random valid start times
    np.random.seed(69)
    t0s = np.random.randint(0, n_t_pred - max(horizons) - 1, size=n_trials)

    mses = {h: [] for h in horizons}
    corrs = {h: [] for h in horizons}

    for t0 in t0s:
        x0 = jnp.asarray(x_reduced[t0])
        max_h = max(horizons)
        u_dummy = jnp.zeros((max_h, params.n_nodes))

        # Roll out max_h steps
        x_pred_traj = rollout_tbptt(x0, u_dummy, params, dt, window=max_h)
        y_pred = np.asarray(eeg_jax(x_pred_traj, eeg_gain))  # (62, max_h)

        y_true = y_data[:, t0 : t0 + max_h]

        for h in horizons:
            yp = y_pred[:, :h]
            yt = y_true[:, :h]

            # Channel-mean MSE
            mse = np.mean((yp - yt) ** 2)
            mses[h].append(mse)

            # Correlation over all flattened samples
            corr = float(np.corrcoef(yp.flatten(), yt.flatten())[0, 1])
            corrs[h].append(corr)

    fig_pred, axes_pred = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")

    h_labels = [str(h) for h in horizons]
    mean_mses = [np.mean(mses[h]) for h in horizons]
    mean_corrs = [np.mean(corrs[h]) for h in horizons]

    axes_pred[0].bar(h_labels, mean_mses, color="#ff7f0e")
    axes_pred[0].set_title("Mean Squared Error vs Horizon")
    axes_pred[0].set_xlabel("Horizon (steps)")
    axes_pred[0].set_ylabel("MSE")

    axes_pred[1].bar(h_labels, mean_corrs, color="#2ca02c")
    axes_pred[1].set_title("Pearson Correlation vs Horizon")
    axes_pred[1].set_xlabel("Horizon (steps)")
    axes_pred[1].set_ylabel("Correlation")
    axes_pred[1].set_ylim(0, 1)

    for ax in axes_pred:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig_pred
    return y_pred, y_true


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Single Trial Traces

    Time-domain comparison for a single prediction trial over the maximum horizon evaluated.
    """)
    return


@app.cell
def _(channel_names, dt_ms, plot_signals, plt, y_pred, y_true):
    _sel = [0, 15, 30, 45, 60]
    _fig_traces, _axes_traces = plt.subplots(1, 2, figsize=(13, 5), layout="constrained", sharex=True, sharey=True)

    plot_signals(
        y_true,
        dt_ms,
        channel_names=channel_names,
        channels_to_plot=_sel,
        title="Ground truth EEG (Short Horizon)",
        color="#1f77b4",
        ax=_axes_traces[0],
    )
    plot_signals(
        y_pred,
        dt_ms,
        channel_names=channel_names,
        channels_to_plot=_sel,
        title="Predicted EEG (Short Horizon)",
        color="#d62728",
        ax=_axes_traces[1],
    )
    _fig_traces
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Multi-Step Prediction Overlay

    Using the `plot_multistep_predictions` function from `utils.plotting` to visualise
    the reduced model's short-horizon predictions overlaid on the continuous ground truth.
    """)
    return


@app.cell
def _(
    components,
    dt,
    eeg_gain,
    eeg_jax,
    jnp,
    np,
    params,
    plot_multistep_predictions,
    rollout_tbptt,
    x_data,
    y_data,
):
    # Project ground truth full states into reduced latent space
    _x_reduced = x_data @ components.T  # (n_steps, 6, N)

    _horizon = 20
    _n_plot_samples = 200
    _C_y = y_data.shape[0]

    # We want predictions for each starting point idx in 0..n_plot_samples
    _Y_pred_abs = []

    # Pre-compute dummy inputs for the prediction horizon
    _u_dummy = jnp.zeros((_horizon, params.n_nodes))

    for _t0 in range(_n_plot_samples):
        _x0 = jnp.asarray(_x_reduced[_t0])
        # Roll out `horizon` steps
        _x_pred_traj = rollout_tbptt(_x0, _u_dummy, params, dt, window=_horizon)
        _y_pred_overlay = np.asarray(eeg_jax(_x_pred_traj, eeg_gain))  # (62, horizon)

        # _y_pred_overlay is (C_y, horizon) -> _y_pred_overlay.T is (horizon, C_y)
        _Y_pred_abs.append(_y_pred_overlay.T)

    _Y_pred_abs = np.array(_Y_pred_abs)
    _Y_val = y_data.T[: _n_plot_samples + _horizon]  # shape (n_samples, 62)

    _fig_overlay, _ = plot_multistep_predictions(
        y_true=_Y_val[:_n_plot_samples],
        y_pred=_Y_pred_abs[:_n_plot_samples],
        dt=dt,
        channels=list(range(min(4, _C_y))),
        stride=_horizon,
        title=f"EEG {_horizon}-Step Ahead Prediction",
    )
    _fig_overlay.savefig("sysid_predictions_overlay.png", dpi=300)

    # Return the current figure so it renders in the notebook
    _fig_overlay
    return


if __name__ == "__main__":
    app.run()
