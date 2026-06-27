import marimo

__generated_with = "0.23.9"
app = marimo.App(
    width="medium",
    app_title="Sysid short-horizon prediction comparison",
)


@app.cell
def _():
    import glob
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import yaml
    from matplotlib import pyplot as plt

    from neuro import prediction as pred
    from neuro.connectome import compute_gamma, load_connectome
    from neuro.jansen_rit_jax import enable_x64
    from neuro.prediction import JaxSysidPredictor, NNPredictor, PredictionWindow
    from utils.plotting import plot_multistep_predictions
    from utils.processing import compute_psd

    enable_x64()  # float64 parity; before any jnp array
    return (
        JaxSysidPredictor,
        NNPredictor,
        Path,
        PredictionWindow,
        compute_gamma,
        compute_psd,
        glob,
        load_connectome,
        mo,
        np,
        plot_multistep_predictions,
        plt,
        pred,
        yaml,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Short-horizon prediction analysis (one model at a time)

    Evaluate a **single** identified model (JAX sysid or NN — pick below) at predicting
    EEG over **200 ms / 400 ms** on **held-out** seeds of the seizure + PRBS-tES regime,
    against ground truth and a persistence baseline. The model is conditioned on a **400 ms
    window of measured EEG only** (no hidden plant state), then runs open-loop with the
    recorded controls applied through the physical tES projection. Metrics are scale-invariant
    (NRMSE, Pearson correlation, functional connectivity) because the EEG units are arbitrary.

    Run as a script to dump metrics + plots to disk (set `model:` in the config to choose):
    `uv run python notebooks/sysid_prediction_comparison.py --config configs/evaluation/prediction_eval.yaml`
    """)
    return


@app.cell
def _(Path, compute_gamma, load_connectome, mo, np, yaml):
    # Resolve the config from --config (CLI / query param) with a sensible default.
    _c = mo.cli_args().get("config")
    if isinstance(_c, list):
        _c = _c[0] if _c else None
    cfg_path = Path(_c) if _c else Path(__file__).parent.parent / "configs/evaluation/prediction_eval.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())

    data_dt = float(cfg["data_dt"])
    horizons_ms = list(cfg["horizons_ms"])
    horizons_s = [h / 1000.0 for h in horizons_ms]
    context_steps = round(float(cfg["context_ms"]) / 1000.0 / data_dt)
    analysis_dt = float(cfg["analysis_dt"])
    n_windows = int(cfg["n_windows"])
    available_models = list(cfg["artifacts"])
    model_default = str(cfg.get("model", available_models[0]))
    rng = np.random.default_rng(int(cfg.get("seed", 69)))
    out_dir = Path(__file__).parent.parent / cfg.get("out_dir", "results/prediction_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Physical tES projection (the reduced physics models need it to respond to stimulation).
    _conn = load_connectome(speed=float(cfg.get("speed", 50.0)))
    _elec = cfg["target_electrode"]
    _elec_list = [_elec] if isinstance(_elec, str) else list(_elec)
    gamma_full = np.atleast_2d(compute_gamma(_conn.centres, _elec_list, float(cfg["gamma_spread"])))
    channel_labels = [str(c) for c in _conn.channel_labels]

    # Plot defaults (overridable interactively, or via config for script mode).
    plot_channels_default = list(cfg.get("plot_channels", [0, 15, 30, 45]))
    plot_horizon_ms_default = float(cfg.get("plot_horizon_ms", max(horizons_ms)))
    trace_interval_default = list(cfg.get("trace_interval_ms", [0.0, float(max(horizons_ms))]))

    in_nb = mo.running_in_notebook()

    def emit_fig(fig, name):
        if not in_nb:
            path = out_dir / f"{name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  saved {path}")
        return fig

    def emit_md(text):
        if in_nb:
            return mo.md(text)
        print(text)
        return None

    print(f"config={cfg_path}  model={model_default}  horizons_ms={horizons_ms}  n_windows={n_windows}")
    return (
        analysis_dt,
        available_models,
        cfg,
        channel_labels,
        context_steps,
        data_dt,
        emit_fig,
        emit_md,
        gamma_full,
        horizons_s,
        model_default,
        n_windows,
        plot_channels_default,
        plot_horizon_ms_default,
        rng,
        trace_interval_default,
    )


@app.cell
def _(Path, cfg, glob, np):
    # Load held-out test recordings (measured EEG + controls only) and assert disjointness.
    _base = Path(__file__).parent.parent
    train_files = set(glob.glob(str(_base / cfg["train_sims_glob"])))
    test_files = [str(_base / p) for p in cfg["test_sims"]]
    overlap = set(test_files) & train_files
    if overlap:
        msg = f"test/train seed overlap: {sorted(overlap)}"
        raise ValueError(msg)

    recordings = []
    for _p in test_files:
        with np.load(_p) as _d:
            _y = np.asarray(_d["universal_y_mea"], dtype=np.float64).T  # (62, T)
            _u = np.asarray(_d["universal_u"], dtype=np.float64)  # (T, n_elec)
        recordings.append((_y, _u))
    print(f"loaded {len(recordings)} test recording(s); train/test disjoint OK")
    return (recordings,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model to evaluate (one at a time — changing this re-runs the evaluation)
    """)
    return


@app.cell
def _(available_models, mo, model_default):
    # Single-model selector: this drives the evaluation, so only one model runs per pass.
    ui_model = mo.ui.dropdown(options=available_models, value=model_default, label="model to evaluate")
    ui_model
    return (ui_model,)


@app.cell
def _(JaxSysidPredictor, NNPredictor, cfg, gamma_full, ui_model):
    # Build ONLY the selected model's predictor from its train-only artifact.
    arts = cfg["artifacts"]
    builders = {
        "jax_sysid": lambda: JaxSysidPredictor.load(arts["jax_sysid"], gamma_full),
        "nn": lambda: NNPredictor.load(arts["nn"]),
    }
    _sel = ui_model.value
    predictors = {}
    try:
        predictors[_sel] = builders[_sel]()
        print(f"loaded predictor: {_sel}  (dt={predictors[_sel].dt:g}s)")
    except (FileNotFoundError, OSError, KeyError) as _e:
        print(f"SKIP {_sel}: artifact not loadable ({_e})")
    return (predictors,)


@app.cell
def _(
    PredictionWindow,
    analysis_dt,
    context_steps,
    data_dt,
    horizons_s,
    n_windows,
    np,
    pred,
    predictors,
    recordings,
    rng,
):
    # Core evaluation: sample windows, predict open-loop once per window at the max horizon,
    # then score each sub-horizon against the true future on the common analysis grid.
    max_h_s = max(horizons_s)
    max_h_data = round(max_h_s / data_dt)
    transient = round(1.0 / data_dt)  # skip the 1 s zero-stim lead so stimulation is active
    a_ds = round(analysis_dt / data_dt)  # data -> analysis decimation

    names = [*predictors.keys(), "persistence"]
    results = {
        m: {h: {"nrmse": [], "corr": [], "fc_err": [], "fc_pcorr": [], "preds": [], "trues": []} for h in horizons_s}
        for m in names
    }
    example = {}

    def _store(slot, ph, yt, fc_t):
        slot["nrmse"].append(pred.nrmse(ph, yt))
        slot["corr"].append(pred.pearson_corr(ph, yt))
        fc_p = pred.fc(ph)
        slot["fc_err"].append(pred.fc_error(fc_p, fc_t))
        slot["fc_pcorr"].append(pred.fc_pattern_corr(fc_p, fc_t))
        slot["preds"].append(ph)
        slot["trues"].append(yt)

    for _y, _u in recordings:
        t_len = _y.shape[1]
        lo, hi = max(context_steps, transient), t_len - max_h_data
        if hi <= lo:
            continue
        for _t0 in rng.integers(lo, hi, size=n_windows):
            t0 = int(_t0)
            window = PredictionWindow(
                y_ctx=_y[:, t0 - context_steps : t0],
                u_ctx=_u[t0 - context_steps : t0],
                u_future=_u[t0 : t0 + max_h_data],
                dt_data=data_dt,
            )
            full = {}
            for m, p in predictors.items():
                try:
                    pf = np.asarray(p.predict(window, max_h_s))
                    full[m] = pf if np.all(np.isfinite(pf)) else None
                except Exception as _e:  # noqa: BLE001
                    print(f"  predict failed [{m} @ t0={t0}]: {_e}")
                    full[m] = None

            for h_s in horizons_s:
                n_data = round(h_s / data_dt)
                yt = _y[:, t0 : t0 + n_data][:, ::a_ds]
                n_a = yt.shape[1]
                fc_t = pred.fc(yt)
                for m, p in predictors.items():
                    pf = full[m]
                    if pf is None:
                        continue
                    ph = pred.decimate_to(pf[:, : round(h_s / p.dt)], p.dt, analysis_dt)[:, :n_a]
                    if ph.shape[1] == n_a:
                        _store(results[m][h_s], ph, yt, fc_t)
                _store(
                    results["persistence"][h_s], pred.persistence_baseline(window, h_s, analysis_dt)[:, :n_a], yt, fc_t
                )

            if not example:
                example = {
                    "true": _y[:, t0 : t0 + max_h_data][:, ::a_ds],
                    "preds": {
                        m: pred.decimate_to(full[m], p.dt, analysis_dt)
                        for m, p in predictors.items()
                        if full[m] is not None
                    },
                }
    print(f"evaluated {sum(len(results['persistence'][h]['nrmse']) for h in horizons_s)} method-windows")
    return example, names, results


@app.cell
def _(np):
    # Colour map + a nan-safe mean used by the summary cells.
    colors = {
        "jax_sysid": "#d62728",
        "nn": "#9467bd",
        "persistence": "#7f7f7f",
    }

    def safe_mean(values):
        arr = np.asarray(values, dtype=np.float64)
        return float(np.nanmean(arr)) if arr.size else float("nan")

    return colors, safe_mean


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Display controls — channels, horizon, and the trace time window
    """)
    return


@app.cell
def _(
    channel_labels,
    horizons_s,
    mo,
    plot_channels_default,
    plot_horizon_ms_default,
    trace_interval_default,
):
    _chan_value = [channel_labels[i] for i in plot_channels_default if 0 <= i < len(channel_labels)] or channel_labels[
        :1
    ]
    _h_options = {f"{_h * 1000:.0f} ms": _h for _h in horizons_s}
    _h_label = next((lbl for lbl, _h in _h_options.items() if abs(_h * 1000 - plot_horizon_ms_default) < 1e-6), None)
    _h_label = _h_label or next(iter(_h_options))
    _max_ms = max(horizons_s) * 1000.0
    _t_lo = max(0.0, float(trace_interval_default[0]))
    _t_hi = min(_max_ms, float(trace_interval_default[1]))

    ui_channels = mo.ui.multiselect(options=channel_labels, value=_chan_value, label="channels (traces)")
    ui_horizon = mo.ui.dropdown(options=_h_options, value=_h_label, label="horizon (FC / PSD)")
    ui_trace = mo.ui.range_slider(start=0.0, stop=_max_ms, step=_max_ms / 40.0, value=[_t_lo, _t_hi], label="trace ms")
    mo.vstack([mo.hstack([ui_horizon, ui_trace]), ui_channels])
    return ui_channels, ui_horizon, ui_trace


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Error vs lead time (NRMSE), one panel per horizon
    """)
    return


@app.cell
def _(
    analysis_dt,
    colors,
    emit_fig,
    horizons_s,
    names,
    np,
    plt,
    pred,
    results,
):
    _fig, _axes = plt.subplots(1, len(horizons_s), figsize=(6.5 * len(horizons_s), 4.2), layout="constrained")
    _axes = np.atleast_1d(_axes)
    for _ax, _h in zip(_axes, horizons_s, strict=True):
        for _m in names:
            _preds, _trues = results[_m][_h]["preds"], results[_m][_h]["trues"]
            if not _preds:
                continue
            _curve = pred.error_vs_leadtime(_preds, _trues)
            _t = (np.arange(_curve.shape[0]) + 1) * analysis_dt * 1000.0
            _ax.plot(_t, _curve, label=_m, color=colors.get(_m), linewidth=2.0, alpha=0.9)
        _ax.set_title(f"{_h * 1000:.0f} ms horizon", fontsize=12, fontweight="bold")
        _ax.set_xlabel("lead time (ms)")
        _ax.set_ylabel("NRMSE")
        _ax.spines[["top", "right"]].set_visible(False)
        _ax.grid(visible=True, linestyle="--", alpha=0.3)
        _ax.legend(framealpha=0.9)
    emit_fig(_fig, "error_vs_leadtime")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Aggregate NRMSE and correlation per method
    """)
    return


@app.cell
def _(emit_fig, horizons_s, names, np, plt, results, safe_mean):
    _methods = [m for m in names if any(results[m][h]["nrmse"] for h in horizons_s)]
    _x = np.arange(len(_methods))
    _w = 0.8 / max(1, len(horizons_s))
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 4.2), layout="constrained")
    for _j, _h in enumerate(horizons_s):
        _nrmse = [safe_mean(results[m][_h]["nrmse"]) for m in _methods]
        _corr = [safe_mean(results[m][_h]["corr"]) for m in _methods]
        _off = (_j - (len(horizons_s) - 1) / 2) * _w
        _axes[0].bar(_x + _off, _nrmse, _w, label=f"{_h * 1000:.0f} ms")
        _axes[1].bar(_x + _off, _corr, _w, label=f"{_h * 1000:.0f} ms")
    for _ax, _title, _yl in (
        (_axes[0], "Mean NRMSE (lower better)", "NRMSE"),
        (_axes[1], "Mean correlation (higher better)", "Pearson r"),
    ):
        _ax.set_xticks(_x)
        _ax.set_xticklabels(_methods)
        _ax.set_title(_title, fontsize=12, fontweight="bold")
        _ax.set_ylabel(_yl)
        _ax.spines[["top", "right"]].set_visible(False)
        _ax.grid(axis="y", linestyle="--", alpha=0.3)
        _ax.legend()
    emit_fig(_fig, "aggregate_metrics")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Functional connectivity (longest horizon)

    Channel-by-channel correlation, averaged over test windows: ground truth vs each
    model, plus the signed difference.
    """)
    return


@app.cell
def _(emit_fig, np, plt, predictors, results, ui_horizon):
    _h = ui_horizon.value
    _models = [m for m in predictors if results[m][_h]["preds"]]
    _fc_true = np.mean([np.corrcoef(yt) for yt in results["persistence"][_h]["trues"]], axis=0)
    _ncol = len(_models) + 1
    _fig, _axes = plt.subplots(1, _ncol, figsize=(4.6 * _ncol, 4.2), layout="constrained")
    _axes = np.atleast_1d(_axes)
    _im = _axes[0].imshow(_fc_true, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    _axes[0].set_title("ground truth FC", fontsize=12, fontweight="bold")
    _fig.colorbar(_im, ax=_axes[0], fraction=0.046, pad=0.04)
    for _ax, _m in zip(_axes[1:], _models, strict=True):
        _fc_m = np.mean([np.corrcoef(p) for p in results[_m][_h]["preds"]], axis=0)
        _diff = _fc_true - _fc_m
        _im = _ax.imshow(_diff, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        _ax.set_title(f"GT - {_m} FC", fontsize=12, fontweight="bold")
        _fig.colorbar(_im, ax=_ax, fraction=0.046, pad=0.04)
    emit_fig(_fig, "functional_connectivity")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Example single-window traces (longest horizon)
    """)
    return


@app.cell
def _(
    analysis_dt,
    channel_labels,
    colors,
    emit_fig,
    example,
    np,
    plot_multistep_predictions,
    plt,
    ui_channels,
    ui_trace,
):
    _chans = [channel_labels.index(c) for c in ui_channels.value] or [0]
    _lo_s, _hi_s = float(ui_trace.value[0]) / 1000.0, float(ui_trace.value[1]) / 1000.0
    _fig, _axes = plt.subplots(len(_chans), 1, figsize=(11, 2.0 * len(_chans) + 1), sharex=True, layout="constrained")
    _axes = np.atleast_1d(_axes)
    if example:
        _true = example["true"]
        _n_samples = _true.shape[1]
        _C = _true.shape[0]

        for _i, (_m, _pf) in enumerate(example["preds"].items()):
            _horizon = _pf.shape[1]
            _y_pred = np.full((_n_samples, _horizon, _C), np.nan)
            _y_pred[0, :_horizon, :] = _pf.T

            plot_multistep_predictions(
                y_true=_true.T,
                y_pred=_y_pred,
                dt=analysis_dt,
                channels=_chans,
                channel_names=channel_labels,
                anchors=[0],
                t_start=_lo_s,
                t_end=_hi_s,
                pred_kwargs={"color": colors.get(_m), "label": _m},
                true_kwargs={"label": "truth"} if _i == 0 else {"label": None},
                title="Example single-window traces (longest horizon)",
                axes=_axes,
            )

        handles, labels = _axes[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles, strict=False))
        # Ensure 'truth' is first in legend if present
        ordered_labels = (
            ["truth"] + [lbl for lbl in by_label if lbl != "truth"] if "truth" in by_label else by_label.keys()
        )
        ordered_handles = [by_label[lbl] for lbl in ordered_labels]
        _axes[0].legend(ordered_handles, ordered_labels, ncol=4, framealpha=0.9)

    emit_fig(_fig, "example_traces")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Aggregate spectral shape

    Channel-mean Welch PSD of the test windows (concatenated) vs each model. Short
    windows give coarse low-frequency resolution -- read this as a gross spectral-shape
    check, not a precise PSD.
    """)
    return


@app.cell
def _(
    analysis_dt,
    colors,
    compute_psd,
    emit_fig,
    np,
    plt,
    predictors,
    results,
    ui_horizon,
):
    _h = ui_horizon.value
    _dt_ms = analysis_dt * 1000.0
    _fig, _ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    _true_cat = np.concatenate(results["persistence"][_h]["trues"], axis=1)
    _f, _p = compute_psd(_true_cat, _dt_ms)
    _ax.semilogy(_f, _p.mean(axis=0), color="#1f77b4", linewidth=2.2, label="truth")
    for _m in predictors:
        if not results[_m][_h]["preds"]:
            continue
        _cat = np.concatenate(results[_m][_h]["preds"], axis=1)
        _fm, _pm = compute_psd(_cat, _dt_ms)
        _ax.semilogy(_fm, _pm.mean(axis=0), color=colors.get(_m), linewidth=1.8, alpha=0.85, label=_m)
    _ax.set_xlabel("frequency (Hz)")
    _ax.set_ylabel("power (a.u.)")
    _ax.set_title("Channel-mean PSD (aggregate)", fontsize=12, fontweight="bold")
    _ax.spines[["top", "right"]].set_visible(False)
    _ax.grid(visible=True, linestyle="--", alpha=0.3)
    _ax.legend()
    emit_fig(_fig, "psd")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Metrics table
    """)
    return


@app.cell
def _(emit_md, horizons_s, names, results, safe_mean):
    _header = "| method | horizon | NRMSE | corr | FC error | FC pattern r |\n| --- | --- | --- | --- | --- | --- |"
    _rows = []
    for _m in names:
        for _h in horizons_s:
            _slot = results[_m][_h]
            if not _slot["nrmse"]:
                continue
            _rows.append(
                f"| {_m} | {_h * 1000:.0f} ms | {safe_mean(_slot['nrmse']):.3f} | "
                f"{safe_mean(_slot['corr']):.3f} | {safe_mean(_slot['fc_err']):.4f} | "
                f"{safe_mean(_slot['fc_pcorr']):.3f} |"
            )
    emit_md("\n".join([_header, *_rows]))
    return


if __name__ == "__main__":
    app.run()
