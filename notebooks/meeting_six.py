import marimo

__generated_with = "0.23.10"
app = marimo.App(
    width="full",
    app_title="Supervisor Update",
    layout_file="layouts/supervisor_update.slides.json",
)


@app.cell(hide_code=True)
def imports():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import optuna
    import optuna.visualization as ov
    import yaml

    from neuro.nn_training import prepare_datasets, reshape_to_trajectory
    from neuro.prediction import MLPArtifact
    from utils.plotting import plot_multistep_predictions

    notebook_dir = Path(__file__).parent
    artifact_base = notebook_dir.parent / "artifacts"
    sweep_dirs = sorted([d.name for d in artifact_base.iterdir() if d.is_dir() and "sweep_nn_predictor" in d.name])
    return (
        MLPArtifact,
        artifact_base,
        mo,
        notebook_dir,
        np,
        optuna,
        ov,
        plot_multistep_predictions,
        plt,
        prepare_datasets,
        reshape_to_trajectory,
        sweep_dirs,
        yaml,
    )


@app.cell(hide_code=True)
def slide_title(mo):
    mo.md(r"""
    # Closed-loop Neurostimulation
    ## Sixth meeting

    **Max Frankenberg** · 30 June 2026
    """)
    return


@app.cell(hide_code=True)
def slide_model(mo, notebook_dir):
    _img_path = notebook_dir / "assets" / "mlp_diagram.png"
    _diagram = (
        mo.image(str(_img_path), width=320)
        if _img_path.exists()
        else mo.md("> *[MLP diagram — add `notebooks/assets/mlp_diagram.png`]*")
    )

    _header = mo.hstack(
        [mo.md("## Model & training setup"), _diagram],
        justify="space-between",
        align="start",
    )

    _col_data = mo.md(
        r"""
    ### Data
    - $100$ excited trajectories
    - $\mathbf{u}(t)$ — 2 stimulation electrodes (**CP5**, **T7**)
    - $\mathbf{y}(t)$ — 62 EEG channels
    - integration step $dt = 0.1\,\text{ms}$

    ### Model: MLP
    - ReLU activation
    - downsample $\times 100$: model step $\Delta t = 100\,dt = 10\,\text{ms}$
    - $\hat{\mathbf{y}}(t+\Delta t)=\hat{\mathbf{y}}_{t+1} = \mathrm{MLP}\big(\mathbf{y}_t,\, \dots,\, \mathbf{y}_{t-n_y+1},\; \mathbf{u}_t,\, \dots,\, \mathbf{u}_{t-n_u+1}\big)$
    - sizes: $n_\text{in} = n_u\,c_u + n_y\,c_y,\ \; n_\text{out} = c_y$
        - Example: $n_y{=}15,\; n_u{=}7,\; c_u{=}2,\; c_y{=}62$
        - $\Rightarrow n_\text{in}=944,\; n_\text{out}=62$
    """
    )

    _col_training = mo.md(
        r"""
    ### Training
    Multi-step predictor, horizon $H$ (e.g. $H = 20$).

    One-step loss (first rollout step):
    $$\mathcal{L}_1 = \big\lVert \hat{\mathbf{y}}_{t+1} - \mathbf{y}_{t+1} \big\rVert^2$$

    Multi-step loss (full $H$-step rollout):
    $$\mathcal{L}_H = \tfrac{1}{H}\textstyle\sum_{h=1}^{H} \big\lVert \hat{\mathbf{y}}_{t+h} - \mathbf{y}_{t+h} \big\rVert^2$$

    Blend, anneal $\alpha: 1 \to 0$ (start one-step, end full multi-step):
    $$\mathcal{L} = \alpha\,\mathcal{L}_1 + (1-\alpha)\,\mathcal{L}_H$$
    """
    )

    _body = mo.hstack([_col_data, _col_training], widths=[1, 1], gap=2, align="start")
    mo.vstack([_header, _body])
    return


@app.cell(hide_code=True)
def hpo_controls(mo, sweep_dirs):
    sweep_dropdown = mo.ui.dropdown(
        options=sweep_dirs, value=sweep_dirs[-1] if sweep_dirs else None, label="Sweep directory"
    )
    return (sweep_dropdown,)


@app.cell(hide_code=True)
def slide_hpo(artifact_base, mo, optuna, ov, sweep_dropdown):
    mo.stop(sweep_dropdown.value is None, "Please select a sweep directory.")
    sweep_path = artifact_base / sweep_dropdown.value
    db_path = sweep_path / "nn_predictor_sweep.db"
    mo.stop(not db_path.exists(), f"No sweep DB found at {db_path}")

    storage_url = f"sqlite:///{db_path.resolve()}"
    study = optuna.load_study(study_name="nn_predictor_sweep", storage=storage_url)
    trials_df = study.trials_dataframe()
    trials_df = trials_df[trials_df.state == "COMPLETE"].drop(
        columns=["number", "datetime_start", "datetime_complete", "state"]
    )

    fig_slice = ov.plot_slice(study)

    mo.vstack(
        [
            mo.md(
                r"""
    ## Hyperparameter search
    - search over: $n_y$, $n_u$, `depth`, `hidden_size`, learning rate, weight decay, etc.
    - **Takeaway:** many setups work. `depth = 0` (a linear autoregressive model) also viable option.
    """
            ),
            mo.as_html(fig_slice),
            sweep_dropdown,
        ]
    )
    return sweep_path, trials_df


@app.cell(hide_code=True)
def pred_trial_controls(mo, trials_df):
    trials_list = trials_df.sort_values("value").index.astype(str).tolist()
    trial_dropdown = mo.ui.dropdown(options=trials_list, value=trials_list[0] if trials_list else None, label="Trial")
    return (trial_dropdown,)


@app.cell(hide_code=True)
def load_trial_data(
    MLPArtifact,
    mo,
    np,
    prepare_datasets,
    sweep_path,
    trial_dropdown,
    yaml,
):
    mo.stop(trial_dropdown.value is None, "Please select a trial.")
    trial_num = trial_dropdown.value
    trial_dir = sweep_path / f"trial_{trial_num}"

    config_path = trial_dir / "trial_config.yaml"
    mo.stop(not config_path.exists(), "Config not found for this trial.")
    with config_path.open() as f:
        config = yaml.safe_load(f)

    artifact_path = trial_dir / "model.eqx"
    mo.stop(not artifact_path.exists(), "Model artifact not found for this trial.")

    artifact = MLPArtifact.load(artifact_path)
    model = artifact.model

    sim_cfg = config.get("simulation", {})
    downsample = sim_cfg.get("downsample", 1)
    n_steps_cfg = sim_cfg.get("n_steps", 2000)
    dt_base = sim_cfg.get("dt", 1e-4)
    dt_real = dt_base * downsample

    data_path_str = sim_cfg.get("data_path", "data/jansen_rit_healthy")
    data_path = sweep_path.parent.parent / data_path_str

    data_files = sorted([str(p) for p in data_path.glob("*.npz")])
    mo.stop(not data_files, f"No .npz data files found in {data_path}")

    model_cfg = config.get("model", {})
    n_y = model_cfg.get("n_y", 5)
    n_u = model_cfg.get("n_u", 5)
    horizon = model_cfg.get("horizon", 5)

    _train_cfg = config.get("training", {})
    train_split = float(_train_cfg.get("train_split", 0.8))
    _train_cfg.get("global_scaling", True)

    projection = (artifact.latent_basis, artifact.latent_mean) if artifact.latent_basis is not None else None

    X_full, Y_full, n_channels = prepare_datasets(data_files, n_steps_cfg, downsample, n_y, n_u, horizon, projection)
    n_controls = (X_full.shape[1] - n_y * n_channels) // (n_u + horizon)

    split_idx = int(train_split * len(X_full))
    _X_train, X_val = X_full[:split_idx], X_full[split_idx:]
    _Y_train, _Y_val = Y_full[:split_idx], Y_full[split_idx:]

    from neuro.prediction import zscore

    y_past = X_val[:, : n_y * n_channels]
    u_past = X_val[:, n_y * n_channels : n_y * n_channels + n_u * n_controls]
    u_future = X_val[:, n_y * n_channels + n_u * n_controls :]

    # In global scaling, the mean/scale are 1D arrays of size 1. In per-channel, they are size n_channels/n_controls.
    # We reshape to (batch, time, channel) to broadcast correctly, then flatten again.
    def scale_flat(data, mean, scale, channels):
        shape = data.shape
        reshaped = data.reshape(shape[0], -1, channels)
        return zscore(reshaped, mean, scale).reshape(shape)

    y_past_s = scale_flat(y_past, artifact.y_mean, artifact.y_scale, n_channels)
    u_past_s = scale_flat(u_past, artifact.u_mean, artifact.u_scale, n_controls)
    u_future_s = scale_flat(u_future, artifact.u_mean, artifact.u_scale, n_controls)

    X_val_s = np.concatenate([y_past_s, u_past_s, u_future_s], axis=1)
    return (
        X_val,
        X_val_s,
        artifact,
        config,
        dt_real,
        horizon,
        model,
        n_channels,
        n_y,
    )


@app.cell(hide_code=True)
def eval_predictions(
    X_val_s,
    artifact,
    config,
    model,
    n_channels,
    np,
    reshape_to_trajectory,
):
    import jax.numpy as jnp

    from neuro.nn_training import predict_batch
    from neuro.prediction import unzscore

    Y_pred_s = np.asarray(predict_batch(model, jnp.asarray(X_val_s)))

    def unscale_flat(data, mean, scale, channels):
        shape = data.shape
        reshaped = data.reshape(shape[0], -1, channels)
        return unzscore(reshaped, mean, scale).reshape(shape)

    Y_pred_unscaled_flat = unscale_flat(Y_pred_s, artifact.y_mean, artifact.y_scale, n_channels)

    Y_pred_traj = reshape_to_trajectory(Y_pred_unscaled_flat, config.get("model", {}).get("horizon", 5), n_channels)
    return (Y_pred_traj,)


@app.cell(hide_code=True)
def pred_slider_controls(Y_pred_traj, dt_real, mo, n_channels):
    max_time_s = len(Y_pred_traj) * dt_real
    channel_options = [str(i) for i in range(n_channels)]
    ui_channels = mo.ui.multiselect(
        options=channel_options, value=[str(i) for i in range(min(4, n_channels))], label="Channels"
    )
    ui_time = mo.ui.slider(start=0.0, stop=max_time_s, step=dt_real * 10, value=0.0, label="Time (s)")
    ui_time_length = mo.ui.slider(start=1, stop=10, step=1, value=5, label="Window (s)")
    ui_stride = mo.ui.slider(start=1, stop=50, step=1, value=5, label="Stride")
    return ui_channels, ui_stride, ui_time, ui_time_length


@app.cell(hide_code=True)
def slide_pred(
    X_val,
    Y_pred_traj,
    dt_real,
    horizon,
    mo,
    n_channels,
    n_y,
    plot_multistep_predictions,
    plt,
    trial_dropdown,
    ui_channels,
    ui_stride,
    ui_time,
    ui_time_length,
):
    import io

    channels_to_plot = [int(c) for c in ui_channels.value] if ui_channels.value else [0]
    y_true_anchors = X_val[:, (n_y - 1) * n_channels : n_y * n_channels]
    _fig_ts, _ = plot_multistep_predictions(
        y_true=y_true_anchors,
        y_pred=Y_pred_traj,
        dt=dt_real,
        channels=channels_to_plot,
        t_start=ui_time.value,
        t_end=ui_time.value + ui_time_length.value,
        stride=ui_stride.value,
        title=f"EEG {horizon}-Step Ahead Prediction",
    )
    _buf = io.BytesIO()
    _fig_ts.savefig(_buf, format="png", bbox_inches="tight")
    plt.close(_fig_ts)

    mo.vstack(
        [
            mo.md(
                "## Prediction quality\n\n20-step-ahead EEG prediction — true vs. predicted on held-out testing data."
            ),
            mo.hstack([trial_dropdown, ui_channels], gap=2, justify="start"),
            mo.hstack([ui_time, ui_time_length, ui_stride], gap=2, justify="start"),
            mo.image(_buf.getvalue(), width=1000),
        ]
    )
    return


@app.cell(hide_code=True)
def slide_wrap(mo):
    mo.md(r"""
    ## Problems
    - **Activation:** ReLU fits well, but MPC needs differentiable dynamics: (**tanh** / **softplus**).
    - **Multiple-shooting:** Each shooting node lifts the input window into a block of decision variables
        $$ \boldsymbol{\xi} = \big[\boldsymbol{\xi}_1, ..., \boldsymbol{\xi}_H \big] \quad \text{with} \quad \boldsymbol{\xi}_k = \big[\, \mathbf{y}_t,\; \dots,\; \mathbf{y}_{t-n_y+1},\;\; \mathbf{u}_t,\; \dots,\; \mathbf{u}_{t-n_u+1} \,\big],\quad n_\xi = H (n_y\,c_y + n_u\,c_u) $$
        with $c_y = 62$ and $n_y{=}15$ we get $n_\xi \approx 20.000$
    - **Direction:** project EEG into a low-dim space using PCA ($c_y: 62 \to k$) to shrink $n_\xi$

    ## Outlook
    - Baseline **continuous MPC**: quadratic / $L_1$-norm cost
        - Define **constraints**
        - integrate the **prediction model**
    - Reading:
        - linear ARX and NNs for MPC
        - **cascaded / 2-DoF** structure


    - **Exam: 13 July** — possibly ~1 week break
    """)
    return


if __name__ == "__main__":
    app.run()
