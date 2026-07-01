import marimo

__generated_with = "0.23.10"
app = marimo.App(
    width="full",
    layout_file="layouts/sweep_analysis.slides.json",
)


@app.cell(hide_code=True)
def imports():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import optuna
    import optuna.visualization as ov
    import pandas as pd
    import seaborn as sns
    import yaml

    from neuro.jansen_rit_jax import enable_x64
    from neuro.nn_training import prepare_datasets, reshape_to_trajectory
    from neuro.prediction import MLPArtifact
    from utils.plotting import plot_multistep_predictions

    enable_x64()

    artifact_base = Path(__file__).parent.parent / "artifacts"
    sweep_dirs = sorted([d.name for d in artifact_base.iterdir() if d.is_dir() and "sweep_nn_predictor" in d.name])
    return (
        MLPArtifact,
        artifact_base,
        mo,
        np,
        optuna,
        ov,
        pd,
        plot_multistep_predictions,
        plt,
        prepare_datasets,
        reshape_to_trajectory,
        sns,
        sweep_dirs,
        yaml,
    )


@app.cell
def ui_sweep_selection(mo, sweep_dirs):
    sweep_dropdown = mo.ui.dropdown(
        options=sweep_dirs, value=sweep_dirs[-1] if sweep_dirs else None, label="Select Sweep Directory"
    )
    sweep_dropdown
    return (sweep_dropdown,)


@app.cell(hide_code=True)
def load_study(artifact_base, mo, optuna, sweep_dropdown):
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

    trials_df
    return study, sweep_path, trials_df


@app.cell
def plot_parallel_coord(ov, study):
    ov.plot_parallel_coordinate(study)
    return


@app.cell
def plot_slice(mo, ov, study):
    fig_slice = ov.plot_slice(study)
    mo.md(f"### Individual Hyperparameter Effects (Slice Plot)\n{mo.as_html(fig_slice)}")
    return


@app.cell
def plot_contour(mo, ov, study):
    fig_contour = ov.plot_contour(study)
    mo.md(f"### Pairwise Hyperparameter Effects (Contour/Heatmap Plot)\n{mo.as_html(fig_contour)}")
    return


@app.cell
def plot_param_importances(mo, ov, study):
    fig_imp = ov.plot_param_importances(study)
    mo.md(f"### Hyperparameter Importances\n{mo.as_html(fig_imp)}")
    return


@app.cell
def ui_trial_selection(mo, trials_df):
    trials_list = trials_df.sort_values("value").index.astype(str).tolist()
    trial_dropdown = mo.ui.dropdown(
        options=trials_list, value=trials_list[0] if trials_list else None, label="Select Trial to Analyze"
    )
    trial_dropdown
    return (trial_dropdown,)


@app.cell
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

    X_full, Y_full, C_y = prepare_datasets(data_files, n_steps_cfg, downsample, n_y, n_u, horizon, projection)
    C_u = (X_full.shape[1] - n_y * C_y) // (n_u + horizon)

    split_idx = int(train_split * len(X_full))
    _X_train, X_val = X_full[:split_idx], X_full[split_idx:]
    _Y_train, Y_val = Y_full[:split_idx], Y_full[split_idx:]

    # Each .npz is one short recording; prepare_datasets concatenates equal-length window
    # blocks per file. The time-series plot must span a single trajectory, else the x-axis
    # (anchor index * dt) runs across all stitched recordings instead of real wall-clock time.
    n_per_traj = X_full.shape[0] // len(data_files)
    _val_offset = (-split_idx) % n_per_traj  # align to the first complete held-out trajectory
    traj_slice = slice(_val_offset, _val_offset + n_per_traj)

    from neuro.prediction import zscore

    y_past = X_val[:, : n_y * C_y]
    u_past = X_val[:, n_y * C_y : n_y * C_y + n_u * C_u]
    u_fut = X_val[:, n_y * C_y + n_u * C_u :]

    # In global scaling, the mean/scale are 1D arrays of size 1. In per-channel, they are size C_y/C_u.
    # We reshape to (batch, time, channel) to broadcast correctly, then flatten again.
    def scale_flat(data, mean, scale, channels):
        shape = data.shape
        reshaped = data.reshape(shape[0], -1, channels)
        return zscore(reshaped, mean, scale).reshape(shape)

    y_past_s = scale_flat(y_past, artifact.y_mean, artifact.y_scale, C_y)
    u_past_s = scale_flat(u_past, artifact.u_mean, artifact.u_scale, C_u)
    u_fut_s = scale_flat(u_fut, artifact.u_mean, artifact.u_scale, C_u)

    X_val_s = np.concatenate([y_past_s, u_past_s, u_fut_s], axis=1)
    return (
        C_y,
        X_val,
        X_val_s,
        Y_val,
        artifact,
        config,
        dt_real,
        horizon,
        model,
        n_per_traj,
        n_y,
        traj_slice,
    )


@app.cell
def eval_and_errors(
    C_y,
    X_val_s,
    Y_val,
    artifact,
    config,
    model,
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

    Y_pred_abs_flat = unscale_flat(Y_pred_s, artifact.y_mean, artifact.y_scale, C_y)

    Y_val_traj = reshape_to_trajectory(Y_val, config.get("model", {}).get("horizon", 5), C_y)
    Y_pred_traj = reshape_to_trajectory(Y_pred_abs_flat, config.get("model", {}).get("horizon", 5), C_y)

    mse_per_channel = np.mean((Y_val_traj - Y_pred_traj) ** 2, axis=(0, 1))
    mae_per_channel = np.mean(np.abs(Y_val_traj - Y_pred_traj), axis=(0, 1))
    return Y_pred_traj, mae_per_channel, mse_per_channel


@app.cell
def plot_channel_errors(mae_per_channel, mse_per_channel, np, pd, plt, sns):
    df_errors = pd.DataFrame(
        {"Channel": np.arange(len(mse_per_channel)), "MSE": mse_per_channel, "MAE": mae_per_channel}
    )

    fig_err, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.barplot(data=df_errors, x="Channel", y="MSE", ax=ax[0])
    ax[0].set_title("MSE per EEG Channel")

    sns.barplot(data=df_errors, x="Channel", y="MAE", ax=ax[1])
    ax[1].set_title("MAE per EEG Channel")

    fig_err.tight_layout()
    plt.gca()
    return


@app.cell
def ui_timeseries_controls(C_y, dt_real, mo, n_per_traj):
    max_time_s = n_per_traj * dt_real

    channel_options = [str(i) for i in range(C_y)]
    ui_channels = mo.ui.multiselect(
        options=channel_options,
        value=[str(i) for i in range(min(4, C_y))],
        label="Channels to Plot",
    )

    ui_time = mo.ui.slider(
        start=0.0,
        stop=max_time_s,
        step=dt_real * 10,
        value=0.0,
        label="Time (s)",
    )

    ui_time_length = mo.ui.slider(
        start=1,
        stop=10,
        step=1,
        value=5,
        label="Time length (s)",
    )

    ui_stride = mo.ui.slider(
        start=1,
        stop=50,
        step=1,
        value=5,
        label="Stride (Anchor Spacing)",
    )

    mo.md(f"#### Time Series Plot Controls\n{mo.vstack([ui_channels, ui_time, ui_time_length, ui_stride])}")
    return ui_channels, ui_stride, ui_time, ui_time_length


@app.cell
def plot_timeseries(
    C_y,
    X_val,
    Y_pred_traj,
    dt_real,
    horizon,
    n_y,
    plot_multistep_predictions,
    plt,
    traj_slice,
    ui_channels,
    ui_stride,
    ui_time,
    ui_time_length,
):
    channels_to_plot = [int(c) for c in ui_channels.value] if ui_channels.value else [0]
    t_start = ui_time.value
    stride = ui_stride.value

    y_true_anchors = X_val[traj_slice, (n_y - 1) * C_y : n_y * C_y]

    _fig_ts, _ = plot_multistep_predictions(
        y_true=y_true_anchors,
        y_pred=Y_pred_traj[traj_slice],
        dt=dt_real,
        channels=channels_to_plot,
        t_start=t_start,
        t_end=t_start + ui_time_length.value,
        stride=stride,
        title=f"EEG {horizon}-Step Ahead Prediction",
    )
    plt.gca()
    return


if __name__ == "__main__":
    app.run()
