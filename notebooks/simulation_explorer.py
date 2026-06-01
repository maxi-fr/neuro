import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium", app_title="Whole-Brain Simulation Explorer")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    from neuro import (
        FHNOutput,
        NativeFHNDynamics,
        TVBFHNDynamics,
        TVBFHNOutput,
        plot_fourier,
        plot_signals,
    )

    return (
        FHNOutput,
        NativeFHNDynamics,
        TVBFHNDynamics,
        TVBFHNOutput,
        mo,
        np,
        plot_fourier,
        plot_signals,
    )


@app.cell
def _(mo):
    mo.md("""
    # 🧠 Whole-Brain Simulation & EEG Explorer

    Welcome to the simulation explorer notebook! This interactive tool allows you to simulate whole-brain network models of FitzHugh-Nagumo dynamics, project the simulated neural activity onto synthetic EEG sensors, and analyze the resulting signals in both time and frequency domains.

    Use the controls below to configure and run the simulation.
    """)
    return


@app.cell
def _(mo):
    # 1. Choose the plant model
    plant_select = mo.ui.dropdown(
        options=["NativeFHNDynamics", "TVBFHNDynamics"],
        value="NativeFHNDynamics",
        label="Plant Model",
    )

    # 2. Simulation duration in milliseconds
    duration_slider = mo.ui.slider(
        start=50,
        stop=1000,
        step=50,
        value=200,
        label="Duration (ms)",
    )

    # 3. Noise controls -- the FHN and TVB plants take *different* noise
    # parameters, so they get independent sliders. sigma_ou is an
    # Ornstein-Uhlenbeck (colored, low-pass) amplitude; nsig is a white
    # additive-noise intensity (per-step kick = sqrt(2*nsig)). They are not
    # physically equivalent; a rough per-step-variance match is
    # nsig ~= dt*sigma_ou**2*tau_ou/4.
    sigma_ou_slider = mo.ui.slider(
        start=0.0,
        stop=0.1,
        step=0.01,
        value=0.05,
        label="FHN Noise (sigma_ou, OU)",
    )

    nsig_slider = mo.ui.slider(
        start=0.0,
        stop=0.02,
        step=0.001,
        value=0.005,
        label="TVB Noise (nsig, white)",
    )

    # 4. Simulation RNG Seed
    seed_input = mo.ui.number(
        start=0,
        stop=1000,
        value=42,
        label="RNG Seed",
    )

    # 5. Number of channels to plot in individual plots
    channels_slider = mo.ui.slider(
        start=1,
        stop=10,
        step=1,
        value=5,
        label="Channels to Display",
    )

    # 6. Max frequency to plot in spectra
    max_freq_slider = mo.ui.slider(
        start=10,
        stop=150,
        step=5,
        value=50,
        label="Max Frequency (Hz)",
    )

    # 7. Stacked vs overlaid selector
    stacked_select = mo.ui.checkbox(
        value=True,
        label="Waterfall/Stacked Stacked Channels",
    )

    # 8. Spectral mode selector
    mode_select = mo.ui.dropdown(
        options=["amplitude", "power"],
        value="power",
        label="Fourier Spectrum Mode",
    )

    # 9. Normalize spectra (compare shape, not absolute magnitude)
    normalize_select = mo.ui.checkbox(
        value=True,
        label="Normalize Spectra",
    )

    # Layout the controls nicely in columns
    mo.md(
        f"### ⚙️ Simulation & Plotting Controls\n{
            mo.as_html(
                mo.hstack(
                    [
                        mo.vstack([plant_select, seed_input, duration_slider, sigma_ou_slider, nsig_slider]),
                        mo.vstack([channels_slider, max_freq_slider, stacked_select, mode_select, normalize_select]),
                    ],
                    justify='start',
                    gap=4,
                )
            )
        }"
    )
    return (
        channels_slider,
        duration_slider,
        max_freq_slider,
        mode_select,
        normalize_select,
        nsig_slider,
        plant_select,
        seed_input,
        sigma_ou_slider,
        stacked_select,
    )


@app.cell
def _(
    FHNOutput,
    NativeFHNDynamics,
    TVBFHNDynamics,
    TVBFHNOutput,
    duration_slider,
    mo,
    np,
    nsig_slider,
    plant_select,
    seed_input,
    sigma_ou_slider,
):
    plant_name = plant_select.value
    seed = int(seed_input.value)
    duration = float(duration_slider.value)

    # Each plant takes its own noise parameter (see controls cell), and is paired
    # with the matching EEG Output component.
    if plant_name == "NativeFHNDynamics":
        noise_param = f"sigma_ou = {sigma_ou_slider.value}"
        dynamics = NativeFHNDynamics(sigma_ou=float(sigma_ou_slider.value), seed=seed)
        output = FHNOutput(dt=dynamics.dt, n_nodes=dynamics.n_nodes)
    else:
        noise_param = f"nsig = {nsig_slider.value}"
        dynamics = TVBFHNDynamics(nsig=float(nsig_slider.value), seed=seed)
        output = TVBFHNOutput(dt=dynamics.dt, n_nodes=dynamics.n_nodes)

    # Advance the dynamics one dt at a time, projecting each state to EEG.
    dt_ms = dynamics.dt
    n_steps = round(duration / dt_ms)
    n_nodes = dynamics.n_nodes
    u = np.zeros((dynamics.b.shape[1], 1))

    activity_cols = []
    eeg_cols = []
    t = 0.0
    for _ in range(n_steps):
        x_raw, _ = dynamics.evaluate(t, u)
        y_raw, _ = output.evaluate(t, x_raw, u)
        activity_cols.append(np.atleast_1d(x_raw)[:n_nodes].reshape(-1, 1))
        eeg_cols.append(np.atleast_1d(y_raw).reshape(-1, 1))
        t += dt_ms

    activity = np.hstack(activity_cols)
    eeg = np.hstack(eeg_cols)

    status_message = mo.md(
        f"""
        **Simulation Status**: Successfully simulated **{plant_name}** for **{duration} ms** with **{noise_param}** (RNG seed = **{seed}**).
        * Output Activity Shape: `{activity.shape}` (nodes, samples)
        * Output EEG Shape: `{eeg.shape}` (sensors, samples)
        * Integration dt: `{dt_ms} ms` (sampling rate: `{1000.0 / dt_ms:.1f} Hz`)
        """
    )
    return activity, dt_ms, eeg, status_message


@app.cell
def _(status_message):
    status_message
    return


@app.cell
def _(mo):
    mo.md("""
    ## 📈 Detached Signal and Fourier Analysis
    """)
    return


@app.cell
def _(
    activity,
    channels_slider,
    dt_ms,
    eeg,
    max_freq_slider,
    mo,
    mode_select,
    normalize_select,
    plot_fourier,
    plot_signals,
    stacked_select,
):
    _n_ch = int(channels_slider.value)
    _max_f = float(max_freq_slider.value)
    is_stacked = bool(stacked_select.value)
    spec_mode = mode_select.value
    normalize_spec = bool(normalize_select.value)

    # Interactive tabs to view Activity or EEG separately
    nodes_tab = mo.vstack(
        [
            mo.md("### Brain Region Activity (V)"),
            plot_signals(
                activity,
                dt_ms=dt_ms,
                channels_to_plot=list(range(min(_n_ch, activity.shape[0]))),
                stacked=is_stacked,
                title="Brain Region Oscillations over Time",
                color="#1f77b4",
            )[0],
            plot_fourier(
                activity,
                dt_ms=dt_ms,
                mode=spec_mode,
                channels_to_plot=list(range(min(_n_ch, activity.shape[0]))),
                plot_mean=False,
                max_freq=_max_f,
                normalize=normalize_spec,
            )[0],
        ]
    )

    eeg_tab = mo.vstack(
        [
            mo.md("### Synthetic EEG Sensors (Leadfield @ V)"),
            plot_signals(
                eeg,
                dt_ms=dt_ms,
                channels_to_plot=list(range(min(_n_ch, eeg.shape[0]))),
                stacked=is_stacked,
                title="EEG Scalp Projection over Time",
                color="#ff7f0e",
            )[0],
            plot_fourier(
                eeg,
                dt_ms=dt_ms,
                mode=spec_mode,
                channels_to_plot=list(range(min(_n_ch, eeg.shape[0]))),
                plot_mean=False,
                max_freq=_max_f,
                normalize=normalize_spec,
            )[0],
        ]
    )

    mo.ui.tabs({"Brain Region Activity": nodes_tab, "EEG Projections": eeg_tab})
    return


if __name__ == "__main__":
    app.run()
