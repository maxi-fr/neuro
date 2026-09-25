import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full", app_title="Closed-loop run explorer")


@app.cell(hide_code=True)
def _():
    import io
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt

    from neuro.prediction_spectra import plot_spectra
    from neuro.run_plot import plot_runs
    from neuro.run_view import PredictionSeries, Run, ViewSettings, discover_runs, load_bookmarks, save_bookmark

    return (
        Path,
        PredictionSeries,
        Run,
        ViewSettings,
        discover_runs,
        io,
        load_bookmarks,
        mo,
        plot_runs,
        plot_spectra,
        plt,
        save_bookmark,
    )


@app.cell(hide_code=True)
def _(mo):
    _intro = mo.md("""
    # Closed-loop run explorer

    Open a collection of saved runs. **Closed-loop runs** compares controllers on their own recordings;
    **Shared recording** compares prepared Predictor replays on one recording.

    Prediction lines start at selected controller decisions. Planned and applied stimulation are shown
    separately because later decisions can change the original plan. Predictor outputs may be filtered EEG,
    LFP, or Observable values; their output indices are separate from raw EEG channel indices.
    """)
    collection = mo.ui.text(value="artifacts", label="Run collection directory", full_width=True).form(
        submit_button_label="Load collection"
    )
    mo.vstack([_intro, collection])
    return (collection,)


@app.cell(hide_code=True)
def _(Path, collection, discover_runs, mo):
    mo.stop(collection.value is None, mo.md("Choose a collection to start."))
    root = Path(collection.value).resolve()
    run_paths = discover_runs(root)
    mo.stop(
        not run_paths,
        mo.md("No runs found. Each run needs `config.yaml` and `log.npz` from the updated simulate package."),
    )
    run_names = [path.relative_to(root).as_posix() for path in run_paths]
    bookmark_path = root / "bookmarks.json"
    return bookmark_path, root, run_names


@app.cell(hide_code=True)
def _(ViewSettings, mo):
    get_restored, set_restored = mo.state(ViewSettings())
    get_revision, set_revision = mo.state(0)
    return get_restored, get_revision, set_restored, set_revision


@app.cell(hide_code=True)
def _(
    ViewSettings,
    bookmark_path,
    get_revision,
    load_bookmarks,
    mo,
    run_names,
):
    get_revision()
    bookmarks = {
        "Overview": ViewSettings(runs=[run_names[0]]),
        "Decision detail": ViewSettings(runs=[run_names[0]], start=5.5, width=1.5, decision=6.0, neighbors=1),
        **load_bookmarks(bookmark_path),
    }
    bookmark_choice = mo.ui.dropdown(options=list(bookmarks), value="Overview", label="Saved view")
    return bookmark_choice, bookmarks


@app.cell(hide_code=True)
def _(bookmark_choice, bookmarks, mo, set_restored):
    restore_button = mo.ui.button(
        label="Restore view", on_click=lambda _: set_restored(bookmarks[bookmark_choice.value])
    )
    mo.hstack([bookmark_choice, restore_button], justify="start")
    return


@app.cell(hide_code=True)
def _(get_restored, mo):
    restored = get_restored()
    mode = mo.ui.dropdown(options=["Closed-loop runs", "Shared recording"], value=restored.mode, label="Comparison")
    mode
    return mode, restored


@app.cell(hide_code=True)
def _(mo, mode, restored, run_names):
    _selected = [name for name in restored.runs if name in run_names] or [run_names[0]]
    run_choice = (
        mo.ui.dropdown(options=run_names, value=_selected[0], label="Source recording")
        if mode.value == "Shared recording"
        else mo.ui.multiselect(options=run_names, value=_selected, label="Closed-loop runs")
    )
    run_choice
    return (run_choice,)


@app.cell(hide_code=True)
def _(Run, mo, mode, root, run_choice):
    selected_names = [run_choice.value] if mode.value == "Shared recording" else run_choice.value
    mo.stop(not selected_names, mo.md("Select at least one run."))
    selected_runs = [Run.load(root / name) for name in selected_names]
    return selected_names, selected_runs


@app.cell(hide_code=True)
def _(mo, selected_runs):
    _rows = [run.metrics() for run in selected_runs]
    _table = mo.ui.table(_rows, selection=None) if _rows else mo.md("")
    mo.accordion({"📊 Run Metrics Summary": _table})
    return


@app.cell
def _(mo, restored, root, selected_runs):
    replay_paths = {
        path.relative_to(root).as_posix(): path
        for run in selected_runs
        for path in sorted((run.directory / "replays").glob("*.npz"))
    }
    _defaults = [name for name in restored.replays if name in replay_paths]
    replay_choice = mo.ui.multiselect(
        options=list(replay_paths),
        value=_defaults if restored.runs else list(replay_paths),
        label="Prepared actual-input predictions",
    )
    mo.vstack(
        [
            replay_choice,
            mo.md(
                "Prepare missing replays offline with `uv run python scripts/prepare_run_predictions.py <collection>`. Use repeated `--config` arguments to compare other Predictors."
            ),
        ]
    )
    return replay_choice, replay_paths


@app.cell
def _(mo, restored, selected_runs):
    duration = max(float(run.config.get("t_end", 12.0)) for run in selected_runs)
    dt = min(float(run.config["controller"]["dt"]) for run in selected_runs)
    start = mo.ui.slider(
        start=0, stop=duration, step=dt, value=min(restored.start, duration), label="Window start (s)", show_value=True
    )
    width = mo.ui.slider(
        start=dt,
        stop=duration,
        step=dt,
        value=min(max(restored.width, dt), duration),
        label="Window width (s)",
        show_value=True,
    )
    decision = mo.ui.slider(
        start=0,
        stop=duration,
        step=dt,
        value=min(restored.decision, duration),
        label="Decision time (s)",
        show_value=True,
    )
    neighbors = mo.ui.slider(
        start=0, stop=5, value=restored.neighbors, label="Neighboring predictions on each side", show_value=True
    )
    spacing = mo.ui.slider(
        start=1, stop=50, value=restored.spacing, label="Spacing between anchors (decisions)", show_value=True
    )
    offset = mo.ui.number(value=restored.offset, label="Vertical offset between channels")
    mo.vstack([mo.hstack([start, width, decision]), mo.hstack([neighbors, spacing, offset])])
    return decision, neighbors, offset, spacing, start, width


@app.cell
def _(mo, restored, selected_runs):
    def channel_picker(key, defaults, label):
        counts = [run.arrays[key].shape[-1] for run in selected_runs if key in run.arrays]
        options = list(range(max(counts, default=0)))
        return mo.ui.multiselect(options=options, value=[i for i in defaults if i in options], label=label)

    _eeg_labels = next(
        (
            run.eeg_channel_labels()
            for run in selected_runs
            if run.config.get("sensors", {}).get("measurement", {}).get("class_path") == "neuro.eeg.EEGMeasurement"
        ),
        [],
    )
    _eeg_options = {label: index for index, label in enumerate(_eeg_labels)}
    eeg_channels = mo.ui.multiselect(
        options=_eeg_options,
        value=[label for label, index in _eeg_options.items() if index in restored.eeg_channels],
        label="EEG channels",
    )
    output_channels = channel_picker("controller.predicted_y", restored.output_channels, "Predictor output indices")
    _elec_labels = next((run.electrode_labels() for run in selected_runs if "controller.u" in run.arrays), [])
    _elec_options = {label: index for index, label in enumerate(_elec_labels)}
    electrodes = (
        mo.ui.multiselect(
            options=_elec_options,
            value=[label for label, index in _elec_options.items() if index in restored.electrodes],
            label="Stimulation electrodes",
        )
        if _elec_options
        else channel_picker("controller.u", restored.electrodes, "Stimulation electrodes")
    )
    regions = channel_picker("dynamics.lfp", restored.regions, "LFP regions (optional)")
    mo.hstack([eeg_channels, output_channels, electrodes, regions])
    return eeg_channels, electrodes, output_channels, regions


@app.cell
def _(loaded_replays, selected_runs):
    spectral_series = {
        f"{run.directory.name} / planned Control Currents": run.planned_predictions()
        for run in selected_runs
        if "controller.predicted_y" in run.arrays
    }
    spectral_series.update({f"{name} / applied Control Currents": value for name, value in loaded_replays.items()})
    return (spectral_series,)


@app.cell
def _(mo, restored, spectral_series):
    _max_horizon = max((value.predicted_y.shape[1] - 1 for value in spectral_series.values()), default=2)
    _channels = max(
        (value.predicted_y.shape[-1] // max(1, len(value.frequencies)) for value in spectral_series.values()), default=0
    )
    show_spectra = mo.ui.checkbox(value=restored.show_spectra, label="Show predicted vs recorded spectra")
    spectral_steps = mo.ui.slider(
        start=2,
        stop=max(2, _max_horizon),
        value=min(restored.spectral_steps, max(2, _max_horizon)),
        label="Waveform Segment length (future samples)",
        show_value=True,
    )
    spectral_frame = mo.ui.slider(
        start=1,
        stop=max(1, _max_horizon),
        value=min(restored.spectral_frame, max(1, _max_horizon)),
        label="Observable future Frame",
        show_value=True,
    )
    spectral_channels = mo.ui.multiselect(
        options=list(range(_channels)),
        value=[i for i in restored.spectral_channels if i < _channels],
        label="Spectral channels",
    )
    mo.vstack(
        [
            show_spectra,
            mo.hstack([spectral_steps, spectral_frame, spectral_channels]),
            mo.md(
                "Waveforms use the same future samples, periodic Hann window and density scaling on both sides. Observable distributions use the selected native log-power Frame. Each panel reports its time support; frequency resolution is limited by Segment length."
            ),
        ]
    )
    return show_spectra, spectral_channels, spectral_frame, spectral_steps


@app.cell
def _(mo, restored):
    bookmark_name = mo.ui.text(label="Bookmark name", value="Interesting decision")
    note = mo.ui.text_area(label="Why this view is interesting", value=restored.note)
    mo.hstack([bookmark_name, note])
    return bookmark_name, note


@app.cell
def _(
    ViewSettings,
    decision,
    eeg_channels,
    electrodes,
    mode,
    neighbors,
    note,
    offset,
    output_channels,
    regions,
    replay_choice,
    selected_names,
    show_spectra,
    spacing,
    spectral_channels,
    spectral_frame,
    spectral_steps,
    start,
    width,
):
    settings = ViewSettings(
        mode=mode.value,
        runs=selected_names,
        replays=replay_choice.value,
        start=start.value,
        width=width.value,
        decision=decision.value,
        neighbors=neighbors.value,
        spacing=spacing.value,
        offset=offset.value,
        eeg_channels=eeg_channels.value,
        output_channels=output_channels.value,
        electrodes=electrodes.value,
        regions=regions.value,
        note=note.value,
        show_spectra=show_spectra.value,
        spectral_steps=spectral_steps.value,
        spectral_frame=spectral_frame.value,
        spectral_channels=spectral_channels.value,
    )
    return (settings,)


@app.cell
def _(bookmark_name, bookmark_path, mo, save_bookmark, set_revision, settings):
    def save_view(_):
        if bookmark_name.value.strip():
            save_bookmark(bookmark_path, bookmark_name.value.strip(), settings)
            set_revision(lambda value: value + 1)
        return "Saved"

    save_button = mo.ui.button(label="Save bookmark", on_click=save_view)
    save_button
    return


@app.cell
def _(PredictionSeries, replay_choice, replay_paths):
    loaded_replays = {name: PredictionSeries.load(replay_paths[name]) for name in replay_choice.value}
    return (loaded_replays,)


@app.cell
def _(io, loaded_replays, mo, plot_runs, plt, selected_runs, settings):
    figure = plot_runs(selected_runs, settings, loaded_replays)
    _png = io.BytesIO()
    figure.savefig(_png, format="png", dpi=150)
    _display = mo.vstack([figure, mo.download(_png.getvalue(), filename="run-view.png", label="Download plot")])
    plt.close(figure)
    _display
    return


@app.cell
def _(io, mo, plot_spectra, plt, settings, spectral_series):
    mo.stop(not settings.show_spectra)
    spectral_figure = plot_spectra(spectral_series, settings)
    _buffer = io.BytesIO()
    spectral_figure.savefig(_buffer, format="png", dpi=150)
    _display = mo.vstack(
        [spectral_figure, mo.download(_buffer.getvalue(), filename="prediction-spectra.png", label="Download spectra")]
    )
    plt.close(spectral_figure)
    _display
    return


if __name__ == "__main__":
    app.run()
