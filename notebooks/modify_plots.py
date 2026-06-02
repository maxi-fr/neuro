import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell
def _():
    import json
    import pickle
    from pathlib import Path
    from typing import Any

    import marimo as mo

    from utils.save_plots import ThesisPlotSaver

    saver: ThesisPlotSaver = ThesisPlotSaver()
    return Any, Path, json, mo, pickle, saver


@app.cell(hide_code=True)
def _(Path, mo):
    mo.md("# 📊 Thesis Plot Inspector & Resizer")

    plots_dir: Path = Path("plots")
    available_plots: list[str] = []

    if plots_dir.exists() and plots_dir.is_dir():
        available_plots = sorted([d.name for d in plots_dir.iterdir() if d.is_dir()])

    plot_selector: mo.ui.dropdown = mo.ui.dropdown(
        options=available_plots, label="Select an Experiment Plot to Review:"
    )

    width_slider: mo.ui.slider = mo.ui.slider(
        start=0.3, stop=1.0, step=0.05, value=1.0, label="Target LaTeX Textwidth Fraction:"
    )

    rebuild_btn: mo.ui.button = mo.ui.button(label="🔄 Re-Save PGF with New Width")

    mo.hstack([plot_selector, width_slider, rebuild_btn], justify="start")
    return plot_selector, rebuild_btn, width_slider


@app.cell
def _(Any, Path, json, mo, plot_selector: "mo.ui.dropdown"):
    mo.stop(not plot_selector.value, mo.md("💡 *Select a plot above to view assets and metadata.*"))

    selected_folder: Path = Path("plots") / plot_selector.value
    base_file: Path = selected_folder / plot_selector.value

    # Load metadata safely with Path objects
    meta_data: dict[str, Any] = {}
    json_path: Path = base_file.with_suffix(".json")
    with json_path.open(encoding="utf-8") as f_json:
        meta_data = json.load(f_json)

    preview_img: mo.image = mo.image(src=str(base_file.with_suffix(".png")), width=450)

    meta_view: mo.vstack = mo.vstack([mo.md("### ⚙️ Experiment Configuration"), mo.dict_view(meta_data)])

    mo.hstack([preview_img, meta_view], gap=3)
    return (base_file,)


@app.cell
def _(
    Path,
    base_file: "Path",
    mo,
    pickle,
    rebuild_btn: "mo.ui.button",
    saver: "ThesisPlotSaver",
    width_slider: "mo.ui.slider",
):
    mo.stop(not rebuild_btn.value)

    # Unpickle using structural path typing
    pkl_path: Path = base_file.with_suffix(".pkl")
    with pkl_path.open("rb") as f:
        fig = pickle.load(f)

    # Recalculate dimensions dynamically
    new_size = saver.calculate_dimensions(fraction=width_slider.value)
    fig.set_size_inches(new_size[0], new_size[1])

    # Overwrite the targeted vector layouts cleanly
    import shutil
    import warnings

    if shutil.which("pdflatex") is not None:
        try:
            fig.savefig(base_file.with_suffix(".pgf"), bbox_inches="tight")
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"Could not save PGF graphic: {e}", RuntimeWarning, stacklevel=2)
    else:
        warnings.warn("pdflatex not found. Skipping PGF export.", RuntimeWarning, stacklevel=2)

    fig.savefig(base_file.with_suffix(".png"), dpi=200, bbox_inches="tight")
    return


if __name__ == "__main__":
    app.run()
