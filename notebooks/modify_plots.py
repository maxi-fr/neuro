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

    plots_dir: Path = Path(__file__).parent.parent / "artifacts"
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

    selected_folder: Path = Path(__file__).parent.parent / "artifacts" / plot_selector.value
    base_file: Path = selected_folder / plot_selector.value

    # Load metadata safely with Path objects
    meta_data: dict[str, Any] = {}
    json_path: Path = base_file.with_suffix(".json")
    with json_path.open(encoding="utf-8") as f_json:
        meta_data = json.load(f_json)

    # Find all PNG files in the folder for preview
    png_files = sorted(selected_folder.glob("*.png"))
    if png_files:
        previews = [mo.image(src=str(p), width=450) for p in png_files]
        preview_img = mo.hstack(previews) if len(previews) > 1 else previews[0]
    else:
        preview_img = mo.md("⚠️ *No PNG preview found.*")

    meta_view: mo.vstack = mo.vstack([mo.md("### ⚙️ Experiment Configuration"), mo.dict_view(meta_data)])

    mo.hstack([preview_img, meta_view], gap=3)
    return (base_file,)


@app.cell
def _(
    base_file: "Path",
    mo,
    pickle,
    rebuild_btn: "mo.ui.button",
    saver: "ThesisPlotSaver",
    width_slider: "mo.ui.slider",
):
    mo.stop(not rebuild_btn.value)

    # Unpickle using structural path typing for all pickle files in the folder
    import shutil
    import warnings

    pkl_files = sorted(base_file.parent.glob("*.pkl"))
    for p_path in pkl_files:
        with p_path.open("rb") as f:
            fig = pickle.load(f)

        # Recalculate dimensions dynamically
        new_size = saver.calculate_dimensions(fraction=width_slider.value)
        fig.set_size_inches(new_size[0], new_size[1])

        # Overwrite the targeted vector layouts cleanly
        if shutil.which("pdflatex") is not None:
            try:
                fig.savefig(p_path.with_suffix(".pgf"), bbox_inches="tight")
            except Exception as e:  # noqa: BLE001
                warnings.warn(f"Could not save PGF graphic: {e}", RuntimeWarning, stacklevel=2)
        else:
            warnings.warn("pdflatex not found. Skipping PGF export.", RuntimeWarning, stacklevel=2)

        fig.savefig(p_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    return


if __name__ == "__main__":
    app.run()
