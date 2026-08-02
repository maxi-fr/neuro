import json
import pickle
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from utils.save_plots import ThesisPlotSaver


def test_thesis_plot_saver_dimensions() -> None:
    """Test golden-ratio dimension calculation."""
    saver = ThesisPlotSaver(textwidth_pt=418.25)
    width, height = saver.calculate_dimensions(fraction=1.0)
    assert width > 0
    assert height > 0
    assert width > height


def test_thesis_plot_saver_save(tmp_path: Path) -> None:
    """Test saving a plot with metadata and files generation."""
    saver = ThesisPlotSaver(base_dir=str(tmp_path))

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    metadata = {"test_key": "test_value"}
    saver.save(fig, "test_plot", metadata=metadata, overwrite=True)
    plt.close(fig)

    target_dir = tmp_path / "test_plot"
    assert target_dir.exists()
    assert target_dir.is_dir()

    json_path = target_dir / "test_plot.json"
    assert json_path.exists()
    with json_path.open("r", encoding="utf-8") as f:
        saved_meta = json.load(f)
    assert saved_meta["test_key"] == "test_value"
    assert "git_commit" in saved_meta
    assert "generated_at" in saved_meta

    pkl_path = target_dir / "test_plot.pkl"
    assert pkl_path.exists()
    with pkl_path.open("rb") as f:
        loaded_fig = pickle.load(f)  # noqa: S301
    assert loaded_fig is not None
    plt.close(loaded_fig)

    png_path = target_dir / "test_plot.png"
    assert png_path.exists()


def test_thesis_plot_saver_increment_vs_overwrite(tmp_path: Path) -> None:
    """Test that overwrite=False increments the folder name and overwrite=True does not."""
    saver = ThesisPlotSaver(base_dir=str(tmp_path))

    fig1, ax1 = plt.subplots()
    ax1.plot([0, 1], [0, 1])
    saver.save(fig1, "my_plot", overwrite=False)
    plt.close(fig1)

    dir1 = tmp_path / "my_plot"
    assert dir1.exists()

    fig2, ax2 = plt.subplots()
    ax2.plot([0, 1], [0, 2])
    saver.save(fig2, "my_plot", overwrite=False)
    plt.close(fig2)

    dir2 = tmp_path / "my_plot_01"
    assert dir2.exists()

    fig3, ax3 = plt.subplots()
    ax3.plot([0, 1], [0, 3])
    saver.save(fig3, "my_plot", overwrite=True)
    plt.close(fig3)

    dir3 = tmp_path / "my_plot_02"
    assert not dir3.exists()


def test_thesis_plot_saver_save_with_data(tmp_path: Path) -> None:
    """Test saving dictionaries of numpy arrays."""
    saver = ThesisPlotSaver(base_dir=str(tmp_path))
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    dict_data = {"eeg": np.array([1, 2]), "activity": np.array([3, 4])}
    saver.save(fig, "plot_dict", data=dict_data, overwrite=True)
    npz_path = tmp_path / "plot_dict" / "plot_dict.npz"
    assert npz_path.exists()
    loaded_npz = np.load(npz_path)
    np.testing.assert_array_equal(loaded_npz["eeg"], dict_data["eeg"])
    np.testing.assert_array_equal(loaded_npz["activity"], dict_data["activity"])

    plt.close(fig)


def test_thesis_plot_saver_save_dict_figs(tmp_path: Path) -> None:
    """Test saving multiple figures as a dictionary."""
    saver = ThesisPlotSaver(base_dir=str(tmp_path))

    fig1, ax1 = plt.subplots()
    ax1.plot([0, 1], [0, 1])
    fig2, ax2 = plt.subplots()
    ax2.plot([0, 1], [0, 2])

    saver.save({"traces": fig1, "heatmap": fig2}, "dict_plots", overwrite=True)
    assert (tmp_path / "dict_plots" / "dict_plots_traces.png").exists()
    assert (tmp_path / "dict_plots" / "dict_plots_heatmap.png").exists()

    plt.close(fig1)
    plt.close(fig2)


def test_thesis_plot_saver_metadata(tmp_path: Path) -> None:
    """Test metadata saving directly in the metadata dictionary."""
    saver = ThesisPlotSaver(base_dir=str(tmp_path))
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    metadata = {
        "generator": "test_generator_func",
        "step_size": 0.05,
        "n_sims": 10,
    }
    saver.save(fig, "gen_plot", metadata=metadata, overwrite=True)
    plt.close(fig)

    json_path = tmp_path / "gen_plot" / "gen_plot.json"
    assert json_path.exists()
    with json_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["generator"] == "test_generator_func"
    assert meta["step_size"] == 0.05
    assert meta["n_sims"] == 10
