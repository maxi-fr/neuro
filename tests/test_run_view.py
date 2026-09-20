from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from neuro.predictor.data import load_trajectory
from neuro.run_plot import plot_runs
from neuro.run_view import Run, ViewSettings, discover_runs, load_bookmarks, save_bookmark


def test_run_uses_component_times_and_bookmarks_restore_every_setting(tmp_path: Path) -> None:
    directory = tmp_path / "arm_s1"
    directory.mkdir()
    (directory / "config.yaml").write_text("controller: {dt: 0.02}\n", encoding="utf-8")
    np.savez(
        directory / "log.npz",
        allow_pickle=False,
        **{
            "sensor_0.t": [0, 0.01, 0.02],
            "sensor_0.y_mea": [[1], [2], [3]],
            "controller.t": [0, 0.02],
            "controller.u": [[0], [1]],
        },
    )
    assert discover_runs(tmp_path) == [directory]
    run = Run.load(directory)
    t, values = run.signal("controller", "u")
    np.testing.assert_allclose(t, [0, 0.02])
    assert values.shape == (2, 1)
    settings = ViewSettings(
        runs=["arm_s1"],
        start=0.01,
        width=0.3,
        decision=0.02,
        neighbors=2,
        eeg_channels=[0],
        electrodes=[0],
        regions=[],
        note="onset",
        spectral_steps=7,
        spectral_frame=3,
        spectral_channels=[1, 2],
    )
    save_bookmark(tmp_path / "bookmarks.json", "onset", settings)
    assert load_bookmarks(tmp_path / "bookmarks.json")["onset"] == settings


def test_eeg_channel_labels_follow_the_configured_sensor_subset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("neuro.run_view.build_eeg_leadfield", lambda: (np.empty((3, 1)), np.array(["F3", "P3", "O1"])))
    run = Run(
        tmp_path,
        {"sensors": {"measurement": {"selected_channels": ["O1", 0]}}},
        {"sensor_0.t": np.array([0.0]), "sensor_0.eeg": np.ones((1, 2))},
    )
    assert run.eeg_channel_labels() == ["O1", "F3"]


def test_prediction_lines_are_anchored_in_absolute_time_and_currents_use_own_clock(tmp_path: Path) -> None:
    t = np.array([0.0, 0.02, 0.04])
    arrays = {
        "controller.t": t,
        "controller.u": np.ones((3, 1)),
        "controller.predicted_y": np.ones((3, 3, 1)),
        "estimator.t": t,
        "estimator.x_hat": np.ones((3, 1)),
        "controller.planned_u": np.ones((3, 2, 1)),
    }
    run = Run(tmp_path, {"controller": {"dt": 0.02}}, arrays)
    view = ViewSettings(decision=0.02, width=0.1)
    fig = plot_runs([run], view, {})
    np.testing.assert_allclose(np.asarray(fig.axes[1].lines[1].get_xdata()), [0.02, 0.04, 0.06])
    np.testing.assert_allclose(np.asarray(fig.axes[3].lines[0].get_xdata()), t)
    plt.close(fig)


def test_training_loader_holds_decision_currents_on_the_eeg_clock(tmp_path: Path) -> None:
    path = tmp_path / "log.npz"
    np.savez(
        path,
        allow_pickle=False,
        **{
            "sensor_0.t": np.arange(9) * 0.01,
            "sensor_0.y_mea": np.ones((9, 1)),
            "controller.t": [0, 0.04, 0.08],
            "controller.u": [[1], [2], [3]],
        },
    )
    controls, eeg = load_trajectory(str(path), None, 2, 0.01)
    np.testing.assert_array_equal(controls[:, 0], [1, 1, 2, 2, 3])
    assert len(eeg) == len(controls)


def test_decision_observations_use_held_estimates_not_controller_predictions(tmp_path: Path) -> None:
    run = Run(
        tmp_path,
        {"controller": {"dt": 0.02}},
        {
            "controller.t": np.array([0.0, 0.02, 0.04]),
            "controller.predicted_y": np.full((3, 2, 1), 999.0),
            "estimator.t": np.array([0.0, 0.015, 0.03, 0.045]),
            "estimator.x_hat": np.array([[1.0], [2.0], [3.0], [4.0]]),
        },
    )
    np.testing.assert_array_equal(run.planned_predictions().observed_y[:, 0], [1, 2, 3])
