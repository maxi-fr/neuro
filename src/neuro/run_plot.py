from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from neuro.run_view import decision_indices

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from neuro.run_view import PredictionSeries, Run, ViewSettings
    from neuro.types import FloatArray


@dataclass
class _Panel:
    """One plotting axis and the view shared by its traces and prediction anchors."""

    ax: Axes
    view: ViewSettings

    def traces(self, t: FloatArray, y: FloatArray, channels: list[int], label: str, *, step: bool = False) -> None:
        """Draw selected channels using their recorded timestamps, clipped to the visible interval."""
        selected = (t >= self.view.start) & (t <= self.view.start + self.view.width)
        for position, channel in enumerate(channels):
            if channel < y.shape[1]:
                self.ax.plot(
                    t[selected],
                    y[selected, channel] + position * self.view.offset,
                    label=f"{label} [{channel}]",
                    linewidth=0.8,
                    drawstyle="steps-post" if step else "default",
                )

    def rollouts(self, times: FloatArray, predictions: FloatArray, dt: float, channels: list[int], label: str) -> None:
        """Overlay recorded Rollouts at selected decision anchors."""
        for anchor in decision_indices(times, self.view):
            t = times[anchor] + np.arange(predictions.shape[1]) * dt
            for position, channel in enumerate(channels):
                if channel < predictions.shape[2] and np.isfinite(predictions[anchor, :, channel]).any():
                    self.ax.plot(
                        t,
                        predictions[anchor, :, channel] + position * self.view.offset,
                        "--",
                        linewidth=1.2,
                        label=f"{label} [{channel}] @ {times[anchor]:.3f}s",
                    )


def _plot_run(axes: list[Axes], run: Run, view: ViewSettings) -> None:
    """Draw recorded EEG, controller plans and applied Control Currents on synchronized axes."""
    panels = [_Panel(ax, view) for ax in axes]
    label = run.directory.name
    if "sensor_0.y_mea" in run.arrays or "sensor_0.eeg" in run.arrays:
        panels[0].traces(*run.eeg(), view.eeg_channels, label)
    if "controller.u" in run.arrays:
        panels[3].traces(*run.signal("controller", "u"), view.electrodes, f"{label} applied", step=True)
    if "controller.predicted_y" in run.arrays:
        series = run.planned_predictions()
        t, dt = series.t, series.dt
        panels[1].traces(t, series.observed_y, view.output_channels, f"{label} observed")
        panels[1].rollouts(t, series.predicted_y, dt, view.output_channels, label)
        _, plans = run.signal("controller", "planned_u")
        for anchor in decision_indices(t, view):
            plan_t = t[anchor] + np.arange(plans.shape[1] + 1) * dt
            panels[4].traces(
                plan_t,
                np.concatenate((plans[anchor], plans[anchor, -1:])),
                view.electrodes,
                f"{label} @ {t[anchor]:.3f}s",
                step=True,
            )
    if view.regions and "dynamics.lfp" in run.arrays:
        panels[5].traces(*run.signal("dynamics", "lfp"), view.regions, label)


def plot_runs(runs: list[Run], view: ViewSettings, replays: dict[str, PredictionSeries]) -> Figure:
    """Render both prediction conditions with raw signals and distinct applied/planned Control Current panels."""
    titles = [
        "Recorded EEG",
        "Predictions under planned Control Currents / observed Predictor outputs",
        "Predictions under applied Control Currents / observed Predictor outputs",
        "Applied Control Currents",
        "Planned Control Currents",
    ]
    if view.regions:
        titles.append("Regional LFP")
    fig, axes_array = plt.subplots(len(titles), 1, sharex=True, figsize=(14, 2.6 * len(titles)), layout="constrained")
    axes = list(axes_array)
    panels = [_Panel(ax, view) for ax in axes]
    for run in runs:
        _plot_run(axes, run, view)
    for label, replay in replays.items():
        panels[2].traces(replay.t, replay.observed_y, view.output_channels, f"{label} observed")
        panels[2].rollouts(replay.t, replay.predicted_y, replay.dt, view.output_channels, label)
    for ax, title in zip(axes, titles, strict=True):
        ax.set_title(title, loc="left", fontsize=10)
        ax.axvline(view.decision, color="0.5", linewidth=0.7)
        ax.grid(alpha=0.2)
        ax.set_xlim(view.start, view.start + view.width)
        if ax.lines and ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=6, loc="upper right", ncols=2)
        else:
            ax.text(0.02, 0.5, "No saved data selected for this panel", transform=ax.transAxes)
    axes[3].set_ylabel("mA")
    axes[4].set_ylabel("mA")
    axes[-1].set_xlabel("Time (s)")
    return fig
