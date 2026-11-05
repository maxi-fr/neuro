from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from neuro.spectral import compute_periodograms

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from neuro.runs.record import PredictionSeries
    from neuro.runs.view import ViewSettings
    from neuro.types import FloatArray


@dataclass(frozen=True)
class SpectralComparison:
    """Predicted and recorded distributions ``(channels, frequencies)`` on identical temporal support."""

    frequencies: FloatArray
    predicted: FloatArray
    observed: FloatArray
    units: str
    support: str


def compare_spectra(series: PredictionSeries, view: ViewSettings) -> SpectralComparison | None:
    """Compare future Frames or Segments against their native observation timestamps, excluding the initial knot."""
    if not len(series.t):
        return None
    anchor = int(np.argmin(np.abs(series.t - view.decision)))
    steps = view.spectral_frame if series.frequencies.size else view.spectral_steps
    if steps >= series.predicted_y.shape[1]:
        return None
    target_times = series.t[anchor] + np.arange(1, steps + 1) * series.dt
    indices = np.searchsorted(series.observation_times, target_times - 1e-9)
    if np.any(indices >= len(series.observation_times)):
        return None
    if not np.allclose(series.observation_times[indices], target_times, rtol=0, atol=1e-9):
        return None
    if series.frequencies.size:
        predicted = series.predicted_y[anchor, steps].reshape(-1, len(series.frequencies))
        observed = series.observed_y[indices[-1]].reshape(predicted.shape)
        result = SpectralComparison(
            series.frequencies, predicted, observed, "log power", f"Frame at {target_times[-1]:.3f}s"
        )
    else:
        if steps <= 1:
            return None
        predicted = series.predicted_y[anchor, 1 : steps + 1]
        observed = series.observed_y[indices]
        fs = 1.0 / series.dt
        result = SpectralComparison(
            np.fft.rfftfreq(steps, series.dt),
            compute_periodograms(predicted, fs=fs, window=steps, hop=steps)[0],
            compute_periodograms(observed, fs=fs, window=steps, hop=steps)[0],
            "Power density (output units²/Hz)",
            f"{target_times[0]:.3f} to {target_times[-1]:.3f}s; Δf={fs / steps:g} Hz",
        )
    return result if np.isfinite(result.predicted).all() and np.isfinite(result.observed).all() else None


def plot_spectra(series: dict[str, PredictionSeries], view: ViewSettings) -> Figure:
    """Draw predicted and recorded distributions separately for each Predictor and input condition."""
    fig, axes = plt.subplots(
        max(1, len(series)), 1, figsize=(12, 3.5 * max(1, len(series))), squeeze=False, layout="constrained"
    )
    if not series:
        axes[0, 0].text(0.1, 0.5, "Select saved predictions to compare spectra.")
    for ax, (label, predictions) in zip(axes[:, 0], series.items(), strict=bool(series)):
        result = compare_spectra(predictions, view)
        ax.set_title(label, loc="left", fontsize=10)
        ax.set_xlabel("Frequency (Hz)")
        if result is None:
            ax.text(
                0.05, 0.5, "No complete finite prediction and recording for this selection.", transform=ax.transAxes
            )
            continue
        for channel in view.spectral_channels:
            if channel < result.predicted.shape[0]:
                color = f"C{channel % 10}"
                ax.plot(result.frequencies, result.observed[channel], color=color, label=f"Recorded [{channel}]")
                ax.plot(
                    result.frequencies, result.predicted[channel], "--", color=color, label=f"Predicted [{channel}]"
                )
        ax.set_title(f"{label} · {result.support}", loc="left", fontsize=10)
        ax.set_ylabel(result.units)
        ax.grid(alpha=0.2)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    return fig
