from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from neuro.spectral import compute_periodograms

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from neuro.run_view import PredictionSeries, ViewSettings
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
    """Compare one future Observable Frame or matching waveform Segments, excluding the initial knot."""
    anchor = int(np.argmin(np.abs(series.t - view.decision)))
    steps = view.spectral_frame if series.frequencies.size else view.spectral_steps
    if steps >= series.predicted_y.shape[1] or anchor + steps >= len(series.t):
        return None
    if series.frequencies.size:
        predicted = series.predicted_y[anchor, steps].reshape(-1, len(series.frequencies))
        observed = series.observed_y[anchor + steps].reshape(predicted.shape)
        result = SpectralComparison(
            series.frequencies, predicted, observed, "log power", f"Frame at {series.t[anchor + steps]:.3f}s"
        )
    else:
        if steps <= 1:
            return None
        predicted = series.predicted_y[anchor, 1 : steps + 1]
        observed = series.observed_y[anchor + 1 : anchor + steps + 1]
        fs = 1.0 / series.dt
        result = SpectralComparison(
            np.fft.rfftfreq(steps, series.dt),
            compute_periodograms(predicted, fs=fs, window=steps, hop=steps)[0],
            compute_periodograms(observed, fs=fs, window=steps, hop=steps)[0],
            "Power density (output units²/Hz)",
            f"{series.t[anchor + 1]:.3f} to {series.t[anchor + steps]:.3f}s; Δf={fs / steps:g} Hz",
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
