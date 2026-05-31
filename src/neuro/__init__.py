"""Model Predictive Control for Neurostimulation."""

from __future__ import annotations

from neuro.plant import FHNPlant, NativeFHNPlant, TVBFHNPlant
from neuro.plotting import plot_dashboard, plot_fourier, plot_signals
from neuro.processing import compute_fft, compute_psd

__all__ = [
    "FHNPlant",
    "NativeFHNPlant",
    "TVBFHNPlant",
    "compute_fft",
    "compute_psd",
    "plot_dashboard",
    "plot_fourier",
    "plot_signals",
]
