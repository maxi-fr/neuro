"""Model Predictive Control for Neurostimulation."""

from __future__ import annotations

from neuro.jansen_rit_plant import TVBJansenRitDynamics, TVBJansenRitOutput
from neuro.plant import (
    FHNOutput,
    NativeFHNDynamics,
    TVBFHNDynamics,
    TVBFHNOutput,
)
from neuro.plotting import plot_dashboard, plot_fourier, plot_signals
from neuro.processing import compute_fft, compute_psd

__all__ = [
    "FHNOutput",
    "NativeFHNDynamics",
    "TVBFHNDynamics",
    "TVBFHNOutput",
    "TVBJansenRitDynamics",
    "TVBJansenRitOutput",
    "compute_fft",
    "compute_psd",
    "plot_dashboard",
    "plot_fourier",
    "plot_signals",
]
