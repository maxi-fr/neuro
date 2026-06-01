"""Model Predictive Control for Neurostimulation."""

from __future__ import annotations

from neuro.control import ZeroController
from neuro.jansen_rit_plant import TVBJansenRitDynamics, TVBJansenRitOutput
from neuro.plant import (
    FHNOutput,
    NativeFHNDynamics,
    TVBFHNDynamics,
    TVBFHNOutput,
)
from neuro.plotting import plot_dashboard, plot_fourier, plot_signals
from neuro.processing import compute_fft, compute_psd
from neuro.sensing import DirectSensor

__all__ = [
    "DirectSensor",
    "FHNOutput",
    "NativeFHNDynamics",
    "TVBFHNDynamics",
    "TVBFHNOutput",
    "TVBJansenRitDynamics",
    "TVBJansenRitOutput",
    "ZeroController",
    "compute_fft",
    "compute_psd",
    "plot_dashboard",
    "plot_fourier",
    "plot_signals",
]
