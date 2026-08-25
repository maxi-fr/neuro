from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from neuro.esn import ESNArtifact
from neuro.esn_predictor_casadi import ESNSymbolicModel
from neuro.nn_predictor_casadi import NNSymbolicModel
from neuro.observable import ObservableArtifact
from neuro.observable_casadi import ObservableSymbolicModel
from neuro.predictor.artifact import MLPArtifact

if TYPE_CHECKING:
    from neuro.types import ObservableModel, SymbolicModel

RolloutArtifact = MLPArtifact | ESNArtifact
"""Artifacts that free-run on the sample grid; the observable one forecasts the Observable instead."""
PredictorArtifact = RolloutArtifact | ObservableArtifact


def load_any_artifact(artifact_path: str | Path) -> PredictorArtifact:
    """Load a single-``.npz`` predictor artifact (MLP, ESN or observable) from disk."""
    p = Path(artifact_path)
    npz_path = p.with_suffix(".npz")
    with np.load(npz_path) as npz:
        model_type = json.loads(str(npz["meta"]))["model_type"]
    if model_type == "mlp":
        return MLPArtifact.load(p)
    if model_type == "esn":
        return ESNArtifact.load(p)
    if model_type == "observable":
        return ObservableArtifact.load(p)
    msg = f"unsupported model_type {model_type!r} in {npz_path}"
    raise ValueError(msg)


def load_rollout_artifact(artifact_path: str | Path) -> RolloutArtifact:
    """Load an artifact that free-runs on the sample grid, rejecting an observable one."""
    art = load_any_artifact(artifact_path)
    if isinstance(art, ObservableArtifact):
        msg = f"{artifact_path} is an observable artifact; it forecasts the Observable and never a waveform."
        raise TypeError(msg)
    return art


def build_symbolic_model(art: PredictorArtifact) -> SymbolicModel | ObservableModel:
    """Build the appropriate symbolic model bridge; the MPC branches on which of the two it gets."""
    if isinstance(art, ESNArtifact):
        return ESNSymbolicModel(art)
    if isinstance(art, ObservableArtifact):
        return ObservableSymbolicModel(art)
    return NNSymbolicModel(art)
