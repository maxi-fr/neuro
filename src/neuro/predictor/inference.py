from __future__ import annotations

from typing import TYPE_CHECKING, Any

from neuro.predictor.base import (
    AutoregressiveModel,
    InferencePredictor,
)
from neuro.predictor.cnn import (
    ObservableCNNModel,
    WaveformCNNModel,
)
from neuro.predictor.mlp import (
    ObservableMLPModel,
    WaveformMLPModel,
)

if TYPE_CHECKING:
    from neuro.types import FloatArray


def inference_from_checkpoint(
    meta: dict[str, Any], arrays: dict[str, FloatArray], *, n_history: int | None = None
) -> InferencePredictor:
    """Construct the runtime adapter selected by checkpoint architecture metadata."""
    if meta.get("model_type") == "cnn":
        if "geometry" in meta:
            return ObservableCNNModel.from_checkpoint(meta, arrays, n_history=n_history)
        return WaveformCNNModel.from_checkpoint(meta, arrays, n_history=n_history)
    if "geometry" in meta:
        return ObservableMLPModel.from_checkpoint(meta, arrays, n_history=n_history)
    if meta.get("model_type") != "mlp":
        msg = f"unsupported neural checkpoint model_type: {meta.get('model_type')!r}"
        raise ValueError(msg)
    return WaveformMLPModel.from_checkpoint(meta, arrays, n_history=n_history)


__all__ = [
    "AutoregressiveModel",
    "InferencePredictor",
    "ObservableCNNModel",
    "ObservableMLPModel",
    "WaveformCNNModel",
    "WaveformMLPModel",
    "inference_from_checkpoint",
]
