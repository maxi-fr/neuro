from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from neuro.types import ACTIVATIONS

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from neuro.types import FloatArray


def save_checkpoint(path: str | Path, *, meta: dict[str, Any], arrays: dict[str, FloatArray]) -> None:
    """Persist ``meta`` and ``arrays`` into one ``.npz`` at ``path``, a suffix-less stem.

    ``meta`` is stored as a 0-d unicode array holding JSON, so loading needs no ``allow_pickle``;
    every weight and standardizer rides as its own NumPy array, which is what lets a torch-free
    reader consume the checkpoint later. The layout is the one the checkpoint readers read, so
    the incumbent control path keeps loading what ``save`` writes.
    """
    file = Path(path).with_suffix(".npz")
    file.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {"meta": np.array(json.dumps(meta))}
    payload.update({key: np.asarray(value) for key, value in arrays.items()})
    np.savez(file, **payload)  # ty: ignore[invalid-argument-type]


def load_checkpoint(path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Read back :func:`save_checkpoint`: the ``meta`` mapping and every stored array, keyed as written."""
    file = Path(path).with_suffix(".npz")
    with np.load(file) as npz:
        meta: dict[str, Any] = json.loads(str(npz["meta"]))
        arrays = {key: np.asarray(npz[key]) for key in npz.files if key != "meta"}
    return meta, arrays


def load_meta(path: str | Path) -> dict[str, Any]:
    """Read only the JSON ``meta`` block of a checkpoint, leaving every weight array on disk."""
    file = Path(path).with_suffix(".npz")
    with np.load(file) as npz:
        return json.loads(str(npz["meta"]))


def require_model_type(meta: dict[str, Any], expected: str) -> None:
    """Raise unless the checkpoint's ``model_type`` is the one this loader rebuilds."""
    if meta.get("model_type") != expected:
        msg = f"checkpoint is model_type {meta.get('model_type')!r}, not {expected!r}."
        raise ValueError(msg)


def require_activation(meta: dict[str, Any]) -> None:
    """Raise unless the checkpoint's recorded activation is one both sides can evaluate."""
    if meta["activation"] not in ACTIVATIONS:
        msg = f"Unsupported activation: {meta['activation']!r}"
        raise ValueError(msg)


def layer_arrays(
    prefix: str, weights: Sequence[npt.ArrayLike], biases: Sequence[npt.ArrayLike]
) -> dict[str, FloatArray]:
    """Key one MLP block's weights and biases as ``<prefix>.<i>.weight`` / ``<prefix>.<i>.bias``.

    This and :func:`layers_from_arrays` are the one place the block key convention is written, so
    the torch writer and the jax reader cannot drift apart on it.
    """
    arrays: dict[str, FloatArray] = {}
    for i, (weight, bias) in enumerate(zip(weights, biases, strict=True)):
        arrays[f"{prefix}.{i}.weight"] = np.asarray(weight, dtype=np.float64)
        arrays[f"{prefix}.{i}.bias"] = np.asarray(bias, dtype=np.float64)
    return arrays


def layers_from_arrays(
    arrays: Mapping[str, FloatArray], prefix: str, n_layers: int
) -> tuple[tuple[FloatArray, ...], tuple[FloatArray, ...]]:
    """Read back :func:`layer_arrays`: the block's ``(weights, biases)`` in layer order."""
    weights = tuple(np.asarray(arrays[f"{prefix}.{i}.weight"], dtype=np.float64) for i in range(n_layers))
    biases = tuple(np.asarray(arrays[f"{prefix}.{i}.bias"], dtype=np.float64) for i in range(n_layers))
    return weights, biases
