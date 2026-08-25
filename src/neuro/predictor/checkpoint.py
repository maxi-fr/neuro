from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
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
