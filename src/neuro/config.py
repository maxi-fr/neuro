"""Configuration utilities."""

from __future__ import annotations

from typing import Any

import numpy as np


def parse_array(val: Any) -> Any:  # noqa: ANN401
    """Parse a configuration value that might be an array or a path to an array.

    If the value is a string ending in .npy, .npz, or .npv, it is loaded via
    np.load. For .npz files, the first array is returned. Otherwise, the original
    value is returned (to be parsed by np.asarray later).
    """
    if isinstance(val, str) and (val.endswith((".npy", ".npz", ".npv"))):
        loaded = np.load(val)
        if isinstance(loaded, np.ndarray):
            return loaded
        # npz file
        keys = list(loaded.keys())
        if keys:
            return loaded[keys[0]]
        msg = f"NPZ file {val} is empty"
        raise ValueError(msg)
    return val
