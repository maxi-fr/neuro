from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from types import ModuleType

_ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"{name}_mod", _ROOT / "scripts" / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_amplitude_bias_reports_the_ratio_the_rollout_underestimates_by() -> None:
    module = _load_script("jansen_rit_horizon_error")
    actual = np.zeros((4, 10, 3))
    actual[:, ::2] = 1.0  # peak-to-peak 1.0 per region
    predicted = 0.75 * actual

    bias = module.amplitude_bias(predicted, actual, {"all": [0, 1, 2]})

    assert bias["ptp_ratio_all"] == pytest.approx(0.75)
    assert np.isnan(module.amplitude_bias(predicted[:, :1], actual[:, :1], {"all": [0]})["ptp_ratio_all"])
