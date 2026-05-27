"""Equivalence tests for FHNPlant (neurolib-backed) and NativeFHNPlant (pure NumPy).

Both classes are constructed with identical kwargs including a fixed ``seed``
so that initial conditions and OU noise are drawn from the same RNG state.
Numerical differences from floating-point non-associativity in the coupling
sum (numpy vectorised vs. numba sequential) are expected to be below 1e-10.
"""

import numpy as np
import pytest

from closed_loop_neurostimulation.plant import FHNPlant, NativeFHNPlant

_SEED = 42
_DT = 0.1
_N_SENSORS = 64
_N_NODES = 80  # HCP connectome


def _make_plants(sigma_ou: float, seed: int = _SEED) -> tuple[FHNPlant, NativeFHNPlant]:
    kwargs: dict[str, object] = {
        "dt": _DT,
        "sigma_ou": sigma_ou,
        "n_sensors": _N_SENSORS,
        "leadfield_seed": 0,
        "seed": seed,
    }
    return FHNPlant(**kwargs), NativeFHNPlant(**kwargs)  # type: ignore


def test_output_shapes() -> None:
    """Activity and EEG arrays have the expected shapes."""
    _, native = _make_plants(sigma_ou=0.0)
    activity, eeg = native.step(100.0)
    n_samples = round(100.0 / _DT)
    assert activity.shape == (_N_NODES, n_samples)
    assert eeg.shape == (_N_SENSORS, n_samples)


def test_deterministic_exact_match() -> None:
    """Zero-noise implementations agree to near machine precision."""
    neurolib_plant, native_plant = _make_plants(sigma_ou=0.0)
    act_nl, eeg_nl = neurolib_plant.step(200.0)
    act_na, eeg_na = native_plant.step(200.0)
    np.testing.assert_allclose(act_na, act_nl, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(eeg_na, eeg_nl, rtol=1e-10, atol=1e-10)


def test_seeded_noisy_exact_match() -> None:
    """With a fixed seed and sigma_ou > 0, both implementations agree."""
    neurolib_plant, native_plant = _make_plants(sigma_ou=0.05)
    act_nl, eeg_nl = neurolib_plant.step(200.0)
    act_na, eeg_na = native_plant.step(200.0)
    np.testing.assert_allclose(act_na, act_nl, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(eeg_na, eeg_nl, rtol=1e-10, atol=1e-10)


def test_multistep_match() -> None:
    """Multi-step runs produce matching outputs across both implementations."""
    neurolib_plant, native_plant = _make_plants(sigma_ou=0.05)
    acts_nl = []
    acts_na = []
    for _ in range(5):
        act_nl, _ = neurolib_plant.step(100.0)
        act_na, _ = native_plant.step(100.0)
        acts_nl.append(act_nl)
        acts_na.append(act_na)
    np.testing.assert_allclose(
        np.concatenate(acts_na, axis=1),
        np.concatenate(acts_nl, axis=1),
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.parametrize("sigma_ou", [0.0, 0.05])
def test_reset_reproduces_first_step(sigma_ou: float) -> None:
    """After reset, NativeFHNPlant reproduces the same output as the first step."""
    _, native = _make_plants(sigma_ou=sigma_ou)
    act_first, _ = native.step(100.0)
    native.reset()
    act_second, _ = native.step(100.0)
    np.testing.assert_array_equal(act_first, act_second)
