"""Tests for the whole-brain FHN plant implementations.

FHNPlant / NativeFHNPlant equivalence
--------------------------------------
Both classes are constructed with identical kwargs including a fixed ``seed``
so that initial conditions and OU noise are drawn from the same RNG state.
Numerical differences from floating-point non-associativity in the coupling
sum (numpy vectorised vs. numba sequential) are expected to be below 1e-10.

TVBFHNPlant
-----------
TVBFHNPlant uses TVB's ``Generic2dOscillator`` model with Heun-stochastic
integration and additive Gaussian noise (``nsig``).  Its dynamics are **not**
numerically comparable to the neurolib plants because the FHN equations,
integrators, and coupling functions all differ.  Tests here cover:

* Correct output shapes
* Determinism (two instances with the same seed produce identical output)
* State continuity (multi-step matches single-step)
* Reset reproducibility
* Finite-valued output with noise
* Structural: same 80-node HCP connectome as the neurolib plants
"""

import numpy as np
import pytest

from closed_loop_neurostimulation.plant import FHNPlant, NativeFHNPlant, TVBFHNPlant

_SEED = 42
_DT = 0.1
_N_SENSORS = 64
_N_SENSORS_TVB = 65  # TVB ships a 65-sensor BEM surface EEG projection
_N_NODES = 80  # neurolib HCP connectome
_N_NODES_TVB = 76  # TVB default connectivity_76.zip (AAL atlas)


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


# ---------------------------------------------------------------------------
# TVBFHNPlant tests
# ---------------------------------------------------------------------------


def _make_tvb_plant(**kwargs: object) -> TVBFHNPlant:
    base: dict[str, object] = {"dt": _DT}
    base.update(kwargs)
    return TVBFHNPlant(**base)  # type: ignore


def test_tvb_output_shapes() -> None:
    """TVBFHNPlant returns activity (n_nodes, n_samples) and EEG (n_sensors, n_samples)."""
    plant = _make_tvb_plant(seed=0)
    activity, eeg = plant.step(100.0)
    n_samples = round(100.0 / _DT)
    assert activity.shape == (_N_NODES_TVB, n_samples)
    assert eeg.shape == (_N_SENSORS_TVB, n_samples)


def test_tvb_leadfield_is_real_projection() -> None:
    """TVBFHNPlant uses TVB's deterministic BEM-derived region-averaged leadfield."""
    lf1 = TVBFHNPlant().leadfield
    lf2 = TVBFHNPlant().leadfield
    assert lf1.shape == (_N_SENSORS_TVB, _N_NODES_TVB)
    np.testing.assert_array_equal(lf1, lf2)
    # Real region-averaged BEM gain — rows are not unit-norm (would be ~1.0 for _make_leadfield).
    row_norms = np.linalg.norm(lf1, axis=1)
    assert not np.allclose(row_norms, 1.0)


def test_tvb_deterministic_consistency() -> None:
    """Two TVBFHNPlants with the same seed and nsig=0 produce identical outputs."""
    p1 = _make_tvb_plant(nsig=0.0, seed=42)
    p2 = _make_tvb_plant(nsig=0.0, seed=42)
    a1, e1 = p1.step(100.0)
    a2, e2 = p2.step(100.0)
    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_array_equal(e1, e2)


def test_tvb_multistep_continuity() -> None:
    """Two 50 ms steps (noiseless) concatenate to the same array as one 100 ms step."""
    p_single = _make_tvb_plant(nsig=0.0, seed=0)
    p_multi = _make_tvb_plant(nsig=0.0, seed=0)
    act_single, _ = p_single.step(100.0)
    act_a, _ = p_multi.step(50.0)
    act_b, _ = p_multi.step(50.0)
    np.testing.assert_array_equal(act_single, np.concatenate([act_a, act_b], axis=1))


@pytest.mark.parametrize("nsig", [0.0, 0.05])
def test_tvb_reset_reproduces_first_step(nsig: float) -> None:
    """After reset(), TVBFHNPlant reproduces the same output as the first step."""
    plant = _make_tvb_plant(nsig=nsig, seed=7)
    act_first, _ = plant.step(100.0)
    plant.reset()
    act_second, _ = plant.step(100.0)
    np.testing.assert_array_equal(act_first, act_second)


def test_tvb_activity_is_finite() -> None:
    """TVBFHNPlant output is finite when noise is non-zero."""
    plant = _make_tvb_plant(nsig=0.01, seed=1)
    activity, eeg = plant.step(200.0)
    assert np.isfinite(activity).all()
    assert np.isfinite(eeg).all()


def test_tvb_default_connectome_is_76_nodes() -> None:
    """TVBFHNPlant defaults to TVB's 76-node AAL atlas (connectivity_76.zip)."""
    assert TVBFHNPlant().n_nodes == _N_NODES_TVB
