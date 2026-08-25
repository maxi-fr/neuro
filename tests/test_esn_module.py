"""Seam 1 -- the torch ESN module over the Predictor protocol, matching the numpy runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING

import casadi as ca
import numpy as np
import pytest
import torch

from neuro.esn import ESNArtifact, generate_reservoir, harvest_normal_equations, solve_ridge
from neuro.esn_predictor_casadi import ESNSymbolicModel
from neuro.predictor.esn_module import ESNModule
from neuro.predictor.ridge import RidgeTrainer
from neuro.transforms import Standardizer
from neuro.types import Predictor, RidgeFittable

if TYPE_CHECKING:
    from pathlib import Path

    from neuro.types import FloatArray

_SEED = 17
_N_RES, _N_EEG, _N_CONTROLS = 50, 2, 2
_LEAK, _RHO, _DENSITY, _IN_SCALE = 0.1, 0.9, 0.1, 0.5
_PRIMING, _HORIZON = 10, 5
_STEPS = 8  # rollout length, deliberately different from the trained horizon
# float32 runtime vs float64 numpy: the reservoir recurrence accumulates rounding, so looser
# than the MLP's 1e-5/1e-6 while still 100x below any real divergence.
_RTOL, _ATOL = 1e-4, 1e-5
_RIDGE_RTOL, _RIDGE_ATOL = 1e-8, 1e-9  # both solves run in float64 (LAPACK)


def _artifact() -> ESNArtifact:
    """A random ESN artifact with nontrivial standardizers, matching the module's weights."""
    rng = np.random.default_rng(_SEED)
    y_std = Standardizer(center=rng.uniform(-1.0, 1.0, _N_EEG), scale=rng.uniform(0.5, 2.0, _N_EEG))
    u_std = Standardizer(center=rng.uniform(-1.0, 1.0, _N_CONTROLS), scale=rng.uniform(0.5, 2.0, _N_CONTROLS))
    w_res, w_in = generate_reservoir(
        reservoir_size=_N_RES,
        spectral_radius=_RHO,
        density=_DENSITY,
        input_scaling=_IN_SCALE,
        in_dim=_N_EEG + _N_CONTROLS + 1,
        seed=_SEED,
    )
    return ESNArtifact(
        w_in=w_in,
        w_out=rng.uniform(-0.1, 0.1, size=(_N_EEG, _N_RES + 1)),
        w_res=w_res,
        dt=0.02,
        downsample=200,
        horizon=_HORIZON,
        reservoir_size=_N_RES,
        leak_rate=_LEAK,
        spectral_radius=_RHO,
        priming_steps=_PRIMING,
        input_scaling=_IN_SCALE,
        density=_DENSITY,
        noise_sigma=0.0,
        ridge_lambda=1e-3,
        seed=_SEED,
        y_std=y_std,
        u_std=u_std,
    )


def _module() -> ESNModule:
    """The torch ESN module carrying the artifact's weights, buffers and generation metadata."""
    art = _artifact()
    return ESNModule(
        w_res=art.w_res,
        w_in=art.w_in,
        w_out=art.w_out,
        leak_rate=art.leak_rate,
        priming_steps=art.priming_steps,
        horizon=art.horizon,
        dt=art.dt,
        spectral_radius=art.spectral_radius,
        density=art.density,
        input_scaling=art.input_scaling,
        noise_sigma=art.noise_sigma,
        ridge_lambda=art.ridge_lambda,
        seed=art.seed,
        y_std=art.y_std,
        u_std=art.u_std,
    )


def _context(seed: int = _SEED + 1, steps: int = _STEPS) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Raw EEG history, raw control history and raw future controls, all ending/starting at the seam."""
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((_PRIMING, _N_EEG)),
        rng.standard_normal((_PRIMING, _N_CONTROLS)),
        rng.standard_normal((steps, _N_CONTROLS)),
    )


def _batch(n_batch: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Independently drawn ``(y_hists, u_hists, u_futures)`` -- no two members share a history."""
    rng = np.random.default_rng(_SEED + 100)
    return (
        rng.standard_normal((n_batch, _PRIMING, _N_EEG)),
        rng.standard_normal((n_batch, _PRIMING, _N_CONTROLS)),
        rng.standard_normal((n_batch, _STEPS, _N_CONTROLS)),
    )


def test_esn_module_satisfies_predictor_protocol() -> None:
    """The torch ESN module is a Predictor, and ``rollout_many`` returns ``(B, positions, outputs)``."""
    model = _module()
    assert isinstance(model, Predictor)
    assert model.n_outputs == _N_EEG
    assert model.n_channels == _N_EEG
    assert model.n_controls == _N_CONTROLS
    assert model.dt == 0.02
    assert model.priming_steps == _PRIMING
    assert model.horizon == _HORIZON

    y_hists, u_hists, u_futures = _batch(5)
    preds = model.rollout_many(model.prime_many(y_hists, u_hists), u_futures)
    assert preds.shape == (5, _STEPS, model.n_outputs)


def test_buffers_hold_weights_and_standardizers() -> None:
    """W_res stays sparse; W_in/W_out and the standardizers are float32 buffers; raw round-trips."""
    art = _artifact()
    model = _module()
    assert model.w_res.layout == torch.sparse_coo
    assert model.w_res.dtype == torch.float32
    assert model.w_in.dtype == torch.float32
    assert model.w_out.dtype == torch.float32
    assert model.y_center.dtype == torch.float32
    assert model.u_center.dtype == torch.float32
    np.testing.assert_allclose(model.w_in.numpy(), art.w_in, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(model.w_out.numpy(), art.w_out, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(model.w_res.to_dense().numpy(), art.w_res.toarray(), rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(model.y_std.center, art.y_std.center, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(model.u_std.center, art.u_std.center, rtol=_RTOL, atol=_ATOL)

    rng = np.random.default_rng(_SEED + 2)
    y = rng.standard_normal((4, _N_EEG))
    np.testing.assert_allclose(model.decode(model.encode(y)), y, rtol=_RTOL, atol=_ATOL)


def test_absorb_matches_numpy_runtime() -> None:
    """Torch ``absorb`` advances the reservoir exactly like ``ESNPredictor.absorb``, step by step."""
    art = _artifact()
    model = _module()
    rng = np.random.default_rng(_SEED + 3)
    y = rng.standard_normal((_PRIMING + 4, _N_EEG))
    u = rng.standard_normal((_PRIMING + 4, _N_CONTROLS))
    z = art.encode(y)
    v = art.u_std.transform(u)

    h_np = np.zeros(_N_RES)
    state = model.initial_state()
    for t in range(len(y)):
        h_np = art.predictor.absorb(h_np, z[t], v[t])
        state = model.absorb(state, y[t], u[t])
        np.testing.assert_allclose(state[:_N_RES], h_np, rtol=_RTOL, atol=_ATOL)
    assert state[_N_RES] == len(y)


def test_step_matches_numpy_runtime() -> None:
    """Torch ``step`` free-runs the same recurrence and emits the same raw output as the numpy runtime."""
    art = _artifact()
    model = _module()
    y_hist, u_hist, u_future = _context()

    state = model.prime(y_hist, u_hist)
    h_np = art.prime(y_hist, u_hist)
    for t in range(_STEPS):
        v = art.u_std.transform(u_future[t])
        z_hat_np = art.predictor.readout(h_np)
        want_y = art.decode(z_hat_np)
        h_np = art.predictor.step(h_np, v)

        state, got_y = model.step(state, u_future[t])
        np.testing.assert_allclose(got_y, want_y, rtol=_RTOL, atol=_ATOL)
        np.testing.assert_allclose(state[:_N_RES], h_np, rtol=_RTOL, atol=_ATOL)


def test_prime_matches_numpy_prime() -> None:
    """``prime`` reproduces the artifact's reservoir state and counts the absorbed history."""
    art = _artifact()
    model = _module()
    y_hist, u_hist, _ = _context()
    np.testing.assert_allclose(model.prime(y_hist, u_hist)[:_N_RES], art.prime(y_hist, u_hist), rtol=_RTOL, atol=_ATOL)


def test_rollout_matches_numpy_rollout() -> None:
    """``rollout`` reproduces the artifact's raw free-run to float32 tolerance."""
    art = _artifact()
    model = _module()
    y_hist, u_hist, u_future = _context()

    got = model.rollout(model.prime(y_hist, u_hist), u_future)
    want = art.rollout(art.prime(y_hist, u_hist), u_future)
    assert got.shape == (_STEPS, _N_EEG)
    np.testing.assert_allclose(got, want, rtol=_RTOL, atol=_ATOL)


def test_rollout_many_matches_numpy_rollout_many() -> None:
    """Batched ``prime_many``/``rollout_many`` reproduce the artifact's batched path."""
    art = _artifact()
    model = _module()
    y_hists, u_hists, u_futures = _batch(5)

    got = model.rollout_many(model.prime_many(y_hists, u_hists), u_futures)
    want = art.rollout_many(art.prime_many(y_hists, u_hists), u_futures)
    assert got.shape == (5, _STEPS, _N_EEG)
    np.testing.assert_allclose(got, want, rtol=_RTOL, atol=_ATOL)


@pytest.mark.parametrize("n_batch", [1, 5])
def test_prime_many_matches_a_loop_of_prime(n_batch: int) -> None:
    """``prime_many`` equals a loop of ``prime`` over per-member histories."""
    model = _module()
    y_hists, u_hists, _ = _batch(n_batch)

    batched = model.prime_many(y_hists, u_hists)
    looped = np.stack([model.prime(y_hists[i], u_hists[i]) for i in range(n_batch)])

    assert batched.shape == looped.shape
    np.testing.assert_allclose(batched, looped, rtol=_RTOL, atol=_ATOL)


@pytest.mark.parametrize("n_batch", [1, 5])
def test_rollout_many_matches_a_loop_of_rollout(n_batch: int) -> None:
    """``rollout_many`` equals a loop of ``rollout`` from per-member states."""
    model = _module()
    y_hists, u_hists, u_futures = _batch(n_batch)

    states = np.stack([model.prime(y_hists[i], u_hists[i]) for i in range(n_batch)])
    batched = model.rollout_many(states, u_futures)
    looped = np.stack([model.rollout(states[i], u_futures[i]) for i in range(n_batch)])

    assert batched.shape == (n_batch, _STEPS, model.n_outputs)
    np.testing.assert_allclose(batched, looped, rtol=_RTOL, atol=_ATOL)


def test_rollout_equals_a_loop_of_step() -> None:
    """``rollout`` is exactly one ``step`` per position, emitting raw output at each."""
    model = _module()
    y_hist, u_hist, u_future = _context()

    state = model.prime(y_hist, u_hist)
    got = model.rollout(state, u_future)

    want = np.empty((_STEPS, _N_EEG), dtype=np.float64)
    for t in range(_STEPS):
        state, y = model.step(state, u_future[t])
        want[t] = y

    np.testing.assert_allclose(got, want, rtol=_RTOL, atol=_ATOL)


def test_initial_state_is_ready_respect_priming_steps() -> None:
    """``initial_state`` starts not-ready; ``prime`` is ready exactly past ``priming_steps``."""
    model = _module()
    state = model.initial_state()
    assert not model.is_ready(state)
    assert (state[:_N_RES] == 0).all()
    assert state[_N_RES] == 0

    rng = np.random.default_rng(_SEED + 5)
    y = rng.standard_normal((_PRIMING + 1, _N_EEG))
    u = rng.standard_normal((_PRIMING + 1, _N_CONTROLS))
    for k in (1, _PRIMING - 1, _PRIMING, _PRIMING + 1):
        state = model.prime(y[:k], u[:k])
        assert model.is_ready(state) == (k >= _PRIMING)
        assert state[_N_RES] == k


def test_absorb_counts_toward_ready() -> None:
    """Absorbing ``priming_steps`` measurements from ``initial_state`` makes the state ready."""
    model = _module()
    rng = np.random.default_rng(_SEED + 6)
    state = model.initial_state()
    y = rng.standard_normal((_PRIMING + 2, _N_EEG))
    u = rng.standard_normal((_PRIMING + 2, _N_CONTROLS))
    for t in range(len(y)):
        assert model.is_ready(state) == (t >= _PRIMING)
        state = model.absorb(state, y[t], u[t])
    assert model.is_ready(state)


def test_solve_ridge_reproduces_the_incumbent() -> None:
    """The torch ridge solve equals ``solve_ridge`` on a real harvest to LAPACK precision."""
    art = _artifact()
    rng = np.random.default_rng(_SEED + 7)
    y = rng.standard_normal((60, _N_EEG))
    u = rng.standard_normal((60, _N_CONTROLS))
    G, P = harvest_normal_equations(
        trajectories=[(u, y)],
        y_std=art.y_std,
        u_std=art.u_std,
        w_res=art.w_res,
        w_in=art.w_in,
        leak_rate=art.leak_rate,
        priming_steps=art.priming_steps,
        noise_sigma=0.0,
        seed=_SEED,
    )
    model = _module()
    np.testing.assert_allclose(
        model.solve_ridge(G, P, 1e-3), solve_ridge(G, P, 1e-3), rtol=_RIDGE_RTOL, atol=_RIDGE_ATOL
    )


def test_design_normal_equations_reproduces_the_incumbent_harvest() -> None:
    """Streaming G/P equals the incumbent numpy harvest at noise_sigma = 0, to float32 tolerance."""
    art = _artifact()
    model = _module()
    rng = np.random.default_rng(_SEED + 7)
    y = rng.standard_normal((60, _N_EEG))
    u = rng.standard_normal((60, _N_CONTROLS))

    G, P = model.design_normal_equations([(u, y)])
    G_want, P_want = harvest_normal_equations(
        trajectories=[(u, y)],
        y_std=art.y_std,
        u_std=art.u_std,
        w_res=art.w_res,
        w_in=art.w_in,
        leak_rate=art.leak_rate,
        priming_steps=art.priming_steps,
        noise_sigma=0.0,
        seed=_SEED,
    )
    np.testing.assert_allclose(G, G_want, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(P, P_want, rtol=_RTOL, atol=_ATOL)


def test_install_readout_writes_w_out() -> None:
    """``install_readout`` overwrites the ``w_out`` buffer with the fitted readout, bias column last."""
    model = _module()
    rng = np.random.default_rng(_SEED + 9)
    A = rng.standard_normal((_N_EEG, _N_RES + 1))
    model.install_readout(A)
    np.testing.assert_array_equal(model.w_out.numpy(), A.astype(np.float32))


def test_ridge_trainer_fits_the_esn_end_to_end() -> None:
    """The Ridge Trainer on the ESN reproduces the incumbent design -> solve -> install pipeline."""
    _artifact()
    model = _module()
    assert isinstance(model, RidgeFittable)
    rng = np.random.default_rng(_SEED + 10)
    y = rng.standard_normal((80, _N_EEG))
    u = rng.standard_normal((80, _N_CONTROLS))

    RidgeTrainer(ridge_lambda=1e-3).fit(model, [(u, y)])
    G, P = model.design_normal_equations([(u, y)])
    w_want = solve_ridge(G, P, 1e-3)
    np.testing.assert_allclose(model.w_out.numpy(), w_want.astype(np.float32), rtol=_RTOL, atol=_ATOL)


def test_bias_column_is_unregularized() -> None:
    """On a diagonal G the solve decouples: the last column sees no ridge, the others do."""
    model = _module()
    rng = np.random.default_rng(_SEED + 8)
    n = model.reservoir_size + 1
    diag = rng.uniform(1.0, 10.0, n)
    diag[-1] = 100.0  # large so the ridge would be visible if it applied
    lam = 5.0
    G = np.diag(diag)
    P = rng.standard_normal((n, _N_EEG))

    W = model.solve_ridge(G, P, lam)
    np.testing.assert_allclose(W[:, -1], P[-1] / diag[-1], rtol=_RIDGE_RTOL, atol=_RIDGE_ATOL)
    for i in range(n - 1):
        np.testing.assert_allclose(W[:, i], P[i] / (diag[i] + lam), rtol=_RIDGE_RTOL, atol=_RIDGE_ATOL)


def test_rollout_accepts_any_length_not_just_the_native_horizon() -> None:
    """``rollout`` is not bounded by the trained ``horizon``; the identity stays available."""
    model = _module()
    y_hist, u_hist, _ = _context()

    state = model.prime(y_hist, u_hist)
    long = model.rollout(state, np.zeros((_HORIZON + 3, _N_CONTROLS)))

    assert long.shape == (_HORIZON + 3, _N_EEG)


@pytest.mark.parametrize("steps", [3, 8, 16])
def test_casadi_adapter_rollout_matches_the_checkpoint_float64_rollout(tmp_path: Path, steps: int) -> None:
    """The CasADi bridge rebuilt from the module's checkpoint reproduces the float64 rollout.

    The module stores float32 weights; the torch-free reader and the artifact loader both cast them
    to float64, so the f_step/f_out chain and the reference are the same arithmetic to round-off.
    """
    model = _module()
    path = tmp_path / "esn_module"
    model.save(path)

    sym = ESNSymbolicModel.from_checkpoint(path)
    art = ESNArtifact.load(path)
    y_hist, u_hist, u_future = _context(steps=steps)

    want = art.rollout(art.prime(y_hist, u_hist), u_future)
    x = art.prime(y_hist, u_hist)
    preds = []
    for u in u_future:
        preds.append(np.asarray(ca.DM(sym.f_out(x))).flatten())
        x = np.asarray(sym.f_step(x, u)).flatten()
    got = np.stack(preds)

    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)


def test_save_checkpoint_is_readable_by_the_torch_free_loader(tmp_path: Path) -> None:
    """``ESNModule.save`` writes the six generation/fit metadata keys, so the torch-free
    :class:`neuro.esn.ESNArtifact` loader reads the checkpoint and its runtime matches the module."""
    model = _module()
    path = tmp_path / "esn_module"
    model.save(path)

    loaded = ESNArtifact.load(path)
    assert loaded.spectral_radius == pytest.approx(_RHO)
    assert loaded.density == pytest.approx(_DENSITY)
    assert loaded.input_scaling == pytest.approx(_IN_SCALE)
    assert loaded.noise_sigma == pytest.approx(0.0)
    assert loaded.ridge_lambda == pytest.approx(1e-3)
    assert loaded.seed == _SEED
    assert loaded.reservoir_size == _N_RES
    assert loaded.leak_rate == pytest.approx(_LEAK)
    assert loaded.priming_steps == _PRIMING

    y_hist, u_hist, u_future = _context()
    want = model.rollout(model.prime(y_hist, u_hist), u_future)
    got = loaded.rollout(loaded.prime(y_hist, u_hist), u_future)
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=_RTOL, atol=_ATOL)
