"""The observable training module's Ridge capability and its inference-side twin."""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from neuro.config import StftGeometry
from neuro.observable import control_means, log_observable
from neuro.predictor.data import build_dataset_for_trajectory, extract_future_windows
from neuro.predictor.inference import InferencePredictor, ObservableModel
from neuro.predictor.module import to_numpy
from neuro.predictor.observable_module import StepwiseObservableMLP
from neuro.predictor.ridge import RidgeTrainer
from neuro.types import RidgeFittable

if TYPE_CHECKING:
    from collections.abc import Callable

    from neuro.types import FloatArray

_SEED = 17
_N_Y, _N_U, _HORIZON = 3, 2, 5
_N_EEG, _N_CONTROLS = 5, 2
_Z_DIM, _HIDDEN = 6, 8
_FS = 50.0
_RTOL, _ATOL = 1e-5, 1e-6


def test_observable_jax_twin_is_an_inference_predictor() -> None:
    """The observable module's jax twin carries the inference ABC and the priming seam."""
    module = StepwiseObservableMLP(
        n_y=_N_Y,
        n_u=_N_U,
        horizon=_HORIZON,
        n_channels=_N_EEG,
        n_controls=_N_CONTROLS,
        geometry=StftGeometry(n_segment=3, n_hop=2),
        fs=_FS,
        z_dim=_Z_DIM,
        lift_hidden=_HIDDEN,
        lift_depth=2,
        transition_hidden=_HIDDEN,
        transition_depth=2,
    )
    model = ObservableModel.from_checkpoint(*module.to_checkpoint())
    assert isinstance(model, InferencePredictor)
    state = model.initial_state()
    assert not model.is_ready(state)


def test_depth0_ridge_fits_the_shared_readout_on_harvested_z_m() -> None:
    """A depth-0 observable ridge fit solves the harvested ``(z_m, target)`` pairs exactly.

    The pairs are re-derived through the module's own blocks, so the normal equations, the
    streaming and the write-back are all pinned against an explicit lstsq at once.
    """
    rng = np.random.default_rng(_SEED + 9)
    t_len = 200
    y = rng.standard_normal((t_len, _N_EEG))
    u = rng.standard_normal((t_len, _N_CONTROLS))
    model = StepwiseObservableMLP(
        n_y=_N_Y,
        n_u=_N_U,
        horizon=_HORIZON,
        n_channels=_N_EEG,
        n_controls=_N_CONTROLS,
        geometry=StftGeometry(n_segment=3, n_hop=2),
        fs=_FS,
        z_dim=_Z_DIM,
        lift_hidden=_HIDDEN,
        lift_depth=0,
        transition_hidden=_HIDDEN,
        transition_depth=0,
    )
    assert isinstance(model, RidgeFittable)

    RidgeTrainer(ridge_lambda=0.0).fit(model, [(u, y)])

    # Re-derive the harvested pairs through the module's own blocks.
    n_hist = _N_Y * _N_EEG + _N_U * _N_CONTROLS
    n_frames = model.n_frames()
    X, _ = build_dataset_for_trajectory(model.u_std.transform(u), model.y_std.transform(y), _N_Y, _N_U, _HORIZON)
    y_fut = extract_future_windows(y, _N_Y, _N_U, _HORIZON)
    targets = model.l_std.transform(
        log_observable(y_fut.reshape(-1, _HORIZON, _N_EEG), model.geometry, _FS).reshape(-1, n_frames, model.n_outputs)
    )
    n_samples = X.shape[0]
    z = model.lift(torch.as_tensor(X[:, :n_hist], dtype=torch.float32))
    u_bar = torch.einsum(
        "mt,btc->bmc",
        model.aggregate,
        torch.as_tensor(X[:, n_hist:].reshape(n_samples, _HORIZON, _N_CONTROLS), dtype=torch.float32),
    )
    H = np.empty((n_samples * n_frames, _Z_DIM + 1))
    H[:, -1] = 1.0
    for m in range(n_frames):
        z = model.transition(torch.cat([z, u_bar[:, m]], dim=1))
        H[m * n_samples : (m + 1) * n_samples, :_Z_DIM] = to_numpy(z)
    T = targets.transpose(1, 0, 2).reshape(-1, model.n_outputs)
    # The residual readout fits per-Frame *deltas*: Frame m's target is its difference from the
    # previous Frame's target, with a zero baseline before the first Frame.
    T = T.copy()
    T[n_samples:] -= T[:-n_samples]

    A_want = np.linalg.lstsq(H, T, rcond=None)[0].T
    installed = np.hstack([model.readout.weight.detach().numpy(), model.readout.bias.detach().numpy()[:, None]])
    np.testing.assert_allclose(installed, A_want, rtol=_RTOL, atol=_ATOL)


def test_rollout_chains_the_recursion_the_controller_steps(
    make_observable_model: Callable[..., ObservableModel],
) -> None:
    """The stateless free run and the controller's absorb -> ``discrete_dynamics`` loop agree.

    ``rollout`` seeds its state the way ``absorb`` does and advances it with the same Frame
    transition, so the free-run scores and the trajectory the MPC optimizes over are one recursion
    rather than two that can drift.
    """
    span = 24
    model = make_observable_model(StftGeometry(n_segment=8, n_hop=4), horizon=span)
    rng = np.random.default_rng(_SEED + 31)
    k = model.priming_steps
    y_hist = rng.standard_normal((k, model.n_channels))
    u_hist = rng.standard_normal((k, model.n_controls))
    u_future = rng.standard_normal((span, model.n_controls))

    state = model.initial_state()
    for y_t, u_t in zip(y_hist, u_hist, strict=True):
        state = model.absorb(state, y_t, u_t)
    assert model.is_ready(state)

    x = jnp.asarray(state)
    stepped = []
    for u_bar in control_means(model.geometry, span, model.fs) @ u_future:
        x = model.discrete_dynamics(x, jnp.asarray(u_bar), 0.0, model.dt)
        stepped.append(np.asarray(model.output(x)))

    got = np.asarray(model.free_run(y_hist[None], u_hist[None], u_future[None]))[0]
    np.testing.assert_allclose(got, np.stack(stepped), rtol=_RTOL, atol=_ATOL)


def test_residual_carry_accumulates_frame_deltas(
    make_observable_model: Callable[..., ObservableModel],
) -> None:
    """With the residual the Frame forecast accumulates: a constant readout ramps per Frame.

    A readout emitting a constant standardized delta makes the recursion add it once per Frame, so
    Frame ``m`` decodes to ``(m + 1) * b`` in standardized space -- the persistence prior applied
    to the previous Frame's prediction, not to the mean.
    """
    span = 24
    model = make_observable_model(StftGeometry(n_segment=8, n_hop=4), horizon=span)
    assert model.residual
    model = eqx.tree_at(
        lambda m: (m.readout_w, m.readout_b),
        model,
        (jnp.zeros_like(model.readout_w), jnp.full_like(model.readout_b, 0.3)),
    )

    rng = np.random.default_rng(_SEED + 37)
    k = model.priming_steps
    y_hist = rng.standard_normal((k, model.n_channels))
    u_hist = rng.standard_normal((k, model.n_controls))
    u_future = rng.standard_normal((span, model.n_controls))

    got = np.asarray(model.free_run(y_hist[None], u_hist[None], u_future[None]))[0]
    n_frames = model.n_frames(span)
    want = np.stack([np.asarray(model.l_center) + np.asarray(model.l_scale) * 0.3 * (m + 1) for m in range(n_frames)])
    np.testing.assert_allclose(got, want, rtol=_RTOL, atol=_ATOL)

    # The carried level lives in the opaque state, after the register and the lifted Frame state.
    state = model.absorb(model.initial_state(), y_hist[-1], u_hist[-1])
    n_hist = model.n_y * model.n_channels + model.n_u * model.n_controls
    assert state.shape == (n_hist + model.z_dim + model.n_outputs,)
    np.testing.assert_array_equal(state[n_hist + model.z_dim :], np.zeros(model.n_outputs))


def test_without_residual_the_state_carries_no_level(
    make_observable_model: Callable[..., ObservableModel],
) -> None:
    """A non-residual observable model keeps the plain state layout: register, lift, no carry."""
    model = make_observable_model(StftGeometry(n_segment=8, n_hop=4), residual=False)
    n_hist = model.n_y * model.n_channels + model.n_u * model.n_controls
    assert model.initial_state().shape == (n_hist + model.z_dim,)
