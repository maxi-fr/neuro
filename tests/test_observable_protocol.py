"""Seam 1 -- the new one-Frame-per-step observable module over the Predictor protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from torch import nn

from neuro.config import EegMsGeometry, ObservableGeometry, StftGeometry
from neuro.observable import control_means
from neuro.predictor.module import AutoregressiveMLP
from neuro.predictor.observable_module import StepwiseObservableMLP
from neuro.transforms import Standardizer
from neuro.types import Predictor

if TYPE_CHECKING:
    from collections.abc import Callable

    from neuro.observable import ObservableArtifact
    from neuro.types import FloatArray

_SEED = 17
_N_Y, _N_U, _HORIZON = 3, 2, 5
_N_EEG, _N_CONTROLS = 5, 2
_Z_DIM, _HIDDEN, _DEPTH = 6, 8, 2
_FS = 50.0
_STEPS = 8  # rollout length, deliberately different from the trained horizon
_RTOL, _ATOL = 1e-5, 1e-6  # float32 tolerance


def _model(geometry: ObservableGeometry | None = None) -> StepwiseObservableMLP:
    """A randomly initialized module with nontrivial standardizers, no artifact in sight."""
    rng = np.random.default_rng(_SEED)
    geometry = geometry or StftGeometry(n_segment=3, n_hop=2)
    n_values = geometry.n_values(_FS)
    return StepwiseObservableMLP(
        n_y=_N_Y,
        n_u=_N_U,
        horizon=_HORIZON,
        n_channels=_N_EEG,
        n_controls=_N_CONTROLS,
        geometry=geometry,
        fs=_FS,
        z_dim=_Z_DIM,
        lift_hidden=_HIDDEN,
        lift_depth=_DEPTH,
        transition_hidden=_HIDDEN,
        transition_depth=_DEPTH,
        y_std=Standardizer(center=rng.uniform(-1.0, 1.0, _N_EEG), scale=rng.uniform(0.5, 2.0, _N_EEG)),
        u_std=Standardizer(center=rng.uniform(-1.0, 1.0, _N_CONTROLS), scale=rng.uniform(0.5, 2.0, _N_CONTROLS)),
        l_std=Standardizer(
            center=rng.uniform(-1.0, 1.0, _N_EEG * n_values), scale=rng.uniform(0.5, 2.0, _N_EEG * n_values)
        ),
    )


def _context(seed: int = _SEED + 1, steps: int = _STEPS) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Raw EEG history, raw control history and raw future controls, all ending/starting at the seam."""
    rng = np.random.default_rng(seed)
    k = max(_N_Y, _N_U)
    return (
        rng.standard_normal((k, _N_EEG)),
        rng.standard_normal((k, _N_CONTROLS)),
        rng.standard_normal((steps, _N_CONTROLS)),
    )


def _batch(n_batch: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Independently drawn ``(y_hists, u_hists, u_futures)`` -- no two members share a history."""
    rng = np.random.default_rng(_SEED + 100)
    k = max(_N_Y, _N_U)
    return (
        rng.standard_normal((n_batch, k, _N_EEG)),
        rng.standard_normal((n_batch, k, _N_CONTROLS)),
        rng.standard_normal((n_batch, _STEPS, _N_CONTROLS)),
    )


def test_observable_mlp_satisfies_predictor_protocol() -> None:
    """The module is a Predictor; ``step`` emits one Frame and ``rollout_many`` ``(B, frames, C * F)``."""
    model = _model()
    assert isinstance(model, Predictor)
    assert model.n_outputs == _N_EEG * model.geometry.n_values(_FS)
    assert model.n_channels == _N_EEG
    assert model.n_controls == _N_CONTROLS
    assert model.dt == 1.0 / _FS
    assert model.priming_steps == max(_N_Y, _N_U)
    assert model.horizon == _HORIZON

    y_hists, u_hists, u_futures = _batch(5)
    preds = model.rollout_many(model.prime_many(y_hists, u_hists), u_futures)
    assert preds.shape == (5, model.geometry.n_frames(_STEPS, _FS), model.n_outputs)


def test_standardizers_are_buffers_and_round_trip_raw() -> None:
    """The standardizers are float32 buffers; raw units round-trip at the boundary."""
    model = _model()
    assert model.y_center.dtype == torch.float32
    assert model.u_center.dtype == torch.float32
    assert model.l_center.dtype == torch.float32

    rng = np.random.default_rng(_SEED + 2)
    y = rng.standard_normal((4, _N_EEG))
    np.testing.assert_allclose(model.decode(model.encode(y)), y, rtol=_RTOL, atol=_ATOL)
    log_obs = rng.standard_normal((4, model.n_outputs))
    np.testing.assert_allclose(
        model.l_std.inverse_transform(model.l_std.transform(log_obs)), log_obs, rtol=_RTOL, atol=_ATOL
    )


def test_step_advances_exactly_one_frame_under_a_frame_mean_control() -> None:
    """``step`` consumes one ``(n_controls,)`` Frame mean, emits ``(C * F,)`` raw log-Observable."""
    model = _model()
    y_hist, u_hist, u_future = _context()
    state = model.prime(y_hist, u_hist)
    n_hist = _N_Y * _N_EEG + _N_U * _N_CONTROLS

    u_bar = control_means(model.geometry, len(u_future), _FS) @ u_future
    state_next, output = model.step(state, u_bar[0])

    assert output.shape == (model.n_outputs,)
    np.testing.assert_allclose(state_next[:n_hist], state[:n_hist], rtol=_RTOL, atol=_ATOL)
    assert not np.allclose(state_next[n_hist:], state[n_hist:])


@pytest.mark.parametrize("n_batch", [1, 5])
def test_prime_many_matches_a_loop_of_prime(n_batch: int) -> None:
    """``prime_many`` equals a loop of ``prime`` over per-member histories."""
    model = _model()
    y_hists, u_hists, _ = _batch(n_batch)

    batched = model.prime_many(y_hists, u_hists)
    looped = np.stack([model.prime(y_hists[i], u_hists[i]) for i in range(n_batch)])

    assert batched.shape == looped.shape
    np.testing.assert_allclose(batched, looped, rtol=_RTOL, atol=_ATOL)


@pytest.mark.parametrize("n_batch", [1, 5])
def test_rollout_many_matches_a_loop_of_rollout(n_batch: int) -> None:
    """``rollout_many`` equals a loop of ``rollout`` from per-member states."""
    model = _model()
    y_hists, u_hists, u_futures = _batch(n_batch)

    states = np.stack([model.prime(y_hists[i], u_hists[i]) for i in range(n_batch)])
    batched = model.rollout_many(states, u_futures)
    looped = np.stack([model.rollout(states[i], u_futures[i]) for i in range(n_batch)])

    n_frames = model.geometry.n_frames(_STEPS, _FS)
    assert batched.shape == (n_batch, n_frames, model.n_outputs)
    np.testing.assert_allclose(batched, looped, rtol=_RTOL, atol=_ATOL)


def test_rollout_equals_a_loop_of_step_over_control_means() -> None:
    """``rollout`` is one ``step`` per Frame, on Frame means aggregated via ``control_means``."""
    model = _model()
    y_hist, u_hist, u_future = _context()

    state = model.prime(y_hist, u_hist)
    got = model.rollout(state, u_future)

    u_bar = control_means(model.geometry, len(u_future), _FS) @ u_future
    want = np.empty((len(u_bar), model.n_outputs), dtype=np.float64)
    for m in range(len(u_bar)):
        state, output = model.step(state, u_bar[m])
        want[m] = output

    np.testing.assert_allclose(got, want, rtol=_RTOL, atol=_ATOL)


def test_absorb_is_ready_initial_state_mirror_the_waveform_shift_register() -> None:
    """``initial_state``/``absorb``/``is_ready`` match the waveform module's NaN-padded register."""
    model = _model()
    wave = AutoregressiveMLP(
        n_y=_N_Y,
        n_u=_N_U,
        horizon=_HORIZON,
        n_channels=_N_EEG,
        n_controls=_N_CONTROLS,
        hidden_size=4,
        depth=1,
        y_std=model.y_std,
        u_std=model.u_std,
    )
    n_hist = _N_Y * _N_EEG + _N_U * _N_CONTROLS

    state = model.initial_state()
    want_state = wave.initial_state()
    np.testing.assert_array_equal(state[:n_hist], want_state)
    assert np.isnan(state[: _N_Y * _N_EEG]).all()
    assert not model.is_ready(state)

    rng = np.random.default_rng(_SEED + 5)
    y = rng.standard_normal((_N_Y + 2, _N_EEG))
    u = rng.standard_normal((_N_Y + 2, _N_CONTROLS))
    for t in range(len(y)):
        state = model.absorb(state, y[t], u[t])
        want_state = wave.absorb(want_state, y[t], u[t])
        np.testing.assert_allclose(state[:n_hist], want_state, rtol=_RTOL, atol=_ATOL)
        assert model.is_ready(state) == wave.is_ready(want_state)
    assert model.is_ready(state)


def test_prime_register_matches_the_waveform_prime() -> None:
    """``prime`` builds the same register the waveform module does, then lifts once."""
    model = _model()
    wave = AutoregressiveMLP(
        n_y=_N_Y,
        n_u=_N_U,
        horizon=_HORIZON,
        n_channels=_N_EEG,
        n_controls=_N_CONTROLS,
        hidden_size=4,
        depth=1,
        y_std=model.y_std,
        u_std=model.u_std,
    )
    y_hist, u_hist, _ = _context()
    n_hist = _N_Y * _N_EEG + _N_U * _N_CONTROLS

    np.testing.assert_allclose(model.prime(y_hist, u_hist)[:n_hist], wave.prime(y_hist, u_hist), rtol=_RTOL, atol=_ATOL)


def test_absorbed_state_matches_prime_of_the_same_history() -> None:
    """``initial_state`` plus ``priming_steps`` absorbs equals ``prime`` of those same samples."""
    model = _model()
    rng = np.random.default_rng(_SEED + 6)
    k = max(_N_Y, _N_U)
    y = rng.standard_normal((k, _N_EEG))
    u = rng.standard_normal((k, _N_CONTROLS))

    state = model.initial_state()
    for t in range(k):
        state = model.absorb(state, y[t], u[t])

    np.testing.assert_allclose(state, model.prime(y, u), rtol=_RTOL, atol=_ATOL)


def test_forward_inverse_standardized_matches_rollout() -> None:
    """The training ``forward`` unrolls the same Frame recursion ``rollout`` does, at the native horizon."""
    model = _model()
    y_hist, u_hist, u_future = _context(steps=_HORIZON)

    row = np.concatenate(
        [
            model.encode(y_hist)[-_N_Y:].reshape(-1),
            model.u_std.transform(u_hist)[-_N_U:].reshape(-1),
            model.u_std.transform(u_future).reshape(-1),
        ]
    )
    state = model.prime(y_hist, u_hist)
    with torch.no_grad():
        standardized = model(torch.as_tensor(row[None], dtype=torch.float32)).numpy()[0].astype(np.float64)
    got = model.l_std.inverse_transform(standardized)
    want = model.rollout(state, u_future)

    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=_RTOL, atol=_ATOL)


@pytest.mark.parametrize(
    "geometry",
    [StftGeometry(n_segment=3, n_hop=2), EegMsGeometry(window_s=0.1, hop_s=0.04)],
)
def test_frame_count_follows_the_geometry_across_horizons(geometry: ObservableGeometry) -> None:
    """The emitted Frame count is the geometry's own, at horizons both below and above the trained one."""
    model = _model(geometry)
    y_hist, u_hist, _ = _context()
    state = model.prime(y_hist, u_hist)

    for horizon in [5, 8, 11, 15]:
        rng = np.random.default_rng(_SEED + horizon)
        u_future = rng.standard_normal((horizon, _N_CONTROLS))
        preds = model.rollout(state, u_future)
        assert preds.shape == (geometry.n_frames(horizon, _FS), model.n_outputs)


def test_rollout_accepts_any_length_not_just_the_native_horizon() -> None:
    """``rollout`` is not bounded by the trained ``horizon``; the identity stays available."""
    model = _model()
    y_hist, u_hist, _ = _context()

    state = model.prime(y_hist, u_hist)
    long = model.rollout(state, np.zeros((_HORIZON + 3, _N_CONTROLS)))

    assert long.shape == (model.geometry.n_frames(_HORIZON + 3, _FS), model.n_outputs)


def test_frame_m_is_invariant_to_controls_after_its_segment() -> None:
    """Frame ``m``'s prediction is invariant to raw controls landing after its Segment ends."""
    model = _model()
    rng = np.random.default_rng(_SEED + 3)
    k = max(_N_Y, _N_U)
    state = model.prime(rng.standard_normal((k, _N_EEG)), rng.standard_normal((k, _N_CONTROLS)))

    u = rng.standard_normal((_STEPS, _N_CONTROLS))
    base = model.rollout(state, u)

    supports = model.geometry.frame_supports(_STEPS, _FS)
    assert len(supports) > 1, "the property is vacuous with a single frame"
    for m, (_, end) in enumerate(supports[:-1]):
        perturbed = u.copy()
        perturbed[end:] += 10.0
        np.testing.assert_allclose(model.rollout(state, perturbed)[: m + 1], base[: m + 1], rtol=1e-12, atol=1e-12)
        assert not np.allclose(model.rollout(state, perturbed)[m + 1 :], base[m + 1 :])


def test_step_recursion_matches_the_incumbent_one_shot_forecast(
    make_observable_artifact: Callable[..., ObservableArtifact],
) -> None:
    """The new step recursion and the incumbent one-shot forecast are the same Frame math."""
    geometry = StftGeometry(n_segment=8, n_hop=4)
    horizon = 20
    art = make_observable_artifact(geometry, horizon=horizon)
    model = StepwiseObservableMLP(
        n_y=art.n_y,
        n_u=art.n_u,
        horizon=art.horizon,
        n_channels=art.n_channels,
        n_controls=art.n_controls,
        geometry=art.geometry,
        fs=art.fs,
        z_dim=art.z_dim,
        lift_hidden=art.lift[0][0].shape[0],
        lift_depth=len(art.lift) - 1,
        transition_hidden=art.transition[0][0].shape[0],
        transition_depth=len(art.transition) - 1,
        activation=art.activation,
        y_std=art.y_std,
        u_std=art.u_std,
        l_std=art.l_std,
    )
    with torch.no_grad():
        for block, weights in ((model.lift, art.lift), (model.transition, art.transition)):
            linears = (m for m in block if isinstance(m, nn.Linear))
            for lin, (w, b) in zip(linears, weights, strict=True):
                lin.weight.copy_(torch.as_tensor(w, dtype=torch.float32))
                lin.bias.copy_(torch.as_tensor(b, dtype=torch.float32))
        model.readout.weight.copy_(torch.as_tensor(art.readout[0], dtype=torch.float32))
        model.readout.bias.copy_(torch.as_tensor(art.readout[1], dtype=torch.float32))

    rng = np.random.default_rng(_SEED + 7)
    y_hist = rng.standard_normal((art.n_y, art.n_channels))
    u_hist = rng.standard_normal((art.n_u, art.n_controls))
    u_future = rng.standard_normal((horizon, art.n_controls))

    got = model.rollout(model.prime(y_hist, u_hist), u_future)
    want = art.forecast(art.prime(y_hist, u_hist), u_future).reshape(-1, art.n_channels * art.n_values)

    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=_RTOL, atol=_ATOL)
