"""Pin the observable checkpoint's float64 forecast against the torch module and the CasADi bridge."""

from __future__ import annotations

from typing import TYPE_CHECKING

import casadi as ca
import numpy as np
import pytest
from _predictor_reference import observable_forecast, observable_prime

from neuro.config import EegMsGeometry, StftGeometry
from neuro.observable_casadi import ObservableSymbolicModel
from neuro.predictor.observable_module import StepwiseObservableMLP

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from neuro.checkpoint import ObservableCheckpoint
    from neuro.types import Activation, FloatArray

_SEED = 31

_GEOMETRIES = [
    StftGeometry(n_segment=8, n_hop=4),
    StftGeometry(n_segment=8, n_hop=8),
    StftGeometry(n_segment=10, n_hop=5, n_bin_pool=2),
    StftGeometry(n_segment=8, n_hop=4, kernel="hann", kernel_width=2),
    EegMsGeometry(window_s=0.16, hop_s=0.08),
]


def _inputs(ckpt: ObservableCheckpoint, horizon: int) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Random history and future controls in raw units, plus the primed state."""

    rng = np.random.default_rng(_SEED)
    y_hist = rng.standard_normal((ckpt.n_y, ckpt.n_channels))
    u_hist = rng.standard_normal((ckpt.n_u, ckpt.n_controls))
    u_future = rng.standard_normal((horizon, ckpt.n_controls))
    return y_hist, u_hist, u_future, observable_prime(ckpt, y_hist, u_hist)


@pytest.mark.parametrize("activation", ["relu", "tanh", "softplus"])
@pytest.mark.parametrize("geometry", _GEOMETRIES)
def test_casadi_forecast_matches_the_checkpoint(
    tmp_path: Path,
    geometry: StftGeometry | EegMsGeometry,
    activation: Activation,
    make_observable_checkpoint: Callable[..., ObservableCheckpoint],
) -> None:
    """The control path never runs torch, so the checkpoint's float64 and CasADi forecasts agree.

    The adapter is rebuilt from the checkpoint's buffers and the reference is the same file read
    through the hand-rolled float64 recursion, so they are one recursion to round-off.
    """

    horizon = 24
    ckpt = make_observable_checkpoint(geometry, horizon=horizon, activation=activation)
    ckpt.save(tmp_path / "obs")
    _, _, u_future, state = _inputs(ckpt, horizon)

    want = observable_forecast(ckpt, state, u_future)
    got = np.asarray(ca.DM(ObservableSymbolicModel(ckpt).f_forecast(state, u_future)))

    assert got.shape == (ckpt.n_channels * ckpt.n_values, ckpt.n_frames())
    np.testing.assert_allclose(got.T.reshape(want.shape), want, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("activation", ["relu", "tanh", "softplus"])
@pytest.mark.parametrize("geometry", _GEOMETRIES)
def test_torch_forecast_matches_the_checkpoint(
    tmp_path: Path,
    geometry: StftGeometry | EegMsGeometry,
    activation: Activation,
    make_observable_checkpoint: Callable[..., ObservableCheckpoint],
) -> None:
    """The float32 module the trainer optimises forecasts what the checkpoint later replays.

    The torch side runs in float32, which sets the tolerance; the checkpoint's averaging of *raw*
    controls and the module's averaging of *standardized* ones agree because both maps are affine.
    """
    import torch  # noqa: PLC0415 -- kept out of module scope so the control-path guard stays honest

    horizon = 24
    ckpt = make_observable_checkpoint(geometry, horizon=horizon, activation=activation)
    ckpt.save(tmp_path / "obs")
    y_hist, u_hist, u_future, state = _inputs(ckpt, horizon)

    row = np.concatenate(
        [
            ckpt.y_std.transform(y_hist).reshape(-1),
            ckpt.u_std.transform(u_hist).reshape(-1),
            ckpt.u_std.transform(u_future).reshape(-1),
        ]
    )
    module = StepwiseObservableMLP.load(tmp_path / "obs")
    with torch.no_grad():
        standardized = module(torch.as_tensor(row[None], dtype=torch.float32)).numpy()[0].astype(np.float64)
    got = ckpt.l_std.inverse_transform(standardized).reshape(ckpt.n_frames(), ckpt.n_channels, ckpt.n_values)

    np.testing.assert_allclose(got, observable_forecast(ckpt, state, u_future), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("horizon", [16, 20, 24, 32])
def test_forecast_frame_count_follows_the_horizon_not_a_frozen_head(
    tmp_path: Path,
    horizon: int,
    make_observable_checkpoint: Callable[..., ObservableCheckpoint],
) -> None:
    """A checkpoint trained at one Frame count deploys at another: geometry, not a frozen horizon."""

    geometry = StftGeometry(n_segment=8, n_hop=4)
    ckpt = make_observable_checkpoint(geometry, horizon=16)
    ckpt.save(tmp_path / "obs")
    _, _, u_future, state = _inputs(ckpt, horizon)

    model = ObservableSymbolicModel(ckpt)
    x_sym = ca.MX.sym("x", *model.state_shape)
    u_sym = ca.MX.sym("u", horizon, model.n_controls)
    forecast = ca.Function("f", [x_sym, u_sym], [model.forecast(x_sym, u_sym)])

    expected_frames = geometry.n_frames(horizon, ckpt.fs)
    assert model.n_frames(horizon) == expected_frames
    got = np.asarray(ca.DM(forecast(state, u_future)))
    assert got.shape == (ckpt.n_channels * ckpt.n_values, expected_frames)
    np.testing.assert_allclose(
        got.T.reshape(expected_frames, ckpt.n_channels, ckpt.n_values),
        observable_forecast(ckpt, state, u_future),
        rtol=1e-10,
        atol=1e-12,
    )


def test_state_seam_carries_over_from_the_autoregressive_model(
    make_observable_checkpoint: Callable[..., ObservableCheckpoint],
) -> None:
    """Priming, State Absorption and readiness behave exactly as they do on the incumbent path."""
    ckpt = make_observable_checkpoint(StftGeometry(n_segment=8, n_hop=4), horizon=16, n_y=3, n_u=2)
    model = ObservableSymbolicModel(ckpt)
    rng = np.random.default_rng(_SEED + 4)

    state = model.initial_state()
    assert not model.is_ready(state)
    for _ in range(ckpt.n_y):
        state = model.absorb(state, rng.standard_normal(ckpt.n_channels), np.zeros(ckpt.n_controls))
    assert model.is_ready(state)
    assert state.shape == (model.state_shape[0],)
