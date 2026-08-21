"""Pin the observable artifact's NumPy forecast against the torch module and the CasADi bridge."""

from __future__ import annotations

from typing import TYPE_CHECKING

import casadi as ca
import numpy as np
import pytest

from neuro.config import EegMsGeometry, StftGeometry
from neuro.observable_casadi import ObservableSymbolicModel
from neuro.predictor.observable_module import ObservableMLP

if TYPE_CHECKING:
    from collections.abc import Callable

    from neuro.observable import ObservableArtifact
    from neuro.predictor.artifact import Activation
    from neuro.types import FloatArray

_SEED = 31

_GEOMETRIES = [
    StftGeometry(n_segment=8, n_hop=4),
    StftGeometry(n_segment=8, n_hop=8),
    StftGeometry(n_segment=10, n_hop=5, n_bin_pool=2),
    StftGeometry(n_segment=8, n_hop=4, kernel="hann", kernel_width=2),
    EegMsGeometry(window_s=0.16, hop_s=0.08),
]


def _inputs(art: ObservableArtifact, horizon: int) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Random history and future controls in raw units, plus the primed state."""
    rng = np.random.default_rng(_SEED)
    y_hist = rng.standard_normal((art.n_y, art.n_channels))
    u_hist = rng.standard_normal((art.n_u, art.n_controls))
    u_future = rng.standard_normal((horizon, art.n_controls))
    return y_hist, u_hist, u_future, art.prime(y_hist, u_hist)


@pytest.mark.parametrize("activation", ["relu", "tanh", "softplus"])
@pytest.mark.parametrize("geometry", _GEOMETRIES)
def test_casadi_forecast_matches_the_artifact(
    geometry: StftGeometry | EegMsGeometry,
    activation: Activation,
    make_observable_artifact: Callable[..., ObservableArtifact],
) -> None:
    """The control path never runs torch, so the NumPy and CasADi forecasts must be one function.

    They are independent implementations of the same recursion, both in float64, so they agree to
    round-off rather than merely to a tolerance.
    """
    horizon = 24
    art = make_observable_artifact(geometry, horizon=horizon, activation=activation)
    _, _, u_future, state = _inputs(art, horizon)

    want = art.forecast(state, u_future)
    got = np.asarray(ca.DM(ObservableSymbolicModel(art).f_forecast(state, u_future)))

    assert got.shape == (art.n_channels * art.n_values, art.n_frames())
    np.testing.assert_allclose(got.T.reshape(want.shape), want, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("activation", ["relu", "tanh", "softplus"])
@pytest.mark.parametrize("geometry", _GEOMETRIES)
def test_torch_forecast_matches_the_artifact(
    geometry: StftGeometry | EegMsGeometry,
    activation: Activation,
    make_observable_artifact: Callable[..., ObservableArtifact],
) -> None:
    """The torch module the trainer optimises forecasts what the frozen artifact later replays.

    The torch side runs in float32, which sets the tolerance; the artifact's averaging of *raw*
    controls and the module's averaging of *standardized* ones agree because both maps are affine.
    """
    import torch  # noqa: PLC0415 -- kept out of module scope so the control-path guard stays honest

    horizon = 24
    art = make_observable_artifact(geometry, horizon=horizon, activation=activation)
    y_hist, u_hist, u_future, state = _inputs(art, horizon)

    row = np.concatenate(
        [
            art.encode(y_hist).reshape(-1),
            art.u_std.transform(u_hist).reshape(-1),
            art.u_std.transform(u_future).reshape(-1),
        ]
    )
    module = ObservableMLP.from_artifact(art)
    with torch.no_grad():
        standardized = module(torch.as_tensor(row[None], dtype=torch.float32)).numpy()[0].astype(np.float64)
    got = art.l_std.inverse_transform(standardized).reshape(art.n_frames(), art.n_channels, art.n_values)

    np.testing.assert_allclose(got, art.forecast(state, u_future), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("horizon", [16, 20, 24, 32])
def test_forecast_frame_count_follows_the_horizon_not_a_frozen_head(
    horizon: int, make_observable_artifact: Callable[..., ObservableArtifact]
) -> None:
    """An artifact trained at one Frame count deploys at another: geometry, not a frozen horizon."""
    geometry = StftGeometry(n_segment=8, n_hop=4)
    art = make_observable_artifact(geometry, horizon=16)
    _, _, u_future, state = _inputs(art, horizon)

    model = ObservableSymbolicModel(art)
    x_sym = ca.MX.sym("x", *model.state_shape)
    u_sym = ca.MX.sym("u", horizon, model.n_controls)
    forecast = ca.Function("f", [x_sym, u_sym], [model.forecast(x_sym, u_sym)])

    expected_frames = geometry.n_frames(horizon, art.fs)
    assert model.n_frames(horizon) == expected_frames
    got = np.asarray(ca.DM(forecast(state, u_future)))
    assert got.shape == (art.n_channels * art.n_values, expected_frames)
    np.testing.assert_allclose(
        got.T.reshape(expected_frames, art.n_channels, art.n_values),
        art.forecast(state, u_future),
        rtol=1e-10,
        atol=1e-12,
    )


def test_state_seam_carries_over_from_the_autoregressive_model(
    make_observable_artifact: Callable[..., ObservableArtifact],
) -> None:
    """Priming, State Absorption and readiness behave exactly as they do on the incumbent path."""
    art = make_observable_artifact(StftGeometry(n_segment=8, n_hop=4), horizon=16, n_y=3, n_u=2)
    model = ObservableSymbolicModel(art)
    rng = np.random.default_rng(_SEED + 4)

    state = model.initial_state()
    assert not model.is_ready(state)
    for _ in range(art.n_y):
        state = model.absorb(state, rng.standard_normal(art.n_channels), np.zeros(art.n_controls))
    assert model.is_ready(state)
    assert state.shape == (model.state_shape[0],)
