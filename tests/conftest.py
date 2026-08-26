from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from neuro.predictor.inference import ObservableModel
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from collections.abc import Callable

    from neuro.config import ObservableGeometry
    from neuro.types import Activation


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add --runslow command line option."""
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip tests marked with @pytest.mark.slow unless --runslow is specified."""
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def _random_observable_params(
    geometry: ObservableGeometry,
    *,
    n_y: int,
    n_u: int,
    horizon: int,
    n_channels: int,
    activation: Activation,
    dt: float,
    residual: bool,
) -> dict[str, Any]:
    """Random lift/transition/readout weights and standardizers, shared by the two fixtures."""
    rng = np.random.default_rng(11)
    n_controls, z_dim, hidden, depth = 2, 5, 6, 2

    def layers(sizes: list[int]) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        return tuple(
            (rng.standard_normal((out, inp)) / np.sqrt(inp), rng.standard_normal(out) * 0.1)
            for inp, out in itertools.pairwise(sizes)
        )

    fs = 1.0 / dt
    n_values = geometry.n_values(fs)
    n_out = n_channels * n_values
    lift = layers([n_y * n_channels + n_u * n_controls, *[hidden] * depth, z_dim])
    transition = layers([z_dim + n_controls, *[hidden] * depth, z_dim])
    return {
        "n_y": n_y,
        "n_u": n_u,
        "horizon": horizon,
        "n_channels": n_channels,
        "n_controls": n_controls,
        "z_dim": z_dim,
        "n_values": n_values,
        "fs": fs,
        "downsample": 1,
        "activation": activation,
        "residual": residual,
        "geometry": geometry,
        "lift_weights": tuple(w for w, _ in lift),
        "lift_biases": tuple(b for _, b in lift),
        "transition_weights": tuple(w for w, _ in transition),
        "transition_biases": tuple(b for _, b in transition),
        "readout_w": rng.standard_normal((n_out, z_dim)) / np.sqrt(z_dim),
        "readout_b": rng.standard_normal(n_out) * 0.1,
        "y_center": rng.uniform(-1.0, 1.0, n_channels),
        "y_scale": rng.uniform(0.5, 2.0, n_channels),
        "u_center": rng.uniform(-1.0, 1.0, n_controls),
        "u_scale": rng.uniform(0.5, 2.0, n_controls),
        "l_center": rng.uniform(-1.0, 1.0, n_out),
        "l_scale": rng.uniform(0.5, 2.0, n_out),
    }


@pytest.fixture
def make_observable_model() -> Callable[..., ObservableModel]:
    """Factory for a tiny synthetic observable jax model with random weights and real standardizers."""

    def build(
        geometry: ObservableGeometry,
        *,
        n_y: int = 4,
        n_u: int = 3,
        horizon: int = 12,
        n_channels: int = 2,
        activation: Activation = "softplus",
        dt: float = 0.02,
        residual: bool = True,
    ) -> ObservableModel:
        return ObservableModel(
            **_random_observable_params(
                geometry,
                n_y=n_y,
                n_u=n_u,
                horizon=horizon,
                n_channels=n_channels,
                activation=activation,
                dt=dt,
                residual=residual,
            )
        )

    return build
