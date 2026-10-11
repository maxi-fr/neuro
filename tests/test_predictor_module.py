from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest

from neuro.predictor.mlp import WaveformMLPModel
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from neuro.types import Activation, FloatArray

_SEED = 17
_N_Y, _N_U, _HORIZON = 3, 2, 5
_N_EEG, _N_CONTROLS, _HIDDEN = 5, 2, 6


def _model(depth: int = 2, activation: Activation = "softplus") -> WaveformMLPModel:
    """A random module with nontrivial standardizers."""
    return WaveformMLPModel(
        n_y=_N_Y,
        n_u=_N_U,
        horizon=_HORIZON,
        n_channels=_N_EEG,
        n_controls=_N_CONTROLS,
        n_outputs=_N_EEG,
        hidden_size=_HIDDEN,
        depth=depth,
        activation=activation,
        dt=0.01,
    )


def _context(seed: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """One model-space input: standardized history plus standardized future controls."""
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((_N_Y, _N_EEG)).astype(np.float64),
        rng.standard_normal((_N_U, _N_CONTROLS)).astype(np.float64),
        rng.standard_normal((_HORIZON, _N_CONTROLS)).astype(np.float64),
    )


def test_forward_is_row_independent() -> None:
    """Stacking two contexts into one batch predicts the same as running them one at a time."""
    model = _model(2, "tanh")

    items = [_context(_SEED + offset) for offset in (2, 3)]
    y_hist_b = jnp.asarray(np.stack([c[0] for c in items]), dtype=jnp.float32)
    u_hist_b = jnp.asarray(np.stack([c[1] for c in items]), dtype=jnp.float32)
    u_future_b = jnp.asarray(np.stack([c[2] for c in items]), dtype=jnp.float32)
    batched = np.asarray(model(y_hist_b, u_hist_b, u_future_b))

    singles = np.stack(
        [
            np.asarray(
                model(
                    jnp.asarray(c[0][None], dtype=jnp.float32),
                    jnp.asarray(c[1][None], dtype=jnp.float32),
                    jnp.asarray(c[2][None], dtype=jnp.float32),
                )
            )[0]
            for c in items
        ]
    )

    assert batched.shape == (2, _HORIZON, _N_EEG)
    np.testing.assert_allclose(batched, singles, rtol=1e-5, atol=1e-6)


def test_residual_skip_makes_a_zero_mlp_predict_pure_persistence() -> None:
    """With the residual skip, a zero-weight MLP predicts the last window sample unchanged.

    ``z_{t+1} = layers(x) + z_t`` degenerates to the identity when ``layers`` is zero, so the
    whole free run is the constant last window sample -- the persistence prior the skip bakes in.
    """
    model = WaveformMLPModel(
        n_y=2,
        n_u=1,
        horizon=3,
        n_channels=2,
        n_controls=1,
        n_outputs=2,
        hidden_size=4,
        depth=2,
        activation="softplus",
        residual=True,
    )
    model.weights = tuple(jnp.zeros_like(w) for w in model.weights)
    model.biases = tuple(jnp.zeros_like(b) for b in model.biases)

    rng = np.random.default_rng(11)
    y_hist = jnp.asarray(rng.standard_normal((1, 2, 2)), dtype=jnp.float32)
    u_hist = jnp.asarray(rng.standard_normal((1, 1, 1)), dtype=jnp.float32)
    u_future = jnp.asarray(rng.standard_normal((1, 3, 1)), dtype=jnp.float32)
    pred = np.asarray(model(y_hist, u_hist, u_future))[0]
    z_t = np.asarray(y_hist[0, -1])
    assert pred.shape == (3, 2)
    np.testing.assert_allclose(pred, np.broadcast_to(z_t, (3, 2)), rtol=1e-6, atol=1e-7)


def test_without_residual_a_zero_mlp_predicts_zero() -> None:
    """Without the skip the same zero-weight MLP emits zeros: the skip is what carries the level."""
    model = WaveformMLPModel(
        n_y=2,
        n_u=1,
        horizon=3,
        n_channels=2,
        n_controls=1,
        n_outputs=2,
        hidden_size=4,
        depth=2,
        activation="softplus",
        residual=False,
    )
    model.weights = tuple(jnp.zeros_like(w) for w in model.weights)
    model.biases = tuple(jnp.zeros_like(b) for b in model.biases)

    rng = np.random.default_rng(12)
    y_hist = jnp.asarray(rng.standard_normal((1, 2, 2)), dtype=jnp.float32)
    u_hist = jnp.asarray(rng.standard_normal((1, 1, 1)), dtype=jnp.float32)
    u_future = jnp.asarray(rng.standard_normal((1, 3, 1)), dtype=jnp.float32)
    pred = np.asarray(model(y_hist, u_hist, u_future))[0]
    np.testing.assert_array_equal(pred, np.zeros((3, 2)))


def test_module_layer_dimensions() -> None:
    """The JAX MLP stores one weight/bias pair per hidden layer and its readout."""
    model = WaveformMLPModel(
        n_y=2,
        n_u=1,
        horizon=3,
        n_channels=4,
        n_controls=2,
        n_outputs=4,
        hidden_size=8,
        depth=2,
        activation="softplus",
    )
    assert [weight.shape for weight in model.weights] == [(8, 10), (8, 8), (4, 8)]
    assert [bias.shape for bias in model.biases] == [(8,), (8,), (4,)]
    assert model.activation == "softplus"


@pytest.mark.parametrize(
    "module",
    [
        "neuro.predictor.checkpoint",
        "neuro.predictor.inference",
        "neuro.control",
        "neuro.control.zero",
        "neuro.control.threshold",
        "neuro.control.schedule",
        "neuro.control.costs",
        "neuro.control.mpc",
    ],
)
def test_runtime_imports_do_not_load_legacy_framework(module: str) -> None:
    """Runtime and training imports work without loading the legacy framework."""
    code = f"import importlib, sys; importlib.import_module({module!r}); assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603


def test_autoregressive_mlp_supports_distinct_output_width() -> None:
    """WaveformMLPModel unrolls with n_outputs independent of n_channels."""
    model = WaveformMLPModel(
        n_y=2,
        n_u=1,
        horizon=3,
        n_channels=2,
        n_controls=1,
        n_outputs=6,
        hidden_size=8,
        depth=2,
        activation="softplus",
        residual=True,
    )
    assert model.n_outputs == 6
    assert model.n_channels == 2
    assert model.y_center.shape == (6,)
    assert model.y_scale.shape == (6,)

    # Zero-weight residual predicts persistence of length-6 output
    model.weights = tuple(jnp.zeros_like(w) for w in model.weights)
    model.biases = tuple(jnp.zeros_like(b) for b in model.biases)

    rng = np.random.default_rng(42)
    y_hist = jnp.asarray(rng.standard_normal((1, 2, 6)), dtype=jnp.float32)
    u_hist = jnp.asarray(rng.standard_normal((1, 1, 1)), dtype=jnp.float32)
    u_future = jnp.asarray(rng.standard_normal((1, 3, 1)), dtype=jnp.float32)
    pred = np.asarray(model(y_hist, u_hist, u_future))[0]
    assert pred.shape == (3, 6)
    z_last = np.asarray(y_hist[0, -1])
    np.testing.assert_allclose(pred, np.broadcast_to(z_last, (3, 6)), rtol=1e-6, atol=1e-7)


def test_standardizer_length_agrees_with_output_width() -> None:
    """The standardizer's length must agree with the model's output width."""
    # Conforming length passes
    m = WaveformMLPModel(
        n_y=2,
        n_u=1,
        horizon=3,
        n_channels=2,
        n_controls=1,
        n_outputs=6,
        hidden_size=8,
        depth=1,
        y_std=Standardizer(center=np.zeros(6), scale=np.ones(6)),
    )
    assert m.n_outputs == 6

    # Mismatched length raises naming the conflicting values
    with pytest.raises(ValueError, match=r"y_center/scale size \(4\) must equal model n_outputs \(6\)"):
        WaveformMLPModel(
            n_y=2,
            n_u=1,
            horizon=3,
            n_channels=2,
            n_controls=1,
            n_outputs=6,
            hidden_size=8,
            depth=1,
            y_std=Standardizer(center=np.zeros(4), scale=np.ones(4)),
        )
