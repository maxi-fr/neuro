from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neuro.config import (
    CurriculumMSESpec,
    LossSpecs,
    ModelConfig,
    NNPredictorConfig,
    SimulationConfig,
    StftGeometry,
    TrainingConfig,
)
from neuro.control.mpc import build_observable_problem
from neuro.predictor.cnn import ObservableCNNModel
from neuro.predictor.inference import InferencePredictor
from neuro.predictor.train import TrainingResult, train
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from pathlib import Path

    from neuro.types import Activation, FloatArray


def _model(
    *, n_values: int = 4, residual: bool = True, n_y: int = 3, n_controls: int = 1, activation: Activation = "relu"
) -> ObservableCNNModel:
    """Build a small structured CNN with nontrivial channel-frequency standardizers."""
    rng = np.random.default_rng(27 + n_values)
    geometry = StftGeometry(n_segment=6, n_hop=2) if n_values == 4 else StftGeometry(n_segment=2, n_hop=1, n_bin_pool=2)
    model = ObservableCNNModel(
        n_y=n_y,
        n_u=2,
        horizon=3,
        n_channels=2,
        n_controls=n_controls,
        n_outputs=2 * n_values,
        hidden_size=4,
        depth=2,
        kernel_size=3,
        frequency_kernel_size=4,
        activation=activation,
        residual=residual,
        dt=0.04,
        geometry=geometry,
        y_std=Standardizer(center=rng.normal(size=(2, n_values)), scale=rng.uniform(0.5, 2.0, (2, n_values))),
        u_std=Standardizer(center=np.full(n_controls, 0.4), scale=np.full(n_controls, 1.7)),
    )
    return eqx.tree_at(
        lambda m: (m.conv_weights, m.conv_biases, m.head_weights, m.head_biases),
        model,
        replace_fn=lambda tree: jax.tree.map(lambda x: jax.random.normal(jax.random.PRNGKey(91), x.shape) * 0.2, tree),
    )


def test_observable_cnn_training_jax_parity_and_frequency_order() -> None:
    """Causal time-frequency convolution preserves frequency order across the runtime boundary."""
    model = _model()
    runtime = ObservableCNNModel.from_checkpoint(*model.to_checkpoint())
    rng = np.random.default_rng(91)
    t0 = 7
    k = t0 - 1
    y_raw = rng.normal(size=(t0 + model.horizon, model.n_channels, model.n_outputs // model.n_channels))
    u_raw = rng.normal(size=(t0 + model.horizon, model.n_controls))
    y_hist = jnp.expand_dims(jnp.asarray(model.y_std.transform(y_raw[k - model.n_y + 1 : k + 1]), dtype=jnp.float32), 0)
    u_hist = jnp.expand_dims(jnp.asarray(model.u_std.transform(u_raw[k - model.n_u : k]), dtype=jnp.float32), 0)
    u_future = jnp.expand_dims(jnp.asarray(model.u_std.transform(u_raw[k : k + model.horizon]), dtype=jnp.float32), 0)
    expected = model.y_std.inverse_transform(np.asarray(model(y_hist, u_hist, u_future))[0])
    actual = np.asarray(runtime.free_run(y_raw[: k + 1][None], u_raw[:k][None], u_raw[k : k + model.horizon][None]))[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    assert actual.shape == (model.horizon, model.n_channels, 4)


def test_observable_cnn_singleton_frequency_and_short_history() -> None:
    """A singleton frequency axis and history shorter than the frequency kernel remain valid."""
    model = _model(n_values=1, residual=False, n_y=1)
    runtime = ObservableCNNModel.from_checkpoint(*model.to_checkpoint())
    rng = np.random.default_rng(44)
    y = rng.normal(size=(1, model.n_y, model.n_channels, 1))
    u_hist_raw = rng.normal(size=(1, model.n_u, model.n_controls))
    future = np.zeros((1, model.horizon, model.n_controls))
    future[:, 0] = rng.normal(size=(1, model.n_controls))
    y_standardized = jnp.expand_dims(jnp.asarray(model.y_std.transform(y[0]), dtype=jnp.float32), 0)
    u_standardized = jnp.expand_dims(jnp.asarray(model.u_std.transform(u_hist_raw[0]), dtype=jnp.float32), 0)
    u_future_standardized = jnp.expand_dims(jnp.asarray(model.u_std.transform(future[0]), dtype=jnp.float32), 0)
    expected = model.y_std.inverse_transform(
        np.asarray(model(y_standardized, u_standardized, u_future_standardized))[0]
    )
    result = np.asarray(runtime.free_run(y, u_hist_raw, future))[0]
    assert result.shape == (model.horizon, model.n_channels, 1)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)


def test_observable_cnn_generic_checkpoint_factory(tmp_path: Path) -> None:
    """Generic checkpoint loading selects the Observable CNN runtime."""
    model = _model()
    artifact = tmp_path / "observable_cnn"
    model.save(artifact)
    loaded = InferencePredictor.load(artifact)
    assert isinstance(loaded, ObservableCNNModel)
    assert loaded.frequency_kernel_size == 4
    assert loaded.geometry == model.geometry

    roundtrip_meta, roundtrip_arrays = loaded.to_checkpoint()
    training_loaded = ObservableCNNModel.from_checkpoint(roundtrip_meta, roundtrip_arrays)
    rng = np.random.default_rng(702)
    y_raw = rng.normal(size=(model.n_y, model.n_channels, model.n_values))
    u_raw = rng.normal(size=(model.n_u + model.horizon, model.n_controls))
    y_hist = jnp.expand_dims(jnp.asarray(model.y_std.transform(y_raw), dtype=jnp.float32), 0)
    u_hist = jnp.expand_dims(jnp.asarray(model.u_std.transform(u_raw[: model.n_u]), dtype=jnp.float32), 0)
    u_future = jnp.expand_dims(jnp.asarray(model.u_std.transform(u_raw[model.n_u :]), dtype=jnp.float32), 0)
    expected = model.y_std.inverse_transform(np.asarray(model(y_hist, u_hist, u_future))[0])
    actual = training_loaded.y_std.inverse_transform(np.asarray(training_loaded(y_hist, u_hist, u_future))[0])
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def _central_jacobian_step(
    runtime: ObservableCNNModel, state: jnp.ndarray, control: jnp.ndarray
) -> tuple[FloatArray, FloatArray]:
    """Estimate Observable CNN state and Control Current Jacobians by central differences."""
    eps = 1e-6
    state_np = np.asarray(state, dtype=np.float64)
    control_np = np.asarray(control, dtype=np.float64)
    state_jac = np.empty((runtime.n, runtime.n), dtype=np.float64)
    control_jac = np.empty((runtime.n, runtime.m), dtype=np.float64)
    for index in range(runtime.n):
        perturbation = np.zeros(runtime.n)
        perturbation[index] = eps
        plus = runtime.discrete_dynamics(jnp.asarray(state_np + perturbation), control, 0.0, runtime.dt)
        minus = runtime.discrete_dynamics(jnp.asarray(state_np - perturbation), control, 0.0, runtime.dt)
        state_jac[:, index] = (np.asarray(plus) - np.asarray(minus)) / (2.0 * eps)
    for index in range(runtime.m):
        perturbation = np.zeros(runtime.m)
        perturbation[index] = eps
        plus = runtime.discrete_dynamics(state, jnp.asarray(control_np + perturbation), 0.0, runtime.dt)
        minus = runtime.discrete_dynamics(state, jnp.asarray(control_np - perturbation), 0.0, runtime.dt)
        control_jac[:, index] = (np.asarray(plus) - np.asarray(minus)) / (2.0 * eps)
    return state_jac, control_jac


def test_observable_cnn_discrete_dynamics_jacobians_match_finite_differences() -> None:
    """Check smooth Observable CNN controller derivatives at a primed state and nonzero control sensitivity."""
    with jax.enable_x64():
        module = _model(activation="tanh")
        meta, arrays = module.to_checkpoint()
        runtime = ObservableCNNModel.from_checkpoint(
            meta, {key: np.asarray(value, dtype=np.float64) for key, value in arrays.items()}
        )
        rng = np.random.default_rng(2405)
        state = runtime.initial_state()
        for _ in range(runtime.n_y):
            state = runtime.absorb(
                state,
                rng.normal(size=(runtime.n_channels, runtime.n_values)),
                rng.normal(size=runtime.n_controls),
            )
        state_jax = jnp.asarray(state)
        control = jnp.asarray(rng.normal(size=runtime.m), dtype=jnp.float64)

        def step(x: jax.Array, u: jax.Array) -> jax.Array:
            return runtime.discrete_dynamics(x, u, 0.0, runtime.dt)

        state_ad, control_ad = jax.jacfwd(step, (0, 1))(state_jax, control)
        state_fd, control_fd = _central_jacobian_step(runtime, state_jax, control)
        np.testing.assert_allclose(np.asarray(state_ad), state_fd, rtol=2e-6, atol=2e-8)
        np.testing.assert_allclose(np.asarray(control_ad), control_fd, rtol=2e-6, atol=2e-8)
        newest = slice((runtime.n_history - 1) * runtime.n_outputs, runtime.n_history * runtime.n_outputs)
        assert np.linalg.norm(np.asarray(control_ad)[newest]) > 1e-6


def test_observable_cnn_train_save_load_and_controller_smoke(tmp_path: Path) -> None:
    """A short Observable CNN fit reloads through the factory and builds an MPC problem."""
    rng = np.random.default_rng(903)
    files: list[str] = []
    for index in range(2):
        u = rng.standard_normal((120, 2))
        y = np.cumsum(0.03 * u[:, :1], axis=0) + 0.1 * rng.standard_normal((120, 3))
        path = tmp_path / f"trajectory_{index}.npz"
        np.savez(path, **{"sensor_0.y_mea": y, "controller.u": u})  # ty: ignore[invalid-argument-type]
        files.append(str(path))
    geometry = StftGeometry(n_segment=8, n_hop=2)
    config = NNPredictorConfig(
        simulation=SimulationConfig(dt=0.01, downsample=1),
        model=ModelConfig(
            architecture="cnn",
            n_y=2,
            n_u=geometry.min_past_controls(),
            hidden_size=4,
            depth=1,
            kernel_size=3,
            frequency_kernel_size=4,
        ),
        training=TrainingConfig(
            epochs=2,
            batch_size=16,
            learning_rate=1e-2,
            weight_decay=0.0,
            train_split=0.5,
            patience=2,
            eval_horizon_s=0.08,
            device="cpu",
            losses=LossSpecs(
                curriculum_mse=CurriculumMSESpec(
                    weight=1.0,
                    span_s=0.08,
                    curr_start=0,
                    curr_end=1,
                )
            ),
        ),
        observable=geometry,
    )
    result = train(config, files)
    assert isinstance(result, TrainingResult)
    artifact = tmp_path / "artifact"
    result.save(artifact)
    runtime = InferencePredictor.load(artifact / "model")
    assert isinstance(runtime, ObservableCNNModel)
    problem = build_observable_problem(artifact / "model", horizon=2, u_max=1.0, w_u=1.0)
    assert problem.model is not None


@pytest.mark.parametrize(
    ("tapering_ratio", "head_depth", "head_width"),
    [
        (0.5, 1, "trunk"),
        (0.5, 0, "trunk"),
        (0.5, 1, "geometric_mean"),
    ],
)
def test_observable_cnn_tapering_and_head_depth_parity(
    tapering_ratio: float,
    head_depth: int,
    head_width: Literal["trunk", "geometric_mean"],
) -> None:
    """Observable CNN supports tapering ratio and configurable head depth with JAX checkpoint parity."""
    rng = np.random.default_rng(808)
    geometry = StftGeometry(n_segment=6, n_hop=2)
    n_values = 4
    hidden_size = 16
    depth = 3
    module = ObservableCNNModel(
        n_y=3,
        n_u=2,
        horizon=3,
        n_channels=2,
        n_controls=2,
        n_outputs=2 * n_values,
        hidden_size=hidden_size,
        depth=depth,
        kernel_size=3,
        frequency_kernel_size=3,
        activation="tanh",
        residual=True,
        dt=0.04,
        geometry=geometry,
        y_std=Standardizer(center=rng.normal(size=(2, n_values)), scale=rng.uniform(0.5, 2.0, (2, n_values))),
        u_std=Standardizer(center=np.full(2, 0.4), scale=np.full(2, 1.7)),
        tapering_ratio=tapering_ratio,
        head_depth=head_depth,
        head_width=head_width,
    )
    module = eqx.tree_at(
        lambda m: (m.conv_weights, m.conv_biases, m.head_weights, m.head_biases),
        module,
        replace_fn=lambda tree: jax.tree.map(lambda x: jax.random.normal(jax.random.PRNGKey(91), x.shape) * 0.2, tree),
    )

    conv_layers = module.conv_weights
    expected_channels = [
        16,
        math.floor(16 * tapering_ratio),
        math.floor(math.floor(16 * tapering_ratio) * tapering_ratio),
    ]
    assert [layer.shape[0] for layer in conv_layers] == expected_channels

    head_linears = module.head_weights
    assert len(head_linears) == (head_depth + 1 if head_depth > 0 else 1)

    meta, arrays = module.to_checkpoint()
    assert meta["tapering_ratio"] == tapering_ratio
    assert meta["head_depth"] == head_depth
    assert meta["head_width"] == head_width

    runtime = ObservableCNNModel.from_checkpoint(meta, arrays)
    assert runtime.tapering_ratio == tapering_ratio
    assert runtime.head_depth == head_depth
    assert runtime.head_width == head_width

    t0 = 7
    k = t0 - 1
    y_raw = rng.normal(size=(t0 + module.horizon, module.n_channels, n_values))
    u_raw = rng.normal(size=(t0 + module.horizon, module.n_controls))
    y_hist = jnp.expand_dims(
        jnp.asarray(module.y_std.transform(y_raw[k - module.n_y + 1 : k + 1]), dtype=jnp.float32), 0
    )
    u_hist = jnp.expand_dims(jnp.asarray(module.u_std.transform(u_raw[k - module.n_u : k]), dtype=jnp.float32), 0)
    u_future = jnp.expand_dims(jnp.asarray(module.u_std.transform(u_raw[k : k + module.horizon]), dtype=jnp.float32), 0)
    expected = module.y_std.inverse_transform(np.asarray(module(y_hist, u_hist, u_future))[0])
    actual = np.asarray(runtime.free_run(y_raw[: k + 1][None], u_raw[:k][None], u_raw[k : k + module.horizon][None]))[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
