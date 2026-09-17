from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import torch

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
from neuro.predictor.inference import InferencePredictor, ObservableCNNModel
from neuro.predictor.module import AutoregressiveCNN
from neuro.predictor.train import TrainingResult, train
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from pathlib import Path

    from neuro.types import Activation, FloatArray


def _model(
    *, n_values: int = 4, residual: bool = True, n_y: int = 3, activation: Activation = "relu"
) -> AutoregressiveCNN:
    """Build a small structured CNN with nontrivial channel-frequency standardizers."""
    rng = np.random.default_rng(27 + n_values)
    torch.manual_seed(27 + n_values)
    geometry = StftGeometry(n_segment=6, n_hop=2) if n_values == 4 else StftGeometry(n_segment=2, n_hop=1, n_bin_pool=2)
    model = AutoregressiveCNN(
        n_y=n_y,
        n_u=2,
        horizon=3,
        n_channels=2,
        n_controls=1,
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
        u_std=Standardizer(center=np.array([0.4]), scale=np.array([1.7])),
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=0.2)
    return model


def test_observable_cnn_torch_jax_parity_and_frequency_order() -> None:
    """Causal time-frequency convolution preserves frequency order across the runtime boundary."""
    model = _model()
    runtime = ObservableCNNModel.from_checkpoint(*model.to_checkpoint())
    rng = np.random.default_rng(91)
    t0 = 7
    y_raw = rng.normal(size=(t0 + model.horizon, model.n_channels, model.n_outputs // model.n_channels))
    u_raw = rng.normal(size=(t0 + model.horizon, model.n_controls))
    y_hist = torch.as_tensor(model.y_std.transform(y_raw[t0 - model.n_y : t0]), dtype=torch.float32).unsqueeze(0)
    u_hist = torch.as_tensor(model.u_std.transform(u_raw[t0 - 1 - model.n_u : t0 - 1]), dtype=torch.float32).unsqueeze(
        0
    )
    u_future = torch.as_tensor(
        model.u_std.transform(u_raw[t0 - 1 : t0 - 1 + model.horizon]), dtype=torch.float32
    ).unsqueeze(0)
    with torch.no_grad():
        expected = model.y_std.inverse_transform(model(y_hist, u_hist, u_future).numpy()[0])
    actual = np.asarray(runtime.free_run(y_raw[:t0][None], u_raw[:t0][None], u_raw[t0:][None]))[0]
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
    y_standardized = torch.as_tensor(model.y_std.transform(y[0]), dtype=torch.float32).unsqueeze(0)
    u_standardized = torch.as_tensor(model.u_std.transform(u_hist_raw[0]), dtype=torch.float32).unsqueeze(0)
    u_future_standardized = torch.as_tensor(model.u_std.transform(future[0]), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        expected = model.y_std.inverse_transform(
            model(y_standardized, u_standardized, u_future_standardized).numpy()[0]
        )
    runtime_u_hist = np.concatenate([u_hist_raw[:, 1:], future[:, :1]], axis=1)
    runtime_future = np.concatenate([future[:, 1:], future[:, -1:]], axis=1)
    result = np.asarray(runtime.free_run(y, runtime_u_hist, runtime_future))[0]
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
    torch_loaded = AutoregressiveCNN.from_checkpoint(roundtrip_meta, roundtrip_arrays)
    rng = np.random.default_rng(702)
    y_raw = rng.normal(size=(model.n_y, model.n_channels, model.n_values))
    u_raw = rng.normal(size=(model.n_u + model.horizon, model.n_controls))
    y_hist = torch.as_tensor(model.y_std.transform(y_raw), dtype=torch.float32).unsqueeze(0)
    u_hist = torch.as_tensor(model.u_std.transform(u_raw[: model.n_u]), dtype=torch.float32).unsqueeze(0)
    u_future = torch.as_tensor(model.u_std.transform(u_raw[model.n_u :]), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        expected = model.y_std.inverse_transform(model(y_hist, u_hist, u_future).numpy()[0])
        actual = torch_loaded.y_std.inverse_transform(torch_loaded(y_hist, u_hist, u_future).numpy()[0])
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
            epochs=1,
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
