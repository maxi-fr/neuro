from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from neuro.config import CurriculumMSESpec, LossSpecs, ModelConfig, NNPredictorConfig, SimulationConfig, TrainingConfig
from neuro.control.mpc import build_waveform_problem
from neuro.jansen_rit import JansenRitParams
from neuro.predictor.inference import InferencePredictor, WaveformCNNModel, WaveformMLPModel
from neuro.predictor.jansen_rit import JansenRitModel
from neuro.predictor.module import AutoregressiveCNN, AutoregressiveMLP
from neuro.predictor.train import TrainingResult, train
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from pathlib import Path

    from neuro.types import Activation, FloatArray


def _cnn(*, depth: int = 2, activation: Activation = "tanh", residual: bool = True) -> AutoregressiveCNN:
    """Build a small causal CNN with nontrivial output and control scaling."""
    rng = np.random.default_rng(91)
    torch.manual_seed(91)
    model = AutoregressiveCNN(
        n_y=4,
        n_u=3,
        horizon=4,
        n_channels=3,
        n_controls=2,
        hidden_size=5,
        depth=depth,
        kernel_size=3,
        activation=activation,
        residual=residual,
        dt=0.01,
        y_std=Standardizer(center=rng.uniform(-1.0, 1.0, 3), scale=rng.uniform(0.5, 2.0, 3)),
        u_std=Standardizer(center=rng.uniform(-1.0, 1.0, 2), scale=rng.uniform(0.5, 2.0, 2)),
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=0.2)
    return model


@pytest.mark.parametrize(
    ("depth", "activation", "residual"),
    [(1, "relu", False), (2, "tanh", True), (3, "softplus", False)],
)
def test_cnn_torch_jax_parity_for_causal_multistep_rollout(
    depth: int,
    activation: Activation,
    residual: bool,  # noqa: FBT001 -- pytest parameterizes this boolean
) -> None:
    """The JAX CNN runtime matches the torch rollout for nonsymmetric causal kernels."""
    module = _cnn(depth=depth, activation=activation, residual=residual)
    runtime = WaveformCNNModel.from_checkpoint(*module.to_checkpoint())
    rng = np.random.default_rng(123)
    t0 = 8
    y_raw = rng.standard_normal((t0 + module.horizon, module.n_channels))
    u_raw = rng.standard_normal((t0 + module.horizon, module.n_controls))
    k = t0 - 1
    y_hist = torch.as_tensor(module.y_std.transform(y_raw[k - module.n_y + 1 : k + 1]), dtype=torch.float32).unsqueeze(
        0
    )
    u_hist = torch.as_tensor(module.u_std.transform(u_raw[k - module.n_u : k]), dtype=torch.float32).unsqueeze(0)
    u_future = torch.as_tensor(module.u_std.transform(u_raw[k : k + module.horizon]), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        expected = module.y_std.inverse_transform(module(y_hist, u_hist, u_future).numpy()[0])
    actual = np.asarray(runtime.free_run(y_raw[: k + 1][None], u_raw[:k][None], u_raw[k : k + module.horizon][None]))[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_cnn_checkpoint_factory_and_typed_loaders(tmp_path: Path) -> None:
    """Generic loading selects CNN while architecture-specific loaders reject mismatches."""
    module = _cnn()
    stem = tmp_path / "cnn"
    module.save(stem)
    meta, arrays = module.to_checkpoint()
    torch_reloaded = AutoregressiveCNN.from_checkpoint(meta, arrays)
    generic = InferencePredictor.load(stem)
    direct = WaveformCNNModel.load(stem)
    assert isinstance(generic, WaveformCNNModel)
    assert isinstance(direct, WaveformCNNModel)
    y_hist = torch.randn(1, module.n_y, module.n_channels)
    u_hist = torch.randn(1, module.n_u, module.n_controls)
    u_future = torch.randn(1, module.horizon, module.n_controls)
    with torch.no_grad():
        torch_expected = module(y_hist, u_hist, u_future)
        torch_actual = torch_reloaded(y_hist, u_hist, u_future)
    np.testing.assert_allclose(torch_actual.numpy(), torch_expected.numpy(), rtol=1e-6, atol=1e-6)
    runtime_stem = tmp_path / "cnn_runtime_roundtrip"
    direct.save(runtime_stem)
    torch_from_runtime = AutoregressiveCNN.load(runtime_stem)
    with torch.no_grad():
        runtime_torch = torch_from_runtime(y_hist, u_hist, u_future)
    np.testing.assert_allclose(runtime_torch.numpy(), torch_expected.numpy(), rtol=1e-6, atol=1e-6)
    rng = np.random.default_rng(405)
    t0 = 8
    y_raw = rng.standard_normal((t0 + module.horizon, module.n_channels))
    u_raw = rng.standard_normal((t0 + module.horizon, module.n_controls))
    k = t0 - 1
    y_hist = torch.as_tensor(module.y_std.transform(y_raw[k - module.n_y + 1 : k + 1]), dtype=torch.float32).unsqueeze(
        0
    )
    u_hist = torch.as_tensor(module.u_std.transform(u_raw[k - module.n_u : k]), dtype=torch.float32).unsqueeze(0)
    u_future = torch.as_tensor(module.u_std.transform(u_raw[k : k + module.horizon]), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        torch_rollout = module.y_std.inverse_transform(module(y_hist, u_hist, u_future).numpy()[0])
    jax_rollout = np.asarray(
        direct.free_run(y_raw[: k + 1][None], u_raw[:k][None], u_raw[k : k + module.horizon][None])
    )[0]
    np.testing.assert_allclose(jax_rollout, torch_rollout, rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match="model_type"):
        WaveformMLPModel.load(stem)


def test_generic_waveform_mlp_load_regression(tmp_path: Path) -> None:
    """The generic loader reconstructs a waveform MLP checkpoint instead of falling into its ABC."""
    module = AutoregressiveMLP(
        n_y=2,
        n_u=2,
        horizon=2,
        n_channels=2,
        n_controls=1,
        n_outputs=2,
        hidden_size=3,
        depth=1,
    )
    stem = tmp_path / "mlp"
    module.save(stem)
    loaded = InferencePredictor.load(stem)
    assert isinstance(loaded, WaveformMLPModel)


def test_cnn_discrete_step_uses_current_action_alignment() -> None:
    """The controller step with a new action equals the training recursion's next prediction."""
    module = _cnn()
    runtime = WaveformCNNModel.from_checkpoint(*module.to_checkpoint())
    rng = np.random.default_rng(77)
    t0 = 8
    y_raw = rng.standard_normal((t0 + 2, module.n_channels))
    u_raw = rng.standard_normal((t0 + 3, module.n_controls))
    state = runtime.initial_state()
    for y_value, u_value in zip(y_raw[: t0 + 1], u_raw[: t0 + 1], strict=True):
        state = runtime.absorb(state, y_value, u_value)
    stepped = runtime.discrete_dynamics(jnp.asarray(state), jnp.asarray(u_raw[t0 + 1]), 0.0, runtime.dt)
    expected = np.asarray(runtime.free_run(y_raw[: t0 + 1][None], u_raw[: t0 + 1][None], u_raw[t0 + 1 :][None]))[0, 0]
    np.testing.assert_allclose(np.asarray(runtime.output(stepped)), expected, rtol=1e-5, atol=1e-6)


def _central_jacobian_step(
    runtime: WaveformCNNModel, state: jnp.ndarray, control: jnp.ndarray
) -> tuple[FloatArray, FloatArray]:
    """Estimate discrete-dynamics state and Control Current Jacobians independently by central differences."""
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


def test_cnn_discrete_dynamics_state_and_control_jacobians_match_finite_differences() -> None:
    """Check smooth waveform CNN controller derivatives at a primed state and nonzero control sensitivity."""
    with jax.enable_x64():
        module = _cnn(activation="tanh")
        meta, arrays = module.to_checkpoint()
        runtime = WaveformCNNModel.from_checkpoint(
            meta, {key: np.asarray(value, dtype=np.float64) for key, value in arrays.items()}
        )
        rng = np.random.default_rng(2404)
        state = runtime.initial_state()
        for _ in range(runtime.n_y):
            state = runtime.absorb(state, rng.normal(size=runtime.n_channels), rng.normal(size=runtime.n_controls))
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


def test_cnn_history_extension_preserves_runtime() -> None:
    """A CNN can use a longer controller history than the trained history."""
    runtime = WaveformCNNModel.from_checkpoint(*_cnn().to_checkpoint())
    extended = runtime.with_history(7)
    assert extended.n_history == 7
    assert extended.priming_steps == runtime.priming_steps
    assert extended.initial_state().shape == (7 * runtime.n_outputs + runtime.n_u * runtime.n_controls,)


def test_cnn_depth_zero_is_rejected_at_config_build() -> None:
    """CNN configuration rejects an architecture with no convolutional layers."""
    with pytest.raises(ValueError, match="depth"):
        ModelConfig(architecture="cnn", depth=0)


@pytest.mark.parametrize("fit", ["ridge", "dmd"])
def test_cnn_closed_form_fits_reject_before_loading_data(fit: Literal["ridge", "dmd"]) -> None:
    """CNN ridge and DMD requests fail before touching a nonexistent trajectory path."""
    config = NNPredictorConfig(
        simulation=SimulationConfig(dt=0.01, downsample=1),
        model=ModelConfig(architecture="cnn", n_y=2, n_u=2, hidden_size=4, depth=1),
        training=TrainingConfig(
            fit=fit,
            eval_horizon_s=0.02,
            losses=LossSpecs(curriculum_mse=CurriculumMSESpec(weight=1.0, span_s=0.02, curr_start=0, curr_end=1)),
        ),
    )
    with pytest.raises(ValueError, match="gradient_descent"):
        train(config, ["does-not-exist.npz"])


def test_jansen_rit_typed_load_does_not_use_neural_factory(tmp_path: Path) -> None:
    """Concrete non-neural predictors continue to load through their own checkpoint format."""
    model = JansenRitModel.from_plant_components(params=JansenRitParams(), dt=1e-4, n_nodes=1)
    stem = tmp_path / "jansen"
    model.save(stem)
    loaded = JansenRitModel.load(stem)
    assert loaded.n == model.n
    np.testing.assert_allclose(np.asarray(loaded.initial_state()), np.asarray(model.initial_state()))


def test_cnn_train_save_load_and_controller_smoke(tmp_path: Path) -> None:
    """A short CNN fit can save, reload through the generic factory, and build an MPC problem."""
    rng = np.random.default_rng(404)
    files: list[str] = []
    for index in range(2):
        u = rng.standard_normal((80, 2))
        y = np.cumsum(0.03 * u[:, :1], axis=0) + 0.1 * rng.standard_normal((80, 3))
        path = tmp_path / f"trajectory_{index}.npz"
        np.savez(path, **{"sensor_0.y_mea": y, "controller.u": u})  # ty: ignore[invalid-argument-type]
        files.append(str(path))
    config = NNPredictorConfig(
        simulation=SimulationConfig(dt=0.01, downsample=1),
        model=ModelConfig(architecture="cnn", n_y=2, n_u=2, hidden_size=4, depth=1, kernel_size=3),
        training=TrainingConfig(
            epochs=1,
            batch_size=16,
            learning_rate=1e-2,
            weight_decay=0.0,
            train_split=0.5,
            patience=2,
            eval_horizon_s=0.02,
            losses=LossSpecs(curriculum_mse=CurriculumMSESpec(weight=1.0, span_s=0.02, curr_start=0, curr_end=1)),
        ),
    )
    result = train(config, files)
    assert isinstance(result, TrainingResult)
    artifact = tmp_path / "artifact"
    result.save(artifact)
    runtime = InferencePredictor.load(artifact / "model")
    assert isinstance(runtime, WaveformCNNModel)
    problem = build_waveform_problem(artifact / "model", horizon=2, u_max=1.0, w_y=0.0, w_u=1.0)
    assert problem.model is not None
