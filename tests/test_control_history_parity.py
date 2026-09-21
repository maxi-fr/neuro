from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from neuro.config import StftGeometry
from neuro.predictor.evaluation import evaluate_observable_free_run, rollout_batches
from neuro.predictor.inference import (
    InferencePredictor,
    ObservableCNNModel,
    ObservableMLPModel,
    WaveformCNNModel,
    WaveformMLPModel,
)
from neuro.predictor.module import AutoregressiveCNN, AutoregressiveMLP
from neuro.predictor.replay import replay_predictions
from neuro.transforms import Standardizer


def _create_deterministic_linear_mlp(
    *,
    n_y: int = 2,
    n_u: int = 2,
    n_channels: int = 2,
    n_controls: int = 2,
    horizon: int = 3,
) -> tuple[AutoregressiveMLP, WaveformMLPModel]:
    """Create a deterministic linear MLP where the immediate candidate control has known weights."""
    module = AutoregressiveMLP(
        n_y=n_y,
        n_u=n_u,
        horizon=horizon,
        n_channels=n_channels,
        n_controls=n_controls,
        n_outputs=n_channels,
        hidden_size=8,
        depth=0,
        activation="relu",
        residual=False,
        dt=0.01,
        y_std=Standardizer(center=np.zeros(n_channels), scale=np.ones(n_channels)),
        u_std=Standardizer(center=np.zeros(n_controls), scale=np.ones(n_controls)),
    )
    layer = module.layers[0]
    assert isinstance(layer, torch.nn.Linear)
    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.zero_()
        y_t_start = (n_y - 1) * n_channels
        layer.weight[0, y_t_start] = 1.0
        layer.weight[1, y_t_start + 1] = 1.0
        u_t_start = n_y * n_channels + (n_u - 1) * n_controls
        layer.weight[0, u_t_start] = 5.0
        layer.weight[1, u_t_start + 1] = 10.0

    meta, arrays = module.to_checkpoint()
    jax_model = WaveformMLPModel.from_checkpoint(meta, arrays)
    return module, jax_model


def test_immediate_control_response_parity() -> None:
    """A deterministic test with changing, channel-distinct inputs verifies immediate control response."""
    n_y, n_u, n_c, n_m, horizon = 2, 2, 2, 2, 3
    module, jax_model = _create_deterministic_linear_mlp(
        n_y=n_y, n_u=n_u, n_channels=n_c, n_controls=n_m, horizon=horizon
    )

    y_hist = np.array([[0.5, 0.5], [1.0, 2.0]], dtype=np.float64)
    u_hist = np.array([[1.0, 1.0], [3.0, 4.0]], dtype=np.float64)
    u_future = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=np.float64)

    expected_first_step = np.array([51.0, 202.0], dtype=np.float64)

    # 1. Training forward pass
    y_hist_t = torch.as_tensor(y_hist, dtype=torch.float32).unsqueeze(0)
    u_hist_t = torch.as_tensor(u_hist, dtype=torch.float32).unsqueeze(0)
    u_fut_t = torch.as_tensor(u_future, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        torch_preds = module(y_hist_t, u_hist_t, u_fut_t).numpy()[0]
    np.testing.assert_allclose(torch_preds[0], expected_first_step, rtol=1e-5, atol=1e-5)

    # 2. MPC discrete_dynamics
    state = jax_model.initial_state()
    for i in range(n_y):
        state = jax_model.absorb(state, y_hist[i], u_hist[i])
    assert jax_model.is_ready(state)

    mpc_preds = []
    curr_state = state
    for t in range(horizon):
        curr_state = jax_model.discrete_dynamics(jnp.asarray(curr_state), jnp.asarray(u_future[t]), 0.0, jax_model.dt)
        mpc_preds.append(np.asarray(jax_model.output(curr_state)))
    mpc_preds_arr = np.stack(mpc_preds, axis=0)

    np.testing.assert_allclose(mpc_preds_arr[0], expected_first_step, rtol=1e-5, atol=1e-5)

    # 3. Offline free_run
    free_run_preds = np.asarray(jax_model.free_run(y_hist[None], u_hist[None], u_future[None]))[0]

    np.testing.assert_allclose(free_run_preds[0], expected_first_step, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(free_run_preds, torch_preds, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(free_run_preds, mpc_preds_arr, rtol=1e-5, atol=1e-5)


def _build_test_model(
    model_kind: str,
    n_y: int,
    n_u: int,
    n_channels: int,
    n_controls: int,
    horizon: int,
    dt: float,
    rng: np.random.Generator,
) -> tuple[AutoregressiveMLP | AutoregressiveCNN, InferencePredictor, np.ndarray, tuple[int, ...]]:
    """Construct matched torch and JAX models and an initial history for parity testing."""
    if model_kind == "waveform_mlp":
        module = AutoregressiveMLP(
            n_y=n_y,
            n_u=n_u,
            horizon=horizon,
            n_channels=n_channels,
            n_controls=n_controls,
            n_outputs=n_channels,
            hidden_size=16,
            depth=1,
            activation="tanh",
            residual=True,
            dt=dt,
            y_std=Standardizer(center=rng.normal(size=n_channels), scale=np.exp(rng.normal(size=n_channels))),
            u_std=Standardizer(center=rng.normal(size=n_controls), scale=np.exp(rng.normal(size=n_controls))),
        )
        meta, arrays = module.to_checkpoint()
        jax_model: InferencePredictor = WaveformMLPModel.from_checkpoint(meta, arrays)
        y_hist = rng.standard_normal((n_y, n_channels))
        return module, jax_model, y_hist, (n_channels,)

    if model_kind == "observable_mlp":
        geometry = StftGeometry(n_segment=16, n_hop=4, band_hz=[4.0, 20.0], n_bin_pool=2, kernel_width=3)
        n_values = geometry.n_values(1.0 / dt)
        n_outputs = n_channels * n_values
        module = AutoregressiveMLP(
            n_y=n_y,
            n_u=n_u,
            horizon=horizon,
            n_channels=n_channels,
            n_controls=n_controls,
            n_outputs=n_outputs,
            hidden_size=16,
            depth=1,
            activation="relu",
            residual=True,
            dt=dt,
            y_std=Standardizer(
                center=rng.normal(size=(n_channels, n_values)),
                scale=np.exp(rng.normal(size=(n_channels, n_values))),
            ),
            u_std=Standardizer(center=rng.normal(size=n_controls), scale=np.exp(rng.normal(size=n_controls))),
            geometry=geometry,
        )
        meta, arrays = module.to_checkpoint()
        jax_model = ObservableMLPModel.from_checkpoint(meta, arrays)
        y_hist = rng.standard_normal((n_y, n_channels, n_values))
        return module, jax_model, y_hist, (n_channels, n_values)

    if model_kind == "waveform_cnn":
        module = AutoregressiveCNN(
            n_y=n_y,
            n_u=n_u,
            horizon=horizon,
            n_channels=n_channels,
            n_controls=n_controls,
            hidden_size=8,
            depth=1,
            kernel_size=2,
            activation="relu",
            residual=True,
            dt=dt,
            y_std=Standardizer(center=rng.normal(size=n_channels), scale=np.exp(rng.normal(size=n_channels))),
            u_std=Standardizer(center=rng.normal(size=n_controls), scale=np.exp(rng.normal(size=n_controls))),
        )
        meta, arrays = module.to_checkpoint()
        jax_model = WaveformCNNModel.from_checkpoint(meta, arrays)
        y_hist = rng.standard_normal((n_y, n_channels))
        return module, jax_model, y_hist, (n_channels,)

    # observable_cnn
    geometry = StftGeometry(n_segment=16, n_hop=4, band_hz=[4.0, 20.0], n_bin_pool=2, kernel_width=3)
    n_values = geometry.n_values(1.0 / dt)
    module = AutoregressiveCNN(
        n_y=n_y,
        n_u=n_u,
        horizon=horizon,
        n_channels=n_channels,
        n_controls=n_controls,
        n_outputs=n_channels * n_values,
        hidden_size=8,
        depth=1,
        kernel_size=2,
        frequency_kernel_size=2,
        activation="relu",
        residual=True,
        dt=dt,
        y_std=Standardizer(
            center=rng.normal(size=(n_channels, n_values)),
            scale=np.exp(rng.normal(size=(n_channels, n_values))),
        ),
        u_std=Standardizer(center=rng.normal(size=n_controls), scale=np.exp(rng.normal(size=n_controls))),
        geometry=geometry,
    )
    meta, arrays = module.to_checkpoint()
    jax_model = ObservableCNNModel.from_checkpoint(meta, arrays)
    y_hist = rng.standard_normal((n_y, n_channels, n_values))
    return module, jax_model, y_hist, (n_channels, n_values)


@pytest.mark.parametrize("model_kind", ["waveform_mlp", "observable_mlp", "waveform_cnn", "observable_cnn"])
@pytest.mark.parametrize("n_u", [1, 2])
def test_multistep_parity_across_models_and_control_history_one(model_kind: str, n_u: int) -> None:
    """Training forward, free_run, and MPC agree over full multistep rollout with n_u=1 and n_u=2."""
    rng = np.random.default_rng(101)
    torch.manual_seed(101)

    n_y, n_channels, n_controls, horizon, dt = 2, 2, 2, 4, 0.02
    module, jax_model, y_hist, output_shape = _build_test_model(
        model_kind, n_y, n_u, n_channels, n_controls, horizon, dt, rng
    )

    u_hist = rng.standard_normal((n_u, n_controls))
    u_future = rng.standard_normal((horizon, n_controls))

    # 1. Training forward pass
    y_hist_std = torch.as_tensor(module.y_std.transform(y_hist), dtype=torch.float32).unsqueeze(0)
    u_hist_std = torch.as_tensor(module.u_std.transform(u_hist), dtype=torch.float32).unsqueeze(0)
    u_fut_std = torch.as_tensor(module.u_std.transform(u_future), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        torch_preds_std = module(y_hist_std, u_hist_std, u_fut_std).numpy()[0]
    torch_preds = module.y_std.inverse_transform(torch_preds_std)

    # 2. Offline free_run
    free_run_preds = np.asarray(jax_model.free_run(y_hist[None], u_hist[None], u_future[None]))[0]

    # 3. MPC discrete_dynamics rollout
    state = jax_model.initial_state()
    for i in range(n_y):
        state = jax_model.absorb(state, y_hist[i], u_hist[min(i, n_u - 1)])
    assert jax_model.is_ready(state)

    mpc_preds = []
    curr_state = state
    for t in range(horizon):
        curr_state = jax_model.discrete_dynamics(jnp.asarray(curr_state), jnp.asarray(u_future[t]), 0.0, jax_model.dt)
        mpc_preds.append(np.asarray(jax_model.output(curr_state)).reshape(output_shape))
    mpc_preds_arr = np.stack(mpc_preds, axis=0)

    # Assert agreement across all steps to numerical tolerance
    np.testing.assert_allclose(free_run_preds, torch_preds, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(mpc_preds_arr, free_run_preds, rtol=1e-4, atol=1e-5)


def test_causality_and_no_future_control_leakage() -> None:
    """Changing future control u_{future}[t] does not alter predictions at earlier steps."""
    rng = np.random.default_rng(202)
    n_y, n_u, n_channels, n_controls, horizon = 2, 2, 2, 2, 4
    _, jax_model = _create_deterministic_linear_mlp(
        n_y=n_y, n_u=n_u, n_channels=n_channels, n_controls=n_controls, horizon=horizon
    )

    y_hist = rng.standard_normal((n_y, n_channels))
    u_hist = rng.standard_normal((n_u, n_controls))
    u_future_a = rng.standard_normal((horizon, n_controls))
    u_future_b = u_future_a.copy()
    u_future_b[1:] += rng.standard_normal(u_future_b[1:].shape) * 5.0  # Only step 0 is unchanged

    preds_a = np.asarray(jax_model.free_run(y_hist[None], u_hist[None], u_future_a[None]))[0]
    preds_b = np.asarray(jax_model.free_run(y_hist[None], u_hist[None], u_future_b[None]))[0]

    # Step 0 must be exactly identical
    np.testing.assert_allclose(preds_a[0], preds_b[0], rtol=1e-6, atol=1e-6)
    # Subsequent steps must differ
    assert not np.allclose(preds_a[1], preds_b[1])


def test_state_absorption_and_extended_history_buffer() -> None:
    """Extending measurement buffer for spectral Frames does not change trained input window or predictions."""
    rng = np.random.default_rng(303)
    n_y, n_u, n_channels, n_controls = 2, 2, 2, 2
    _, base_model = _create_deterministic_linear_mlp(
        n_y=n_y, n_u=n_u, n_channels=n_channels, n_controls=n_controls, horizon=3
    )

    # Extend history buffer to 8 steps
    extended_model = base_model.with_history(8)
    assert extended_model.n_history == 8
    assert extended_model.n_y == n_y
    assert extended_model.n_u == n_u

    y_extended = rng.standard_normal((8, n_channels))
    u_extended = rng.standard_normal((8, n_controls))

    # Prime both models: base model primes on last n_y steps
    base_state = base_model.initial_state()
    for i in range(8 - n_y, 8):
        base_state = base_model.absorb(base_state, y_extended[i], u_extended[i])

    ext_state = extended_model.initial_state()
    for i in range(8):
        ext_state = extended_model.absorb(ext_state, y_extended[i], u_extended[i])

    assert base_model.is_ready(base_state)
    assert extended_model.is_ready(ext_state)

    # Next candidate control
    u_candidate = np.array([7.0, -3.0])
    base_next = base_model.discrete_dynamics(jnp.asarray(base_state), jnp.asarray(u_candidate), 0.0, base_model.dt)
    ext_next = extended_model.discrete_dynamics(
        jnp.asarray(ext_state), jnp.asarray(u_candidate), 0.0, extended_model.dt
    )

    np.testing.assert_allclose(
        np.asarray(base_model.output(base_next)),
        np.asarray(extended_model.output(ext_next)),
        rtol=1e-6,
        atol=1e-6,
    )


def test_evaluation_rollout_batches_timing_matches_training_and_mpc() -> None:
    """rollout_batches yields targets y_{t0+1:t0+1+steps} matching forward pass and discrete dynamics."""
    rng = np.random.default_rng(404)
    n_y, n_u, n_c, n_m, steps = 2, 2, 2, 2, 3
    module, jax_model = _create_deterministic_linear_mlp(
        n_y=n_y, n_u=n_u, n_channels=n_c, n_controls=n_m, horizon=steps
    )

    t_len = 30
    u_traj = rng.standard_normal((t_len, n_m))
    y_traj = rng.standard_normal((t_len, n_c))
    trajs = [(u_traj, y_traj)]

    for y_pred, y_true in rollout_batches(jax_model, trajs, steps, stride=5):
        k = jax_model.priming_steps
        t0s = list(range(k, t_len - steps, 5))
        assert len(y_pred) == len(t0s)
        for i, t0 in enumerate(t0s):
            # Physical verification:
            # y_hist ends at y[t0], u_hist ends at u[t0-1]
            y_hist_t = torch.as_tensor(y_traj[t0 - n_y + 1 : t0 + 1], dtype=torch.float32).unsqueeze(0)
            u_hist_t = torch.as_tensor(u_traj[t0 - n_u : t0], dtype=torch.float32).unsqueeze(0)
            u_fut_t = torch.as_tensor(u_traj[t0 : t0 + steps], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                expected_pred = module(y_hist_t, u_hist_t, u_fut_t).numpy()[0]
            np.testing.assert_allclose(y_pred[i], expected_pred, rtol=1e-5, atol=1e-5)
            # Target verification:
            np.testing.assert_allclose(y_true[i], y_traj[t0 + 1 : t0 + 1 + steps], rtol=1e-5, atol=1e-5)


def test_replay_predictions_matches_free_run_and_mpc() -> None:
    """replay_predictions agrees with free_run and MPC discrete dynamics under actual recorded controls."""
    rng = np.random.default_rng(505)
    n_y, n_u, n_c, n_m, horizon = 2, 2, 2, 2, 3
    _, jax_model = _create_deterministic_linear_mlp(n_y=n_y, n_u=n_u, n_channels=n_c, n_controls=n_m, horizon=horizon)

    n_samples = 15
    measurements = rng.standard_normal((n_samples, n_c))
    controls = rng.standard_normal((n_samples, n_m))
    times = np.arange(n_samples, dtype=np.float64) * jax_model.dt

    predictions, observed = replay_predictions(jax_model, measurements, controls, times, horizon=horizon)

    # For a primed decision index t0:
    t0 = 5
    # replay_predictions[t0, 0] is the observed output at t0
    np.testing.assert_allclose(predictions[t0, 0], observed[t0])
    np.testing.assert_allclose(observed[t0], measurements[t0])

    # replay_predictions[t0, 1:] are the forecast outputs under candidate controls starting at controls[t0]
    # This must equal free_run called with the exact same physical history and future controls:
    y_hist = measurements[t0 - n_y + 1 : t0 + 1]
    u_hist = controls[t0 - n_u : t0]
    u_future = controls[t0 : t0 + horizon]
    free_run_pred = np.asarray(jax_model.free_run(y_hist[None], u_hist[None], u_future[None]))[0]

    np.testing.assert_allclose(predictions[t0, 1:], free_run_pred, rtol=1e-5, atol=1e-5)


def test_observable_evaluation_scores_an_exact_control_response() -> None:
    """Observable evaluation scores y[t+1] = u[t] with zero error for changing inputs."""
    geom = StftGeometry(n_segment=4, n_hop=1)
    n_values = geom.n_values(50.0)
    module = AutoregressiveMLP(
        n_y=2,
        n_u=1,
        horizon=3,
        n_channels=1,
        n_controls=1,
        n_outputs=n_values,
        hidden_size=8,
        depth=0,
        activation="relu",
        residual=False,
        dt=0.02,
        y_std=Standardizer(center=np.zeros((1, n_values)), scale=np.ones((1, n_values))),
        u_std=Standardizer(center=np.zeros(1), scale=np.ones(1)),
        geometry=geom,
    )
    layer = module.layers[0]
    assert isinstance(layer, torch.nn.Linear)
    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.zero_()
        layer.weight[:, -1] = 1.0
    meta, arrays = module.to_checkpoint()
    model = ObservableMLPModel.from_checkpoint(meta, arrays)
    u = np.arange(10, dtype=np.float64)[:, None]
    y = np.zeros((10, 1, n_values))
    y[1:] = u[:-1, :, None]
    result = evaluate_observable_free_run(model, [(u, y)], 3)
    np.testing.assert_allclose(result.per_step, 0.0, atol=1e-12)
