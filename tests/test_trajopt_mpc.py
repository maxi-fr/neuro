"""Ticket 01: the trajopt waveform MPC model adapter and controller."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from neuro.checkpoint import load_mlp
from neuro.control.trajopt_mpc import (
    TrajOptMPCController,
    TrajOptMPCLog,
    WaveformMLPModel,
    build_waveform_problem,
)
from neuro.predictor.module import AutoregressiveMLP
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from pathlib import Path

    from neuro.types import FloatArray

_SEED = 7


def _random_layers(rng: np.random.Generator, sizes: list[int]) -> tuple[tuple[FloatArray, FloatArray], ...]:
    """Random ``(weight (out, in), bias (out,))`` pairs, drawn uniformly from ``+-1/sqrt(fan_in)``."""
    return tuple(
        (rng.uniform(-1.0, 1.0, (out, inp)) / np.sqrt(inp), rng.uniform(-1.0, 1.0, out) / np.sqrt(inp))
        for inp, out in itertools.pairwise(sizes)
    )


def _build_checkpoint(
    tmp_path: Path,
    *,
    n_y: int = 4,
    n_u: int = 3,
    horizon: int = 3,
    n_channels: int = 2,
    n_controls: int = 2,
    depth: int = 2,
) -> Path:
    """Save a tiny synthetic MLP checkpoint and return its suffix-less stem."""
    rng = np.random.default_rng(_SEED)
    in_size = n_y * n_channels + n_u * n_controls
    scalers = {
        "u_mean": rng.uniform(-1.0, 1.0, n_controls),
        "u_scale": rng.uniform(0.5, 2.0, n_controls),
        "y_mean": rng.uniform(-1.0, 1.0, n_channels),
        "y_scale": rng.uniform(0.5, 2.0, n_channels),
    }
    model = AutoregressiveMLP(
        n_y=n_y,
        n_u=n_u,
        horizon=horizon,
        n_channels=n_channels,
        n_controls=n_controls,
        hidden_size=5,
        depth=depth,
        activation="relu",
        dt=0.01,
        y_std=Standardizer(center=scalers["y_mean"], scale=scalers["y_scale"]),
        u_std=Standardizer(center=scalers["u_mean"], scale=scalers["u_scale"]),
    )
    linears = [m for m in model.layers if isinstance(m, torch.nn.Linear)]
    with torch.no_grad():
        for lin, (w, b) in zip(linears, _random_layers(rng, [in_size, *([5] * depth), n_channels]), strict=True):
            lin.weight.copy_(torch.as_tensor(w, dtype=torch.float32))
            lin.bias.copy_(torch.as_tensor(b, dtype=torch.float32))
    path = tmp_path / "art"
    model.save(path)
    return path


def _lagged_state(artifact: Path, y_ctx: FloatArray, u_ctx: FloatArray) -> FloatArray:
    """MPC-convention state carrying the same trajectory as ``prime(y_ctx, u_ctx)``.

    The controller chain rolls a state whose y-window ends at ``t`` while its u-window ends at
    ``t - 1``; the torch module's ``prime`` ends both windows at ``t - 1``. Lagging the
    u-window and prepending ``u_ctx``'s last entry converts one to the other.
    """
    ckpt = load_mlp(artifact)
    return np.concatenate(
        [
            ckpt.y_std.transform(y_ctx)[-ckpt.n_y :].reshape(-1),
            u_ctx[-ckpt.n_u - 1 : -1].reshape(-1),
        ]
    )


def test_discrete_dynamics_matches_torch_step(tmp_path: Path) -> None:
    """One adapter step predicts the same raw sample as the torch module's step.

    The adapter rolls in the incumbent MPC state convention (u-window lagged one behind the
    y-window), so its prediction input equals the module's primed windows and the predicted
    sample must agree to floating-point tolerance.
    """
    n_y, n_u, n_channels, n_controls = 3, 2, 2, 2
    artifact = _build_checkpoint(tmp_path, n_y=n_y, n_u=n_u, n_channels=n_channels, n_controls=n_controls)
    adapter = WaveformMLPModel.from_checkpoint(artifact)
    module = AutoregressiveMLP.load(artifact)
    ckpt = load_mlp(artifact)

    rng = np.random.default_rng(_SEED + 1)
    ctx = n_y + 2
    y_ctx = rng.standard_normal((ctx, n_channels))
    u_ctx = rng.standard_normal((ctx, n_controls))
    u = u_ctx[-1]

    state = _lagged_state(artifact, y_ctx, u_ctx)
    state_next = np.asarray(adapter.discrete_dynamics(jnp.asarray(state), jnp.asarray(u), 0.0, 0.01))
    y_next = state_next[(n_y - 1) * n_channels : n_y * n_channels] * ckpt.y_std.scale + ckpt.y_std.center

    _, y_want = module.step(module.prime(y_ctx, u_ctx), u)
    np.testing.assert_allclose(y_next, y_want, rtol=1e-5, atol=1e-6)


def test_rollout_matches_torch_rollout(tmp_path: Path) -> None:
    """Chaining adapter steps reproduces the torch module's rollout on the same trajectory."""
    n_y, n_u, n_channels, n_controls = 3, 2, 2, 2
    artifact = _build_checkpoint(tmp_path, n_y=n_y, n_u=n_u, n_channels=n_channels, n_controls=n_controls)
    adapter = WaveformMLPModel.from_checkpoint(artifact)
    module = AutoregressiveMLP.load(artifact)
    ckpt = load_mlp(artifact)

    rng = np.random.default_rng(_SEED + 2)
    ctx = n_y + 2
    horizon = 5
    y_ctx = rng.standard_normal((ctx, n_channels))
    u_ctx = rng.standard_normal((ctx, n_controls))
    u_future = rng.standard_normal((horizon, n_controls))

    want = module.rollout(module.prime(y_ctx, u_ctx), u_future)

    state = _lagged_state(artifact, y_ctx, u_ctx)
    u_seq = np.concatenate([u_ctx[-1:], u_future[: horizon - 1]])
    got = []
    for u in u_seq:
        state = np.asarray(adapter.discrete_dynamics(jnp.asarray(state), jnp.asarray(u), 0.0, 0.01))
        z_last = state[(n_y - 1) * n_channels : n_y * n_channels]
        got.append(z_last * ckpt.y_std.scale + ckpt.y_std.center)
    np.testing.assert_allclose(np.array(got), want, rtol=1e-5, atol=1e-6)


def test_absorb_matches_torch_module(tmp_path: Path) -> None:
    """The adapter's priming seam (initial_state/absorb/is_ready) matches the torch module's."""
    n_y, n_u, n_channels, n_controls = 4, 3, 2, 2
    artifact = _build_checkpoint(tmp_path, n_y=n_y, n_u=n_u, n_channels=n_channels, n_controls=n_controls)
    adapter = WaveformMLPModel.from_checkpoint(artifact)
    module = AutoregressiveMLP.load(artifact)

    rng = np.random.default_rng(_SEED + 3)
    state = adapter.initial_state()
    module_state = module.initial_state()
    np.testing.assert_array_equal(state, module_state)
    assert not adapter.is_ready(state)
    assert not module.is_ready(module_state)

    y_seq = rng.standard_normal((n_y, n_channels))
    u_seq = rng.standard_normal((n_y, n_controls))
    for t in range(n_y):
        state = adapter.absorb(state, y_seq[t], u_seq[t])
        module_state = module.absorb(module_state, y_seq[t], u_seq[t])
        assert adapter.is_ready(state) == module.is_ready(module_state)
        np.testing.assert_array_equal(state, module_state)
    assert adapter.is_ready(state)


def _drive(controller: TrajOptMPCController, n_steps: int, n_channels: int) -> list[tuple[FloatArray, TrajOptMPCLog]]:
    """Feed ``n_steps`` random EEG measurements through ``update`` and collect the outputs."""
    rng = np.random.default_rng(_SEED + 4)
    out = []
    for k in range(n_steps):
        u, log = controller.update(k * controller.dt, ref=np.array([0.0]), x_hat=rng.standard_normal(n_channels))
        out.append((np.atleast_1d(np.asarray(u, dtype=np.float64)), log))
    return out


def test_warmup_emits_zero_until_window_filled(tmp_path: Path) -> None:
    """While the EEG window is still NaN-padded, the controller holds off and emits zeros."""
    n_y = 4
    artifact = _build_checkpoint(tmp_path, n_y=n_y)
    problem = build_waveform_problem(artifact, horizon=3, u_max=0.5, w_y=1.0)
    controller = TrajOptMPCController(dt=0.01, problem=problem)

    results = _drive(controller, n_steps=n_y, n_channels=controller.model.n_channels)
    for u, log in results[: n_y - 1]:
        assert log.warmup
        np.testing.assert_array_equal(u, np.zeros(controller.model.m))

    assert not results[-1][1].warmup


def test_update_respects_bounds(tmp_path: Path) -> None:
    """Past warm-up, update returns a finite ``(n_controls,)`` control within the box bounds."""
    u_max = 0.5
    artifact = _build_checkpoint(tmp_path, n_y=4)
    problem = build_waveform_problem(artifact, horizon=3, u_max=u_max, w_y=1.0, w_u=0.0)
    controller = TrajOptMPCController(dt=0.01, problem=problem)

    u, _ = _drive(controller, n_steps=6, n_channels=controller.model.n_channels)[-1]
    assert u.shape == (controller.model.m,)
    assert np.isfinite(u).all()
    assert np.all(np.abs(u) <= u_max + 1e-6)


def test_pure_effort_cost_yields_zero_control(tmp_path: Path) -> None:
    """With w_y=0 the cost is sum||u||^2, whose unconstrained minimizer is u=0."""
    artifact = _build_checkpoint(tmp_path, n_y=4)
    problem = build_waveform_problem(artifact, horizon=3, u_max=1.0, w_y=0.0, w_u=1.0)
    controller = TrajOptMPCController(dt=0.01, problem=problem)

    u, log = _drive(controller, n_steps=6, n_channels=controller.model.n_channels)[-1]
    assert log.success
    np.testing.assert_allclose(u, np.zeros(controller.model.m), atol=1e-4)


def test_from_config_dispatches_problem_factory(tmp_path: Path) -> None:
    """from_config routes the problem through the ``{class_path, ...}`` factory pattern."""
    artifact = _build_checkpoint(tmp_path, horizon=5)
    controller = TrajOptMPCController.from_config(
        {
            "dt": 0.01,
            "problem": {
                "class_path": "neuro.control.trajopt_mpc.build_waveform_problem",
                "artifact": str(artifact),
                "horizon": 5,
                "u_max": 0.5,
                "w_y": 1.0,
            },
        }
    )
    assert controller.dt == 0.01
    assert controller.problem.N == 6  # horizon + 1 knot points
    assert controller.model.m == 2


def test_per_electrode_bounds_rejected_when_mismatched(tmp_path: Path) -> None:
    """A u_max length that is neither 1 nor n_controls is rejected by the box builder."""
    artifact = _build_checkpoint(tmp_path, n_controls=2)
    with pytest.raises(ValueError, match="could not be broadcast"):
        build_waveform_problem(artifact, horizon=3, u_max=[1.0, 2.0, 3.0])


def test_controller_keeps_absorbed_state_private(tmp_path: Path) -> None:
    """``update`` persists the absorbed state and ``u_last`` as its own attributes.

    ``state.x0`` after ``shift()`` is the prior solve's second knot -- a model prediction, not
    the measurement-corrected state -- so the controller must not be reading its persistent
    state back from the post-solve trajectory.
    """
    artifact = _build_checkpoint(tmp_path, n_y=4)
    problem = build_waveform_problem(artifact, horizon=3, u_max=0.5, w_y=1.0)
    controller = TrajOptMPCController(dt=0.01, problem=problem)
    ckpt = load_mlp(artifact)

    n_z = ckpt.n_y * ckpt.n_channels
    results = _drive(controller, n_steps=6, n_channels=controller.model.n_channels)
    u_last, log_last = results[-1]
    assert not log_last.warmup
    assert not np.isnan(controller._state[:n_z]).any()  # noqa: SLF001
    np.testing.assert_array_equal(controller._u_last, u_last)  # noqa: SLF001
    assert not np.array_equal(np.asarray(controller.state.x0), controller._state)  # noqa: SLF001
