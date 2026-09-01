from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from trajopt.constraints.linear import LinearConstraint
from trajopt.problem import MPCState
from trajopt.solvers.altro import ALTRO
from trajopt.transcription.ipopt import Ipopt

from neuro.config import StftGeometry
from neuro.control.costs import (
    L1ControlCost,
    ObservableFrameHingeCost,
    ObservableHingeCost,
    SpectralHingeCost,
    StateOutputs,
    SumCost,
    has_whole_horizon_cost,
    jax_compute_log_power_frames,
    jax_compute_observable_frames,
)
from neuro.control.mpc import build_waveform_problem
from neuro.predictor.inference import WaveformMLPModel
from neuro.predictor.module import AutoregressiveMLP
from neuro.spectral import ObservableEnvelope, PsdEnvelope, compute_log_power_frames
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from pathlib import Path

    from trajopt.constraints.constraint_list import BuiltConstraintList

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
    horizon: int = 8,
    n_channels: int = 2,
    n_controls: int = 2,
    depth: int = 0,
) -> Path:
    """Save a tiny synthetic (linear when ``depth=0``) MLP checkpoint and return its stem."""
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
        n_outputs=n_channels,
        hidden_size=5,
        depth=depth,
        activation="relu",
        dt=0.01,
        y_std=Standardizer(center=scalers["y_mean"], scale=scalers["y_scale"]),
        u_std=Standardizer(center=scalers["u_mean"], scale=scalers["u_scale"]),
    )
    linears = [m for m in model.layers if isinstance(m, torch.nn.Linear)]
    sizes = [in_size, *([5] * depth), n_channels]
    with torch.no_grad():
        for lin, (w, b) in zip(linears, _random_layers(rng, sizes), strict=True):
            lin.weight.copy_(torch.as_tensor(w, dtype=torch.float32))
            lin.bias.copy_(torch.as_tensor(b, dtype=torch.float32))
    path = tmp_path / "art"
    model.save(path)
    return path


def _ready_state(artifact: Path, rng: np.random.Generator) -> FloatArray:
    """A NaN-free ready model state with a random EEG window, for direct solves."""
    probe = WaveformMLPModel.load(artifact)
    state = np.asarray(probe.initial_state())
    state[: probe.n_y * probe.n_channels] = rng.standard_normal(probe.n_y * probe.n_channels)
    return state


def _full_parity_solver() -> Ipopt:
    """The general Ipopt transcription with L-BFGS and acceptable-level termination.

    The exact-Hessian mode is unusable for the whole-horizon spectral hinge (its per-knot
    expansion misses the hinge's cross-knot terms and converges to a wrong point), and the
    non-smooth L1 kink breaks the limited-memory search direction, so the smooth surrogate plus
    acceptable-level stopping is what makes the full set solve cleanly.
    """
    return Ipopt(
        options={
            "hessian_approximation": "limited-memory",
            "print_level": 0,
            "max_iter": 500,
            "acceptable_tol": 1e-5,
            "acceptable_iter": 5,
            "acceptable_constr_viol_tol": 1e-4,
        }
    )


def test_kirchhoff_constraint_registered_and_enforced(tmp_path: Path) -> None:
    """The full builder adds a ``trajopt.constraints.linear`` Kirchhoff equality; solves enforce it.

    The single-shooting default solver cannot carry the linear equality, so the full-constraint
    problem is solved with the general Ipopt transcription.
    """
    n_controls = 2
    artifact = _build_checkpoint(tmp_path, horizon=6, n_controls=n_controls)

    def _has_linear(constraints: BuiltConstraintList) -> bool:
        return any(
            isinstance(con, LinearConstraint)
            for evaluator in constraints.knot_evaluators
            for con in evaluator.constraints
        )

    without = build_waveform_problem(artifact, horizon=6, u_max=0.8, kirchhoff=False)
    assert not _has_linear(without.constraints)

    with_kirchhoff = build_waveform_problem(artifact, horizon=6, u_max=0.8, kirchhoff=True)
    linear = [
        con
        for evaluator in with_kirchhoff.constraints.knot_evaluators
        for con in evaluator.constraints
        if isinstance(con, LinearConstraint)
    ]
    # One fused copy per active (non-terminal) knot of the single registered constraint.
    assert len({id(con) for con in linear}) == 1
    assert len(linear) == 6
    assert linear[0].A.shape == (1, n_controls)

    rng = np.random.default_rng(_SEED + 4)
    x0 = _ready_state(artifact, rng)
    state = MPCState.initial(with_kirchhoff, x0=jnp.asarray(x0), dt=0.01)
    solved = with_kirchhoff.solve(state, solver=_full_parity_solver())
    assert solved.status == "converged"
    np.testing.assert_allclose(np.sum(np.asarray(solved.controls), axis=1), np.zeros(6), atol=1e-6)


def test_native_solver_converges_on_smooth_l1(tmp_path: Path) -> None:
    """ALTRO (native JAX) solves the local quadratic + smooth-L1 + box + Kirchhoff problem.

    The whole-horizon spectral hinge is invisible to per-knot Taylor expansions, so the native
    path carries only the local costs -- which includes the smooth L1 surrogate, the data behind
    keeping it rather than restricting L1 sparsity to transcription-based solvers.
    """
    horizon = 6
    artifact = _build_checkpoint(tmp_path, horizon=horizon, depth=0)
    problem = build_waveform_problem(
        artifact, horizon=horizon, u_max=0.8, w_y=1.0, w_u=0.05, w_u_l1=0.5, kirchhoff=True
    )
    rng = np.random.default_rng(_SEED + 7)
    x0 = _ready_state(artifact, rng)
    state = MPCState.initial(problem, x0=jnp.asarray(x0), dt=0.01)
    solved = problem.solve(state, solver=ALTRO())
    assert solved.status == "converged"
    np.testing.assert_allclose(np.sum(np.asarray(solved.controls), axis=1), np.zeros(horizon), atol=1e-4)


def test_l1_cost_stage_values_match_epigraph() -> None:
    """The smooth surrogate's per-knot values match the epigraph's on fixed controls, up to ``eps``."""
    rng = np.random.default_rng(_SEED + 8)
    horizon, m = 5, 2
    u_seq = rng.uniform(-0.8, 0.8, (horizon, m))
    w_l1 = 0.5
    epigraph = (w_l1 / horizon) * np.sum(np.abs(u_seq))
    surrogate = L1ControlCost(n=4, m=m, w_l1=w_l1, horizon=horizon).stage_costs(
        jnp.zeros((horizon, 4)), jnp.asarray(u_seq), jnp.zeros(horizon)
    )
    np.testing.assert_allclose(float(jnp.sum(surrogate)), epigraph, atol=1e-3)


def test_sum_cost_composes_evaluate_and_stage_costs() -> None:
    """``SumCost`` routes per-knot and whole-horizon sub-costs through their own evaluation paths."""
    rng = np.random.default_rng(_SEED + 9)
    n, m, horizon = 4, 2, 5
    x_seq = rng.standard_normal((horizon, n))
    u_seq = rng.standard_normal((horizon, m))
    quadratic = L1ControlCost(n=n, m=m, w_l1=0.0, horizon=horizon)
    l1 = L1ControlCost(n=n, m=m, w_l1=0.5, horizon=horizon)
    combined = SumCost([quadratic, l1])
    per_knot = combined.evaluate(jnp.asarray(x_seq[0]), jnp.asarray(u_seq[0]))
    np.testing.assert_allclose(float(per_knot), float(l1.evaluate(jnp.asarray(x_seq[0]), jnp.asarray(u_seq[0]))))
    total = float(jnp.sum(combined.stage_costs(jnp.asarray(x_seq), jnp.asarray(u_seq), jnp.zeros(horizon))))
    np.testing.assert_allclose(
        total, float(jnp.sum(l1.stage_costs(jnp.asarray(x_seq), jnp.asarray(u_seq), jnp.zeros(horizon))))
    )


def test_spectral_hinge_jax_reduction_agrees_with_canonical_numpy() -> None:
    """The jax reduction helper inside SpectralHingeCost agrees with canonical NumPy to float tolerance."""
    rng = np.random.default_rng(_SEED + 10)
    horizon, n_channels, fs, window, hop = 100, 3, 50.0, 50, 25
    y = rng.standard_normal((horizon, n_channels))

    geom = StftGeometry(n_segment=window, n_hop=hop)
    numpy_frames = compute_log_power_frames(y, geom, fs=fs)

    jax_frames = jax_compute_log_power_frames(jnp.asarray(y), fs=fs, window=window, hop=hop)

    assert jax_frames.shape == numpy_frames.shape
    np.testing.assert_allclose(np.asarray(jax_frames), numpy_frames, rtol=1e-10, atol=1e-12)


def test_spectral_hinge_cost_is_model_free_and_scores_stage_trajectory() -> None:
    """SpectralHingeCost constructs without a model instance and scores every stage Frame."""
    rng = np.random.default_rng(_SEED + 11)
    n_y, n_channels, n_u, n_controls = 4, 3, 2, 2
    horizon, window, hop, fs = 100, 50, 25, 50.0
    n = n_y * n_channels + n_u * n_controls
    m = n_controls

    y_center = rng.uniform(-1.0, 1.0, n_channels)
    y_scale = rng.uniform(0.5, 2.0, n_channels)
    envelope_power = rng.uniform(0.1, 1.0, (n_channels, window // 2 + 1))
    envelope = PsdEnvelope(
        power=envelope_power,
        fs=fs,
        window=window,
        hop=hop,
    )

    outputs = StateOutputs(n=n, m=m, n_y=n_y, n_outputs=n_channels, center=y_center, scale=y_scale)
    cost = SpectralHingeCost(outputs, envelope, w_psd=10.0, horizon=horizon)

    # evaluate returns 0 for native expansions
    x_single = jnp.asarray(rng.standard_normal(n))
    np.testing.assert_equal(float(cost.evaluate(x_single)), 0.0)

    # stage_costs decodes every stage state and scores the exact windowed hinge
    X = rng.standard_normal((horizon, n))
    U = rng.standard_normal((horizon, m))
    stage_vals = cost.stage_costs(jnp.asarray(X), jnp.asarray(U), jnp.zeros(horizon))

    # Entries past 0 must be 0
    np.testing.assert_array_equal(np.asarray(stage_vals[1:]), np.zeros(horizon - 1))

    # Check value at index 0 against NumPy
    z_last = slice((n_y - 1) * n_channels, n_y * n_channels)
    y_stage = X[:, z_last] * y_scale + y_center  # (horizon, n_channels)
    geom = StftGeometry(n_segment=window, n_hop=hop)
    numpy_frames = compute_log_power_frames(y_stage, geom, fs=fs)
    assert numpy_frames.shape[0] == (horizon - window) // hop + 1
    log_excess = numpy_frames - np.log(envelope_power[None, :, 1:])
    want_value = 10.0 * float(np.mean(np.maximum(0.0, log_excess) ** 2))
    np.testing.assert_allclose(float(stage_vals[0]), want_value, rtol=1e-10, atol=1e-12)


def test_spectral_hinge_window_grid_spans_a_whole_control_horizon() -> None:
    """The grid holds the ``horizon`` windows the Control Horizon implies, not ``horizon - 1``.

    ``configs/simulation/mse02_psd_mpc_spectral.yaml`` runs horizon 75 at window 50, hop 25:
    two windows. Scoring one Frame fewer silently drops the second, leaving the horizon's last
    third unpriced.
    """
    rng = np.random.default_rng(_SEED + 13)
    n_y, n_channels, n_u, n_controls = 4, 2, 2, 2
    horizon, window, hop, fs = 75, 50, 25, 50.0
    n = n_y * n_channels + n_u * n_controls

    envelope = PsdEnvelope(
        power=rng.uniform(0.1, 1.0, (n_channels, window // 2 + 1)),
        fs=fs,
        window=window,
        hop=hop,
    )
    outputs = StateOutputs(
        n=n, m=n_controls, n_y=n_y, n_outputs=n_channels, center=np.zeros(n_channels), scale=np.ones(n_channels)
    )
    SpectralHingeCost(outputs, envelope, w_psd=1.0, horizon=horizon)

    X = jnp.asarray(rng.standard_normal((horizon, n)))
    frames = jax_compute_log_power_frames(outputs.decode(X), fs=fs, window=window, hop=hop)
    assert frames.shape[0] == 2

    # One Frame short of the Control Horizon the grid loses a whole window, so the value moves.
    short = jax_compute_log_power_frames(outputs.decode(X[1:]), fs=fs, window=window, hop=hop)
    assert short.shape[0] == 1


def test_spectral_hinge_pins_a_seeded_rollout(tmp_path: Path) -> None:
    """Pin the hinge on a fixed seeded model rollout, guarding the window grid's anchor.

    Before the model-free refactor the Cost stepped the model once more to recover the terminal
    knot's Frame and scored ``y_1 .. y_H``; this exact rollout scored ``100.35113257399541``
    there, over 3 windows. A Cost sees the terminal knot one state at a time, and an FFT window
    straddling it does not split into a stage term plus a terminal term, so the grid is anchored
    one Frame earlier and scores ``y_0 .. y_{H-1}`` -- the same 3 windows spanning the same
    Control Horizon, shifted by one sample.
    """
    n_y, n_u, n_channels, n_controls = 4, 3, 2, 2
    horizon, window, hop, fs = 100, 50, 25, 50.0
    artifact = _build_checkpoint(
        tmp_path, n_y=n_y, n_u=n_u, horizon=horizon, n_channels=n_channels, n_controls=n_controls
    )
    model = WaveformMLPModel.load(artifact)

    rng = np.random.default_rng(_SEED + 21)
    envelope = PsdEnvelope(power=rng.uniform(0.1, 1.0, (n_channels, window // 2 + 1)), fs=fs, window=window, hop=hop)
    x0 = np.asarray(model.initial_state())
    x0[: n_y * n_channels] = rng.standard_normal(n_y * n_channels)
    U = jnp.asarray(rng.standard_normal((horizon, n_controls)))
    states = [jnp.asarray(x0)]
    for u in U:
        states.append(model.discrete_dynamics(states[-1], u, 0.0, 0.0))
    X = jnp.stack(states[:-1])

    outputs = StateOutputs(
        n=model.n, m=model.m, n_y=n_y, n_outputs=n_channels, center=model.y_center, scale=model.y_scale
    )
    cost = SpectralHingeCost(outputs, envelope, w_psd=10.0, horizon=horizon)
    assert jax_compute_log_power_frames(outputs.decode(X), fs=fs, window=window, hop=hop).shape[0] == 3

    value = float(cost.stage_costs(X, U, jnp.zeros(horizon))[0])
    np.testing.assert_allclose(value, 95.680650452903251, rtol=1e-9)


def test_spectral_hinge_cost_validation() -> None:
    """SpectralHingeCost rejects mismatched channel counts or horizons shorter than window."""
    envelope = PsdEnvelope(
        power=np.ones((3, 26)),
        fs=50.0,
        window=50,
        hop=25,
    )
    # Channel count mismatch (envelope has 3, n_channels=2)
    with pytest.raises(ValueError, match="envelope has 3 channels but the model outputs 2"):
        SpectralHingeCost(
            StateOutputs(n=10, m=2, n_y=2, n_outputs=2, center=np.zeros(2), scale=np.ones(2)),
            envelope,
            w_psd=1.0,
            horizon=60,
        )

    # Horizon equal to the window still scores one whole window; anything shorter cannot.
    outputs = StateOutputs(n=12, m=2, n_y=2, n_outputs=3, center=np.zeros(3), scale=np.ones(3))
    assert SpectralHingeCost(outputs, envelope, w_psd=1.0, horizon=50).window == 50
    with pytest.raises(ValueError, match="horizon \\(40\\) is shorter than the envelope window \\(50\\)"):
        SpectralHingeCost(outputs, envelope, w_psd=1.0, horizon=40)


def test_observable_hinge_cost_matches_numpy_reference() -> None:
    """ObservableHingeCost constructs without a model instance and scores stage states."""
    rng = np.random.default_rng(_SEED + 12)
    n_y, n_channels, n_values, n_u, n_controls = 4, 3, 5, 2, 2
    horizon, fs = 10, 50.0
    n_outputs = n_channels * n_values
    n = n_y * n_outputs + n_u * n_controls
    m = n_controls

    geom = StftGeometry(n_segment=20, n_hop=5, band_hz=(4.0, 24.0), n_bin_pool=2)
    y_center = rng.uniform(-1.0, 1.0, n_outputs)
    y_scale = rng.uniform(0.5, 2.0, n_outputs)
    envelope_power = rng.uniform(-5.0, 0.0, (n_channels, n_values))
    envelope = ObservableEnvelope(
        power=envelope_power,
        fs=fs,
        geometry=geom,
    )

    outputs = StateOutputs(n=n, m=m, n_y=n_y, n_outputs=n_outputs, center=y_center, scale=y_scale)
    cost = ObservableHingeCost(outputs, envelope, w_hinge=10.0, horizon=horizon)
    terminal = ObservableHingeCost(outputs, envelope, w_hinge=10.0, horizon=horizon, terminal=True)

    # evaluate returns 0 at a stage knot, so native expansions cannot mis-score it
    x_single = jnp.asarray(rng.standard_normal(n))
    np.testing.assert_equal(float(cost.evaluate(x_single)), 0.0)

    # stage_costs decodes X[1:] and scores the predicted Frames the stage trajectory carries
    X = rng.standard_normal((horizon + 1, n))
    U = rng.standard_normal((horizon, m))
    stage_vals = cost.stage_costs(jnp.asarray(X[:-1]), jnp.asarray(U), jnp.zeros(horizon))

    # Entries past 0 must be 0
    np.testing.assert_array_equal(np.asarray(stage_vals[1:]), np.zeros(horizon - 1))

    # Stage plus terminal is the NumPy mean over every Frame of the Control Horizon
    total = float(jnp.sum(stage_vals)) + float(terminal.evaluate(jnp.asarray(X[-1])))
    z_last = slice((n_y - 1) * n_outputs, n_y * n_outputs)
    y_horizon = X[1:, z_last] * y_scale + y_center  # (horizon, n_outputs)
    log_excess = y_horizon - envelope_power.reshape(1, -1)
    want_value = 10.0 * float(np.mean(np.maximum(0.0, log_excess) ** 2))
    np.testing.assert_allclose(total, want_value, rtol=1e-10, atol=1e-12)


def test_observable_hinge_scores_every_control_horizon_frame() -> None:
    """The stage and terminal Costs together price exactly ``horizon`` Frames, none dropped."""
    horizon, n_channels, n_values, n_controls = 4, 3, 5, 2
    n_outputs = n_channels * n_values
    n = n_outputs + 2 * n_controls

    envelope = ObservableEnvelope(
        power=np.zeros((n_channels, n_values)),
        fs=50.0,
        geometry=StftGeometry(n_segment=20, n_hop=5),
    )
    outputs = StateOutputs(
        n=n, m=n_controls, n_y=1, n_outputs=n_outputs, center=np.zeros(n_outputs), scale=np.ones(n_outputs)
    )
    stage = ObservableHingeCost(outputs, envelope, w_hinge=1.0, horizon=horizon)
    terminal = ObservableHingeCost(outputs, envelope, w_hinge=1.0, horizon=horizon, terminal=True)

    # Every Frame sits exactly one unit above the envelope, so each scored Frame adds 1 / horizon.
    X = jnp.ones((horizon + 1, n))
    stage_total = float(jnp.sum(stage.stage_costs(X[:-1], jnp.zeros((horizon, n_controls)), jnp.zeros(horizon))))
    np.testing.assert_allclose(stage_total, (horizon - 1) / horizon, rtol=1e-12)
    np.testing.assert_allclose(stage_total + float(terminal.evaluate(X[-1])), 1.0, rtol=1e-12)


def test_observable_hinge_cost_validation() -> None:
    """ObservableHingeCost rejects mismatched output dimensions or non-positive horizon."""
    geom = StftGeometry(n_segment=20, n_hop=5)
    envelope = ObservableEnvelope(
        power=np.ones((3, 5)),
        fs=50.0,
        geometry=geom,
    )
    # Output width mismatch (envelope has 3 * 5 = 15, n_outputs=10)
    with pytest.raises(
        ValueError, match="envelope has 3 channels and 5 values \\(15 total\\) but the model output width is 10"
    ):
        ObservableHingeCost(
            StateOutputs(n=20, m=2, n_y=2, n_outputs=10, center=np.zeros(10), scale=np.ones(10)),
            envelope,
            w_hinge=1.0,
            horizon=5,
        )

    # Horizon < 1
    with pytest.raises(ValueError, match="horizon \\(0\\) must be at least 1"):
        ObservableHingeCost(
            StateOutputs(n=34, m=2, n_y=2, n_outputs=15, center=np.zeros(15), scale=np.ones(15)),
            envelope,
            w_hinge=1.0,
            horizon=0,
        )


def _observable_geometry() -> StftGeometry:
    """A pooled, kernel-smoothed geometry, so the test exercises every reduction stage."""
    return StftGeometry(
        n_segment=32,
        n_hop=16,
        band_hz=(4.0, 30.0),
        n_bin_pool=2,
        kernel="hann",
        kernel_width=3,
    )


def test_observable_frame_jax_reduction_agrees_with_canonical_numpy() -> None:
    """The jax Observable reduction reproduces :func:`compute_log_power_frames` stage for stage."""
    rng = np.random.default_rng(_SEED + 20)
    geom = _observable_geometry()
    fs = 100.0
    y = rng.standard_normal((200, 4))

    numpy_frames = compute_log_power_frames(y, geom, fs=fs)
    jax_frames = jax_compute_observable_frames(jnp.asarray(y), geom, fs=fs)

    assert jax_frames.shape == numpy_frames.shape
    np.testing.assert_allclose(np.asarray(jax_frames), numpy_frames, rtol=1e-10, atol=1e-12)


def test_observable_frame_hinge_cost_scores_the_stage_waveform() -> None:
    """ObservableFrameHingeCost reduces the stage waveform to Frames and hinges them, whole-horizon."""
    rng = np.random.default_rng(_SEED + 21)
    geom = _observable_geometry()
    n_y, n_channels, n_u, n_controls = 4, 3, 2, 2
    horizon, fs = 200, 100.0
    n = n_y * n_channels + n_u * n_controls
    m = n_controls

    y_center = rng.uniform(-1.0, 1.0, n_channels)
    y_scale = rng.uniform(0.5, 2.0, n_channels)
    envelope = ObservableEnvelope(
        power=rng.uniform(-2.0, 2.0, (n_channels, geom.n_values(fs))),
        fs=fs,
        geometry=geom,
    )
    outputs = StateOutputs(n=n, m=m, n_y=n_y, n_outputs=n_channels, center=y_center, scale=y_scale)
    cost = ObservableFrameHingeCost(outputs, envelope, w_hinge=10.0, horizon=horizon)

    np.testing.assert_equal(float(cost.evaluate(jnp.asarray(rng.standard_normal(n)))), 0.0)

    X = rng.standard_normal((horizon, n))
    stage_vals = cost.stage_costs(jnp.asarray(X), jnp.asarray(rng.standard_normal((horizon, m))), jnp.zeros(horizon))
    np.testing.assert_array_equal(np.asarray(stage_vals[1:]), np.zeros(horizon - 1))

    newest = slice((n_y - 1) * n_channels, n_y * n_channels)
    y_stage = X[:, newest] * y_scale + y_center
    numpy_frames = compute_log_power_frames(y_stage, geom, fs=fs)
    want = 10.0 * float(np.mean(np.maximum(0.0, numpy_frames - envelope.power[None]) ** 2))
    np.testing.assert_allclose(float(stage_vals[0]), want, rtol=1e-10, atol=1e-12)


def test_observable_frame_hinge_is_zero_under_the_envelope_and_registered_whole_horizon() -> None:
    """The hinge vanishes when every Frame sits under the envelope, and the solver guard sees it."""
    rng = np.random.default_rng(_SEED + 22)
    geom = _observable_geometry()
    n_channels, horizon, fs = 2, 200, 100.0
    n, m = n_channels, 1

    outputs = StateOutputs(
        n=n,
        m=m,
        n_y=1,
        n_outputs=n_channels,
        center=np.zeros(n_channels),
        scale=np.ones(n_channels),
    )
    X = 1e-3 * rng.standard_normal((horizon, n))
    quiet = ObservableEnvelope(
        power=np.full((n_channels, geom.n_values(fs)), 10.0),
        fs=fs,
        geometry=geom,
    )
    cost = ObservableFrameHingeCost(outputs, quiet, w_hinge=1.0, horizon=horizon)
    stage_vals = cost.stage_costs(jnp.asarray(X), jnp.zeros((horizon, m)), jnp.zeros(horizon))
    np.testing.assert_allclose(float(stage_vals[0]), 0.0, atol=0.0)

    assert has_whole_horizon_cost(cost)
    assert has_whole_horizon_cost(SumCost([L1ControlCost(n=n, m=m, w_l1=1.0, horizon=horizon), cost]))


def test_observable_frame_hinge_cost_validation() -> None:
    """The cost rejects envelopes of the wrong width and horizons shorter than one Frame's support."""
    geom = _observable_geometry()
    fs = 100.0
    n_channels = 3
    outputs = StateOutputs(
        n=n_channels,
        m=1,
        n_y=1,
        n_outputs=n_channels,
        center=np.zeros(n_channels),
        scale=np.ones(n_channels),
    )
    envelope = ObservableEnvelope(power=np.zeros((n_channels, geom.n_values(fs))), fs=fs, geometry=geom)

    with pytest.raises(ValueError, match="shorter than the sample support"):
        ObservableFrameHingeCost(outputs, envelope, w_hinge=1.0, horizon=geom.sample_support_steps(fs) - 1)

    wide = ObservableEnvelope(power=np.zeros((n_channels + 1, geom.n_values(fs))), fs=fs, geometry=geom)
    with pytest.raises(ValueError, match="channels but the model outputs"):
        ObservableFrameHingeCost(outputs, wide, w_hinge=1.0, horizon=200)
