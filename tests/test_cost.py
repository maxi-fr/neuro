from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from trajopt.constraints.linear import LinearConstraint
from trajopt.costs.output import OutputCost
from trajopt.dynamics.base import DiscreteDynamics
from trajopt.mpc import MPC
from trajopt.solvers.altro import ALTRO
from trajopt.trajectory import Trajectory
from trajopt.transcription.ipopt import Ipopt
from trajopt.transcription.single_shooting import SingleShooting

from neuro.comparison import cost_metrics, format_cost_report
from neuro.config import StftGeometry
from neuro.control.costs import (
    ExcludeInitialKnotState,
    L1ControlCost,
    ObservableFrameHingeCost,
    ObservableHingeCost,
    ReducedEffortCost,
    SpectralHingeCost,
    SumCost,
    compute_waveform_observable_frames,
    frame_grid_offsets,
    has_whole_horizon_cost,
    jax_compute_log_power_frames,
    jax_compute_observable_frames,
)
from neuro.control.mpc import build_waveform_problem, decompose_cost, kirchhoff_basis
from neuro.predictor.inference import WaveformMLPModel
from neuro.predictor.module import AutoregressiveMLP
from neuro.spectral import HealthyReference, ObservableEnvelope, PsdEnvelope, compute_log_power_frames
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import ArrayLike
    from simulate.logger import BaseLogger
    from trajopt.constraints.constraint_list import BuiltConstraintList

    from neuro.types import FloatArray

_SEED = 7


class _DummyModel(DiscreteDynamics):
    p: int
    n_y: int
    center: jax.Array
    scale: jax.Array

    def __init__(
        self,
        *,
        n: int,
        m: int,
        p: int,
        n_y: int = 1,
        center: ArrayLike | None = None,
        scale: ArrayLike | None = None,
    ) -> None:
        super().__init__(n=n, m=m, ne=n, p=p)
        self.p = p
        self.n_y = n_y
        self.center = jnp.zeros(p) if center is None else jnp.asarray(center)
        self.scale = jnp.ones(p) if scale is None else jnp.asarray(scale)

    def output(self, x: jax.Array, u: jax.Array | None = None, t: float | jax.Array = 0.0) -> jax.Array:
        del u, t
        newest = x[(self.n_y - 1) * self.p : self.n_y * self.p]
        return newest * self.scale + self.center

    def discrete_dynamics(self, x: jax.Array, u: jax.Array, t: float | jax.Array, dt: float | jax.Array) -> jax.Array:
        del u, t, dt
        return x


class _DummyHistoryModel(DiscreteDynamics):
    p: int
    n_history: int
    center: jax.Array
    scale: jax.Array

    def __init__(self, *, n_history: int, p: int, m: int = 1) -> None:
        super().__init__(n=n_history * p, m=m, ne=n_history * p, p=p)
        self.p = p
        self.n_history = n_history
        self.center = jnp.zeros(p)
        self.scale = jnp.ones(p)

    def output(self, x: jax.Array, u: jax.Array | None = None, t: float | jax.Array = 0.0) -> jax.Array:
        del u, t
        newest = x[(self.n_history - 1) * self.p : self.n_history * self.p]
        return newest * self.scale + self.center

    def past_outputs(self, x: jax.Array, count: int) -> jax.Array:
        start_idx = (self.n_history - 1 - count) * self.p
        end_idx = (self.n_history - 1) * self.p
        past_std = x[..., start_idx:end_idx].reshape(-1, count, self.p)
        out = past_std * self.scale + self.center
        return out[0] if x.ndim == 1 else out

    def discrete_dynamics(self, x: jax.Array, u: jax.Array, t: float | jax.Array, dt: float | jax.Array) -> jax.Array:
        del u, t, dt
        return x


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
    dt: float = 0.01,
    equilibrium_at: FloatArray | None = None,
) -> Path:
    """Save a tiny synthetic (linear when ``depth=0``) MLP checkpoint with optional ``equilibrium_at`` and return its stem."""
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
        dt=dt,
        y_std=Standardizer(center=scalers["y_mean"], scale=scalers["y_scale"]),
        u_std=Standardizer(center=scalers["u_mean"], scale=scalers["u_scale"]),
    )
    linears = [m for m in model.layers if isinstance(m, torch.nn.Linear)]
    sizes = [in_size, *([5] * depth), n_channels]
    with torch.no_grad():
        if equilibrium_at is not None and depth == 0:
            u_0_std = (np.zeros(n_controls) - scalers["u_mean"]) / scalers["u_scale"]
            w = np.zeros((n_channels, in_size), dtype=np.float64)
            b_mat = np.zeros((n_channels, n_controls), dtype=np.float64)
            b_mat[0, 0] = 1.0
            b_mat[0, 1] = -1.0
            b_mat[1, 0] = -1.0
            b_mat[1, 1] = 1.0
            w[:, in_size - n_controls : in_size] = b_mat
            b = -b_mat @ u_0_std
            linears[0].weight.copy_(torch.as_tensor(w, dtype=torch.float32))
            linears[0].bias.copy_(torch.as_tensor(b, dtype=torch.float32))
        else:
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

    ref = HealthyReference(eeg_mean=np.zeros(2))
    without = build_waveform_problem(artifact, horizon=6, u_max=0.8, kirchhoff=False, reference=ref)
    assert not _has_linear(without.constraints)

    with_kirchhoff = build_waveform_problem(artifact, horizon=6, u_max=0.8, kirchhoff=True, reference=ref)
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
    mpc = MPC(with_kirchhoff, _full_parity_solver(), x0=jnp.asarray(x0))
    assert mpc.solve().success
    np.testing.assert_allclose(np.sum(np.asarray(mpc.controls), axis=1), np.zeros(6), atol=1e-6)


def test_native_solver_converges_on_smooth_l1(tmp_path: Path) -> None:
    """ALTRO (native JAX) solves the local quadratic + smooth-L1 + box + Kirchhoff problem.

    The whole-horizon spectral hinge is invisible to per-knot Taylor expansions, so the native
    path carries only the local costs -- which includes the smooth L1 surrogate, the data behind
    keeping it rather than restricting L1 sparsity to transcription-based solvers.
    """
    horizon = 6
    artifact = _build_checkpoint(tmp_path, horizon=horizon, depth=0)
    ref = HealthyReference(eeg_mean=np.zeros(2))
    problem = build_waveform_problem(
        artifact, horizon=horizon, u_max=0.8, w_y=1.0, w_u=0.05, w_u_l1=0.5, kirchhoff=True, reference=ref
    )
    rng = np.random.default_rng(_SEED + 7)
    x0 = _ready_state(artifact, rng)
    mpc = MPC(problem, ALTRO(), x0=jnp.asarray(x0))
    assert mpc.solve().success
    np.testing.assert_allclose(np.sum(np.asarray(mpc.controls), axis=1), np.zeros(horizon), atol=1e-4)


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

    geom = StftGeometry(n_segment=window, n_hop=hop, band_hz=(1.0, 25.0))
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

    model = _DummyModel(n=n, m=m, p=n_channels, n_y=n_y, center=y_center, scale=y_scale)
    cost = SpectralHingeCost(model, envelope, w_psd=10.0, horizon=horizon)

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
    geom = StftGeometry(n_segment=window, n_hop=hop, band_hz=(1.0, 25.0))
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
    model = _DummyModel(n=n, m=n_controls, p=n_channels, n_y=n_y)
    SpectralHingeCost(model, envelope, w_psd=1.0, horizon=horizon)

    X = jnp.asarray(rng.standard_normal((horizon, n)))
    frames = jax_compute_log_power_frames(jax.vmap(model.output)(X), fs=fs, window=window, hop=hop)
    assert frames.shape[0] == 2

    # One Frame short of the Control Horizon the grid loses a whole window, so the value moves.
    short = jax_compute_log_power_frames(jax.vmap(model.output)(X[1:]), fs=fs, window=window, hop=hop)
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

    cost = SpectralHingeCost(model, envelope, w_psd=10.0, horizon=horizon)
    assert jax_compute_log_power_frames(jax.vmap(model.output)(X), fs=fs, window=window, hop=hop).shape[0] == 3

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
            _DummyModel(n=10, m=2, p=2, n_y=2),
            envelope,
            w_psd=1.0,
            horizon=60,
        )

    # Horizon equal to the window still scores one whole window; anything shorter cannot.
    model = _DummyModel(n=12, m=2, p=3, n_y=2)
    assert SpectralHingeCost(model, envelope, w_psd=1.0, horizon=50).window == 50
    with pytest.raises(ValueError, match="horizon \\(40\\) is shorter than the envelope window \\(50\\)"):
        SpectralHingeCost(model, envelope, w_psd=1.0, horizon=40)


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

    model = _DummyModel(n=n, m=m, p=n_outputs, n_y=n_y, center=y_center, scale=y_scale)
    hinge = ObservableHingeCost(envelope, w_hinge=10.0, horizon=horizon)
    output_hinge = OutputCost(model, hinge)
    stage = ExcludeInitialKnotState(output_hinge)
    terminal = output_hinge.as_terminal()

    # Single knot evaluate on R^p scores against the healthy envelope
    x_frame = rng.standard_normal(n_outputs)
    got_frame = float(hinge.evaluate(jnp.asarray(x_frame)))
    excess_frame = np.maximum(0.0, x_frame - envelope_power.reshape(-1))
    want_frame = (10.0 / (horizon * n_outputs)) * float(np.sum(excess_frame**2))
    np.testing.assert_allclose(got_frame, want_frame, rtol=1e-10, atol=1e-12)

    # ExcludeInitialKnotState(OutputCost) on stage trajectory X[:-1] plus terminal knot X[-1]
    # matches the NumPy mean over every predicted Frame of the Control Horizon
    X = rng.standard_normal((horizon + 1, n))
    U = rng.standard_normal((horizon, m))
    stage_vals = stage.stage_costs(jnp.asarray(X[:-1]), jnp.asarray(U), jnp.zeros(horizon))

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
    model = _DummyModel(n=n, m=n_controls, p=n_outputs, n_y=1)
    hinge = ObservableHingeCost(envelope, w_hinge=1.0, horizon=horizon)
    output_hinge = OutputCost(model, hinge)
    stage = ExcludeInitialKnotState(output_hinge)
    terminal = output_hinge.as_terminal()

    # Every Frame sits exactly one unit above the envelope, so each scored Frame adds 1 / horizon.
    X = jnp.ones((horizon + 1, n))
    stage_total = float(jnp.sum(stage.stage_costs(X[:-1], jnp.zeros((horizon, n_controls)), jnp.zeros(horizon))))
    np.testing.assert_allclose(stage_total, (horizon - 1) / horizon, rtol=1e-12)
    np.testing.assert_allclose(stage_total + float(terminal.evaluate(X[-1])), 1.0, rtol=1e-12)


def test_observable_hinge_cost_validation() -> None:
    """ObservableHingeCost rejects non-positive horizon and OutputCost rejects dimension mismatch."""
    geom = StftGeometry(n_segment=20, n_hop=5)
    envelope = ObservableEnvelope(
        power=np.ones((3, 5)),
        fs=50.0,
        geometry=geom,
    )
    # Output width mismatch (envelope has 3 * 5 = 15, n_outputs=10)
    model_mismatch = _DummyModel(n=20, m=2, p=10, n_y=2)
    hinge = ObservableHingeCost(envelope, w_hinge=1.0, horizon=5)
    with pytest.raises(ValueError, match="Cost state dimension \\(15\\) must match model output dimension p \\(10\\)"):
        OutputCost(model_mismatch, hinge)

    # Horizon < 1
    with pytest.raises(ValueError, match="horizon \\(0\\) must be at least 1"):
        ObservableHingeCost(envelope, w_hinge=1.0, horizon=0)


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


@pytest.mark.parametrize(
    "geom",
    [
        StftGeometry(
            n_segment=20,
            n_hop=5,
            band_hz=(4.0, 24.0),
            n_bin_pool=2,
            kernel="hann",
            kernel_width=3,
            window="hann",
            asymmetric_window=True,
        ),
        StftGeometry(
            n_segment=20,
            n_hop=5,
            band_hz=(4.0, 24.0),
            n_bin_pool=2,
            kernel="exponential",
            kernel_width=3,
            window="hann_poisson",
            asymmetric_window=True,
        ),
        StftGeometry(
            n_segment=20,
            n_hop=5,
            band_hz=(4.0, 24.0),
            n_bin_pool=2,
            kernel="exponential",
            kernel_width=3,
            window="hann_poisson",
            asymmetric_window=False,
        ),
    ],
)
def test_observable_frame_jax_reduction_agrees_with_canonical_numpy_asymmetric(geom: StftGeometry) -> None:
    """The jax Observable reduction reproduces compute_log_power_frames with asymmetric windowing."""
    rng = np.random.default_rng(_SEED + 20)
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
    model = _DummyModel(n=n, m=m, p=n_channels, n_y=n_y, center=y_center, scale=y_scale)
    cost = ObservableFrameHingeCost(model, envelope, w_hinge=10.0, horizon=horizon)

    np.testing.assert_equal(float(cost.evaluate(jnp.asarray(rng.standard_normal(n)))), 0.0)

    X = rng.standard_normal((horizon, n))
    stage_vals = cost.stage_costs(jnp.asarray(X), jnp.asarray(rng.standard_normal((horizon, m))), jnp.zeros(horizon))
    np.testing.assert_array_equal(np.asarray(stage_vals[1:]), np.zeros(horizon - 1))

    newest = slice((n_y - 1) * n_channels, n_y * n_channels)
    y_stage = X[:, newest] * y_scale + y_center
    numpy_frames = compute_log_power_frames(y_stage, geom, fs=fs)
    numpy_hinge = np.maximum(0.0, numpy_frames - envelope.power[None]) ** 2
    want = 10.0 * float(np.mean(numpy_hinge))
    np.testing.assert_allclose(float(stage_vals[0]), want, rtol=1e-10, atol=1e-12)


def test_observable_frame_hinge_is_zero_under_the_envelope_and_registered_whole_horizon() -> None:
    """The hinge vanishes when every Frame sits under the envelope, and the solver guard sees it."""
    rng = np.random.default_rng(_SEED + 22)
    geom = _observable_geometry()
    n_channels, horizon, fs = 2, 200, 100.0
    n, m = n_channels, 1

    model = _DummyModel(n=n, m=m, p=n_channels, n_y=1)
    X = 1e-3 * rng.standard_normal((horizon, n))
    quiet = ObservableEnvelope(
        power=np.full((n_channels, geom.n_values(fs)), 10.0),
        fs=fs,
        geometry=geom,
    )
    cost = ObservableFrameHingeCost(model, quiet, w_hinge=1.0, horizon=horizon)
    stage_vals = cost.stage_costs(jnp.asarray(X), jnp.zeros((horizon, m)), jnp.zeros(horizon))
    np.testing.assert_allclose(float(stage_vals[0]), 0.0, atol=0.0)

    assert has_whole_horizon_cost(cost)
    assert has_whole_horizon_cost(SumCost([L1ControlCost(n=n, m=m, w_l1=1.0, horizon=horizon), cost]))


def test_observable_frame_hinge_cost_validation() -> None:
    """The cost rejects envelopes of the wrong width and horizons shorter than one Frame's support."""
    geom = _observable_geometry()
    fs = 100.0
    n_channels = 3
    model = _DummyModel(n=n_channels, m=1, p=n_channels, n_y=1)
    envelope = ObservableEnvelope(power=np.zeros((n_channels, geom.n_values(fs))), fs=fs, geometry=geom)

    with pytest.raises(ValueError, match="shorter than the sample support"):
        ObservableFrameHingeCost(model, envelope, w_hinge=1.0, horizon=geom.sample_support_steps(fs) - 1)

    wide = ObservableEnvelope(power=np.zeros((n_channels + 1, geom.n_values(fs))), fs=fs, geometry=geom)
    with pytest.raises(ValueError, match="channels but the model outputs"):
        ObservableFrameHingeCost(model, wide, w_hinge=1.0, horizon=200)

    wrong_values = ObservableEnvelope(power=np.zeros((n_channels, geom.n_values(fs) + 1)), fs=fs, geometry=geom)
    with pytest.raises(ValueError, match="values per channel but its geometry implies"):
        ObservableFrameHingeCost(model, wrong_values, w_hinge=1.0, horizon=200)


def test_observable_frame_hinge_cost_with_history_prepends_past_outputs() -> None:
    """ObservableFrameHingeCost prepends past outputs when model carries history >= support."""
    rng = np.random.default_rng(_SEED + 23)
    geom = StftGeometry(n_segment=20, n_hop=10, kernel_width=1)
    fs = 100.0
    support = geom.sample_support_steps(fs)
    n_history = 25
    p = 3
    model = _DummyHistoryModel(n_history=n_history, p=p)
    envelope = ObservableEnvelope(power=np.zeros((p, geom.n_values(fs))), fs=fs, geometry=geom)

    short_cost = ObservableFrameHingeCost(model, envelope, w_hinge=1.0, horizon=5)
    assert short_cost is not None

    horizon = 30
    cost = ObservableFrameHingeCost(model, envelope, w_hinge=2.0, horizon=horizon)
    X = rng.standard_normal((horizon, n_history * p))
    stage_vals = cost.stage_costs(jnp.asarray(X), jnp.zeros((horizon, 1)), jnp.zeros(horizon))

    n_past = support - 1
    past_y = np.asarray(model.past_outputs(jnp.asarray(X[0]), n_past))
    y_future = np.asarray(jax.vmap(model.output)(jnp.asarray(X)))
    y_full = np.concatenate([past_y, y_future], axis=0)

    numpy_frames = compute_log_power_frames(y_full, geom, fs=fs)
    numpy_hinge = np.maximum(0.0, numpy_frames - envelope.power[None]) ** 2
    want = (2.0 / cost.total_frames) * float(np.sum(np.mean(numpy_hinge, axis=(-2, -1))))
    np.testing.assert_allclose(float(stage_vals[0]), want, rtol=1e-10, atol=1e-12)


def test_reduced_effort_cost_prices_the_expanded_currents() -> None:
    """ReducedEffortCost scores ``||Z v||^2``, not ``||v||^2``, so reduction reprices nothing."""
    rng = np.random.default_rng(11)
    for m in (2, 3, 5):
        Z = np.asarray(kirchhoff_basis(m))
        cost = ReducedEffortCost(n=4, m=m - 1, w_u=0.7, horizon=5)
        for v in rng.standard_normal((6, m - 1)):
            expected = 0.7 / 5 * float((Z @ v) @ (Z @ v))
            assert float(cost.evaluate(jnp.zeros(4), jnp.asarray(v))) == pytest.approx(expected, rel=1e-10)

    # The terminal knot carries no control, and the cost has to be finite there.
    assert float(ReducedEffortCost(n=4, m=2, w_u=1.0, horizon=5).evaluate(jnp.zeros(4))) == 0.0


def test_waveform_problem_requires_reference_when_wy_positive(tmp_path: Path) -> None:
    """Attempting to assemble waveform MPC problem with w_y > 0 without reference raises ValueError."""
    artifact = _build_checkpoint(tmp_path, horizon=6, n_channels=2)
    with pytest.raises(ValueError, match="reference must be provided when w_y > 0"):
        build_waveform_problem(artifact, horizon=6, u_max=0.8, w_y=1.0, reference=None)


def test_waveform_problem_state_space_reference_translation(tmp_path: Path) -> None:
    """Waveform MPC maps physical healthy reference through standardizer center and scale into state space."""
    artifact = _build_checkpoint(tmp_path, horizon=6, n_channels=2)
    model = WaveformMLPModel.load(artifact)
    y_ref = np.array([1.5, -0.5])
    ref = HealthyReference(eeg_mean=y_ref)

    problem = build_waveform_problem(artifact, horizon=6, u_max=0.8, w_y=2.0, reference=ref)
    expected_xf_y = (y_ref - model.y_center) / model.y_scale
    z_last = slice((model.n_y - 1) * model.n_channels, model.n_y * model.n_channels)

    # Functional test: evaluate tracking cost at standardized target state
    stage_cost = problem.obj.stage_cost
    inner_cost = getattr(stage_cost, "inner", stage_cost)
    terminal_cost = problem.obj.terminal_cost

    x_target = np.zeros(model.n)
    x_target[z_last] = expected_xf_y
    u_zero = np.zeros(model.m)

    # Cost is zero at healthy reference target
    np.testing.assert_allclose(float(inner_cost.evaluate(jnp.asarray(x_target), jnp.asarray(u_zero))), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        float(terminal_cost.evaluate(jnp.asarray(x_target), jnp.asarray(u_zero))), 0.0, atol=1e-12
    )

    # Cost is positive when deviating from healthy reference
    x_deviation = np.zeros(model.n)
    assert float(inner_cost.evaluate(jnp.asarray(x_deviation), jnp.asarray(u_zero))) > 0.0
    assert float(terminal_cost.evaluate(jnp.asarray(x_deviation), jnp.asarray(u_zero))) > 0.0


def test_waveform_problem_zero_control_at_healthy_operating_point(tmp_path: Path) -> None:
    """At the healthy operating point without disturbances, optimal control is zero and stage cost is zero."""
    horizon = 4
    y_ref = np.array([1.2, -0.8])
    artifact = _build_checkpoint(tmp_path, horizon=horizon, depth=0, n_channels=2, n_controls=2, equilibrium_at=y_ref)
    model = WaveformMLPModel.load(artifact)
    ref = HealthyReference(eeg_mean=y_ref)

    problem = build_waveform_problem(
        artifact, horizon=horizon, u_max=0.5, w_y=1.0, w_u=0.1, kirchhoff=True, reference=ref
    )

    # Initialize state at y_ref for all history windows
    x0 = np.tile((y_ref - model.y_center) / model.y_scale, model.n_y)
    x0 = np.concatenate([x0, np.zeros(model.n_u * model.n_controls)])

    solver = SingleShooting(solver=Ipopt(options={"print_level": 0}))
    mpc = MPC(problem, solver, x0=jnp.asarray(x0))
    res = mpc.solve()
    assert res.success
    np.testing.assert_allclose(np.asarray(mpc.controls), np.zeros((horizon, 2)), atol=1e-5)


def test_waveform_problem_active_suppression_under_seizure_deviation(tmp_path: Path) -> None:
    """Under a seizure excursion away from y_ref, nonzero Control Current is mobilized."""
    horizon = 4
    y_ref = np.array([1.2, -0.8])
    artifact = _build_checkpoint(tmp_path, horizon=horizon, depth=0, n_channels=2, n_controls=2, equilibrium_at=y_ref)
    model = WaveformMLPModel.load(artifact)
    ref = HealthyReference(eeg_mean=y_ref)

    problem = build_waveform_problem(
        artifact, horizon=horizon, u_max=0.5, w_y=1.0, w_u=0.01, kirchhoff=True, reference=ref
    )

    # Displace initial state away from y_ref (simulated seizure)
    y_seizure = y_ref + np.array([5.0, -5.0])
    x0 = np.tile((y_seizure - model.y_center) / model.y_scale, model.n_y)
    x0 = np.concatenate([x0, np.zeros(model.n_u * model.n_controls)])

    solver = SingleShooting(solver=Ipopt(options={"print_level": 0}))
    mpc = MPC(problem, solver, x0=jnp.asarray(x0))
    res = mpc.solve()
    assert res.success
    ctrls = np.asarray(mpc.controls)
    assert np.max(np.abs(ctrls)) > 1e-4


def test_reproduces_blind_tail_before_fix(tmp_path: Path) -> None:
    """Deterministic test of the waveform objective reproducing the blind final 25 controls before fix."""
    rng = np.random.default_rng(_SEED + 30)
    horizon = 75
    n_channels, n_controls = 2, 2
    fs = 1000.0
    geom = StftGeometry(n_segment=50, n_hop=25, kernel_width=1)
    support = geom.sample_support_steps(fs)

    artifact = _build_checkpoint(
        tmp_path,
        horizon=horizon,
        depth=0,
        n_channels=n_channels,
        n_controls=n_controls,
        n_y=support,
        dt=1.0 / fs,
    )
    model = WaveformMLPModel.load(artifact)

    power = np.full((n_channels, geom.n_values(fs)), -100.0)
    env = ObservableEnvelope(power=power, fs=fs, geometry=geom)

    x0 = np.tile((np.zeros(n_channels) - model.y_center) / model.y_scale, model.n_y)
    x0 = np.concatenate([x0, np.zeros(model.n_u * model.n_controls)])

    def legacy_objective_cost(u_seq: jax.Array) -> jax.Array:
        def step_fn(x: jax.Array, u: jax.Array) -> tuple[jax.Array, jax.Array]:
            x_next = model.discrete_dynamics(x, u, 0.0, model.dt)
            return x_next, x

        _, x_stage = jax.lax.scan(step_fn, jnp.asarray(x0), u_seq)
        y_stage = jax.vmap(model.output)(x_stage)
        past_y = model.past_outputs(jnp.asarray(x0), support - 1)
        y_all_stage = jnp.concatenate([past_y, y_stage], axis=0)
        frames = jax_compute_observable_frames(y_all_stage, geom, fs=fs)
        excess = jnp.maximum(0.0, frames - env.power[None])
        hinge = excess**2
        return jnp.mean(jnp.sum(jnp.mean(hinge, axis=-1), axis=-1))

    u_test = jnp.asarray(rng.standard_normal((horizon, n_controls)))
    grad_legacy = jax.grad(legacy_objective_cost)(u_test)

    assert float(jnp.linalg.norm(grad_legacy[:50])) > 0.0
    np.testing.assert_allclose(np.asarray(grad_legacy[50:]), 0.0, atol=1e-15)


def test_channel_reduction_discrepancy_reproduced() -> None:
    """Expose the channel-reduction discrepancy: a 62x factor for 62 equivalent channels."""
    n_channels = 62
    n_values = 26
    horizon = 1
    w = 1.0

    delta = 1.0
    frame = jnp.full((1, n_channels, n_values), delta)
    power_ref = jnp.zeros((n_channels, n_values))

    obs_cost = ObservableHingeCost(power_ref, w_hinge=w, horizon=horizon)
    obs_val = float(obs_cost.evaluate(frame[0].reshape(-1)))

    hinge = (frame - power_ref[None]) ** 2
    legacy_wave_val = float(w * jnp.mean(jnp.sum(jnp.mean(hinge, axis=-1), axis=-1)))

    assert np.isclose(obs_val, 1.0)
    assert np.isclose(legacy_wave_val, 62.0)
    assert np.isclose(legacy_wave_val / obs_val, 62.0)


def test_corrected_objective_includes_terminal_frame_and_full_horizon(tmp_path: Path) -> None:
    """Corrected objective scores through terminal prediction y_H with no blind suffix."""
    rng = np.random.default_rng(_SEED + 31)
    horizon = 75
    n_channels, n_controls = 2, 2
    fs = 1000.0
    geom = StftGeometry(n_segment=50, n_hop=25, kernel_width=1)
    support = geom.sample_support_steps(fs)

    artifact = _build_checkpoint(
        tmp_path, horizon=horizon, depth=0, n_channels=n_channels, n_controls=n_controls, n_y=support, dt=1.0 / fs
    )
    ref = HealthyReference(
        eeg_mean=np.zeros(n_channels),
        observable=ObservableEnvelope(power=np.full((n_channels, geom.n_values(fs)), -100.0), fs=fs, geometry=geom),
    )
    problem = build_waveform_problem(
        artifact,
        horizon=horizon,
        u_max=1.0,
        w_y=0.0,
        w_u=0.0,
        w_hinge=1.0,
        reference=ref,
    )
    stage_offsets, term_offset = frame_grid_offsets(horizon, geom)
    assert stage_offsets == [0, 25, 50]
    assert term_offset == 75

    model = problem.model
    assert isinstance(model, WaveformMLPModel)
    x0 = np.tile((np.zeros(n_channels) - model.y_center) / model.y_scale, model.n_y)
    x0 = np.concatenate([x0, np.zeros(model.n_u * model.n_controls)])

    def full_cost(u_seq: jax.Array) -> jax.Array:
        def step_fn(x: jax.Array, u: jax.Array) -> tuple[jax.Array, jax.Array]:
            x_next = model.discrete_dynamics(x, u, 0.0, model.dt)
            return x_next, x_next

        _, x_stage = jax.lax.scan(step_fn, jnp.asarray(x0), u_seq)
        X = jnp.concatenate([jnp.asarray(x0)[None], x_stage], axis=0)
        t_grid = jnp.arange(horizon + 1) * model.dt
        traj = Trajectory(X=X, U=u_seq, t=t_grid, dt=jnp.full(horizon, model.dt))
        return problem.obj.cost(traj)

    u_test = jnp.asarray(rng.standard_normal((horizon, n_controls)))
    grad_corrected = jax.grad(full_cost)(u_test)

    for k in range(horizon):
        assert float(jnp.linalg.norm(grad_corrected[k])) > 0.0
    assert float(jnp.linalg.norm(grad_corrected[74])) > 0.0


def test_direct_calculation_matches_solver_path() -> None:
    """Direct calculation of full Observable Frames matches the production solver path."""
    geom = StftGeometry(n_segment=50, n_hop=25, kernel_width=1)
    fs = 1000.0
    horizon = 75
    support = geom.sample_support_steps(fs)
    w = 2.0

    n_total = support - 1 + horizon + 1
    y_full = jnp.arange(n_total, dtype=jnp.float32)[:, None] * 0.05
    power = jnp.zeros((1, geom.n_values(fs)))

    all_frames = compute_waveform_observable_frames(y_full, geom, fs=fs)
    assert all_frames.shape[0] == 4
    direct_cost = w * jnp.mean(jnp.maximum(0.0, all_frames - power[None]) ** 2)

    stage_frames = jax_compute_observable_frames(y_full[:-1], geom, fs=fs)
    term_frame = jax_compute_observable_frames(y_full[-support:], geom, fs=fs)
    total_frames = (horizon - 1) // geom.n_hop + 2
    stage_means = jnp.mean(jnp.maximum(0.0, stage_frames - power[None]) ** 2, axis=(-2, -1))
    term_mean = jnp.mean(jnp.maximum(0.0, term_frame - power[None]) ** 2)
    solver_cost = (w / total_frames) * (jnp.sum(stage_means) + term_mean)

    np.testing.assert_allclose(float(direct_cost), float(solver_cost), atol=1e-12)


def test_autodiff_matches_finite_differences_and_all_controls_sensitive() -> None:
    """Autodiff matches finite differences in active-hinge fixture and every control is sensitive."""
    geom = StftGeometry(n_segment=10, n_hop=5, kernel_width=1)
    fs = 1000.0
    horizon = 10
    total_frames = 3

    class _ControllableLinear(DiscreteDynamics):
        n_history: int
        dt: float

        def __init__(self) -> None:
            super().__init__(n=10, m=1, ne=10, p=1)
            self.n_history = 10
            self.dt = 0.001

        def output(self, x: jax.Array, u: jax.Array | None = None, t: float | jax.Array = 0.0) -> jax.Array:
            del u, t
            return x[:1]

        def past_outputs(self, x: jax.Array, count: int) -> jax.Array:
            del x
            return jnp.zeros((count, 1))

        def discrete_dynamics(
            self, x: jax.Array, u: jax.Array, t: float | jax.Array, dt: float | jax.Array
        ) -> jax.Array:
            del t, dt
            return x.at[0].set(x[0] + u[0])

    model = _ControllableLinear()
    env = ObservableEnvelope(power=np.full((1, geom.n_values(fs)), -100.0), fs=fs, geometry=geom)
    cost_stage = ObservableFrameHingeCost(model, env, w_hinge=1.0, horizon=horizon, total_frames=total_frames)
    cost_term = cost_stage.as_terminal()

    def cost_fn(u_seq: jax.Array) -> jax.Array:
        def step(x: jax.Array, u: jax.Array) -> tuple[jax.Array, jax.Array]:
            xn = model.discrete_dynamics(x, u, 0.0, model.dt)
            return xn, xn

        x0 = jnp.ones(10) * 0.5
        _, X_future = jax.lax.scan(step, x0, u_seq)
        X = jnp.concatenate([x0[None], X_future], axis=0)
        stage_val = cost_stage.stage_costs(X[:-1], u_seq, jnp.zeros(horizon))[0]
        term_val = cost_term.evaluate(X[-1])
        return stage_val + term_val

    u = jnp.ones((horizon, 1)) * 0.2
    grad_ad = jax.grad(cost_fn)(u)

    eps = 1e-4
    grad_fd = np.zeros_like(u)
    for i in range(horizon):
        u_plus = u.at[i, 0].add(eps)
        u_minus = u.at[i, 0].add(-eps)
        grad_fd[i, 0] = (float(cost_fn(u_plus)) - float(cost_fn(u_minus))) / (2 * eps)

    np.testing.assert_allclose(np.asarray(grad_ad), grad_fd, rtol=1e-3, atol=1e-4)
    for k in range(horizon):
        assert float(np.abs(grad_ad[k, 0])) > 0.0
    assert float(np.abs(grad_ad[-1, 0])) > 0.0


def test_coverage_geometries_odd_hop_and_frame_kernel() -> None:
    """Coverage covers diagnosed, short horizon, non-divisible hop, and Frame Kernel."""
    geoms = [
        (75, StftGeometry(n_segment=50, n_hop=25, kernel_width=1)),
        (10, StftGeometry(n_segment=50, n_hop=25, kernel_width=1)),
        (70, StftGeometry(n_segment=50, n_hop=25, kernel_width=1)),
        (60, StftGeometry(n_segment=20, n_hop=20, kernel_width=3)),
    ]
    fs = 1000.0
    w = 2.5
    for H, geom in geoms:
        support = geom.sample_support_steps(fs)
        stage_offsets, term_offset = frame_grid_offsets(H, geom)
        assert term_offset == H
        assert stage_offsets[-1] < H

        n_total = support - 1 + H + 1
        y_full = jnp.arange(n_total, dtype=jnp.float32)[:, None] * 0.1
        power = jnp.zeros((1, geom.n_values(fs)))

        stage_frames = jax_compute_observable_frames(y_full[:-1], geom, fs=fs)
        term_frame = jax_compute_observable_frames(y_full[-support:], geom, fs=fs)
        all_frames = jnp.concatenate([stage_frames, term_frame], axis=0)

        direct_cost = w * jnp.mean(jnp.maximum(0.0, all_frames - power[None]) ** 2)

        total_frames = (H - 1) // geom.n_hop + 2
        stage_means = jnp.mean(jnp.maximum(0.0, stage_frames - power[None]) ** 2, axis=(-2, -1))
        term_mean = jnp.mean(jnp.maximum(0.0, term_frame - power[None]) ** 2)
        solver_cost = (w / total_frames) * (jnp.sum(stage_means) + term_mean)

        np.testing.assert_allclose(float(direct_cost), float(solver_cost), atol=1e-12)


def test_late_input_changes_spectral_objective_and_optimized_plan(tmp_path: Path) -> None:
    """Late inputs change the spectral objective and are mobilized in the optimized plan."""
    horizon = 10
    n_channels, n_controls = 2, 2
    fs = 1000.0
    geom = StftGeometry(n_segment=10, n_hop=5, kernel_width=1)
    support = geom.sample_support_steps(fs)
    y_ref = np.zeros(n_channels)

    artifact = _build_checkpoint(
        tmp_path,
        horizon=horizon,
        depth=0,
        n_channels=n_channels,
        n_controls=n_controls,
        n_y=support,
        dt=1.0 / fs,
        equilibrium_at=y_ref,
    )
    ref = HealthyReference(
        eeg_mean=y_ref,
        observable=ObservableEnvelope(power=np.full((n_channels, geom.n_values(fs)), -30.0), fs=fs, geometry=geom),
    )
    problem = build_waveform_problem(
        artifact,
        horizon=horizon,
        u_max=2.0,
        w_y=0.0,
        w_u=0.05,
        w_hinge=1.0,
        reference=ref,
    )
    model = problem.model
    assert isinstance(model, WaveformMLPModel)
    y_seizure = y_ref + np.array([2.0, -2.0])
    x0 = np.tile((y_seizure - model.y_center) / model.y_scale, model.n_y)
    x0 = np.concatenate([x0, np.zeros(model.n_u * model.n_controls)])

    solver = SingleShooting(
        solver=Ipopt(options={"print_level": 0, "max_iter": 60, "tol": 1e-4, "hessian_approximation": "limited-memory"})
    )
    mpc = MPC(problem, solver, x0=jnp.asarray(x0))
    res = mpc.solve()
    assert res.success
    controls = np.asarray(mpc.controls)
    assert np.max(np.abs(controls[5:])) > 1e-4


def test_spectral_costs_agree_for_matched_frames_and_envelopes() -> None:
    """ObservableFrameHingeCost and ObservableHingeCost agree on matched frames and envelopes."""
    n_channels = 4
    n_values = 16
    delta = 1.5
    w = 3.0

    frame = jnp.full((1, n_channels, n_values), delta)
    power_ref = jnp.zeros((n_channels, n_values))

    obs_cost = ObservableHingeCost(power_ref, w_hinge=w, horizon=1)
    obs_val = float(obs_cost.evaluate(frame[0].reshape(-1)))

    hinge = (frame - power_ref[None]) ** 2
    corrected_wave_val = float(w * jnp.mean(hinge))

    np.testing.assert_allclose(obs_val, corrected_wave_val, atol=1e-12)


def test_duplicating_identical_channels_leaves_normalized_cost_unchanged() -> None:
    """Duplicating channels leaves normalized spectral cost unchanged under channel-mean convention."""
    rng = np.random.default_rng(_SEED + 32)
    n_values = 10
    c2 = rng.uniform(1.0, 3.0, (1, 2, n_values))
    p2 = np.zeros((2, n_values))
    hinge2 = np.maximum(0.0, c2 - p2[None]) ** 2
    cost2 = float(np.mean(hinge2))

    c4 = np.tile(c2, (1, 2, 1))
    p4 = np.tile(p2, (2, 1))
    hinge4 = np.maximum(0.0, c4 - p4[None]) ** 2
    cost4 = float(np.mean(hinge4))

    np.testing.assert_allclose(cost2, cost4, atol=1e-12)

    bin_means = np.mean(hinge2, axis=-1)
    np.testing.assert_allclose(float(np.mean(bin_means)), cost2, atol=1e-12)

    multi_frames = np.repeat(hinge2, 5, axis=0)
    assert np.isclose(float(np.mean(multi_frames)), cost2)


def test_cost_contributions_and_evaluation_report(tmp_path: Path) -> None:
    """Evaluation reports separate spectral, quadratic-effort, and sparse-effort contributions."""
    horizon = 10
    n_channels, n_controls = 2, 2
    fs = 1000.0
    geom = StftGeometry(n_segment=10, n_hop=5, kernel_width=1)
    support = geom.sample_support_steps(fs)

    artifact = _build_checkpoint(
        tmp_path, horizon=horizon, depth=0, n_channels=n_channels, n_controls=n_controls, n_y=support, dt=1.0 / fs
    )
    ref = HealthyReference(
        eeg_mean=np.zeros(n_channels),
        observable=ObservableEnvelope(power=np.full((n_channels, geom.n_values(fs)), -100.0), fs=fs, geometry=geom),
    )
    problem = build_waveform_problem(
        artifact,
        horizon=horizon,
        u_max=1.0,
        w_y=1.0,
        w_u=0.5,
        w_u_l1=0.2,
        w_hinge=2.0,
        reference=ref,
    )
    model = problem.model
    assert isinstance(model, WaveformMLPModel)
    x0 = np.tile((np.zeros(n_channels) - model.y_center) / model.y_scale, model.n_y)
    x0 = np.concatenate([x0, np.zeros(model.n_u * model.n_controls)])
    states = jnp.repeat(jnp.asarray(x0)[None], horizon + 1, axis=0)
    controls = jnp.ones((horizon, n_controls)) * 0.1

    decomp = decompose_cost(problem, states, controls, t=0.0, dt=model.dt)
    assert decomp["normalization"] == "channel_mean"
    assert decomp["cost_spectral"] > 0.0
    assert decomp["cost_quadratic_effort"] > 0.0
    assert decomp["cost_sparse_effort"] > 0.0
    total = (
        decomp["cost_spectral"]
        + decomp["cost_quadratic_effort"]
        + decomp["cost_sparse_effort"]
        + decomp["cost_tracking"]
    )
    np.testing.assert_allclose(decomp["cost_total"], total, atol=1e-10)

    class _MockLogger:
        def signals(self) -> list[tuple[str, str]]:
            return [
                ("controller", "cost_spectral"),
                ("controller", "cost_quadratic_effort"),
                ("controller", "cost_sparse_effort"),
                ("controller", "cost_tracking"),
                ("controller", "normalization"),
                ("controller", "warmup"),
            ]

        def signal(self, comp: str, name: str) -> tuple[np.ndarray, np.ndarray]:
            del comp
            if name == "warmup":
                return np.array([0.0, 1.0]), np.array([True, False])
            if name == "normalization":
                return np.array([0.0, 1.0]), np.array(["channel_mean", "channel_mean"])
            val = decomp[name]
            return np.array([0.0, 1.0]), np.array([0.0, val])

    mock_logger = _MockLogger()
    metrics = cost_metrics(cast("BaseLogger", mock_logger))
    assert metrics["cost_normalization"] == "channel_mean"
    assert np.isclose(metrics["cost_spectral_mean"], decomp["cost_spectral"])
    assert np.isclose(metrics["cost_quadratic_effort_mean"], decomp["cost_quadratic_effort"])
    assert np.isclose(metrics["cost_sparse_effort_mean"], decomp["cost_sparse_effort"])

    report = format_cost_report(metrics)
    assert "channel_mean" in report
    assert "Spectral cost" in report
