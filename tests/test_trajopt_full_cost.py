"""Ticket 02: full cost and constraint parity -- hinge/L1 costs, Kirchhoff, controller parity."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import casadi as ca
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from trajopt.constraints.linear import LinearConstraint
from trajopt.problem import MPCState
from trajopt.solvers.altro import ALTRO
from trajopt.transcription.ipopt import Ipopt

from neuro.config import StftGeometry
from neuro.control.nlp import _observable_hinge_cost, _spectral_hinge_cost
from neuro.control.nonlinear_mpc import MPCController
from neuro.control.trajopt_costs import L1ControlCost, ObservableHingeCost, SpectralHingeCost, SumCost
from neuro.control.trajopt_mpc import (
    TrajOptMPCController,
    WaveformMLPModel,
    build_waveform_problem,
    kirchhoff_constraint,
)
from neuro.nn_predictor_casadi import NNSymbolicModel
from neuro.observable import envelope_log_reference
from neuro.predictor.module import AutoregressiveMLP
from neuro.spectral import PsdEnvelope, compute_periodograms
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
    probe = WaveformMLPModel.from_checkpoint(artifact)
    state = np.asarray(probe.initial_state())
    state[: probe.n_y * probe.n_channels] = rng.standard_normal(probe.n_y * probe.n_channels)
    return state


def _biting_envelope(tmp_path: Path, artifact: Path, horizon: int, *, length: int, hop: int, fs: float) -> Path:
    """Write a healthy-envelope npz whose power the model's zero-control rollout exceeds.

    The envelope is the per-``(channel, bin)`` median of the rollout's own periodograms, so
    roughly half of every cell sits over it and the hinge actually bites.
    """
    probe = WaveformMLPModel.from_checkpoint(artifact)
    rng = np.random.default_rng(_SEED + 1)
    x0 = np.asarray(probe.initial_state())
    x0[: probe.n_y * probe.n_channels] = rng.standard_normal(probe.n_y * probe.n_channels)
    x = jnp.asarray(x0)
    y_traj = []
    for _ in range(horizon):
        x = probe.discrete_dynamics(x, jnp.zeros(probe.m), 0.0, 0.01)
        z = np.asarray(x)[(probe.n_y - 1) * probe.n_channels : probe.n_y * probe.n_channels]
        y_traj.append(z * np.asarray(probe.y_scale) + np.asarray(probe.y_center))
    windows = compute_periodograms(np.array(y_traj), fs=fs, window=length, hop=hop)
    path = tmp_path / "healthy.npz"
    np.savez(
        path,
        Pref=np.median(windows, axis=0),
        freqs=np.fft.rfftfreq(length, 1.0 / fs),
        fs=fs,
        L=length,
        R=hop,
        quantile=0.9,
        n_windows=10,
        plant_fingerprint="test",
    )
    return path


def _roll_trajectory(artifact: Path, x0: FloatArray, u_seq: FloatArray) -> tuple[list[FloatArray], FloatArray]:
    """Roll the adapter under ``u_seq``; returns the predicted raw outputs and the state trajectory."""
    probe = WaveformMLPModel.from_checkpoint(artifact)
    states = [np.asarray(x0)]
    y_nodes = []
    x = jnp.asarray(x0)
    for u in u_seq:
        x = probe.discrete_dynamics(x, jnp.asarray(u), 0.0, 0.01)
        states.append(np.asarray(x))
        z = np.asarray(x)[(probe.n_y - 1) * probe.n_channels : probe.n_y * probe.n_channels]
        y_nodes.append(z * np.asarray(probe.y_scale) + np.asarray(probe.y_center))
    return y_nodes, np.array(states)


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


def test_spectral_hinge_matches_casadi_graph(tmp_path: Path) -> None:
    """The trajopt spectral hinge totals the CasADi graph's value on a fixed trajectory."""
    n_y, n_u, n_channels, n_controls = 4, 3, 2, 2
    horizon, length, hop, fs = 6, 4, 2, 50.0
    artifact = _build_checkpoint(
        tmp_path, n_y=n_y, n_u=n_u, horizon=horizon, n_channels=n_channels, n_controls=n_controls
    )
    probe = WaveformMLPModel.from_checkpoint(artifact)
    env_path = _biting_envelope(tmp_path, artifact, horizon, length=length, hop=hop, fs=fs)
    envelope = PsdEnvelope.load(env_path)

    rng = np.random.default_rng(_SEED + 2)
    x0 = _ready_state(artifact, rng)
    u_seq = rng.uniform(-0.5, 0.5, (horizon, n_controls))
    y_nodes, states = _roll_trajectory(artifact, x0, u_seq)

    casadi_value = float(ca.evalf(_spectral_hinge_cost([ca.MX(y.reshape(-1, 1)) for y in y_nodes], envelope, horizon)))

    cost = SpectralHingeCost(model=probe, envelope=envelope, w_psd=1.0, horizon=horizon)
    stage = cost.stage_costs(jnp.asarray(states[:-1]), jnp.asarray(u_seq), jnp.zeros(horizon))
    assert casadi_value > 0.0, "the envelope must actually bite for the parity check to mean anything"
    np.testing.assert_allclose(float(jnp.sum(stage)), casadi_value, rtol=1e-10, atol=1e-12)


def test_observable_hinge_matches_casadi_graph() -> None:
    """The trajopt observable hinge totals the CasADi graph's value on fixed forecast frames.

    The reference is reduced from the healthy envelope through the ``ObservableGeometry`` -- the
    same shared source of truth the training-time Loss scores against.
    """
    rng = np.random.default_rng(_SEED + 3)
    n_channels, n_values, n_frames = 3, 4, 5
    geometry = StftGeometry(n_segment=8, n_hop=4)
    envelope = PsdEnvelope(power=np.full((n_channels, 5), 1.5), fs=50.0, window=8, hop=4)
    log_reference = envelope_log_reference(envelope, geometry, 50.0)
    assert log_reference.shape == (n_channels, n_values)

    l_hat = rng.standard_normal((n_channels * n_values, n_frames)) + 1.0
    casadi_value = float(ca.evalf(_observable_hinge_cost(ca.MX(l_hat), log_reference)))
    assert casadi_value > 0.0, "the reference must actually bite for the parity check to mean anything"

    cost = ObservableHingeCost(n=0, m=0, log_reference=log_reference, w_psd=1.0, n_frames=n_frames)
    stage = cost.stage_costs(jnp.asarray(l_hat.T), jnp.zeros((n_frames, 1)), jnp.zeros(n_frames))
    np.testing.assert_allclose(float(jnp.sum(stage)), casadi_value, rtol=1e-12, atol=1e-12)


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


def test_full_cost_controller_reproduces_incumbent(tmp_path: Path) -> None:
    """The controller reproduces the incumbent's control sequence with the full cost and constraint set.

    Quadratic (power + effort) plus the spectral hinge and the smooth L1 surrogate, against the
    box bounds and the Kirchhoff equality -- the incumbent's epigraph L1 and hand-rolled DFT
    hinge replaced by the custom ``CostFunction``s, both sides solved by Ipopt on the same
    scripted trajectory. A depth-0 (linear) checkpoint keeps the problem well-behaved so both
    land on the same optimum.
    """
    n_y, n_u, n_channels, n_controls = 4, 3, 2, 2
    horizon, length, hop, fs = 8, 4, 2, 50.0
    artifact = _build_checkpoint(
        tmp_path, n_y=n_y, n_u=n_u, horizon=horizon, n_channels=n_channels, n_controls=n_controls, depth=0
    )
    env_path = _biting_envelope(tmp_path, artifact, horizon, length=length, hop=hop, fs=fs)

    w_y, w_u, w_l1, w_psd, u_max = 1.0, 0.05, 0.5, 50.0, 0.8

    incumbent = MPCController(
        dt=0.01,
        model=NNSymbolicModel.from_checkpoint(artifact),
        u_max=u_max,
        horizon=horizon,
        w_y=w_y,
        w_u=w_u,
        w_u_l1=w_l1,
        w_psd=w_psd,
        psd_ref=str(env_path),
        solver="ipopt",
    )
    problem = build_waveform_problem(
        artifact,
        horizon=horizon,
        u_max=u_max,
        w_y=w_y,
        w_u=w_u,
        w_u_l1=w_l1,
        w_psd=w_psd,
        psd_ref=str(env_path),
        kirchhoff=True,
    )
    controller = TrajOptMPCController(dt=0.01, problem=problem, solver=_full_parity_solver())

    rng = np.random.default_rng(_SEED + 5)
    want = []
    got = []
    for k in range(6):
        measurement = rng.standard_normal(n_channels)
        u_inc, _ = incumbent.update(k * 0.01, ref=np.array([0.0]), x_hat=measurement)
        u_new, log_new = controller.update(k * 0.01, ref=np.array([0.0]), x_hat=measurement)
        if not log_new.warmup:
            assert log_new.success, "the trajopt solve must converge"
        want.append(np.atleast_1d(np.asarray(u_inc, dtype=np.float64)))
        got.append(np.atleast_1d(np.asarray(u_new, dtype=np.float64)))

    np.testing.assert_allclose(np.array(got[3:]), np.array(want[3:]), atol=1e-4)
    # The Kirchhoff equality holds on the emitted controls.
    np.testing.assert_allclose(np.sum(np.array(got[3:]), axis=1), np.zeros(3), atol=1e-6)


def test_l1_surrogate_matches_incumbent_epigraph(tmp_path: Path) -> None:
    """The smooth L1 surrogate reproduces the incumbent's epigraph controls on the transcription path.

    With only the quadratic and L1 terms active (no spectral hinge), the problem is a convex QP,
    so both sides converge to the same unique optimum: the data behind keeping the smooth
    surrogate for every solver rather than restricting L1 to transcription-based ones.
    """
    n_y, n_u, n_channels, n_controls = 4, 3, 2, 2
    horizon = 6
    artifact = _build_checkpoint(
        tmp_path, n_y=n_y, n_u=n_u, horizon=horizon, n_channels=n_channels, n_controls=n_controls, depth=0
    )

    w_y, w_u, w_l1, u_max = 1.0, 0.05, 0.5, 0.8

    incumbent = MPCController(
        dt=0.01,
        model=NNSymbolicModel.from_checkpoint(artifact),
        u_max=u_max,
        horizon=horizon,
        w_y=w_y,
        w_u=w_u,
        w_u_l1=w_l1,
        solver="ipopt",
    )
    problem = build_waveform_problem(
        artifact, horizon=horizon, u_max=u_max, w_y=w_y, w_u=w_u, w_u_l1=w_l1, kirchhoff=True
    )
    controller = TrajOptMPCController(dt=0.01, problem=problem, solver=_full_parity_solver())

    rng = np.random.default_rng(_SEED + 6)
    want = []
    got = []
    for k in range(6):
        measurement = rng.standard_normal(n_channels)
        u_inc, _ = incumbent.update(k * 0.01, ref=np.array([0.0]), x_hat=measurement)
        u_new, log_new = controller.update(k * 0.01, ref=np.array([0.0]), x_hat=measurement)
        assert log_new.success
        want.append(np.atleast_1d(np.asarray(u_inc, dtype=np.float64)))
        got.append(np.atleast_1d(np.asarray(u_new, dtype=np.float64)))

    np.testing.assert_allclose(np.array(got[3:]), np.array(want[3:]), atol=1e-3)


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
