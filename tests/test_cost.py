from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import torch
from trajopt.constraints.linear import LinearConstraint
from trajopt.problem import MPCState
from trajopt.solvers.altro import ALTRO
from trajopt.transcription.ipopt import Ipopt

from neuro.config import StftGeometry
from neuro.control.costs import L1ControlCost, SpectralHingeCost, SumCost, jax_compute_log_power_frames
from neuro.control.mpc import build_waveform_problem
from neuro.predictor.inference import WaveformMLPModel
from neuro.predictor.module import AutoregressiveMLP
from neuro.spectral import PsdEnvelope, compute_log_power_frames
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
