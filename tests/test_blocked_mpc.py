from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
import pytest
from test_mpc import _build_checkpoint, _ref
from test_waveform_cnn import _cnn
from trajopt.program import WarmStart
from trajopt.solvers.altro import ALTRO
from trajopt.transcription.ipopt import Ipopt

from experiments.benchmark import describe_problem
from neuro.config import StftGeometry
from neuro.control.candidate import CandidateMPCController
from neuro.control.casadi import CasADiMPCController, build_casadi_waveform_problem, decompose_casadi_cost
from neuro.control.diagnostics import _rollout_states, decompose_cost, planned_outputs
from neuro.control.mpc import TrajOptMPCController
from neuro.control.problems import NeuroProblem, build_waveform_problem
from neuro.spectral import HealthyReference, ObservableEnvelope

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("kind", ["mlp", "cnn"])
def test_blocked_native_plan_and_backend_parity(tmp_path: Path, kind: str) -> None:
    artifact = _build_checkpoint(tmp_path, horizon=12, depth=0)
    if kind == "cnn":
        artifact = tmp_path / "cnn"
        _cnn(depth=1, activation="tanh").save(artifact)
    kwargs: dict[str, Any] = {
        "horizon": 12,
        "control_block_steps": 6,
        "u_max": 0.5,
        "w_y": 0.4,
        "w_u": 2.0,
        "reference": _ref(3 if kind == "cnn" else 2),
        "kirchhoff": True,
    }
    problem = build_waveform_problem(artifact, **kwargs)
    trajopt = TrajOptMPCController(0.01, problem, replan_dt=0.06)
    casadi = CasADiMPCController(0.01, build_casadi_waveform_problem(artifact, **kwargs), replan_dt=0.06)
    state = np.random.default_rng(17).normal(size=problem.model.n) * 0.05
    controls = np.repeat([[0.1, -0.1], [-0.2, 0.2]], 6, axis=0)
    states = _rollout_states(problem.model, jnp.asarray(state), jnp.asarray(controls), 0.01)
    expected_y = planned_outputs(trajopt.model, states, jnp.asarray(controls), 0.0, 0.01)
    np.testing.assert_allclose(
        np.asarray(casadi._predicted_y_fn(state, controls.reshape(-1))).T,  # noqa: SLF001 -- verify retained native evaluator
        expected_y,
        atol=1e-8,
    )
    costs = decompose_cost(problem, states, jnp.asarray(controls), 0.0, 0.01)
    assert casadi.decompose_cost(state, controls)["cost_total"] == pytest.approx(costs["cost_total"], rel=1e-7)
    assert casadi._solver.size_in("x0") == (4, 1)  # noqa: SLF001 -- reduced NLP dimension is the contract
    assert casadi._solver.size_in("lbg") == (2, 1)  # noqa: SLF001 -- no redundant Kirchhoff equalities
    assert "SS 4 vars" in describe_problem(problem)
    for controller in (trajopt, casadi):
        log = controller.solve_state(state, initial_u=controls)
        assert log.success, log.status
        assert log.planned_u.shape == (12, 2)
        assert log.predicted_y.shape == (13, 3 if kind == "cnn" else 2)
        np.testing.assert_allclose(
            log.planned_u.reshape(2, 6, 2), np.repeat(log.planned_u[::6, None, :], 6, axis=1), atol=1e-10
        )
        np.testing.assert_allclose(log.planned_u.sum(axis=1), 0.0, atol=1e-7)
        assert np.max(np.abs(log.planned_u)) <= 0.500001
    if kind == "mlp":
        a, b = trajopt.solve_state(state), casadi.solve_state(state)
        np.testing.assert_allclose(a.planned_u, b.planned_u, atol=3e-3)
        assert a.cost == pytest.approx(b.cost, rel=1e-4, abs=1e-6)
    program = trajopt.mpc.program
    trajopt.mpc.measure(jnp.asarray(state))
    warm_start = trajopt.mpc.warm_start
    trajopt.mpc._ws = WarmStart(  # noqa: SLF001 -- seed stale canonical Multipliers at the driver seam
        Z=warm_start.Z, lam=jnp.ones_like(warm_start.lam), mu=jnp.ones_like(warm_start.mu)
    )
    result = trajopt.mpc.solve()
    assert result.trajectory.X.shape == (13, problem.model.n)
    np.testing.assert_array_equal(result.lam, 0.0)
    np.testing.assert_array_equal(result.mu, 0.0)
    np.testing.assert_array_equal(trajopt.mpc.warm_start.lam, 0.0)
    np.testing.assert_array_equal(trajopt.mpc.warm_start.mu, 0.0)
    handle = program.handles["single_shooting"]
    trajopt.mpc.advance(6)
    trajopt.mpc.measure(jnp.asarray(state))
    trajopt.mpc.solve()
    assert trajopt.mpc.program is program
    assert program.handles["single_shooting"] is handle


@pytest.mark.parametrize("backend", ["trajopt", "casadi"])
@pytest.mark.parametrize("block", [True, 0, 2.5, 5])
def test_invalid_block_structure(tmp_path: Path, backend: str, block: object) -> None:
    artifact = _build_checkpoint(tmp_path, horizon=12)
    builder = build_waveform_problem if backend == "trajopt" else build_casadi_waveform_problem
    with pytest.raises((TypeError, ValueError), match=r"control_block_steps|divis"):
        builder(artifact, horizon=12, control_block_steps=block, u_max=0.5, w_y=0.0)  # ty: ignore[invalid-argument-type]


def test_blocked_forwarding_epigraph_and_unsupported_paths(tmp_path: Path) -> None:
    artifact = _build_checkpoint(tmp_path, horizon=12)
    problem = build_waveform_problem(artifact, horizon=12, control_block_steps=6, u_max=0.5, w_y=0.0)
    epi = build_waveform_problem(
        artifact, horizon=12, control_block_steps=6, u_max=0.5, w_y=0.0, w_u_l1=0.1, l1_mode="epigraph"
    )
    assert epi.control_block_steps == 6
    TrajOptMPCController(0.01, epi, replan_dt=0.06)
    for solver in (ALTRO(), Ipopt()):
        with pytest.raises(ValueError, match=r"blocked|blocking"):
            TrajOptMPCController(0.01, problem, solver=solver, replan_dt=0.06)
    with pytest.raises(ValueError, match="control_block_steps"):
        TrajOptMPCController(0.01, problem, replan_dt=0.03)
    with pytest.raises(ValueError, match=r"blocked|blocking"):
        CandidateMPCController(0.01, problem)
    binary = NeuroProblem(
        model=problem.model,
        obj=problem.obj,
        N=13,
        dt=0.01,
        binary_control_indices=(0,),
        control_block_steps=6,
        horizon=12,
        u_max=0.5,
    )
    with pytest.raises(ValueError, match=r"continuous|activation"):
        TrajOptMPCController(0.01, binary, replan_dt=0.06)
    with pytest.raises(ValueError, match=r"continuous|activation"):
        CasADiMPCController(
            0.01,
            build_casadi_waveform_problem(
                artifact, horizon=12, control_block_steps=6, u_max=0.5, w_y=0.0, max_active_intervals=6
            ),
            replan_dt=0.06,
        )


@pytest.mark.parametrize("kind", ["mlp", "cnn"])
def test_blocked_native_cost_components_with_spectral_history(tmp_path: Path, kind: str) -> None:
    """Blocking preserves all native tracking, effort and spectral Cost components."""
    artifact = _build_checkpoint(tmp_path, horizon=12, depth=0)
    channels = 2
    if kind == "cnn":
        artifact = tmp_path / "cnn"
        _cnn(depth=1).save(artifact)
        channels = 3
    geometry = StftGeometry(n_segment=4, n_hop=2, kernel="hann", kernel_width=2)
    reference = HealthyReference(
        eeg_mean=np.zeros(channels),
        observable=ObservableEnvelope(
            power=np.full((channels, geometry.n_values(100.0)), -0.5), fs=100.0, geometry=geometry
        ),
    )
    kwargs: dict[str, Any] = {
        "horizon": 12,
        "control_block_steps": 6,
        "u_max": 0.5,
        "w_y": 0.4,
        "w_y_terminal": 0.7,
        "w_u": 0.2,
        "w_u_l1": 0.3,
        "w_hinge": 0.6,
        "w_hinge_terminal": 0.8,
        "reference": reference,
    }
    problem = build_waveform_problem(artifact, **kwargs)
    casadi = build_casadi_waveform_problem(artifact, **kwargs)
    state = np.random.default_rng(8).normal(size=problem.model.n)
    controls = np.repeat([[0.1, -0.1], [-0.2, 0.2]], 6, axis=0)
    states = _rollout_states(problem.model, jnp.asarray(state), jnp.asarray(controls), 0.01)
    expected = decompose_cost(problem, states, jnp.asarray(controls), dt=0.01)
    actual = decompose_casadi_cost(casadi, state, controls)
    for component in ("cost_tracking", "cost_quadratic_effort", "cost_sparse_effort", "cost_spectral", "cost_total"):
        assert actual[component] == pytest.approx(expected[component], rel=1e-7, abs=1e-8)


@pytest.mark.parametrize("backend", ["trajopt", "casadi"])
def test_blocked_configuration_forwards_structure(tmp_path: Path, backend: str) -> None:
    """Configuration dictionaries preserve blocking without a second controller block setting."""
    artifact = _build_checkpoint(tmp_path, horizon=12)
    factory = (
        "neuro.control.problems.build_waveform_problem"
        if backend == "trajopt"
        else "neuro.control.casadi.build_casadi_waveform_problem"
    )
    config: dict[str, Any] = {
        "dt": 0.01,
        "replan_dt": 0.06,
        "problem": {
            "class_path": factory,
            "artifact": str(artifact),
            "horizon": 12,
            "control_block_steps": 6,
            "u_max": 0.5,
            "w_y": 0.0,
        },
    }
    cls = TrajOptMPCController if backend == "trajopt" else CasADiMPCController
    controller = cls.from_config(config)
    assert getattr(controller.problem, "control_block_steps", 1) == 6
    with pytest.raises(ValueError, match="control_block_steps"):
        cls.from_config({**config, "replan_dt": 0.03})


def test_blocked_epigraph_solve_and_physical_history(tmp_path: Path) -> None:
    """Blocked slack decisions remain feasible while playback absorbs only physical currents."""
    artifact = _build_checkpoint(tmp_path, horizon=12, depth=0)
    problem = build_waveform_problem(
        artifact,
        horizon=12,
        control_block_steps=6,
        u_max=0.5,
        reference=_ref(),
        w_y=0.5,
        w_u=0.1,
        w_u_l1=0.2,
        l1_mode="epigraph",
        kirchhoff=True,
    )
    controller = TrajOptMPCController(0.01, problem, replan_dt=0.06, control_start_s=0.06)
    absorbed = controller.model.initial_state()
    previous = np.zeros(2)
    for k in range(13):
        measurement = np.full(2, 0.1 * k)
        absorbed = controller.model.absorb(absorbed, measurement, previous)
        command, log = controller.update(k * 0.01, np.zeros(1), measurement)
        assert command.shape == (2,)
        if log.solve_attempted:
            assert log.success, log.status
            assert log.planned_u.shape == (12, 2)
            np.testing.assert_allclose(controller.mpc.x0, absorbed)
            decisions = np.asarray(controller.mpc.controls)
            assert decisions.shape == (12, 4)
            assert np.min(decisions[:, 2:] - np.abs(decisions[:, :2])) >= -1e-5
            np.testing.assert_allclose(
                decisions.reshape(2, 6, 4), np.repeat(decisions[::6, None], 6, axis=1), atol=1e-10
            )
        previous = command
