from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from test_mpc import _build_checkpoint, _build_observable_checkpoint
from trajopt.problem import MPCState
from trajopt.solvers.altro import ALTRO
from trajopt.solvers.boxqp import BoxQP
from trajopt.transcription.ipopt import Ipopt
from trajopt.transcription.osqp import OSQP
from trajopt.transcription.single_shooting import SingleShooting

from neuro.control.benchmark import (
    format_closed_loop_table,
    format_open_loop_table,
    get_benchmark_solver,
    run_observable_benchmark,
    run_waveform_benchmark,
)
from neuro.control.mpc import (
    BipolarReducedModel,
    build_bipolar_waveform_problem,
    build_waveform_problem,
)
from neuro.predictor.inference import InferencePredictor, WaveformMLPModel

if TYPE_CHECKING:
    from pathlib import Path


def test_get_benchmark_solver_instantiation() -> None:
    """get_benchmark_solver creates configured solver instances across all names."""
    assert type(get_benchmark_solver("single_shooting")) is SingleShooting
    assert type(get_benchmark_solver("ipopt")) is Ipopt
    assert type(get_benchmark_solver("altro")) is ALTRO
    assert type(get_benchmark_solver("boxqp")) is BoxQP
    assert type(get_benchmark_solver("osqp")) is OSQP

    with pytest.raises(ValueError, match="Unknown solver name"):
        get_benchmark_solver("invalid_solver_name")


def test_osqp_exact_linear_predictor_optimality(tmp_path: Path) -> None:
    """On a linear predictor with quadratic cost, OSQP matches IPOPT Single Shooting to high precision."""
    art = _build_checkpoint(tmp_path, depth=0, n_y=4, n_u=3, horizon=4, n_channels=2, n_controls=2)
    problem = build_waveform_problem(art, horizon=4, u_max=0.5, w_y=1.0, w_u=0.1, kirchhoff=True)
    model = problem.model
    assert isinstance(model, InferencePredictor)

    rng = np.random.default_rng(123)
    x0 = rng.standard_normal(model.n)
    state = MPCState.initial(problem, x0=jnp.asarray(x0), dt=model.dt)

    s_ipopt = SingleShooting(solver=Ipopt(options={"print_level": 0}))
    s_osqp = OSQP(options={"eps_abs": 1e-8, "eps_rel": 1e-8, "max_iter": 10000})

    res_ipopt = problem.solve(state, solver=s_ipopt)
    res_osqp = problem.solve(state, solver=s_osqp)

    assert res_ipopt.status == "converged"
    assert res_osqp.status == "converged"

    np.testing.assert_allclose(res_ipopt.controls[0], res_osqp.controls[0], atol=1e-4)
    np.testing.assert_allclose(float(problem.cost(res_ipopt)), float(problem.cost(res_osqp)), rtol=1e-4)


def test_bipolar_boxqp_waveform_parity(tmp_path: Path) -> None:
    """Bipolar reduced Box-iLQR matches Single Shooting with hard Kirchhoff on 2-electrode montage."""
    art = _build_checkpoint(tmp_path, depth=0, n_y=4, n_u=3, horizon=4, n_channels=2, n_controls=2)

    prob_full = build_waveform_problem(art, horizon=4, u_max=0.5, w_y=1.0, w_u=0.1, kirchhoff=True)
    prob_bipolar = build_bipolar_waveform_problem(art, horizon=4, u_max=0.5, w_y=1.0, w_u=0.1)

    model_full = prob_full.model
    model_bipolar = prob_bipolar.model
    assert isinstance(model_full, InferencePredictor)
    assert isinstance(model_bipolar, InferencePredictor)

    rng = np.random.default_rng(456)
    x0 = rng.standard_normal(model_full.n)
    state_full = MPCState.initial(prob_full, x0=jnp.asarray(x0), dt=model_full.dt)
    state_bipolar = MPCState.initial(prob_bipolar, x0=jnp.asarray(x0), dt=model_bipolar.dt)

    s_ipopt = SingleShooting(solver=Ipopt(options={"print_level": 0}))
    s_boxqp = BoxQP()

    res_full = prob_full.solve(state_full, solver=s_ipopt)
    res_bipolar = prob_bipolar.solve(state_bipolar, solver=s_boxqp)

    v0 = float(res_bipolar.controls[0][0])
    u_bipolar_0 = np.array([v0, -v0])

    np.testing.assert_allclose(res_full.controls[0], u_bipolar_0, atol=1e-3)
    np.testing.assert_allclose(float(prob_full.cost(res_full)), float(prob_bipolar.cost(res_bipolar)), rtol=1e-3)


def test_bipolar_reduced_model_methods(tmp_path: Path) -> None:
    """BipolarReducedModel properly wraps dynamics, absorb, free_run, and initial_state."""
    art = _build_checkpoint(tmp_path, depth=0, n_y=3, n_u=2, horizon=3, n_channels=2, n_controls=2)
    base = WaveformMLPModel.load(art)
    reduced = BipolarReducedModel(base)

    assert reduced.m == 1
    assert reduced.n == base.n
    assert reduced.n_controls == 1

    x0 = reduced.initial_state()
    assert np.isnan(x0[: base.n_y * base.n_channels]).all()
    assert not reduced.is_ready(x0)

    y_meas = np.array([0.1, -0.2])
    x_primed = reduced.absorb(x0, y_meas, np.array([0.3]))
    assert not np.isnan(x_primed[-base.n_u * base.n_controls :]).any()


def test_kirchhoff_penalty_cost_reduces_violation(tmp_path: Path) -> None:
    """Higher w_kirchhoff penalty weight reduces Kirchhoff current sum deviation."""
    art = _build_checkpoint(tmp_path, depth=0, n_y=4, n_u=3, horizon=3, n_channels=2, n_controls=2)

    prob_low_pen = build_waveform_problem(art, horizon=3, u_max=0.5, w_y=1.0, w_u=0.0, w_kirchhoff=0.1, kirchhoff=False)
    prob_high_pen = build_waveform_problem(
        art, horizon=3, u_max=0.5, w_y=1.0, w_u=0.0, w_kirchhoff=100.0, kirchhoff=False
    )

    model_low = prob_low_pen.model
    model_high = prob_high_pen.model
    assert isinstance(model_low, InferencePredictor)
    assert isinstance(model_high, InferencePredictor)

    rng = np.random.default_rng(789)
    x0 = rng.standard_normal(model_low.n)
    state_low = MPCState.initial(prob_low_pen, x0=jnp.asarray(x0), dt=model_low.dt)
    state_high = MPCState.initial(prob_high_pen, x0=jnp.asarray(x0), dt=model_high.dt)

    s_ipopt = SingleShooting(solver=Ipopt(options={"print_level": 0}))
    res_low = prob_low_pen.solve(state_low, solver=s_ipopt)
    res_high = prob_high_pen.solve(state_high, solver=s_ipopt)

    sum_low = np.abs(np.sum(res_low.controls[0]))
    sum_high = np.abs(np.sum(res_high.controls[0]))

    assert sum_high < sum_low


def test_run_waveform_benchmark_integration(tmp_path: Path) -> None:
    """run_waveform_benchmark runs open-loop and closed-loop comparisons across solvers."""
    art = _build_checkpoint(tmp_path, depth=0, n_y=3, n_u=2, horizon=3, n_channels=2, n_controls=2)
    open_comp, closed_comp = run_waveform_benchmark(
        art,
        horizon=3,
        u_max=0.5,
        n_repeats=2,
        num_steps=3,
        include_bipolar_boxqp=True,
    )
    assert len(open_comp.rows) == 4  # SingleShooting, ALTRO, OSQP, Bipolar-BoxQP
    assert len(closed_comp.rows) == 4

    open_table = format_open_loop_table(open_comp)
    closed_table = format_closed_loop_table(closed_comp)
    assert "Open-Loop Solver Comparison" in open_table
    assert "Closed-Loop Receding Horizon MPC Comparison" in closed_table


def test_run_observable_benchmark_integration(tmp_path: Path) -> None:
    """run_observable_benchmark runs open-loop and closed-loop comparisons on Observable OCP."""
    art, geom = _build_observable_checkpoint(tmp_path, n_y=2, n_u=2, horizon=3, n_channels=2, n_controls=2)
    n_values = geom.n_values(50.0)
    env_path = tmp_path / "obs_env.npz"
    np.savez_compressed(
        env_path,
        Pref_frames=np.full((2, n_values), -2.0),
        fs=50.0,
        n_segment=geom.n_segment,
        n_hop=geom.n_hop,
        band_hz=np.asarray(geom.band_hz if geom.band_hz is not None else [-1.0, -1.0]),
        n_bin_pool=geom.n_bin_pool,
        kernel=geom.kernel,
        kernel_width=geom.kernel_width,
    )

    open_comp, closed_comp = run_observable_benchmark(
        art,
        env_path,
        horizon=3,
        u_max=0.5,
        w_u=1.0,
        w_hinge=2.0,
        n_repeats=2,
        num_steps=3,
    )
    assert len(open_comp.rows) == 3  # SingleShooting, ALTRO, OSQP
    assert len(closed_comp.rows) == 3
