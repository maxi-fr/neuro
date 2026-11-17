from __future__ import annotations

import dataclasses
import itertools
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import jax.numpy as jnp
import numpy as np
import pytest
from trajopt.costs.base import CostFunction
from trajopt.costs.l1 import L1ControlCost
from trajopt.costs.objective import Objective
from trajopt.costs.output import OutputCost
from trajopt.costs.sum import cost_terms
from trajopt.mpc import MPC
from trajopt.problem import BoundaryConditions, Problem
from trajopt.program import Program, WarmStart
from trajopt.solvers.altro import ALTRO
from trajopt.solvers.boxqp import BoxQP
from trajopt.trajectory import Trajectory
from trajopt.transcription.ipopt import Ipopt, IpoptResult
from trajopt.transcription.osqp import OSQP
from trajopt.transcription.single_shooting import SingleShooting

from neuro.config import StftGeometry
from neuro.control.candidate import CandidateMPCController
from neuro.control.casadi import build_casadi_observable_problem
from neuro.control.condensed import CondensedOSQP
from neuro.control.condensed_casadi import CondensedBonminController
from neuro.control.costs import ObservableHingeCost
from neuro.control.diagnostics import _rollout_states, decompose_cost
from neuro.control.model_wrappers import AugmentedActivationModel
from neuro.control.mpc import TrajOptMPCController, TrajOptMPCLog
from neuro.control.problems import NeuroProblem, build_observable_problem, build_waveform_problem
from neuro.control.solvers import _build_solver
from neuro.filtering import ObservableEstimator
from neuro.predictor.mlp import ObservableMLPModel, WaveformMLPModel
from neuro.spectral import HealthyReference, ObservableEnvelope
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from pathlib import Path

    import jax

    from neuro.types import FloatArray

_SEED = 7


def _ref(n: int = 2) -> HealthyReference:
    return HealthyReference(eeg_mean=np.zeros(n))


# Pinned parity values from the incumbent CasADi MPCController at fd0d244 (solver="ipopt"), run
# on the depth-0 checkpoint and the _SEED + 5 measurement trajectory below, with Kirchhoff
# applied unconditionally as the incumbent always does.
_WAVEFORM_PARITY_CONTROLS = np.array(
    [
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [-0.5000000097735464, 0.5000000097735464],
        [-0.2197725016350544, 0.2197725016350544],
        [-0.5000000099748614, 0.5000000099748614],
        [-0.5000000099693410, 0.5000000099693410],
        [-0.5000000099527547, 0.5000000099527547],
    ]
)
_WAVEFORM_PARITY_COSTS = [
    0.0,
    0.0,
    0.0,
    1.4970273984042963,
    1.6780944598038890,
    1.8600552355429352,
    1.7721805744284902,
    2.4054735588528806,
]


def _full_parity_solver() -> Ipopt:
    """The general Ipopt transcription that carries the Kirchhoff linear equality."""
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


def _drive_golden(controller: TrajOptMPCController, n_steps: int, n_channels: int) -> tuple[FloatArray, list[float]]:
    """Feed the fixed golden trajectory through ``update``, returning controls and reported costs."""
    rng = np.random.default_rng(_SEED + 5)
    controls = []
    costs = []
    for k in range(n_steps):
        u, log = controller.update(k * controller.dt, ref=np.array([0.0]), x_hat=rng.standard_normal(n_channels))
        controls.append(np.atleast_1d(np.asarray(u, dtype=np.float64)))
        costs.append(log.cost)
    return np.array(controls), costs


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
    model = WaveformMLPModel(
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
        # The golden control/cost values were pinned from the incumbent CasADi controller on the
        # plain (non-residual) semantics, so the parity artifacts keep the skip off; the residual
        # path is pinned separately by the cross-side parity tests.
        residual=False,
        y_std=Standardizer(center=scalers["y_mean"], scale=scalers["y_scale"]),
        u_std=Standardizer(center=scalers["u_mean"], scale=scalers["u_scale"]),
    )
    layers = _random_layers(rng, [in_size, *([5] * depth), n_channels])
    model.weights = tuple(jnp.asarray(w, dtype=jnp.float32) for w, _ in layers)
    model.biases = tuple(jnp.asarray(b, dtype=jnp.float32) for _, b in layers)
    path = tmp_path / "art"
    model.save(path)
    return path


def _drive(controller: TrajOptMPCController, n_steps: int, n_channels: int) -> list[tuple[FloatArray, TrajOptMPCLog]]:
    """Feed ``n_steps`` random EEG measurements through ``update`` and collect the outputs."""
    rng = np.random.default_rng(_SEED + 4)
    out = []
    for k in range(n_steps):
        u, log = controller.update(k * controller.dt, ref=np.array([0.0]), x_hat=rng.standard_normal(n_channels))
        out.append((np.atleast_1d(np.asarray(u, dtype=np.float64)), log))
    return out


def test_absorb_is_ready_initial_state(tmp_path: Path) -> None:
    """The jax model's priming seam holds NaN until ``n_y`` samples are absorbed, then is ready."""
    n_y, n_u, n_channels, n_controls = 4, 3, 2, 2
    artifact = _build_checkpoint(tmp_path, n_y=n_y, n_u=n_u, n_channels=n_channels, n_controls=n_controls)
    adapter = WaveformMLPModel.load(artifact)

    state = adapter.initial_state()
    assert np.isnan(state[: n_y * n_channels]).all()
    assert not adapter.is_ready(state)

    rng = np.random.default_rng(_SEED + 3)
    y_seq = rng.standard_normal((n_y, n_channels))
    u_seq = rng.standard_normal((n_y, n_controls))
    for t in range(n_y):
        state = adapter.absorb(state, y_seq[t], u_seq[t])
        assert adapter.is_ready(state) == (t == n_y - 1)
    assert adapter.is_ready(state)
    assert not np.isnan(state[: n_y * n_channels]).any()


def test_warmup_emits_zero_until_window_filled(tmp_path: Path) -> None:
    """While the EEG window is still NaN-padded, the controller holds off and emits zeros."""
    n_y = 4
    artifact = _build_checkpoint(tmp_path, n_y=n_y)
    problem = build_waveform_problem(artifact, horizon=3, u_max=0.5, w_y=1.0, reference=_ref(2))
    controller = TrajOptMPCController(dt=0.01, problem=problem)

    results = _drive(controller, n_steps=n_y, n_channels=controller.model.n_channels)
    for u, log in results[: n_y - 1]:
        assert log.warmup
        np.testing.assert_array_equal(u, np.zeros(controller.model.m))

    assert not results[-1][1].warmup


def test_decision_log_preserves_unshifted_predictions_and_physical_plan(tmp_path: Path) -> None:
    artifact = _build_checkpoint(tmp_path)
    problem = build_waveform_problem(artifact, horizon=3, u_max=0.5, w_y=1.0, reference=_ref(2), kirchhoff=True)
    controller = TrajOptMPCController(dt=0.01, problem=problem)
    results = _drive(controller, n_steps=5, n_channels=2)
    warmup = results[0][1]
    assert warmup.predicted_y.shape == (4, 2)
    assert np.isnan(warmup.predicted_y).all()
    for u, log in results[3:]:
        assert log.planned_u.shape == (3, 2)
        np.testing.assert_allclose(log.planned_u[0], u)
        np.testing.assert_allclose(log.planned_u.sum(axis=1), 0, atol=1e-10)
        assert not hasattr(log, "measurement")
        assert not hasattr(log, "observed_y")
        assert np.isfinite(log.predicted_y).all()
    saved = results[-1][1].predicted_y.copy()
    _drive(controller, n_steps=2, n_channels=2)
    np.testing.assert_array_equal(results[-1][1].predicted_y, saved)


def test_update_respects_bounds(tmp_path: Path) -> None:
    """Past warm-up, update returns a finite ``(n_controls,)`` control within the box bounds."""
    u_max = 0.5
    artifact = _build_checkpoint(tmp_path, n_y=4)
    problem = build_waveform_problem(artifact, horizon=3, u_max=u_max, w_y=1.0, w_u=0.0, reference=_ref(2))
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
                "class_path": "neuro.control.problems.build_waveform_problem",
                "artifact": str(artifact),
                "horizon": 5,
                "u_max": 0.5,
                "w_y": 1.0,
                "reference": _ref(2),
            },
        }
    )
    assert controller.dt == 0.01
    assert controller.mpc.program.problem.N == 6  # horizon + 1 knot points
    assert controller.model.m == 2


def test_per_electrode_bounds_rejected_when_mismatched(tmp_path: Path) -> None:
    """A u_max length that is neither 1 nor n_controls is rejected by the box builder."""
    artifact = _build_checkpoint(tmp_path, n_controls=2)
    with pytest.raises(ValueError, match="could not be broadcast"):
        build_waveform_problem(artifact, horizon=3, u_max=[1.0, 2.0, 3.0], reference=_ref(2))


def test_controller_keeps_absorbed_state_private(tmp_path: Path) -> None:
    """Each native solve pins its own State-Absorbed history and stores the emitted current."""
    artifact = _build_checkpoint(tmp_path, n_y=4)
    problem = build_waveform_problem(artifact, horizon=3, u_max=0.5, w_y=1.0, reference=_ref(2))
    controller = TrajOptMPCController(dt=0.01, problem=problem)

    n_z = controller.model.n_y * controller.model.n_channels
    results = _drive(controller, n_steps=6, n_channels=controller.model.n_channels)
    u_last, log_last = results[-1]
    assert not log_last.warmup
    assert not np.isnan(controller._state[:n_z]).any()  # noqa: SLF001 -- the test inspects the absorbed state it owns
    np.testing.assert_array_equal(controller._u_last, u_last)  # noqa: SLF001 -- the test verifies the private u_last persistence
    seed = controller.mpc.x0
    np.testing.assert_array_equal(seed, controller._state)  # noqa: SLF001 -- the solved initial state is measured history


def test_reproduces_mpc_controller_control_sequence(tmp_path: Path) -> None:
    """The trajopt controller reproduces the incumbent CasADi control sequence and reported cost.

    The golden values are pinned from the incumbent ``MPCController`` (``solver="ipopt"``) at
    fd0d244 on this same depth-0 (linear, hence convex) checkpoint and measurement trajectory.
    Both the controls and the per-step reported cost must match: the cost assertion is what
    catches the absorbed-measurement term the incumbent graph never scores.
    """
    artifact = _build_checkpoint(tmp_path, depth=0)
    problem = build_waveform_problem(
        artifact, horizon=3, u_max=0.5, w_y=1.0, w_u=0.0, kirchhoff=True, reference=_ref(2)
    )
    controller = TrajOptMPCController(dt=0.01, problem=problem, solver=_full_parity_solver())

    controls, costs = _drive_golden(controller, n_steps=8, n_channels=controller.model.n_channels)
    np.testing.assert_allclose(controls, _WAVEFORM_PARITY_CONTROLS, atol=1e-4)
    np.testing.assert_allclose(costs[3:], _WAVEFORM_PARITY_COSTS[3:], atol=1e-4)
    assert np.isnan(costs[:3]).all()


def test_single_shooting_solver_succeeds_when_kirchhoff(tmp_path: Path) -> None:
    """An injected single-shooting solver on a Kirchhoff problem constructs and solves without error."""
    artifact = _build_checkpoint(tmp_path, depth=0)
    problem = build_waveform_problem(artifact, horizon=3, u_max=0.5, w_y=1.0, kirchhoff=True, reference=_ref(2))
    controller = TrajOptMPCController(
        dt=0.01,
        problem=problem,
        solver=SingleShooting(solver=Ipopt(options={"print_level": 0})),
    )
    assert type(controller.solver) is SingleShooting


def _build_observable_checkpoint(
    tmp_path: Path,
    *,
    n_y: int = 3,
    n_u: int = 2,
    horizon: int = 4,
    n_channels: int = 2,
    n_controls: int = 2,
    depth: int = 1,
    geom: StftGeometry | None = None,
) -> tuple[Path, StftGeometry]:
    """Save a synthetic Observable MLP checkpoint and return its stem and geometry."""
    rng = np.random.default_rng(_SEED + 1)
    if geom is None:
        geom = StftGeometry(n_segment=20, n_hop=5, band_hz=(4.0, 16.0), n_bin_pool=2)
    fs = 50.0
    n_values = geom.n_values(fs)
    n_outputs = n_channels * n_values
    in_size = n_y * n_outputs + n_u * n_controls
    scalers = {
        "u_mean": rng.uniform(-1.0, 1.0, n_controls),
        "u_scale": rng.uniform(0.5, 2.0, n_controls),
        "y_mean": rng.uniform(-1.0, 1.0, n_outputs),
        "y_scale": rng.uniform(0.5, 2.0, n_outputs),
    }
    model = ObservableMLPModel(
        n_y=n_y,
        n_u=n_u,
        horizon=horizon,
        n_channels=n_channels,
        n_controls=n_controls,
        hidden_size=5,
        depth=depth,
        activation="relu",
        dt=geom.n_hop / fs,
        n_outputs=n_outputs,
        geometry=geom,
        residual=False,
        y_std=Standardizer(center=scalers["y_mean"], scale=scalers["y_scale"]),
        u_std=Standardizer(center=scalers["u_mean"], scale=scalers["u_scale"]),
    )
    sizes = [in_size, *([5] * depth), n_outputs]
    layers = _random_layers(rng, sizes)
    model.weights = tuple(jnp.asarray(w, dtype=jnp.float32) for w, _ in layers)
    model.biases = tuple(jnp.asarray(b, dtype=jnp.float32) for _, b in layers)
    path = tmp_path / "obs_art"
    model.save(path)
    return path, geom


def _write_envelope(tmp_path: Path, geom: StftGeometry, n_channels: int = 2) -> Path:
    n_values = geom.n_values(50.0)
    env_path = tmp_path / f"obs_env_{n_channels}.npz"
    np.savez_compressed(
        env_path,
        Pref_frames=np.full((n_channels, n_values), -2.0),
        fs=50.0,
        n_segment=geom.n_segment,
        n_hop=geom.n_hop,
        band_hz=np.asarray(geom.band_hz if geom.band_hz is not None else [-1.0, -1.0]),
        n_bin_pool=geom.n_bin_pool,
        kernel=geom.kernel,
        kernel_width=geom.kernel_width,
    )
    return env_path


def test_build_observable_problem_assembles_and_solves(tmp_path: Path) -> None:
    """The observable problem builder wires the hinge, L1, quadratic, box bounds and Kirchhoff, and solves."""
    artifact, geom = _build_observable_checkpoint(tmp_path, n_channels=2, n_controls=2)
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

    problem = build_observable_problem(
        artifact,
        horizon=4,
        u_max=0.5,
        w_u=1.0,
        w_u_l1=0.2,
        w_hinge=5.0,
        reference=HealthyReference.load(env_path),
        kirchhoff=True,
    )
    assert problem.N == 5
    # The stage trajectory carries every Frame of the Control Horizon but the last; the terminal
    # Cost prices that one, so no predicted Frame the controls move goes unscored.
    assert isinstance(problem.obj.terminal_cost, OutputCost)
    assert isinstance(problem.obj.terminal_cost.cost, ObservableHingeCost)
    assert problem.obj.terminal_cost.terminal
    assert isinstance(problem.model, ObservableMLPModel)
    assert problem.model.n_outputs == 2 * n_values
    assert problem.model.m == 2

    # Solve from a valid ready state
    rng = np.random.default_rng(_SEED + 2)
    model = ObservableMLPModel.load(artifact)
    x0 = np.asarray(model.initial_state())
    x0[: model.n_y * model.n_outputs] = rng.uniform(-1.0, 1.0, model.n_y * model.n_outputs)
    # Ipopt's 1e-8 default is below the noise floor of a float32 objective whose optimum sits
    # inside PseudoHuberControlCost's delta=1e-3 smoothing radius, where curvature is 1/delta; the solve
    # stalls there at a dual infeasibility of ~1e-2 with the objective already flat to 1e-6.
    solver = Ipopt(options={"print_level": 0, "hessian_approximation": "limited-memory", "tol": 1e-3})
    mpc = MPC(problem, solver, x0=jnp.asarray(x0))
    assert mpc.solve().success

    controls = np.asarray(mpc.controls)
    assert controls.shape == (4, 2)
    assert np.all(np.abs(controls) <= 0.5 + 1e-6)
    np.testing.assert_allclose(np.sum(controls, axis=1), np.zeros(4), atol=1e-5)


def test_observable_closed_loop_warmup_and_emission(tmp_path: Path) -> None:
    """Closed-loop run primes the predictor, emits zeros during Warm-up Period, then finite controls."""
    n_y, n_u, n_channels = 3, 2, 2
    artifact, geom = _build_observable_checkpoint(
        tmp_path, n_y=n_y, n_u=n_u, horizon=4, n_channels=n_channels, n_controls=2
    )
    n_values = geom.n_values(50.0)
    env_path = tmp_path / "obs_env.npz"
    np.savez_compressed(
        env_path,
        Pref_frames=np.full((n_channels, n_values), -2.0),
        fs=50.0,
        n_segment=geom.n_segment,
        n_hop=geom.n_hop,
        band_hz=np.asarray(geom.band_hz if geom.band_hz is not None else [-1.0, -1.0]),
        n_bin_pool=geom.n_bin_pool,
        kernel=geom.kernel,
        kernel_width=geom.kernel_width,
    )

    problem = build_observable_problem(
        artifact,
        horizon=4,
        u_max=0.5,
        w_u=1.0,
        w_u_l1=0.2,
        w_hinge=5.0,
        reference=HealthyReference.load(env_path),
        kirchhoff=True,
    )
    # Plant fs = 1000 Hz (dt = 0.001s), downsample = 20 -> fs_decimated = 50.0 Hz.
    # Hop = 5 decimated samples -> hop duration = 5 / 50.0 = 0.10s = controller dt.
    plant_dt = 0.001
    downsample = 20
    controller_dt = geom.n_hop * downsample * plant_dt  # 0.10s
    controller = TrajOptMPCController(dt=controller_dt, problem=problem)
    estimator = ObservableEstimator(dt=plant_dt, geometry=geom, downsample=downsample)

    plant_steps = 850
    rng = np.random.default_rng(_SEED + 3)
    y_plant = rng.standard_normal((plant_steps, n_channels))

    u_applied = np.zeros(2)
    controller_outputs: list[tuple[float, FloatArray, TrajOptMPCLog]] = []

    for k in range(plant_steps):
        t = k * plant_dt
        x_hat, _ = estimator.evaluate(t, y_plant[k], u_applied)
        # Controller ticks every 100 plant steps (every 0.10 s)
        if k % 100 == 0:
            u_applied, log = controller.update(t, ref=np.array([0.0]), x_hat=x_hat)
            controller_outputs.append((t, u_applied.copy(), log))

    # Controller ticks at t = 0.0, 0.1, 0.2, 0.3, 0.4, 0.5 (indices 0..5):
    # Estimator warms up for first 4 ticks; controller primes for next 2 ticks.
    # At tick 6 (t = 0.6s), controller is primed and emits finite control!
    for i in range(6):
        t, u_cmd, log = controller_outputs[i]
        assert log.warmup
        np.testing.assert_array_equal(u_cmd, np.zeros(2))

    for i in range(6, len(controller_outputs)):
        t, u_cmd, log = controller_outputs[i]
        assert not log.warmup
        assert log.success
        assert np.isfinite(u_cmd).all()
        assert np.any(u_cmd != 0.0)
        assert np.all(np.abs(u_cmd) <= 0.5 + 1e-6)
        np.testing.assert_allclose(np.sum(u_cmd), 0.0, atol=1e-5)


def test_observable_controller_from_config(tmp_path: Path) -> None:
    """A full controller dict with build_observable_problem builds through from_config."""
    artifact, geom = _build_observable_checkpoint(tmp_path, n_channels=2, n_controls=2)
    env_path = _write_envelope(tmp_path, geom)

    cfg = {
        "class_path": "neuro.control.mpc.TrajOptMPCController",
        "dt": 0.10,
        "problem": {
            "class_path": "neuro.control.problems.build_observable_problem",
            "artifact": str(artifact),
            "horizon": 4,
            "u_max": 1.0,
            "w_u": 5.0,
            "w_hinge": 2.0,
            "reference": str(env_path),
            "kirchhoff": True,
        },
    }
    controller = TrajOptMPCController.from_config(cfg)
    assert controller.dt == 0.10


def test_build_observable_problem_envelope_cross_validation(tmp_path: Path) -> None:
    """build_observable_problem validates envelope channel count, sampling rate, and geometry."""
    artifact, geom = _build_observable_checkpoint(tmp_path, n_channels=2, n_controls=2)
    n_values = geom.n_values(50.0)

    # Mismatched channel count raises
    bad_ch = tmp_path / "bad_ch.npz"
    np.savez_compressed(
        bad_ch,
        Pref_frames=np.full((3, n_values), -2.0),
        fs=50.0,
        n_segment=geom.n_segment,
        n_hop=geom.n_hop,
        band_hz=np.asarray(geom.band_hz if geom.band_hz is not None else [-1.0, -1.0]),
        n_bin_pool=geom.n_bin_pool,
        kernel=geom.kernel,
        kernel_width=geom.kernel_width,
    )
    with pytest.raises(ValueError, match=r"envelope channel count \(3\) does not match model channel count \(2\)"):
        build_observable_problem(artifact, horizon=4, u_max=0.5, w_hinge=1.0, reference=HealthyReference.load(bad_ch))

    # Mismatched sampling rate raises
    bad_fs = tmp_path / "bad_fs.npz"
    np.savez_compressed(
        bad_fs,
        Pref_frames=np.full((2, geom.n_values(100.0)), -2.0),
        fs=100.0,
        n_segment=geom.n_segment,
        n_hop=geom.n_hop,
        band_hz=np.asarray(geom.band_hz if geom.band_hz is not None else [-1.0, -1.0]),
        n_bin_pool=geom.n_bin_pool,
        kernel=geom.kernel,
        kernel_width=geom.kernel_width,
    )
    with pytest.raises(ValueError, match=r"envelope sampling rate \(100 Hz\) is a Frame rate of 20 Hz at hop 5"):
        build_observable_problem(artifact, horizon=4, u_max=0.5, w_hinge=1.0, reference=HealthyReference.load(bad_fs))

    # Mismatched geometry band_hz raises
    bad_band = tmp_path / "bad_band.npz"
    np.savez_compressed(
        bad_band,
        Pref_frames=np.full((2, 2), -2.0),
        fs=50.0,
        n_segment=geom.n_segment,
        n_hop=geom.n_hop,
        band_hz=np.array([2.0, 10.0]),
        n_bin_pool=geom.n_bin_pool,
        kernel=geom.kernel,
        kernel_width=geom.kernel_width,
    )
    with pytest.raises(
        ValueError,
        match=r"envelope geometry does not match model geometry: band_hz \(\(2\.0, 10\.0\) vs \(4\.0, 16\.0\)\)",
    ):
        build_observable_problem(artifact, horizon=4, u_max=0.5, w_hinge=1.0, reference=HealthyReference.load(bad_band))


def test_default_solver_selection(tmp_path: Path) -> None:
    """The default is stock SingleShooting(Ipopt) for every deployed formulation."""
    art_wf = _build_checkpoint(tmp_path / "wf", depth=0, n_y=3, n_u=2, horizon=4, n_channels=2, n_controls=3)

    geometry = StftGeometry(n_segment=4, n_hop=2)
    envelope = ObservableEnvelope(power=np.full((2, 3), -2.0), fs=100.0, geometry=geometry)
    problems = (
        build_waveform_problem(art_wf, horizon=4, u_max=0.5, w_y=1.0, kirchhoff=True, reference=_ref(2)),
        build_waveform_problem(
            art_wf,
            horizon=4,
            u_max=0.5,
            w_y=0.0,
            w_hinge=1.0,
            reference=HealthyReference(observable=envelope),
        ),
    )
    for problem in problems:
        solver = _build_solver(None, problem)
        assert isinstance(solver, SingleShooting)
        assert isinstance(solver.solver, Ipopt)


def test_controller_from_config_solver_variants(tmp_path: Path) -> None:
    """TrajOptMPCController.from_config dispatches class_path solver configs across backends."""
    art = _build_checkpoint(tmp_path, depth=0, n_y=3, n_u=2, horizon=4, n_channels=2, n_controls=2)

    # 1. Default (omitted solver on waveform model -> SingleShooting)
    cfg_default = {
        "dt": 0.01,
        "problem": {
            "class_path": "neuro.control.problems.build_waveform_problem",
            "artifact": str(art),
            "horizon": 4,
            "u_max": 0.5,
            "reference": _ref(2),
        },
    }
    ctrl_def = TrajOptMPCController.from_config(cfg_default)
    assert isinstance(ctrl_def.solver, SingleShooting)

    # 2. Explicit ALTRO with options
    cfg_altro = {
        "dt": 0.01,
        "solver": {
            "class_path": "trajopt.solvers.altro.ALTRO",
            "options": {"constraint_tolerance": 1e-4, "cost_tolerance": 1e-4},
        },
        "problem": {
            "class_path": "neuro.control.problems.build_waveform_problem",
            "artifact": str(art),
            "horizon": 4,
            "u_max": 0.5,
            "kirchhoff": True,
            "reference": _ref(2),
        },
    }
    ctrl_altro = TrajOptMPCController.from_config(cfg_altro)
    assert isinstance(ctrl_altro.solver, ALTRO)
    assert ctrl_altro.solver.options.constraint_tolerance == 1e-4

    # 3. Explicit OSQP with options
    cfg_osqp = {
        "dt": 0.01,
        "solver": {
            "class_path": "trajopt.transcription.osqp.OSQP",
            "options": {"eps_abs": 1e-5},
        },
        "problem": {
            "class_path": "neuro.control.problems.build_waveform_problem",
            "artifact": str(art),
            "horizon": 4,
            "u_max": 0.5,
            "reference": _ref(2),
        },
    }
    ctrl_osqp = TrajOptMPCController.from_config(cfg_osqp)
    assert isinstance(ctrl_osqp.solver, OSQP)

    # 4. Nested SingleShooting(solver=Ipopt(...))
    cfg_ss = {
        "dt": 0.01,
        "solver": {
            "class_path": "trajopt.transcription.single_shooting.SingleShooting",
            "solver": {
                "class_path": "trajopt.transcription.ipopt.Ipopt",
                "options": {"print_level": 0},
            },
        },
        "problem": {
            "class_path": "neuro.control.problems.build_waveform_problem",
            "artifact": str(art),
            "horizon": 4,
            "u_max": 0.5,
            "reference": _ref(2),
        },
    }
    ctrl_ss = TrajOptMPCController.from_config(cfg_ss)
    assert isinstance(ctrl_ss.solver, SingleShooting)
    assert isinstance(ctrl_ss.solver.solver, Ipopt)


def test_controller_from_config_loud_validation_errors(tmp_path: Path) -> None:
    """from_config raises immediately on malformed configs or incompatible solvers."""
    art = _build_checkpoint(tmp_path, depth=0, n_y=3, n_u=2, horizon=4, n_channels=2, n_controls=2)

    # 1. Missing class_path in solver dict raises ValueError
    cfg_missing = {
        "dt": 0.01,
        "solver": {"name": "altro"},
        "problem": {
            "class_path": "neuro.control.problems.build_waveform_problem",
            "artifact": str(art),
            "horizon": 4,
            "u_max": 0.5,
            "reference": _ref(2),
        },
    }
    with pytest.raises(ValueError, match="solver config must contain 'class_path'"):
        TrajOptMPCController.from_config(cfg_missing)

    # 2. String instead of dict raises TypeError
    cfg_str = {
        "dt": 0.01,
        "solver": "altro",
        "problem": {
            "class_path": "neuro.control.problems.build_waveform_problem",
            "artifact": str(art),
            "horizon": 4,
            "u_max": 0.5,
            "reference": _ref(2),
        },
    }
    with pytest.raises(TypeError, match="must be a dict with 'class_path'"):
        TrajOptMPCController.from_config(cfg_str)

    # 3. Expansion solver on whole-horizon PSD cost raises ValueError
    geometry = StftGeometry(n_segment=4, n_hop=2)
    obs_ref = tmp_path / "observable_ref.npz"
    np.savez_compressed(
        obs_ref,
        Pref_frames=np.full((2, geometry.n_values(100.0)), -2.0),
        fs=100.0,
        n_segment=geometry.n_segment,
        n_hop=geometry.n_hop,
        band_hz=np.asarray([-1.0, -1.0]),
        n_bin_pool=geometry.n_bin_pool,
        kernel=geometry.kernel,
        kernel_width=geometry.kernel_width,
    )
    cfg_incompatible = {
        "dt": 0.01,
        "solver": {"class_path": "trajopt.solvers.altro.ALTRO"},
        "problem": {
            "class_path": "neuro.control.problems.build_waveform_problem",
            "artifact": str(art),
            "horizon": 4,
            "u_max": 0.5,
            "w_y": 0.0,
            "w_hinge": 1.0,
            "reference": str(obs_ref),
        },
    }
    with pytest.raises(ValueError, match="cannot score the whole-horizon hinge cost"):
        TrajOptMPCController.from_config(cfg_incompatible)

    # 4. BoxQP on unhandled coupled linear equality constraints raises ValueError
    cfg_boxqp_coupled = {
        "dt": 0.01,
        "solver": {"class_path": "trajopt.solvers.boxqp.BoxQP"},
        "problem": {
            "class_path": "neuro.control.problems.build_waveform_problem",
            "artifact": str(art),
            "horizon": 4,
            "u_max": 0.5,
            "kirchhoff": True,
            "reference": _ref(2),
        },
    }
    with pytest.raises(ValueError, match="BoxQP only supports uncoupled box bounds"):
        TrajOptMPCController.from_config(cfg_boxqp_coupled)


def test_problem_carries_the_predictors_step_as_its_time_grid(tmp_path: Path) -> None:
    """The native controller and Problem grids match the Predictor while solves may run slower."""
    art = _build_checkpoint(tmp_path, depth=0, n_y=3, n_u=2, horizon=4, n_channels=2, n_controls=3)
    model_dt = WaveformMLPModel.load(art).dt

    problem = build_waveform_problem(art, horizon=4, u_max=0.5, kirchhoff=True, reference=_ref(2))
    np.testing.assert_allclose(np.asarray(problem.dt), model_dt)

    # Slower replanning leaves native State Absorption and the prediction grid unchanged.
    controller = TrajOptMPCController.from_config(
        {
            "dt": model_dt,
            "replan_dt": 4.0 * model_dt,
            "problem": {
                "class_path": "neuro.control.problems.build_waveform_problem",
                "artifact": str(art),
                "horizon": 4,
                "u_max": 0.5,
                "reference": _ref(2),
            },
        }
    )
    assert controller.dt == pytest.approx(model_dt)
    assert controller.replan_dt == pytest.approx(4.0 * model_dt)
    with pytest.raises(ValueError, match="Predictor step"):
        TrajOptMPCController(dt=10.0 * model_dt, problem=problem)
    np.testing.assert_allclose(np.asarray(controller.mpc.program.problem.dt), model_dt)


def test_single_shooting_canonical_duals_survive_native_advancement(tmp_path: Path) -> None:
    """Stock single shooting returns canonical Multipliers usable by the persistent driver."""
    art = _build_checkpoint(tmp_path, depth=0, n_y=3, n_u=2, horizon=4, n_channels=2, n_controls=3)
    problem = build_waveform_problem(art, horizon=4, u_max=0.5, w_y=1.0, kirchhoff=True, reference=_ref(2))
    x0 = jnp.asarray(np.random.default_rng(7).standard_normal(problem.model.n))
    driver = MPC(problem, SingleShooting(solver=Ipopt(options={"print_level": 0})), x0=x0)
    program = driver.program
    first = driver.solve()
    N, n, m = int(problem.N), int(problem.model.n), int(problem.model.m)
    assert first.success
    assert len(first.mu) == N * n + (N - 1) * m
    assert len(first.lam) == N * n + sum(problem.constraints.p)
    driver.advance(1)
    driver.measure(x0, 0.01)
    assert driver.solve().success
    assert driver.program is program


def test_default_ipopt_solver_options(tmp_path: Path) -> None:
    """Default single shooting retains the tuned IPOPT tolerances and Hessian approximation."""
    art = _build_checkpoint(tmp_path, depth=0, n_y=2, n_u=2, horizon=3, n_channels=2, n_controls=3)
    problem = build_waveform_problem(art, horizon=3, u_max=0.5, reference=_ref(2))
    solver = _build_solver(None, problem)
    assert isinstance(solver, SingleShooting)
    assert isinstance(solver.solver, Ipopt)
    opts = dict(solver.solver.options)
    assert opts["tol"] == 1e-3
    assert opts["acceptable_tol"] == 1e-2
    assert opts["acceptable_iter"] == 5
    assert opts["print_level"] == 0
    assert opts["hessian_approximation"] == "limited-memory"
    assert opts["max_iter"] == 300


def test_build_observable_problem_mixed_integer_validation(tmp_path: Path) -> None:
    """build_observable_problem validates w_active and max_active_intervals domain rules."""
    artifact, _ = _build_observable_checkpoint(tmp_path, n_channels=2, n_controls=2)

    with pytest.raises(ValueError, match="mutually exclusive"):
        build_observable_problem(artifact, horizon=3, u_max=1.0, w_active=0.1, max_active_intervals=2)

    with pytest.raises(ValueError, match="w_active must be finite and nonnegative"):
        build_observable_problem(artifact, horizon=3, u_max=1.0, w_active=-0.5)

    with pytest.raises(ValueError, match="max_active_intervals must be an integer"):
        build_observable_problem(artifact, horizon=3, u_max=1.0, max_active_intervals=-1)

    with pytest.raises(ValueError, match="max_active_intervals must be an integer"):
        build_observable_problem(artifact, horizon=3, u_max=1.0, max_active_intervals=5)


def test_build_observable_problem_mixed_integer_structure(tmp_path: Path) -> None:
    """build_observable_problem configures binary coordinates and horizon constraints."""
    artifact, _ = _build_observable_checkpoint(tmp_path, n_channels=2, n_controls=2)
    prob_w = build_observable_problem(artifact, horizon=3, u_max=1.0, w_active=0.2, kirchhoff=True)
    assert prob_w.binary_control_indices == (2,)
    assert isinstance(prob_w.model, AugmentedActivationModel)
    assert prob_w.model.m == 3
    assert prob_w.model.n_controls == 2
    assert prob_w.horizon == 3
    assert prob_w.w_active == 0.2

    prob_cap = build_observable_problem(artifact, horizon=3, u_max=1.0, max_active_intervals=1, kirchhoff=True)
    assert prob_cap.binary_control_indices == (2,)
    assert len(prob_cap.constraints.horizon_constraints) == 1
    assert prob_cap.max_active_intervals == 1


@dataclasses.dataclass(frozen=True)
class _PlanCheck:
    objective: float
    hinge_cost: float
    effort_cost: float
    activation_cost: float
    current_violation: float
    balance_violation: float
    integrality_violation: float
    predicted_y: FloatArray


def _check_plan(
    problem: NeuroProblem,
    state: FloatArray,
    planned_u: FloatArray,
    planned_active: FloatArray,
) -> _PlanCheck:
    model: Any = problem.model
    h = problem.horizon
    m = model.n_controls
    p = model.n_outputs
    u = np.asarray(planned_u, dtype=np.float64).reshape(h, m)
    active = np.asarray(planned_active, dtype=np.float64).reshape(h)
    state_arr = np.asarray(state, dtype=np.float64).reshape(-1)
    n_output_state = model.n_history * p
    y_history = state_arr[:n_output_state].reshape(model.n_history, p)
    y_history = y_history * np.asarray(model.y_scale) + np.asarray(model.y_center)
    u_history = state_arr[n_output_state:].reshape(model.n_u, m)
    forecast = np.asarray(model.free_run(y_history[None], u_history[None], u[None])[0], dtype=np.float64).reshape(h, p)
    initial_y = y_history[-1:]
    predicted_y = np.concatenate((initial_y, forecast), axis=0)
    envelope = problem.envelope
    excess = np.maximum(forecast - np.asarray(envelope.power).reshape(1, p), 0.0) if envelope is not None else 0.0
    hinge_cost = float(problem.w_hinge * np.sum(np.square(excess)) / (h * p))
    effort_cost = float(problem.w_u * np.sum(u**2) / h)
    activation_cost = float((problem.w_active or 0.0) * np.sum(active) / h)
    sparse_cost = float(problem.w_u_l1 * np.sum(np.sqrt(u**2 + 1e-6)) / h)
    current_violation = float(np.max(np.maximum(np.abs(u) - np.asarray(problem.u_max)[None, :] * active[:, None], 0.0)))
    balance_violation = float(np.max(np.abs(np.sum(u, axis=1)))) if problem.kirchhoff else 0.0
    integrality_violation = float(np.max(np.abs(active - np.round(active))))
    return _PlanCheck(
        objective=hinge_cost + effort_cost + activation_cost + sparse_cost,
        hinge_cost=hinge_cost,
        effort_cost=effort_cost + sparse_cost,
        activation_cost=activation_cost,
        current_violation=current_violation,
        balance_violation=balance_violation,
        integrality_violation=integrality_violation,
        predicted_y=predicted_y,
    )


def test_mixed_integer_mpc_controller_solve_and_update(tmp_path: Path) -> None:
    """Native-rate binary execution retains plans certified by _check_plan."""
    artifact, geom = _build_observable_checkpoint(tmp_path, n_channels=2, n_controls=2)
    n_values = geom.n_values(50.0)
    env_path = tmp_path / "obs_env_mi.npz"
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

    prob = build_observable_problem(
        artifact,
        horizon=2,
        u_max=1.0,
        w_u=1.0,
        w_active=0.3,
        reference=HealthyReference.load(env_path),
        kirchhoff=True,
    )
    ctrl = TrajOptMPCController(dt=float(prob.dt[0]), problem=prob)
    assert ctrl.n_controls == 2
    assert ctrl.n_electrodes == 2
    assert ctrl.is_integer_mode

    model = ObservableMLPModel.load(artifact)
    x0 = np.asarray(model.initial_state())
    x0[: model.n_y * model.n_outputs] = 0.1

    log = ctrl.solve_state(x0)
    assert log.success
    assert log.planned_u.shape == (2, 2)
    assert log.planned_active is not None
    assert log.planned_active.shape == (2,)
    assert log.active_count is not None

    assert isinstance(ctrl.problem, NeuroProblem)
    checked = _check_plan(ctrl.problem, x0, log.planned_u, log.planned_active)
    assert checked.integrality_violation < 1e-4
    assert checked.balance_violation < 1e-6
    assert checked.current_violation < 1e-6
    assert checked.objective == pytest.approx(log.cost, rel=1e-5)

    u_cmd, update_log = ctrl.update(0.0, np.zeros(2), np.zeros(model.n_outputs))
    assert u_cmd.shape == (2,)
    assert update_log.planned_u.shape == (2, 2)


def test_candidate_mpc_controller_update_and_config(tmp_path: Path) -> None:
    """CandidateMPCController respects its configured release while absorbing history."""
    artifact, _ = _build_observable_checkpoint(tmp_path, n_channels=2, n_controls=2)
    prob = build_observable_problem(artifact, horizon=3, u_max=1.0, kirchhoff=True)

    ctrl = CandidateMPCController(
        dt=0.06,
        problem=prob,
        blocks=[[0], [1, 2]],
        w_active=0.3,
    )
    assert ctrl.n_controls == 2
    assert ctrl.n_electrodes == 2
    assert len(ctrl.schedules) == 4

    model = ObservableMLPModel.load(artifact)
    u_cmd, log = ctrl.update(0.0, np.zeros(2), np.zeros(model.n_outputs))
    assert log.warmup
    assert u_cmd.shape == (2,)

    cfg = {
        "class_path": "neuro.control.candidate.CandidateMPCController",
        "dt": 0.06,
        "control_start_s": 1.38,
        "blocks": [[0], [1, 2]],
        "w_active": 0.3,
        "problem": {
            "class_path": "neuro.control.problems.build_observable_problem",
            "artifact": str(artifact),
            "horizon": 3,
            "u_max": 1.0,
            "kirchhoff": True,
        },
    }
    loaded = CandidateMPCController.from_config(cfg)
    assert loaded.dt == 0.06
    assert len(loaded.schedules) == 4
    for tick in range(23):
        current, log = loaded.update(tick * 0.06, np.zeros(2), np.zeros(model.n_outputs))
        assert log.warmup
        np.testing.assert_array_equal(current, np.zeros(2))
    _, log = loaded.update(1.38, np.zeros(2), np.zeros(model.n_outputs))
    assert log.solve_attempted


def test_trajopt_mpc_controller_cont_polish(tmp_path: Path) -> None:
    """Capped binary updates reuse stock drivers and retain the original reported Cost."""
    artifact, geom = _build_observable_checkpoint(tmp_path, n_channels=2, n_controls=2)
    n_values = geom.n_values(50.0)
    env_path = tmp_path / "obs_env_polish.npz"
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

    prob = build_observable_problem(
        artifact,
        horizon=2,
        u_max=1.0,
        max_active_intervals=1,
        reference=HealthyReference.load(env_path),
        kirchhoff=True,
    )
    assert prob.continuous_problem is not None

    ctrl = TrajOptMPCController(
        dt=float(prob.dt[0]),
        problem=prob,
        init_mode="cont_polish",
        solver={
            "class_path": "trajopt.transcription.single_shooting.SingleShooting",
            "solver": {
                "class_path": "trajopt.transcription.ipopt.Ipopt",
                "beta_initial": 2.0,
                "beta_growth": 8.0,
                "max_passes": 10,
                "binary_tolerance": 1e-4,
            },
        },
    )
    assert ctrl.init_mode == "cont_polish"
    assert ctrl._cont_mpc is not None  # noqa: SLF001 -- test verifies internal continuous driver creation

    model = ObservableMLPModel.load(artifact)
    u_cmd, log = ctrl.update(0.0, np.zeros(2), np.zeros(model.n_outputs))
    assert log.warmup
    assert u_cmd.shape == (2,)

    assert isinstance(ctrl.solver, SingleShooting)
    assert ctrl.solver.solver.beta_initial == 2.0
    assert ctrl.solver.solver.beta_growth == 8.0
    assert ctrl.solver.solver.max_passes == 10
    program = ctrl.mpc.program
    continuous_program = ctrl._cont_mpc.program  # noqa: SLF001 -- both persistent drivers are the reuse boundary
    for k in range(1, ctrl.model.n_history + 2):
        u_cmd, log = ctrl.update(k * ctrl.dt, np.zeros(2), np.zeros(model.n_outputs))
        if log.warmup:
            continue
        assert log.success
        assert np.sum(log.planned_active) <= 1 + 1e-4
        assert np.max(np.abs(log.planned_active - np.round(log.planned_active))) < 1e-4
        checked = _check_plan(prob, np.asarray(ctrl.mpc.x0), log.planned_u, log.planned_active)
        assert log.cost == pytest.approx(checked.objective, rel=1e-5)
        np.testing.assert_allclose(u_cmd, log.planned_u[0])
        assert ctrl.mpc.program is program
        assert ctrl._cont_mpc.program is continuous_program  # noqa: SLF001 -- relaxation Program remains persistent


def test_build_waveform_problem_epigraph_construction(tmp_path: Path) -> None:
    """Waveform exact L1 exposes a stage term with physical Control Current dimensions."""
    artifact = _build_checkpoint(tmp_path, n_channels=2, n_controls=2)
    prob = build_waveform_problem(
        artifact,
        horizon=3,
        u_max=1.0,
        w_y=0.0,
        w_u=0.1,
        w_u_l1=0.5,
        l1_mode="epigraph",
    )
    assert any(isinstance(cost, L1ControlCost) for cost in cost_terms(prob.obj.stage_cost))
    assert prob.model.m == 2
    assert isinstance(prob.model, WaveformMLPModel)
    assert prob.w_u_l1 == 0.5

    # Check that invalid l1_mode raises ValueError
    with pytest.raises(ValueError, match="l1_mode must be 'smooth' or 'epigraph'"):
        build_waveform_problem(
            artifact,
            horizon=3,
            u_max=1.0,
            w_y=0.0,
            w_u_l1=0.5,
            l1_mode="invalid",
        )


def test_build_observable_problem_epigraph_construction(tmp_path: Path) -> None:
    """Observable exact L1 leaves physical Control Current dimensions intact."""
    artifact, geom = _build_observable_checkpoint(tmp_path, n_channels=2, n_controls=2)
    env_path = _write_envelope(tmp_path, geom)
    prob = build_observable_problem(
        artifact,
        horizon=3,
        u_max=1.0,
        w_u_l1=0.5,
        reference=HealthyReference.load(env_path),
        l1_mode="epigraph",
    )
    assert any(isinstance(cost, L1ControlCost) for cost in cost_terms(prob.obj.stage_cost))
    assert prob.model.m == 2
    assert isinstance(prob.model, ObservableMLPModel)
    assert prob.w_u_l1 == 0.5


def test_boxqp_rejects_exact_l1_epigraph(tmp_path: Path) -> None:
    """Reject exact L1 constraints even when the physical Problem has only box bounds."""
    artifact = _build_checkpoint(tmp_path, depth=0)
    problem = build_waveform_problem(artifact, horizon=3, u_max=0.5, w_y=0.0, w_u_l1=0.2, l1_mode="epigraph")
    with pytest.raises(ValueError, match="BoxQP only supports uncoupled box bounds"):
        TrajOptMPCController(0.01, problem, solver=BoxQP())


def test_epigraph_exact_zero_recovery(tmp_path: Path) -> None:
    """Exact L1 recovers zero Control Currents in the physical warm start."""
    artifact = _build_checkpoint(tmp_path, n_channels=2, n_controls=2)
    # With tracking disabled (w_y=0) and w_u_l1 > 0, the optimal control is exactly 0.
    prob = build_waveform_problem(
        artifact,
        horizon=4,
        u_max=1.0,
        w_y=0.0,
        w_u=1e-3,
        w_u_l1=1.0,
        l1_mode="epigraph",
    )
    ctrl = TrajOptMPCController(dt=0.01, problem=prob)

    # Prime controller past warmup
    for k in range(ctrl.model.n_history):
        u_cmd, log = ctrl.update(k * 0.01, ref=np.zeros(1), x_hat=np.zeros(2))

    assert not log.warmup
    assert log.success
    # u_cmd and planned_u must be physical dimensions
    assert u_cmd.shape == (2,)
    assert log.planned_u.shape == (4, 2)
    np.testing.assert_allclose(u_cmd, 0.0, atol=1e-5)
    np.testing.assert_allclose(log.planned_u, 0.0, atol=1e-5)


def test_trajopt_mpc_controller_epigraph_closed_loop(tmp_path: Path) -> None:
    """Exact L1 execution exposes physical plans and warm starts across replans."""
    artifact = _build_checkpoint(tmp_path, n_channels=2, n_controls=2)
    prob = build_waveform_problem(
        artifact,
        horizon=3,
        u_max=0.5,
        w_y=1.0,
        w_u=0.1,
        w_u_l1=0.2,
        reference=_ref(2),
        kirchhoff=True,
        l1_mode="epigraph",
    )
    ctrl = TrajOptMPCController(dt=0.01, problem=prob)
    assert ctrl.n_electrodes == 2

    controller_program = ctrl.mpc.program

    # Warmup steps
    for k in range(ctrl.model.n_history - 1):
        u_cmd, log = ctrl.update(k * 0.01, ref=np.zeros(1), x_hat=np.zeros(2))
        assert log.warmup
        assert u_cmd.shape == (2,)
        np.testing.assert_array_equal(u_cmd, np.zeros(2))

    # Stepping when ready
    t_ready = (ctrl.model.n_history - 1) * 0.01
    u_cmd, log = ctrl.update(t_ready, ref=np.zeros(1), x_hat=np.ones(2) * 0.5)
    assert not log.warmup
    assert log.success
    assert u_cmd.shape == (2,)
    assert log.planned_u.shape == (3, 2)
    # Check that planned_active matches physical currents
    expected_active = (np.abs(log.planned_u).max(axis=-1) > 1e-3).astype(np.float64)
    np.testing.assert_allclose(log.planned_active, expected_active)
    assert controller_program is ctrl.mpc.program
    native_plan = np.asarray(ctrl.mpc.controls)
    np.testing.assert_allclose(native_plan, log.planned_u)
    states, controls = ctrl.mpc.warm_start.unpack(prob)
    assert states.shape == (4, prob.model.n)
    assert controls.shape == (3, 2)
    np.testing.assert_allclose(log.cost_sparse_effort, 0.2 * np.abs(native_plan).sum() / 3)
    next_u, next_log = ctrl.update(t_ready + 0.01, ref=np.zeros(1), x_hat=np.ones(2))
    assert next_log.success
    assert next_u.shape == (2,)
    assert controller_program is ctrl.mpc.program


def test_epigraph_weight_matches_horizon_normalized_surrogate(tmp_path: Path) -> None:
    """Problem with l1_mode='epigraph' sets slack weight to w_u_l1 / horizon."""
    artifact = _build_checkpoint(tmp_path, n_channels=2, n_controls=2)
    horizon = 5
    w_u_l1 = 0.5
    prob = build_waveform_problem(
        artifact,
        horizon=horizon,
        u_max=1.0,
        w_y=0.0,
        w_u=0.0,
        w_u_l1=w_u_l1,
        l1_mode="epigraph",
    )
    term = next(cost for cost in cost_terms(prob.obj.stage_cost) if isinstance(cost, L1ControlCost))
    np.testing.assert_allclose(float(term.weight), w_u_l1 / horizon)


def test_candidate_logs_cost_on_nonuniform_solver_grid(tmp_path: Path) -> None:
    """Price time-dependent stage and terminal Costs at the solver's knot times."""

    class TimeCost(CostFunction):
        """Price the squared knot time."""

        def evaluate(self, x: jax.Array, u: jax.Array | None = None, t: float | jax.Array = 0.0) -> jax.Array:
            """Return the squared time independently of state and Control Current."""
            del x, u
            return jnp.asarray(t) ** 2

    artifact = _build_checkpoint(tmp_path, n_y=1, n_u=1, depth=0)
    model = WaveformMLPModel.load(artifact)
    objective = Objective(TimeCost(model.n, model.m, terminal=False), TimeCost(model.n, model.m, terminal=True), N=4)
    problem = Problem(model, objective, dt=jnp.array([0.01, 0.02, 0.03]))
    controller = CandidateMPCController(0.01, problem, blocks=[])
    _, log = controller.update(1.0, np.zeros(1), np.zeros(model.n_outputs))

    assert log.success
    expected = np.square([1.0, 1.01, 1.03, 1.06]).sum()
    np.testing.assert_allclose(log.cost, expected, atol=1e-12)
    np.testing.assert_allclose(log.cost_tracking, expected, atol=1e-12)


def test_exact_l1_activation_and_candidate(tmp_path: Path) -> None:
    """Keep activation costs and exact L1 effort separate through both controller paths."""
    artifact, geometry = _build_observable_checkpoint(tmp_path, n_channels=2, n_controls=2, depth=0)
    reference = HealthyReference.load(_write_envelope(tmp_path, geometry))
    problem = build_observable_problem(
        artifact,
        horizon=3,
        u_max=0.5,
        w_u=0.1,
        w_u_l1=0.2,
        w_hinge=1,
        reference=reference,
        kirchhoff=True,
        l1_mode="epigraph",
        w_active=0.0,
    )
    assert problem.continuous_problem is not None
    dt = float(problem.dt[0])
    controller = TrajOptMPCController(dt, problem, init_mode="cont_polish")
    candidate = CandidateMPCController(dt, problem.continuous_problem, blocks=[[0], [1, 2]], w_active=0.0)
    for k in range(controller.model.n_history + 1):
        measurement = np.full(controller.model.p, 2.0)
        _, binary_log = controller.update(k * dt, np.zeros(1), measurement)
        _, candidate_log = candidate.update(k * dt, np.zeros(1), measurement)
        for log, controls in ((binary_log, controller.mpc.controls), (candidate_log, candidate_log.planned_u)):
            if log.warmup:
                continue
            assert log.success, log.status
            assert log.planned_u.shape == (3, 2)
            np.testing.assert_allclose(log.cost_sparse_effort, 0.2 * np.abs(controls).sum() / 3)
            np.testing.assert_allclose(
                log.cost, log.cost_spectral + log.cost_quadratic_effort + log.cost_sparse_effort, atol=1e-10
            )


def test_affine_observable_hinge_and_condensed_mpc_parity() -> None:
    """Compare objective placement and three configured MPC paths across shifted replans."""
    geometry = StftGeometry(n_segment=4, n_hop=1, band_hz=(0.0, 1.0))
    model = ObservableMLPModel(
        geometry=geometry,
        n_y=1,
        n_u=2,
        horizon=3,
        n_channels=1,
        n_controls=2,
        n_outputs=1,
        hidden_size=1,
        depth=0,
        dt=0.01,
        residual=True,
        y_center=np.array([1.2]),
        y_scale=np.array([2.5]),
        u_center=np.array([0.2, -0.4]),
        u_scale=np.array([0.5, 2.0]),
        weights=(np.array([[-1.0, 0.1, 0.0, 0.6, -0.3]]),),
        biases=(np.array([0.17]),),
    )
    reference = HealthyReference(observable=ObservableEnvelope(power=np.array([[1.4]]), fs=100, geometry=geometry))
    config = {
        "dt": model.dt,
        "problem": {
            "class_path": "neuro.control.problems.build_observable_problem",
            "artifact": model,
            "horizon": 3,
            "u_max": [0.5, 0.7],
            "w_u": 0.3,
            "w_hinge": 2.0,
            "kirchhoff": True,
            "reference": reference,
        },
        "solver": {
            "class_path": "trajopt.transcription.single_shooting.SingleShooting",
            "solver": {
                "class_path": "trajopt.transcription.ipopt.Ipopt",
                "options": {"tol": 1e-10, "acceptable_iter": 0},
            },
        },
    }
    incumbent = TrajOptMPCController.from_config(config)
    affine_config = {**config, "problem": {**config["problem"], "affine_hinge": True}}
    affine = TrajOptMPCController.from_config(affine_config)
    with pytest.raises(ValueError, match="affine_hinge=True"):
        TrajOptMPCController(model.dt, incumbent.problem, CondensedOSQP())
    with pytest.raises(ValueError, match="affine_hinge requires"):
        TrajOptMPCController(model.dt, affine.problem, OSQP())
    condensed = TrajOptMPCController.from_config(
        {
            **affine_config,
            "solver": {
                "class_path": "neuro.control.condensed.CondensedOSQP",
                "options": {"eps_abs": 1e-9, "eps_rel": 1e-9},
            },
        }
    )
    assert isinstance(condensed.solver, CondensedOSQP)
    handle = condensed.mpc.program.handles["condensed_osqp"]
    state = jnp.asarray(model.absorb(model.initial_state(), np.array([5.0]), np.array([0.1, -0.1])))
    controls = jnp.array([[-0.3, 0.3], [-0.3, 0.3], [0.5, -0.5]])
    states = _rollout_states(model, state, controls, model.dt)
    frames = np.asarray(model.output(states)).ravel()
    np.testing.assert_array_equal(frames > 1.4, [True, False, False, True])
    physical = decompose_cost(incumbent.problem, states, controls, dt=model.dt)
    rewritten = decompose_cost(affine.problem, states, controls, dt=model.dt)
    for name in ("cost_spectral", "cost_quadratic_effort", "cost_total"):
        np.testing.assert_allclose(rewritten[name], physical[name], atol=1e-12)
    np.testing.assert_allclose(physical["cost_spectral"], 2 / 3 * (frames[-1] - 1.4) ** 2, atol=1e-12)

    for k, measured in enumerate([5.0, 2.0, 0.8, 1.9]):
        t = k * model.dt
        _, expected = incumbent.update(t, np.zeros(1), np.array([measured]))
        for ctrl in (affine, condensed):
            _, actual = ctrl.update(t, np.zeros(1), np.array([measured]))
            if expected.warmup:
                assert actual.warmup
                continue
            assert expected.success, expected.status
            assert actual.success, actual.status
            assert 0.0 <= actual.solve_time < actual.replan_time
            np.testing.assert_allclose(actual.u, expected.u, atol=1e-6)
            np.testing.assert_allclose(actual.planned_u, expected.planned_u, atol=1e-6)
            np.testing.assert_allclose(actual.predicted_y, expected.predicted_y, atol=1e-6)
            np.testing.assert_allclose(actual.planned_u.sum(axis=1), 0.0, atol=1e-8)
            assert np.all(np.abs(actual.planned_u) <= np.array([0.5, 0.7]) + 1e-8)
            np.testing.assert_allclose(actual.cost, expected.cost, atol=1e-8)
            rescore = decompose_cost(incumbent.problem, ctrl.mpc.states, ctrl.mpc.controls, t, model.dt)
            np.testing.assert_allclose(actual.cost_spectral, rescore["cost_spectral"], atol=1e-12)
            np.testing.assert_allclose(actual.cost_quadratic_effort, rescore["cost_quadratic_effort"], atol=1e-12)
            np.testing.assert_allclose(actual.cost, rescore["cost_total"], atol=1e-12)
        assert condensed.mpc.program.handles["condensed_osqp"] is handle


@pytest.mark.parametrize("u_max", [0.5, [0.0, 0.5, 0.5], 0.0])
def test_condensed_osqp_and_bonmin_agree_on_fixed_l1_schedule(tmp_path: Path, u_max: float | list[float]) -> None:
    """Price the same schedule in both epigraphs, including fixed-zero electrodes."""
    artifact, geometry = _build_observable_checkpoint(tmp_path, depth=0, n_controls=3)
    reference = HealthyReference.load(_write_envelope(tmp_path, geometry))
    settings = {
        "horizon": 4,
        "u_max": u_max,
        "kirchhoff": True,
        "w_hinge": 2.0,
        "w_u_l1": 0.1,
        "reference": reference,
    }
    problem = build_observable_problem(artifact, **settings, affine_hinge=True, l1_mode="epigraph")
    solver = CondensedOSQP({"eps_abs": 1e-9, "eps_rel": 1e-9})
    program = Program(problem, solver)
    model = cast("ObservableMLPModel", problem.model)
    state = model.initial_state()
    for _ in range(model.n_history):
        state = model.absorb(state, np.full(model.n_outputs, 1.0), np.zeros(model.m))
    boundary = BoundaryConditions(x0=state, t0=jnp.asarray(0.0))
    seed = WarmStart.cold(problem, state)
    schedule = np.array([True, False, True, True])
    qp = solver.solve(program, boundary, seed, active=schedule)
    assert qp.success, qp.message
    bonmin_problem = build_casadi_observable_problem(
        artifact,
        **settings,
        w_active=0.0,
        solver_options={"bonmin.bb_log_level": 0},
    )
    bonmin = CondensedBonminController(model.dt, bonmin_problem)
    gate_indices = slice(model.m * schedule.size, (model.m + 1) * schedule.size)
    bonmin._lbx[gate_indices] = schedule  # noqa: SLF001 -- fix the same schedule in both solvers
    bonmin._ubx[gate_indices] = schedule  # noqa: SLF001 -- fix the same schedule in both solvers
    mixed = bonmin.solve_state(np.asarray(state), np.asarray(qp.trajectory.U), schedule.astype(float))
    assert mixed.success, mixed.status
    np.testing.assert_array_equal(mixed.planned_active, schedule)
    np.testing.assert_equal(mixed.planned_u[~schedule], 0.0)
    np.testing.assert_equal(mixed.planned_u[:, np.broadcast_to(u_max, (model.m,)) == 0.0], 0.0)
    np.testing.assert_allclose(mixed.planned_u.sum(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(mixed.predicted_y, model.output(qp.trajectory.X), atol=1e-6)
    np.testing.assert_allclose(mixed.cost, qp.cost, atol=1e-6)
    np.testing.assert_allclose(mixed.cost_sparse_effort, 0.1 * np.abs(mixed.planned_u).sum() / 4, atol=1e-12)
    unrestricted = solver.solve(program, boundary, seed)
    assert unrestricted.success, unrestricted.message
    assert unrestricted.cost <= qp.cost + 1e-7


def test_condensed_candidate_search_prices_all_17_holds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Evaluate all 16 block schedules and prefer the cheaper four-hold activation."""
    artifact, geometry = _build_observable_checkpoint(tmp_path, depth=0, horizon=17)
    reference = HealthyReference.load(_write_envelope(tmp_path, geometry))
    problem = build_observable_problem(
        artifact, horizon=17, u_max=0.5, kirchhoff=True, w_hinge=2.0, reference=reference, affine_hinge=True
    )
    model = cast("ObservableMLPModel", problem.model)
    blocks = [list(range(start, stop)) for start, stop in ((0, 4), (4, 8), (8, 12), (12, 17))]
    controller = CandidateMPCController(model.dt, problem, blocks=blocks, w_active=0.51, condensed_osqp={})
    schedules = []

    def price_schedule(*_args: object, active: np.ndarray) -> SimpleNamespace:
        """Make the five-hold QP slightly cheaper before pricing each enabled hold."""
        schedules.append(active.copy())
        enabled = np.flatnonzero(active).tolist()
        if enabled == blocks[0]:
            cost = -1.0
        elif enabled == blocks[-1]:
            cost = -1.02
        else:
            cost = 10.0
        return SimpleNamespace(cost=cost, status=1, success=True, trajectory=SimpleNamespace(U=np.zeros((17, model.m))))

    monkeypatch.setattr(CondensedOSQP, "_solve_at_boundary", price_schedule)
    for tick in range(model.n_history):
        _, log = controller.update(tick * model.dt, np.zeros(1), np.ones(model.n_outputs))
        if log.solve_attempted:
            break
    assert log.success
    assert len(schedules) == 15
    assert len({schedule.tobytes() for schedule in schedules}) == 15
    assert all(schedule.any() for schedule in schedules)
    np.testing.assert_array_equal(np.flatnonzero(log.planned_active), blocks[0])
    assert log.cost_active == pytest.approx(0.51 * 4 / 17)
    assert 0.0 <= log.solve_time < log.replan_time


def test_activation_homotopy_can_lower_the_affine_hinge(tmp_path: Path) -> None:
    """Exercise the existing IPOPT continuation with an affine hinge lifted through JAX-TrajOpt."""
    artifact, geometry = _build_observable_checkpoint(tmp_path, depth=0)
    reference = HealthyReference.load(_write_envelope(tmp_path, geometry))
    problem = build_observable_problem(
        artifact,
        horizon=4,
        u_max=0.5,
        kirchhoff=True,
        w_hinge=2.0,
        w_active=0.1,
        reference=reference,
        affine_hinge=True,
    )
    solver = SingleShooting(solver=Ipopt(options={"print_level": 0, "sb": "yes"}))
    program = Program(problem, solver)
    model = cast("AugmentedActivationModel", problem.model)
    state = model.initial_state()
    for _ in range(model.n_history):
        state = model.absorb(state, np.full(model.n_outputs, 1.0), np.zeros(model.n_controls))
    solution = program.solve(BoundaryConditions(x0=state, t0=jnp.asarray(0.0)), WarmStart.cold(problem, state))
    assert solution.success, solution.message
    scores = decompose_cost(problem, solution.trajectory.X, solution.trajectory.U, dt=model.dt)
    assert scores["cost_spectral"] > 0
    np.testing.assert_allclose(solution.cost, scores["cost_total"], atol=1e-6)


@pytest.mark.parametrize("mode", ["physical", "epigraph", "binary"])
def test_failed_solve_absorbs_applied_zero_current(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    """A failed nonzero iterate remains diagnostic; the next State Absorption receives fallback zero."""
    if mode == "binary":
        artifact, _ = _build_observable_checkpoint(tmp_path, n_y=2, n_u=2)
        problem = build_observable_problem(artifact, horizon=3, u_max=0.5, w_active=0.2)
    else:
        artifact = _build_checkpoint(tmp_path, depth=0, n_y=2, n_u=2)
        problem = build_waveform_problem(
            artifact,
            horizon=3,
            u_max=0.5,
            w_y=0.0,
            w_u_l1=0.2 if mode == "epigraph" else 0.0,
            l1_mode="epigraph" if mode == "epigraph" else "smooth",
        )
    dt = float(problem.dt[0])
    controller = TrajOptMPCController(dt=dt, problem=problem)
    program = controller.mpc.program

    def failed_solve(_program: Program, bc: BoundaryConditions, ws: WarmStart) -> IpoptResult:
        """Return an intentionally failed, nonzero native iterate from the Program seam."""
        failed_u = jnp.full((3, problem.model.m), 0.25)
        failed_ws = ws.with_primal(problem, U=failed_u)
        states, controls = failed_ws.unpack(problem)
        times = bc.t0 + jnp.arange(4) * dt
        return IpoptResult(
            trajectory=Trajectory(X=states, U=controls, t=times, dt=problem.dt),
            success=False,
            status=-1,
            message="forced failure",
            cost=1.0,
            Z=failed_ws.Z,
            info={},
            constraint_violation=0.0,
        )

    monkeypatch.setattr(type(program), "solve", failed_solve)
    n_outputs = controller.model.p
    ready_index = controller.model.n_history - 1
    for k in range(ready_index):
        controller.update(k * dt, np.zeros(1), np.zeros(n_outputs))
    command, failed = controller.update(ready_index * dt, np.zeros(1), np.ones(n_outputs))
    assert not failed.success
    np.testing.assert_array_equal(command, np.zeros(2))
    np.testing.assert_array_equal(failed.planned_u, np.full((3, 2), 0.25))
    absorbed = np.asarray(controller.mpc.x0)
    expected = controller.model.absorb(absorbed, np.full(n_outputs, 2.0), command)
    controller.update((ready_index + 1) * dt, np.zeros(1), np.full(n_outputs, 2.0))
    np.testing.assert_allclose(controller.mpc.x0, expected)
    assert controller.mpc.program is program
