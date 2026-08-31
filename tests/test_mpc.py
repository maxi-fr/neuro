from __future__ import annotations

import itertools
from pathlib import Path
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from simulate.config import load_config as load_sim_config
from simulate.simulation import Simulation
from trajopt.problem import MPCState
from trajopt.solvers.altro import ALTRO
from trajopt.solvers.boxqp import BoxQP
from trajopt.transcription.ipopt import Ipopt
from trajopt.transcription.osqp import OSQP
from trajopt.transcription.single_shooting import SingleShooting
from yaml import safe_load

from neuro.config import StftGeometry
from neuro.control.costs import ObservableHingeCost
from neuro.control.mpc import (
    TrajOptMPCController,
    TrajOptMPCLog,
    _default_solver,
    build_bipolar_waveform_problem,
    build_observable_problem,
    build_waveform_problem,
)
from neuro.filtering import ObservableEstimator
from neuro.predictor.inference import ObservableMLPModel, WaveformMLPModel
from neuro.predictor.module import AutoregressiveMLP
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from neuro.types import FloatArray

_SEED = 7

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
        # The golden control/cost values were pinned from the incumbent CasADi controller on the
        # plain (non-residual) semantics, so the parity artifacts keep the skip off; the residual
        # path is pinned separately by the cross-side parity tests.
        residual=False,
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
                "class_path": "neuro.control.mpc.build_waveform_problem",
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

    n_z = controller.model.n_y * controller.model.n_channels
    results = _drive(controller, n_steps=6, n_channels=controller.model.n_channels)
    u_last, log_last = results[-1]
    assert not log_last.warmup
    assert not np.isnan(controller._state[:n_z]).any()  # noqa: SLF001 -- the test inspects the absorbed state it owns
    np.testing.assert_array_equal(controller._u_last, u_last)  # noqa: SLF001 -- the test verifies the private u_last persistence
    seed = controller.state.x0
    assert not np.array_equal(np.asarray(seed), controller._state)  # noqa: SLF001 -- absorbed state vs post-solve seed


def test_reproduces_mpc_controller_control_sequence(tmp_path: Path) -> None:
    """The trajopt controller reproduces the incumbent CasADi control sequence and reported cost.

    The golden values are pinned from the incumbent ``MPCController`` (``solver="ipopt"``) at
    fd0d244 on this same depth-0 (linear, hence convex) checkpoint and measurement trajectory.
    Both the controls and the per-step reported cost must match: the cost assertion is what
    catches the absorbed-measurement term the incumbent graph never scores.
    """
    artifact = _build_checkpoint(tmp_path, depth=0)
    problem = build_waveform_problem(artifact, horizon=3, u_max=0.5, w_y=1.0, w_u=0.0, kirchhoff=True)
    controller = TrajOptMPCController(dt=0.01, problem=problem, solver=_full_parity_solver())

    controls, costs = _drive_golden(controller, n_steps=8, n_channels=controller.model.n_channels)
    np.testing.assert_allclose(controls, _WAVEFORM_PARITY_CONTROLS, atol=1e-4)
    np.testing.assert_allclose(costs, _WAVEFORM_PARITY_COSTS, atol=1e-4)


def test_migrated_config_reproduces_incumbent_end_to_end(tmp_path: Path) -> None:
    """The migrated YAML, dispatched through ``from_config``, reproduces the incumbent sequence and cost.

    Loads ``mse02_psd_mpc.yaml`` and swaps in the synthetic checkpoint and the golden weights,
    keeping the config's ``kirchhoff: true``; the controller is then built through the config's
    class-path dispatch rather than direct instantiation.
    """
    with Path("configs/simulation/mse02_psd_mpc.yaml").open() as file:
        sim_config = safe_load(file)
    controller_cfg = sim_config["controller"]
    problem_cfg = {
        **controller_cfg["problem"],
        "artifact": str(_build_checkpoint(tmp_path, depth=0)),
        "horizon": 3,
        "u_max": 0.5,
        "w_y": 1.0,
        "w_u": 0.0,
    }
    controller = TrajOptMPCController.from_config(
        {"dt": controller_cfg["dt"], "problem": problem_cfg, "solver": _full_parity_solver()}
    )

    controls, costs = _drive_golden(controller, n_steps=8, n_channels=controller.model.n_channels)
    np.testing.assert_allclose(controls, _WAVEFORM_PARITY_CONTROLS, atol=1e-4)
    np.testing.assert_allclose(costs, _WAVEFORM_PARITY_COSTS, atol=1e-4)


def test_migrated_config_reproduces_incumbent_with_default_solver(tmp_path: Path) -> None:
    """A migrated ``kirchhoff: true`` config runs through ``from_config`` with no injected solver."""
    with Path("configs/simulation/mse02_psd_mpc.yaml").open() as file:
        sim_config = safe_load(file)
    controller_cfg = sim_config["controller"]
    problem_cfg = {
        **controller_cfg["problem"],
        "artifact": str(_build_checkpoint(tmp_path, depth=0)),
        "horizon": 3,
        "u_max": 0.5,
        "w_y": 1.0,
        "w_u": 0.0,
    }
    controller = TrajOptMPCController.from_config({"dt": controller_cfg["dt"], "problem": problem_cfg})
    assert type(controller.solver) is SingleShooting

    controls, costs = _drive_golden(controller, n_steps=8, n_channels=controller.model.n_channels)
    np.testing.assert_allclose(controls, _WAVEFORM_PARITY_CONTROLS, atol=1e-4)
    np.testing.assert_allclose(costs, _WAVEFORM_PARITY_COSTS, atol=1e-4)


def test_single_shooting_solver_succeeds_when_kirchhoff(tmp_path: Path) -> None:
    """An injected single-shooting solver on a Kirchhoff problem constructs and solves without error."""
    artifact = _build_checkpoint(tmp_path, depth=0)
    problem = build_waveform_problem(artifact, horizon=3, u_max=0.5, w_y=1.0, kirchhoff=True)
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
    model = AutoregressiveMLP(
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
    linears = [m for m in model.layers if isinstance(m, torch.nn.Linear)]
    sizes = [in_size, *([5] * depth), n_outputs]
    with torch.no_grad():
        for lin, (w, b) in zip(linears, _random_layers(rng, sizes), strict=True):
            lin.weight.copy_(torch.as_tensor(w, dtype=torch.float32))
            lin.bias.copy_(torch.as_tensor(b, dtype=torch.float32))
    path = tmp_path / "obs_art"
    model.save(path)
    return path, geom


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
        envelope_ref=env_path,
        kirchhoff=True,
    )
    assert problem.N == 5
    # The stage trajectory carries every Frame of the Control Horizon but the last; the terminal
    # Cost prices that one, so no predicted Frame the controls move goes unscored.
    assert isinstance(problem.obj.terminal_cost, ObservableHingeCost)
    assert problem.obj.terminal_cost.terminal
    assert isinstance(problem.model, ObservableMLPModel)
    assert problem.model.n_outputs == 2 * n_values
    assert problem.model.m == 2

    # Solve from a valid ready state
    rng = np.random.default_rng(_SEED + 2)
    model = ObservableMLPModel.load(artifact)
    x0 = np.asarray(model.initial_state())
    x0[: model.n_y * model.n_outputs] = rng.uniform(-1.0, 1.0, model.n_y * model.n_outputs)
    state = MPCState.initial(problem, x0=jnp.asarray(x0), dt=model.dt)
    # Ipopt's 1e-8 default is below the noise floor of a float32 objective whose optimum sits
    # inside L1ControlCost's eps=1e-3 smoothing radius, where curvature is 1/eps; the solve
    # stalls there at a dual infeasibility of ~1e-2 with the objective already flat to 1e-6.
    solver = Ipopt(options={"print_level": 0, "hessian_approximation": "limited-memory", "tol": 1e-3})
    solved = problem.solve(state, solver=solver)
    assert solved.status == "converged"

    controls = np.asarray(solved.controls)
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
        envelope_ref=env_path,
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
    """Observable controller instantiates from config and routes problem factory."""
    artifact, geom = _build_observable_checkpoint(tmp_path)
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

    cfg = {
        "dt": 0.10,
        "problem": {
            "class_path": "neuro.control.mpc.build_observable_problem",
            "artifact": str(artifact),
            "horizon": 4,
            "u_max": 1.0,
            "w_u": 5.0,
            "w_hinge": 2.0,
            "envelope_ref": str(env_path),
            "kirchhoff": True,
        },
    }
    controller = TrajOptMPCController.from_config(cfg)
    assert controller.dt == 0.10
    assert controller.problem.N == 5
    assert controller.model.m == 2


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
        build_observable_problem(artifact, horizon=4, u_max=0.5, w_hinge=1.0, envelope_ref=bad_ch)

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
        build_observable_problem(artifact, horizon=4, u_max=0.5, w_hinge=1.0, envelope_ref=bad_fs)

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
        build_observable_problem(artifact, horizon=4, u_max=0.5, w_hinge=1.0, envelope_ref=bad_band)


def test_example_observable_config_runs_simulation_start_to_finish(tmp_path: Path) -> None:
    """An example config with build_observable_problem runs the loop start to finish."""
    sim_dict = load_sim_config(Path("configs/simulation/observable_psd_mpc.yaml"))
    geom = StftGeometry(n_segment=50, n_hop=25, kernel="boxcar", kernel_width=1)
    artifact, _ = _build_observable_checkpoint(
        tmp_path, n_y=2, n_u=2, horizon=4, n_channels=62, n_controls=3, geom=geom
    )
    env_path = tmp_path / "healthy_psd.npz"
    np.savez_compressed(
        env_path,
        Pref_frames=np.full((62, geom.n_values(50.0)), -2.0),
        fs=50.0,
        n_segment=geom.n_segment,
        n_hop=geom.n_hop,
        band_hz=np.asarray(geom.band_hz if geom.band_hz is not None else [-1.0, -1.0]),
        n_bin_pool=geom.n_bin_pool,
        kernel=geom.kernel,
        kernel_width=geom.kernel_width,
    )
    sim_dict["t_end"] = 0.5
    sim_dict["controller"]["problem"]["artifact"] = str(artifact)
    sim_dict["controller"]["problem"]["envelope_ref"] = str(env_path)
    sim_dict["estimator"]["geometry"] = geom.model_dump()
    sim_dict["estimator"]["downsample"] = 200

    sim = Simulation.from_config(sim_dict)
    sim.run(output_dir=tmp_path / "sim_out", use_mmap=True)
    assert sim.logger is not None
    us = sim.logger.signal("controller", "u")
    assert us.shape[0] > 0
    assert np.isfinite(us).all()


def test_default_solver_selection(tmp_path: Path) -> None:
    """_default_solver chooses BoxQP for bipolar and SingleShooting for general problems."""
    art_wf = _build_checkpoint(tmp_path / "wf", depth=0, n_y=3, n_u=2, horizon=4, n_channels=2, n_controls=2)

    # 1. Bipolar reduced waveform OCP -> BoxQP
    prob_bip = build_bipolar_waveform_problem(art_wf, horizon=4, u_max=0.5, w_y=1.0)
    assert isinstance(_default_solver(prob_bip), BoxQP)

    # 2. Standard waveform OCP -> SingleShooting(Ipopt)
    prob_wf = build_waveform_problem(art_wf, horizon=4, u_max=0.5, w_y=1.0, kirchhoff=True)
    assert isinstance(_default_solver(prob_wf), SingleShooting)

    # 3. Whole-horizon PSD cost -> SingleShooting(Ipopt)
    psd_ref = tmp_path / "psd_ref.npz"
    np.savez_compressed(psd_ref, Pref=np.full((2, 3), -2.0), fs=100.0, L=4, R=2)
    prob_psd = build_waveform_problem(art_wf, horizon=4, u_max=0.5, w_psd=1.0, psd_ref=psd_ref)
    default_psd = _default_solver(prob_psd)
    assert isinstance(default_psd, SingleShooting)
    assert isinstance(default_psd.solver, Ipopt)


def test_controller_from_config_solver_variants(tmp_path: Path) -> None:
    """TrajOptMPCController.from_config dispatches class_path solver configs across backends."""
    art = _build_checkpoint(tmp_path, depth=0, n_y=3, n_u=2, horizon=4, n_channels=2, n_controls=2)

    # 1. Default (omitted solver on waveform model -> SingleShooting)
    cfg_default = {
        "dt": 0.01,
        "problem": {
            "class_path": "neuro.control.mpc.build_waveform_problem",
            "artifact": str(art),
            "horizon": 4,
            "u_max": 0.5,
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
            "class_path": "neuro.control.mpc.build_waveform_problem",
            "artifact": str(art),
            "horizon": 4,
            "u_max": 0.5,
            "kirchhoff": True,
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
            "class_path": "neuro.control.mpc.build_waveform_problem",
            "artifact": str(art),
            "horizon": 4,
            "u_max": 0.5,
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
            "class_path": "neuro.control.mpc.build_waveform_problem",
            "artifact": str(art),
            "horizon": 4,
            "u_max": 0.5,
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
            "class_path": "neuro.control.mpc.build_waveform_problem",
            "artifact": str(art),
            "horizon": 4,
            "u_max": 0.5,
        },
    }
    with pytest.raises(ValueError, match="solver config must contain 'class_path'"):
        TrajOptMPCController.from_config(cfg_missing)

    # 2. String instead of dict raises TypeError
    cfg_str = {
        "dt": 0.01,
        "solver": "altro",
        "problem": {
            "class_path": "neuro.control.mpc.build_waveform_problem",
            "artifact": str(art),
            "horizon": 4,
            "u_max": 0.5,
        },
    }
    with pytest.raises(TypeError, match="must be a dict with 'class_path'"):
        TrajOptMPCController.from_config(cfg_str)

    # 3. Expansion solver on whole-horizon PSD cost raises ValueError
    psd_ref = tmp_path / "psd_ref.npz"
    np.savez_compressed(psd_ref, Pref=np.full((2, 3), -2.0), fs=100.0, L=4, R=2)
    cfg_incompatible = {
        "dt": 0.01,
        "solver": {"class_path": "trajopt.solvers.altro.ALTRO"},
        "problem": {
            "class_path": "neuro.control.mpc.build_waveform_problem",
            "artifact": str(art),
            "horizon": 4,
            "u_max": 0.5,
            "w_psd": 1.0,
            "psd_ref": str(psd_ref),
        },
    }
    with pytest.raises(ValueError, match="cannot score the whole-horizon hinge cost"):
        TrajOptMPCController.from_config(cfg_incompatible)

    # 4. BoxQP on unhandled coupled linear equality constraints raises ValueError
    cfg_boxqp_coupled = {
        "dt": 0.01,
        "solver": {"class_path": "trajopt.solvers.boxqp.BoxQP"},
        "problem": {
            "class_path": "neuro.control.mpc.build_waveform_problem",
            "artifact": str(art),
            "horizon": 4,
            "u_max": 0.5,
            "kirchhoff": True,
        },
    }
    with pytest.raises(ValueError, match="BoxQP only supports uncoupled box bounds"):
        TrajOptMPCController.from_config(cfg_boxqp_coupled)
