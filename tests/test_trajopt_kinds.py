from __future__ import annotations

import itertools
from pathlib import Path
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from _predictor_reference import esn_prime
from trajopt.problem import MPCState
from trajopt.solvers.altro import ALTRO
from trajopt.transcription.ipopt import Ipopt
from yaml import safe_load

from neuro.checkpoint import ESNCheckpoint, load_esn
from neuro.config import StftGeometry
from neuro.control.trajopt_costs import ESNAutoRegressiveCost
from neuro.control.trajopt_mpc import (
    ESNModel,
    ObservableModel,
    TrajOptMPCController,
    TrajOptMPCLog,
    build_esn_problem,
    build_observable_problem,
)
from neuro.esn import generate_reservoir
from neuro.observable import control_means
from neuro.predictor.esn_module import ESNModule
from neuro.predictor.module import AutoregressiveMLP
from neuro.predictor.observable_module import StepwiseObservableMLP
from neuro.spectral import compute_periodograms
from neuro.transforms import Standardizer

if TYPE_CHECKING:
    from collections.abc import Callable

    from neuro.checkpoint import ObservableCheckpoint
    from neuro.types import FloatArray

_SEED = 7
_N_RES, _N_EEG, _N_CONTROLS = 30, 2, 2
_LEAK, _RHO, _DENSITY, _IN_SCALE = 0.1, 0.9, 0.2, 0.1
_PRIMING, _HORIZON = 4, 3
_L, _R, _FS = 8, 4, 50.0
# float32 torch runtime vs float64 checkpoint weights: the reservoir recurrence accumulates
# rounding, but on these fixtures it stays inside ticket 01's waveform bar (measured ~2e-8).


def _esn_checkpoint(tmp_path: Path) -> Path:
    """Save a tiny synthetic ESN checkpoint and return its suffix-less stem."""
    rng = np.random.default_rng(_SEED)
    w_res, w_in = generate_reservoir(
        reservoir_size=_N_RES,
        spectral_radius=_RHO,
        density=_DENSITY,
        input_scaling=_IN_SCALE,
        in_dim=_N_EEG + _N_CONTROLS + 1,
        seed=_SEED,
    )
    y_std = Standardizer(center=rng.uniform(-1.0, 1.0, _N_EEG), scale=rng.uniform(0.5, 2.0, _N_EEG))
    u_std = Standardizer(center=rng.uniform(-1.0, 1.0, _N_CONTROLS), scale=rng.uniform(0.5, 2.0, _N_CONTROLS))
    checkpoint = tmp_path / "esn_art"
    ESNCheckpoint(
        w_in=w_in,
        w_out=rng.uniform(-0.1, 0.1, size=(_N_EEG, _N_RES + 1)),
        w_res=w_res,
        dt=0.01,
        downsample=1,
        horizon=_HORIZON,
        reservoir_size=_N_RES,
        leak_rate=_LEAK,
        spectral_radius=_RHO,
        priming_steps=_PRIMING,
        input_scaling=_IN_SCALE,
        density=_DENSITY,
        noise_sigma=0.0,
        ridge_lambda=1e-3,
        seed=_SEED,
        y_std=y_std,
        u_std=u_std,
    ).save(checkpoint)
    return checkpoint


def _drive(controller: TrajOptMPCController, n_steps: int, n_channels: int) -> list[tuple[FloatArray, TrajOptMPCLog]]:
    """Feed ``n_steps`` random EEG measurements through ``update`` and collect the outputs."""
    rng = np.random.default_rng(_SEED + 4)
    out = []
    for k in range(n_steps):
        u, log = controller.update(k * controller.dt, ref=np.array([0.0]), x_hat=rng.standard_normal(n_channels))
        out.append((np.atleast_1d(np.asarray(u, dtype=np.float64)), log))
    return out


def _envelope_npz(tmp_path: Path, power: FloatArray, *, window: int = _L, hop: int = _R) -> Path:
    """Write a healthy-envelope npz in the layout ``scripts/build_healthy_psd.py`` produces."""
    path = tmp_path / "healthy.npz"
    np.savez(
        path,
        Pref=power,
        freqs=np.fft.rfftfreq(window, 1.0 / _FS),
        fs=_FS,
        L=window,
        R=hop,
        quantile=0.9,
        n_windows=10,
        plant_fingerprint="test",
    )
    return path


def _full_parity_solver() -> Ipopt:
    """The general Ipopt transcription with L-BFGS and acceptable-level termination."""
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


def test_esn_discrete_dynamics_matches_torch_step(tmp_path: Path) -> None:
    """One adapter step reproduces the torch module's ``step`` state advance and output."""
    artifact = _esn_checkpoint(tmp_path)
    adapter = ESNModel.from_checkpoint(artifact)
    module = ESNModule.load(artifact)

    rng = np.random.default_rng(_SEED + 1)
    y_ctx = rng.standard_normal((_PRIMING, _N_EEG))
    u_ctx = rng.standard_normal((_PRIMING, _N_CONTROLS))
    u = u_ctx[-1]
    state = module.prime(y_ctx, u_ctx)
    state_next, y_want = module.step(state, u)
    x = jnp.asarray(state)
    x_next = np.asarray(adapter.discrete_dynamics(x, jnp.asarray(u), 0.0, 0.01))
    np.testing.assert_allclose(x_next, state_next, rtol=1e-5, atol=1e-6)
    # The emitted output is the readout of the *pre*-step reservoir on both sides.
    np.testing.assert_allclose(np.asarray(adapter.output(x)), y_want, rtol=1e-5, atol=1e-6)


def test_esn_rollout_matches_torch_rollout(tmp_path: Path) -> None:
    """Chaining adapter steps reproduces the torch module's rollout on the same trajectory.

    The adapter emits the readout of the pre-step reservoir exactly as the module's ``rollout``
    does, so the chained outputs equal the module's predictions one-for-one.
    """
    artifact = _esn_checkpoint(tmp_path)
    adapter = ESNModel.from_checkpoint(artifact)
    module = ESNModule.load(artifact)

    rng = np.random.default_rng(_SEED + 2)
    y_ctx = rng.standard_normal((_PRIMING + 2, _N_EEG))
    u_ctx = rng.standard_normal((_PRIMING + 2, _N_CONTROLS))
    u_future = rng.standard_normal((6, _N_CONTROLS))

    state = module.prime(y_ctx, u_ctx)
    want = module.rollout(state, u_future)
    x = jnp.asarray(state)
    got = []
    for u in u_future:
        got.append(np.asarray(adapter.output(x)))
        x = adapter.discrete_dynamics(x, jnp.asarray(u), 0.0, 0.01)
    np.testing.assert_allclose(np.stack(got), want, rtol=1e-5, atol=1e-6)


def test_esn_absorb_is_ready_initial_state_match_torch_module(tmp_path: Path) -> None:
    """The adapter's priming seam (initial_state/absorb/is_ready) matches the torch module's."""
    artifact = _esn_checkpoint(tmp_path)
    adapter = ESNModel.from_checkpoint(artifact)
    module = ESNModule.load(artifact)
    ckpt = load_esn(artifact)

    rng = np.random.default_rng(_SEED + 3)
    state = adapter.initial_state()
    module_state = module.initial_state()
    np.testing.assert_array_equal(state, module_state)
    assert not adapter.is_ready(state)
    assert not module.is_ready(module_state)

    y_seq = rng.standard_normal((_PRIMING, _N_EEG))
    u_seq = rng.standard_normal((_PRIMING, _N_CONTROLS))
    for t in range(_PRIMING):
        state = adapter.absorb(state, y_seq[t], u_seq[t])
        module_state = module.absorb(module_state, y_seq[t], u_seq[t])
        assert adapter.is_ready(state) == module.is_ready(module_state)
    np.testing.assert_allclose(state, module_state, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(state[:_N_RES], esn_prime(ckpt, y_seq, u_seq), rtol=1e-5, atol=1e-6)
    assert adapter.is_ready(state)


def test_observable_rollout_matches_torch_rollout(
    tmp_path: Path, make_observable_checkpoint: Callable[..., ObservableCheckpoint]
) -> None:
    """Chaining adapter Frame steps reproduces the torch module's rollout on the same trajectory.

    The module's ``rollout`` aggregates raw future controls into Frame means; the adapter steps
    one Frame per call, so the test aggregates the same way and compares the decoded Frames
    one-for-one.
    """
    ckpt = make_observable_checkpoint(StftGeometry(n_segment=_L, n_hop=_R), horizon=16, n_y=4, n_u=3, dt=0.02)
    artifact = tmp_path / "observable"
    ckpt.save(artifact)
    adapter = ObservableModel.from_checkpoint(artifact)
    module = StepwiseObservableMLP.load(artifact)

    rng = np.random.default_rng(_SEED + 5)
    horizon = 16
    y_ctx = rng.standard_normal((8, ckpt.n_channels))
    u_ctx = rng.standard_normal((8, ckpt.n_controls))
    u_future = rng.standard_normal((horizon, ckpt.n_controls))

    state = module.prime(y_ctx, u_ctx)
    want = module.rollout(state, u_future)
    u_bar = control_means(ckpt.geometry, horizon, ckpt.fs) @ u_future
    x = jnp.asarray(state)
    got = []
    for u in u_bar:
        x = adapter.discrete_dynamics(x, jnp.asarray(u), 0.0, 0.0)
        got.append(np.asarray(adapter.output(x)))
    np.testing.assert_allclose(np.stack(got), want, rtol=1e-5, atol=1e-6)


def test_observable_absorb_is_ready_initial_state_match_torch_module(
    tmp_path: Path, make_observable_checkpoint: Callable[..., ObservableCheckpoint]
) -> None:
    """The adapter's priming seam matches the torch module's shift-register discipline."""
    ckpt = make_observable_checkpoint(StftGeometry(n_segment=_L, n_hop=_R), horizon=16, n_y=4, n_u=3, dt=0.02)
    artifact = tmp_path / "observable"
    ckpt.save(artifact)
    adapter = ObservableModel.from_checkpoint(artifact)
    module = StepwiseObservableMLP.load(artifact)

    rng = np.random.default_rng(_SEED + 6)
    state = adapter.initial_state()
    module_state = module.initial_state()
    np.testing.assert_array_equal(state, module_state)
    assert not adapter.is_ready(state)
    assert not module.is_ready(module_state)

    y_seq = rng.standard_normal((4, ckpt.n_channels))
    u_seq = rng.standard_normal((4, ckpt.n_controls))
    for t in range(4):
        state = adapter.absorb(state, y_seq[t], u_seq[t])
        module_state = module.absorb(module_state, y_seq[t], u_seq[t])
        assert adapter.is_ready(state) == module.is_ready(module_state)
    # The register part is float64 on both sides and matches exactly; the lifted Frame state
    # differs by the module's float32 lift, so the whole state compares to float32 tolerance.
    np.testing.assert_allclose(state, module_state, rtol=1e-5, atol=1e-6)
    assert adapter.is_ready(state)


def test_esn_problem_drives_controller(tmp_path: Path) -> None:
    """The ESN problem runs through the ticket-01/02 controller: warmup, then bounded active steps."""
    artifact = _esn_checkpoint(tmp_path)
    u_max = 0.5
    problem = build_esn_problem(artifact, horizon=_HORIZON, u_max=u_max, w_y=1.0, w_u=0.1)
    controller = TrajOptMPCController(dt=0.01, problem=problem)

    results = _drive(controller, n_steps=_PRIMING + 3, n_channels=_N_EEG)
    for u, log in results[: _PRIMING - 1]:
        assert log.warmup
        np.testing.assert_array_equal(u, np.zeros(_N_CONTROLS))
    for u, log in results[_PRIMING - 1 :]:
        assert not log.warmup
        assert log.success
        assert u.shape == (_N_CONTROLS,)
        assert np.isfinite(u).all()
        assert np.all(np.abs(u) <= u_max + 1e-6)


def test_observable_problem_drives_controller(
    tmp_path: Path, make_observable_checkpoint: Callable[..., ObservableCheckpoint]
) -> None:
    """The observable problem runs through the ticket-01/02 controller on the Frame grid."""
    ckpt = make_observable_checkpoint(StftGeometry(n_segment=_L, n_hop=_R), horizon=16, n_y=4, dt=0.02)
    artifact = tmp_path / "observable"
    ckpt.save(artifact)
    envelope = _envelope_npz(tmp_path, np.full((ckpt.n_channels, _L // 2 + 1), 1e-3))
    u_max = 2.0
    problem = build_observable_problem(artifact, horizon=16, u_max=u_max, w_u=0.01, w_psd=10.0, psd_ref=envelope)
    controller = TrajOptMPCController(dt=ckpt.dt, problem=problem)

    results = _drive(controller, n_steps=8, n_channels=ckpt.n_channels)
    for u, log in results[:3]:
        assert log.warmup
        np.testing.assert_array_equal(u, np.zeros(ckpt.n_controls))
    for u, log in results[3:]:
        assert not log.warmup
        assert log.success
        assert u.shape == (ckpt.n_controls,)
        assert np.isfinite(u).all()
        assert np.all(np.abs(u) <= u_max + 1e-6)


def test_esn_from_config_dispatches_factory(tmp_path: Path) -> None:
    """from_config routes the ESN problem through the ``{class_path, ...}`` factory pattern."""
    artifact = _esn_checkpoint(tmp_path)
    controller = TrajOptMPCController.from_config(
        {
            "dt": 0.01,
            "problem": {
                "class_path": "neuro.control.trajopt_mpc.build_esn_problem",
                "artifact": str(artifact),
                "horizon": _HORIZON,
                "u_max": 0.5,
                "w_y": 1.0,
            },
        }
    )
    assert controller.dt == 0.01
    assert controller.problem.N == _HORIZON + 1
    assert isinstance(controller.model, ESNModel)
    assert controller.model.m == _N_CONTROLS


def test_observable_from_config_dispatches_factory(
    tmp_path: Path, make_observable_checkpoint: Callable[..., ObservableCheckpoint]
) -> None:
    """from_config routes the observable problem through the factory, defaulting no horizon."""
    ckpt = make_observable_checkpoint(StftGeometry(n_segment=_L, n_hop=_R), horizon=16, n_y=4, dt=0.02)
    artifact = tmp_path / "observable"
    ckpt.save(artifact)
    envelope = _envelope_npz(tmp_path, np.full((ckpt.n_channels, _L // 2 + 1), 1e-3))

    controller = TrajOptMPCController.from_config(
        {
            "dt": ckpt.dt,
            "problem": {
                "class_path": "neuro.control.trajopt_mpc.build_observable_problem",
                "artifact": str(artifact),
                "horizon": 16,
                "u_max": 2.0,
                "w_u": 1.0,
                "w_psd": 10.0,
                "psd_ref": str(envelope),
            },
        }
    )
    assert controller.dt == ckpt.dt
    assert isinstance(controller.model, ObservableModel)
    assert ckpt.n_frames(16) + 1 == controller.problem.N


@pytest.mark.parametrize(
    ("kind", "factory", "weights"),
    [
        ("mlp", "build_waveform_problem", {"w_y": 1.0, "w_u": 10.0}),
        ("esn", "build_esn_problem", {"w_y": 1.0, "w_u": 10.0}),
        (
            "observable",
            "build_observable_problem",
            {"w_y": 0.0, "w_u": 10.0, "w_psd": 1000.0, "psd_ref": None},
        ),
    ],
    ids=["waveform", "esn", "observable"],
)
def test_migrated_yaml_controller_block_dispatches_each_kind(
    tmp_path: Path,
    make_observable_checkpoint: Callable[..., ObservableCheckpoint],
    kind: str,
    factory: str,
    weights: dict[str, object],
) -> None:
    """The checked-in ``mse02_psd_mpc.yaml`` controller block dispatches each predictor kind.

    The config carries the migrated layout: ``class_path`` at ``TrajOptMPCController`` and the
    cost-weight fields (``artifact``, ``horizon``, ``u_max``, ``w_y``, ``w_u``) nested under
    ``problem`` with the per-kind factory. Swapping in a synthetic checkpoint for each kind and
    adding the observable-only weights drives it through ``from_config`` directly.
    """
    with Path("configs/simulation/mse02_psd_mpc.yaml").open() as file:
        sim_config = safe_load(file)
    controller_cfg = sim_config["controller"]
    assert controller_cfg["class_path"] == "neuro.control.trajopt_mpc.TrajOptMPCController"
    assert controller_cfg["problem"]["class_path"] == "neuro.control.trajopt_mpc.build_waveform_problem"

    if kind == "mlp":
        artifact = _mlp_checkpoint(tmp_path)
    elif kind == "esn":
        artifact = _esn_checkpoint(tmp_path)
    else:
        ckpt = make_observable_checkpoint(StftGeometry(n_segment=_L, n_hop=_R), horizon=50, n_y=4, dt=0.02)
        artifact = tmp_path / "observable"
        ckpt.save(artifact)
        weights["psd_ref"] = str(_envelope_npz(tmp_path, np.full((ckpt.n_channels, _L // 2 + 1), 1e-3)))

    problem_cfg = {
        **controller_cfg["problem"],
        "class_path": f"neuro.control.trajopt_mpc.{factory}",
        "artifact": str(artifact),
        **weights,
    }
    controller = TrajOptMPCController.from_config({"dt": controller_cfg["dt"], "problem": problem_cfg})
    assert controller.dt == controller_cfg["dt"]
    u, log = controller.update(0.0, np.array([0.0]), np.zeros(controller.model.n_channels))
    assert u.shape == (controller.model.m,)
    assert log.warmup  # nothing absorbed yet


def _mlp_checkpoint(tmp_path: Path) -> Path:
    """Save a tiny synthetic depth-0 MLP checkpoint and return its suffix-less stem."""
    rng = np.random.default_rng(_SEED)
    n_y, n_u, n_channels, n_controls = 4, 3, 2, 2
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
        horizon=3,
        n_channels=n_channels,
        n_controls=n_controls,
        hidden_size=5,
        depth=0,
        activation="relu",
        dt=0.01,
        y_std=Standardizer(center=scalers["y_mean"], scale=scalers["y_scale"]),
        u_std=Standardizer(center=scalers["u_mean"], scale=scalers["u_scale"]),
    )
    linears = [m for m in model.layers if isinstance(m, torch.nn.Linear)]
    sizes = [in_size, n_channels]
    with torch.no_grad():
        for lin, (w, b) in zip(linears, _random_layers(rng, sizes), strict=True):
            lin.weight.copy_(torch.as_tensor(w, dtype=torch.float32))
            lin.bias.copy_(torch.as_tensor(b, dtype=torch.float32))
    path = tmp_path / "mlp_art"
    model.save(path)
    return path


def _random_layers(rng: np.random.Generator, sizes: list[int]) -> tuple[tuple[FloatArray, FloatArray], ...]:
    """Random ``(weight (out, in), bias (out,))`` pairs, drawn uniformly from ``+-1/sqrt(fan_in)``."""
    return tuple(
        (rng.uniform(-1.0, 1.0, (out, inp)) / np.sqrt(inp), rng.uniform(-1.0, 1.0, out) / np.sqrt(inp))
        for inp, out in itertools.pairwise(sizes)
    )


def test_esn_full_cost_set_solves_with_kirchhoff(tmp_path: Path) -> None:
    """The full ESN cost set (spectral hinge + smooth L1 + Kirchhoff) assembles and solves."""
    horizon, length, hop, fs = 6, 4, 2, 50.0
    artifact = _esn_checkpoint(tmp_path)
    probe = ESNModel.from_checkpoint(artifact)
    rng = np.random.default_rng(_SEED + 10)
    x0 = np.concatenate([rng.standard_normal(_N_RES), [0.0]])
    x = jnp.asarray(x0)
    y_traj = []
    for _ in range(horizon):
        x = probe.discrete_dynamics(x, jnp.zeros(probe.m), 0.0, 0.01)
        y_traj.append(np.asarray(probe.output(x)))
    windows = compute_periodograms(np.array(y_traj), fs=fs, window=length, hop=hop)
    env_path = _envelope_npz(tmp_path, np.median(windows, axis=0), window=length, hop=hop)

    problem = build_esn_problem(
        artifact,
        horizon=horizon,
        u_max=0.8,
        w_y=1.0,
        w_u=0.05,
        w_u_l1=0.5,
        w_psd=50.0,
        psd_ref=env_path,
        kirchhoff=True,
    )
    state = MPCState.initial(problem, x0=jnp.asarray(x0), dt=0.01)
    solved = problem.solve(state, solver=_full_parity_solver())
    assert solved.status == "converged"
    np.testing.assert_allclose(np.sum(np.asarray(solved.controls), axis=1), np.zeros(horizon), atol=1e-6)


def test_esn_autoregressive_cost_matches_incumbent_formula(tmp_path: Path) -> None:
    """The ESN stage cost reproduces ``w_y * ||f_out(x)||^2 / H + w_u * ||u||^2 / H`` on fixed inputs."""
    artifact = _esn_checkpoint(tmp_path)
    probe = ESNModel.from_checkpoint(artifact)
    rng = np.random.default_rng(_SEED + 11)
    x = jnp.asarray(np.concatenate([rng.standard_normal(_N_RES), [3.0]]))
    u = jnp.asarray(rng.standard_normal(_N_CONTROLS))
    w_y, w_u, horizon = 1.5, 0.25, 4

    expected = w_y * np.sum(np.asarray(probe.output(x)) ** 2) / horizon + w_u * np.sum(np.asarray(u) ** 2) / horizon
    cost = ESNAutoRegressiveCost(model=probe, w_y=w_y, w_u=w_u, horizon=horizon)
    np.testing.assert_allclose(float(cost.evaluate(x, u)), expected, rtol=1e-12, atol=1e-12)
    terminal = ESNAutoRegressiveCost(model=probe, w_y=w_y, w_u=0.0, horizon=horizon, terminal=True)
    np.testing.assert_allclose(
        float(terminal.evaluate(x, None)), w_y * np.sum(np.asarray(probe.output(x)) ** 2) / horizon
    )


def test_observable_problem_rejects_w_y_and_requires_psd(
    tmp_path: Path, make_observable_checkpoint: Callable[..., ObservableCheckpoint]
) -> None:
    """The observable path rejects ``w_y`` and requires ``w_psd`` + ``psd_ref``, like the incumbent."""
    ckpt = make_observable_checkpoint(StftGeometry(n_segment=_L, n_hop=_R), horizon=16, n_y=4, dt=0.02)
    artifact = tmp_path / "observable"
    ckpt.save(artifact)
    envelope = _envelope_npz(tmp_path, np.full((ckpt.n_channels, _L // 2 + 1), 1e-3))

    with pytest.raises(ValueError, match="w_y"):
        build_observable_problem(artifact, horizon=16, u_max=2.0, w_y=1.0, w_psd=1.0, psd_ref=envelope)
    with pytest.raises(ValueError, match="w_psd > 0"):
        build_observable_problem(artifact, horizon=16, u_max=2.0, w_u=1.0)
    with pytest.raises(ValueError, match="w_psd > 0"):
        build_observable_problem(artifact, horizon=16, u_max=2.0, w_psd=1.0)
    with pytest.raises(ValueError, match=r"no .* frame"):
        build_observable_problem(artifact, horizon=4, u_max=2.0, w_psd=1.0, psd_ref=envelope)


def test_observable_problem_rejects_native_solver_with_hinge(
    tmp_path: Path, make_observable_checkpoint: Callable[..., ObservableCheckpoint]
) -> None:
    """A native expansion-only solver is rejected when the objective carries a whole-horizon hinge.

    The hinge returns ``0`` from ``evaluate``, so a native backend would silently solve a problem
    missing it; the controller raises instead. The transcription path still scores the hinge and
    is covered by the parity tests.
    """
    ckpt = make_observable_checkpoint(StftGeometry(n_segment=_L, n_hop=_R), horizon=16, n_y=4, dt=0.02)
    artifact = tmp_path / "observable"
    ckpt.save(artifact)
    envelope = _envelope_npz(tmp_path, np.full((ckpt.n_channels, _L // 2 + 1), 1e-3))
    problem = build_observable_problem(artifact, horizon=16, u_max=2.0, w_u=0.1, w_psd=10.0, psd_ref=envelope)
    with pytest.raises(ValueError, match="whole-horizon"):
        TrajOptMPCController(dt=ckpt.dt, problem=problem, solver=ALTRO())
