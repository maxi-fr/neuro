from __future__ import annotations

import dataclasses
import importlib
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import casadi as ca
import numpy as np
from simulate.controller import Controller

from neuro.predictor.inference import (
    InferencePredictor,
    WaveformMLPModel,
)
from neuro.spectral import HealthyReference

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from neuro.types import Activation, FloatArray


@dataclasses.dataclass(frozen=True)
class CasADiMPCLog:
    """Decision diagnostics and unshifted plans: outputs ``(H+1, p)``, Control Currents ``(H, m)``."""

    u: FloatArray
    cost: float
    success: bool
    warmup: bool
    status: str
    solve_time: float
    predicted_y: FloatArray
    planned_u: FloatArray
    cost_spectral: float = 0.0
    cost_quadratic_effort: float = 0.0
    cost_sparse_effort: float = 0.0
    cost_tracking: float = 0.0
    normalization: str = "channel_mean"


@dataclasses.dataclass(frozen=True)
class CasADiWaveformProblem:
    """Continuous waveform optimal control problem definition for CasADi."""

    model: WaveformMLPModel
    horizon: int
    u_max: FloatArray
    w_y: float
    w_u: float
    w_y_terminal: float | None
    reference: HealthyReference | None
    target: FloatArray | None
    kirchhoff: bool
    solver_options: dict[str, Any] = dataclasses.field(default_factory=dict)


def _resolve_waveform_target(
    ref: HealthyReference | None,
    base: WaveformMLPModel,
    *,
    tracking_active: bool,
) -> FloatArray | None:
    """Resolve and validate the physical tracking target from a HealthyReference."""
    if tracking_active and ref is None:
        msg = "reference must be provided when w_y > 0"
        raise ValueError(msg)
    if not tracking_active or ref is None:
        return None
    if ref.eeg_mean is not None and len(ref.eeg_mean) == base.n_channels:
        y_ref = ref.eeg_mean
    elif ref.lfp_mean is not None and len(ref.lfp_mean) == base.n_channels:
        y_ref = ref.lfp_mean
    else:
        msg = f"reference mean vector does not match model channel count ({base.n_channels})"
        raise ValueError(msg)
    return np.asarray(y_ref, dtype=np.float64)


def _mlp_forward_ca(
    x: ca.SX,
    weights: tuple[FloatArray, ...],
    biases: tuple[FloatArray, ...],
    activation: Activation,
) -> ca.SX:
    """Evaluate an MLP forward pass symbolically."""
    for w, b in zip(weights[:-1], biases[:-1], strict=True):
        z = ca.mtimes(ca.SX(np.asarray(w, dtype=np.float64)), x) + ca.SX(np.asarray(b, dtype=np.float64).reshape(-1, 1))
        if activation == "relu":
            x = ca.fmax(z, 0.0)
        elif activation == "tanh":
            x = ca.tanh(z)
        elif activation == "softplus":
            m = ca.fmax(z, 0.0)
            x = m + ca.log(ca.exp(-m) + ca.exp(z - m))
        else:
            msg = f"unsupported activation: {activation}"
            raise ValueError(msg)
    w_last = np.asarray(weights[-1], dtype=np.float64)
    b_last = np.asarray(biases[-1], dtype=np.float64).reshape(-1, 1)
    return ca.mtimes(ca.SX(w_last), x) + ca.SX(b_last)


def _step_casadi(
    x: ca.SX,
    u: ca.SX,
    model: WaveformMLPModel,
) -> tuple[ca.SX, ca.SX]:
    """Advance one step in the CasADi graph: return (x_next, y_next_physical)."""
    n_history = model.n_history
    n_outputs = model.n_outputs
    n_y = model.n_y
    n_u = model.n_u
    n_controls = model.n_controls
    n_z = n_history * n_outputs

    y_window = x[:n_z]
    u_window_raw = x[n_z:]
    u_window = ca.vertcat(u_window_raw[n_controls:], u)

    u_center_tiled = np.tile(np.asarray(model.u_center, dtype=np.float64), n_u).reshape(-1, 1)
    u_scale_tiled = np.tile(np.asarray(model.u_scale, dtype=np.float64), n_u).reshape(-1, 1)
    u_std = (u_window - ca.SX(u_center_tiled)) / ca.SX(u_scale_tiled)

    y_mlp_in = y_window[(n_history - n_y) * n_outputs :]
    mlp_in = ca.vertcat(y_mlp_in, u_std)

    z = _mlp_forward_ca(
        mlp_in,
        tuple(np.asarray(w, dtype=np.float64) for w in model.weights),
        tuple(np.asarray(b, dtype=np.float64) for b in model.biases),
        model.activation,
    )
    if model.residual:
        y_last = y_window[(n_history - 1) * n_outputs :]
        z_next = z + y_last
    else:
        z_next = z

    next_y_window = ca.vertcat(y_window[n_outputs:], z_next)
    next_x = ca.vertcat(next_y_window, u_window)

    y_center = np.asarray(model.y_center, dtype=np.float64).reshape(-1, 1)
    y_scale = np.asarray(model.y_scale, dtype=np.float64).reshape(-1, 1)
    y_out = z_next * ca.SX(y_scale) + ca.SX(y_center)

    return next_x, y_out


def _output_casadi(x: ca.SX, model: WaveformMLPModel) -> ca.SX:
    """Evaluate physical output for a state in the CasADi graph."""
    n_history = model.n_history
    n_outputs = model.n_outputs
    y_last = x[(n_history - 1) * n_outputs : n_history * n_outputs]
    y_center = np.asarray(model.y_center, dtype=np.float64).reshape(-1, 1)
    y_scale = np.asarray(model.y_scale, dtype=np.float64).reshape(-1, 1)
    return y_last * ca.SX(y_scale) + ca.SX(y_center)


def build_casadi_waveform_problem(  # noqa: PLR0913 -- checkpoint plus the MPC cost/bound knobs
    artifact: str | Path | WaveformMLPModel,
    *,
    horizon: int,
    u_max: ArrayLike,
    w_y: float = 1.0,
    w_u: float = 0.0,
    w_y_terminal: float | None = None,
    reference: HealthyReference | str | Path | None = None,
    kirchhoff: bool = True,
    solver_options: dict[str, Any] | None = None,
) -> CasADiWaveformProblem:
    """Assemble a CasADi waveform optimal control problem with waveform tracking and current effort."""
    if isinstance(artifact, WaveformMLPModel):
        base = artifact
    else:
        loaded = InferencePredictor.load(artifact)
        if not isinstance(loaded, WaveformMLPModel):
            msg = f"waveform MLP checkpoint required, got {type(loaded).__name__}"
            raise TypeError(msg)
        base = loaded

    if isinstance(reference, (str, Path)):
        ref: HealthyReference | None = HealthyReference.load(reference)
    else:
        ref = reference

    w_y_final = w_y_terminal if w_y_terminal is not None else w_y
    target_val = _resolve_waveform_target(
        ref,
        base,
        tracking_active=(w_y > 0 or w_y_final > 0),
    )

    u_max_arr = np.broadcast_to(np.atleast_1d(np.asarray(u_max, dtype=np.float64)), (base.n_controls,)).copy()

    return CasADiWaveformProblem(
        model=base,
        horizon=horizon,
        u_max=u_max_arr,
        w_y=w_y,
        w_u=w_u,
        w_y_terminal=w_y_terminal,
        reference=ref,
        target=target_val,
        kirchhoff=kirchhoff,
        solver_options=dict(solver_options) if solver_options is not None else {},
    )


class CasADiMPCController(Controller[CasADiMPCLog]):
    """Receding-horizon continuous MPC for the Waveform Predictor, driven by CasADi IPOPT."""

    def __init__(
        self,
        dt: float,
        problem: CasADiWaveformProblem,
    ) -> None:
        """Initialize the CasADi continuous waveform MPC controller."""
        super().__init__(dt)
        self.problem = problem
        self.model = problem.model
        self.horizon = int(problem.horizon)
        self.n_controls = int(self.model.n_controls)
        self.n_electrodes = self.n_controls
        self.n_channels = int(self.model.n_channels)
        self.dt = float(dt)

        self._state = np.asarray(self.model.initial_state(), dtype=np.float64)
        self._u_last = np.zeros(self.n_controls, dtype=np.float64)
        self._u_guess = np.zeros((self.horizon, self.n_controls), dtype=np.float64)

        self._build_solver()

    def _build_solver(self) -> None:
        """Build the symbolic single-shooting NLP graph and compile the IPOPT solver."""
        h = self.horizon
        m = self.n_controls
        n_state = self.model.n

        x0_sym = ca.SX.sym("x0", n_state)
        u_sym = ca.SX.sym("u", h * m)

        x_curr = x0_sym
        y_0 = _output_casadi(x_curr, self.model)
        y_preds = [y_0]
        cost_tracking = ca.SX(0.0)
        cost_effort = ca.SX(0.0)
        g_parts: list[ca.SX] = []

        w_y = self.problem.w_y
        w_y_terminal = self.problem.w_y_terminal
        w_u = self.problem.w_u
        target = self.problem.target
        target_sym = ca.SX(np.asarray(target, dtype=np.float64).reshape(-1, 1)) if target is not None else None

        for k in range(h):
            u_k = u_sym[k * m : (k + 1) * m]
            x_next, y_next = _step_casadi(x_curr, u_k, self.model)
            y_preds.append(y_next)

            is_terminal = k == h - 1
            w_track = (w_y_terminal if (is_terminal and w_y_terminal is not None) else w_y) / h
            if w_track > 0:
                err = (y_next - target_sym) if target_sym is not None else y_next
                cost_tracking = cost_tracking + w_track * ca.sumsqr(err)

            if w_u > 0:
                cost_effort = cost_effort + (w_u / h) * ca.sumsqr(u_k)

            if self.problem.kirchhoff:
                g_parts.append(ca.sum1(u_k))

            x_curr = x_next

        cost_total = cost_tracking + cost_effort

        if g_parts:
            g_sym = ca.vertcat(*g_parts)
            self._lbg = np.zeros(len(g_parts))
            self._ubg = np.zeros(len(g_parts))
        else:
            g_sym = ca.SX(0.0)
            self._lbg = -np.inf
            self._ubg = np.inf

        self._lbx = np.tile(-self.problem.u_max, h)
        self._ubx = np.tile(self.problem.u_max, h)

        nlp = {"x": u_sym, "p": x0_sym, "f": cost_total, "g": g_sym}

        opts: dict[str, Any] = {
            "print_time": False,
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.tol": 1e-3,
            "ipopt.acceptable_tol": 1e-2,
            "ipopt.acceptable_iter": 5,
            "ipopt.max_iter": 300,
            "ipopt.hessian_approximation": "limited-memory",
        }
        opts.update(self.problem.solver_options)

        self._solver = ca.nlpsol("solver", "ipopt", nlp, opts)
        self._predicted_y_fn = ca.Function("pred_y", [x0_sym, u_sym], [ca.horzcat(*y_preds)])
        self._costs_fn = ca.Function("costs", [x0_sym, u_sym], [cost_total, cost_tracking, cost_effort])

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate controller from a configuration dictionary."""
        dt = float(config["dt"])
        prob_cfg = config["problem"]
        if isinstance(prob_cfg, CasADiWaveformProblem):
            problem = prob_cfg
        elif isinstance(prob_cfg, dict):
            cfg = prob_cfg.copy()
            if "reference" in cfg and isinstance(cfg["reference"], (str, Path)):
                cfg["reference"] = HealthyReference.load(cfg["reference"])
            if "class_path" in cfg:
                class_path: str = cfg.pop("class_path")
                module_name, func_name = class_path.rsplit(".", 1)
                target = getattr(importlib.import_module(module_name), func_name)
                problem = target(**cfg)
            else:
                problem = build_casadi_waveform_problem(**cfg)
        else:
            msg = f"Unexpected problem configuration: {type(prob_cfg)}"
            raise TypeError(msg)
        return cls(dt=dt, problem=problem)

    def update(
        self,
        t: float,  # noqa: ARG002 -- the goal is baked into the objective
        ref: FloatArray,  # noqa: ARG002 -- reference target is set in problem
        x_hat: FloatArray,
    ) -> tuple[FloatArray, CasADiMPCLog]:
        """Absorb measurement, solve the continuous MPC problem, and emit the first Control Current."""
        self._state = np.asarray(self.model.absorb(self._state, np.asarray(x_hat).reshape(-1), self._u_last))

        if not self.model.is_ready(self._state):
            u_zero = np.zeros(self.n_controls, dtype=np.float64)
            self._u_last = u_zero
            return u_zero, CasADiMPCLog(
                u=u_zero,
                cost=0.0,
                success=True,
                warmup=True,
                status="warmup",
                solve_time=0.0,
                predicted_y=np.full((self.horizon + 1, self.n_channels), np.nan),
                planned_u=np.full((self.horizon, self.n_controls), np.nan),
                cost_spectral=0.0,
                cost_quadratic_effort=0.0,
                cost_sparse_effort=0.0,
                cost_tracking=0.0,
                normalization="channel_mean",
            )

        u_guess_flat = self._u_guess.reshape(-1)
        started = time.perf_counter()
        try:
            res = self._solver(
                x0=u_guess_flat,
                p=self._state,
                lbx=self._lbx,
                ubx=self._ubx,
                lbg=self._lbg,
                ubg=self._ubg,
            )
            solve_time = time.perf_counter() - started
            stats = self._solver.stats()
            success = bool(stats["success"])
            status = str(stats.get("return_status", "unknown"))
        except Exception as exc:  # noqa: BLE001 -- handle unexpected solver failure
            solve_time = time.perf_counter() - started
            success = False
            status = str(exc)
            res = None

        if not success or res is None:
            u_zero = np.zeros(self.n_controls, dtype=np.float64)
            self._u_last = u_zero
            if res is not None:
                planned_u = np.asarray(res["x"], dtype=np.float64).reshape(self.horizon, self.n_controls)
                predicted_y = np.asarray(self._predicted_y_fn(self._state, res["x"]), dtype=np.float64).T
                cost_tot, cost_track, cost_effort = (
                    float(v) for v in self._costs_fn(self._state, res["x"])
                )
            else:
                planned_u = np.full((self.horizon, self.n_controls), np.nan)
                predicted_y = np.full((self.horizon + 1, self.n_channels), np.nan)
                cost_tot, cost_track, cost_effort = 0.0, 0.0, 0.0

            return u_zero, CasADiMPCLog(
                u=u_zero,
                cost=cost_tot,
                success=False,
                warmup=False,
                status=status,
                solve_time=solve_time,
                predicted_y=predicted_y,
                planned_u=planned_u,
                cost_spectral=0.0,
                cost_quadratic_effort=cost_effort,
                cost_sparse_effort=0.0,
                cost_tracking=cost_track,
                normalization="channel_mean",
            )

        u_plan = np.asarray(res["x"], dtype=np.float64).reshape(self.horizon, self.n_controls)
        u_cmd = u_plan[0].copy()
        self._u_last = u_cmd
        self._u_guess = np.vstack([u_plan[1:], u_plan[-1:]])

        cost_tot, cost_track, cost_effort = (
            float(v) for v in self._costs_fn(self._state, res["x"])
        )
        predicted_y = np.asarray(self._predicted_y_fn(self._state, res["x"]), dtype=np.float64).T

        return u_cmd, CasADiMPCLog(
            u=u_cmd,
            cost=cost_tot,
            success=True,
            warmup=False,
            status=status,
            solve_time=solve_time,
            predicted_y=predicted_y,
            planned_u=u_plan.copy(),
            cost_spectral=0.0,
            cost_quadratic_effort=cost_effort,
            cost_sparse_effort=0.0,
            cost_tracking=cost_track,
            normalization="channel_mean",
        )
