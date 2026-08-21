from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Literal, Self

import casadi as ca
import numpy as np
from pydantic import Field
from simulate.controller import Controller

from neuro.artifacts import load_any_artifact
from neuro.config import StrictConfig
from neuro.control.nlp import MPCNlp, _l1_epigraph, _spectral_hinge_cost, _sum_to_zero
from neuro.control.nonlinear_mpc import MPCControllerLog
from neuro.control.solvers import IpoptMPCSolver, MPCSolver, SqpFallbackMPCSolver, SqpMPCSolver
from neuro.nn_predictor_casadi import NNSymbolicModel
from neuro.predictor.artifact import MLPArtifact
from neuro.spectral import PsdEnvelope

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import ArrayLike

    from neuro.types import FloatArray


@dataclasses.dataclass(frozen=True)
class NarxMPCNlp(MPCNlp):
    """Symbolic output-lifted NARX NLP formulation and bounds."""

    @classmethod
    def build(  # noqa: PLR0913
        cls,
        model: NNSymbolicModel,
        *,
        horizon: int,
        shooting_depth: int | None = None,
        n_controls: int,
        u_max: FloatArray,
        w_y: float,
        w_y_terminal: float | None = None,
        w_u: float = 0.0,
        w_u_l1: float = 0.0,
        w_psd: float = 0.0,
        psd_envelope: PsdEnvelope | None = None,
    ) -> Self:
        """Build the output-lifted or partially-condensed NARX symbolic NLP graph and bounds."""
        D = int(shooting_depth) if shooting_depth is not None else 1
        n_y = model.artifact.n_y
        n_u = model.artifact.n_u
        n_ch = model.n_channels
        n_ctrl = n_controls
        h = horizon
        state_dim = model.state_shape[0]
        y_base = n_y * n_ch

        x0_p = ca.MX.sym("x0", state_dim)
        u_vars = [ca.MX.sym(f"u_{k}", n_ctrl) for k in range(h)]

        def u_at(idx: int) -> ca.MX:
            """Control at window index ``idx``."""
            if idx >= 0:
                return u_vars[idx]
            pos = n_u + idx
            return x0_p[y_base + pos * n_ctrl : y_base + (pos + 1) * n_ctrl]

        def decode(z: ca.MX) -> ca.SX | ca.MX:
            """Decode a lifted model-space output to raw EEG."""
            pad_before = ca.MX.zeros((n_y - 1) * n_ch, 1)
            pad_after = ca.MX.zeros(n_u * n_ctrl, 1)
            return model.output(ca.vertcat(pad_before, z, pad_after))

        if D == 1:
            y_vars = [ca.MX.sym(f"y_{k}", n_ch) for k in range(h)]

            def y_at(idx: int) -> ca.MX:
                """Model-space output at window index ``idx``."""
                if idx >= 0:
                    return y_vars[idx]
                pos = n_y + idx
                return x0_p[pos * n_ch : (pos + 1) * n_ch]

            defects = []
            y_nodes = []
            cost = ca.MX(0)
            for k in range(h):
                y_win = ca.vertcat(*[y_at(k - n_y + j) for j in range(n_y)])
                u_win = ca.vertcat(*[u_at(k - n_u + 1 + j) for j in range(n_u)])
                defects.append(y_vars[k] - model.predict_output(y_win, u_win))
                y_raw = decode(y_vars[k])
                y_nodes.append(y_raw)
                is_terminal = (k == h - 1) and (w_y_terminal is not None)
                w_y_step = w_y_terminal if is_terminal else w_y
                cost = cost + w_y_step * ca.sumsqr(y_raw) + w_u * ca.sumsqr(u_vars[k])

            node_vars = y_vars
            n_def_eq = len(defects) * n_ch
            n_node_vars = h * n_ch
        else:
            n_segments = (h - 1) // D
            phi_vars = [ca.MX.sym(f"y_node_{s}", n_y * n_ch) for s in range(1, n_segments + 1)]

            def get_y_win(seg: int) -> ca.MX:
                return x0_p[:y_base] if seg == 0 else phi_vars[seg - 1]

            defects = []
            y_nodes = []
            cost = ca.MX(0)
            for s in range(n_segments + 1):
                y_curr_win = get_y_win(s)
                start_step = s * D
                end_step = min((s + 1) * D, h)
                for k in range(start_step, end_step):
                    u_win = ca.vertcat(*[u_at(k - n_u + 1 + j) for j in range(n_u)])
                    y_pred = model.predict_output(y_curr_win, u_win)
                    y_raw = decode(y_pred)
                    y_nodes.append(y_raw)
                    is_terminal = (k == h - 1) and (w_y_terminal is not None)
                    w_y_step = w_y_terminal if is_terminal else w_y
                    cost = cost + w_y_step * ca.sumsqr(y_raw) + w_u * ca.sumsqr(u_vars[k])
                    y_curr_win = ca.vertcat(y_curr_win[n_ch:], y_pred)

                if s < n_segments:
                    defects.append(y_curr_win - get_y_win(s + 1))

            node_vars = phi_vars
            n_def_eq = len(defects) * n_y * n_ch
            n_node_vars = len(phi_vars) * n_y * n_ch

        cost = cost / h

        if w_psd > 0:
            if psd_envelope is None:
                msg = "psd_envelope must be provided when w_psd > 0"
                raise ValueError(msg)
            cost = cost + w_psd * _spectral_hinge_cost(y_nodes, psd_envelope, h)

        x_parts = [*u_vars, *node_vars]
        g_parts = [*defects, _sum_to_zero(u_vars)]
        n_eq = n_def_eq + h
        lbx = np.concatenate([np.tile(-u_max, h), np.full(n_node_vars, -np.inf)])
        ubx = np.concatenate([np.tile(u_max, h), np.full(n_node_vars, np.inf)])

        if w_u_l1 > 0:
            slacks, l1_cost, l1_g = _l1_epigraph(u_vars, w_u_l1)
            cost = cost + l1_cost
            x_parts += slacks
            g_parts.append(l1_g)
            n_l1 = l1_g.numel()
            lbg = np.concatenate([np.zeros(n_eq), np.zeros(n_l1)])
            ubg = np.concatenate([np.zeros(n_eq), np.full(n_l1, np.inf)])
            lbx = np.concatenate([lbx, np.zeros(h * n_ctrl)])
            ubx = np.concatenate([ubx, np.full(h * n_ctrl, np.inf)])
        else:
            lbg = ubg = 0.0

        x_nlp = ca.vertcat(*x_parts)
        g_nlp = ca.vertcat(*g_parts)
        nlp = {"x": x_nlp, "f": cost, "g": g_nlp, "p": x0_p}
        return cls(nlp=nlp, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)


class _NarxMPCControllerConfig(StrictConfig):
    """Config schema for :class:`NarxMPCController`."""

    dt: float = Field(gt=0)
    artifact: str
    u_max: float | list[float]
    horizon: int | None = Field(default=None, ge=1)
    shooting_depth: int | None = Field(default=None, ge=1)
    w_y: float = Field(default=1.0, ge=0)
    w_y_terminal: float | None = Field(default=None, ge=0)
    w_u: float = Field(default=0.0, ge=0)
    w_u_l1: float = Field(default=0.0, ge=0)
    w_psd: float = Field(default=0.0, ge=0)
    psd_ref: str | None = Field(default=None)
    psd_window_s: float | None = Field(default=None, gt=0)
    psd_hop_s: float | None = Field(default=None, gt=0)
    solver: Literal["ipopt", "sqp", "sqp_fallback"] = "sqp_fallback"
    max_iter: int = Field(default=100, ge=1)
    max_cpu_time: float | None = Field(default=None, gt=0)
    expand: bool = False
    ipopt_options: dict[str, Any] | None = None
    sqp_qpsol: Literal["qpoases", "osqp", "qrqp"] = "qpoases"
    sqp_hessian: Literal["limited-memory", "exact"] = "limited-memory"
    sqp_max_iter: int = Field(default=15, ge=1)
    sqp_lbfgs_memory: int = Field(default=10, ge=1)
    sqp_options: dict[str, Any] | None = None


class NarxMPCController(Controller[MPCControllerLog]):
    """Receding-horizon MPC using the output-lifted NARX formulation."""

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        dt: float,
        model: NNSymbolicModel,
        u_max: ArrayLike,
        horizon: int | None = None,
        w_y: float = 1.0,
        w_u: float = 0.0,
        w_u_l1: float = 0.0,
        w_psd: float = 0.0,
        *,
        psd_ref: str | Path | None = None,
        w_y_terminal: float | None = None,
        shooting_depth: int | None = None,
        solver: Literal["ipopt", "sqp", "sqp_fallback"] = "sqp_fallback",
        max_iter: int = 100,
        max_cpu_time: float | None = None,
        expand: bool = False,
        ipopt_options: dict[str, Any] | None = None,
        sqp_qpsol: Literal["qpoases", "osqp", "qrqp"] = "qpoases",
        sqp_hessian: Literal["limited-memory", "exact"] = "limited-memory",
        sqp_max_iter: int = 15,
        sqp_lbfgs_memory: int = 10,
        sqp_options: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the NARX MPC and build its CasADi solver(s)."""
        super().__init__(dt)
        if solver not in ("ipopt", "sqp", "sqp_fallback"):
            msg = f"solver must be 'ipopt', 'sqp', or 'sqp_fallback', got {solver!r}"
            raise ValueError(msg)
        if not isinstance(model, NNSymbolicModel):
            msg = f"NarxMPCController requires an NNSymbolicModel, got {type(model).__name__}"
            raise TypeError(msg)

        self.model = model
        self.horizon = int(horizon) if horizon is not None else model.native_horizon
        self.w_y = float(w_y)
        self.w_y_terminal = float(w_y_terminal) if w_y_terminal is not None else None
        self.w_u = float(w_u)
        self.w_u_l1 = float(w_u_l1)
        self.w_psd = float(w_psd)
        self.psd_ref = psd_ref
        self.psd_envelope = PsdEnvelope.load(psd_ref) if psd_ref is not None else None
        self.shooting_depth = int(shooting_depth) if shooting_depth is not None else 1
        self.solver = solver
        self.max_iter = int(max_iter)
        self.max_cpu_time = float(max_cpu_time) if max_cpu_time is not None else None
        self.expand = bool(expand)
        self.ipopt_options = dict(ipopt_options) if ipopt_options is not None else {}
        self.sqp_qpsol = sqp_qpsol
        self.sqp_hessian = sqp_hessian
        self.sqp_max_iter = int(sqp_max_iter)
        self.sqp_lbfgs_memory = int(sqp_lbfgs_memory)
        self.sqp_options = dict(sqp_options) if sqp_options is not None else {}

        self.n_controls = model.n_controls

        u_max_arr = np.atleast_1d(np.asarray(u_max, dtype=np.float64))
        if u_max_arr.size == 1:
            u_max_arr = np.broadcast_to(u_max_arr, (self.n_controls,))
        elif u_max_arr.size != self.n_controls:
            msg = f"u_max has {u_max_arr.size} entries but n_controls is {self.n_controls}"
            raise ValueError(msg)
        self.u_max = np.ascontiguousarray(u_max_arr)

        self._state = model.initial_state()
        self._u_last = np.zeros(self.n_controls, dtype=np.float64)
        self._u_prev: FloatArray | None = None

        self._build_solver()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate from a config dict, loading the predictor artifact from disk."""
        cfg = _NarxMPCControllerConfig.model_validate(config)
        artifact = load_any_artifact(cfg.artifact)
        if not isinstance(artifact, MLPArtifact):
            msg = f"NarxMPCController requires an MLPArtifact, got {type(artifact).__name__}"
            raise TypeError(msg)
        return cls(
            dt=cfg.dt,
            model=NNSymbolicModel(artifact),
            u_max=cfg.u_max,
            horizon=cfg.horizon,
            w_y=cfg.w_y,
            w_u=cfg.w_u,
            w_u_l1=cfg.w_u_l1,
            w_psd=cfg.w_psd,
            psd_ref=cfg.psd_ref,
            w_y_terminal=cfg.w_y_terminal,
            shooting_depth=cfg.shooting_depth,
            solver=cfg.solver,
            max_iter=cfg.max_iter,
            max_cpu_time=cfg.max_cpu_time,
            expand=cfg.expand,
            ipopt_options=cfg.ipopt_options,
            sqp_qpsol=cfg.sqp_qpsol,
            sqp_hessian=cfg.sqp_hessian,
            sqp_max_iter=cfg.sqp_max_iter,
            sqp_lbfgs_memory=cfg.sqp_lbfgs_memory,
            sqp_options=cfg.sqp_options,
        )

    def _build_solver(self) -> None:
        """Build the output-lifted NARX NLP graph and its solver object."""
        self._mpc_nlp = NarxMPCNlp.build(
            self.model,
            horizon=self.horizon,
            shooting_depth=self.shooting_depth,
            n_controls=self.n_controls,
            u_max=self.u_max,
            w_y=self.w_y,
            w_y_terminal=self.w_y_terminal,
            w_u=self.w_u,
            w_u_l1=self.w_u_l1,
            w_psd=self.w_psd,
            psd_envelope=self.psd_envelope,
        )
        if self.solver == "sqp_fallback":
            self._solver_obj: MPCSolver = SqpFallbackMPCSolver.build(
                self._mpc_nlp,
                max_iter=self.max_iter,
                max_cpu_time=self.max_cpu_time,
                expand=self.expand,
                ipopt_options=self.ipopt_options,
                sqp_qpsol=self.sqp_qpsol,
                sqp_hessian=self.sqp_hessian,
                sqp_max_iter=self.sqp_max_iter,
                sqp_lbfgs_memory=self.sqp_lbfgs_memory,
                sqp_options=self.sqp_options,
            )
        elif self.solver == "sqp":
            self._solver_obj = SqpMPCSolver.build(
                self._mpc_nlp,
                qpsol=self.sqp_qpsol,
                hessian_approximation=self.sqp_hessian,
                max_iter=self.sqp_max_iter,
                lbfgs_memory=self.sqp_lbfgs_memory,
                expand=self.expand,
                sqp_options=self.sqp_options,
            )
        else:
            self._solver_obj = IpoptMPCSolver.build(
                self._mpc_nlp,
                max_iter=self.max_iter,
                max_cpu_time=self.max_cpu_time,
                expand=self.expand,
                ipopt_options=self.ipopt_options,
            )

    def _solve(self, x0: FloatArray) -> tuple[FloatArray, float, bool, int, bool, bool]:
        """Solve the NLP for window-state ``x0``; return ``(u_0*, cost, success, n_iter, capped, fallback)``."""
        m, h = self.n_controls, self.horizon
        D = self.shooting_depth
        n_y, n_ch = self.model.artifact.n_y, self.model.n_channels
        u_guess = self._u_prev if self._u_prev is not None else np.zeros((h, m))

        if D == 1:
            x = x0
            y_guess = []
            for step in range(h):
                x = np.asarray(self.model.f_step(x, u_guess[step])).reshape(-1)
                y_guess.append(x[(n_y - 1) * n_ch : n_y * n_ch])
            seed = [u_guess.reshape(-1), *y_guess]
        else:
            x = x0
            phi_guess = []
            for step in range(h):
                x = np.asarray(self.model.f_step(x, u_guess[step])).reshape(-1)
                if (step + 1) % D == 0 and (step + 1) < h:
                    phi_guess.append(x[: n_y * n_ch])
            seed = [u_guess.reshape(-1), *phi_guess]

        if self.w_u_l1 > 0:
            seed.append(np.abs(u_guess).reshape(-1))
        w0 = np.concatenate(seed)

        res = self._solver_obj.solve(x0, w0)
        u_opt = res.u_opt[: h * m].reshape(h, m)
        self._u_prev = np.vstack([u_opt[1:], u_opt[-1:]])
        return u_opt[0], res.cost, res.success, res.n_iter, res.capped, res.fallback

    def update(
        self,
        t: float,  # noqa: ARG002
        ref: FloatArray,  # noqa: ARG002
        x_hat: FloatArray,
    ) -> tuple[FloatArray, MPCControllerLog]:
        """Ingest current measurement, solve the NLP, and emit the first control."""
        self._state = self.model.absorb(self._state, x_hat.reshape(-1), self._u_last)

        if not self.model.is_ready(self._state):
            u_zero = np.zeros(self.n_controls, dtype=np.float64)
            self._u_last = u_zero
            return u_zero, MPCControllerLog(u=u_zero, cost=0.0, success=True, warmup=True)

        u0, cost, success, n_iter, capped, fallback = self._solve(self._state)
        self._u_last = u0
        return u0, MPCControllerLog(
            u=u0.copy(),
            cost=cost,
            success=success,
            warmup=False,
            n_iter=n_iter,
            capped=capped,
            fallback=fallback,
        )
