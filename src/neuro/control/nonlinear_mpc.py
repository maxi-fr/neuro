from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Literal, Self, cast

import numpy as np
from pydantic import Field
from simulate.controller import Controller

from neuro.checkpoint import build_symbolic_model, load_any
from neuro.config import StrictConfig
from neuro.control.nlp import MPCNlp
from neuro.control.solvers import IpoptMPCSolver, MPCSolver, SqpFallbackMPCSolver, SqpMPCSolver
from neuro.observable import load_log_reference
from neuro.observable_casadi import ObservableSymbolicModel
from neuro.spectral import PsdEnvelope

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import ArrayLike

    from neuro.esn_predictor_casadi import ESNSymbolicModel
    from neuro.nn_predictor_casadi import NNSymbolicModel
    from neuro.types import FloatArray


@dataclasses.dataclass(frozen=True)
class MPCControllerLog:
    """Per-step diagnostics for any MPC controller: the applied control, optimal cost, solver success, iteration count, iteration-cap flag, warm-up flag, and fallback flag."""

    u: FloatArray
    cost: float
    success: bool
    warmup: bool
    n_iter: int = 0
    capped: bool = False
    fallback: bool = False


class _MPCControllerConfig(StrictConfig):
    """Config schema for :class:`MPCController`."""

    dt: float = Field(gt=0)
    artifact: str
    u_max: float | list[float]
    horizon: int | None = Field(default=None, ge=1)
    w_y: float = Field(default=1.0, ge=0)
    w_y_terminal: float | None = Field(default=None, ge=0)
    w_u: float = Field(default=0.0, ge=0)
    w_u_l1: float = Field(default=0.0, ge=0)
    w_psd: float = Field(default=0.0, ge=0)
    psd_ref: str | None = Field(default=None)
    # Declared so the YAML states the geometry it expects; the envelope npz is the single source of
    # truth for the cost, and neuro.validation raises when the two disagree.
    psd_window_s: float | None = Field(default=None, gt=0)
    psd_hop_s: float | None = Field(default=None, gt=0)
    shooting_depth: int | None = Field(default=None, ge=1)
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


class MPCController(Controller[MPCControllerLog]):
    """Receding-horizon MPC that suppresses EEG power using a CasADi symbolic predictor."""

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        dt: float,
        model: NNSymbolicModel | ESNSymbolicModel | ObservableSymbolicModel,
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
        """Initialize the MPC and build its CasADi solver(s).

        Parameters
        ----------
        dt
            Controller update step in seconds; should equal the predictor's native dt.
        model
            The CasADi symbolic predictor used as the prediction model.
        u_max
            Per-electrode amplitude bound: a scalar shared by every electrode or a
            length-``n_controls`` vector. The box constraint is ``-u_max <= u <= u_max``.
        horizon
            Prediction/control horizon in steps; defaults to the model's native horizon.
        w_y
            Weight on predicted EEG power in the cost.
        w_u
            Weight on control effort (quadratic) in the cost.
        w_psd
            Weight on the spectral cost: the mean squared amount by which the predicted EEG
            spectrum exceeds ``psd_ref``'s healthy envelope, in log power. One-sided, so it is
            exactly ``0`` once the spectrum is under the envelope everywhere, leaving ``w_u``
            alone to ask for the least stimulation that keeps it there. ``0`` (default) disables
            it; any nonzero ``w_y`` destroys that "done" point.
        psd_ref
            Path to the healthy reference envelope npz written by ``scripts/build_healthy_psd.py``.
            Required when ``w_psd > 0``; its stored window geometry drives the cost.
        w_y_terminal
            Weight on predicted EEG power at the *final* horizon step, replacing ``w_y`` there;
            ``None`` (default) keeps ``w_y`` uniform over the horizon.
        w_u_l1
            Weight on the L1 norm of the control effort (a sparse-stimulation penalty); ``0``
            disables it (default), leaving the pure-quadratic problem unchanged.
        shooting_depth
            Size ``D`` of the shooting segments: a state shooting root is introduced every ``D``
            steps. ``D = 1`` is full multiple shooting (a root at every step), ``D >= horizon``
            is single shooting (no roots; the states are condensed out and the NLP is over the
            controls alone). Defaults to ``horizon``, i.e. single shooting
        solver
            Solver mode: ``"sqp_fallback"`` (default, SQP with warm-started IPOPT fallback),
            ``"sqp"`` (standalone SQP), or ``"ipopt"`` (pure IPOPT).
        max_iter
            Hard cap on IPOPT iterations per solve.
        max_cpu_time
            Optional per-solve wall-time budget in seconds (IPOPT ``max_cpu_time``); ``None``
            leaves it unbounded.
        expand
            Expand the NLP from MX to SX before building the solver.
        ipopt_options
            Optional dictionary of extra IPOPT solver options.
        sqp_qpsol
            QP subsolver plugin for SQP: ``"qpoases"`` (default), ``"osqp"``, or ``"qrqp"``.
        sqp_hessian
            Hessian approximation for SQP: ``"limited-memory"`` (default L-BFGS) or ``"exact"``.
        sqp_max_iter
            Maximum number of SQP iterations per solve before fallback.
        sqp_lbfgs_memory
            L-BFGS history memory size for SQP quasi-Newton Hessian updates.
        sqp_options
            Optional dictionary of extra options passed to the SQP solver.
        """
        super().__init__(dt)
        if solver not in ("ipopt", "sqp", "sqp_fallback"):
            msg = f"solver must be 'ipopt', 'sqp', or 'sqp_fallback', got {solver!r}"
            raise ValueError(msg)
        self.model = model
        self.horizon = int(horizon) if horizon is not None else model.native_horizon
        self.w_y = float(w_y)
        self.w_y_terminal = float(w_y_terminal) if w_y_terminal is not None else None
        self.w_u = float(w_u)
        self.w_u_l1 = float(w_u_l1)
        self.w_psd = float(w_psd)
        self.psd_ref = psd_ref
        # An observable model forecasts the Observable directly, so it hinges against the envelope
        # reduced onto its own Frame value grid rather than against a per-(channel, bin) spectrum.
        self.is_observable = isinstance(model, ObservableSymbolicModel)
        self.psd_envelope = PsdEnvelope.load(psd_ref) if psd_ref is not None and not self.is_observable else None
        self.log_reference = (
            load_log_reference(psd_ref, model.geometry, model.fs)
            if psd_ref is not None and isinstance(model, ObservableSymbolicModel)
            else None
        )
        self.shooting_depth = int(shooting_depth) if shooting_depth is not None else self.horizon
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
        """Instantiate from a config dict, loading the predictor checkpoint from disk."""
        cfg = _MPCControllerConfig.model_validate(config)
        return cls(
            dt=cfg.dt,
            model=build_symbolic_model(load_any(cfg.artifact)),
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
        """Build the PCMS multiple-shooting NLP graph and its solver object, once."""
        self._mpc_nlp = MPCNlp.build(
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
            log_reference=self.log_reference,
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
        u_guess = self._u_prev if self._u_prev is not None else np.zeros((h, m))

        # With no shooting roots the seed collapses to the shifted controls plus their L1 slacks.
        phi_guess = [] if self.is_observable else self._shooting_roots(x0, u_guess)

        seed = [u_guess.reshape(-1), *phi_guess]
        if self.w_u_l1 > 0:
            seed.append(np.abs(u_guess).reshape(-1))
        w0 = np.concatenate(seed)

        res = self._solver_obj.solve(x0, w0)
        u_opt = res.u_opt[: h * m].reshape(h, m)
        self._u_prev = np.vstack([u_opt[1:], u_opt[-1:]])
        return u_opt[0], res.cost, res.success, res.n_iter, res.capped, res.fallback

    def _shooting_roots(self, x0: FloatArray, u_guess: FloatArray) -> list[FloatArray]:
        """Roll ``f_step`` forward under ``u_guess`` to seed the multiple-shooting state variables."""
        model = cast("NNSymbolicModel | ESNSymbolicModel", self.model)
        x = x0
        roots = []
        for step in range(self.horizon):
            x = np.asarray(model.f_step(x, u_guess[step])).reshape(-1)
            if (step + 1) % self.shooting_depth == 0 and (step + 1) < self.horizon:
                roots.append(x)
        return roots

    def update(
        self,
        t: float,  # noqa: ARG002
        ref: FloatArray,  # noqa: ARG002
        x_hat: FloatArray,
    ) -> tuple[FloatArray, MPCControllerLog]:
        """Ingest the current EEG measurement, solve the NLP, and emit the first control."""
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
