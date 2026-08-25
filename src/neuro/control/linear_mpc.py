from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Self, cast

import casadi as ca
import numpy as np
from pydantic import Field
from simulate.controller import Controller

from neuro.checkpoint import build_symbolic_model, load_rollout
from neuro.config import StrictConfig
from neuro.control.nlp import _l1_epigraph, _sum_to_zero
from neuro.control.nonlinear_mpc import MPCControllerLog

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from neuro.esn_predictor_casadi import ESNSymbolicModel
    from neuro.nn_predictor_casadi import NNSymbolicModel
    from neuro.types import FloatArray


class _LinearMPCControllerConfig(StrictConfig):
    """Config schema for :class:`LinearMPCController`."""

    dt: float = Field(gt=0)
    artifact: str
    u_max: float | list[float]
    horizon: int | None = Field(default=None, ge=1)
    w_y: float = Field(default=1.0, ge=0)
    w_u: float = Field(default=0.0, ge=0)
    w_u_l1: float = Field(default=0.0, ge=0)
    formulation: Literal["sparse", "dense"] = "sparse"
    osqp_eps: float = Field(default=1e-9, gt=0)


class LinearMPCController(Controller[MPCControllerLog]):
    """Receding-horizon MPC for a *linear* (0-hidden-layer) NN predictor, solved as a QP."""

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        dt: float,
        model: NNSymbolicModel | ESNSymbolicModel,
        u_max: ArrayLike,
        horizon: int | None = None,
        w_y: float = 1.0,
        w_u: float = 0.0,
        w_u_l1: float = 0.0,
        *,
        formulation: str = "sparse",
        osqp_eps: float = 1e-9,
    ) -> None:
        """Initialize the linear MPC and build its (re-used) QP solver.

        Parameters
        ----------
        dt
            Controller update step in seconds; should equal the predictor's native dt.
        model
            The CasADi symbolic predictor used as the prediction model. Must be linear: its inner
            structure must have 0 hidden layers, otherwise the QP would silently linearize it.
        u_max
            Per-electrode amplitude bound: a scalar shared by every electrode or a
            length-``n_controls`` vector. The box constraint is ``-u_max <= u <= u_max``.
        horizon
            Prediction/control horizon in steps; defaults to the model's native horizon.
        w_y
            Weight on predicted EEG power in the cost.
        w_u
            Weight on control effort (quadratic) in the cost.
        w_u_l1
            Weight on the L1 norm of the control effort (a sparse-stimulation penalty); ``0``
            disables it (default). When positive the QP gains slack variables and the
            inequalities ``t >= |u|`` so it stays a convex QP.
        formulation
            ``"sparse"`` (stacked states, OSQP) or ``"dense"`` (condensed, qpOASES).
        osqp_eps
            OSQP absolute/relative convergence tolerance (``"sparse"`` only); the default
            ``1e-9`` is far tighter than OSQP's loose ``1e-3`` default so the suppression QP is
            solved accurately.
        """
        super().__init__(dt)
        if formulation not in ("sparse", "dense"):
            msg = f"formulation must be 'sparse' or 'dense', got {formulation!r}"
            raise ValueError(msg)
        if not model.is_linear:
            msg = (
                "LinearMPCController requires a linear predictor (0 hidden layers); got a nonlinear "
                "predictor. Use MPCController for a nonlinear predictor."
            )
            raise ValueError(msg)

        self.model = model
        self.horizon = int(horizon) if horizon is not None else model.native_horizon
        self.w_y = float(w_y)
        self.w_u = float(w_u)
        self.w_u_l1 = float(w_u_l1)
        self.formulation = formulation
        self.osqp_eps = float(osqp_eps)

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
        cfg = _LinearMPCControllerConfig.model_validate(config)
        return cls(
            dt=cfg.dt,
            model=cast("NNSymbolicModel | ESNSymbolicModel", build_symbolic_model(load_rollout(cfg.artifact))),
            u_max=cfg.u_max,
            horizon=cfg.horizon,
            w_y=cfg.w_y,
            w_u=cfg.w_u,
            w_u_l1=cfg.w_u_l1,
            formulation=cfg.formulation,
            osqp_eps=cfg.osqp_eps,
        )

    def _build_solver(self) -> None:  # noqa: PLR0915
        """Build the QP and its OSQP (``"sparse"``) or qpOASES (``"dense"``) solver, once.

        ``"sparse"`` stacks the controls and every intermediate state as decision variables and
        ties them with the affine continuity constraints; ``"dense"`` keeps only the controls
        and unrolls the affine state map from the ``x0`` parameter. Both accumulate the same
        quadratic cost and both enforce the Kirchhoff sum-to-zero equality on the controls (so
        ``"dense"`` also carries equality constraints). The inline symbolic ``model.step``/
        ``model.output`` (rather than the compiled ``f_step``/``f_out``) keep the graph a flat
        affine expression, so ``ca.qpsol`` extracts an exact constant Hessian.
        """
        n_state = self.model.state_shape[0]
        n_ctrl, h = self.n_controls, self.horizon

        x0_p = ca.MX.sym("x0", n_state)
        u_vars = [ca.MX.sym(f"u_{k}", n_ctrl) for k in range(h)]

        cost = ca.MX(0)
        # Kirchhoff current law: the montage currents sum to zero at every step (h equalities).
        kcl = _sum_to_zero(u_vars)
        if self.formulation == "dense":
            x_curr = x0_p
            for k in range(h):
                x_curr = self.model.step([x_curr], u_vars[k])
                y_next = self.model.output(x_curr)
                cost = cost + self.w_y * ca.sumsqr(y_next) + self.w_u * ca.sumsqr(u_vars[k])
            x_parts = [*u_vars]
            g_parts: list[ca.MX] = [kcl]
            lbx = np.tile(-self.u_max, h)
            ubx = np.tile(self.u_max, h)
            lbg = np.zeros(h)
            ubg = np.zeros(h)
        else:  # sparse
            x_vars = [ca.MX.sym(f"x_{k}", n_state) for k in range(1, h + 1)]
            defects, x_prev = [], x0_p
            for k in range(h):
                x_lift = x_vars[k]
                defects.append(x_lift - self.model.step([x_prev], u_vars[k]))
                y_next = self.model.output(x_lift)
                cost = cost + self.w_y * ca.sumsqr(y_next) + self.w_u * ca.sumsqr(u_vars[k])
                x_prev = x_lift
            x_parts = [*u_vars, *x_vars]
            g_parts = [*defects, kcl]
            n_x_vars = h * n_state
            lbx = np.concatenate([np.tile(-self.u_max, h), np.full(n_x_vars, -np.inf)])
            ubx = np.concatenate([np.tile(self.u_max, h), np.full(n_x_vars, np.inf)])
            lbg = np.zeros(h * n_state + h)  # continuity defects + sum-to-zero (equalities)
            ubg = np.zeros(h * n_state + h)

        cost = cost / h

        if self.w_u_l1 > 0:
            slacks, l1_cost, l1_g = _l1_epigraph(u_vars, self.w_u_l1)
            cost = cost + l1_cost
            x_parts += slacks
            g_parts.append(l1_g)
            n_l1 = l1_g.numel()
            lbg = np.concatenate([lbg, np.zeros(n_l1)])
            ubg = np.concatenate([ubg, np.full(n_l1, np.inf)])
            lbx = np.concatenate([lbx, np.zeros(h * n_ctrl)])
            ubx = np.concatenate([ubx, np.full(h * n_ctrl, np.inf)])

        qp: dict[str, Any] = {"x": ca.vertcat(*x_parts), "f": cost, "p": x0_p}
        if g_parts:
            qp["g"] = ca.vertcat(*g_parts)
        self._lbx, self._ubx = lbx, ubx
        self._lbg, self._ubg = lbg, ubg
        self._has_g = bool(g_parts)

        opts: dict[str, Any] = {"print_time": False, "error_on_fail": False}
        if self.formulation == "sparse":
            plugin = "osqp"
            opts["osqp"] = {"verbose": False, "eps_abs": self.osqp_eps, "eps_rel": self.osqp_eps}
        else:
            plugin = "qpoases"
            opts["printLevel"] = "none"
        self._solver = ca.qpsol("mpc", plugin, qp, opts)

    def _solve(self, x0: FloatArray) -> tuple[FloatArray, float, bool]:
        """Solve the QP for window-state ``x0``; return ``(u_0*, cost, success)``."""
        m, h = self.n_controls, self.horizon
        u_guess = self._u_prev if self._u_prev is not None else np.zeros((h, m))

        if self.formulation == "dense":
            seed = [u_guess.reshape(-1)]
        else:
            # Forward-simulate from x0 to seed the lifted states with a feasible guess.
            x = x0
            x_guess = []
            for step in range(h):
                x = np.asarray(self.model.f_step(x, u_guess[step])).reshape(-1)
                x_guess.append(x)
            seed = [u_guess.reshape(-1), *x_guess]
        if self.w_u_l1 > 0:
            seed.append(np.abs(u_guess).reshape(-1))  # slacks t = |u|
        w0 = np.concatenate(seed)

        call: dict[str, Any] = {"x0": w0, "lbx": self._lbx, "ubx": self._ubx, "p": x0}
        if self._has_g:
            call["lbg"], call["ubg"] = self._lbg, self._ubg
        sol = self._solver(**call)

        u_opt = np.asarray(sol["x"]).reshape(-1)[: h * m].reshape(h, m)
        self._u_prev = np.vstack([u_opt[1:], u_opt[-1:]])
        success = bool(self._solver.stats()["success"])
        return u_opt[0], float(sol["f"]), success

    def update(
        self,
        t: float,  # noqa: ARG002
        ref: FloatArray,  # noqa: ARG002
        x_hat: FloatArray,
    ) -> tuple[FloatArray, MPCControllerLog]:
        """Ingest the current EEG measurement, solve the QP, and emit the first control."""
        self._state = self.model.absorb(self._state, x_hat.reshape(-1), self._u_last)

        if not self.model.is_ready(self._state):
            u_zero = np.zeros(self.n_controls, dtype=np.float64)
            self._u_last = u_zero
            return u_zero, MPCControllerLog(u=u_zero, cost=0.0, success=True, warmup=True)

        u0, cost, success = self._solve(self._state)
        self._u_last = u0
        return u0, MPCControllerLog(u=u0.copy(), cost=cost, success=success, warmup=False)
