"""Model Predictive Control (MPC) for the Jansen-Rit network.

This controller optimizes a multi-step sequence of control inputs to minimize
the error between the predicted EEG lfp and a reference signal, subject to
control bounds and optionally smoothness penalties.
"""
# ruff: noqa: N806

from __future__ import annotations

import dataclasses
from typing import Any, Self, cast

import casadi as ca
import numpy as np
from simulate.controller import Controller

from neuro.jansen_rit import JansenRitParams
from neuro.jansen_rit_casadi import (
    JRSymbolicParams,
    eeg,
    heun_step,
    project_control,
)


@dataclasses.dataclass(frozen=True)
class MPCLog:
    """Log for MPC containing internal prediction and control info if desired."""


class MPCController(Controller[MPCLog]):
    """Model Predictive Controller for the Jansen-Rit network."""

    def __init__(  # noqa: PLR0913
        self,
        dt: float,
        params: JRSymbolicParams,
        horizon_steps: int,
        R: float,  # noqa: N803
        R_du: float,  # noqa: N803
        bounds: tuple[float, float],
    ) -> None:
        """Initialize the MPC controller.

        Constructs the CasADi Opti problem once, parametrised over the rolling
        history of states and the current reference signal.

        Parameters
        ----------
        dt : float
            Simulation time step.
        params : JRSymbolicParams
            Symbolic parameters for the Jansen-Rit network.
        horizon_steps : int
            Number of steps to look ahead in the predictive horizon.
        R : float
            Control effort penalty weight.
        R_du : float
            Control derivative (smoothness) penalty weight.
        bounds : tuple[float, float]
            Minimum and maximum limits for the control signal.
        """
        super().__init__(dt)
        self.opti = ca.Opti()

        # gamma is structural here (always numeric, never symbolic) for the MPC plant.
        gamma = cast("np.ndarray", params.gamma)
        if gamma.ndim == 1:
            gamma = gamma.reshape(1, -1)
        self.n_elec = gamma.shape[0]
        self.n_nodes = gamma.shape[1]
        self.horizon_steps = horizon_steps

        self.max_delay_steps = int(np.max(params.delay_steps))

        # Parameters (fixed for the optimization but updated each step)
        self.x0_history_p = [self.opti.parameter(6, self.n_nodes) for _ in range(self.max_delay_steps + 1)]
        self.u_prev_p = self.opti.parameter(self.n_elec, 1)

        n_channels = params.eeg_gain.shape[0]
        self.y_ref_p = self.opti.parameter(n_channels, horizon_steps)

        # Decision variables
        self.U = self.opti.variable(self.n_elec, horizon_steps)
        U_proj = [project_control(self.U[:, k], gamma) for k in range(horizon_steps)]

        cost = 0

        # Multiple shooting: explicit state variables per step
        X_vars = [self.opti.variable(6, self.n_nodes) for _ in range(horizon_steps)]
        history = list(self.x0_history_p)
        for k in range(horizon_steps):
            x_next = heun_step(history, U_proj[k], params, dt)
            self.opti.subject_to(X_vars[k] == x_next)

            history.insert(0, X_vars[k])
            history.pop()

            y_k = eeg(X_vars[k], params.eeg_gain)
            err = y_k - self.y_ref_p[:, k]
            cost += ca.sumsqr(err)

        for k in range(horizon_steps):
            cost += R * ca.sumsqr(self.U[:, k])
            if k == 0:
                cost += R_du * ca.sumsqr(self.U[:, k] - self.u_prev_p)
            else:
                cost += R_du * ca.sumsqr(self.U[:, k] - self.U[:, k - 1])

        self.opti.minimize(cost)

        lb, ub = bounds
        self.opti.subject_to(self.opti.bounded(lb, self.U, ub))

        opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes"}
        self.opti.solver("ipopt", opts)

        # Rolling state tracking
        self._past_states: list[np.ndarray] = []
        self._u_prev_val = np.zeros(self.n_elec)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the controller from a configuration dictionary.

        Parameters
        ----------
        config : dict[str, Any]
            Dictionary containing the controller configuration.

        Returns
        -------
        Self
            An instantiated MPCController object.
        """
        dt = float(config["dt"])
        horizon_steps = int(config["horizon"] / dt)
        R = float(config.get("R", 1.0))
        R_du = float(config.get("R_du", 0.0))
        bounds = (float(config.get("u_min", -1.0)), float(config.get("u_max", 1.0)))

        if "target_electrode" not in config:
            config["target_electrode"] = "CP5"

        base = JansenRitParams.from_config(config)
        n_nodes = base.w_weights.shape[0]

        params = JRSymbolicParams(
            A=np.broadcast_to(np.asarray(base.A), (1, n_nodes)),
            B=base.B,
            a=base.a,
            b=base.b,
            C1=base.C1,
            C2=base.C2,
            C3=base.C3,
            C4=base.C4,
            e0=base.e0,
            v0=base.v0,
            r=base.r,
            mean_input=np.broadcast_to(np.asarray(base.mean_input), (1, n_nodes)),
            sigma=base.sigma,
            eeg_gain=base.eeg_gain,
            gamma=base.gamma,
            K=base.K,
            w_weights=base.w_weights,
            delay_steps=base.delay_steps,
        )

        return cls(
            dt=dt,
            params=params,
            horizon_steps=horizon_steps,
            R=R,
            R_du=R_du,
            bounds=bounds,
        )

    def update(
        self,
        t: float,  # noqa: ARG002
        ref: float | np.ndarray,
        x_hat: float | np.ndarray,
    ) -> tuple[float | np.ndarray, MPCLog]:
        """Solve the MPC problem to find the optimal current control.

        Parameters
        ----------
        t : float
            Current time.
        ref : float | np.ndarray
            Target reference signal.
        x_hat : float | np.ndarray
            Estimated or true current state of the plant.

        Returns
        -------
        tuple[float | np.ndarray, MPCLog]
            The optimal control signal for the current step, and a log object.
        """
        x_hat_arr = np.asarray(x_hat, dtype=np.float64).reshape(6, self.n_nodes)

        self._past_states.insert(0, x_hat_arr.copy())
        if len(self._past_states) > self.max_delay_steps + 1:
            self._past_states.pop()

        # Pad with the oldest state if we don't have enough history yet
        hist = self._past_states + [self._past_states[-1]] * (self.max_delay_steps + 1 - len(self._past_states))

        for k in range(self.max_delay_steps + 1):
            self.opti.set_value(self.x0_history_p[k], hist[k])

        self.opti.set_value(self.u_prev_p, self._u_prev_val)

        ref_arr = np.asarray(ref, dtype=np.float64)
        if ref_arr.ndim in {0, 1}:
            y_ref_mat = np.tile(np.reshape(ref_arr, (-1, 1)), (1, self.horizon_steps))
        else:
            y_ref_mat = ref_arr

        self.opti.set_value(self.y_ref_p, y_ref_mat)

        try:
            sol = self.opti.solve()
            u_opt = sol.value(self.U)
            self.opti.set_initial(self.opti.x, sol.value(self.opti.x))
        except RuntimeError:
            u_opt = self.opti.debug.value(self.U)
            # Soft reset the initial guess if it failed
            self.opti.set_initial(self.opti.x, 0)

        if not isinstance(u_opt, np.ndarray):
            u_opt = np.array([u_opt])

        u_opt = u_opt.reshape((self.n_elec, self.horizon_steps))

        u_next = u_opt[:, 0]
        self._u_prev_val = u_next
        return u_next, MPCLog()
