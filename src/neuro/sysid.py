"""System Identification for the Jansen-Rit network using CasADi."""
# ruff: noqa: N806, PLR2004

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar

import casadi as ca
import numpy as np

from neuro.jansen_rit import JansenRitParams
from neuro.jansen_rit_casadi import (
    JRSymbolicParams,
    heun_step,
    measure,
    project_control,
)


@dataclasses.dataclass
class SysIDResult:
    """Result of a system identification run."""

    params: JansenRitParams
    eeg_gain: np.ndarray | None
    cost: float
    trajectory: np.ndarray  # Predicted output


class SysIDSolver:
    """System identification solver for the Jansen-Rit network."""

    # Rough nominal bounds for scaling and constraints
    NOMINALS: ClassVar[dict[str, float]] = {
        "A": 3.25,
        "B": 22.0,
        "a": 100.0,
        "b": 50.0,
        "C1": 135.0,
        "C2": 108.0,
        "C3": 33.75,
        "C4": 33.75,
        "e0": 2.5,
        "v0": 6.0,
        "r": 0.56,
        "mean_input": 90.0,
        "sigma": 0.0,
        "K": 1.0,
        "eeg_gain": 1.0,
    }

    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {
        "A": (0.0, 10.0),
        "B": (0.0, 100.0),
        "a": (10.0, 500.0),
        "b": (1.0, 200.0),
        "C1": (10.0, 500.0),
        "C2": (10.0, 500.0),
        "C3": (1.0, 200.0),
        "C4": (1.0, 200.0),
        "e0": (0.1, 10.0),
        "v0": (0.1, 20.0),
        "r": (0.1, 2.0),
        "mean_input": (0.0, 500.0),
        "sigma": (0.0, 0.0),
        "K": (0.0, 10.0),
        "eeg_gain": (0.0, 10.0),
    }

    def __init__(  # noqa: C901, PLR0913, PLR0915
        self,
        dt: float,
        base_params: JansenRitParams,
        free_params: list[str],
        gamma: np.ndarray,
        gain: np.ndarray,
        w_weights: np.ndarray,
        delay_steps: np.ndarray,
        n_steps: int,
        lambda_reg: float = 0.01,
    ) -> None:
        """Initialize the SysID NLP.

        Parameters
        ----------
        dt : float
            Simulation time step.
        base_params : JansenRitParams
            Base parameters for the network. Unfree parameters will remain at these values.
        free_params : list[str]
            List of parameter names to estimate (e.g. ["A", "mean_input", "eeg_gain"]).
        gamma : np.ndarray
            Spatial profile mapping electrode stimulation to brain regions.
        gain : np.ndarray
            Leadfield matrix for observing the brain regions.
        w_weights : np.ndarray
            Network connectivity weight matrix.
        delay_steps : np.ndarray
            Network delay matrix in integer steps.
        n_steps : int
            Number of steps to fit the data over.
        lambda_reg : float
            Regularization penalty for keeping free parameters close to nominal values.
        """
        self.opti = ca.Opti()
        self.dt = dt
        self.base = base_params
        self.free_params = free_params

        self.n_nodes = w_weights.shape[0]
        self.n_elec = gamma.shape[0] if gamma.ndim == 2 else 1
        if gamma.ndim == 1:
            gamma = gamma.reshape(1, -1)
        self.n_channels = gain.shape[0]

        self.max_delay = int(np.max(delay_steps))
        self.n_steps = n_steps

        # Data parameters
        self.u_data_p = self.opti.parameter(self.n_elec, n_steps)
        self.y_data_p = self.opti.parameter(self.n_channels, n_steps)
        self.y_std_p = self.opti.parameter(self.n_channels, 1)

        # Initial history parameter
        self.x0_history_p = [
            self.opti.parameter(6, self.n_nodes) for _ in range(self.max_delay + 1)
        ]

        # Build decision variables
        self._var_map = {}
        self._actual_map = {}

        sym_kwargs: dict[str, Any] = {
            "w_weights": w_weights,
            "delay_steps": delay_steps,
        }

        # Initialize sym_kwargs with base values broadcasted if needed
        for field in dataclasses.fields(JansenRitParams):
            val = getattr(self.base, field.name)
            if field.name in ["A", "mean_input"]:
                val = np.broadcast_to(np.asarray(val), (1, self.n_nodes))
            sym_kwargs[field.name] = val
        sym_kwargs["K"] = 1.0
        sym_kwargs["eeg_gain"] = gain

        # For each free parameter, create an Opti variable
        for p in self.free_params:
            if p not in self.NOMINALS:
                msg = f"Unknown parameter {p}"
                raise ValueError(msg)

            # Determine shape
            if p in ["A", "mean_input"]:
                shape = (1, self.n_nodes)
            elif p == "eeg_gain":
                shape = gain.shape
            else:
                shape = (1, 1)

            # Scaled variable near 1.0
            v_scaled = self.opti.variable(*shape)
            self._var_map[p] = v_scaled

            # Nominal scaling
            nominal = self.NOMINALS[p]
            if p == "eeg_gain":
                nominal = np.abs(gain).mean() or 1.0
            actual = v_scaled * nominal
            self._actual_map[p] = actual
            sym_kwargs[p] = actual

            # Bounds
            lb, ub = self.BOUNDS[p]
            self.opti.subject_to(self.opti.bounded(lb / nominal, v_scaled, ub / nominal))

        self.sym_params = JRSymbolicParams(**sym_kwargs)

        # Build the dynamic graph
        cost = 0
        reg_cost = 0

        # Regularization: pull scaled parameters to 1.0 (their nominal base values)
        for p in self.free_params:
            reg_cost += lambda_reg * ca.sumsqr(self._var_map[p] - 1.0)

        # Unroll the graph
        U_proj = [project_control(self.u_data_p[:, k], gamma) for k in range(n_steps)]

        # We need a way to track the predicted Y to output it later
        Y_pred = []

        # Multiple shooting: explicit state variables per step
        X_vars = [self.opti.variable(6, self.n_nodes) for _ in range(n_steps)]
        history = list(self.x0_history_p)

        for k in range(n_steps):
            x_next = heun_step(history, U_proj[k], self.sym_params, dt)
            self.opti.subject_to(X_vars[k] == x_next)

            history.insert(0, X_vars[k])
            history.pop()

            y_k = measure(X_vars[k], self.sym_params.eeg_gain)
            err = (y_k - self.y_data_p[:, k]) / self.y_std_p
            cost += ca.sumsqr(err)
            Y_pred.append(y_k)

        self.Y_pred_sym = ca.horzcat(*Y_pred)
        self.opti.minimize(cost + reg_cost)

        opts = {"ipopt.print_level": 5, "print_time": 1, "ipopt.sb": "yes"}
        self.opti.solver("ipopt", opts)

    def solve(
        self,
        u_data: np.ndarray,
        y_data: np.ndarray,
        x0_history: list[np.ndarray],
    ) -> SysIDResult:
        """Run the SysID optimization on the provided dataset.

        Parameters
        ----------
        u_data : np.ndarray
            Control input array of shape (n_steps, n_elec).
        y_data : np.ndarray
            Measured EEG output data of shape (n_steps, n_channels).
        x0_history : list[np.ndarray]
            List of initial state matrices representing the history before step 0.

        Returns
        -------
        SysIDResult
            A result object containing the estimated parameters, cost, and trajectory.
        """
        # Check shapes
        if u_data.shape != (self.n_steps, self.n_elec):
            msg = f"u_data shape {u_data.shape} must be ({self.n_steps}, {self.n_elec})"
            raise ValueError(msg)
        if y_data.shape != (self.n_steps, self.n_channels):
            msg = f"y_data shape {y_data.shape} must be ({self.n_steps}, {self.n_channels})"
            raise ValueError(msg)

        # Set data
        self.opti.set_value(self.u_data_p, u_data.T)
        self.opti.set_value(self.y_data_p, y_data.T)

        # Channel std
        y_std = np.std(y_data, axis=0).reshape(-1, 1)
        y_std[y_std < 1e-6] = 1.0  # avoid division by zero
        self.opti.set_value(self.y_std_p, y_std)

        # Initial history
        for k in range(self.max_delay + 1):
            self.opti.set_value(self.x0_history_p[k], x0_history[k])

        # Initial guesses: start at 1.0 (nominal)
        for v in self._var_map.values():
            self.opti.set_initial(v, 1.0)

        # Try solving
        try:
            sol = self.opti.solve()
            cost_val = sol.value(self.opti.f)
            y_pred_val = sol.value(self.Y_pred_sym).T
        except RuntimeError:
            sol = self.opti.debug
            cost_val = sol.value(self.opti.f)
            y_pred_val = sol.value(self.Y_pred_sym).T

        # Extract parameters
        kwargs = {}
        eeg_gain = None
        for p in self.free_params:
            val = sol.value(self._actual_map[p])
            if p == "eeg_gain":
                eeg_gain = val
            elif p in ["A", "mean_input"]:
                kwargs[p] = np.asarray(val).flatten()
            else:
                kwargs[p] = float(val)

        new_params = dataclasses.replace(self.base, **kwargs)

        return SysIDResult(
            params=new_params,
            eeg_gain=eeg_gain,
            cost=cost_val,
            trajectory=y_pred_val,
        )
