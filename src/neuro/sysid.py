"""System Identification for the Jansen-Rit network using CasADi."""
# ruff: noqa: N806, PLR2004

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar

import casadi as ca
import numpy as np

from neuro.jansen_rit import JansenRitParams
from neuro.jansen_rit_casadi import JRSymbolicParams
from neuro.symbolic_model import JRSymbolicModel


@dataclasses.dataclass
class SysIDResult:
    """Result of a system identification run."""

    params: JansenRitParams
    eeg_gain: np.ndarray | None
    cost: float
    trajectory: np.ndarray  # Predicted lfp


class SysIDSolver:
    """System identification solver for the Jansen-Rit network.

    This solver uses the Partially Constrained Multiple Shooting (PCMS) formulation
    to identify network parameters from EEG data.

    Optimization Problem Size & Conditioning
    ----------------------------------------
    The size and conditioning of the Nonlinear Programming (NLP) problem depend
    heavily on the trajectory length (`n_steps`) and the subrecord length (`D`):

    - **Decision Variables:**
      The number of state decision variables scales as `O((n_steps / D) * n_nodes * 6)`.
      Multiple Shooting (D=1) creates new state variables at every step, leading
      to a massive NLP that consumes a lot of memory but has a smoother objective
      landscape. Single Shooting (D >= n_steps) creates no intermediate state
      variables, keeping the NLP small but creating a deeply nested simulation graph.

    - **Conditioning and Stability:**
      If the system exhibits chaotic or unstable dynamics for certain parameter
      combinations, Single Shooting (large D) can become non-contractive, leading
      to gradient explosion and causing the IPOPT solver to fail or get stuck in
      local minima. Smaller D breaks the simulation into smaller, stable subrecords,
      making it significantly more robust against poor initial parameter guesses.

    - **Trade-offs:**
      - **D = 1 (Multiple Shooting):** Most robust, but memory intensive. If `n_steps`
        is very large (e.g., > 10,000), Ipopt may run out of memory or take too
        long per iteration due to the massive Jacobian/Hessian size.
      - **D >= n_steps (Single Shooting):** Least memory usage, fastest per-iteration
        computation, but highly susceptible to getting stuck in local minima or
        crashing if the initial guess falls in an unstable parameter regime.
      - **1 < D < n_steps (PCMS):** The sweet spot. Tuning D (e.g., D=10 to D=100)
        reduces the number of decision variables compared to Multiple Shooting
        while still preventing the gradients from exploding over long horizons.
    """

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

    # ``w_weights`` is identified as a symmetric, zero-diagonal matrix: its strict
    # upper triangle is optimised (nonnegative, scaled by the base matrix magnitude)
    # and mirrored. This caps each scaled edge weight.
    _W_SCALED_UB: ClassVar[float] = 10.0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SysIDSolver:
        """Create a SysIDSolver from a configuration dictionary.

        Parameters
        ----------
        config : dict[str, Any]
            Dictionary containing the solver configuration.

        Returns
        -------
        SysIDSolver
            A configured SysIDSolver instance.
        """
        base_params = JansenRitParams.from_config(config["params"])
        return cls(
            dt=config["dt"],
            base_params=base_params,
            free_params=config["free_params"],
            n_steps=config["n_steps"],
            lambda_reg=config.get("lambda_reg", 0.01),
            D=config.get("D", 1),
        )

    def __init__(  # noqa: C901, PLR0912, PLR0913, PLR0915
        self,
        dt: float,
        base_params: JansenRitParams,
        free_params: list[str],
        n_steps: int,
        lambda_reg: float = 0.01,
        D: int = 1,  # noqa: N803
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
        n_steps : int
            Number of steps to fit the data over.
        lambda_reg : float
            Regularization penalty for keeping free parameters close to nominal values.
        D : int
            Subrecord length for Partially Constrained Multiple Shooting (PCMS).
            D=1 corresponds to Multiple Shooting. D>=n_steps corresponds to Single Shooting.
        """
        self.opti = ca.Opti()
        self.dt = dt
        self.base = base_params
        self.free_params = free_params

        self.n_nodes = base_params.w_weights.shape[0]
        gamma = base_params.gamma
        if gamma is None:
            self.n_elec = 1
            gamma = np.zeros((1, self.n_nodes))
        elif gamma.ndim == 1:
            gamma = gamma.reshape(1, -1)
            self.n_elec = 1
        else:
            self.n_elec = gamma.shape[0]
        self.n_channels = base_params.eeg_gain.shape[0]

        self.n_steps = n_steps

        # Data parameters
        self.u_data_p = self.opti.parameter(self.n_elec, n_steps)
        self.y_data_p = self.opti.parameter(self.n_channels, n_steps)
        self.y_std_p = self.opti.parameter(self.n_channels, 1)

        # Build decision variables for the free parameters and assemble the model.
        self._var_map = {}
        self._actual_map = {}

        sym_kwargs: dict[str, Any] = {
            "w_weights": base_params.w_weights,
            "delay_steps": base_params.delay_steps,
        }

        # Initialize sym_kwargs with base values broadcasted if needed
        for field in dataclasses.fields(JansenRitParams):
            val = getattr(self.base, field.name)
            if field.name in ["A", "mean_input"]:
                val = np.broadcast_to(np.asarray(val), (1, self.n_nodes))
            sym_kwargs[field.name] = val
        sym_kwargs["K"] = base_params.K
        sym_kwargs["eeg_gain"] = base_params.eeg_gain
        sym_kwargs["gamma"] = gamma

        # For each free parameter, create an Opti variable
        for p in self.free_params:
            if p == "w_weights":
                # Identify a symmetric, zero-diagonal connectivity matrix: optimise the
                # strict upper triangle and mirror it, so symmetry holds by construction.
                iu = np.triu_indices(self.n_nodes, k=1)
                base_tri = np.asarray(base_params.w_weights, dtype=np.float64)[iu]
                scale = float(np.abs(base_params.w_weights).max()) or 1.0

                w_scaled = self.opti.variable(1, base_tri.size)
                self._var_map[p] = w_scaled
                self.opti.subject_to(self.opti.bounded(0.0, w_scaled, self._W_SCALED_UB))

                edges = {}
                for idx, (i, j) in enumerate(zip(*iu, strict=True)):
                    edges[int(i), int(j)] = edges[int(j), int(i)] = w_scaled[0, idx] * scale
                w_sym = ca.vertcat(
                    *[
                        ca.horzcat(*[edges.get((i, j), ca.MX(0.0)) for j in range(self.n_nodes)])
                        for i in range(self.n_nodes)
                    ]
                )
                self._actual_map[p] = w_sym
                sym_kwargs[p] = w_sym
                self._w_tri_target = (base_tri / scale).reshape(1, -1)
                continue

            if p not in self.NOMINALS:
                msg = f"Unknown parameter {p}"
                raise ValueError(msg)

            # Determine shape
            if p in ["A", "mean_input"]:
                shape = (1, self.n_nodes)
            elif p == "eeg_gain":
                shape = base_params.eeg_gain.shape
            else:
                shape = (1, 1)

            # Scaled variable near 1.0
            v_scaled = self.opti.variable(*shape)
            self._var_map[p] = v_scaled

            # Nominal scaling
            nominal = self.NOMINALS[p]
            if p == "eeg_gain":
                nominal = np.abs(base_params.eeg_gain).mean() or 1.0
            actual = v_scaled * nominal
            self._actual_map[p] = actual
            sym_kwargs[p] = actual

            # Bounds
            lb, ub = self.BOUNDS[p]
            self.opti.subject_to(self.opti.bounded(lb / nominal, v_scaled, ub / nominal))

        self.sym_params = JRSymbolicParams(**sym_kwargs)
        self.model = JRSymbolicModel(self.sym_params, dt)
        self.max_delay = self.model.history_depth

        # State stays unbounded: the parameter bounds keep the dynamics in a stable
        # regime and the data-fitting cost + dynamics defect constraints pin the states.

        # Initial state variable (at step 0)
        self.x0_var = self.opti.variable(*self.model.state_shape)

        # Intermediate state variables for steps 1 to max_delay
        self.x_vars = [self.opti.variable(*self.model.state_shape) for _ in range(self.max_delay)]

        # Build the dynamic graph
        cost = 0
        reg_cost = 0

        # Regularization: pull scaled parameters to 1.0 (their nominal base values);
        # the weight triangle is pulled to its base structural values instead.
        for p in self.free_params:
            target = self._w_tri_target if p == "w_weights" else 1.0
            reg_cost += lambda_reg * ca.sumsqr(self._var_map[p] - target)

        # We need a way to track the predicted Y to output it later
        Y_pred = []

        # For the first max_delay steps, the states are already decision variables.
        # We compute the cost and predictions directly from them.
        for k in range(self.max_delay):
            current_x = self.x_vars[k]
            y_k = self.model.output(current_x)
            err = (y_k - self.y_data_p[:, k]) / self.y_std_p
            cost += ca.sumsqr(err)
            Y_pred.append(y_k)

        # Initial history for starting the loop after max_delay steps
        history = [self.x_vars[i] for i in range(self.max_delay - 1, -1, -1)] + [self.x0_var]

        self.ms_vars = []

        for k in range(self.max_delay, n_steps):
            x_next = self.model.step(history, self.u_data_p[:, k])

            if (k + 1) % D == 0 and (k + 1) < n_steps:
                current_x = self.opti.variable(*self.model.state_shape)
                self.opti.subject_to(current_x == x_next)
                self.ms_vars.append((current_x, k + 1))
            else:
                current_x = x_next

            history.insert(0, current_x)
            history.pop()

            y_k = self.model.output(current_x)
            err = (y_k - self.y_data_p[:, k]) / self.y_std_p
            cost += ca.sumsqr(err)
            Y_pred.append(y_k)

        self.Y_pred_sym = ca.horzcat(*Y_pred)
        self.opti.minimize(cost + reg_cost)

        opts = {"ipopt.print_level": 5, "print_time": 1, "ipopt.sb": "yes"}
        self.opti.solver("ipopt", opts)

    def solve(  # noqa: C901, PLR0912
        self,
        u_data: np.ndarray,
        y_data: np.ndarray,
        x0_guess: np.ndarray | None = None,
    ) -> SysIDResult:
        """Run the SysID optimization on the provided dataset.

        Parameters
        ----------
        u_data : np.ndarray
            Control input array of shape (n_steps, n_elec).
        y_data : np.ndarray
            Measured EEG data of shape (n_steps, n_channels).
        x0_guess : np.ndarray | None
            Optional initial guess for the state matrix at step 0, of shape (6, n_nodes).

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

        # Initial guess for state at step 0
        if x0_guess is None:
            x0_guess = np.zeros((6, self.n_nodes))

        # Warm-start state variables using a forward simulation rollout
        from neuro.jansen_rit import JansenRitDynamics  # noqa: PLC0415

        dyn = JansenRitDynamics(
            dt=self.dt,
            params=self.base,
            initial_state=x0_guess,
        )
        x_guess = [x0_guess]
        for k in range(self.n_steps - 1):
            dyn.evaluate(k * self.dt, u_data[k])
            x_guess.append(dyn.x.copy())

        # Set initial guesses for state variables
        self.opti.set_initial(self.x0_var, x_guess[0])

        for i, x_var in enumerate(self.x_vars):
            self.opti.set_initial(x_var, x_guess[i + 1])

        for x_var, step_idx in self.ms_vars:
            self.opti.set_initial(x_var, x_guess[step_idx])

        # Initial guesses: scaled params start at 1.0 (nominal); the weight triangle
        # starts at its base structural values.
        for p, v in self._var_map.items():
            self.opti.set_initial(v, self._w_tri_target if p == "w_weights" else 1.0)

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
            elif p == "w_weights":
                kwargs[p] = np.asarray(val, dtype=np.float64)
            else:
                kwargs[p] = float(val)

        new_params = dataclasses.replace(self.base, **kwargs)

        return SysIDResult(
            params=new_params,
            eeg_gain=eeg_gain,
            cost=cost_val,
            trajectory=y_pred_val,
        )
