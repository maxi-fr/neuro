"""System Identification for the Jansen-Rit network using CasADi."""
# ruff: noqa: N806

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

    free_params: dict[str, Any]
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

    # ``w_weights`` is identified as a symmetric, zero-diagonal matrix: its strict
    # upper triangle is optimised (nonnegative) and mirrored. Each edge is capped at
    # this multiple of the largest base weight.
    _W_UB_FACTOR: ClassVar[float] = 10.0

    def __init__(  # noqa: C901, PLR0912, PLR0913, PLR0915
        self,
        dt: float,
        base_params: JansenRitParams,
        free_params: list[str],
        n_steps: int,
        D: int = 1,  # noqa: N803
        ipopt_options: dict[str, Any] | None = None,
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

        D : int
            Subrecord length for Partially Constrained Multiple Shooting (PCMS).
            D=1 corresponds to Multiple Shooting. D>=n_steps corresponds to Single Shooting.
        ipopt_options : dict[str, Any] | None
            Extra solver options merged over the defaults, e.g.
            ``{"ipopt.hessian_approximation": "limited-memory"}`` for a low-memory L-BFGS
            solve. Do not pass ``{"expand": True}`` (see the solver setup below).
        """
        self.opti = ca.Opti()
        self.dt = dt
        self.free_params = free_params

        self.n_nodes = base_params.w_weights.shape[0]
        gamma = base_params.gamma

        if gamma.ndim == 1:
            gamma = gamma.reshape(1, -1)
            self.n_elec = 1
        else:
            self.n_elec = gamma.shape[0]
        self.n_channels = base_params.eeg_gain.shape[0]

        self.n_steps = n_steps

        # Data parameters
        self.u_data_p = self.opti.parameter(self.n_elec, n_steps)
        self.y_data_p = self.opti.parameter(self.n_channels, n_steps)

        sym_kwargs: dict[str, Any] = {
            "w_weights": base_params.w_weights,
            "delay_steps": base_params.delay_steps,
        }

        # Initialize sym_kwargs with base values broadcasted if needed
        for field in dataclasses.fields(JansenRitParams):
            val = getattr(base_params, field.name)
            if field.name in ["A", "mean_input"]:
                val = np.broadcast_to(np.asarray(val), (1, self.n_nodes))
            sym_kwargs[field.name] = val
        sym_kwargs["K"] = base_params.K
        sym_kwargs["eeg_gain"] = base_params.eeg_gain
        sym_kwargs["gamma"] = gamma

        # Placeholder params mirror ``sym_kwargs`` but swap each free parameter for a fresh
        # ``MX.sym``. The single-step ca.Functions built below close over these placeholders
        # (not the live Opti variables), so each free param is passed in as a Function input.
        ph_kwargs: dict[str, Any] = dict(sym_kwargs)
        free_placeholders: list[ca.MX] = []
        free_actuals: list[ca.MX] = []

        # For each free parameter, create an Opti variable
        for p in self.free_params:
            if p == "w_weights":
                iu = np.triu_indices(self.n_nodes, k=1)
                base_tri = np.asarray(base_params.w_weights, dtype=np.float64)[iu]
                w_ub = self._W_UB_FACTOR * (float(np.abs(base_params.w_weights).max()) or 1.0)

                w_var = self.opti.variable(1, base_tri.size)
                prior = base_tri.reshape(1, -1)
                self.opti.set_initial(w_var, prior)
                self.opti.subject_to(self.opti.bounded(0.0, w_var, w_ub))

                edges = {}
                for idx, (i, j) in enumerate(zip(*iu, strict=True)):
                    edges[int(i), int(j)] = edges[int(j), int(i)] = w_var[0, idx]
                w_sym = ca.vertcat(
                    *[
                        ca.horzcat(*[edges.get((i, j), ca.MX(0.0)) for j in range(self.n_nodes)])
                        for i in range(self.n_nodes)
                    ]
                )
                sym_kwargs[p] = w_sym

                w_ph = ca.MX.sym("w_weights", self.n_nodes, self.n_nodes)
                ph_kwargs[p] = w_ph
                free_placeholders.append(w_ph)
                free_actuals.append(w_sym)
                continue

            if not hasattr(base_params, p):
                msg = f"Unknown parameter {p}"
                raise ValueError(msg)

            # Determine shape
            if p in ["A", "mean_input"]:
                shape = (1, self.n_nodes)
            elif p == "eeg_gain":
                shape = base_params.eeg_gain.shape
            elif p == "gamma":
                shape = (self.n_elec, self.n_nodes)
            else:
                shape = (1, 1)

            v = self.opti.variable(*shape)
            sym_kwargs[p] = v

            v_ph = ca.MX.sym(p, *shape)
            ph_kwargs[p] = v_ph
            free_placeholders.append(v_ph)
            free_actuals.append(v)

            prior = getattr(base_params, p)
            if p == "eeg_gain":
                prior = np.abs(base_params.eeg_gain).mean() or 1.0
            elif p == "gamma":
                prior = gamma  # 2D-reshaped base gamma, shape (n_elec, n_nodes)
            self.opti.set_initial(v, prior)

            # Bounds
            lb = -np.inf
            ub = np.inf
            for f in dataclasses.fields(base_params):
                if f.name == p:
                    lb, ub = f.metadata.get("bounds", (-np.inf, np.inf))
                    break
            self.opti.subject_to(self.opti.bounded(lb, v, ub))

        self.sym_params = JRSymbolicParams(**sym_kwargs)
        self.model = JRSymbolicModel(self.sym_params, dt)
        self.max_delay = self.model.history_depth

        # Compile a single Heun step and the output map into ca.Functions. Calling these in
        # the cost loop inserts one call node per step instead of inlining the whole O(N^2)
        # step graph n_steps times, which is what otherwise exhausts memory on long horizons.
        # The free-parameter decision variables are passed in as Function inputs (the
        # ``free_actuals``) via matching placeholders in ``ph_model``.
        ph_model = JRSymbolicModel(JRSymbolicParams(**ph_kwargs), dt)
        self._free_actuals = free_actuals

        xh_syms = [ca.MX.sym(f"xh{i}", *self.model.state_shape) for i in range(self.max_delay + 1)]
        u_step = ca.MX.sym("u_step", self.n_elec, 1)
        self._F_step = ca.Function("F_step", [*xh_syms, u_step, *free_placeholders], [ph_model.step(xh_syms, u_step)])

        x_out = ca.MX.sym("x_out", *self.model.state_shape)
        self._F_out = ca.Function("F_out", [x_out, *free_placeholders], [ph_model.output(x_out)])

        # Initial state variable (at step 0)
        self.x0_var = self.opti.variable(*self.model.state_shape)

        # Intermediate state variables for steps 1 to max_delay
        self.x_vars = [self.opti.variable(*self.model.state_shape) for _ in range(self.max_delay)]

        Y_pred = []

        # For the first max_delay steps, the states are already decision variables.
        # We compute the predictions directly from them.
        for k in range(self.max_delay):
            current_x = self.x_vars[k]
            y_k = self._F_out(current_x, *self._free_actuals)
            Y_pred.append(y_k)

        # Initial history for starting the loop after max_delay steps
        history = [self.x_vars[i] for i in range(self.max_delay - 1, -1, -1)] + [self.x0_var]

        self.ms_vars = []

        for k in range(self.max_delay, n_steps):
            x_next = self._F_step(*history, self.u_data_p[:, k], *self._free_actuals)

            if (k + 1) % D == 0 and (k + 1) < n_steps:
                current_x = self.opti.variable(*self.model.state_shape)
                self.opti.subject_to(current_x == x_next)
                self.ms_vars.append((current_x, k + 1))
            else:
                current_x = x_next

            history.insert(0, current_x)
            history.pop()

            y_k = self._F_out(current_x, *self._free_actuals)
            Y_pred.append(y_k)

        # Single least-squares cost over the whole horizon; ``Y_pred`` is ordered to match
        # the columns of ``y_data_p``, so this equals the per-step sum of ``sumsqr`` terms.
        self.Y_pred_sym = ca.horzcat(*Y_pred)
        cost = ca.sumsqr(self.Y_pred_sym - self.y_data_p)
        self.opti.minimize(cost)

        opts: dict[str, Any] = {
            "ipopt.print_level": 5,
            "print_time": 1,
            "ipopt.sb": "yes",
            "ipopt.nlp_scaling_method": "gradient-based",
        }
        # Caller overrides merge last (e.g. ``ipopt.hessian_approximation: limited-memory``
        # for a low-memory L-BFGS solve). Never enable ``expand``: it re-inlines the per-step
        # ca.Function call nodes into one flat graph and reintroduces the construction OOM.
        if ipopt_options:
            opts.update(ipopt_options)
        self.opti.solver("ipopt", opts)

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
            D=config.get("D", 1),
            ipopt_options=config.get("ipopt_options"),
        )

    def solve(self, u_data: np.ndarray, y_data: np.ndarray) -> SysIDResult:
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

        # Try solving
        try:
            sol = self.opti.solve()
            cost_val = sol.value(self.opti.f)
            y_pred_val = sol.value(self.Y_pred_sym).T
        except RuntimeError:
            sol = self.opti.debug
            cost_val = sol.value(self.opti.f)
            y_pred_val = sol.value(self.Y_pred_sym).T

        kwargs = {k: sol.value(getattr(self.sym_params, k)) for k in self.free_params}

        return SysIDResult(
            free_params=kwargs,
            cost=cost_val,
            trajectory=y_pred_val,
        )
