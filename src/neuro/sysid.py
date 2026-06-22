"""System Identification for the Jansen-Rit network using CasADi."""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar

import casadi as ca
import numpy as np

from neuro.jansen_rit import JansenRitParams
from neuro.symbolic_model import JRSymbolicModel, shoot


@dataclasses.dataclass
class SysIDResult:
    """Result of a system identification run."""

    free_params: dict[str, Any]
    cost: float
    trajectory: np.ndarray  # Predicted lfp


@dataclasses.dataclass(frozen=True)
class _WelchPlan:
    """Precomputed constants for a differentiable one-sided Welch PSD.

    The DFT is the linear map ``Re = Y_seg @ cw``, ``Im = Y_seg @ sw`` with the periodic
    Hann window folded into the cosine/sine matrices; power is ``Re**2 + Im**2`` averaged
    over the overlapping segments. Comparing in log-power makes the constant per-channel
    leadfield scaling cancel in the loss difference.
    """

    starts: list[int]  # segment start indices into the time axis
    seg_len: int
    cw: ca.DM  # (seg_len, n_freqs) windowed cosine matrix
    sw: ca.DM  # (seg_len, n_freqs) windowed -sine matrix
    idx: list[int]  # retained one-sided frequency-bin indices (after band masking)
    freqs: np.ndarray  # frequencies (Hz) of the retained bins


def _welch_plan(n_steps: int, dt: float, nperseg: int, noverlap: int, band: tuple[float, float] | None) -> _WelchPlan:
    """Build the constant matrices for a differentiable Welch PSD over an ``n_steps`` horizon."""
    seg_len = min(nperseg, n_steps)
    noverlap = min(noverlap, seg_len - 1)
    hop = seg_len - noverlap
    starts = list(range(0, n_steps - seg_len + 1, hop)) or [0]

    n = np.arange(seg_len)
    k = np.arange(seg_len // 2 + 1)  # one-sided bins
    ang = 2.0 * np.pi * np.outer(n, k) / seg_len  # (seg_len, n_freqs)
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / seg_len)  # periodic Hann

    cw = ca.DM(window[:, None] * np.cos(ang))
    sw = ca.DM(window[:, None] * -np.sin(ang))

    freqs = k * (1.0 / dt) / seg_len
    idx = np.where((freqs >= band[0]) & (freqs <= band[1]))[0] if band is not None else np.arange(k.size)
    return _WelchPlan(starts, seg_len, cw, sw, idx.tolist(), freqs[idx])


def _log_welch_psd(y: ca.MX, plan: _WelchPlan, eps: float = 1e-12) -> ca.MX:
    """Differentiable one-sided log Welch PSD of a ``(n_channels, n_steps)`` signal."""
    power = ca.MX.zeros(y.shape[0], plan.cw.shape[1])
    for s in plan.starts:
        y_seg = y[:, s : s + plan.seg_len]
        power = power + (ca.mtimes(y_seg, plan.cw) ** 2 + ca.mtimes(y_seg, plan.sw) ** 2)
    power = power / len(plan.starts)
    return ca.log(power[:, plan.idx] + eps)


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

    def __init__(  # noqa: PLR0913
        self,
        dt: float,
        base_params: JansenRitParams,
        free_params: list[str],
        n_steps: int,
        D: int = 1,
        ipopt_options: dict[str, Any] | None = None,
        psd_loss: dict[str, Any] | None = None,
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
        psd_loss : dict[str, Any] | None
            When given, adds a differentiable log Welch power-spectrum term to the cost:
            ``time_weight * sumsqr(Y - Y_data) + weight * sumsqr(logPSD(Y) - logPSD(Y_data))``.
            Keys: ``weight`` (required, spectral term coefficient), ``time_weight`` (default
            1.0; set 0.0 for a pure-spectral fit), ``nperseg`` (default ``min(256, n_steps)``;
            must be large enough at ``fs = 1/dt`` to resolve the rhythm), ``noverlap``
            (default ``nperseg // 2``), and ``band`` (optional ``[fmin, fmax]`` in Hz). A
            spectral term is phase-blind, so it complements rather than replaces the
            time-domain term, especially for identifying ``gamma``.
        """
        self.opti = ca.Opti()
        self.dt = dt
        self.free_params = free_params
        self.n_steps = n_steps

        # One model whose free parameters are symbolic Function inputs (it owns the
        # placeholders); the compiled f_step/f_out take them as trailing inputs.
        self.model = JRSymbolicModel.with_free_params(base_params, dt, free_params)
        self.n_nodes = self.model.state_shape[1]
        self.n_elec = self.model.n_elec
        self.n_channels = self.model.n_channels
        self.max_delay = self.model.history_depth

        # Free-parameter decision variables, grafted onto the model placeholders at every
        # f_step/f_out call site and read back after the solve.
        self._free_vars: dict[str, ca.MX] = {p: self._build_param_var(p, base_params) for p in free_params}
        free_actuals = [self._free_vars[p] for p in free_params]
        f_step, f_out = self.model.f_step, self.model.f_out

        # Data parameters
        self.u_data_p = self.opti.parameter(self.n_elec, n_steps)
        self.y_data_p = self.opti.parameter(self.n_channels, n_steps)

        # Unknown initial history: the first max_delay+1 states are decision variables, and
        # the first max_delay predictions come straight from them.
        self.x0_var = self.opti.variable(*self.model.state_shape)
        self.x_vars = [self.opti.variable(*self.model.state_shape) for _ in range(self.max_delay)]
        Y_pred = [f_out(self.x_vars[k], *free_actuals) for k in range(self.max_delay)]

        # Roll the rest of the horizon forward with PCMS sparsification (stride D).
        history = [self.x_vars[i] for i in range(self.max_delay - 1, -1, -1)] + [self.x0_var]
        controls = [self.u_data_p[:, k] for k in range(self.max_delay, n_steps)]
        y_rest, self.ms_vars = shoot(
            self.opti,
            f_step,
            f_out,
            history,
            controls,
            self.model.state_shape,
            free=free_actuals,
            stride=D,
            step0=self.max_delay,
        )
        Y_pred.extend(y_rest)

        # Single least-squares cost over the whole horizon; ``Y_pred`` is ordered to match
        # the columns of ``y_data_p``, so this equals the per-step sum of ``sumsqr`` terms.
        self.Y_pred_sym = ca.horzcat(*Y_pred)
        cost = ca.sumsqr(self.Y_pred_sym - self.y_data_p)

        # Optional differentiable log Welch PSD term. ``y_data_p`` is a parameter, so its
        # PSD is the same symbolic map and updates whenever the data is set in ``solve``.
        self.psd_freqs: np.ndarray | None = None
        if psd_loss:
            nperseg = int(psd_loss.get("nperseg", min(256, n_steps)))
            band = psd_loss.get("band")
            plan = _welch_plan(
                n_steps,
                dt,
                nperseg,
                int(psd_loss.get("noverlap", nperseg // 2)),
                tuple(band) if band is not None else None,
            )
            self.psd_freqs = plan.freqs
            psd_cost = ca.sumsqr(_log_welch_psd(self.Y_pred_sym, plan) - _log_welch_psd(self.y_data_p, plan))
            cost = float(psd_loss.get("time_weight", 1.0)) * cost + float(psd_loss["weight"]) * psd_cost

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

    def _build_param_var(self, p: str, base_params: JansenRitParams) -> ca.MX:
        """Build the Opti decision variable (prior + bounds) for free parameter ``p``.

        Returns the actual MX grafted onto ``model.free_syms[p]`` and read back after the
        solve -- a plain bounded variable, or for ``w_weights`` the symmetric zero-diagonal
        matrix mirrored from an optimised strict-upper-triangle variable.
        """
        if p == "w_weights":
            return self._build_w_weights_var(base_params)

        if not hasattr(base_params, p):
            msg = f"Unknown parameter {p}"
            raise ValueError(msg)

        v = self.opti.variable(*self.model.free_syms[p].shape)

        prior = getattr(base_params, p)
        if p == "eeg_gain":
            prior = np.abs(base_params.eeg_gain).mean() or 1.0
        elif p == "gamma":
            g = np.asarray(base_params.gamma, dtype=np.float64)
            prior = g.reshape(1, -1) if g.ndim == 1 else g
        self.opti.set_initial(v, prior)

        lb, ub = -np.inf, np.inf
        for f in dataclasses.fields(base_params):
            if f.name == p:
                lb, ub = f.metadata.get("bounds", (-np.inf, np.inf))
                break
        self.opti.subject_to(self.opti.bounded(lb, v, ub))

        return v

    def _build_w_weights_var(self, base_params: JansenRitParams) -> ca.MX:
        """Build the symmetric, zero-diagonal ``w_weights`` decision variable.

        Only the strict upper triangle is optimised (nonnegative, capped at ``_W_UB_FACTOR``
        times the largest base weight); the returned MX is the mirrored full matrix.
        """
        iu = np.triu_indices(self.n_nodes, k=1)
        base_tri = np.asarray(base_params.w_weights, dtype=np.float64)[iu]
        w_ub = self._W_UB_FACTOR * (float(np.abs(base_params.w_weights).max()) or 1.0)

        w_var = self.opti.variable(1, base_tri.size)
        self.opti.set_initial(w_var, base_tri.reshape(1, -1))
        self.opti.subject_to(self.opti.bounded(0.0, w_var, w_ub))

        edges = {}
        for idx, (i, j) in enumerate(zip(*iu, strict=True)):
            edges[int(i), int(j)] = edges[int(j), int(i)] = w_var[0, idx]
        return ca.vertcat(
            *[ca.horzcat(*[edges.get((i, j), ca.MX(0.0)) for j in range(self.n_nodes)]) for i in range(self.n_nodes)]
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SysIDSolver:
        """Create a SysIDSolver from a configuration dictionary.

        The base network is read from the nested ``base_model`` block (a
        :class:`JansenRitParams` config, optionally with a ``w_weights_scale`` to rescale
        the prior connectivity). ``D`` defaults to ``n_steps`` (single shooting).
        """
        base_cfg = config.get("base_model", {})
        base_params = JansenRitParams.from_config(base_cfg)
        if "w_weights_scale" in base_cfg:
            base_params = dataclasses.replace(
                base_params, w_weights=base_params.w_weights * float(base_cfg["w_weights_scale"])
            )
        n_steps = int(config["n_steps"])
        return cls(
            dt=float(config["dt"]),
            base_params=base_params,
            free_params=config.get("free_params", ["A"]),
            n_steps=n_steps,
            D=int(config.get("D", n_steps)),
            ipopt_options=config.get("ipopt_options"),
            psd_loss=config.get("psd_loss"),
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

        kwargs = {k: sol.value(self._free_vars[k]) for k in self.free_params}

        return SysIDResult(
            free_params=kwargs,
            cost=cost_val,
            trajectory=y_pred_val,
        )
