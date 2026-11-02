from __future__ import annotations

import dataclasses
import time
from typing import TYPE_CHECKING, Any, Self

import jax.numpy as jnp
import numpy as np
from simulate.controller import Controller
from trajopt.mpc import MPC
from trajopt.problem import (
    BoundaryConditions,
    Problem,
)
from trajopt.program import WarmStart
from trajopt.transcription.ipopt import Ipopt
from trajopt.transcription.single_shooting import SingleShooting

from neuro.control.diagnostics import (
    ACTIVE_CURRENT_THRESHOLD,
    _rollout_states,
    decompose_cost,
    planned_outputs,
)
from neuro.control.model_wrappers import EpigraphInferenceModel
from neuro.control.problems import _build_problem
from neuro.control.solvers import (
    IPOPT_DEFAULTS,
    _build_solver,
    canonicalize_duals,
)
from neuro.predictor.inference import InferencePredictor

if TYPE_CHECKING:
    from trajopt.transcription.result import Solver

    from neuro.types import FloatArray


@dataclasses.dataclass(frozen=True)
class TrajOptMPCLog:
    """Decision diagnostics and unshifted plans: outputs ``(H+1, p)``, Control Currents ``(H, m)``."""

    u: FloatArray
    cost: float
    success: bool
    warmup: bool
    solve_time: float
    predicted_y: FloatArray
    planned_u: FloatArray
    status: str = "Solve_Succeeded"
    planned_active: FloatArray = dataclasses.field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    active_count: float = 0.0
    cost_active: float = 0.0
    cost_spectral: float = 0.0
    cost_quadratic_effort: float = 0.0
    cost_sparse_effort: float = 0.0
    cost_tracking: float = 0.0
    normalization: str = "channel_mean"


_CONTINUOUS_POLISH_THRESHOLD = 0.20


class TrajOptMPCController(Controller[TrajOptMPCLog]):
    """Receding-horizon MPC for the Waveform Predictor, driven by trajopt's :class:`~trajopt.mpc.MPC`.

    The driver owns the horizon's boundary conditions and warm start, and holds one compiled
    ``Program`` for the life of the run, so the solver's cores are built once rather than per
    step. This controller owns what the driver does not: the true State-Absorbed Predictor state and
    ``u_last``, which are what turn a measurement into the driver's ``x0``.
    """

    def __init__(
        self,
        dt: float,
        problem: Problem,
        solver: Solver | dict[str, Any] | None = None,
        init_mode: str = "shift",
    ) -> None:
        """Initialize the persistent MPC driver and physical Control Current history.

        Parameters
        ----------
        dt
            Controller update step in seconds; should equal the Predictor's native dt.
        problem
            The trajopt optimal-control problem: model adapter + objective + constraint list.
        solver
            Solver backend instance or ``{class_path, ...}`` config dict. When omitted,
            `_default_solver` picks ``SingleShooting(Ipopt)``.
        init_mode
            Warm-start initialization mode: ``"shift"`` (default receding horizon) or ``"cont_polish"``.
        """
        super().__init__(dt)
        model = problem.model
        if not isinstance(model, InferencePredictor):
            msg = f"problem.model ({type(model).__name__}) does not implement the InferencePredictor priming seam"
            raise TypeError(msg)
        self.problem = problem
        self.model = model
        self.solver = _build_solver(solver, problem)
        self.horizon = int(problem.N - 1)
        self._integer_mode = bool(problem.binary_control_indices)
        self.init_mode = str(init_mode)

        # The unprimed EEG window is NaN padding; the driver seeds a full state trajectory from
        # x0, not just the controls, so it is zeroed to keep that seed finite for every solver.
        unprimed = jnp.nan_to_num(jnp.asarray(self.model.initial_state()), nan=0.0)
        solver_driver = (
            self.solver
            if self._integer_mode or problem.constraints.horizon_constraints
            else canonicalize_duals(self.solver)
        )
        self.mpc = MPC(problem, solver_driver, x0=unprimed)
        self._state = np.asarray(self.model.initial_state(), dtype=np.float64)
        self._u_last = np.zeros(model.m, dtype=np.float64)
        if (
            self._integer_mode
            or isinstance(model, EpigraphInferenceModel)
            or model.m == 2 * getattr(model, "n_controls", -1)
        ):
            self.n_controls = int(self.model.n_controls)
        else:
            self.n_controls = int(model.m)
        self.n_electrodes = self.n_controls
        self._epigraph_mode = bool(self.model.m == 2 * self.n_electrodes)
        m_guess = self.model.m if self._epigraph_mode else self.n_controls
        self._u_guess = np.zeros((self.horizon, m_guess), dtype=np.float64)
        self._active_guess = np.zeros(self.horizon, dtype=np.float64) if self._integer_mode else None

        cont_prob = getattr(problem, "continuous_problem", None)
        if self._integer_mode and self.init_mode == "cont_polish" and cont_prob is not None:
            self._cont_mpc: MPC | None = MPC(
                cont_prob, canonicalize_duals(SingleShooting(solver=Ipopt(options=IPOPT_DEFAULTS))), x0=unprimed
            )
            self._cont_u_max = float(getattr(problem, "u_max", [2.0])[0])
            self._max_cap = getattr(problem, "max_active_intervals", None)
        else:
            self._cont_mpc = None

    def _extract_plan(self, plan: FloatArray, *, success: bool) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Extract physical Control Currents and activation flags, applying zero on solve failure."""
        if self._integer_mode:
            planned_u = plan[:, :-1].copy()
            planned_active = plan[:, -1].copy()
        else:
            planned_u = plan[:, : self.n_electrodes].copy()
            planned_active = np.any(np.abs(planned_u) >= ACTIVE_CURRENT_THRESHOLD, axis=-1).astype(np.float64)
        u_cmd = planned_u[0].copy() if success else np.zeros(self.n_electrodes, dtype=np.float64)
        return planned_u, planned_active, u_cmd

    def _prepare_warm_start_u(self, initial_u: FloatArray, initial_active: FloatArray | None) -> FloatArray:
        """Append activation or epigraph coordinates to a physical Control Current guess."""
        u_init = np.asarray(initial_u, dtype=np.float64).reshape(self.horizon, -1)
        if self._integer_mode:
            if initial_active is not None:
                act_init = np.asarray(initial_active, dtype=np.float64).reshape(self.horizon, 1)
            else:
                act_init = (np.any(u_init != 0, axis=1, keepdims=True)).astype(np.float64)
            return np.hstack([u_init, act_init])
        if self._epigraph_mode and u_init.shape[1] == self.n_electrodes:
            return np.hstack([u_init, np.abs(u_init)])
        return u_init

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate from a config dict, dispatching the Problem and Solver factories.

        Follows the standard ``{class_path, ...}`` pattern: ``problem`` is either a Problem instance
        or a ``{class_path, ...}`` dict naming a Problem-building factory (e.g.
        ``neuro.control.problems.build_waveform_problem``); ``solver`` is an optional
        ``{class_path, ...}`` dict naming a solver backend (e.g. ``trajopt.solvers.altro.ALTRO``).
        When ``solver`` is omitted, the benchmark-winning default is selected automatically.
        """
        problem = _build_problem(config["problem"])
        return cls(
            dt=float(config["dt"]),
            problem=problem,
            solver=_build_solver(config.get("solver"), problem),
            init_mode=str(config.get("init_mode", "shift")),
        )

    @property
    def is_integer_mode(self) -> bool:
        """Whether this controller solves mixed-integer activation decisions."""
        return self._integer_mode

    def update(
        self,
        t: float,
        ref: FloatArray,  # noqa: ARG002 -- the goal is baked into the objective
        x_hat: FloatArray,
    ) -> tuple[FloatArray, TrajOptMPCLog]:
        """Absorb the measurement and log physical Control Currents and Costs before shifting."""
        self._state = np.asarray(self.model.absorb(self._state, np.asarray(x_hat).reshape(-1), self._u_last))

        if not self.model.is_ready(self._state):
            self._u_last = np.zeros(self.model.m, dtype=np.float64)
            u_zero = np.zeros(self.n_electrodes, dtype=np.float64)
            knots = self.mpc.problem.N
            planned_active = np.full(knots - 1, np.nan, dtype=np.float64)
            active_count = float("nan")
            return u_zero, TrajOptMPCLog(
                u=u_zero,
                cost=0.0,
                success=True,
                warmup=True,
                status="warmup",
                solve_time=0.0,
                predicted_y=np.full((knots, self.model.p), np.nan),
                planned_u=np.full((knots - 1, self.n_electrodes), np.nan),
                planned_active=planned_active,
                active_count=active_count,
                cost_active=0.0,
                cost_spectral=0.0,
                cost_quadratic_effort=0.0,
                cost_sparse_effort=0.0,
                cost_tracking=0.0,
                normalization="channel_mean",
            )

        self.mpc.measure(jnp.asarray(self._state), t)
        if self._cont_mpc is not None:
            self._cont_mpc.measure(jnp.asarray(self._state), t)
            res_cont = self._cont_mpc.solve()
            self._cont_mpc.shift(self.dt)
            u_cont = np.asarray(res_cont.trajectory.U)
            envelope = np.max(np.abs(u_cont), axis=1) / self._cont_u_max
            delta_thresh = np.where(envelope > _CONTINUOUS_POLISH_THRESHOLD, 1.0, 0.0)
            if self._max_cap is not None and np.sum(delta_thresh) > self._max_cap:
                top_k = np.argsort(envelope)[-self._max_cap :]
                delta_thresh = np.zeros(self.horizon)
                delta_thresh[top_k] = 1.0
            u_full = np.hstack([u_cont, delta_thresh[:, None]])
            u_full_jax = jnp.asarray(u_full, dtype=jnp.float64)
            x0_arr = jnp.asarray(self._state, dtype=jnp.float64)
            x_rollout = _rollout_states(self.problem.model, x0_arr, u_full_jax, float(self.problem.dt[0]))
            self.mpc._ws = self.mpc._ws.with_primal(  # noqa: SLF001 -- inject continuous relaxation primal guess into driver warm start
                self.problem, X=x_rollout, U=u_full_jax
            )
        started = time.perf_counter()
        solved = self.mpc.solve()
        solve_time = time.perf_counter() - started
        u_solved = np.asarray(self.mpc.controls[0], dtype=np.float64)
        cost = float(self.mpc.cost())
        predicted_y = np.asarray(planned_outputs(self.model, self.mpc.states, self.mpc.controls, t, self.dt)).copy()
        planned_u, planned_active, u_cmd = self._extract_plan(
            np.asarray(self.mpc.controls), success=bool(solved.success)
        )
        costs_decomp = decompose_cost(self.mpc.problem, self.mpc.states, self.mpc.controls, t, self.dt)
        self.mpc.shift(self.dt)
        self._u_last = u_solved
        if solved.success:
            if self._epigraph_mode:
                u_shifted = np.vstack((planned_u[1:], planned_u[-1:]))
                self._u_guess = np.hstack([u_shifted, np.abs(u_shifted)])
            else:
                self._u_guess = np.vstack((planned_u[1:], planned_u[-1:]))
            if self._integer_mode and planned_active is not None:
                self._active_guess = np.concatenate((planned_active[1:], planned_active[-1:]))
        return u_cmd, TrajOptMPCLog(
            u=u_cmd.copy(),
            cost=cost,
            success=bool(solved.success),
            warmup=False,
            status=str(solved.status),
            solve_time=solve_time,
            predicted_y=predicted_y,
            planned_u=planned_u,
            planned_active=planned_active,
            active_count=costs_decomp["active_count"],
            cost_active=costs_decomp["cost_active"],
            cost_spectral=costs_decomp["cost_spectral"],
            cost_quadratic_effort=costs_decomp["cost_quadratic_effort"],
            cost_sparse_effort=costs_decomp["cost_sparse_effort"],
            cost_tracking=costs_decomp["cost_tracking"],
            normalization=costs_decomp["normalization"],
        )

    def solve_state(
        self,
        x0: FloatArray,
        initial_u: FloatArray | None = None,
        initial_active: FloatArray | None = None,
        time_limit_s: float | None = None,  # noqa: ARG002 -- interface compatibility with MINLP controllers
    ) -> TrajOptMPCLog:
        """Solve a fixed Predictor state with an optional primal guess, without advancing controller history.

        Parameters
        ----------
        x0
            Predictor state vector of shape ``(n,)``.
        initial_u
            Control guess of shape ``(horizon, n_controls)``.
        initial_active
            Binary activation guess of shape ``(horizon,)`` for integer mode.
        time_limit_s
            Optional solver execution time limit in seconds.
        """
        x0_arr = jnp.asarray(x0, dtype=jnp.float64).reshape(self.model.n)
        bc = BoundaryConditions(x0=x0_arr, t0=jnp.asarray(0.0, dtype=jnp.float64))
        if initial_u is not None:
            u_full = self._prepare_warm_start_u(initial_u, initial_active)
            ws = WarmStart.cold(self.problem, x0_arr)
            u_full_jax = jnp.asarray(u_full, dtype=jnp.float64)
            x_rollout = _rollout_states(self.problem.model, x0_arr, u_full_jax, float(self.problem.dt[0]))
            ws = ws.with_primal(self.problem, X=x_rollout, U=u_full_jax)
        else:
            ws = WarmStart.cold(self.problem, x0_arr)

        started = time.perf_counter()
        try:
            res = self.mpc.program.solve(bc, ws)
            solve_time = time.perf_counter() - started
            success = bool(res.success)
            status = str(res.status)
        except Exception as exc:  # noqa: BLE001 -- report solver failure cleanly
            solve_time = time.perf_counter() - started
            success = False
            status = str(exc)
            res = None

        if res is None:
            planned_u = np.full((self.horizon, self.n_controls), np.nan)
            planned_active = np.full(self.horizon, np.nan, dtype=np.float64)
            predicted_y = np.full((self.horizon + 1, self.model.p), np.nan)
            u_cmd = np.zeros(self.n_controls, dtype=np.float64)
            return TrajOptMPCLog(
                u=u_cmd,
                cost=float("inf"),
                success=False,
                warmup=False,
                status=status,
                solve_time=solve_time,
                predicted_y=predicted_y,
                planned_u=planned_u,
                planned_active=planned_active,
                active_count=float("nan"),
                cost_active=0.0,
                cost_spectral=0.0,
                cost_quadratic_effort=0.0,
                cost_sparse_effort=0.0,
                cost_tracking=0.0,
                normalization="channel_mean",
            )

        planned_u, planned_active, u_cmd = self._extract_plan(np.asarray(res.trajectory.U), success=success)

        predicted_y = np.asarray(planned_outputs(self.model, res.trajectory.X, res.trajectory.U, 0.0, self.dt)).copy()
        costs_decomp = decompose_cost(self.problem, res.trajectory.X, res.trajectory.U, 0.0, self.dt)
        return TrajOptMPCLog(
            u=u_cmd,
            cost=float(res.cost),
            success=success,
            warmup=False,
            status=status,
            solve_time=solve_time,
            predicted_y=predicted_y,
            planned_u=planned_u,
            planned_active=planned_active,
            active_count=costs_decomp["active_count"],
            cost_active=costs_decomp["cost_active"],
            cost_spectral=costs_decomp["cost_spectral"],
            cost_quadratic_effort=costs_decomp["cost_quadratic_effort"],
            cost_sparse_effort=costs_decomp["cost_sparse_effort"],
            cost_tracking=costs_decomp["cost_tracking"],
            normalization=costs_decomp["normalization"],
        )
