from __future__ import annotations

import dataclasses
import time
from typing import TYPE_CHECKING, cast

import casadi as ca
import jax.numpy as jnp
import numpy as np

from neuro.control.casadi import CasADiMPCController, CasADiMPCLog, CasADiObservableProblem
from neuro.control.condensed import condense_affine_observable
from neuro.control.problems import _affine_observable_hinge
from neuro.predictor.inference import ObservableMLPModel

if TYPE_CHECKING:
    from neuro.types import FloatArray


_FEASIBILITY_TOL = 1e-6


class CondensedBonminController(CasADiMPCController):
    """Solve DMDc activation with an affine prediction map and a convex hinge objective."""

    def _build_solver(self) -> None:
        """Compile the condensed direct hinge and optional L1 epigraph with binary gates."""
        model = self.model
        problem = self.problem
        if not isinstance(model, ObservableMLPModel):
            msg = "Condensed Bonmin requires an affine Observable Predictor."
            raise TypeError(msg)
        if (
            not isinstance(problem, CasADiObservableProblem)
            or not self._integer_mode
            or problem.integer_solver != "bonmin"
            or model.depth != 0
            or model.residual
            or problem.envelope is None
        ):
            msg = "Condensed Bonmin requires an Observable activation problem with a hinge reference."
            raise ValueError(msg)
        self._used_backup = False
        h, m = self.horizon, self.n_controls
        p = model.n_outputs
        exact_l1 = problem.w_u_l1 > 0.0
        hinge_cost = _affine_observable_hinge(model, problem.envelope, problem.w_hinge / h)
        self._evaluate_affine, self._gain = condense_affine_observable(model, hinge_cost, h, self.dt)
        self._reference = np.tile(np.asarray(problem.envelope.power).reshape(-1), h)
        u = ca.MX.sym("current", h * m)
        a = ca.MX.sym("active", h)
        z = ca.MX.sym("abs_current", h * m if exact_l1 else 0)
        p_sym = ca.MX.sym("free_residual", h * p)
        residual = p_sym + ca.mtimes(ca.sparsify(ca.DM(self._gain), 0.0), u)
        hinge = problem.w_hinge / (h * p) * ca.sumsqr(ca.fmax(residual, 0.0))
        effort = problem.w_u / h * ca.sumsqr(u)
        active_cost = (problem.w_active or 0.0) / h * ca.sum1(a)
        sparse_cost = problem.w_u_l1 / h * ca.sum1(z) if exact_l1 else 0.0
        total = hinge + effort + active_cost + sparse_cost
        gates = ca.vec(ca.mtimes(ca.DM(problem.u_max), a.T))
        g, lower, upper = [u - gates, -u - gates], [-np.inf] * (2 * h * m), [0.0] * (2 * h * m)
        if exact_l1:
            g.extend((u - z, -u - z))
            lower.extend([-np.inf] * (2 * h * m))
            upper.extend([0.0] * (2 * h * m))
        if problem.kirchhoff:
            g.append(ca.sum1(ca.reshape(u, m, h)).T)
            lower.extend([0.0] * h)
            upper.extend([0.0] * h)
        if problem.max_active_intervals is not None:
            g.append(ca.sum1(a))
            lower.append(-np.inf)
            upper.append(float(problem.max_active_intervals))
        self._lbx = np.r_[np.tile(-problem.u_max, h), np.zeros(h + z.numel())]
        self._ubx = np.r_[np.tile(problem.u_max, h), np.ones(h), np.full(z.numel(), np.inf)]
        self._lbg = np.asarray(lower)
        self._ubg = np.asarray(upper)
        self._solver = ca.nlpsol(
            "condensed_bonmin",
            "bonmin",
            {"x": ca.vertcat(u, a, z), "p": p_sym, "f": total, "g": ca.vertcat(*g)},
            {
                "print_time": False,
                "calc_lam_p": False,
                "discrete": [False] * (h * m) + [True] * h + [False] * z.numel(),
                "bonmin.algorithm": "B-BB",
                "bonmin.print_level": 0,
                **problem.solver_options,
            },
        )
        self._evaluate_affine(jnp.zeros(model.n), jnp.zeros((h, m)))[1].block_until_ready()

    def _plan_costs(
        self, residual: FloatArray, currents: FloatArray, gates: FloatArray
    ) -> tuple[float, float, float, float, float]:
        """Price a physical plan using the same residual as the Bonmin epigraph."""
        problem = self.problem
        hinge = problem.w_hinge * float(np.mean(np.maximum(residual, 0.0) ** 2))
        effort = problem.w_u * float(np.sum(currents**2)) / self.horizon
        sparse = problem.w_u_l1 * float(np.sum(np.abs(currents))) / self.horizon
        activation = (problem.w_active or 0.0) * float(np.mean(gates))
        return hinge + effort + sparse + activation, hinge, effort, sparse, activation

    def _feasible_plan(self, currents: FloatArray, activation: FloatArray) -> tuple[FloatArray, FloatArray] | None:
        """Validate a binary plan and clean numerical leakage within the active current bounds."""
        tol = _FEASIBILITY_TOL
        binary = np.rint(activation)
        cap = self.problem.max_active_intervals
        if (
            not np.isfinite(currents).all()
            or not np.isfinite(activation).all()
            or np.max(np.abs(activation - binary)) > tol
            or np.any((binary < 0) | (binary > 1))
            or np.max(np.abs(currents) - binary[:, None] * self.problem.u_max) > tol
            or (cap is not None and binary.sum() > cap)
            or (self.problem.kirchhoff and np.max(np.abs(currents.sum(axis=1))) > tol)
        ):
            return None
        movable = self.problem.u_max > 0.0
        cleaned = np.where(movable, currents, 0.0)
        if self.problem.kirchhoff:
            correction = cleaned.sum(axis=1, keepdims=True) / max(1, np.count_nonzero(movable))
            cleaned[:, movable] -= correction
        cleaned[binary == 0] = 0.0
        ratios = np.divide(np.abs(cleaned), self.problem.u_max, out=np.zeros_like(cleaned), where=movable)
        scale = np.maximum(1.0, ratios.max(axis=1, keepdims=True))
        return cleaned / scale, binary

    def solve_state(
        self,
        x0: FloatArray,
        initial_u: FloatArray | None = None,
        initial_active: FloatArray | None = None,
    ) -> CasADiMPCLog:
        """Select a validated Bonmin incumbent or a feasible shifted backup at the measured state."""
        started = time.perf_counter()
        currents = np.zeros((self.horizon, self.n_controls))
        activation = np.zeros(self.horizon)
        state = jnp.asarray(x0, dtype=jnp.float64).reshape(self.model.n)
        _, free_residual = self._evaluate_affine(state, jnp.asarray(currents))
        free = np.asarray(free_residual)
        residual = free
        costs = self._plan_costs(residual, currents, activation)
        backup_source = "FALLBACK_ZERO"
        if initial_u is not None and initial_active is not None:
            shifted = self._feasible_plan(initial_u, initial_active)
            if shifted is not None:
                shifted_residual = free + self._gain @ shifted[0].ravel()
                shifted_costs = self._plan_costs(shifted_residual, *shifted)
                if shifted_costs[0] < costs[0]:
                    currents, activation = shifted
                    residual, costs = shifted_residual, shifted_costs
                    backup_source = "FALLBACK_SHIFT"

        incumbent, status, solve_time = self._solve_incumbent(free, currents, activation)
        if incumbent is not None:
            incumbent_residual = free + self._gain @ incumbent[0].ravel()
            incumbent_costs = self._plan_costs(incumbent_residual, *incumbent)
            if incumbent_costs[0] <= costs[0] + 1e-8:
                currents, activation = incumbent
                residual, costs = incumbent_residual, incumbent_costs
            else:
                incumbent = None

        total, hinge, effort, sparse, active = costs
        future = (residual + self._reference).reshape(self.horizon, self.model.n_outputs)
        predicted_y = np.vstack((np.asarray(self.model.output(state)), future))
        return CasADiMPCLog(
            u=currents[0].copy(),
            success=incumbent is not None,
            warmup=False,
            status=status if incumbent is not None else f"{backup_source}:{status}",
            cost=total,
            cost_quadratic_effort=effort,
            cost_sparse_effort=sparse,
            cost_spectral=hinge,
            cost_active=active,
            solve_time=solve_time,
            planned_u=currents,
            planned_active=activation,
            predicted_y=predicted_y,
            active_count=float(activation.sum()),
            fallback=incumbent is None,
            replan_time=time.perf_counter() - started,
        )

    def _solve_incumbent(
        self, free: FloatArray, currents: FloatArray, activation: FloatArray
    ) -> tuple[tuple[FloatArray, FloatArray] | None, str, float]:
        """Solve from a feasible lifted seed and validate the returned binary incumbent."""
        seed_u = currents.ravel()
        seed = np.r_[seed_u, activation]
        if self.problem.w_u_l1 > 0.0:
            seed = np.r_[seed, np.abs(seed_u)]
        started = time.perf_counter()
        try:
            result = self._solver(x0=seed, p=free, lbx=self._lbx, ubx=self._ubx, lbg=self._lbg, ubg=self._ubg)
        except RuntimeError as exc:
            return None, str(exc), time.perf_counter() - started
        solve_time = time.perf_counter() - started
        stats = self._solver.stats()
        status = str(stats.get("return_status", "unknown"))
        if not stats["success"] and status != "LIMIT_EXCEEDED":
            return None, status, solve_time
        decision = np.asarray(result["x"], dtype=np.float64).ravel()
        planned_u = decision[: currents.size].reshape(currents.shape)
        planned_active = decision[currents.size : currents.size + self.horizon]
        incumbent = self._feasible_plan(planned_u, planned_active)
        if incumbent is not None and not stats["success"]:
            status = "FEASIBLE_AT_LIMIT"
        return incumbent, status, solve_time

    def _plan(self) -> CasADiMPCLog:
        """Shift the last feasible plan with an off tail and play the validated replacement."""
        started = time.perf_counter()
        steps = self._playback.steps
        shifted_u = np.roll(self._u_guess, -steps, axis=0)
        shifted_a = np.roll(self._active_guess, -steps)
        shifted_u[-steps:] = 0.0
        shifted_a[-steps:] = 0.0
        log = self.solve_state(self._state, shifted_u, shifted_a)
        self._u_guess = log.planned_u.copy()
        self._active_guess = cast("FloatArray", log.planned_active).copy()
        self._used_backup = log.fallback
        # Playback accepts the validated backup, while the exported status still records the failed solve.
        return dataclasses.replace(log, success=True, replan_time=time.perf_counter() - started)

    def update(self, t: float, ref: FloatArray, x_hat: FloatArray) -> tuple[FloatArray, CasADiMPCLog]:
        """Report a usable backup as fallback while emitting its validated currents."""
        current, log = super().update(t, ref, x_hat)
        if self._used_backup:
            return current, dataclasses.replace(log, success=False, fallback=True)
        return current, log
