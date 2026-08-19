from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self

import casadi as ca
import numpy as np

if TYPE_CHECKING:
    from neuro.control.nlp import MPCNlp
    from neuro.types import FloatArray


@dataclasses.dataclass(frozen=True)
class MPCSolveResult:
    """Numerical result and diagnostics of an MPC NLP solve."""

    u_opt: FloatArray
    cost: float
    success: bool
    n_iter: int
    capped: bool
    fallback: bool


class MPCSolver(Protocol):
    """Protocol for MPC NLP numerical solvers."""

    def solve(self, x0: FloatArray, w0: FloatArray) -> MPCSolveResult:
        """Solve the NLP for parameter ``x0`` and initial guess ``w0``."""
        ...


class IpoptMPCSolver:
    """IPOPT solver for MPC."""

    def __init__(self, solver: ca.Function, mpc_nlp: MPCNlp) -> None:
        """Initialize with a compiled IPOPT nlpsol function and the MPCNlp."""
        self._solver = solver
        self.mpc_nlp = mpc_nlp

    @property
    def function(self) -> ca.Function:
        """Underlying compiled CasADi nlpsol function."""
        return self._solver

    @classmethod
    def build(
        cls,
        mpc_nlp: MPCNlp,
        *,
        max_iter: int = 100,
        max_cpu_time: float | None = None,
        expand: bool = False,
        ipopt_options: dict[str, Any] | None = None,
    ) -> Self:
        """Build a CasADi IPOPT solver from an MPCNlp."""
        opts: dict[str, Any] = {
            "print_time": False,
            "expand": expand,
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.max_iter": max_iter,
            "ipopt.hessian_approximation": "limited-memory",
        }
        if max_cpu_time is not None:
            opts["ipopt.max_cpu_time"] = max_cpu_time
        if ipopt_options:
            for k, v in ipopt_options.items():
                opts[f"ipopt.{k}" if not k.startswith("ipopt.") else k] = v
        solver = ca.nlpsol("mpc_ipopt", "ipopt", mpc_nlp.nlp, opts)
        return cls(solver, mpc_nlp)

    def solve(self, x0: FloatArray, w0: FloatArray) -> MPCSolveResult:
        """Solve using IPOPT."""
        sol = self._solver(
            x0=w0,
            lbx=self.mpc_nlp.lbx,
            ubx=self.mpc_nlp.ubx,
            lbg=self.mpc_nlp.lbg,
            ubg=self.mpc_nlp.ubg,
            p=x0,
        )
        stats = self._solver.stats()
        status = stats["return_status"]
        success = status in {"Solve_Succeeded", "Solved_To_Acceptable_Level"}
        return MPCSolveResult(
            u_opt=np.asarray(sol["x"]).reshape(-1),
            cost=float(sol["f"]),
            success=success,
            n_iter=int(stats.get("iter_count", 0)),
            capped=status == "Maximum_Iterations_Exceeded",
            fallback=False,
        )


class SqpMPCSolver:
    """Standalone SQP solver for MPC."""

    def __init__(self, solver: ca.Function, mpc_nlp: MPCNlp) -> None:
        """Initialize with a compiled SQP nlpsol function and the MPCNlp."""
        self._solver = solver
        self.mpc_nlp = mpc_nlp

    @property
    def function(self) -> ca.Function:
        """Underlying compiled CasADi nlpsol function."""
        return self._solver

    @classmethod
    def build(  # noqa: PLR0913
        cls,
        mpc_nlp: MPCNlp,
        *,
        qpsol: Literal["qpoases", "osqp", "qrqp"] = "qpoases",
        hessian_approximation: Literal["limited-memory", "exact"] = "limited-memory",
        max_iter: int = 15,
        lbfgs_memory: int = 10,
        expand: bool = False,
        sqp_options: dict[str, Any] | None = None,
    ) -> Self:
        """Build a CasADi SQP solver from an MPCNlp."""
        qpsol_opts: dict[str, Any] = {"print_time": False}
        if qpsol == "osqp":
            qpsol_opts["osqp"] = {"verbose": False, "eps_abs": 1e-4, "eps_rel": 1e-4}
        elif qpsol == "qpoases":
            qpsol_opts["printLevel"] = "none"
        elif qpsol == "qrqp":
            qpsol_opts.update({"print_header": False, "print_iter": False, "print_info": False})

        opts: dict[str, Any] = {
            "print_time": False,
            "print_header": False,
            "print_iteration": False,
            "print_status": False,
            "expand": expand,
            "qpsol": qpsol,
            "qpsol_options": qpsol_opts,
            "hessian_approximation": hessian_approximation,
            "max_iter": max_iter,
        }
        if hessian_approximation == "limited-memory":
            opts["lbfgs_memory"] = lbfgs_memory
        if sqp_options:
            opts.update(sqp_options)
        solver = ca.nlpsol("mpc_sqp", "sqpmethod", mpc_nlp.nlp, opts)
        return cls(solver, mpc_nlp)

    def solve(self, x0: FloatArray, w0: FloatArray) -> MPCSolveResult:
        """Solve using standalone SQP."""
        try:
            sol = self._solver(
                x0=w0,
                lbx=self.mpc_nlp.lbx,
                ubx=self.mpc_nlp.ubx,
                lbg=self.mpc_nlp.lbg,
                ubg=self.mpc_nlp.ubg,
                p=x0,
            )
            stats = self._solver.stats()
            status = stats["return_status"]
            success = status == "Solve_Succeeded"
            return MPCSolveResult(
                u_opt=np.asarray(sol["x"]).reshape(-1),
                cost=float(sol["f"]),
                success=success,
                n_iter=int(stats.get("iter_count", 0)),
                capped=status == "Maximum_Iterations_Exceeded",
                fallback=False,
            )
        except Exception:  # noqa: BLE001
            return MPCSolveResult(
                u_opt=np.zeros(w0.size, dtype=np.float64),
                cost=float("nan"),
                success=False,
                n_iter=0,
                capped=False,
                fallback=False,
            )


class SqpFallbackMPCSolver:
    """SQP solver with warm-started IPOPT fallback on non-convergence or exception."""

    def __init__(self, sqp_solver: SqpMPCSolver, ipopt_solver: IpoptMPCSolver) -> None:
        """Initialize with SQP and IPOPT solver objects."""
        self._sqp_solver = sqp_solver
        self._ipopt_solver = ipopt_solver

    @property
    def sqp_function(self) -> ca.Function:
        """Underlying compiled CasADi SQP nlpsol function."""
        return self._sqp_solver.function

    @property
    def ipopt_function(self) -> ca.Function:
        """Underlying compiled CasADi IPOPT nlpsol function."""
        return self._ipopt_solver.function

    @classmethod
    def build(  # noqa: PLR0913
        cls,
        mpc_nlp: MPCNlp,
        *,
        max_iter: int = 100,
        max_cpu_time: float | None = None,
        expand: bool = False,
        ipopt_options: dict[str, Any] | None = None,
        sqp_qpsol: Literal["qpoases", "osqp", "qrqp"] = "qpoases",
        sqp_hessian: Literal["limited-memory", "exact"] = "limited-memory",
        sqp_max_iter: int = 15,
        sqp_lbfgs_memory: int = 10,
        sqp_options: dict[str, Any] | None = None,
    ) -> Self:
        """Build both SQP and IPOPT solvers from an MPCNlp."""
        sqp_obj = SqpMPCSolver.build(
            mpc_nlp,
            qpsol=sqp_qpsol,
            hessian_approximation=sqp_hessian,
            max_iter=sqp_max_iter,
            lbfgs_memory=sqp_lbfgs_memory,
            expand=expand,
            sqp_options=sqp_options,
        )
        ipopt_obj = IpoptMPCSolver.build(
            mpc_nlp,
            max_iter=max_iter,
            max_cpu_time=max_cpu_time,
            expand=expand,
            ipopt_options=ipopt_options,
        )
        return cls(sqp_obj, ipopt_obj)

    def solve(self, x0: FloatArray, w0: FloatArray) -> MPCSolveResult:
        """Attempt SQP, falling back to IPOPT warm-started from the SQP iterate if SQP fails."""
        try:
            res_sqp = self._sqp_solver.solve(x0, w0)
        except Exception:  # noqa: BLE001
            res_sqp = MPCSolveResult(
                u_opt=w0,
                cost=float("nan"),
                success=False,
                n_iter=0,
                capped=False,
                fallback=False,
            )

        if res_sqp.success:
            return res_sqp

        warm_w0 = res_sqp.u_opt if np.isfinite(res_sqp.u_opt).all() else w0
        res_ipopt = self._ipopt_solver.solve(x0, warm_w0)
        return dataclasses.replace(
            res_ipopt,
            n_iter=res_sqp.n_iter + res_ipopt.n_iter,
            fallback=True,
        )
