"""Controller components for the simulate framework.

:class:`ZeroController` is a no-op controller that always emits a zero control
vector -- the controller to use for *open-loop* runs through the
:class:`~simulate.simulation.Simulation` orchestrator (which always requires a
controller). :class:`StimWindowController` is an open-loop tES schedule: it holds
a fixed stimulation amplitude over a ``[onset, offset)`` time window and emits zero
otherwise, the orchestrated counterpart to the ``stim_window`` argument of
:func:`~neuro.jansen_rit.simulate_network`. :class:`WaveformController` plays back a
precomputed persistently-exciting tES schedule for plant identification.
:class:`MPCController` closes the loop: it embeds the CasADi NN predictor
(:class:`~neuro.nn_predictor_casadi.NNSymbolicModel`) as the dynamics model of a
receding-horizon NLP that minimizes predicted EEG power under per-electrode
amplitude bounds. :class:`LinearMPCController` is the same receding-horizon objective for a
*linear* (0-hidden-layer) predictor, posed as a convex QP and solved with OSQP (a
stacked/sparse formulation) or qpOASES (a condensed/dense one). :class:`NarxMPCController`
solves the same nonlinear suppression objective on a *minimal realization*: because the
predictor is a strict NARX map, its stacked window-state is a non-minimal shift register, so
this controller lifts only the per-step model-space output ``y_k`` (via
:func:`_build_output_lifted_nlp`) rather than the full state -- fewer decision variables, a
banded Jacobian, and mathematically equivalent to :class:`MPCController`.
:class:`LinearNarxMPCController` is that output-lifted formulation for a linear predictor,
posed as a convex QP and solved with OSQP.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Literal, Self

import casadi as ca
import numpy as np
from pydantic import Field
from simulate.controller import Controller

from neuro.config import StrictConfig
from neuro.nn_predictor_casadi import NNSymbolicModel

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from neuro.types import FloatArray


@dataclasses.dataclass(frozen=True)
class ZeroControllerLog:
    """Dataclass for ZeroController logging."""


class _ZeroControllerConfig(StrictConfig):
    """Config schema for :class:`ZeroController`."""

    dt: float = Field(gt=0)
    n_u: int = Field(default=1, ge=1)


class ZeroController(Controller[ZeroControllerLog]):
    """Controller that ignores its inputs and always outputs a zero ``(n_u,)`` vector.

    This is the controller for *open-loop* runs through the
    :class:`~simulate.simulation.Simulation` orchestrator, which always requires a
    controller; with all-zero control the plant's ``project_control`` is a no-op.
    """

    def __init__(self, dt: float, n_u: int = 1) -> None:
        """Initialize the zero controller for an ``n_u``-dimensional control."""
        super().__init__(dt)
        self.n_u = n_u

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary."""
        cfg = _ZeroControllerConfig.model_validate(config)
        return cls(dt=cfg.dt, n_u=cfg.n_u)

    def update(
        self,
        t: float,  # noqa: ARG002
        ref: FloatArray,  # noqa: ARG002
        x_hat: FloatArray,  # noqa: ARG002
    ) -> tuple[FloatArray, ZeroControllerLog]:
        """Return a zero control vector regardless of reference or state."""
        return np.zeros(self.n_u, dtype=np.float64), ZeroControllerLog()


@dataclasses.dataclass(frozen=True)
class StimWindowControllerLog:
    """Dataclass for StimWindowController logging."""

    active: bool


class _StimWindowControllerConfig(StrictConfig):
    """Config schema for :class:`StimWindowController`."""

    dt: float = Field(gt=0)
    onset: float = Field(ge=0)
    offset: float = Field(ge=0)
    amplitude: float | list[float]
    n_u: int = Field(default=1, ge=1)


class StimWindowController(Controller[StimWindowControllerLog]):
    """Open-loop tES schedule: a fixed amplitude held over a ``[onset, offset)`` window.

    The control is the per-electrode tES current the plant projects to nodes through
    ``connectome.gamma``; ``amplitude`` is a scalar shared by every electrode or a
    per-electrode vector of length ``n_u``. The window is half-open in seconds, so
    stimulation is active for ``onset <= t < offset`` and zero elsewhere. This is the
    orchestrated equivalent of the ``u_hat_tES`` / ``stim_window`` arguments of
    :func:`~neuro.jansen_rit.simulate_network`.
    """

    def __init__(
        self,
        dt: float,
        onset: float,
        offset: float,
        amplitude: ArrayLike,
        n_u: int = 1,
    ) -> None:
        """Initialize the windowed stimulation schedule.

        Parameters
        ----------
        dt
            Controller update step in seconds.
        onset, offset
            Half-open stimulation window ``[onset, offset)`` in seconds.
        amplitude
            tES current held during the window: a scalar shared by every electrode
            or a length-``n_u`` per-electrode vector.
        n_u
            Number of stimulation electrodes (control dimension).
        """
        super().__init__(dt)
        if offset < onset:
            msg = f"offset ({offset}) must be >= onset ({onset})"
            raise ValueError(msg)
        self.onset = onset
        self.offset = offset
        self.n_u = n_u

        amp = np.atleast_1d(np.asarray(amplitude, dtype=np.float64))
        if amp.size == 1:
            amp = np.broadcast_to(amp, (n_u,))
        elif amp.size != n_u:
            msg = f"amplitude has {amp.size} entries but n_u is {n_u}"
            raise ValueError(msg)
        self.amplitude = amp.reshape((n_u, 1))

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary."""
        cfg = _StimWindowControllerConfig.model_validate(config)
        return cls(dt=cfg.dt, onset=cfg.onset, offset=cfg.offset, amplitude=cfg.amplitude, n_u=cfg.n_u)

    def update(
        self,
        t: float,
        ref: FloatArray,  # noqa: ARG002
        x_hat: FloatArray,  # noqa: ARG002
    ) -> tuple[FloatArray, StimWindowControllerLog]:
        """Emit the stimulation amplitude inside the window, zero outside it."""
        active = self.onset <= t < self.offset
        u = self.amplitude.reshape(-1) if active else np.zeros(self.n_u, dtype=np.float64)
        return u, StimWindowControllerLog(active=active)


_MULTISINE_F_MIN_HZ = 1
_MULTISINE_F_MAX_HZ = 15
_EPS = 1e-12


def _multisine(n_samples: int, n_controls: int, amp: float, dt: float, rng: np.random.Generator) -> FloatArray:
    """Build a random-phase multisine of peak amplitude ``amp``, one column per electrode."""
    t = np.arange(n_samples) * dt
    freqs = np.arange(_MULTISINE_F_MIN_HZ, _MULTISINE_F_MAX_HZ + 1)
    out = np.zeros((n_samples, n_controls))
    for elec in range(n_controls):
        phases = rng.uniform(0.0, 2.0 * np.pi, size=freqs.size)
        sig = np.sin(2.0 * np.pi * freqs[:, None] * t[None, :] + phases[:, None]).sum(axis=0)
        out[:, elec] = amp * sig / max(np.abs(sig).max(), _EPS)
    return out


def build_input_schedule(  # noqa: PLR0913
    *,
    input_type: str,
    n_steps: int,
    transient_steps: int,
    n_controls: int,
    amp: float,
    hold_ms: float,
    dt: float,
    rng: np.random.Generator,
) -> FloatArray:
    """Build the per-step tES schedule ``(n_steps, n_controls)``; zero during the leading transient.

    ``ras`` holds a random uniform amplitude per block, ``prbs`` a random binary +/-amp, and
    ``multisine`` a random-phase sum of sinusoids; ``hold_ms`` sets the block length for the
    first two.
    """
    u = np.zeros((n_steps, n_controls))
    active = n_steps - transient_steps
    if active <= 0:
        return u

    if input_type in ("ras", "prbs"):
        hold = max(1, round(hold_ms / (dt * 1000.0)))
        n_blocks = (active + hold - 1) // hold
        if input_type == "ras":
            block_vals = rng.uniform(-amp, amp, size=(n_blocks, n_controls))
        else:
            block_vals = rng.choice(np.array([-amp, amp]), size=(n_blocks, n_controls))
        seq = np.repeat(block_vals, hold, axis=0)[:active]
    elif input_type == "multisine":
        seq = _multisine(active, n_controls, amp, dt, rng)
    else:
        msg = f"unknown input_type {input_type!r}"
        raise ValueError(msg)

    u[transient_steps:] = seq
    return u


@dataclasses.dataclass(frozen=True)
class WaveformControllerLog:
    """Dataclass for WaveformController logging (the emitted control is logged universally)."""


class _WaveformControllerConfig(StrictConfig):
    """Config schema for :class:`WaveformController`."""

    dt: float = Field(gt=0)
    input_type: Literal["ras", "prbs", "multisine"]
    duration: float = Field(gt=0)
    n_u: int = Field(ge=1)
    amp: float = Field(ge=0)
    input_seed: int = Field(ge=0)
    transient_ms: float = Field(default=0.0, ge=0)
    hold_ms: float = Field(default=50.0, gt=0)


class WaveformController(Controller[WaveformControllerLog]):
    """Open-loop controller that plays back a precomputed per-electrode tES waveform.

    Ignores the reference and estimated state; at time ``t`` it emits the schedule sample for
    step ``k = round(t / dt)`` (clamped to the last sample). Used to inject persistently-exciting
    tES inputs (random-amplitude steps ``ras``, a random binary signal ``prbs``, or a
    ``multisine``) for plant identification -- configured by
    ``configs/simulation/jansen_rit_seizure_excited.yaml``.
    """

    def __init__(self, dt: float, schedule: ArrayLike) -> None:
        """Initialize from a precomputed ``(n_steps, n_u)`` per-electrode schedule."""
        super().__init__(dt)
        self.schedule = np.atleast_2d(np.asarray(schedule, dtype=np.float64))
        self.n_u = self.schedule.shape[1]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Build the schedule from the excitation parameters in the config dict."""
        cfg = _WaveformControllerConfig.model_validate(config)
        schedule = build_input_schedule(
            input_type=cfg.input_type,
            n_steps=round(cfg.duration / cfg.dt),
            transient_steps=round(cfg.transient_ms / (cfg.dt * 1000.0)),
            n_controls=cfg.n_u,
            amp=cfg.amp,
            hold_ms=cfg.hold_ms,
            dt=cfg.dt,
            rng=np.random.default_rng(cfg.input_seed),
        )
        return cls(dt=cfg.dt, schedule=schedule)

    def update(
        self,
        t: float,
        ref: FloatArray,  # noqa: ARG002
        x_hat: FloatArray,  # noqa: ARG002
    ) -> tuple[FloatArray, WaveformControllerLog]:
        """Emit the scheduled per-electrode current for the current step."""
        k = round(t / self.dt)
        if k >= self.schedule.shape[0]:
            return np.zeros(self.n_u, dtype=np.float64), WaveformControllerLog()
        return self.schedule[k], WaveformControllerLog()


@dataclasses.dataclass(frozen=True)
class MPCControllerLog:
    """Per-step diagnostics for any MPC controller: the optimal cost, solver success, and a warm-up flag."""

    cost: float
    success: bool
    warmup: bool


class _MPCControllerConfig(StrictConfig):
    """Config schema for :class:`MPCController`."""

    dt: float = Field(gt=0)
    artifact: str
    u_max: float | list[float]
    horizon: int | None = Field(default=None, ge=1)
    w_y: float = Field(default=1.0, ge=0)
    w_u: float = Field(default=0.0, ge=0)
    shooting_depth: int = Field(default=1, ge=1)
    max_iter: int = Field(default=100, ge=1)
    max_cpu_time: float | None = Field(default=None, gt=0)
    expand: bool = False
    ipopt_options: dict[str, Any] | None = None


class MPCController(Controller[MPCControllerLog]):
    """Receding-horizon MPC that suppresses EEG power using the CasADi NN predictor.

    The :class:`~neuro.nn_predictor_casadi.NNSymbolicModel` is the prediction model: its
    "state" is the rolling window of the last ``n_y`` EEG measurements and ``n_u`` applied
    controls (raw units), and ``f_step``/``f_out`` give an exact one-step ``x_{k+1} =
    F(x_k, u_k)`` map and its EEG output. Each call builds the current window-state from
    the measurement/control history, solves a **multiple-shooting** NLP over ``horizon``
    steps -- minimizing ``sum_k ( w_y ||y_k||^2 + w_u ||u_k||^2 )`` subject to the
    continuity defects ``x_{k+1} = F(x_k, u_k)`` and per-electrode box bounds ``|u| <=
    u_max`` -- and applies the first control (receding horizon).

    When the artifact carries a fixed PCA projection, each measurement is encoded to the
    ``k`` latent components before entering the window, so the shooting state (and hence the
    NLP) shrinks from ``n_y * n_eeg + ...`` to ``n_y * k + ...``; ``f_out`` decodes back to
    raw EEG so the ``||y_k||^2`` cost is still the (retained-subspace) EEG power.

    The controller ``dt`` should equal the predictor's native step
    (:attr:`MLPArtifact.dt`); the orchestrator's zero-order hold then applies each control
    across the corresponding number of plant steps. The ``reference`` input is ignored --
    the objective is suppression, not tracking.
    """

    def __init__(  # noqa: PLR0913
        self,
        dt: float,
        model: NNSymbolicModel,
        u_max: ArrayLike,
        horizon: int | None = None,
        w_y: float = 1.0,
        w_u: float = 0.0,
        *,
        shooting_depth: int = 1,
        max_iter: int = 100,
        max_cpu_time: float | None = None,
        expand: bool = False,
        ipopt_options: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the MPC and build its (re-used) IPOPT solver.

        Parameters
        ----------
        dt
            Controller update step in seconds; should equal the predictor's native dt.
        model
            The CasADi NN predictor used as the prediction model.
        u_max
            Per-electrode amplitude bound: a scalar shared by every electrode or a
            length-``n_controls`` vector. The box constraint is ``-u_max <= u <= u_max``.
        horizon
            Prediction/control horizon in steps; defaults to the model's native horizon.
        w_y
            Weight on predicted EEG power in the cost.
        w_u
            Weight on control effort (quadratic) in the cost.
        max_iter
            Hard cap on IPOPT iterations per solve. When the cap is hit the best warm-started iterate is applied
            and ``success`` is ``False`` (capped, not failed).
        max_cpu_time
            Optional per-solve wall-time budget in seconds (IPOPT ``max_cpu_time``); ``None``
            leaves it unbounded.
        expand
            Expand the NLP from MX to SX before building the solver. Off by default: (for this
            MLP-heavy graphthe SX expansion is huge).
        """
        super().__init__(dt)
        self.model = model
        self.horizon = int(horizon) if horizon is not None else model.artifact.horizon
        self.w_y = float(w_y)
        self.w_u = float(w_u)
        self.shooting_depth = int(shooting_depth)
        self.max_iter = int(max_iter)
        self.max_cpu_time = float(max_cpu_time) if max_cpu_time is not None else None
        self.expand = bool(expand)
        self.ipopt_options = dict(ipopt_options) if ipopt_options is not None else {}

        self.n_y = model.artifact.n_y
        self.n_u = model.artifact.n_u
        self.n_channels = model.n_channels
        self.n_controls = model.n_controls

        u_max_arr = np.atleast_1d(np.asarray(u_max, dtype=np.float64))
        if u_max_arr.size == 1:
            u_max_arr = np.broadcast_to(u_max_arr, (self.n_controls,))
        elif u_max_arr.size != self.n_controls:
            msg = f"u_max has {u_max_arr.size} entries but n_controls is {self.n_controls}"
            raise ValueError(msg)
        self.u_max = np.ascontiguousarray(u_max_arr)

        # History windows (oldest first, newest last)
        self._y_buf = np.zeros((self.n_y, self.n_channels), dtype=np.float64)
        self._u_buf = np.zeros((self.n_u, self.n_controls), dtype=np.float64)
        self._n_seen = 0
        self._u_prev: FloatArray | None = None

        self._build_solver()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate from a config dict, loading the NN predictor artifact from disk."""
        from neuro.nn_predictor_casadi import NNSymbolicModel  # noqa: PLC0415

        cfg = _MPCControllerConfig.model_validate(config)
        return cls(
            dt=cfg.dt,
            model=NNSymbolicModel.from_artifact(cfg.artifact),
            u_max=cfg.u_max,
            horizon=cfg.horizon,
            w_y=cfg.w_y,
            w_u=cfg.w_u,
            shooting_depth=cfg.shooting_depth,
            max_iter=cfg.max_iter,
            max_cpu_time=cfg.max_cpu_time,
            expand=cfg.expand,
            ipopt_options=cfg.ipopt_options,
        )

    def _build_solver(self) -> None:
        """Build the PCMS multiple-shooting NLP and its IPOPT solver, once.

        The problem is partitioned into segments of size D (shooting_depth). The decision
        variables are the control sequence ``[u_0, ..., u_{H-1}]`` and the sparse sequence of
        intermediate state shooting roots ``[phi_1, ..., phi_{floor((H-1)/D)}]``.
        The parameter ``x0`` is the current window-state (phi_0). The state maps exactly
        via the CasADi f_step block and constraints close the gaps between the segments.
        """
        n_state = self.model.state_shape[0]
        n_ctrl, h = self.n_controls, self.horizon
        D = self.shooting_depth

        x0_p = ca.MX.sym("x0", n_state)
        u_vars = [ca.MX.sym(f"u_{k}", n_ctrl) for k in range(h)]

        n_segments = (h - 1) // D
        phi_vars = [ca.MX.sym(f"phi_{k}", n_state) for k in range(1, n_segments + 1)]

        def get_phi(idx: int) -> ca.MX:
            return x0_p if idx == 0 else phi_vars[idx - 1]

        defects, cost = [], ca.MX(0)
        for k in range(n_segments + 1):
            x_curr = get_phi(k)
            start_step = k * D
            end_step = min((k + 1) * D, h)

            for step in range(start_step, end_step):
                u_curr = u_vars[step]
                x_next = self.model.f_step(x_curr, u_curr)
                y_next = self.model.f_out(x_next)

                cost = cost + self.w_y * ca.sumsqr(y_next) + self.w_u * ca.sumsqr(u_curr)
                x_curr = x_next

            if k < n_segments:
                defects.append(x_curr - get_phi(k + 1))

        x_nlp = ca.vertcat(*u_vars, *phi_vars) if phi_vars else ca.vertcat(*u_vars)

        g_nlp = ca.vertcat(*defects) if defects else ca.MX(0)
        nlp = {"x": x_nlp, "f": cost, "g": g_nlp, "p": x0_p}
        opts = {
            "print_time": False,
            "expand": self.expand,
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.max_iter": self.max_iter,
            "ipopt.hessian_approximation": "limited-memory",
        }
        if self.max_cpu_time is not None:
            opts["ipopt.max_cpu_time"] = self.max_cpu_time

        for k, v in self.ipopt_options.items():
            opts[f"ipopt.{k}" if not k.startswith("ipopt.") else k] = v

        self._solver = ca.nlpsol("mpc", "ipopt", nlp, opts)

        n_phi_vars = len(phi_vars) * n_state
        self._lbx = np.concatenate([np.tile(-self.u_max, h), np.full(n_phi_vars, -np.inf)])
        self._ubx = np.concatenate([np.tile(self.u_max, h), np.full(n_phi_vars, np.inf)])

    def _solve(self, x0: FloatArray) -> tuple[FloatArray, float, bool]:
        """Solve the NLP for window-state ``x0``; return ``(u_0*, cost, success)``."""
        m, h = self.n_controls, self.horizon
        D = self.shooting_depth
        u_guess = self._u_prev if self._u_prev is not None else np.zeros((h, m))

        # Forward-simulate from x0 to seed the lifted states with a (near-)feasible guess.
        x = x0
        phi_guess = []
        for step in range(h):
            x = np.asarray(self.model.f_step(x, u_guess[step])).reshape(-1)
            if (step + 1) % D == 0 and (step + 1) < h:
                phi_guess.append(x)

        w0 = np.concatenate([u_guess.reshape(-1), *phi_guess]) if phi_guess else u_guess.reshape(-1)

        sol = self._solver(x0=w0, lbx=self._lbx, ubx=self._ubx, lbg=0.0, ubg=0.0, p=x0)
        u_opt = np.asarray(sol["x"]).reshape(-1)[: h * m].reshape(h, m)
        self._u_prev = np.vstack([u_opt[1:], u_opt[-1:]])

        status = self._solver.stats()["return_status"]
        success = status in {"Solve_Succeeded", "Solved_To_Acceptable_Level"}
        return u_opt[0], float(sol["f"]), success

    def update(
        self,
        t: float,  # noqa: ARG002
        ref: FloatArray,  # noqa: ARG002
        x_hat: FloatArray,
    ) -> tuple[FloatArray, MPCControllerLog]:
        """Ingest the current EEG measurement, solve the NLP, and emit the first control."""
        y_eeg = x_hat.reshape(-1)
        # Encode to the model's state space: latent components with a projection, else a no-op.
        y = self.model.artifact.encode(y_eeg)
        self._y_buf = np.vstack([self._y_buf[1:], y])
        self._n_seen += 1

        # While the window is still zero-padded, hold off stimulating.
        if self._n_seen < self.n_y:
            self._u_buf = np.vstack([self._u_buf[1:], np.zeros((1, self.n_controls))])
            return np.zeros(self.n_controls, dtype=np.float64), MPCControllerLog(cost=0.0, success=True, warmup=True)

        x0 = np.concatenate([self._y_buf.reshape(-1), self._u_buf.reshape(-1)])
        u0, cost, success = self._solve(x0)
        self._u_buf = np.vstack([self._u_buf[1:], u0.reshape(1, self.n_controls)])
        return u0, MPCControllerLog(cost=cost, success=success, warmup=False)


class _LinearMPCControllerConfig(StrictConfig):
    """Config schema for :class:`LinearMPCController`."""

    dt: float = Field(gt=0)
    artifact: str
    u_max: float | list[float]
    horizon: int | None = Field(default=None, ge=1)
    w_y: float = Field(default=1.0, ge=0)
    w_u: float = Field(default=0.0, ge=0)
    formulation: Literal["sparse", "dense"] = "sparse"
    osqp_eps: float = Field(default=1e-9, gt=0)


class LinearMPCController(Controller[MPCControllerLog]):
    """Receding-horizon MPC for a *linear* (0-hidden-layer) NN predictor, solved as a QP.

    A depth-0 :class:`~neuro.nn_predictor_casadi.NNSymbolicModel` is an exact affine map
    ``x_{k+1} = F(x_k, u_k)`` with an affine EEG output, so suppressing predicted EEG power
    under box bounds is a convex quadratic program -- no need for the nonlinear NLP that
    :class:`MPCController` builds. As there, the "state" is the rolling window of the last
    ``n_y`` EEG measurements and ``n_u`` applied controls; each call builds the window-state,
    solves the QP over ``horizon`` steps -- minimizing ``sum_k ( w_y ||y_k||^2 + w_u ||u_k||^2
    )`` subject to per-electrode box bounds ``|u| <= u_max`` -- and applies the first control.

    Two equivalent QP formulations are selectable via ``formulation``:

    * ``"sparse"`` -- the *stacked* multiple-shooting QP whose decision variables are the
      controls ``[u_0, ..., u_{H-1}]`` and every intermediate state ``[x_1, ..., x_H]``, tied
      by the affine continuity constraints ``x_{k+1} = F(x_k, u_k)``. The constraint Jacobian
      is sparse/banded, solved with **OSQP** (the ``ca.qpsol`` ``"osqp"`` plugin).
    * ``"dense"`` -- the *condensed* QP whose only decision variables are the controls; the
      states are eliminated by unrolling ``F`` from the current window-state, giving a dense
      Hessian solved with the active-set **qpOASES** (the ``ca.qpsol`` ``"qpoases"`` plugin).

    Both solve the same problem (to solver tolerance); ``"sparse"`` scales better to long
    horizons, ``"dense"`` to small ``horizon * n_controls``. Projection artifacts are handled
    exactly as in :class:`MPCController` (the window holds latent components; ``f_out``/the
    output map decodes back to EEG). The controller ``dt`` should equal the predictor's native
    step and the reference is ignored -- the objective is suppression, not tracking.
    """

    def __init__(  # noqa: PLR0913
        self,
        dt: float,
        model: NNSymbolicModel,
        u_max: ArrayLike,
        horizon: int | None = None,
        w_y: float = 1.0,
        w_u: float = 0.0,
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
            The CasADi NN predictor used as the prediction model. Must be linear: its inner
            ``eqx.nn.MLP`` must have ``depth == 0`` (a single affine layer), otherwise the QP
            would silently use the quadratic Taylor expansion of a nonlinear map.
        u_max
            Per-electrode amplitude bound: a scalar shared by every electrode or a
            length-``n_controls`` vector. The box constraint is ``-u_max <= u <= u_max``.
        horizon
            Prediction/control horizon in steps; defaults to the model's native horizon.
        w_y
            Weight on predicted EEG power in the cost.
        w_u
            Weight on control effort (quadratic) in the cost.
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
        depth = model.artifact.model.model.depth
        if depth != 0:
            msg = (
                "LinearMPCController requires a linear predictor (0 hidden layers); got an MLP "
                f"with depth {depth}. Use MPCController for a nonlinear predictor."
            )
            raise ValueError(msg)

        self.model = model
        self.horizon = int(horizon) if horizon is not None else model.artifact.horizon
        self.w_y = float(w_y)
        self.w_u = float(w_u)
        self.formulation = formulation
        self.osqp_eps = float(osqp_eps)

        self.n_y = model.artifact.n_y
        self.n_u = model.artifact.n_u
        self.n_channels = model.n_channels
        self.n_controls = model.n_controls

        u_max_arr = np.atleast_1d(np.asarray(u_max, dtype=np.float64))
        if u_max_arr.size == 1:
            u_max_arr = np.broadcast_to(u_max_arr, (self.n_controls,))
        elif u_max_arr.size != self.n_controls:
            msg = f"u_max has {u_max_arr.size} entries but n_controls is {self.n_controls}"
            raise ValueError(msg)
        self.u_max = np.ascontiguousarray(u_max_arr)

        # History windows (oldest first, newest last)
        self._y_buf = np.zeros((self.n_y, self.n_channels), dtype=np.float64)
        self._u_buf = np.zeros((self.n_u, self.n_controls), dtype=np.float64)
        self._n_seen = 0
        self._u_prev: FloatArray | None = None

        self._build_solver()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate from a config dict, loading the NN predictor artifact from disk."""
        cfg = _LinearMPCControllerConfig.model_validate(config)
        return cls(
            dt=cfg.dt,
            model=NNSymbolicModel.from_artifact(cfg.artifact),
            u_max=cfg.u_max,
            horizon=cfg.horizon,
            w_y=cfg.w_y,
            w_u=cfg.w_u,
            formulation=cfg.formulation,
            osqp_eps=cfg.osqp_eps,
        )

    def _build_solver(self) -> None:
        """Build the QP and its OSQP (``"sparse"``) or qpOASES (``"dense"``) solver, once.

        ``"sparse"`` stacks the controls and every intermediate state as decision variables and
        ties them with the affine continuity constraints; ``"dense"`` keeps only the controls
        and unrolls the affine state map from the ``x0`` parameter (no equality constraints).
        Both accumulate the same quadratic cost. The inline symbolic ``model.step``/
        ``model.output`` (rather than the compiled ``f_step``/``f_out``) keep the graph a flat
        affine expression, so ``ca.qpsol`` extracts an exact constant Hessian.
        """
        n_state = self.model.state_shape[0]
        n_ctrl, h = self.n_controls, self.horizon

        x0_p = ca.MX.sym("x0", n_state)
        u_vars = [ca.MX.sym(f"u_{k}", n_ctrl) for k in range(h)]

        cost = ca.MX(0)
        if self.formulation == "dense":
            x_curr = x0_p
            for k in range(h):
                x_curr = self.model.step([x_curr], u_vars[k])
                y_next = self.model.output(x_curr)
                cost = cost + self.w_y * ca.sumsqr(y_next) + self.w_u * ca.sumsqr(u_vars[k])
            qp = {"x": ca.vertcat(*u_vars), "f": cost, "p": x0_p}
            self._lbx = np.tile(-self.u_max, h)
            self._ubx = np.tile(self.u_max, h)
            self._has_g = False
        else:  # sparse
            x_vars = [ca.MX.sym(f"x_{k}", n_state) for k in range(1, h + 1)]
            defects, x_prev = [], x0_p
            for k in range(h):
                x_lift = x_vars[k]
                defects.append(x_lift - self.model.step([x_prev], u_vars[k]))
                y_next = self.model.output(x_lift)
                cost = cost + self.w_y * ca.sumsqr(y_next) + self.w_u * ca.sumsqr(u_vars[k])
                x_prev = x_lift
            qp = {"x": ca.vertcat(*u_vars, *x_vars), "f": cost, "g": ca.vertcat(*defects), "p": x0_p}
            n_x_vars = h * n_state
            self._lbx = np.concatenate([np.tile(-self.u_max, h), np.full(n_x_vars, -np.inf)])
            self._ubx = np.concatenate([np.tile(self.u_max, h), np.full(n_x_vars, np.inf)])
            self._has_g = True

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
            w0 = u_guess.reshape(-1)
        else:
            # Forward-simulate from x0 to seed the lifted states with a feasible guess.
            x = x0
            x_guess = []
            for step in range(h):
                x = np.asarray(self.model.f_step(x, u_guess[step])).reshape(-1)
                x_guess.append(x)
            w0 = np.concatenate([u_guess.reshape(-1), *x_guess])

        call: dict[str, Any] = {"x0": w0, "lbx": self._lbx, "ubx": self._ubx, "p": x0}
        if self._has_g:
            call["lbg"], call["ubg"] = 0.0, 0.0
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
        y_eeg = x_hat.reshape(-1)
        # Encode to the model's state space: latent components with a projection, else a no-op.
        y = self.model.artifact.encode(y_eeg)
        self._y_buf = np.vstack([self._y_buf[1:], y])
        self._n_seen += 1

        # While the window is still zero-padded, hold off stimulating.
        if self._n_seen < self.n_y:
            self._u_buf = np.vstack([self._u_buf[1:], np.zeros((1, self.n_controls))])
            return np.zeros(self.n_controls, dtype=np.float64), MPCControllerLog(cost=0.0, success=True, warmup=True)

        x0 = np.concatenate([self._y_buf.reshape(-1), self._u_buf.reshape(-1)])
        u0, cost, success = self._solve(x0)
        self._u_buf = np.vstack([self._u_buf[1:], u0.reshape(1, self.n_controls)])
        return u0, MPCControllerLog(cost=cost, success=success, warmup=False)


def _build_output_lifted_nlp(
    model: NNSymbolicModel, horizon: int, w_y: float, w_u: float, u_max: FloatArray
) -> dict[str, Any]:
    """Assemble the output-lifted (minimal-realization NARX) suppression problem.

    The predictor is a strict NARX map, so its stacked window-state is a non-minimal
    shift-register realization whose copy-rows carry no information. This lifts only the
    genuinely new quantity at each node -- the model-space output ``y_k`` -- alongside the
    controls ``u_k``, and closes one defect per node
    ``y_k - phi(y_{k-n_y..k-1}, u_{k-n_u+1..k}) = 0`` via
    :meth:`~neuro.nn_predictor_casadi.NNSymbolicModel.predict_output`. Any window index reaching
    before the horizon reads from the measured-past parameter ``x0`` (the current window-state),
    never a variable, so the shift registers are never instantiated: the lifted-state part
    shrinks from ``H*(n_y*n_ch + n_u*n_ctrl)`` to ``H*n_ch`` and the defect Jacobian stays
    banded. The ``||y_k||^2`` cost is measured in raw EEG units by decoding each lifted output
    with the (reused, unchanged) :meth:`~neuro.nn_predictor_casadi.NNSymbolicModel.output`, which
    reads only the last y-slot of the state it is given.

    Parameters
    ----------
    model
        The CasADi NN predictor supplying ``predict_output``/``output`` and the dimensions.
    horizon
        Prediction/control horizon in steps (``H``).
    w_y, w_u
        Weights on predicted EEG power and (quadratic) control effort.
    u_max
        Per-electrode amplitude bound, already broadcast to length ``n_controls``.

    Returns
    -------
    dict
        The symbolic problem pieces ``x``/``f``/``g``/``p`` (decision vector, cost, defects,
        measured-past parameter) plus the box bounds ``lbx``/``ubx``. The nonlinear (IPOPT) and
        linear (OSQP) controllers each wrap these with their own solver.
    """
    n_y, n_u = model.artifact.n_y, model.artifact.n_u
    n_ch, n_ctrl = model.n_channels, model.n_controls
    h = horizon

    x0_p = ca.MX.sym("x0", model.state_shape[0])
    u_vars = [ca.MX.sym(f"u_{k}", n_ctrl) for k in range(h)]
    y_vars = [ca.MX.sym(f"y_{k}", n_ch) for k in range(h)]
    y_base = n_y * n_ch

    def y_at(idx: int) -> ca.MX:
        """Model-space output at window index ``idx`` (measured past when ``idx < 0``)."""
        if idx >= 0:
            return y_vars[idx]
        pos = n_y + idx  # idx in [-n_y, -1] -> measured slot [0, n_y-1]
        return x0_p[pos * n_ch : (pos + 1) * n_ch]

    def u_at(idx: int) -> ca.MX:
        """Control at window index ``idx`` (measured past when ``idx < 0``)."""
        if idx >= 0:
            return u_vars[idx]
        pos = n_u + idx  # idx in [-n_u+1, -1] -> measured slot [1, n_u-1]
        return x0_p[y_base + pos * n_ctrl : y_base + (pos + 1) * n_ctrl]

    def decode(z: ca.MX) -> ca.SX | ca.MX:
        """Decode a lifted model-space output to raw EEG via the model's ``output`` map."""
        pad_before = ca.MX.zeros((n_y - 1) * n_ch, 1)
        pad_after = ca.MX.zeros(n_u * n_ctrl, 1)
        return model.output(ca.vertcat(pad_before, z, pad_after))

    defects, cost = [], ca.MX(0)
    for k in range(h):
        y_win = ca.vertcat(*[y_at(k - n_y + j) for j in range(n_y)])  # y_{k-n_y..k-1}
        u_win = ca.vertcat(*[u_at(k - n_u + 1 + j) for j in range(n_u)])  # u_{k-n_u+1..k}
        defects.append(y_vars[k] - model.predict_output(y_win, u_win))
        cost = cost + w_y * ca.sumsqr(decode(y_vars[k])) + w_u * ca.sumsqr(u_vars[k])

    lbx = np.concatenate([np.tile(-u_max, h), np.full(h * n_ch, -np.inf)])
    ubx = np.concatenate([np.tile(u_max, h), np.full(h * n_ch, np.inf)])
    return {
        "x": ca.vertcat(*u_vars, *y_vars),
        "f": cost,
        "g": ca.vertcat(*defects),
        "p": x0_p,
        "lbx": lbx,
        "ubx": ubx,
    }


def _seed_output_lifted(model: NNSymbolicModel, horizon: int, u_prev: FloatArray | None, x0: FloatArray) -> FloatArray:
    """Warm-start vector for the output-lifted problem: ``[u_guess, y_guess]``.

    The controls are the previous solve shifted by one (or zeros on the first call); the lifted
    outputs are seeded by rolling the full window-state ``x0`` forward through the compiled
    ``f_step`` and reading each next state's newest model-space sample, giving a feasible guess
    for the defects.
    """
    m = model.n_controls
    n_y, n_ch = model.artifact.n_y, model.n_channels
    u_guess = u_prev if u_prev is not None else np.zeros((horizon, m))
    x = x0
    y_guess = []
    for step in range(horizon):
        x = np.asarray(model.f_step(x, u_guess[step])).reshape(-1)
        y_guess.append(x[(n_y - 1) * n_ch : n_y * n_ch])
    return np.concatenate([u_guess.reshape(-1), *y_guess])


class _NarxMPCControllerConfig(StrictConfig):
    """Config schema for :class:`NarxMPCController`."""

    dt: float = Field(gt=0)
    artifact: str
    u_max: float | list[float]
    horizon: int | None = Field(default=None, ge=1)
    w_y: float = Field(default=1.0, ge=0)
    w_u: float = Field(default=0.0, ge=0)
    max_iter: int = Field(default=100, ge=1)
    max_cpu_time: float | None = Field(default=None, gt=0)
    expand: bool = False
    ipopt_options: dict[str, Any] | None = None


class NarxMPCController(Controller[MPCControllerLog]):
    """Receding-horizon MPC that suppresses EEG power using the output-lifted NARX formulation.

    Same objective and predictor as :class:`MPCController` -- minimize
    ``sum_k ( w_y ||y_k||^2 + w_u ||u_k||^2 )`` under per-electrode box bounds ``|u| <= u_max``
    -- but posed on a *minimal* realization. Because the
    :class:`~neuro.nn_predictor_casadi.NNSymbolicModel` is a strict NARX map, its stacked
    window-state is a non-minimal shift register; this controller lifts only the per-step
    model-space output ``y_k`` (and the controls ``u_k``), closing one defect per node
    ``y_k = phi(y_{k-n_y..k-1}, u_{k-n_u+1..k})`` whose earlier-than-horizon indices read from
    the measured window-state (see :func:`_build_output_lifted_nlp`). The lifted-state part is
    thus ``H*n_channels`` rather than :class:`MPCController`'s ``H*(n_y*n_channels +
    n_u*n_controls)`` shooting roots, with a banded defect Jacobian handed to IPOPT. The two are
    mathematically equivalent; this one trades the full-state multiple-shooting structure for far
    fewer decision variables.

    PCA-projection artifacts are handled exactly as in :class:`MPCController` (the lifted outputs
    are latent components; the reused ``output`` map decodes back to EEG for the ``||y_k||^2``
    cost). The controller ``dt`` should equal the predictor's native step and the reference is
    ignored -- the objective is suppression, not tracking.
    """

    def __init__(  # noqa: PLR0913
        self,
        dt: float,
        model: NNSymbolicModel,
        u_max: ArrayLike,
        horizon: int | None = None,
        w_y: float = 1.0,
        w_u: float = 0.0,
        *,
        max_iter: int = 100,
        max_cpu_time: float | None = None,
        expand: bool = False,
        ipopt_options: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the output-lifted MPC and build its (re-used) IPOPT solver.

        Parameters mirror :class:`MPCController` (minus ``shooting_depth``, which output lifting
        makes moot -- every node is lifted): ``dt`` the update step (predictor's native dt),
        ``model`` the CasADi NN predictor, ``u_max`` the per-electrode amplitude bound (scalar or
        length-``n_controls``), ``horizon`` the control horizon (default the model's native
        horizon), ``w_y``/``w_u`` the EEG-power/effort weights, ``max_iter`` the per-solve IPOPT
        iteration cap (a capped iterate is still applied, with ``success`` False), ``max_cpu_time``
        an optional per-solve wall-time budget in seconds, ``expand`` whether to expand the NLP
        from MX to SX, and ``ipopt_options`` an extra IPOPT-option pass-through.
        """
        super().__init__(dt)
        self.model = model
        self.horizon = int(horizon) if horizon is not None else model.artifact.horizon
        self.w_y = float(w_y)
        self.w_u = float(w_u)
        self.max_iter = int(max_iter)
        self.max_cpu_time = float(max_cpu_time) if max_cpu_time is not None else None
        self.expand = bool(expand)
        self.ipopt_options = dict(ipopt_options) if ipopt_options is not None else {}

        self.n_y = model.artifact.n_y
        self.n_u = model.artifact.n_u
        self.n_channels = model.n_channels
        self.n_controls = model.n_controls

        u_max_arr = np.atleast_1d(np.asarray(u_max, dtype=np.float64))
        if u_max_arr.size == 1:
            u_max_arr = np.broadcast_to(u_max_arr, (self.n_controls,))
        elif u_max_arr.size != self.n_controls:
            msg = f"u_max has {u_max_arr.size} entries but n_controls is {self.n_controls}"
            raise ValueError(msg)
        self.u_max = np.ascontiguousarray(u_max_arr)

        # History windows (oldest first, newest last)
        self._y_buf = np.zeros((self.n_y, self.n_channels), dtype=np.float64)
        self._u_buf = np.zeros((self.n_u, self.n_controls), dtype=np.float64)
        self._n_seen = 0
        self._u_prev: FloatArray | None = None

        self._build_solver()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate from a config dict, loading the NN predictor artifact from disk."""
        cfg = _NarxMPCControllerConfig.model_validate(config)
        return cls(
            dt=cfg.dt,
            model=NNSymbolicModel.from_artifact(cfg.artifact),
            u_max=cfg.u_max,
            horizon=cfg.horizon,
            w_y=cfg.w_y,
            w_u=cfg.w_u,
            max_iter=cfg.max_iter,
            max_cpu_time=cfg.max_cpu_time,
            expand=cfg.expand,
            ipopt_options=cfg.ipopt_options,
        )

    def _build_solver(self) -> None:
        """Build the output-lifted NLP and its IPOPT solver, once."""
        prob = _build_output_lifted_nlp(self.model, self.horizon, self.w_y, self.w_u, self.u_max)
        nlp = {"x": prob["x"], "f": prob["f"], "g": prob["g"], "p": prob["p"]}
        opts = {
            "print_time": False,
            "expand": self.expand,
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.max_iter": self.max_iter,
            "ipopt.hessian_approximation": "limited-memory",
        }
        if self.max_cpu_time is not None:
            opts["ipopt.max_cpu_time"] = self.max_cpu_time
        for k, v in self.ipopt_options.items():
            opts[f"ipopt.{k}" if not k.startswith("ipopt.") else k] = v

        self._solver = ca.nlpsol("mpc", "ipopt", nlp, opts)
        self._lbx, self._ubx = prob["lbx"], prob["ubx"]

    def _solve(self, x0: FloatArray) -> tuple[FloatArray, float, bool]:
        """Solve the NLP for window-state ``x0``; return ``(u_0*, cost, success)``."""
        m, h = self.n_controls, self.horizon
        w0 = _seed_output_lifted(self.model, h, self._u_prev, x0)
        sol = self._solver(x0=w0, lbx=self._lbx, ubx=self._ubx, lbg=0.0, ubg=0.0, p=x0)
        u_opt = np.asarray(sol["x"]).reshape(-1)[: h * m].reshape(h, m)
        self._u_prev = np.vstack([u_opt[1:], u_opt[-1:]])
        status = self._solver.stats()["return_status"]
        success = status in {"Solve_Succeeded", "Solved_To_Acceptable_Level"}
        return u_opt[0], float(sol["f"]), success

    def update(
        self,
        t: float,  # noqa: ARG002
        ref: FloatArray,  # noqa: ARG002
        x_hat: FloatArray,
    ) -> tuple[FloatArray, MPCControllerLog]:
        """Ingest the current EEG measurement, solve the NLP, and emit the first control."""
        y = self.model.artifact.encode(x_hat.reshape(-1))
        self._y_buf = np.vstack([self._y_buf[1:], y])
        self._n_seen += 1

        # While the window is still zero-padded, hold off stimulating.
        if self._n_seen < self.n_y:
            self._u_buf = np.vstack([self._u_buf[1:], np.zeros((1, self.n_controls))])
            return np.zeros(self.n_controls, dtype=np.float64), MPCControllerLog(cost=0.0, success=True, warmup=True)

        x0 = np.concatenate([self._y_buf.reshape(-1), self._u_buf.reshape(-1)])
        u0, cost, success = self._solve(x0)
        self._u_buf = np.vstack([self._u_buf[1:], u0.reshape(1, self.n_controls)])
        return u0, MPCControllerLog(cost=cost, success=success, warmup=False)


class _LinearNarxMPCControllerConfig(StrictConfig):
    """Config schema for :class:`LinearNarxMPCController`."""

    dt: float = Field(gt=0)
    artifact: str
    u_max: float | list[float]
    horizon: int | None = Field(default=None, ge=1)
    w_y: float = Field(default=1.0, ge=0)
    w_u: float = Field(default=0.0, ge=0)
    osqp_eps: float = Field(default=1e-9, gt=0)


class LinearNarxMPCController(Controller[MPCControllerLog]):
    """Output-lifted NARX MPC for a *linear* (0-hidden-layer) predictor, solved as a convex QP.

    The output-lifted counterpart of :class:`LinearMPCController`: the same receding-horizon
    suppression objective and box bounds, but lifting only the per-step model-space output
    ``y_k`` (and the controls ``u_k``) instead of the full window-state. For a depth-0 predictor
    the defects ``y_k = phi(...)`` are affine and the cost quadratic, so the problem is a convex
    QP solved with OSQP; the banded, output-lifted constraint set has ``H*n_channels`` lifted
    variables rather than :class:`LinearMPCController`'s ``H*n_state`` stacked states. As there,
    the predictor must be linear (``depth == 0``); projection artifacts, warm-up, and the ignored
    reference behave exactly as in :class:`NarxMPCController`.
    """

    def __init__(  # noqa: PLR0913
        self,
        dt: float,
        model: NNSymbolicModel,
        u_max: ArrayLike,
        horizon: int | None = None,
        w_y: float = 1.0,
        w_u: float = 0.0,
        *,
        osqp_eps: float = 1e-9,
    ) -> None:
        """Initialize the linear output-lifted MPC and build its (re-used) OSQP solver.

        Parameters mirror the nonlinear sibling :class:`NarxMPCController` (``dt``, ``model``,
        ``u_max``, ``horizon``, ``w_y``, ``w_u``) plus ``osqp_eps``, the OSQP absolute/relative
        convergence tolerance (default ``1e-9``, far tighter than OSQP's loose ``1e-3`` default so
        the suppression QP is solved accurately). The predictor's inner MLP must have ``depth ==
        0`` (a single affine layer); a nonlinear model raises ``ValueError``.
        """
        super().__init__(dt)
        depth = model.artifact.model.model.depth
        if depth != 0:
            msg = (
                "LinearNarxMPCController requires a linear predictor (0 hidden layers); got an MLP "
                f"with depth {depth}. Use NarxMPCController for a nonlinear predictor."
            )
            raise ValueError(msg)

        self.model = model
        self.horizon = int(horizon) if horizon is not None else model.artifact.horizon
        self.w_y = float(w_y)
        self.w_u = float(w_u)
        self.osqp_eps = float(osqp_eps)

        self.n_y = model.artifact.n_y
        self.n_u = model.artifact.n_u
        self.n_channels = model.n_channels
        self.n_controls = model.n_controls

        u_max_arr = np.atleast_1d(np.asarray(u_max, dtype=np.float64))
        if u_max_arr.size == 1:
            u_max_arr = np.broadcast_to(u_max_arr, (self.n_controls,))
        elif u_max_arr.size != self.n_controls:
            msg = f"u_max has {u_max_arr.size} entries but n_controls is {self.n_controls}"
            raise ValueError(msg)
        self.u_max = np.ascontiguousarray(u_max_arr)

        # History windows (oldest first, newest last)
        self._y_buf = np.zeros((self.n_y, self.n_channels), dtype=np.float64)
        self._u_buf = np.zeros((self.n_u, self.n_controls), dtype=np.float64)
        self._n_seen = 0
        self._u_prev: FloatArray | None = None

        self._build_solver()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate from a config dict, loading the NN predictor artifact from disk."""
        cfg = _LinearNarxMPCControllerConfig.model_validate(config)
        return cls(
            dt=cfg.dt,
            model=NNSymbolicModel.from_artifact(cfg.artifact),
            u_max=cfg.u_max,
            horizon=cfg.horizon,
            w_y=cfg.w_y,
            w_u=cfg.w_u,
            osqp_eps=cfg.osqp_eps,
        )

    def _build_solver(self) -> None:
        """Build the output-lifted QP and its OSQP solver, once.

        Uses the inline symbolic ``predict_output``/``output`` (via :func:`_build_output_lifted_nlp`)
        rather than the compiled ``f_step``/``f_out`` so the graph stays a flat affine expression
        and ``ca.qpsol`` extracts an exact constant Hessian.
        """
        prob = _build_output_lifted_nlp(self.model, self.horizon, self.w_y, self.w_u, self.u_max)
        qp = {"x": prob["x"], "f": prob["f"], "g": prob["g"], "p": prob["p"]}
        opts = {
            "print_time": False,
            "error_on_fail": False,
            "osqp": {"verbose": False, "eps_abs": self.osqp_eps, "eps_rel": self.osqp_eps},
        }
        self._solver = ca.qpsol("mpc", "osqp", qp, opts)
        self._lbx, self._ubx = prob["lbx"], prob["ubx"]

    def _solve(self, x0: FloatArray) -> tuple[FloatArray, float, bool]:
        """Solve the QP for window-state ``x0``; return ``(u_0*, cost, success)``."""
        m, h = self.n_controls, self.horizon
        w0 = _seed_output_lifted(self.model, h, self._u_prev, x0)
        sol = self._solver(x0=w0, lbx=self._lbx, ubx=self._ubx, lbg=0.0, ubg=0.0, p=x0)
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
        y = self.model.artifact.encode(x_hat.reshape(-1))
        self._y_buf = np.vstack([self._y_buf[1:], y])
        self._n_seen += 1

        # While the window is still zero-padded, hold off stimulating.
        if self._n_seen < self.n_y:
            self._u_buf = np.vstack([self._u_buf[1:], np.zeros((1, self.n_controls))])
            return np.zeros(self.n_controls, dtype=np.float64), MPCControllerLog(cost=0.0, success=True, warmup=True)

        x0 = np.concatenate([self._y_buf.reshape(-1), self._u_buf.reshape(-1)])
        u0, cost, success = self._solve(x0)
        self._u_buf = np.vstack([self._u_buf[1:], u0.reshape(1, self.n_controls)])
        return u0, MPCControllerLog(cost=cost, success=success, warmup=False)
