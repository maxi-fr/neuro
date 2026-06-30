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
amplitude bounds.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Self

import casadi as ca
import numpy as np
from simulate.controller import Controller

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from neuro.nn_predictor_casadi import NNSymbolicModel


@dataclasses.dataclass(frozen=True)
class ZeroControllerLog:
    """Dataclass for ZeroController logging."""


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
        return cls(dt=float(config["dt"]), n_u=int(config.get("n_u", 1)))

    def update(
        self,
        t: float,  # noqa: ARG002
        ref: float | np.ndarray,  # noqa: ARG002
        x_hat: float | np.ndarray,  # noqa: ARG002
    ) -> tuple[float | np.ndarray, ZeroControllerLog]:
        """Return a zero control vector regardless of reference or state."""
        return np.zeros(self.n_u, dtype=np.float64), ZeroControllerLog()


@dataclasses.dataclass(frozen=True)
class StimWindowControllerLog:
    """Dataclass for StimWindowController logging."""

    active: bool


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
        return cls(
            dt=float(config["dt"]),
            onset=float(config["onset"]),
            offset=float(config["offset"]),
            amplitude=config["amplitude"],
            n_u=int(config.get("n_u", 1)),
        )

    def update(
        self,
        t: float,
        ref: float | np.ndarray,  # noqa: ARG002
        x_hat: float | np.ndarray,  # noqa: ARG002
    ) -> tuple[float | np.ndarray, StimWindowControllerLog]:
        """Emit the stimulation amplitude inside the window, zero outside it."""
        active = self.onset <= t < self.offset
        u = self.amplitude.reshape(-1) if active else np.zeros(self.n_u, dtype=np.float64)
        return u, StimWindowControllerLog(active=active)


_MULTISINE_F_MIN_HZ = 1
_MULTISINE_F_MAX_HZ = 15
_EPS = 1e-12


def _multisine(n_samples: int, n_elec: int, amp: float, dt: float, rng: np.random.Generator) -> np.ndarray:
    """Build a random-phase multisine of peak amplitude ``amp``, one column per electrode."""
    t = np.arange(n_samples) * dt
    freqs = np.arange(_MULTISINE_F_MIN_HZ, _MULTISINE_F_MAX_HZ + 1)
    out = np.zeros((n_samples, n_elec))
    for elec in range(n_elec):
        phases = rng.uniform(0.0, 2.0 * np.pi, size=freqs.size)
        sig = np.sin(2.0 * np.pi * freqs[:, None] * t[None, :] + phases[:, None]).sum(axis=0)
        out[:, elec] = amp * sig / max(np.abs(sig).max(), _EPS)
    return out


def build_input_schedule(  # noqa: PLR0913
    *,
    input_type: str,
    n_steps: int,
    transient_steps: int,
    n_elec: int,
    amp: float,
    hold_ms: float,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build the per-step tES schedule ``(n_steps, n_elec)``; zero during the leading transient.

    ``ras`` holds a random uniform amplitude per block, ``prbs`` a random binary +/-amp, and
    ``multisine`` a random-phase sum of sinusoids; ``hold_ms`` sets the block length for the
    first two.
    """
    u = np.zeros((n_steps, n_elec))
    active = n_steps - transient_steps
    if active <= 0:
        return u

    if input_type in ("ras", "prbs"):
        hold = max(1, round(hold_ms / (dt * 1000.0)))
        n_blocks = (active + hold - 1) // hold
        if input_type == "ras":
            block_vals = rng.uniform(-amp, amp, size=(n_blocks, n_elec))
        else:
            block_vals = rng.choice(np.array([-amp, amp]), size=(n_blocks, n_elec))
        seq = np.repeat(block_vals, hold, axis=0)[:active]
    elif input_type == "multisine":
        seq = _multisine(active, n_elec, amp, dt, rng)
    else:
        msg = f"unknown input_type {input_type!r}"
        raise ValueError(msg)

    u[transient_steps:] = seq
    return u


@dataclasses.dataclass(frozen=True)
class WaveformControllerLog:
    """Dataclass for WaveformController logging (the emitted control is logged universally)."""


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
        dt = float(config["dt"])
        schedule = build_input_schedule(
            input_type=str(config["input_type"]),
            n_steps=round(float(config["duration"]) / dt),
            transient_steps=round(float(config.get("transient_ms", 0.0)) / (dt * 1000.0)),
            n_elec=int(config["n_u"]),
            amp=float(config["amp"]),
            hold_ms=float(config.get("hold_ms", 50.0)),
            dt=dt,
            rng=np.random.default_rng(int(config["input_seed"])),
        )
        return cls(dt=dt, schedule=schedule)

    def update(
        self,
        t: float,
        ref: float | np.ndarray,  # noqa: ARG002
        x_hat: float | np.ndarray,  # noqa: ARG002
    ) -> tuple[float | np.ndarray, WaveformControllerLog]:
        """Emit the scheduled per-electrode current for the current step."""
        k = round(t / self.dt)
        if k >= self.schedule.shape[0]:
            return np.zeros(self.n_u, dtype=np.float64), WaveformControllerLog()
        return self.schedule[k], WaveformControllerLog()


@dataclasses.dataclass(frozen=True)
class MPCControllerLog:
    """Per-step MPC diagnostics: the optimal cost, solver success, and a warm-up flag."""

    cost: float
    success: bool
    warmup: bool


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
        max_iter: int = 100,
        max_cpu_time: float | None = None,
        expand: bool = False,
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
            length-``n_elec`` vector. The box constraint is ``-u_max <= u <= u_max``.
        horizon
            Prediction/control horizon in steps; defaults to the model's native horizon.
        w_y
            Weight on predicted EEG power in the cost.
        w_u
            Weight on control effort (quadratic) in the cost.
        max_iter
            Hard cap on IPOPT iterations per solve. MPC only needs a good first move, not a
            certified optimum; when the cap is hit the best warm-started iterate is applied
            and ``success`` is ``False`` (capped, not failed).
        max_cpu_time
            Optional per-solve wall-time budget in seconds (IPOPT ``max_cpu_time``); ``None``
            leaves it unbounded.
        expand
            Expand the NLP from MX to SX before building the solver. Off by default: for this
            MLP-heavy graph (a deep net unrolled over the horizon) the SX expansion is huge --
            it inflates build time by ~10x and is slower per IPOPT iteration than MX, whose
            compact matrix ops stay BLAS-backed. Benchmarks bear this out; leave it off.
        """
        super().__init__(dt)
        self.model = model
        self.horizon = int(horizon) if horizon is not None else model.artifact.horizon
        self.w_y = float(w_y)
        self.w_u = float(w_u)
        self.max_iter = int(max_iter)
        self.max_cpu_time = float(max_cpu_time) if max_cpu_time is not None else None
        self.expand = bool(expand)

        self.n_y = model.artifact.n_y
        self.n_u_steps = model.artifact.n_u
        self.n_channels = model.n_channels
        self.n_elec = model.n_elec

        u_max_arr = np.atleast_1d(np.asarray(u_max, dtype=np.float64))
        if u_max_arr.size == 1:
            u_max_arr = np.broadcast_to(u_max_arr, (self.n_elec,))
        elif u_max_arr.size != self.n_elec:
            msg = f"u_max has {u_max_arr.size} entries but n_elec is {self.n_elec}"
            raise ValueError(msg)
        self.u_max = np.ascontiguousarray(u_max_arr)

        # History windows (oldest first, newest last), zero-padded until filled.
        self._y_buf = np.zeros((self.n_y, self.n_channels), dtype=np.float64)
        self._u_buf = np.zeros((self.n_u_steps, self.n_elec), dtype=np.float64)
        self._n_seen = 0
        self._u_prev: np.ndarray | None = None  # last optimal control sequence, shifted

        self._build_solver()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate from a config dict, loading the NN predictor artifact from disk."""
        from neuro.nn_predictor_casadi import NNSymbolicModel  # noqa: PLC0415

        max_cpu_time = config.get("max_cpu_time")
        return cls(
            dt=float(config["dt"]),
            model=NNSymbolicModel.from_artifact(config["artifact"]),
            u_max=config["u_max"],
            horizon=int(config["horizon"]) if config.get("horizon") is not None else None,
            w_y=float(config.get("w_y", 1.0)),
            w_u=float(config.get("w_u", 0.0)),
            max_iter=int(config.get("max_iter", 100)),
            max_cpu_time=float(max_cpu_time) if max_cpu_time is not None else None,
            expand=bool(config.get("expand", False)),
        )

    def _build_solver(self) -> None:
        """Build the output-condensed multiple-shooting NLP and its IPOPT solver, once.

        Lifting the full shift-register state per step would make ~94% of the variables and
        defects trivial identity copies (the y-window is just past predictions; the u-window
        is a deterministic function of the decision controls and the known history). So the
        decision vector lifts **only the predicted EEG output** per step: the controls
        ``[u_0, ..., u_{H-1}]`` (box-bounded by ``u_max``) followed by the outputs ``[y_1, ...,
        y_H]`` (free). The parameter ``x0`` is the current window-state; its y-/u-windows are
        sliced out and rolled forward symbolically, appending each decision ``u_k``/``y_k`` and
        dropping the oldest block. The defects ``y_k - predict_output(window_k) = 0`` reproduce
        the full-state continuity exactly. IPOPT uses a limited-memory Hessian approximation
        because the predictor's ReLU activations make the exact Hessian non-smooth/degenerate.
        """
        n_state = self.model.state_shape[0]
        n_ch, n_ctrl, h = self.n_channels, self.n_elec, self.horizon

        x0_p = ca.MX.sym("x0", n_state)
        u_vars = [ca.MX.sym(f"u_{k}", n_ctrl) for k in range(h)]
        y_vars = [ca.MX.sym(f"y_{k}", n_ch) for k in range(1, h + 1)]

        # Initial rolling windows (oldest first), sliced from the parameter window-state.
        y_win = [x0_p[i * n_ch : (i + 1) * n_ch] for i in range(self.n_y)]
        split = self.n_y * n_ch
        u_win = [x0_p[split + i * n_ctrl : split + (i + 1) * n_ctrl] for i in range(self.n_u_steps)]

        defects, cost = [], ca.MX(0)
        for k in range(h):
            u_win = [*u_win[1:], u_vars[k]]  # drop oldest control, append the new one
            y_pred = self.model.predict_output(ca.vertcat(*y_win), ca.vertcat(*u_win))
            defects.append(y_vars[k] - y_pred)
            cost = cost + self.w_y * ca.sumsqr(y_vars[k]) + self.w_u * ca.sumsqr(u_vars[k])
            y_win = [*y_win[1:], y_vars[k]]  # drop oldest output, append the lifted prediction

        nlp = {"x": ca.vertcat(*u_vars, *y_vars), "f": cost, "g": ca.vertcat(*defects), "p": x0_p}
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
        self._solver = ca.nlpsol("mpc", "ipopt", nlp, opts)

        self._lbx = np.concatenate([np.tile(-self.u_max, h), np.full(h * n_ch, -np.inf)])
        self._ubx = np.concatenate([np.tile(self.u_max, h), np.full(h * n_ch, np.inf)])

    def _solve(self, x0: np.ndarray) -> tuple[np.ndarray, float, bool]:
        """Solve the NLP for window-state ``x0``; return ``(u_0*, cost, success)``."""
        m, h = self.n_elec, self.horizon
        u_guess = self._u_prev if self._u_prev is not None else np.zeros((h, m))

        # Forward-simulate from x0 to seed the lifted outputs with a (near-)feasible guess.
        x = x0
        y_rows = []
        for k in range(h):
            x = np.asarray(self.model.f_step(x, u_guess[k])).reshape(-1)
            y_rows.append(np.asarray(self.model.f_out(x)).reshape(-1))
        w0 = np.concatenate([u_guess.reshape(-1), *y_rows])

        sol = self._solver(x0=w0, lbx=self._lbx, ubx=self._ubx, lbg=0.0, ubg=0.0, p=x0)
        u_opt = np.asarray(sol["x"]).reshape(-1)[: h * m].reshape(h, m)
        self._u_prev = np.vstack([u_opt[1:], u_opt[-1:]])  # shift one step for next warm start
        # The first control is applied regardless; the ReLU kinks can keep IPOPT from
        # certifying full KKT convergence, so an "acceptable" point also counts as success.
        status = self._solver.stats()["return_status"]
        success = status in {"Solve_Succeeded", "Solved_To_Acceptable_Level"}
        return u_opt[0], float(sol["f"]), success

    def update(
        self,
        t: float,  # noqa: ARG002
        ref: float | np.ndarray,  # noqa: ARG002
        x_hat: float | np.ndarray,
    ) -> tuple[float | np.ndarray, MPCControllerLog]:
        """Ingest the current EEG measurement, solve the NLP, and emit the first control."""
        y = np.atleast_1d(np.asarray(x_hat, dtype=np.float64)).reshape(-1)
        self._y_buf = np.vstack([self._y_buf[1:], y])
        self._n_seen += 1

        # While the window is still zero-padded, hold off stimulating.
        if self._n_seen < self.n_y:
            self._u_buf = np.vstack([self._u_buf[1:], np.zeros((1, self.n_elec))])
            return np.zeros(self.n_elec, dtype=np.float64), MPCControllerLog(cost=0.0, success=True, warmup=True)

        x0 = np.concatenate([self._y_buf.reshape(-1), self._u_buf.reshape(-1)])
        u0, cost, success = self._solve(x0)
        self._u_buf = np.vstack([self._u_buf[1:], u0.reshape(1, self.n_elec)])
        return u0, MPCControllerLog(cost=cost, success=success, warmup=False)
