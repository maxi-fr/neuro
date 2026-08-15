from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Literal, Self

import casadi as ca
import numpy as np
from pydantic import Field, PositiveFloat
from simulate.controller import Controller

from neuro.config import StrictConfig
from neuro.nn_predictor_casadi import NNSymbolicModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike

    from neuro.types import FloatArray, SymbolicModel


class _ZeroControllerConfig(StrictConfig):
    """Config schema for :class:`ZeroController`."""

    dt: float = Field(gt=0)
    n_u: int = Field(default=1, ge=1)


@dataclasses.dataclass(frozen=True)
class ZeroControllerLog:
    """Log carrying the zero control vector."""

    u: FloatArray


class ZeroController(Controller[ZeroControllerLog]):
    """Controller that ignores its inputs and always outputs a zero ``(n_u,)`` vector."""

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
        u = np.zeros(self.n_u, dtype=np.float64)
        return u, ZeroControllerLog(u=u)


@dataclasses.dataclass(frozen=True)
class AmplitudeThresholdControllerLog:
    """Dataclass for AmplitudeThresholdController logging."""

    u: FloatArray
    amplitude: float
    triggered: bool
    active: bool


class _AmplitudeThresholdControllerConfig(StrictConfig):
    """Config schema for :class:`AmplitudeThresholdController`."""

    dt: float = Field(gt=0)
    amplitude: float | list[float]
    threshold: float = Field(gt=0)
    burst_duration: float = Field(gt=0)
    window: float = Field(default=1.0, gt=0)
    channel: int | None = Field(default=None, ge=0)
    n_u: int = Field(default=1, ge=1)


class AmplitudeThresholdController(Controller[AmplitudeThresholdControllerLog]):
    """Trigger a fixed-length constant stimulus whenever EEG amplitude crosses a threshold."""

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        dt: float,
        amplitude: ArrayLike,
        threshold: float,
        burst_duration: float,
        window: float = 1.0,
        channel: int | None = None,
        n_u: int = 1,
    ) -> None:
        """Initialize the trigger.

        Parameters
        ----------
        dt
            Controller update step in seconds.
        amplitude
            tES current held during a burst: a scalar shared by every electrode or a
            length-``n_u`` per-electrode vector. Should sum to zero (Kirchhoff).
        threshold
            Peak-to-peak amplitude that starts a burst, in the measurement's units.
        burst_duration
            Length of one stimulation burst in seconds (the paper's tau).
        window
            Length of the trailing window the peak-to-peak is measured over, in seconds.
        channel
            Index of the monitored measurement channel; ``None`` (default) monitors every
            channel and triggers on the largest peak-to-peak among them.
        n_u
            Number of stimulation electrodes (control dimension).
        """
        super().__init__(dt)
        self.threshold = float(threshold)
        self.burst_duration = float(burst_duration)
        self.window = float(window)
        self.channel = channel
        self.n_u = n_u

        amp = np.atleast_1d(np.asarray(amplitude, dtype=np.float64))
        if amp.size == 1:
            amp = np.broadcast_to(amp, (n_u,))
        elif amp.size != n_u:
            msg = f"amplitude has {amp.size} entries but n_u is {n_u}"
            raise ValueError(msg)
        self.amplitude = np.ascontiguousarray(amp)

        self._n_window = max(1, round(self.window / dt))
        self._buf: FloatArray | None = None
        self._n_seen = 0
        self._until = -np.inf

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary."""
        cfg = _AmplitudeThresholdControllerConfig.model_validate(config)
        return cls(
            dt=cfg.dt,
            amplitude=cfg.amplitude,
            threshold=cfg.threshold,
            burst_duration=cfg.burst_duration,
            window=cfg.window,
            channel=cfg.channel,
            n_u=cfg.n_u,
        )

    def update(
        self,
        t: float,
        ref: FloatArray,  # noqa: ARG002
        x_hat: FloatArray,
    ) -> tuple[FloatArray, AmplitudeThresholdControllerLog]:
        """Ingest the EEG measurement and emit the burst amplitude while a burst is running."""
        y = np.atleast_1d(np.asarray(x_hat, dtype=np.float64).reshape(-1))
        if self._buf is None:
            self._buf = np.zeros((self._n_window, y.size), dtype=np.float64)
        self._buf = np.vstack([self._buf[1:], y])
        self._n_seen += 1

        filled = self._n_seen >= self._n_window
        ptp = np.ptp(self._buf, axis=0)
        amplitude = float(ptp[self.channel] if self.channel is not None else ptp.max()) if filled else 0.0

        triggered = t >= self._until and amplitude > self.threshold
        if triggered:
            self._until = t + self.burst_duration
        active = t < self._until
        u = self.amplitude if active else np.zeros(self.n_u, dtype=np.float64)
        return u, AmplitudeThresholdControllerLog(u=u.copy(), amplitude=amplitude, triggered=triggered, active=active)


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
    input_type: Literal["ras", "prbs", "multisine"],
    n_steps: int,
    transient_steps: int,
    n_controls: int,
    amp: float,
    hold_ms: float | Sequence[float],
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
        holds = np.atleast_1d(np.asarray(hold_ms, dtype=np.float64))
        hold_steps = np.maximum(1, np.round(holds / (dt * 1000.0)).astype(int))
        n_blocks = int(np.ceil(active / hold_steps.min()))
        if hold_steps.size > 1:
            # Draw p ~ 1 / length so each entry gets an equal share of *time*: drawn uniformly, a
            # value's share is proportional to its own length and the longest hold eats the run.
            p = (1.0 / hold_steps) / np.sum(1.0 / hold_steps)
            lengths = rng.choice(hold_steps, size=n_blocks, p=p)
        else:
            lengths = np.repeat(hold_steps, n_blocks)
        n_blocks = int(np.searchsorted(np.cumsum(lengths), active) + 1)
        lengths = lengths[:n_blocks]
        if input_type == "ras":
            block_vals = rng.uniform(-amp, amp, size=(n_blocks, n_controls))
        else:
            block_vals = rng.choice(np.array([-amp, amp]), size=(n_blocks, n_controls))
        seq = np.repeat(block_vals, lengths, axis=0)[:active]
    elif input_type == "multisine":
        seq = _multisine(active, n_controls, amp, dt, rng)
    else:
        msg = f"unknown input_type {input_type!r}"
        raise ValueError(msg)

    zero_sum = seq - seq.mean(axis=1, keepdims=True)
    peak = np.abs(zero_sum).max(axis=1, keepdims=True)
    zero_sum *= np.minimum(1.0, amp / np.where(peak > 0.0, peak, 1.0))
    u[transient_steps:] = zero_sum
    return u


class _WaveformControllerConfig(StrictConfig):
    """Config schema for :class:`WaveformController`."""

    dt: float = Field(gt=0)
    input_type: Literal["ras", "prbs", "multisine"]
    duration: float = Field(gt=0)
    n_u: int = Field(ge=1)
    amp: float = Field(ge=0)
    input_seed: int = Field(ge=0)
    transient_ms: float = Field(default=0.0, ge=0)
    hold_ms: PositiveFloat | list[PositiveFloat] = 50.0


@dataclasses.dataclass(frozen=True)
class WaveformControllerLog:
    """Log carrying the applied control; excitation datasets are identified against it."""

    u: FloatArray


class WaveformController(Controller[WaveformControllerLog]):
    """Open-loop controller that plays back a precomputed per-electrode tES waveform."""

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
        u = np.zeros(self.n_u, dtype=np.float64) if k >= self.schedule.shape[0] else self.schedule[k]
        return u, WaveformControllerLog(u=u.copy())


@dataclasses.dataclass(frozen=True)
class MPCControllerLog:
    """Per-step diagnostics for any MPC controller: the applied control, optimal cost, solver success, iteration count, iteration-cap flag, and a warm-up flag."""

    u: FloatArray
    cost: float
    success: bool
    warmup: bool
    n_iter: int = 0
    capped: bool = False


def _l1_epigraph(u_vars: list[ca.MX], w_l1: float) -> tuple[list[ca.MX], ca.MX, ca.MX]:
    """Epigraph pieces encoding ``w_l1 * sum_k ||u_k||_1`` without breaking quadraticity.

    A slack ``t_k`` is added per control node so the non-smooth L1 penalty becomes a *linear*
    cost ``w_l1 * sum(t_k)`` plus the linear inequalities ``t_k >= |u_k|`` (i.e. ``t_k - u_k >= 0``
    and ``t_k + u_k >= 0``). Keeping the objective quadratic lets ``ca.qpsol`` extract a constant
    Hessian (OSQP/qpOASES) and keeps the IPOPT graph smooth; the active-set QP can then drive
    controls to exact zero. Returns the slack variables, the (linear) cost term, and the stacked
    ``>= 0`` inequalities ``[t_k - u_k, t_k + u_k]``.
    """
    slacks = [ca.MX.sym(f"t_{k}", u.numel()) for k, u in enumerate(u_vars)]
    cost = w_l1 * ca.sum1(ca.vertcat(*slacks))
    g = []
    for t, u in zip(slacks, u_vars, strict=True):
        g += [t - u, t + u]
    return slacks, cost, ca.vertcat(*g)


def _sum_to_zero(u_vars: list[ca.MX]) -> ca.MX:
    """Kirchhoff current-law equality: the per-electrode currents sum to zero at each step."""
    return ca.vertcat(*[ca.sum1(u) for u in u_vars])


def build_mpc_nlp(  # noqa: PLR0913
    model: SymbolicModel,
    *,
    horizon: int,
    shooting_depth: int,
    n_controls: int,
    u_max: FloatArray,
    w_y: float,
    w_y_terminal: float | None,
    w_u: float,
    w_u_l1: float,
    max_iter: int,
    max_cpu_time: float | None,
    expand: bool,
    ipopt_options: dict[str, Any],
) -> tuple[ca.Function, FloatArray, FloatArray, FloatArray | float, FloatArray | float]:
    """Build the PCMS shooting NLP and its IPOPT solver; returns (solver, lbx, ubx, lbg, ubg)."""
    n_state = model.state_shape[0]
    n_ctrl, h = n_controls, horizon
    D = shooting_depth

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
            x_next = model.f_step(x_curr, u_curr)
            y_next = model.f_out(x_next)

            is_terminal = step == h - 1 and w_y_terminal is not None
            w_y_step = w_y_terminal if is_terminal else w_y
            cost = cost + w_y_step * ca.sumsqr(y_next) + w_u * ca.sumsqr(u_curr)
            x_curr = x_next

        if k < n_segments:
            defects.append(x_curr - get_phi(k + 1))

    x_parts = [*u_vars, *phi_vars]
    g_parts = [*defects, _sum_to_zero(u_vars)]
    n_eq = len(defects) * n_state + h
    n_phi_vars = len(phi_vars) * n_state
    lbx = np.concatenate([np.tile(-u_max, h), np.full(n_phi_vars, -np.inf)])
    ubx = np.concatenate([np.tile(u_max, h), np.full(n_phi_vars, np.inf)])

    if w_u_l1 > 0:
        slacks, l1_cost, l1_g = _l1_epigraph(u_vars, w_u_l1)
        cost = cost + l1_cost
        x_parts += slacks
        g_parts.append(l1_g)
        n_l1 = l1_g.numel()
        lbg = np.concatenate([np.zeros(n_eq), np.zeros(n_l1)])
        ubg = np.concatenate([np.zeros(n_eq), np.full(n_l1, np.inf)])
        lbx = np.concatenate([lbx, np.zeros(h * n_ctrl)])
        ubx = np.concatenate([ubx, np.full(h * n_ctrl, np.inf)])
    else:
        lbg = ubg = 0.0

    x_nlp = ca.vertcat(*x_parts)
    g_nlp = ca.vertcat(*g_parts) if g_parts else ca.MX(0)
    nlp = {"x": x_nlp, "f": cost, "g": g_nlp, "p": x0_p}
    opts = {
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

    solver = ca.nlpsol("mpc", "ipopt", nlp, opts)
    return solver, lbx, ubx, lbg, ubg


class _MPCControllerConfig(StrictConfig):
    """Config schema for :class:`MPCController`."""

    dt: float = Field(gt=0)
    artifact: str
    u_max: float | list[float]
    horizon: int | None = Field(default=None, ge=1)
    w_y: float = Field(default=1.0, ge=0)
    w_y_terminal: float | None = Field(default=None, ge=0)
    w_u: float = Field(default=0.0, ge=0)
    w_u_l1: float = Field(default=0.0, ge=0)
    shooting_depth: int | None = Field(default=None, ge=1)
    max_iter: int = Field(default=100, ge=1)
    max_cpu_time: float | None = Field(default=None, gt=0)
    expand: bool = False
    ipopt_options: dict[str, Any] | None = None


class MPCController(Controller[MPCControllerLog]):
    """Receding-horizon MPC that suppresses EEG power using the CasADi NN predictor."""

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        dt: float,
        model: NNSymbolicModel,
        u_max: ArrayLike,
        horizon: int | None = None,
        w_y: float = 1.0,
        w_u: float = 0.0,
        w_u_l1: float = 0.0,
        *,
        w_y_terminal: float | None = None,
        shooting_depth: int | None = None,
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
        w_y_terminal
            Weight on predicted EEG power at the *final* horizon step, replacing ``w_y`` there;
            ``None`` (default) keeps ``w_y`` uniform over the horizon.
        w_u_l1
            Weight on the L1 norm of the control effort (a sparse-stimulation penalty); ``0``
            disables it (default), leaving the pure-quadratic problem unchanged.
        shooting_depth
            Size ``D`` of the shooting segments: a state shooting root is introduced every ``D``
            steps. ``D = 1`` is full multiple shooting (a root at every step), ``D >= horizon``
            is single shooting (no roots; the states are condensed out and the NLP is over the
            controls alone). Defaults to ``horizon``, i.e. single shooting
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
        self.w_y_terminal = float(w_y_terminal) if w_y_terminal is not None else None
        self.w_u = float(w_u)
        self.w_u_l1 = float(w_u_l1)
        self.shooting_depth = int(shooting_depth) if shooting_depth is not None else self.horizon
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
            w_u_l1=cfg.w_u_l1,
            w_y_terminal=cfg.w_y_terminal,
            shooting_depth=cfg.shooting_depth,
            max_iter=cfg.max_iter,
            max_cpu_time=cfg.max_cpu_time,
            expand=cfg.expand,
            ipopt_options=cfg.ipopt_options,
        )

    def _build_solver(self) -> None:
        """Build the PCMS multiple-shooting NLP and its IPOPT solver, once."""
        self._solver, self._lbx, self._ubx, self._lbg, self._ubg = build_mpc_nlp(
            self.model,
            horizon=self.horizon,
            shooting_depth=self.shooting_depth,
            n_controls=self.n_controls,
            u_max=self.u_max,
            w_y=self.w_y,
            w_y_terminal=self.w_y_terminal,
            w_u=self.w_u,
            w_u_l1=self.w_u_l1,
            max_iter=self.max_iter,
            max_cpu_time=self.max_cpu_time,
            expand=self.expand,
            ipopt_options=self.ipopt_options,
        )

    def _solve(self, x0: FloatArray) -> tuple[FloatArray, float, bool, int, bool]:
        """Solve the NLP for window-state ``x0``; return ``(u_0*, cost, success, n_iter, capped)``."""
        m, h = self.n_controls, self.horizon
        D = self.shooting_depth
        u_guess = self._u_prev if self._u_prev is not None else np.zeros((h, m))

        x = x0
        phi_guess = []
        for step in range(h):
            x = np.asarray(self.model.f_step(x, u_guess[step])).reshape(-1)
            if (step + 1) % D == 0 and (step + 1) < h:
                phi_guess.append(x)

        seed = [u_guess.reshape(-1), *phi_guess]
        if self.w_u_l1 > 0:
            seed.append(np.abs(u_guess).reshape(-1))
        w0 = np.concatenate(seed)

        sol = self._solver(x0=w0, lbx=self._lbx, ubx=self._ubx, lbg=self._lbg, ubg=self._ubg, p=x0)
        u_opt = np.asarray(sol["x"]).reshape(-1)[: h * m].reshape(h, m)
        self._u_prev = np.vstack([u_opt[1:], u_opt[-1:]])

        stats = self._solver.stats()
        status = stats["return_status"]
        success = status in {"Solve_Succeeded", "Solved_To_Acceptable_Level"}
        return (
            u_opt[0],
            float(sol["f"]),
            success,
            int(stats.get("iter_count", 0)),
            status == "Maximum_Iterations_Exceeded",
        )

    def update(
        self,
        t: float,  # noqa: ARG002
        ref: FloatArray,  # noqa: ARG002
        x_hat: FloatArray,
    ) -> tuple[FloatArray, MPCControllerLog]:
        """Ingest the current EEG measurement, solve the NLP, and emit the first control."""
        y_eeg = x_hat.reshape(-1)
        y = self.model.artifact.encode(y_eeg)
        self._y_buf = np.vstack([self._y_buf[1:], y])
        self._n_seen += 1

        if self._n_seen < self.n_y:
            u_zero = np.zeros(self.n_controls, dtype=np.float64)
            self._u_buf = np.vstack([self._u_buf[1:], u_zero.reshape(1, self.n_controls)])
            return u_zero, MPCControllerLog(u=u_zero, cost=0.0, success=True, warmup=True)

        x0 = np.concatenate([self._y_buf.reshape(-1), self._u_buf.reshape(-1)])
        u0, cost, success, n_iter, capped = self._solve(x0)
        self._u_buf = np.vstack([self._u_buf[1:], u0.reshape(1, self.n_controls)])
        return u0, MPCControllerLog(u=u0.copy(), cost=cost, success=success, warmup=False, n_iter=n_iter, capped=capped)


class _LinearMPCControllerConfig(StrictConfig):
    """Config schema for :class:`LinearMPCController`."""

    dt: float = Field(gt=0)
    artifact: str
    u_max: float | list[float]
    horizon: int | None = Field(default=None, ge=1)
    w_y: float = Field(default=1.0, ge=0)
    w_u: float = Field(default=0.0, ge=0)
    w_u_l1: float = Field(default=0.0, ge=0)
    formulation: Literal["sparse", "dense"] = "sparse"
    osqp_eps: float = Field(default=1e-9, gt=0)


class LinearMPCController(Controller[MPCControllerLog]):
    """Receding-horizon MPC for a *linear* (0-hidden-layer) NN predictor, solved as a QP."""

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        dt: float,
        model: NNSymbolicModel,
        u_max: ArrayLike,
        horizon: int | None = None,
        w_y: float = 1.0,
        w_u: float = 0.0,
        w_u_l1: float = 0.0,
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
        w_u_l1
            Weight on the L1 norm of the control effort (a sparse-stimulation penalty); ``0``
            disables it (default). When positive the QP gains slack variables and the
            inequalities ``t >= |u|`` so it stays a convex QP.
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
        self.w_u_l1 = float(w_u_l1)
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
            w_u_l1=cfg.w_u_l1,
            formulation=cfg.formulation,
            osqp_eps=cfg.osqp_eps,
        )

    def _build_solver(self) -> None:  # noqa: PLR0915
        """Build the QP and its OSQP (``"sparse"``) or qpOASES (``"dense"``) solver, once.

        ``"sparse"`` stacks the controls and every intermediate state as decision variables and
        ties them with the affine continuity constraints; ``"dense"`` keeps only the controls
        and unrolls the affine state map from the ``x0`` parameter. Both accumulate the same
        quadratic cost and both enforce the Kirchhoff sum-to-zero equality on the controls (so
        ``"dense"`` also carries equality constraints). The inline symbolic ``model.step``/
        ``model.output`` (rather than the compiled ``f_step``/``f_out``) keep the graph a flat
        affine expression, so ``ca.qpsol`` extracts an exact constant Hessian.
        """
        n_state = self.model.state_shape[0]
        n_ctrl, h = self.n_controls, self.horizon

        x0_p = ca.MX.sym("x0", n_state)
        u_vars = [ca.MX.sym(f"u_{k}", n_ctrl) for k in range(h)]

        cost = ca.MX(0)
        # Kirchhoff current law: the montage currents sum to zero at every step (h equalities).
        kcl = _sum_to_zero(u_vars)
        if self.formulation == "dense":
            x_curr = x0_p
            for k in range(h):
                x_curr = self.model.step([x_curr], u_vars[k])
                y_next = self.model.output(x_curr)
                cost = cost + self.w_y * ca.sumsqr(y_next) + self.w_u * ca.sumsqr(u_vars[k])
            x_parts = [*u_vars]
            g_parts: list[ca.MX] = [kcl]
            lbx = np.tile(-self.u_max, h)
            ubx = np.tile(self.u_max, h)
            lbg = np.zeros(h)
            ubg = np.zeros(h)
        else:  # sparse
            x_vars = [ca.MX.sym(f"x_{k}", n_state) for k in range(1, h + 1)]
            defects, x_prev = [], x0_p
            for k in range(h):
                x_lift = x_vars[k]
                defects.append(x_lift - self.model.step([x_prev], u_vars[k]))
                y_next = self.model.output(x_lift)
                cost = cost + self.w_y * ca.sumsqr(y_next) + self.w_u * ca.sumsqr(u_vars[k])
                x_prev = x_lift
            x_parts = [*u_vars, *x_vars]
            g_parts = [*defects, kcl]
            n_x_vars = h * n_state
            lbx = np.concatenate([np.tile(-self.u_max, h), np.full(n_x_vars, -np.inf)])
            ubx = np.concatenate([np.tile(self.u_max, h), np.full(n_x_vars, np.inf)])
            lbg = np.zeros(h * n_state + h)  # continuity defects + sum-to-zero (equalities)
            ubg = np.zeros(h * n_state + h)

        if self.w_u_l1 > 0:
            slacks, l1_cost, l1_g = _l1_epigraph(u_vars, self.w_u_l1)
            cost = cost + l1_cost
            x_parts += slacks
            g_parts.append(l1_g)
            n_l1 = l1_g.numel()
            lbg = np.concatenate([lbg, np.zeros(n_l1)])
            ubg = np.concatenate([ubg, np.full(n_l1, np.inf)])
            lbx = np.concatenate([lbx, np.zeros(h * n_ctrl)])
            ubx = np.concatenate([ubx, np.full(h * n_ctrl, np.inf)])

        qp: dict[str, Any] = {"x": ca.vertcat(*x_parts), "f": cost, "p": x0_p}
        if g_parts:
            qp["g"] = ca.vertcat(*g_parts)
        self._lbx, self._ubx = lbx, ubx
        self._lbg, self._ubg = lbg, ubg
        self._has_g = bool(g_parts)

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
            seed = [u_guess.reshape(-1)]
        else:
            # Forward-simulate from x0 to seed the lifted states with a feasible guess.
            x = x0
            x_guess = []
            for step in range(h):
                x = np.asarray(self.model.f_step(x, u_guess[step])).reshape(-1)
                x_guess.append(x)
            seed = [u_guess.reshape(-1), *x_guess]
        if self.w_u_l1 > 0:
            seed.append(np.abs(u_guess).reshape(-1))  # slacks t = |u|
        w0 = np.concatenate(seed)

        call: dict[str, Any] = {"x0": w0, "lbx": self._lbx, "ubx": self._ubx, "p": x0}
        if self._has_g:
            call["lbg"], call["ubg"] = self._lbg, self._ubg
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
            u_zero = np.zeros(self.n_controls, dtype=np.float64)
            self._u_buf = np.vstack([self._u_buf[1:], u_zero.reshape(1, self.n_controls)])
            return u_zero, MPCControllerLog(u=u_zero, cost=0.0, success=True, warmup=True)

        x0 = np.concatenate([self._y_buf.reshape(-1), self._u_buf.reshape(-1)])
        u0, cost, success = self._solve(x0)
        self._u_buf = np.vstack([self._u_buf[1:], u0.reshape(1, self.n_controls)])
        return u0, MPCControllerLog(u=u0.copy(), cost=cost, success=success, warmup=False)
