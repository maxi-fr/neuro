from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, Self

import numba
import numpy as np
import numpy.typing as npt
from pydantic import Field
from simulate.component import NoLog
from simulate.dynamics import Dynamics

from neuro.config import StrictConfig, parse_array
from neuro.connectome import Connectome, _ConnectomeConfig
from neuro.stimulation import NullStim, StimulationConfig, build_stimulation
from neuro.stimulation.base import _NullConfig

if TYPE_CHECKING:
    from neuro.stimulation import StimulationModel
    from neuro.types import FloatArray


class _JansenRitParamsConfig(StrictConfig):
    """Config schema for :meth:`JansenRitParams.from_config` (the biophysical scalars)."""

    A: float | list[float] | str = 3.25
    B: float = 22.0
    a: float = 100.0
    b: float = 50.0
    C1: float = 135.0
    C2: float = 108.0
    C3: float = 33.75
    C4: float = 33.75
    e0: float = 2.5
    v0: float = 6.0
    r: float = 0.56
    mean_input: float = 90.0
    sigma: float = 500.0


@dataclass(frozen=True)
class JansenRitParams:
    """Jansen-Rit parameters, using the paper's symbols (SI seconds).

    Attributes
    ----------
    A
        Excitatory synaptic gain in mV -- the bifurcation knob (3.25 = background,
        3.6 = limit cycle). Scalar or a vector of shape (N,) for network nodes.
    B
        Inhibitory synaptic gain in mV.
    a
        Excitatory rate constant in 1/s (inverse time constant).
    b
        Inhibitory rate constant in 1/s.
    C1, C2, C3, C4
        Intra-column connectivity constants, the standard ``C`` fractions with
        ``C = 135``: ``C1 = C``, ``C2 = 0.8 C``, ``C3 = C4 = 0.25 C``.
    e0
        Half of the maximum population firing rate in 1/s (so ``2 e0`` is the
        sigmoid ceiling and ``S(v0) = e0``).
    v0
        Sigmoid midpoint in mV.
    r
        Sigmoid steepness in 1/mV.
    mean_input
        Constant excitatory input ``I`` to the excitatory population.
    sigma
        Standard deviation of the additive Gaussian white noise ``zeta`` on the
        ``x5'`` equation; ``sigma = 0`` gives a noiseless (deterministic) run.

    The network structure (weights, delays, EEG gain, global coupling ``K``) lives on
    :class:`~neuro.connectome.Connectome`, and the tES projection on a
    :class:`~neuro.stimulation.base.StimulationModel`, not here.
    """

    A: float | FloatArray = field(default=3.25)
    B: float = field(default=22.0)
    a: float = field(default=100.0)
    b: float = field(default=50.0)
    C1: float = field(default=135.0)
    C2: float = field(default=108.0)
    C3: float = field(default=33.75)
    C4: float = field(default=33.75)
    e0: float = field(default=2.5)
    v0: float = field(default=6.0)
    r: float = field(default=0.56)
    mean_input: float = field(default=90.0)
    sigma: float = field(default=500.0)

    def to_numba_tuple(self, n_nodes: int) -> tuple[Any, ...]:
        """Convert params to a JIT-friendly tuple, broadcasting regional parameters."""
        a_gains = self.A
        if n_nodes == 1:
            a_gains_val = float(a_gains.item()) if isinstance(a_gains, np.ndarray) else float(a_gains)
        else:
            a_gains_val = (
                np.asarray(a_gains, dtype=np.float64)
                if isinstance(a_gains, np.ndarray)
                else np.full(n_nodes, a_gains, dtype=np.float64)
            )

        return (
            a_gains_val,
            self.B,
            self.a,
            self.b,
            self.C1,
            self.C2,
            self.C3,
            self.C4,
            self.e0,
            self.v0,
            self.r,
            self.mean_input,
            self.sigma,
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> JansenRitParams:
        """Build :class:`JansenRitParams` from a strictly-validated config dict."""
        kwargs = _JansenRitParamsConfig.model_validate(config).model_dump()
        a = parse_array(kwargs["A"])
        kwargs["A"] = np.asarray(a, dtype=np.float64) if isinstance(a, (list, tuple, np.ndarray)) else float(a)
        return cls(**kwargs)


@numba.njit(fastmath=True, cache=True)
def sigmoid_jit(v: FloatArray, e0: float, v0: float, r: float) -> FloatArray:
    """Evaluate the firing-rate sigmoid ``S(v) = 2 e0 / (1 + exp(r (v0 - v)))``."""
    return 2.0 * e0 / (1.0 + np.exp(r * (v0 - v)))


@numba.njit(fastmath=True, cache=True)
def _jr_rhs_jit(x: FloatArray, params_tuple: tuple[Any, ...], coupling: FloatArray, u_tes: FloatArray) -> FloatArray:
    A, B, a, b, C1, C2, C3, C4, e0, v0, r, mean_input, _ = params_tuple

    x1, x2, x3, x4, x5, x6 = x

    out = sigmoid_jit(x2 - x3 + u_tes, e0, v0, r)
    exc = sigmoid_jit(C1 * x1, e0, v0, r)
    inh = sigmoid_jit(C3 * x1, e0, v0, r)

    res = np.empty_like(x)
    res[0, :] = x4
    res[3, :] = A * a * out - 2.0 * a * x4 - a * a * x1
    res[1, :] = x5
    res[4, :] = A * a * (mean_input + C2 * exc + coupling) - 2.0 * a * x5 - a * a * x2
    res[2, :] = x6
    res[5, :] = B * b * C4 * inh - 2.0 * b * x6 - b * b * x3
    return res


@numba.njit(fastmath=True, cache=True)
def _heun_step_jit(  # noqa: PLR0913, PLR0917
    x: FloatArray,
    u_tes: FloatArray,
    params_tuple: tuple[Any, ...],
    dt: float,
    xi: FloatArray,
    coupling: FloatArray,
) -> FloatArray:
    sigma = params_tuple[12]

    dw = np.zeros_like(x)
    dw[4, :] = sigma * np.sqrt(dt) * xi

    f0 = _jr_rhs_jit(x, params_tuple, coupling, u_tes)
    x_pred = x + dt * f0 + dw
    f1 = _jr_rhs_jit(x_pred, params_tuple, coupling, u_tes)
    return x + 0.5 * dt * (f0 + f1) + dw


@numba.njit(fastmath=True, cache=True)
def _dynamics_history_coupling_jit(  # noqa: PLR0913, PLR0917
    history: FloatArray,
    k: int,
    max_history_len: int,
    delay_steps: npt.NDArray[np.int64],
    w_weights: FloatArray,
    coupling_k: float,
    s_y: FloatArray,
) -> FloatArray:
    history[k % max_history_len, :] = s_y

    n_nodes = w_weights.shape[0]
    coupling = np.zeros(n_nodes, dtype=np.float64)
    for i in range(n_nodes):
        c_i = 0.0
        for j in range(n_nodes):
            delay = delay_steps[i, j]
            row = (k - delay) % max_history_len
            c_i += w_weights[i, j] * history[row, j]
        coupling[i] = coupling_k * c_i
    return coupling


def simulate_network(
    *,
    dyn: JansenRitDynamics,
    duration: float,
    u_hat_tES: float | FloatArray = 0.0,
    stim_window: tuple[float, float] | None = None,
    t0: float = 0.0,
) -> tuple[FloatArray, FloatArray]:
    """Step a prebuilt plant from its current state and return the full trajectory.

    The plant ``dyn`` already owns its step ``dt``, RNG seed, connectome and initial
    state; a single-node ``dyn`` (``N = 1``) is the degenerate isolated node.

    Parameters
    ----------
    dyn
        The Jansen-Rit plant to integrate; stepped ``round(duration / dyn.dt)`` times.
    duration
        Simulated time in seconds.
    u_hat_tES
        Constant tES current applied during ``stim_window``. A scalar (shared by
        every electrode) or shape ``(n_electrodes,)`` for per-electrode currents.
    stim_window
        Time window (start_s, end_s) during which tES stimulation is active, on the
        same absolute clock as ``t0``.
    t0
        Absolute start time of this segment in seconds. Components hold their own
        monotonic clock, so continuing a plant with a second call requires handing it
        the time it left off at; leaving ``t0`` at 0 would make the plant treat the
        whole segment as already elapsed and silently hold its state.

    Returns
    -------
    t
        Time vector in seconds, shape ``(n_samples,)``, starting at ``t0``.
    x_traj
        State trajectory, shape ``(6, N, n_samples)`` (``N = 1`` for a single node).
    """
    dt = dyn.dt
    n_steps = round(duration / dt)
    n_nodes = dyn.x.shape[1]

    n_controls = dyn.n_controls
    u_amp = np.atleast_1d(np.asarray(u_hat_tES, dtype=np.float64))
    if u_amp.shape[0] == 1:
        u_amp = np.broadcast_to(u_amp, (n_controls,))
    elif u_amp.shape[0] != n_controls:
        msg = f"u_hat_tES has {u_amp.shape[0]} electrodes but the plant has {n_controls}"
        raise ValueError(msg)

    u_sched = np.zeros((n_steps, n_controls), dtype=np.float64)
    if stim_window is not None:
        t_grid = t0 + np.arange(n_steps) * dt
        u_sched[(t_grid >= stim_window[0]) & (t_grid < stim_window[1])] = u_amp

    x_traj = np.zeros((6, n_nodes, n_steps + 1), dtype=np.float64)
    x_traj[:, :, 0] = dyn.x
    for k in range(n_steps):
        dyn.evaluate(t0 + k * dt, u_sched[k])
        x_traj[:, :, k + 1] = dyn.x

    t = t0 + np.arange(n_steps + 1, dtype=np.float64) * dt
    return t, x_traj


def lfp(x_traj: FloatArray) -> FloatArray:
    """Observed output ``y = x2 - x3`` from a state trajectory or state."""
    return x_traj[1] - x_traj[2]


def resting_state(conn: Connectome, dt: float, settle_s: float = 2.0) -> FloatArray:
    """Find the healthy network's fixed point, shape ``(6, N)``, to start a run from."""
    dyn = JansenRitDynamics(dt=dt, params=JansenRitParams(sigma=0.0), conn=conn)
    _, x_traj = simulate_network(dyn=dyn, duration=settle_s)
    return x_traj[:, :, -1].copy()


_ZERO_SUM_CURRENT_ATOL = 1e-6


def _assert_zero_sum_current(u: FloatArray) -> None:
    """Raise if the per-electrode currents violate Kirchhoff's current law."""
    total = float(np.sum(u))
    if abs(total) > _ZERO_SUM_CURRENT_ATOL * (1.0 + float(np.abs(u).max())):
        msg = (
            f"stimulation currents violate Kirchhoff's current law: they sum to {total:.3e} "
            f"(u={np.asarray(u, dtype=np.float64).tolist()}), but a tES montage injects no net "
            "current. Balance the montage so the currents sum to zero, or construct the plant "
            "with enforce_zero_sum_current=False."
        )
        raise ValueError(msg)


@dataclass
class JansenRitStateLog:
    """Log carrying the full network state x of shape (6, n_nodes)."""

    x: FloatArray


@dataclass
class JansenRitLFPLog:
    """Log carrying the network local field potential y = x2 - x3 of shape (n_nodes,)."""

    lfp: FloatArray


class _JansenRitDynamicsConfig(StrictConfig):
    """Config schema for :meth:`JansenRitDynamics.from_config`."""

    dt: float = Field(gt=0)
    seed: int | None = None
    enforce_zero_sum_current: bool = True
    initial_state: Literal["zeros", "rest"] = "zeros"
    log: Literal["none", "lfp", "state"] = "none"
    connectome: _ConnectomeConfig = Field(default_factory=_ConnectomeConfig)
    stimulation: StimulationConfig = Field(default_factory=_NullConfig)
    params: _JansenRitParamsConfig = Field(default_factory=_JansenRitParamsConfig)


class JansenRitDynamics(Dynamics[JansenRitStateLog | JansenRitLFPLog | NoLog]):
    """Whole-brain Jansen-Rit network as a ``simulate`` :class:`Dynamics` plant."""

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        dt: float,
        params: JansenRitParams,
        conn: Connectome,
        stim: StimulationModel | None = None,
        seed: int | None = None,
        initial_state: FloatArray | None = None,
        *,
        enforce_zero_sum_current: bool = True,
        log: Literal["none", "lfp", "state"] = "none",
    ) -> None:
        """Initialize the network plant from ``params``, the structural ``conn`` and ``stim``.

        ``stim=None`` means unstimulated: a single control electrode that drives nothing.
        """
        super().__init__(dt, integrator=None)
        self.K = conn.K
        self.enforce_zero_sum_current = enforce_zero_sum_current
        self.log_mode = log

        n_nodes = conn.weights.shape[0]
        weights = conn.weights
        delays = conn.delay_steps(dt)

        a_vec = params.A
        if np.isscalar(a_vec):
            a_vec = np.full(n_nodes, a_vec, dtype=np.float64)
        self.net_params = replace(params, A=a_vec)
        self.params_tuple = self.net_params.to_numba_tuple(n_nodes)

        self.stim = NullStim(n_nodes) if stim is None else stim
        self.n_controls = self.n_inputs = self.stim.n_controls

        if initial_state is not None:
            if initial_state.shape != (6, n_nodes):
                msg = f"initial_state must have shape (6, {n_nodes}), got {initial_state.shape}"
                raise ValueError(msg)
            self.x = initial_state.astype(np.float64).copy()
        else:
            self.x = np.zeros((6, n_nodes), dtype=np.float64)

        self.delay_steps = delays
        self.max_history_len = int(np.max(self.delay_steps)) + 1

        self.history = np.zeros((self.max_history_len, n_nodes), dtype=np.float64)
        self.history[:, :] = sigmoid_jit(self.x[1] - self.x[2], params.e0, params.v0, params.r)
        self.w_weights = weights

        self.rng = np.random.default_rng(seed)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the plant from a raw config dict."""
        cfg = _JansenRitDynamicsConfig.model_validate(config)
        conn = Connectome.from_config(cfg.connectome.model_dump())
        params = JansenRitParams.from_config(cfg.params.model_dump())

        return cls(
            dt=cfg.dt,
            params=params,
            conn=conn,
            stim=build_stimulation(cfg.stimulation, conn),
            seed=cfg.seed,
            initial_state=resting_state(conn, cfg.dt) if cfg.initial_state == "rest" else None,
            enforce_zero_sum_current=cfg.enforce_zero_sum_current,
            log=cfg.log,
        )

    def dynamics(self, t: float, x: FloatArray, u: FloatArray) -> FloatArray:
        """Advance the network one stochastic-Heun step (``sigma = 0`` is noiseless)."""
        n_nodes = x.shape[1]
        k = round(t / self.dt)

        s_y = sigmoid_jit(x[1] - x[2], self.net_params.e0, self.net_params.v0, self.net_params.r)
        coupling = _dynamics_history_coupling_jit(
            self.history,
            k,
            self.max_history_len,
            self.delay_steps,
            self.w_weights,
            self.K,
            s_y,
        )

        if self.enforce_zero_sum_current:
            _assert_zero_sum_current(u)
        u_node = self.stim.project(u)
        xi = self.rng.standard_normal(n_nodes)
        return _heun_step_jit(x, u_node, self.params_tuple, self.dt, xi, coupling)

    def _make_log(self) -> JansenRitStateLog | JansenRitLFPLog | NoLog:
        """Build a snapshot log of the current network state or LFP, if requested."""
        if self.log_mode == "state":
            return JansenRitStateLog(x=self.x.copy())
        if self.log_mode == "lfp":
            return JansenRitLFPLog(lfp=lfp(self.x))
        return NoLog()
