"""Stage 1 -- single-node Jansen-Rit neural mass, hand-rolled in NumPy + numba.

Implements the Yu et al. 2024 formulation (their Eq. 1-4) for one isolated,
uncontrolled node. The state vector is ``x = [x1, x2, x3, x4, x5, x6]`` with the
paper's 1-6 indexing -- pyramidal pair ``(x1, x4)``, excitatory pair ``(x2, x5)``,
inhibitory pair ``(x3, x6)`` -- and the observed output is ``y = x2 - x3``::

    x1' = x4
    x4' = A a [ S(x2 - x3 + U_tES) ]                       - 2a x4 - a^2 x1
    x2' = x5
    x5' = A a [ I + C2 S(C1 x1) + K sum_j w_ij S(x2-x3) ]  - 2a x5 - a^2 x2 + zeta
    x3' = x6
    x6' = B b [ C4 S(C3 x1) ]                              - 2b x6 - b^2 x3
    S(v) = 2 e0 / (1 + exp(r (v0 - v)))

The tES polarization ``U_tES`` (Stage 3) and the network coupling sum
``K sum_j w_ij S(.)`` (Stage 2) enter :func:`_jr_rhs_jit`; both are zero for an
isolated, uncontrolled node. Default parameters follow Yu et al. 2024, Table 1; the only free knob is the
white-noise std ``sigma`` (the paper gives no value), tuned so the ``A = 3.25``
background sits near peak-to-peak amplitude 2.

Integration is stochastic Heun with additive noise on the ``x5'`` equation; a
noiseless run is just ``sigma = 0``. The right-hand side is written element-wise, so
the same code runs a single node (state shape ``(6,)``) or ``N`` nodes (``(6, N)``).
Everything is in SI seconds: ``a = 100``, ``b = 50`` per second, ``dt = 1e-4`` s
(0.1 ms, matching the Stage 7 TVB reference step).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Self

import numba
import numpy as np
import numpy.typing as npt
from simulate.component import NoLog
from simulate.dynamics import Dynamics

from neuro.config import parse_array

FloatArray = npt.NDArray[np.float64]


def delays_to_steps(delays_ms: FloatArray, dt: float) -> npt.NDArray[np.int64]:
    """Convert conduction delays (ms) to integer step lags for integration step ``dt`` (s)."""
    return np.round(delays_ms / (dt * 1000.0)).astype(np.int64)


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
    eeg_gain
        Leadfield matrix for observing the brain regions, mapping source to sensors.
    w_weights
        Network connectivity weight matrix.
    delay_steps
        Network delay matrix in integer steps.
    K
        Global coupling scaling factor.
    gamma
        Spatial profile mapping electrode stimulation to brain regions.
    """

    A: float | FloatArray = field(default=3.25, metadata={"bounds": (0.0, 10.0)})
    B: float = field(default=22.0, metadata={"bounds": (0.0, 100.0)})
    a: float = field(default=100.0, metadata={"bounds": (10.0, 500.0)})
    b: float = field(default=50.0, metadata={"bounds": (1.0, 200.0)})
    C1: float = field(default=135.0, metadata={"bounds": (10.0, 500.0)})
    C2: float = field(default=108.0, metadata={"bounds": (10.0, 500.0)})  # 0.8 * C, C = 135
    C3: float = field(default=33.75, metadata={"bounds": (1.0, 200.0)})  # 0.25 * C
    C4: float = field(default=33.75, metadata={"bounds": (1.0, 200.0)})  # 0.25 * C
    e0: float = field(default=2.5, metadata={"bounds": (0.1, 10.0)})
    v0: float = field(default=6.0, metadata={"bounds": (0.1, 20.0)})
    r: float = field(default=0.56, metadata={"bounds": (0.1, 2.0)})
    mean_input: float | FloatArray = field(default=90.0, metadata={"bounds": (0.0, 300.0)})
    sigma: float = field(default=500.0, metadata={"bounds": (0.0, 0.0)})

    # Network parameters (mirrors JRSymbolicParams)
    eeg_gain: FloatArray = field(
        default_factory=lambda: np.ones((1, 1), dtype=np.float64), metadata={"bounds": (-10.0, 10.0)}
    )
    w_weights: FloatArray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float64))
    delay_steps: npt.NDArray[np.int64] = field(default_factory=lambda: np.zeros((1, 1), dtype=np.int64))
    K: float = field(default=1.0, metadata={"bounds": (0.0, 5.0)})
    gamma: FloatArray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float64))
    initial_bounds: FloatArray | None = field(default=None, metadata={"bounds": None})

    def to_numba_tuple(self, n_nodes: int) -> tuple[Any, ...]:
        """Convert params to a JIT-friendly tuple, broadcasting regional parameters."""
        a_gains = self.A
        if n_nodes == 1:
            a_gains_val = float(a_gains.item()) if isinstance(a_gains, np.ndarray) else float(a_gains)  # ty:ignore[no-matching-overload]
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
        """Build :class:`JansenRitParams` from a global config dict.

        Extracts the network topology via ``Connectome.from_config`` (or uses a provided
        `connectome` key) and the scalar parameters strictly from the nested `"params"` key.
        """
        # Load the connectome
        conn = config.get("connectome")
        if conn is None:
            from neuro.connectome import Connectome  # noqa: PLC0415

            conn = Connectome.from_config(config)

        dt = float(config.get("dt", 0.001))

        if "params" not in config:
            msg = "JansenRitParams.from_config requires a nested 'params' dictionary in the config."
            raise ValueError(msg)

        params_cfg = dict(config["params"])

        network_kwargs = {
            "eeg_gain": conn.gain,
            "w_weights": conn.weights,
            "delay_steps": delays_to_steps(conn.delays, dt),
            "gamma": conn.gamma,
        }

        conflicts = network_kwargs.keys() & params_cfg.keys()
        if conflicts:
            msg = (
                f"Network parameters {sorted(conflicts)} come from the connectome and cannot be overridden in 'params'"
            )
            raise ValueError(msg)

        for key in ["A", "mean_input"]:
            if params_cfg.get(key) is not None:
                params_cfg[key] = parse_array(params_cfg[key])

        if params_cfg.get("A") is not None:
            a_config = params_cfg.pop("A")
            params_cfg["A"] = (
                np.asarray(a_config, dtype=np.float64)
                if isinstance(a_config, (list, tuple, np.ndarray))
                else float(a_config)
            )

        if params_cfg.get("initial_bounds") is not None:
            params_cfg["initial_bounds"] = np.asarray(params_cfg["initial_bounds"], dtype=np.float64)

        return cls(**network_kwargs, **params_cfg)  # type: ignore


@numba.njit(fastmath=True, cache=True)
def sigmoid_jit(v: FloatArray, e0: float, v0: float, r: float) -> FloatArray:
    """Evaluate the firing-rate sigmoid ``S(v) = 2 e0 / (1 + exp(r (v0 - v)))``."""
    return 2.0 * e0 / (1.0 + np.exp(r * (v0 - v)))


@numba.njit(fastmath=True, cache=True)
def _jr_rhs_jit(x: FloatArray, params_tuple: tuple[Any, ...], coupling: FloatArray, u_tes: FloatArray) -> FloatArray:
    A, B, a, b, C1, C2, C3, C4, e0, v0, r, mean_input, _ = params_tuple

    x1, x2, x3, x4, x5, x6 = x

    out = sigmoid_jit(x2 - x3 + u_tes, e0, v0, r)  # pyramidal output; + U_tES (Stage 3)
    exc = sigmoid_jit(C1 * x1, e0, v0, r)  # drive to excitatory interneurons
    inh = sigmoid_jit(C3 * x1, e0, v0, r)  # drive to inhibitory interneurons

    res = np.empty_like(x)
    res[0, :] = x4
    res[3, :] = A * a * out - 2.0 * a * x4 - a * a * x1
    res[1, :] = x5
    res[4, :] = A * a * (mean_input + C2 * exc + coupling) - 2.0 * a * x5 - a * a * x2
    res[2, :] = x6
    res[5, :] = B * b * C4 * inh - 2.0 * b * x6 - b * b * x3
    return res


@numba.njit(fastmath=True, cache=True)
def _heun_step_jit(  # noqa: PLR0913
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


def heun_step(  # noqa: PLR0913
    x: FloatArray,
    u_tes: FloatArray,
    params: JansenRitParams,
    dt: float,
    xi: FloatArray,
    coupling: FloatArray,
) -> FloatArray:
    """Advance one stochastic-Heun step with additive noise on the ``x5'`` equation.

    Parameters
    ----------
    x
        Current state, shape ``(6, N)``.
    u_tes
        tES stimulation entering the pyramidal sigmoid, shape ``(N,)``.
    params
        Model parameters (``params.sigma`` scales the noise).
    dt
        Integration step in seconds.
    xi
        Standard-normal draws for this step, shape ``(N,)``.
    coupling
        Network coupling term, shape ``(N,)``.

    Returns
    -------
    FloatArray
        Next state, same shape as ``x``.
    """
    return _heun_step_jit(x, u_tes, params.to_numba_tuple(x.shape[1]), dt, xi, coupling)


@numba.njit(fastmath=True, cache=True)
def _dynamics_history_coupling_jit(  # noqa: PLR0913
    history: FloatArray,
    k: int,
    max_history_len: int,
    delay_steps: npt.NDArray[np.int64],
    w_weights: FloatArray,
    coupling_k: float,
    s_y: FloatArray,
) -> FloatArray:
    # 1. Update circular history buffer
    history[k % max_history_len, :] = s_y

    # 2. Compute delayed coupling (O(N^2) loop with zero allocations)
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


def simulate_network(  # noqa: PLR0913
    *,
    params: JansenRitParams,
    duration: float,
    dt: float,
    seed: int | None = None,
    initial_state: FloatArray | None = None,
    u_hat_tES: float | FloatArray = 0.0,
    stim_window: tuple[float, float] | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Integrate a node or coupled network from rest and return the full trajectory.

    Passing ``connectome=None`` runs the degenerate ``N = 1`` isolated node (no
    coupling, no stimulation); a real connectome runs the whole-brain network.

    Parameters
    ----------
    params
        Model parameters. If ``params.A`` is an array, it specifies regional gains.
        ``params.sigma = 0`` gives a noiseless (deterministic) run.
    duration
        Simulated time in seconds.
    dt
        Integration step in seconds.
    seed
        Seed for the noise stream.
    initial_state
        Initial state array of shape ``(6, N)``. If ``None``, defaults to zeros.
    u_hat_tES
        Constant tES current applied during ``stim_window``. A scalar (shared by
        every electrode) or shape ``(n_electrodes,)`` for per-electrode currents.
    stim_window
        Time window (start_s, end_s) during which tES stimulation is active.

    Returns
    -------
    t
        Time vector in seconds, shape ``(n_samples,)``.
    x_traj
        State trajectory, shape ``(6, N, n_samples)`` (``N = 1`` for a single node).
    """
    n_steps = round(duration / dt)

    dyn = JansenRitDynamics(
        dt=dt,
        params=params,
        seed=seed,
        initial_state=initial_state,
    )
    n_nodes = dyn.x.shape[1]

    # Per-electrode constant-current schedule over the stim window. Built once here so
    # the integration loop stays branch-free; the gamma projection lives in the component.
    n_elec = dyn.n_elec
    u_amp = np.atleast_1d(np.asarray(u_hat_tES, dtype=np.float64))
    if u_amp.shape[0] == 1:
        u_amp = np.broadcast_to(u_amp, (n_elec,))
    elif u_amp.shape[0] != n_elec:
        msg = f"u_hat_tES has {u_amp.shape[0]} electrodes but gamma has {n_elec}"
        raise ValueError(msg)

    u_sched = np.zeros((n_steps, n_elec), dtype=np.float64)
    if stim_window is not None:
        t_grid = np.arange(n_steps) * dt
        u_sched[(t_grid >= stim_window[0]) & (t_grid < stim_window[1])] = u_amp

    x_traj = np.zeros((6, n_nodes, n_steps + 1), dtype=np.float64)
    x_traj[:, :, 0] = dyn.x
    for k in range(n_steps):
        dyn.evaluate(k * dt, u_sched[k])
        x_traj[:, :, k + 1] = dyn.x

    t = np.arange(n_steps + 1, dtype=np.float64) * dt
    return t, x_traj


def lfp(x_traj: FloatArray) -> FloatArray:
    """Observed output ``y = x2 - x3`` from a state trajectory or state."""
    return x_traj[1] - x_traj[2]


def project_control(u: np.ndarray, gamma_2d: FloatArray, n_elec: int) -> FloatArray:
    """Project per-electrode tES current ``u`` onto nodes via ``gamma``.

    Parameters
    ----------
    u:
        Per-electrode control input, shape ``(n_elec,)``.
    gamma_2d:
        Steering matrix of shape ``(n_elec, n_nodes)``.
    n_elec:
        Number of electrodes (must match ``gamma_2d.shape[0]``).

    Returns
    -------
    FloatArray
        Node-level stimulation ``u @ gamma_2d`` of shape ``(n_nodes,)``.
    """
    if not np.any(u):
        return np.zeros(gamma_2d.shape[1], dtype=np.float64)
    if u.size != n_elec:
        msg = f"control has {u.size} electrodes but gamma has {n_elec}"
        raise ValueError(msg)
    return u @ gamma_2d


class JansenRitDynamics(Dynamics[NoLog]):
    """Whole-brain Jansen-Rit network as a ``simulate`` :class:`Dynamics` plant.

    Wraps the standalone :func:`simulate_network` integration as a stateful,
    single-step component for the :class:`~simulate.simulation.Simulation`
    orchestrator. It owns its own stepping (``integrator=None``, so the base
    :meth:`~simulate.dynamics.Dynamics.update` treats :meth:`dynamics` as a discrete
    ``x_next`` transition) and uses :func:`_heun_step_jit` so the stochastic ``x5``
    noise and delayed network coupling are preserved.

    The state ``self.x`` is the ``(6, N)`` network array; the orchestrator logs it as
    the flattened ``(6 N,)`` vector that :meth:`~simulate.component.Component.from_col_vec`
    produces. The control input ``u`` is the per-electrode tES current, projected to
    nodes through ``connectome.gamma`` (no-op when ``gamma`` is unset, i.e. open loop).

    Passing ``connectome=None`` yields the degenerate ``N=1`` isolated node (no coupling,
    no stimulation). Everything is in SI seconds: ``dt`` is the integration step and
    ``connectome.delays`` (ms) convert to steps via ``round(delays / (dt * 1000))``.
    """

    def __init__(
        self,
        dt: float,
        params: JansenRitParams,
        seed: int | None = None,
        initial_state: FloatArray | None = None,
    ) -> None:
        """Initialize the network plant from params."""
        super().__init__(dt, integrator=None)
        self.K = params.K

        n_nodes = params.w_weights.shape[0]
        weights = params.w_weights
        delays = params.delay_steps
        gamma = params.gamma

        a_vec = params.A
        if np.isscalar(a_vec):
            a_vec = np.full(n_nodes, a_vec, dtype=np.float64)
        self.net_params = replace(params, A=a_vec)
        self.params_tuple = self.net_params.to_numba_tuple(n_nodes)

        # tES steering matrix gamma_2d of shape (n_elec, n_nodes); single electrode is n_elec=1.
        self.gamma_2d = np.atleast_2d(gamma)
        self.n_elec = self.gamma_2d.shape[0]
        # The control input is the per-electrode tES current; the orchestrator seeds u with this width.
        self.n_inputs = self.n_elec

        self.rng = np.random.default_rng(seed)

        if initial_state is not None:
            if initial_state.shape != (6, n_nodes):
                msg = f"initial_state must have shape (6, {n_nodes}), got {initial_state.shape}"
                raise ValueError(msg)
            self.x = initial_state.astype(np.float64).copy()
        elif params.initial_bounds is not None:
            bounds = np.asarray(params.initial_bounds, dtype=np.float64)
            if bounds.shape != (6, 2):
                msg = f"initial_bounds must have shape (6, 2), got {bounds.shape}"
                raise ValueError(msg)
            lo = bounds[:, 0:1]
            hi = bounds[:, 1:2]
            self.x = self.rng.uniform(lo, hi, size=(6, n_nodes))
        else:
            self.x = np.zeros((6, n_nodes), dtype=np.float64)

        # Delays (ms) -> integer step lag; a connectome with zero delays is instantaneous.
        self.delay_steps = delays
        self.max_history_len = int(np.max(self.delay_steps)) + 1

        # Circular history buffer of S(y), seeded from the initial state.
        self.history = np.zeros((self.max_history_len, n_nodes), dtype=np.float64)
        if params.initial_bounds is not None and self.max_history_len > 0:
            bounds = np.asarray(params.initial_bounds, dtype=np.float64)
            lo = bounds[:, 0:1]
            hi = bounds[:, 1:2]
            hist_x1 = self.rng.uniform(lo[1, 0], hi[1, 0], size=(self.max_history_len, n_nodes))
            hist_x2 = self.rng.uniform(lo[2, 0], hi[2, 0], size=(self.max_history_len, n_nodes))
            hist_y = hist_x1 - hist_x2
            self.history[:, :] = sigmoid_jit(hist_y, params.e0, params.v0, params.r)
        else:
            self.history[:, :] = sigmoid_jit(self.x[1] - self.x[2], params.e0, params.v0, params.r)
        self.w_weights = weights

        self.rng = np.random.default_rng(seed)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate from a raw config dict.

        Config keys for the dynamics component are flat. Parameters for the physical
        model should be nested under the ``params`` key.
        """
        params = JansenRitParams.from_config(config)

        return cls(
            dt=float(config["dt"]),
            params=params,
            seed=config.get("seed"),
        )

    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Advance the network one stochastic-Heun step (``sigma = 0`` is noiseless).

        The circular-history write/read index is recovered from ``t`` via
        ``k = round(t / dt)``; this is exact as long as the plant is stepped at its own
        ``dt`` (so each call lands on ``t = k * dt``), which holds for single-rate runs
        and any multirate setup where ``dt`` is an integer multiple of the base tick.
        """
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

        u_node = project_control(u, self.gamma_2d, self.n_elec)
        xi = self.rng.standard_normal(n_nodes)
        return _heun_step_jit(x, u_node, self.params_tuple, self.dt, xi, coupling)

    def _make_log(self) -> NoLog:
        """Build a snapshot log of the current network state."""
        return NoLog()
