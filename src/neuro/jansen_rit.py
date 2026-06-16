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

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Self

import numba
import numpy as np
import numpy.typing as npt
from simulate.dynamics import Dynamics

if TYPE_CHECKING:
    from neuro.connectome import Connectome

FloatArray = npt.NDArray[np.float64]


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
    """

    A: float | FloatArray = 3.25
    B: float = 22.0
    a: float = 100.0
    b: float = 50.0
    C1: float = 135.0
    C2: float = 108.0  # 0.8 * C, C = 135
    C3: float = 33.75  # 0.25 * C
    C4: float = 33.75  # 0.25 * C
    e0: float = 2.5
    v0: float = 6.0
    r: float = 0.56
    mean_input: float = 90.0
    sigma: float = 500.0

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
        """Build :class:`JansenRitParams` from a config dict merged with the defaults."""
        if config.get("A") is None:
            return JansenRitParams(**config)

        a_config = config.pop("A")
        a_val = (
            np.asarray(a_config, dtype=np.float64)
            if isinstance(a_config, (list, tuple, np.ndarray))
            else float(a_config)
        )
        return JansenRitParams(A=a_val, **config)


def _to_scalar(val: FloatArray | float) -> float:
    if isinstance(val, np.ndarray):
        return float(val.item())
    return float(val)


def _to_array(val: FloatArray | float, n_nodes: int) -> FloatArray:
    if isinstance(val, np.ndarray):
        return np.asarray(val, dtype=np.float64)
    return np.full(n_nodes, val, dtype=np.float64)


@numba.njit(fastmath=True, cache=True)
def sigmoid_jit(v: FloatArray | float, e0: float, v0: float, r: float) -> FloatArray | float:
    """Evaluate the firing-rate sigmoid ``S(v) = 2 e0 / (1 + exp(r (v0 - v)))``."""
    return 2.0 * e0 / (1.0 + np.exp(r * (v0 - v)))


@numba.njit(fastmath=True, cache=True)
def _jr_rhs_jit(
    x: FloatArray, params_tuple: tuple[Any, ...], coupling: FloatArray | float, u_tes: FloatArray | float
) -> FloatArray:
    A, B, a, b, C1, C2, C3, C4, e0, v0, r, mean_input, _ = params_tuple  # noqa: N806

    x1, x2, x3, x4, x5, x6 = x

    out = sigmoid_jit(x2 - x3 + u_tes, e0, v0, r)  # pyramidal output; + U_tES (Stage 3)
    exc = sigmoid_jit(C1 * x1, e0, v0, r)  # drive to excitatory interneurons
    inh = sigmoid_jit(C3 * x1, e0, v0, r)  # drive to inhibitory interneurons

    dx1 = x4
    dx4 = A * a * out - 2.0 * a * x4 - a * a * x1
    dx2 = x5
    dx5 = A * a * (mean_input + C2 * exc + coupling) - 2.0 * a * x5 - a * a * x2
    dx3 = x6
    dx6 = B * b * C4 * inh - 2.0 * b * x6 - b * b * x3

    if x.ndim == 1:
        res = np.empty(6, dtype=np.float64)
        res[0] = dx1
        res[1] = dx2
        res[2] = dx3
        res[3] = dx4
        res[4] = dx5
        res[5] = dx6
        return res
    res = np.empty((6, x.shape[1]), dtype=np.float64)
    res[0, :] = dx1
    res[1, :] = dx2
    res[2, :] = dx3
    res[3, :] = dx4
    res[4, :] = dx5
    res[5, :] = dx6
    return res


@numba.njit(fastmath=True, cache=True)
def _heun_step_jit(  # noqa: PLR0913
    x: FloatArray,
    u_tes: FloatArray | float,
    params_tuple: tuple[Any, ...],
    dt: float,
    xi: FloatArray | float,
    coupling: FloatArray | float,
) -> FloatArray:
    sigma = params_tuple[12]

    dw = np.zeros(x.shape, dtype=np.float64)
    dw[4] = sigma * np.sqrt(dt) * xi  # additive noise enters x5 only

    f0 = _jr_rhs_jit(x, params_tuple, coupling, u_tes)
    x_pred = x + dt * f0 + dw
    f1 = _jr_rhs_jit(x_pred, params_tuple, coupling, u_tes)
    return x + 0.5 * dt * (f0 + f1) + dw


def heun_step(  # noqa: PLR0913
    x: FloatArray,
    u_tes: FloatArray | float,
    params: JansenRitParams,
    dt: float,
    xi: FloatArray | float,
    coupling: FloatArray | float = 0.0,
) -> FloatArray:
    """Advance one stochastic-Heun step with additive noise on the ``x5'`` equation.

    Parameters
    ----------
    x
        Current state, shape ``(6,)`` or ``(6, N)``.
    u_tes
        tES stimulation vector/scalar entering the pyramidal sigmoid.
    params
        Model parameters (``params.sigma`` scales the noise).
    dt
        Integration step in seconds.
    xi
        Standard-normal draw(s) for this step: a scalar for one node, shape
        ``(N,)`` for ``N`` nodes.
    coupling
        Network coupling term, shape ``(N,)`` or scalar.

    Returns
    -------
    FloatArray
        Next state, same shape as ``x``.
    """
    n_nodes = 1 if x.ndim == 1 else x.shape[1]
    params_tuple = params.to_numba_tuple(n_nodes)

    if n_nodes == 1:
        return _heun_step_jit(x, _to_scalar(u_tes), params_tuple, dt, _to_scalar(xi), _to_scalar(coupling))
    return _heun_step_jit(
        x, _to_array(u_tes, n_nodes), params_tuple, dt, _to_array(xi, n_nodes), _to_array(coupling, n_nodes)
    )


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
    connectome: Connectome | None = None,
    K: float = 0.0,  # noqa: N803
    seed: int | None = None,
    initial_state: FloatArray | None = None,
    u_hat_tES: float | FloatArray = 0.0,  # noqa: N803
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
    connectome
        Structural connectome (weights + conduction delays), or ``None`` for one node.
        A connectome with zero delays gives instantaneous coupling.
    K
        Global coupling scaling factor.
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
        connectome=connectome,
        K=K,
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


def output(x_traj: FloatArray) -> FloatArray:
    """Observed output ``y = x2 - x3`` from a state trajectory or state."""
    return x_traj[1] - x_traj[2]


@dataclass(frozen=True)
class JansenRitDynamicsLog:
    """Dataclass log snapshot of the Jansen-Rit network state."""

    x: np.ndarray


class JansenRitDynamics(Dynamics[JansenRitDynamicsLog]):
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

    def __init__(  # noqa: PLR0913
        self,
        *,
        dt: float,
        params: JansenRitParams,
        connectome: Connectome | None = None,
        K: float = 0.0,  # noqa: N803
        seed: int | None = None,
        initial_state: FloatArray | None = None,
    ) -> None:
        """Initialize the network plant from a connectome and coupling ``K``."""
        super().__init__(dt, integrator=None)
        self.K = K

        if connectome is None:
            n_nodes = 1
            weights = np.zeros((1, 1), dtype=np.float64)
            delays = np.zeros((1, 1), dtype=np.float64)
            gamma = None
        else:
            n_nodes = connectome.weights.shape[0]
            weights = connectome.weights
            delays = connectome.delays
            gamma = connectome.gamma

        a_vec = params.A
        if np.isscalar(a_vec):
            a_vec = np.full(n_nodes, a_vec, dtype=np.float64)
        self.net_params = replace(params, A=a_vec)
        self.params_tuple = self.net_params.to_numba_tuple(n_nodes)

        # tES steering matrix gamma_2d of shape (n_elec, n_nodes); single electrode is n_elec=1.
        self.gamma_2d = None if gamma is None else np.atleast_2d(gamma)
        self.n_elec = 1 if self.gamma_2d is None else self.gamma_2d.shape[0]

        if initial_state is not None:
            if initial_state.shape != (6, n_nodes):
                msg = f"initial_state must have shape (6, {n_nodes}), got {initial_state.shape}"
                raise ValueError(msg)
            self.x = initial_state.astype(np.float64).copy()
        else:
            self.x = np.zeros((6, n_nodes), dtype=np.float64)

        # Delays (ms) -> integer step lag; a connectome with zero delays is instantaneous.
        self.delay_steps = np.round(delays / (dt * 1000.0)).astype(np.int64)
        self.max_history_len = int(np.max(self.delay_steps)) + 1

        # Circular history buffer of S(y), seeded from the initial state.
        self.history = np.zeros((self.max_history_len, n_nodes), dtype=np.float64)
        self.history[:, :] = sigmoid_jit(self.x[1] - self.x[2], params.e0, params.v0, params.r)
        self.w_weights = weights

        self.rng = np.random.default_rng(seed)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate from a raw config dict, loading the TVB connectome by ``speed``.

        Config keys for the dynamics component are flat. Parameters for the physical
        model should be nested under the ``params`` key. ``speed``, ``K`` and ``dt``
        are required.
        """
        from neuro.connectome import compute_gamma, load_connectome  # noqa: PLC0415

        connectome = load_connectome(speed=float(config["speed"]))

        n_nodes_val = config.get("n_nodes")
        if n_nodes_val is not None:
            n_nodes = int(n_nodes_val)
            connectome = replace(
                connectome,
                weights=connectome.weights[:n_nodes, :n_nodes],
                tract_lengths=connectome.tract_lengths[:n_nodes, :n_nodes],
                centres=connectome.centres[:n_nodes],
                region_labels=connectome.region_labels[:n_nodes],
                hemispheres=connectome.hemispheres[:n_nodes],
                delays=connectome.delays[:n_nodes, :n_nodes],
                gain=connectome.gain[:, :n_nodes],
                region_index={label: idx for idx, label in enumerate(connectome.region_labels[:n_nodes])},
            )

        target_electrode = config.get("target_electrode")
        if target_electrode is not None:
            gamma = compute_gamma(
                connectome.centres,
                target_electrode=target_electrode,
                sigma=config.get("gamma_sigma", 20.0),
            )
            connectome = replace(connectome, gamma=gamma)

        params_config = config.get("params", {})
        params = JansenRitParams.from_config(params_config)

        return cls(
            dt=float(config["dt"]),
            connectome=connectome,
            K=float(config["K"]),
            params=params,
            seed=config.get("seed"),
        )

    def _project_control(self, u: np.ndarray) -> FloatArray | float:
        """Project per-electrode tES current ``u`` onto nodes via ``gamma``."""
        u_vec = np.asarray(u, dtype=np.float64).reshape(-1)
        if not np.any(u_vec):
            return 0.0
        if self.gamma_2d is None:
            msg = "tES stimulation is active but connectome.gamma is not configured."
            raise ValueError(msg)
        if u_vec.size == 1:
            u_vec = np.broadcast_to(u_vec, (self.n_elec,))
        elif u_vec.size != self.n_elec:
            msg = f"control has {u_vec.size} electrodes but gamma has {self.n_elec}"
            raise ValueError(msg)
        return u_vec @ self.gamma_2d

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

        u_node = self._project_control(u)
        xi = self.rng.standard_normal(n_nodes)
        return _heun_step_jit(x, _to_array(u_node, n_nodes), self.params_tuple, self.dt, xi, coupling)

    def _make_log(self) -> JansenRitDynamicsLog:
        """Build a snapshot log of the current network state."""
        return JansenRitDynamicsLog(x=self.x.copy())
