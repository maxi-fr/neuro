"""Stage 1 -- single-node Jansen-Rit neural mass, hand-rolled in NumPy.

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

At this stage the tES polarization ``U_tES`` (Stage 3) and the network coupling
sum ``K sum_j w_ij S(.)`` (Stage 2) are zero/absent; their slots are marked in
:func:`jr_rhs`. Parameters follow Yu et al. 2024, Table 1 (``I = 90``); the only
free knob is the white-noise std ``sigma`` (the paper gives no value), tuned here
so the ``A = 3.25`` background sits near peak-to-peak amplitude 2, then frozen.

Everything is in SI seconds: ``a = 100``, ``b = 50`` per second, ``dt = 1e-3`` s.
The right-hand side is written element-wise, so the same code runs a single node
(state shape ``(6,)``) or, later, ``N`` nodes at once (state shape ``(6, N)``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Self

import numba
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict
from simulate.dynamics import Dynamics

if TYPE_CHECKING:
    from neuro.connectome import Connectome

FloatArray = npt.NDArray[np.float64]

DT_DEFAULT = 1e-3
"""Locked integration step in seconds (fs = 1000 Hz). Unspecified by the paper."""


@dataclass(frozen=True)
class JansenRitParams:
    """Jansen-Rit parameters, using the paper's symbols (SI seconds).

    Attributes
    ----------
    A
        Excitatory synaptic gain in mV -- the bifurcation knob (3.25 = background,
        3.6 = limit cycle). Can be a scalar or a vector of shape (N,) for network nodes.
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
        Constant excitatory input ``I`` to the excitatory population (Yu et al.
        Table 1: ``I = 90``). At this drive ``A = 3.25`` is a stable fixed point
        and the oscillation onset sits between ``A = 3.4`` and ``A = 3.6``.
    sigma
        Standard deviation of the additive Gaussian white noise ``zeta`` on the
        ``x5'`` equation. Yu et al. specify "Gaussian white noise" but give no SD,
        so it is tuned here so the noisy ``A = 3.25`` background reaches
        peak-to-peak amplitude of roughly 2 (calibrating the closed-loop amplitude
        threshold of 5 used in later stages).
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
    sigma: float = 550.0

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

    out = sigmoid_jit(x2 - x3 + u_tes, e0, v0, r)
    exc = sigmoid_jit(C1 * x1, e0, v0, r)
    inh = sigmoid_jit(C3 * x1, e0, v0, r)

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


def jr_rhs(
    x: FloatArray,
    params: JansenRitParams,
    coupling: FloatArray | float = 0.0,
    u_tes: FloatArray | float = 0.0,
) -> FloatArray:
    """Right-hand side of the single-node or network Jansen-Rit ODEs.

    Parameters
    ----------
    x
        State, shape ``(6,)`` for one node or ``(6, N)`` for ``N`` nodes, ordered
        ``[x1, x2, x3, x4, x5, x6]``.
    params
        Model parameters.
    coupling
        Network coupling term, shape ``(N,)`` or scalar; enters the ``x5'`` equation.
    u_tes
        tES stimulation vector/scalar entering the pyramidal sigmoid in the ``x4'`` equation.

    Returns
    -------
    FloatArray
        Time derivative ``x'`` with the same shape as ``x``.
    """
    n_nodes = 1 if x.ndim == 1 else x.shape[1]
    params_tuple = params.to_numba_tuple(n_nodes)

    if n_nodes == 1:
        return _jr_rhs_jit(x, params_tuple, _to_scalar(coupling), _to_scalar(u_tes))
    return _jr_rhs_jit(x, params_tuple, _to_array(coupling, n_nodes), _to_array(u_tes, n_nodes))


@numba.njit(fastmath=True, cache=True)
def _rk4_step_jit(
    x: FloatArray, params_tuple: tuple[Any, ...], dt: float, coupling: FloatArray | float, u_tes: FloatArray | float
) -> FloatArray:
    k1 = _jr_rhs_jit(x, params_tuple, coupling, u_tes)
    k2 = _jr_rhs_jit(x + 0.5 * dt * k1, params_tuple, coupling, u_tes)
    k3 = _jr_rhs_jit(x + 0.5 * dt * k2, params_tuple, coupling, u_tes)
    k4 = _jr_rhs_jit(x + dt * k3, params_tuple, coupling, u_tes)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4_step(
    x: FloatArray,
    params: JansenRitParams,
    dt: float,
    coupling: FloatArray | float = 0.0,
    u_tes: FloatArray | float = 0.0,
) -> FloatArray:
    """Advance one deterministic RK4 step (noise off)."""
    n_nodes = 1 if x.ndim == 1 else x.shape[1]
    params_tuple = params.to_numba_tuple(n_nodes)

    if n_nodes == 1:
        return _rk4_step_jit(x, params_tuple, dt, _to_scalar(coupling), _to_scalar(u_tes))
    return _rk4_step_jit(x, params_tuple, dt, _to_array(coupling, n_nodes), _to_array(u_tes, n_nodes))


@numba.njit(fastmath=True, cache=True)
def _stochastic_rk4_step_jit(  # noqa: PLR0913
    x: FloatArray,
    params_tuple: tuple[Any, ...],
    dt: float,
    xi: FloatArray | float,
    coupling: FloatArray | float,
    u_tes: FloatArray | float,
) -> FloatArray:
    sigma = params_tuple[12]

    dw = np.zeros(x.shape, dtype=np.float64)
    dw[4] = sigma * np.sqrt(dt) * xi  # additive noise enters x5 only

    k1 = _jr_rhs_jit(x, params_tuple, coupling, u_tes) * dt + dw
    k2 = _jr_rhs_jit(x + 0.5 * k1, params_tuple, coupling, u_tes) * dt + dw
    k3 = _jr_rhs_jit(x + 0.5 * k2, params_tuple, coupling, u_tes) * dt + dw
    k4 = _jr_rhs_jit(x + k3, params_tuple, coupling, u_tes) * dt + dw

    return x + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def stochastic_rk4_step(  # noqa: PLR0913
    x: FloatArray,
    params: JansenRitParams,
    dt: float,
    xi: FloatArray | float,
    coupling: FloatArray | float = 0.0,
    u_tes: FloatArray | float = 0.0,
) -> FloatArray:
    """Advance one stochastic Runge-Kutta 4th-order step with additive noise.

    Parameters
    ----------
    x
        Current state, shape ``(6,)`` or ``(6, N)``.
    params
        Model parameters (``params.sigma`` scales the noise).
    dt
        Integration step in seconds.
    xi
        Standard-normal draw(s) for this step: a scalar for one node, shape
        ``(N,)`` for ``N`` nodes.
    coupling
        Network coupling term, shape ``(N,)`` or scalar.
    u_tes
        tES stimulation vector/scalar entering the pyramidal sigmoid.

    Returns
    -------
    FloatArray
        Next state, same shape as ``x``.
    """
    n_nodes = 1 if x.ndim == 1 else x.shape[1]
    params_tuple = params.to_numba_tuple(n_nodes)

    if n_nodes == 1:
        return _stochastic_rk4_step_jit(x, params_tuple, dt, _to_scalar(xi), _to_scalar(coupling), _to_scalar(u_tes))
    return _stochastic_rk4_step_jit(
        x, params_tuple, dt, _to_array(xi, n_nodes), _to_array(coupling, n_nodes), _to_array(u_tes, n_nodes)
    )


@numba.njit(fastmath=True, cache=True)
def _heun_step_jit(  # noqa: PLR0913
    x: FloatArray,
    params_tuple: tuple[Any, ...],
    dt: float,
    xi: FloatArray | float,
    coupling: FloatArray | float,
    u_tes: FloatArray | float,
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
    params: JansenRitParams,
    dt: float,
    xi: FloatArray | float,
    coupling: FloatArray | float = 0.0,
    u_tes: FloatArray | float = 0.0,
) -> FloatArray:
    """Advance one stochastic-Heun step with additive noise on the ``x5'`` equation.

    Parameters
    ----------
    x
        Current state, shape ``(6,)`` or ``(6, N)``.
    params
        Model parameters (``params.sigma`` scales the noise).
    dt
        Integration step in seconds.
    xi
        Standard-normal draw(s) for this step: a scalar for one node, shape
        ``(N,)`` for ``N`` nodes.
    coupling
        Network coupling term, shape ``(N,)`` or scalar.
    u_tes
        tES stimulation vector/scalar entering the pyramidal sigmoid.

    Returns
    -------
    FloatArray
        Next state, same shape as ``x``.
    """
    n_nodes = 1 if x.ndim == 1 else x.shape[1]
    params_tuple = params.to_numba_tuple(n_nodes)

    if n_nodes == 1:
        return _heun_step_jit(x, params_tuple, dt, _to_scalar(xi), _to_scalar(coupling), _to_scalar(u_tes))
    return _heun_step_jit(
        x, params_tuple, dt, _to_array(xi, n_nodes), _to_array(coupling, n_nodes), _to_array(u_tes, n_nodes)
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


def simulate_node(  # noqa: PLR0913
    *,
    params: JansenRitParams,
    duration: float,
    dt: float = DT_DEFAULT,
    seed: int | None = None,
    deterministic: bool = False,
    use_stochastic_rk4: bool = False,
) -> tuple[FloatArray, FloatArray]:
    """Integrate a single node from rest and return its full trajectory.

    Parameters
    ----------
    params
        Model parameters (``params.A`` selects the regime).
    duration
        Simulated time in seconds.
    dt
        Integration step in seconds; defaults to the locked :data:`DT_DEFAULT`.
    seed
        Seed for the noise stream (ignored when ``deterministic``).
    deterministic
        If ``True``, integrate with RK4 and no noise; otherwise use stochastic Heun
        (or stochastic RK4 if ``use_stochastic_rk4`` is ``True``).
    use_stochastic_rk4
        If ``True`` and ``deterministic`` is ``False``, integrate with stochastic RK4;
        otherwise use stochastic Heun.

    Returns
    -------
    t
        Time vector in seconds, shape ``(n_samples,)``.
    x_traj
        State trajectory, shape ``(6, n_samples)`` ordered ``[x1, ..., x6]``.
    """
    n_steps = round(duration / dt)
    # An isolated node is the degenerate N=1 network: no connectome, no coupling.
    dyn = JansenRitDynamics(
        dt=dt,
        connectome=None,
        K=0.0,
        params=params,
        seed=seed,
        deterministic=deterministic,
        use_stochastic_rk4=use_stochastic_rk4,
        use_delays=False,
    )

    x_traj = np.zeros((6, n_steps + 1), dtype=np.float64)
    x_traj[:, 0] = dyn.x[:, 0]
    for k in range(n_steps):
        dyn.evaluate(k * dt, 0.0)
        x_traj[:, k + 1] = dyn.x[:, 0]

    t = np.arange(n_steps + 1, dtype=np.float64) * dt
    return t, x_traj


def simulate_network(  # noqa: PLR0913
    *,
    params: JansenRitParams,
    connectome: Connectome,
    K: float,  # noqa: N803
    duration: float,
    dt: float = DT_DEFAULT,
    seed: int | None = None,
    deterministic: bool = False,
    use_stochastic_rk4: bool = False,
    use_delays: bool = True,
    initial_state: FloatArray | None = None,
    u_hat_tES: float | FloatArray = 0.0,  # noqa: N803
    stim_window: tuple[float, float] | None = None,
    u_tES: FloatArray | None = None,  # noqa: N803
) -> tuple[FloatArray, FloatArray]:
    """Integrate a coupled network of nodes and return the full trajectory.

    Parameters
    ----------
    params
        Model parameters. If ``params.A`` is an array, it specifies regional gains.
    connectome
        The structural connectome containing weights and conduction delays.
    K
        Global coupling scaling factor.
    duration
        Simulated time in seconds.
    dt
        Integration step in seconds; defaults to ``DT_DEFAULT``.
    seed
        Seed for the noise stream (ignored when ``deterministic``).
    deterministic
        If ``True``, integrate with RK4 and no noise; otherwise use stochastic Heun
        (or stochastic RK4 if ``use_stochastic_rk4`` is ``True``).
    use_stochastic_rk4
        If ``True`` and ``deterministic`` is ``False``, integrate with stochastic RK4;
        otherwise use stochastic Heun.
    use_delays
        If ``True``, use delayed coupling from tract lengths. If ``False``, use
        instantaneous coupling.
    initial_state
        Initial state array of shape ``(6, N)``. If ``None``, defaults to zeros.
    u_hat_tES
        Constant tES current applied during ``stim_window``. A scalar (shared by
        every electrode) or shape ``(n_electrodes,)`` for per-electrode currents.
    stim_window
        Time window (start_s, end_s) during which tES stimulation is active.
    u_tES
        Pre-computed time-varying tES current. Shape ``(n_steps + 1,)`` for a
        single shared envelope, or ``(n_steps + 1, n_electrodes)`` per electrode.
        If provided, overrides ``u_hat_tES`` and ``stim_window``. The per-node
        stimulus is ``u_vec @ gamma`` with ``gamma`` of shape ``(n_electrodes, 76)``.

    Returns
    -------
    t
        Time vector in seconds, shape ``(n_samples,)``.
    x_traj
        State trajectory, shape ``(6, N, n_samples)``.
    """
    n_steps = round(duration / dt)
    n_nodes = connectome.weights.shape[0]

    if u_tES is not None and u_tES.shape[0] < n_steps:
        msg = f"u_tES array length must be at least {n_steps}, got {u_tES.shape[0]}"
        raise ValueError(msg)

    dyn = JansenRitDynamics(
        dt=dt,
        connectome=connectome,
        K=K,
        params=params,
        seed=seed,
        deterministic=deterministic,
        use_stochastic_rk4=use_stochastic_rk4,
        use_delays=use_delays,
        initial_state=initial_state,
    )

    # Per-electrode current schedule; the gamma projection lives in the component.
    n_elec = dyn.n_elec
    u_amp = np.atleast_1d(np.asarray(u_hat_tES, dtype=np.float64))
    if u_amp.shape[0] == 1:
        u_amp = np.broadcast_to(u_amp, (n_elec,))
    elif u_amp.shape[0] != n_elec:
        msg = f"u_hat_tES has {u_amp.shape[0]} electrodes but gamma has {n_elec}"
        raise ValueError(msg)
    if u_tES is not None and u_tES.ndim > 1 and u_tES.shape[1] != n_elec:
        msg = f"u_tES has {u_tES.shape[1]} electrodes but gamma has {n_elec}"
        raise ValueError(msg)

    x_traj = np.zeros((6, n_nodes, n_steps + 1), dtype=np.float64)
    x_traj[:, :, 0] = dyn.x

    for k in range(n_steps):
        if u_tES is not None:
            u_step = u_tES[k]
            u_vec: float | FloatArray = np.full(n_elec, u_step) if np.ndim(u_step) == 0 else u_step
        elif stim_window is not None and stim_window[0] <= k * dt < stim_window[1]:
            u_vec = u_amp
        else:
            u_vec = 0.0

        dyn.evaluate(k * dt, u_vec)
        x_traj[:, :, k + 1] = dyn.x

    t = np.arange(n_steps + 1, dtype=np.float64) * dt
    return t, x_traj


def output(x_traj: FloatArray) -> FloatArray:
    """Observed output ``y = x2 - x3`` from a state trajectory or state."""
    return x_traj[1] - x_traj[2]


class JansenRitDynamicsLog(BaseModel):
    """Pydantic log snapshot of the Jansen-Rit network state."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    x: np.ndarray


class JansenRitDynamics(Dynamics[JansenRitDynamicsLog]):
    """Whole-brain Jansen-Rit network as a ``simulate`` :class:`Dynamics` plant.

    Wraps the standalone :func:`simulate_network` integration as a stateful,
    single-step component for the :class:`~simulate.simulation.Simulation`
    orchestrator. It owns its own stepping (``integrator=None``, so the base
    :meth:`~simulate.dynamics.Dynamics.update` treats :meth:`dynamics` as a discrete
    ``x_next`` transition) and reuses :func:`heun_step` so the stochastic ``x5`` noise
    and delayed network coupling are preserved.

    The state ``self.x`` is the ``(6, N)`` network array; the orchestrator logs it as
    the flattened ``(6 N,)`` vector that :meth:`~simulate.component.Component.from_col_vec`
    produces. The control input ``u`` is the per-electrode tES current, projected to
    nodes through ``connectome.gamma`` (no-op when ``gamma`` is unset, i.e. open loop).

    Passing ``connectome=None`` yields the degenerate ``N=1`` isolated node (no coupling,
    no stimulation) that backs :func:`simulate_node`.

    Everything is in SI seconds, matching :func:`simulate_network`: ``dt`` is the
    integration step in seconds and ``connectome.delays`` (ms) convert to steps via
    ``round(delays / (dt * 1000))``.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        dt: float,
        connectome: Connectome | None = None,
        K: float = 0.0,  # noqa: N803
        params: JansenRitParams | None = None,
        seed: int | None = None,
        deterministic: bool = False,
        use_stochastic_rk4: bool = False,
        use_delays: bool = True,
        initial_state: FloatArray | None = None,
    ) -> None:
        """Initialize the network plant from a connectome and coupling ``K``."""
        super().__init__(dt, integrator=None)
        params = params or JansenRitParams()
        self.K = K
        self.deterministic = deterministic
        self.use_stochastic_rk4 = use_stochastic_rk4

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

        if use_delays:
            # Delays in connectome are in ms. Convert to steps: round(delays / (dt * 1000.0)).
            delay_steps = np.round(delays / (dt * 1000.0)).astype(np.int64)
        else:
            delay_steps = np.zeros((n_nodes, n_nodes), dtype=np.int64)
        self.delay_steps = delay_steps
        self.max_history_len = int(np.max(delay_steps)) + 1

        # Circular history buffer of S(y), seeded from the initial state.
        self.history = np.zeros((self.max_history_len, n_nodes), dtype=np.float64)
        self.history[:, :] = sigmoid_jit(self.x[1] - self.x[2], params.e0, params.v0, params.r)
        self.col_indices = np.arange(n_nodes)[np.newaxis, :]
        self.w_weights = weights

        self.rng = np.random.default_rng(seed)
        self._k = 0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate from a raw config dict, loading the TVB connectome by ``speed``."""
        from neuro.connectome import compute_gamma, load_connectome  # noqa: PLC0415

        connectome = load_connectome(speed=float(config.get("speed", 50.0)))

        target_electrode = config.get("target_electrode")
        if target_electrode is not None:
            gamma = compute_gamma(
                connectome.centres,
                target_electrode=target_electrode,
                sigma=config.get("sigma", 20.0),
            )
            connectome = replace(connectome, gamma=gamma)

        a_cfg = config.get("A", 3.25)
        a_val = np.asarray(a_cfg, dtype=np.float64) if isinstance(a_cfg, (list, tuple)) else float(a_cfg)
        params = JansenRitParams(A=a_val)

        return cls(
            dt=float(config["dt"]),
            connectome=connectome,
            K=float(config["K"]),
            params=params,
            seed=config.get("seed"),
            deterministic=bool(config.get("deterministic", False)),
            use_stochastic_rk4=bool(config.get("use_stochastic_rk4", False)),
            use_delays=bool(config.get("use_delays", True)),
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

    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """Advance the network one step, mirroring the :func:`simulate_network` loop body."""
        n_nodes = x.shape[1]
        params_tuple = self.params_tuple

        # 1. Update history and compute delayed coupling.
        s_y = sigmoid_jit(x[1] - x[2], self.net_params.e0, self.net_params.v0, self.net_params.r)
        coupling = _dynamics_history_coupling_jit(
            self.history,
            self._k,
            self.max_history_len,
            self.delay_steps,
            self.w_weights,
            self.K,
            s_y,
        )

        # 2. Project the tES control onto nodes.
        u_node = self._project_control(u)

        # 3. Integrate one step.
        if self.deterministic:
            if n_nodes == 1:
                x_next = _rk4_step_jit(x, params_tuple, self.dt, _to_scalar(coupling), _to_scalar(u_node))
            else:
                x_next = _rk4_step_jit(
                    x, params_tuple, self.dt, _to_array(coupling, n_nodes), _to_array(u_node, n_nodes)
                )
        else:
            xi = self.rng.standard_normal(n_nodes)
            if self.use_stochastic_rk4:
                if n_nodes == 1:
                    x_next = _stochastic_rk4_step_jit(
                        x, params_tuple, self.dt, _to_scalar(xi), _to_scalar(coupling), _to_scalar(u_node)
                    )
                else:
                    x_next = _stochastic_rk4_step_jit(
                        x,
                        params_tuple,
                        self.dt,
                        _to_array(xi, n_nodes),
                        _to_array(coupling, n_nodes),
                        _to_array(u_node, n_nodes),
                    )
            elif n_nodes == 1:
                x_next = _heun_step_jit(
                    x, params_tuple, self.dt, _to_scalar(xi), _to_scalar(coupling), _to_scalar(u_node)
                )
            else:
                x_next = _heun_step_jit(
                    x,
                    params_tuple,
                    self.dt,
                    _to_array(xi, n_nodes),
                    _to_array(coupling, n_nodes),
                    _to_array(u_node, n_nodes),
                )

        self._k += 1
        return x_next

    def _make_log(self) -> JansenRitDynamicsLog:
        """Build a snapshot log of the current network state."""
        return JansenRitDynamicsLog(x=self.x.copy())
