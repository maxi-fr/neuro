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
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

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


def sigmoid(v: FloatArray | float, params: JansenRitParams) -> FloatArray:
    """Evaluate the firing-rate sigmoid ``S(v) = 2 e0 / (1 + exp(r (v0 - v)))``."""
    return 2.0 * params.e0 / (1.0 + np.exp(params.r * (params.v0 - v)))


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
    x1, x2, x3, x4, x5, x6 = x
    a, b = params.a, params.b

    out = sigmoid(x2 - x3 + u_tes, params)  # pyramidal output rate; + U_tES slot (Stage 3)
    exc = sigmoid(params.C1 * x1, params)  # drive to excitatory interneurons
    inh = sigmoid(params.C3 * x1, params)  # drive to inhibitory interneurons

    dx1 = x4
    dx4 = params.A * a * out - 2.0 * a * x4 - a * a * x1
    dx2 = x5
    # Stage 2 adds + A a K sum_j w_ij S(y_j) - step-wise constant coupling; zeta is injected in heun_step.
    dx5 = params.A * a * (params.mean_input + params.C2 * exc + coupling) - 2.0 * a * x5 - a * a * x2
    dx3 = x6
    dx6 = params.B * b * params.C4 * inh - 2.0 * b * x6 - b * b * x3

    return np.array([dx1, dx2, dx3, dx4, dx5, dx6])


def rk4_step(
    x: FloatArray,
    params: JansenRitParams,
    dt: float,
    coupling: FloatArray | float = 0.0,
    u_tes: FloatArray | float = 0.0,
) -> FloatArray:
    """Advance one deterministic RK4 step (noise off)."""
    k1 = jr_rhs(x, params, coupling, u_tes)
    k2 = jr_rhs(x + 0.5 * dt * k1, params, coupling, u_tes)
    k3 = jr_rhs(x + 0.5 * dt * k2, params, coupling, u_tes)
    k4 = jr_rhs(x + dt * k3, params, coupling, u_tes)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


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
    dw = np.zeros_like(x, dtype=np.float64)
    dw[4] = params.sigma * np.sqrt(dt) * xi  # additive noise enters x5 only

    k1 = jr_rhs(x, params, coupling, u_tes) * dt + dw
    k2 = jr_rhs(x + 0.5 * k1, params, coupling, u_tes) * dt + dw
    k3 = jr_rhs(x + 0.5 * k2, params, coupling, u_tes) * dt + dw
    k4 = jr_rhs(x + k3, params, coupling, u_tes) * dt + dw

    return x + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


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
    dw = np.zeros_like(x, dtype=np.float64)
    dw[4] = params.sigma * np.sqrt(dt) * xi  # additive noise enters x5 only

    f0 = jr_rhs(x, params, coupling, u_tes)
    x_pred = x + dt * f0 + dw
    f1 = jr_rhs(x_pred, params, coupling, u_tes)
    return x + 0.5 * dt * (f0 + f1) + dw


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
    x_traj = np.zeros((6, n_steps + 1), dtype=np.float64)
    rng = np.random.default_rng(seed)

    x = x_traj[:, 0]
    for k in range(n_steps):
        if deterministic:
            x = rk4_step(x, params, dt)
        elif use_stochastic_rk4:
            x = stochastic_rk4_step(x, params, dt, rng.standard_normal())
        else:
            x = heun_step(x, params, dt, rng.standard_normal())
        x_traj[:, k + 1] = x

    t = np.arange(n_steps + 1, dtype=np.float64) * dt
    return t, x_traj


def simulate_network(  # noqa: PLR0913, C901, PLR0912, PLR0915
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

    # tES steering matrix gamma_2d of shape (n_elec, n_nodes); single electrode is n_elec=1.
    gamma_2d = None if connectome.gamma is None else np.atleast_2d(connectome.gamma)
    n_elec = 1 if gamma_2d is None else gamma_2d.shape[0]
    u_amp = np.atleast_1d(np.asarray(u_hat_tES, dtype=np.float64))
    if u_amp.shape[0] == 1:
        u_amp = np.broadcast_to(u_amp, (n_elec,))
    elif u_amp.shape[0] != n_elec:
        msg = f"u_hat_tES has {u_amp.shape[0]} electrodes but gamma has {n_elec}"
        raise ValueError(msg)
    if u_tES is not None and u_tES.ndim > 1 and u_tES.shape[1] != n_elec:
        msg = f"u_tES has {u_tES.shape[1]} electrodes but gamma has {n_elec}"
        raise ValueError(msg)

    a_vec = params.A
    if np.isscalar(a_vec):
        a_vec = np.full(n_nodes, a_vec, dtype=np.float64)
    net_params = replace(params, A=a_vec)

    if initial_state is not None:
        if initial_state.shape != (6, n_nodes):
            msg = f"initial_state must have shape (6, {n_nodes}), got {initial_state.shape}"
            raise ValueError(msg)
        x = initial_state.copy()
    else:
        x = np.zeros((6, n_nodes), dtype=np.float64)

    x_traj = np.zeros((6, n_nodes, n_steps + 1), dtype=np.float64)
    x_traj[:, :, 0] = x

    if use_delays:
        # Delays in connectome are in ms. Convert to steps: round(delays / (dt * 1000.0))
        delay_steps = np.round(connectome.delays / (dt * 1000.0)).astype(np.int64)
    else:
        delay_steps = np.zeros((n_nodes, n_nodes), dtype=np.int64)

    max_delay_steps = int(np.max(delay_steps))
    max_history_len = max_delay_steps + 1

    # History buffer of S(y) of shape (max_history_len, N)
    history = np.zeros((max_history_len, n_nodes), dtype=np.float64)
    # Initialize history by repeating the initial state's S(y)
    s_y_init = sigmoid(x[1] - x[2], net_params)
    history[:, :] = s_y_init

    rng = np.random.default_rng(seed)
    w_weights = connectome.weights

    col_indices = np.arange(n_nodes)[np.newaxis, :]

    for k in range(n_steps):
        # 1. Update history with the current state's S(y)
        s_y = sigmoid(x[1] - x[2], net_params)
        history[k % max_history_len, :] = s_y

        # 2. Compute coupling
        row_indices = (k - delay_steps) % max_history_len
        s_y_delayed = history[row_indices, col_indices]
        coupling = K * np.sum(w_weights * s_y_delayed, axis=1)

        # 3. Compute tES stimulation: per-electrode current u_vec, superposed via u_vec @ gamma.
        if u_tES is not None:
            u_step = u_tES[k]
            u_vec = np.full(n_elec, u_step) if np.ndim(u_step) == 0 else u_step
        elif stim_window is not None and stim_window[0] <= k * dt < stim_window[1]:
            u_vec = u_amp
        else:
            u_vec = None

        if u_vec is not None and np.any(u_vec != 0.0):
            if gamma_2d is None:
                msg = "tES stimulation is active but connectome.gamma is not configured."
                raise ValueError(msg)
            u_node = u_vec @ gamma_2d
        else:
            u_node = 0.0

        # 4. Integrate
        if deterministic:
            x = rk4_step(x, net_params, dt, coupling=coupling, u_tes=u_node)
        else:
            xi = rng.standard_normal(n_nodes)
            if use_stochastic_rk4:
                x = stochastic_rk4_step(x, net_params, dt, xi, coupling=coupling, u_tes=u_node)
            else:
                x = heun_step(x, net_params, dt, xi, coupling=coupling, u_tes=u_node)

        x_traj[:, :, k + 1] = x

    t = np.arange(n_steps + 1, dtype=np.float64) * dt
    return t, x_traj


def output(x_traj: FloatArray) -> FloatArray:
    """Observed output ``y = x2 - x3`` from a state trajectory or state."""
    return x_traj[1] - x_traj[2]
