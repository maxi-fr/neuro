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

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

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
        3.6 = limit cycle).
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

    A: float = 3.25
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


def jr_rhs(x: FloatArray, params: JansenRitParams) -> FloatArray:
    """Right-hand side of the single-node Jansen-Rit ODEs (no noise, no tES).

    Parameters
    ----------
    x
        State, shape ``(6,)`` for one node or ``(6, N)`` for ``N`` nodes, ordered
        ``[x1, x2, x3, x4, x5, x6]``.
    params
        Model parameters.

    Returns
    -------
    FloatArray
        Time derivative ``x'`` with the same shape as ``x``.
    """
    x1, x2, x3, x4, x5, x6 = x
    a, b = params.a, params.b

    out = sigmoid(x2 - x3, params)  # pyramidal output rate; + U_tES slot (Stage 3)
    exc = sigmoid(params.C1 * x1, params)  # drive to excitatory interneurons
    inh = sigmoid(params.C3 * x1, params)  # drive to inhibitory interneurons

    dx1 = x4
    dx4 = params.A * a * out - 2.0 * a * x4 - a * a * x1
    dx2 = x5
    # Stage 2 adds + A a K sum_j w_ij S(x2-x3); zeta is injected in heun_step.
    dx5 = params.A * a * (params.mean_input + params.C2 * exc) - 2.0 * a * x5 - a * a * x2
    dx3 = x6
    dx6 = params.B * b * params.C4 * inh - 2.0 * b * x6 - b * b * x3

    return np.array([dx1, dx2, dx3, dx4, dx5, dx6])


def rk4_step(x: FloatArray, params: JansenRitParams, dt: float) -> FloatArray:
    """Advance one deterministic RK4 step (noise off)."""
    k1 = jr_rhs(x, params)
    k2 = jr_rhs(x + 0.5 * dt * k1, params)
    k3 = jr_rhs(x + 0.5 * dt * k2, params)
    k4 = jr_rhs(x + dt * k3, params)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def stochastic_rk4_step(x: FloatArray, params: JansenRitParams, dt: float, xi: FloatArray | float) -> FloatArray:
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

    Returns
    -------
    FloatArray
        Next state, same shape as ``x``.
    """
    dw = np.zeros_like(x, dtype=np.float64)
    dw[4] = params.sigma * np.sqrt(dt) * xi  # additive noise enters x5 only

    k1 = jr_rhs(x, params) * dt + dw
    k2 = jr_rhs(x + 0.5 * k1, params) * dt + dw
    k3 = jr_rhs(x + 0.5 * k2, params) * dt + dw
    k4 = jr_rhs(x + k3, params) * dt + dw

    return x + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def heun_step(x: FloatArray, params: JansenRitParams, dt: float, xi: FloatArray | float) -> FloatArray:
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

    Returns
    -------
    FloatArray
        Next state, same shape as ``x``.
    """
    dw = np.zeros_like(x, dtype=np.float64)
    dw[4] = params.sigma * np.sqrt(dt) * xi  # additive noise enters x5 only

    f0 = jr_rhs(x, params)
    x_pred = x + dt * f0 + dw
    f1 = jr_rhs(x_pred, params)
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


def output(x_traj: FloatArray) -> FloatArray:
    """Observed output ``y = x2 - x3`` from a state trajectory or state."""
    return x_traj[1] - x_traj[2]
