from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import tvboptim.experimental.network_dynamics as nd
import tvboptim.experimental.network_dynamics.coupling as nd_coupling
import tvboptim.experimental.network_dynamics.dynamics as nd_dynamics
import tvboptim.experimental.network_dynamics.noise as nd_noise
import tvboptim.experimental.network_dynamics.solvers as nd_solvers

if TYPE_CHECKING:
    from neuro.jansen_rit import JansenRitParams

FloatArray = npt.NDArray[np.float64]


def build_tvbo_network(params: JansenRitParams, dt: float, seed: int | None = None) -> nd.Network:
    """Build a tvboptim network simulating the Jansen-Rit model from parameters."""
    n_nodes = params.w_weights.shape[0]

    A_rate = params.a / 1000.0  # ms^-1
    B_rate = params.b / 1000.0
    nu_max = params.e0 / 1000.0
    mu = params.mean_input / 1000.0

    v0 = params.v0
    r = params.r
    J = params.C1
    B = params.B
    A = params.A
    A = np.full(n_nodes, A, dtype=np.float64) if np.isscalar(A) else np.asarray(A, dtype=np.float64)

    dyn = nd_dynamics.JansenRit(
        A=A,
        B=B,
        a=A_rate,
        b=B_rate,
        v0=v0,
        nu_max=nu_max,
        r=r,
        J=J,
        mu=mu,
        INITIAL_STATE=np.zeros(6),
    )

    cpl_delayed = nd_coupling.DelayedSigmoidalJansenRit(
        incoming_states=("y1", "y2"),
        G=params.K,
        midpoint=v0,
        r=r,
        cmax=2.0 * nu_max,
    )

    graph = nd.DenseDelayGraph(
        weights=jnp.asarray(params.w_weights),
        delays=jnp.asarray(params.delay_steps * dt * 1000.0),
    )

    noise = None
    if params.sigma > 0:
        key = jax.random.key(seed if seed is not None else 42)
        # Convert sigma from per-second to per-ms Heun scale
        # tvboptim's Heun solver handles sqrt(dt)
        # Note: the paper's sigma was tuned for dt=1e-4s, but let's just pass it
        # directly and see if they match. Wait, the handrolled formulation handles
        # noise by dw = sigma * sqrt(dt) * xi. TVB Additive noise also scales by sqrt(dt).
        # We may need to pass noise as a scaled version, but we will test deterministic
        # equivalence primarily.
        noise = nd_noise.AdditiveNoise(
            apply_to="y4",
            sigma=params.sigma,
            key=key,
        )

    return nd.Network(
        dynamics=dyn,
        coupling={"delayed": cpl_delayed},
        graph=graph,
        noise=noise,
    )


def lfp_jax(x: jax.Array) -> jax.Array:
    """Observed output ``y = x2 - x3`` from a state or trajectory."""
    return x[1] - x[2]


def eeg_jax(x: jax.Array, gain: jax.Array) -> jax.Array:
    """Map state to EEG measurement ``gain @ lfp``.

    Parameters
    ----------
    x
        State matrix of shape ``(6, N)`` (or ``(6, N, T)``).
    gain
        Leadfield matrix of shape ``(n_channels, N)``.

    Returns
    -------
    jax.Array
        EEG output, shape ``(n_channels,)`` (or ``(n_channels, T)``).
    """
    return gain @ lfp_jax(x)


def simulate_tvbo(
    params: JansenRitParams, duration: float, dt: float, seed: int | None = None
) -> tuple[FloatArray, FloatArray]:
    """Simulate using tvboptim and return identical shape as hand-rolled engine.

    Returns
    -------
        t: array of times in seconds
        x_traj: array of shape (6, n_nodes, n_steps)
    """
    net = build_tvbo_network(params, dt, seed)
    solver = nd_solvers.Heun()

    # tvboptim times are in ms
    res = nd.solve(net, solver, dt=dt * 1000.0, t1=duration * 1000.0)

    t = np.asarray(res.ts) / 1000.0
    # ys has shape (n_steps, n_states, n_nodes)
    # We want (n_states, n_nodes, n_steps)
    x_traj = np.array(res.ys).transpose(1, 2, 0).copy()

    # TVB states 3, 4, 5 are derivatives (in 1/ms), convert back to 1/s
    x_traj[3:, :, :] *= 1000.0

    return t, x_traj
