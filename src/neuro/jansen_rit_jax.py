"""JAX implementation of the Jansen-Rit neural mass model.

Pure, ``jax.lax.scan``-based functions so the same code serves three purposes:

* **Plant simulation** -- :func:`simulate_jax` integrates a (possibly noisy) network
  trajectory, JIT-compiled and vectorised over nodes; :func:`step_jax` is the same
  single-step update exposed standalone for the stateful
  :class:`~neuro.jansen_rit.JansenRitDynamics` plant.
* **System identification** -- :func:`rollout_jax` is a deterministic, differentiable
  forward model, so ``jax.grad`` / ``jax.jacobian`` flow through it w.r.t. the
  continuous parameters.

State ``x = [x1..x6]`` is laid out as ``(6, N)`` (rows are the paper's 1-6 indexing),
observed output ``y = x2 - x3``, stochastic Heun with additive noise on the ``x5'``
equation, and delayed network coupling through a circular history buffer of ``S(y)``.

.. important::
   JAX defaults to ``float32``, which silently breaks parity with the float64 NumPy
   reference (errors collapse from ~1e-12 to ~1e-5). Enable 64-bit mode **before**
   creating any array -- call :func:`enable_x64` (or set ``jax_enable_x64``) at the
   start of your program. This module deliberately does not flip the global flag on
   import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from neuro.jansen_rit import JansenRitParams

FloatArray = npt.NDArray[np.float64]
JaxArray = jax.Array


def enable_x64() -> None:
    """Enable JAX 64-bit precision for float64 parity with the NumPy reference.

    Must be called before any JAX array is created. Equivalent to
    ``jax.config.update("jax_enable_x64", True)``.
    """
    jax.config.update("jax_enable_x64", True)  # noqa: FBT003


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class JRParamsJax:
    """Jansen-Rit parameters as a JAX pytree.

    The continuous physical parameters are pytree leaves, so ``jax.grad`` /
    ``jax.jacobian`` differentiate through them. The structural fields
    (``delay_steps``, ``n_nodes``, ``max_history_len``) are static auxiliary data:
    they fix array shapes and gather indices and must be Python-level constants at
    trace time.

    Attributes
    ----------
    A
        Excitatory synaptic gain, shape ``(N,)``.
    B, a, b, C1, C2, C3, C4, e0, v0, r
        Scalar Jansen-Rit parameters (0-d arrays).
    mean_input
        Constant excitatory input ``I``, shape ``(N,)``.
    sigma
        Noise std for the additive white noise on ``x5'`` (scalar).
    K
        Global coupling scaling factor (scalar).
    w_weights
        Connectivity weight matrix, shape ``(N, N)``.
    eeg_gain
        Leadfield matrix, shape ``(n_channels, N)``.
    gamma
        tES steering matrix, shape ``(n_elec, N)``.
    delay_steps
        Integer delay matrix, shape ``(N, N)`` -- static aux.
    n_nodes
        Number of network nodes ``N`` -- static aux.
    max_history_len
        Circular-buffer length ``max(delay_steps) + 1`` -- static aux.
    """

    A: JaxArray
    B: JaxArray
    a: JaxArray
    b: JaxArray
    C1: JaxArray
    C2: JaxArray
    C3: JaxArray
    C4: JaxArray
    e0: JaxArray
    v0: JaxArray
    r: JaxArray
    mean_input: JaxArray
    sigma: JaxArray
    K: JaxArray
    w_weights: JaxArray
    eeg_gain: JaxArray
    gamma: JaxArray
    delay_steps: npt.NDArray[np.int64]
    n_nodes: int
    max_history_len: int

    def tree_flatten(self) -> tuple[tuple[JaxArray, ...], tuple[tuple[tuple[int, ...], ...], int, int]]:
        """Split into differentiable leaves and hashable static aux data."""
        children = (
            self.A,
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
            self.K,
            self.w_weights,
            self.eeg_gain,
            self.gamma,
        )
        # delay_steps is flattened to nested tuples so the aux is hashable for jit.
        aux = (tuple(map(tuple, self.delay_steps.tolist())), self.n_nodes, self.max_history_len)
        return children, aux

    @classmethod
    def tree_unflatten(
        cls, aux_data: tuple[tuple[tuple[int, ...], ...], int, int], children: tuple[JaxArray, ...]
    ) -> JRParamsJax:
        """Rebuild from the static aux data and the leaf children."""
        delay_tuple, n_nodes, max_history_len = aux_data
        (A, B, a, b, C1, C2, C3, C4, e0, v0, r, mean_input, sigma, K, w_weights, eeg_gain, gamma) = children
        delay_steps = np.asarray(delay_tuple, dtype=np.int64).reshape(n_nodes, n_nodes)
        return cls(
            A=A,
            B=B,
            a=a,
            b=b,
            C1=C1,
            C2=C2,
            C3=C3,
            C4=C4,
            e0=e0,
            v0=v0,
            r=r,
            mean_input=mean_input,
            sigma=sigma,
            K=K,
            w_weights=w_weights,
            eeg_gain=eeg_gain,
            gamma=gamma,
            delay_steps=delay_steps,
            n_nodes=n_nodes,
            max_history_len=max_history_len,
        )


def from_jansen_rit_params(params: JansenRitParams, n_nodes: int) -> JRParamsJax:
    """Convert a :class:`~neuro.jansen_rit.JansenRitParams` into a :class:`JRParamsJax`.

    ``A`` and ``mean_input`` are broadcast to shape ``(N,)`` and ``gamma`` to 2-D, so
    the JAX code carries no scalar/array branching. All leaf arrays are coerced to
    ``float64`` (request 64-bit mode first; see :func:`enable_x64`).

    Parameters
    ----------
    params
        Source parameters; its arrays must be consistent with ``n_nodes``.
    n_nodes
        Number of network nodes ``N``.

    Returns
    -------
    JRParamsJax
        Pytree-typed parameters ready for the JAX dynamics.
    """
    delay_steps = np.asarray(params.delay_steps, dtype=np.int64).reshape(n_nodes, n_nodes)
    return JRParamsJax(
        A=jnp.broadcast_to(jnp.asarray(params.A, dtype=jnp.float64), (n_nodes,)),
        B=jnp.asarray(params.B, dtype=jnp.float64),
        a=jnp.asarray(params.a, dtype=jnp.float64),
        b=jnp.asarray(params.b, dtype=jnp.float64),
        C1=jnp.asarray(params.C1, dtype=jnp.float64),
        C2=jnp.asarray(params.C2, dtype=jnp.float64),
        C3=jnp.asarray(params.C3, dtype=jnp.float64),
        C4=jnp.asarray(params.C4, dtype=jnp.float64),
        e0=jnp.asarray(params.e0, dtype=jnp.float64),
        v0=jnp.asarray(params.v0, dtype=jnp.float64),
        r=jnp.asarray(params.r, dtype=jnp.float64),
        mean_input=jnp.broadcast_to(jnp.asarray(params.mean_input, dtype=jnp.float64), (n_nodes,)),
        sigma=jnp.asarray(params.sigma, dtype=jnp.float64),
        K=jnp.asarray(params.K, dtype=jnp.float64),
        w_weights=jnp.asarray(params.w_weights, dtype=jnp.float64),
        eeg_gain=jnp.asarray(params.eeg_gain, dtype=jnp.float64),
        gamma=jnp.asarray(np.atleast_2d(params.gamma), dtype=jnp.float64),
        delay_steps=delay_steps,
        n_nodes=n_nodes,
        max_history_len=int(delay_steps.max()) + 1,
    )


def sigmoid_jax(v: JaxArray | float, e0: JaxArray | float, v0: JaxArray | float, r: JaxArray | float) -> JaxArray:
    """Evaluate the firing-rate sigmoid ``S(v) = 2 e0 / (1 + exp(r (v0 - v)))``.

    Parameters
    ----------
    v
        Average membrane potential.
    e0, v0, r
        Sigmoid ceiling-half, midpoint, and steepness.

    Returns
    -------
    JaxArray
        Average firing rate, same shape as ``v``.
    """
    return 2.0 * e0 / (1.0 + jnp.exp(r * (v0 - v)))


def rhs_jax(x: JaxArray, p: JRParamsJax, coupling: JaxArray | float, u_tes: JaxArray | float) -> JaxArray:
    """Continuous-time right-hand side of the Jansen-Rit network.

    Parameters
    ----------
    x
        State matrix of shape ``(6, N)`` (rows ``x1..x6``).
    p
        Network parameters.
    coupling
        Delayed network coupling, shape ``(N,)`` or scalar.
    u_tes
        tES stimulation entering the pyramidal sigmoid, shape ``(N,)`` or scalar.

    Returns
    -------
    JaxArray
        State derivatives of shape ``(6, N)``.
    """
    x1, x2, x3, x4, x5, x6 = x

    out = sigmoid_jax(x2 - x3 + u_tes, p.e0, p.v0, p.r)  # pyramidal output + tES
    exc = sigmoid_jax(p.C1 * x1, p.e0, p.v0, p.r)  # drive to excitatory interneurons
    inh = sigmoid_jax(p.C3 * x1, p.e0, p.v0, p.r)  # drive to inhibitory interneurons

    dx1 = x4
    dx4 = p.A * p.a * out - 2.0 * p.a * x4 - p.a**2 * x1
    dx2 = x5
    dx5 = p.A * p.a * (p.mean_input + p.C2 * exc + coupling) - 2.0 * p.a * x5 - p.a**2 * x2
    dx3 = x6
    dx6 = p.B * p.b * p.C4 * inh - 2.0 * p.b * x6 - p.b**2 * x3

    return jnp.stack([dx1, dx2, dx3, dx4, dx5, dx6])


def lfp_jax(x: JaxArray) -> JaxArray:
    """Observed output ``y = x2 - x3`` from a state or trajectory."""
    return x[1] - x[2]


def eeg_jax(x: JaxArray, gain: JaxArray) -> JaxArray:
    """Map state to EEG measurement ``gain @ lfp``.

    Parameters
    ----------
    x
        State matrix of shape ``(6, N)`` (or ``(6, N, T)``).
    gain
        Leadfield matrix of shape ``(n_channels, N)``.

    Returns
    -------
    JaxArray
        EEG output, shape ``(n_channels,)`` (or ``(n_channels, T)``).
    """
    return gain @ lfp_jax(x)


def project_control_jax(u: JaxArray, gamma_2d: JaxArray) -> JaxArray:
    """Project per-electrode tES current onto nodes via ``u @ gamma_2d``.

    Unlike the NumPy/CasADi versions there is no all-zero short-circuit (a
    data-dependent branch that is neither jit-able nor needed: ``0 @ gamma == 0`` and
    the result stays differentiable w.r.t. ``gamma``).

    Parameters
    ----------
    u
        Per-electrode control, shape ``(n_elec,)``.
    gamma_2d
        Steering matrix, shape ``(n_elec, N)``.

    Returns
    -------
    JaxArray
        Node-level stimulation, shape ``(N,)``.
    """
    return u @ gamma_2d


def heun_step_det_jax(
    x: JaxArray, u_tes: JaxArray | float, coupling: JaxArray | float, p: JRParamsJax, dt: float
) -> JaxArray:
    """Advance one deterministic Heun step (no noise).

    Matches the CasADi step and the NumPy reference with ``sigma = 0``.

    Parameters
    ----------
    x
        State matrix of shape ``(6, N)``.
    u_tes
        tES stimulation, shape ``(N,)`` or scalar.
    coupling
        Delayed network coupling, shape ``(N,)`` or scalar.
    p
        Network parameters.
    dt
        Integration step in seconds.

    Returns
    -------
    JaxArray
        Next state of shape ``(6, N)``.
    """
    f0 = rhs_jax(x, p, coupling, u_tes)
    x_pred = x + dt * f0
    f1 = rhs_jax(x_pred, p, coupling, u_tes)
    return x + 0.5 * dt * (f0 + f1)


def heun_step_stoch_jax(  # noqa: PLR0913
    x: JaxArray,
    u_tes: JaxArray | float,
    coupling: JaxArray | float,
    p: JRParamsJax,
    dt: float,
    xi: JaxArray | float,
) -> JaxArray:
    """Advance one stochastic Heun step with additive noise on the ``x5'`` equation.

    The noise increment ``sigma * sqrt(dt) * xi`` enters row 4 only and is added to
    both the predictor and the final state.

    Parameters
    ----------
    x
        State matrix of shape ``(6, N)``.
    u_tes
        tES stimulation, shape ``(N,)`` or scalar.
    coupling
        Delayed network coupling, shape ``(N,)`` or scalar.
    p
        Network parameters (``p.sigma`` scales the noise).
    dt
        Integration step in seconds.
    xi
        Standard-normal draw(s) for this step, shape ``(N,)``.

    Returns
    -------
    JaxArray
        Next state of shape ``(6, N)``.
    """
    dw = jnp.zeros_like(x).at[4].set(p.sigma * jnp.sqrt(dt) * xi)
    f0 = rhs_jax(x, p, coupling, u_tes)
    x_pred = x + dt * f0 + dw
    f1 = rhs_jax(x_pred, p, coupling, u_tes)
    return x + 0.5 * dt * (f0 + f1) + dw


def coupling_from_history_jax(history: JaxArray, k: JaxArray | int, p: JRParamsJax) -> JaxArray:
    """Delayed network coupling from a circular history buffer of ``S(y)``.

    Vectorised form of :func:`neuro.jansen_rit._dynamics_history_coupling_jit`:
    ``coupling[i] = K * sum_j w[i, j] * history[(k - delay[i, j]) % L, j]``.

    Parameters
    ----------
    history
        Circular buffer of ``S(y)`` values, shape ``(max_history_len, N)``.
    k
        Current absolute step index.
    p
        Network parameters (uses ``w_weights``, ``K``, ``delay_steps``).

    Returns
    -------
    JaxArray
        Coupling vector of shape ``(N,)``.
    """
    length = p.max_history_len
    delays = jnp.asarray(p.delay_steps)
    rows = (k - delays) % length  # (N, N) read indices
    cols = jnp.arange(p.n_nodes)[None, :]  # (1, N) -> broadcasts to (N, N)
    s_past = history[rows, cols]  # (N, N): s_past[i, j] = history[(k - delay[i, j]) % L, j]
    return p.K * jnp.sum(p.w_weights * s_past, axis=1)


def seed_history(x0: JaxArray, p: JRParamsJax) -> JaxArray:
    """Seed a circular history buffer from the initial state, shape ``(max_history_len, N)``."""
    s0 = sigmoid_jax(lfp_jax(x0), p.e0, p.v0, p.r)
    return jnp.broadcast_to(s0, (p.max_history_len, p.n_nodes))


def update_history_and_coupling(
    x: JaxArray, history: JaxArray, k: JaxArray | int, p: JRParamsJax
) -> tuple[JaxArray, JaxArray]:
    """Write ``S(y)`` for ``x`` into the history buffer at step ``k``, then read the delayed coupling."""
    s_y = sigmoid_jax(lfp_jax(x), p.e0, p.v0, p.r)
    history = history.at[k % p.max_history_len].set(s_y)
    coupling = coupling_from_history_jax(history, k, p)
    return coupling, history


@jax.jit
def step_jax(  # noqa: PLR0913
    x: JaxArray, history: JaxArray, k: JaxArray | int, u_node: JaxArray, p: JRParamsJax, dt: float, xi: JaxArray
) -> tuple[JaxArray, JaxArray]:
    """One stochastic-Heun step with delayed coupling; advances both state and history.

    JIT-compiled standalone (unlike the ``lax.scan`` bodies above) since this is the
    single step :class:`~neuro.jansen_rit.JansenRitDynamics` calls once per ``dt`` from
    a Python loop.
    """
    coupling, history = update_history_and_coupling(x, history, k, p)
    x_next = heun_step_stoch_jax(x, u_node, coupling, p, dt, xi)
    return x_next, history


def rollout_jax(x0: JaxArray, controls_node: JaxArray, p: JRParamsJax, dt: float) -> tuple[JaxArray, JaxArray]:
    """Deterministic, differentiable forward rollout (the sysid forward model).

    Parameters
    ----------
    x0
        Initial state of shape ``(6, N)``.
    controls_node
        Per-step tES already projected to nodes, shape ``(T, N)`` (use
        :func:`project_control_jax` to build it).
    p
        Network parameters.
    dt
        Integration step in seconds.

    Returns
    -------
    x_traj
        State trajectory of shape ``(6, N, T)`` (step 1 .. T).
    y
        LFP output ``x2 - x3`` of shape ``(T, N)``.
    """

    def body(
        carry: tuple[JaxArray, JaxArray, JaxArray], u_node: JaxArray
    ) -> tuple[tuple[JaxArray, JaxArray, JaxArray], JaxArray]:
        """Advance one deterministic Heun step over the scan carry."""
        x, history, k = carry
        coupling, history = update_history_and_coupling(x, history, k, p)
        x_next = heun_step_det_jax(x, u_node, coupling, p, dt)
        return (x_next, history, k + 1), x_next

    init = (x0, seed_history(x0, p), jnp.array(0, dtype=jnp.int64))
    _, x_seq = jax.lax.scan(body, init, controls_node)
    x_traj = jnp.transpose(x_seq, (1, 2, 0))  # (T, 6, N) -> (6, N, T)
    return x_traj, lfp_jax(x_traj).T


def simulate_jax(  # noqa: PLR0913
    p: JRParamsJax,
    *,
    duration: float,
    dt: float,
    key: JaxArray | None = None,
    x0: JaxArray | None = None,
    history0: JaxArray | None = None,
    u_node_schedule: JaxArray | None = None,
) -> tuple[JaxArray, JaxArray]:
    """Integrate a network trajectory as a plant simulator.

    Deterministic when ``key is None``; otherwise additive white noise
    ``sigma * sqrt(dt) * xi`` is injected on ``x5`` each step. The contract matches
    :func:`neuro.jansen_rit.simulate_network`: the returned trajectory has shape
    ``(6, N, n_steps + 1)`` with the initial state prepended.

    Parameters
    ----------
    p
        Network parameters.
    duration
        Simulated time in seconds (``n_steps = round(duration / dt)``).
    dt
        Integration step in seconds.
    key
        PRNG key for the noise stream; ``None`` runs deterministically. Note the JAX
        noise stream does not match NumPy's, so stochastic runs are not bit-equivalent
        to :func:`simulate_network`.
    x0
        Initial state of shape ``(6, N)``; defaults to zeros.
    history0
        Initial history buffer of shape ``(max_history_len, N)``.
    u_node_schedule
        Per-step node-projected tES of shape ``(n_steps, N)``; defaults to zeros.

    Returns
    -------
    t
        Time vector of shape ``(n_steps + 1,)``.
    x_traj
        State trajectory of shape ``(6, N, n_steps + 1)``.
    """
    n_steps = round(duration / dt)
    if x0 is None:
        x0 = jnp.zeros((6, p.n_nodes), dtype=jnp.float64)
    u = jnp.zeros((n_steps, p.n_nodes), dtype=jnp.float64) if u_node_schedule is None else jnp.asarray(u_node_schedule)
    xi = (
        jnp.zeros((n_steps, p.n_nodes), dtype=jnp.float64)
        if key is None
        else jax.random.normal(key, (n_steps, p.n_nodes), dtype=jnp.float64)
    )

    def body(
        carry: tuple[JaxArray, JaxArray, JaxArray], inp: tuple[JaxArray, JaxArray]
    ) -> tuple[tuple[JaxArray, JaxArray, JaxArray], JaxArray]:
        """Advance one stochastic Heun step over the scan carry."""
        x, history, k = carry
        u_node, xi_k = inp
        coupling, history = update_history_and_coupling(x, history, k, p)
        x_next = heun_step_stoch_jax(x, u_node, coupling, p, dt, xi_k)
        return (x_next, history, k + 1), x_next

    init_history = history0 if history0 is not None else seed_history(x0, p)
    init = (x0, init_history, jnp.array(0, dtype=jnp.int64))
    _, x_seq = jax.lax.scan(body, init, (u, xi))
    x_traj = jnp.concatenate([x0[:, :, None], jnp.transpose(x_seq, (1, 2, 0))], axis=2)
    t = jnp.arange(n_steps + 1, dtype=jnp.float64) * dt
    return t, x_traj
