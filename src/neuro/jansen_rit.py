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
from typing import TYPE_CHECKING, Any, Self

import numba
import numpy as np
import numpy.typing as npt
from pydantic import Field
from simulate.component import NoLog
from simulate.dynamics import Dynamics

from neuro.config import StrictConfig, parse_array
from neuro.connectome import Connectome, _ConnectomeConfig

if TYPE_CHECKING:
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

    The network structure (weights, delays, EEG gain, ``gamma``, global coupling ``K``)
    lives on :class:`~neuro.connectome.Connectome`, not here.
    """

    A: float | FloatArray = field(default=3.25)
    B: float = field(default=22.0)
    a: float = field(default=100.0)
    b: float = field(default=50.0)
    C1: float = field(default=135.0)
    C2: float = field(default=108.0)  # 0.8 * C, C = 135
    C3: float = field(default=33.75)  # 0.25 * C
    C4: float = field(default=33.75)  # 0.25 * C
    e0: float = field(default=2.5)
    v0: float = field(default=6.0)
    r: float = field(default=0.56)
    mean_input: float = field(default=90.0)
    sigma: float = field(default=500.0)

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
        """Build :class:`JansenRitParams` from a strictly-validated config dict.

        ``A`` may be a scalar gain, a per-region list, or a path to an ``.npy``/``.npz``
        array (resolved via :func:`~neuro.config.parse_array`); every other field is a scalar.
        Unknown keys raise ``pydantic.ValidationError``.
        """
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

    out = sigmoid_jit(x2 - x3 + u_tes, e0, v0, r)  # pyramidal output; + U_tES
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
        Time window (start_s, end_s) during which tES stimulation is active.

    Returns
    -------
    t
        Time vector in seconds, shape ``(n_samples,)``.
    x_traj
        State trajectory, shape ``(6, N, n_samples)`` (``N = 1`` for a single node).
    """
    dt = dyn.dt
    n_steps = round(duration / dt)
    n_nodes = dyn.x.shape[1]

    # Per-electrode constant-current schedule over the stim window. Built once here so
    # the integration loop stays branch-free; the gamma projection lives in the component.
    n_controls = dyn.n_controls
    u_amp = np.atleast_1d(np.asarray(u_hat_tES, dtype=np.float64))
    if u_amp.shape[0] == 1:
        u_amp = np.broadcast_to(u_amp, (n_controls,))
    elif u_amp.shape[0] != n_controls:
        msg = f"u_hat_tES has {u_amp.shape[0]} electrodes but gamma has {n_controls}"
        raise ValueError(msg)

    u_sched = np.zeros((n_steps, n_controls), dtype=np.float64)
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


def project_control(u: FloatArray, gamma_2d: FloatArray, n_controls: int) -> FloatArray:
    """Project per-electrode tES current ``u`` onto nodes via ``gamma``.

    Parameters
    ----------
    u:
        Per-electrode control input, shape ``(n_controls,)``.
    gamma_2d:
        Steering matrix of shape ``(n_controls, n_nodes)``.
    n_controls:
        Number of control electrodes (must match ``gamma_2d.shape[0]``).

    Returns
    -------
    FloatArray
        Node-level stimulation ``u @ gamma_2d`` of shape ``(n_nodes,)``.
    """
    if not np.any(u):
        return np.zeros(gamma_2d.shape[1], dtype=np.float64)
    if u.size != n_controls:
        msg = f"control has {u.size} electrodes but gamma has {n_controls}"
        raise ValueError(msg)
    return u @ gamma_2d


class _JansenRitDynamicsConfig(StrictConfig):
    """Config schema for :meth:`JansenRitDynamics.from_config`.

    ``connectome`` and ``params`` nest the two sub-schemas, so a single validation pass
    strictly type-checks the whole tree and rejects unknown keys at every level.
    """

    dt: float = Field(gt=0)
    seed: int | None = None
    connectome: _ConnectomeConfig = Field(default_factory=_ConnectomeConfig)
    params: _JansenRitParamsConfig = Field(default_factory=_JansenRitParamsConfig)


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
    nodes through ``conn.gamma`` (no-op when ``gamma`` is zero, i.e. open loop).

    A single-node ``conn`` (``N = 1``, zero weights/delays) is the degenerate isolated
    node. Everything is in SI seconds: ``dt`` is the integration step and ``conn.delays``
    (ms) convert to steps via ``round(delays / (dt * 1000))``.
    """

    def __init__(
        self,
        dt: float,
        params: JansenRitParams,
        conn: Connectome,
        seed: int | None = None,
        initial_state: FloatArray | None = None,
    ) -> None:
        """Initialize the network plant from ``params`` and the structural ``conn``.

        ``initial_state`` is the ``(6, N)`` starting state; when ``None`` the network
        starts from rest (all zeros). ``seed`` seeds the noise RNG.
        """
        super().__init__(dt, integrator=None)
        self.K = conn.K

        n_nodes = conn.weights.shape[0]
        weights = conn.weights
        delays = conn.delay_steps(dt)
        gamma = conn.gamma

        a_vec = params.A
        if np.isscalar(a_vec):
            a_vec = np.full(n_nodes, a_vec, dtype=np.float64)
        self.net_params = replace(params, A=a_vec)
        self.params_tuple = self.net_params.to_numba_tuple(n_nodes)

        # tES steering matrix gamma_2d of shape (n_controls, n_nodes); single electrode is n_controls=1.
        self.gamma_2d = np.atleast_2d(gamma)
        self.n_controls = self.gamma_2d.shape[0]
        # The control input is the per-electrode tES current; the orchestrator seeds u with this width.
        self.n_inputs = self.n_controls

        if initial_state is not None:
            if initial_state.shape != (6, n_nodes):
                msg = f"initial_state must have shape (6, {n_nodes}), got {initial_state.shape}"
                raise ValueError(msg)
            self.x = initial_state.astype(np.float64).copy()
        else:
            self.x = np.zeros((6, n_nodes), dtype=np.float64)

        # Delays (ms) -> integer step lag; a connectome with zero delays is instantaneous.
        self.delay_steps = delays
        self.max_history_len = int(np.max(self.delay_steps)) + 1

        # Circular history buffer of S(y), seeded from the initial state.
        self.history = np.zeros((self.max_history_len, n_nodes), dtype=np.float64)
        self.history[:, :] = sigmoid_jit(self.x[1] - self.x[2], params.e0, params.v0, params.r)
        self.w_weights = weights

        self.rng = np.random.default_rng(seed)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the plant from a raw config dict.

        The dict is validated strictly against :class:`_JansenRitDynamicsConfig`: ``dt`` and
        ``seed`` are flat, the structural settings nest under ``connectome`` and the
        biophysical scalars under ``params``. Any unknown key (at any level) raises
        ``pydantic.ValidationError``.
        """
        cfg = _JansenRitDynamicsConfig.model_validate(config)
        conn = Connectome.from_config(cfg.connectome.model_dump())
        params = JansenRitParams.from_config(cfg.params.model_dump())

        return cls(dt=cfg.dt, params=params, conn=conn, seed=cfg.seed)

    def dynamics(self, t: float, x: FloatArray, u: FloatArray) -> FloatArray:
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

        u_node = project_control(u, self.gamma_2d, self.n_controls)
        xi = self.rng.standard_normal(n_nodes)
        return _heun_step_jit(x, u_node, self.params_tuple, self.dt, xi, coupling)

    def _make_log(self) -> NoLog:
        """Build a snapshot log of the current network state."""
        return NoLog()
