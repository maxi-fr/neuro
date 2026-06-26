"""Whole-brain Jansen-Rit neural mass network, JAX-backed.

Implements the Yu et al. 2024 formulation (their Eq. 1-4). The state vector is
``x = [x1, x2, x3, x4, x5, x6]`` with the paper's 1-6 indexing -- pyramidal pair
``(x1, x4)``, excitatory pair ``(x2, x5)``, inhibitory pair ``(x3, x6)`` -- and the
observed output is ``y = x2 - x3``::

    x1' = x4
    x4' = A a [ S(x2 - x3 + U_tES) ]                       - 2a x4 - a^2 x1
    x2' = x5
    x5' = A a [ I + C2 S(C1 x1) + K sum_j w_ij S(x2-x3) ]  - 2a x5 - a^2 x2 + zeta
    x3' = x6
    x6' = B b [ C4 S(C3 x1) ]                              - 2b x6 - b^2 x3
    S(v) = 2 e0 / (1 + exp(r (v0 - v)))

The tES polarization ``U_tES`` and the network coupling sum ``K sum_j w_ij S(.)`` are
zero for an isolated, uncontrolled node. Default parameters follow Yu et al. 2024,
Table 1; the only free knob is the white-noise std ``sigma`` (the paper gives no
value), tuned so the ``A = 3.25`` background sits near peak-to-peak amplitude 2.

Integration is stochastic Heun with additive noise on the ``x5'`` equation; a
noiseless run is just ``sigma = 0``. Everything is in SI seconds: ``a = 100``,
``b = 50`` per second, ``dt = 1e-4`` s (0.1 ms). The actual dynamics equations live in
:mod:`neuro.jansen_rit_jax`; :class:`JansenRitDynamics` below is the stateful
``simulate.Dynamics`` plant wrapping them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Self

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from simulate.dynamics import Dynamics

from neuro.config import parse_array
from neuro.jansen_rit_jax import enable_x64, from_jansen_rit_params, project_control_jax, seed_history, step_jax

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
    mean_input: float | FloatArray = field(default=90.0, metadata={"bounds": (0.0, 500.0)})
    sigma: float = field(default=500.0, metadata={"bounds": (0.0, 0.0)})

    # Network parameters (mirrors JRSymbolicParams)
    eeg_gain: FloatArray = field(
        default_factory=lambda: np.ones((1, 1), dtype=np.float64), metadata={"bounds": (0.0, 10.0)}
    )
    w_weights: FloatArray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float64))
    delay_steps: npt.NDArray[np.int64] = field(default_factory=lambda: np.zeros((1, 1), dtype=np.int64))
    K: float = field(default=1.0, metadata={"bounds": (0.0, 10.0)})
    gamma: FloatArray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float64))

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

        return cls(**network_kwargs, **params_cfg)  # type: ignore


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


@dataclass(frozen=True)
class JansenRitDynamicsLog:
    """Dataclass log snapshot of the Jansen-Rit network state."""

    x: np.ndarray


class JansenRitDynamics(Dynamics[JansenRitDynamicsLog]):
    """Whole-brain Jansen-Rit network as a ``simulate`` :class:`Dynamics` plant.

    A thin stateful wrapper around the JAX step function :func:`~neuro.jansen_rit_jax.step_jax`:
    it owns ``self.x`` (the ``(6, N)`` state, required by the orchestrator's logging),
    the delayed-coupling history buffer, and the per-step noise RNG, and delegates the
    actual physics to :mod:`neuro.jansen_rit_jax`. ``integrator=None`` so the base
    :meth:`~simulate.dynamics.Dynamics.update` treats :meth:`dynamics` as a discrete
    ``x_next`` transition.

    The control input ``u`` is the per-electrode tES current, projected to nodes
    through ``connectome.gamma`` (no-op when ``gamma`` is unset, i.e. open loop).
    Passing ``connectome=None`` yields the degenerate ``N=1`` isolated node (no
    coupling, no stimulation). Everything is in SI seconds: ``dt`` is the integration
    step and ``connectome.delays`` (ms) convert to steps via
    ``round(delays / (dt * 1000))``.
    """

    def __init__(
        self,
        *,
        dt: float,
        params: JansenRitParams,
        seed: int | None = None,
        initial_state: FloatArray | None = None,
    ) -> None:
        """Initialize the network plant from params."""
        super().__init__(dt, integrator=None)
        enable_x64()

        n_nodes = params.w_weights.shape[0]
        self.p = from_jansen_rit_params(params, n_nodes)
        self.n_elec = self.p.gamma.shape[0]
        # The control input is the per-electrode tES current; the orchestrator seeds u with this width.
        self.n_inputs = self.n_elec

        if initial_state is not None:
            if initial_state.shape != (6, n_nodes):
                msg = f"initial_state must have shape (6, {n_nodes}), got {initial_state.shape}"
                raise ValueError(msg)
            self.x = initial_state.astype(np.float64).copy()
        else:
            self.x = np.zeros((6, n_nodes), dtype=np.float64)

        self.history = seed_history(jnp.asarray(self.x), self.p)
        self.rng = np.random.default_rng(seed)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate from a raw config dict.

        Config keys for the dynamics component are flat. Parameters for the physical
        model should be nested under the ``params`` key. ``speed``, ``K`` and ``dt``
        are required.
        """
        # Ensure we use "sigma" consistently instead of "gamma_sigma"
        if "gamma_sigma" in config and "sigma" not in config:
            config["sigma"] = config["gamma_sigma"]

        # Ensure K is passed properly if it's top-level
        if "params" not in config:
            config["params"] = {}
        if "K" in config and "K" not in config["params"]:
            config["params"]["K"] = float(config["K"])

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
        k = round(t / self.dt)
        u_arr = np.atleast_1d(np.asarray(u, dtype=np.float64))
        if u_arr.shape[0] == 1 and self.n_elec != 1:
            u_arr = np.broadcast_to(u_arr, (self.n_elec,))
        u_node = project_control_jax(jnp.asarray(u_arr), self.p.gamma)
        xi = jnp.asarray(self.rng.standard_normal(self.p.n_nodes))
        x_next, self.history = step_jax(jnp.asarray(x), self.history, k, u_node, self.p, self.dt, xi)
        return np.asarray(x_next)

    def _make_log(self) -> JansenRitDynamicsLog:
        """Build a snapshot log of the current network state."""
        return JansenRitDynamicsLog(x=self.x.copy())
