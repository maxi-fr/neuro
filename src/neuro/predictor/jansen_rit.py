from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from trajopt.constraints.bounds import ControlBound
from trajopt.constraints.constraint_list import ConstraintList
from trajopt.costs.objective import Objective
from trajopt.costs.quadratic import DiagonalCost
from trajopt.dynamics.base import DiscreteDynamics
from trajopt.problem import Problem

from neuro.connectome import Connectome
from neuro.control.costs import ExcludeInitialKnotState, KirchhoffPenaltyCost, L1ControlCost, SumCost
from neuro.control.mpc import kirchhoff_constraint
from neuro.jansen_rit import JansenRitDynamics, JansenRitParams
from neuro.predictor.inference import InferencePredictor
from neuro.stimulation import build_stimulation
from neuro.stimulation.base import StimulationModel, _AnalyticalConfig, _DynamicYuConfig, _NullConfig, _Roast3DConfig

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import ArrayLike
    from trajopt.costs.base import CostFunction

    from neuro.types import FloatArray


def enable_x64() -> None:
    """Enable JAX 64-bit precision for float64 parity with the NumPy/Numba reference."""
    jax.config.update("jax_enable_x64", True)  # noqa: FBT003


def sigmoid_jax(v: jax.Array | float, e0: jax.Array | float, v0: jax.Array | float, r: jax.Array | float) -> jax.Array:
    """Evaluate the firing-rate sigmoid ``S(v) = 2 e0 / (1 + exp(r (v0 - v)))``."""
    return 2.0 * e0 / (1.0 + jnp.exp(r * (v0 - v)))


def lfp_jax(x: jax.Array) -> jax.Array:
    """Observed Local Field Potential ``y = x2 - x3`` from state or trajectory."""
    return x[1] - x[2]


def eeg_jax(x: jax.Array, gain: jax.Array) -> jax.Array:
    """Map state to Raw EEG measurement ``gain @ lfp``."""
    return gain @ lfp_jax(x)


def project_control_jax(u: jax.Array, gamma_2d: jax.Array) -> jax.Array:
    """Project per-electrode Control Current onto nodes via ``u @ gamma_2d``."""
    return u @ gamma_2d


class JansenRitModel(DiscreteDynamics, InferencePredictor):
    """Whole-brain Jansen-Rit network as an equinox module for trajopt MPC."""

    # Continuous biophysical and network parameters (PyTree leaves)
    A: jax.Array
    B: jax.Array
    a: jax.Array
    b: jax.Array
    C1: jax.Array
    C2: jax.Array
    C3: jax.Array
    C4: jax.Array
    e0: jax.Array
    v0: jax.Array
    r: jax.Array
    mean_input: jax.Array
    sigma: jax.Array
    K: jax.Array
    w_weights: jax.Array
    eeg_gain: jax.Array
    gamma: jax.Array

    # Static structural dimensions and indices
    delays_tuple: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    n_nodes: int = eqx.field(static=True)
    max_history_len: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    n_channels: int = eqx.field(static=True)
    n_controls: int = eqx.field(static=True)
    n_outputs: int = eqx.field(static=True)
    n_y: int = eqx.field(static=True)
    n_u: int = eqx.field(static=True)
    _init_x0: jax.Array | None = None

    def __init__(  # noqa: PLR0913 -- physical parameter leaves and static metadata
        self,
        *,
        A: jax.Array,
        B: jax.Array,
        a: jax.Array,
        b: jax.Array,
        C1: jax.Array,
        C2: jax.Array,
        C3: jax.Array,
        C4: jax.Array,
        e0: jax.Array,
        v0: jax.Array,
        r: jax.Array,
        mean_input: jax.Array,
        sigma: jax.Array,
        K: jax.Array,
        w_weights: jax.Array,
        eeg_gain: jax.Array,
        gamma: jax.Array,
        delay_steps: np.ndarray,
        dt: float,
        x0: FloatArray | None = None,
    ) -> None:
        """Initialize the model adapter directly from arrays and metadata."""
        n_nodes = int(w_weights.shape[0])
        delays_arr = np.asarray(delay_steps, dtype=np.int64).reshape(n_nodes, n_nodes)
        max_history_len = int(delays_arr.max()) + 1
        n_states = 6 * n_nodes + max_history_len * n_nodes + 1
        n_controls = int(gamma.shape[0])

        super().__init__(n=n_states, m=n_controls, ne=n_states)

        self.A = A
        self.B = B
        self.a = a
        self.b = b
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.C4 = C4
        self.e0 = e0
        self.v0 = v0
        self.r = r
        self.mean_input = mean_input
        self.sigma = sigma
        self.K = K
        self.w_weights = w_weights
        self.eeg_gain = eeg_gain
        self.gamma = gamma
        self._init_x0 = jnp.asarray(x0, dtype=jnp.float64) if x0 is not None else None

        self.delays_tuple = tuple(map(tuple, delays_arr.tolist()))
        self.n_nodes = n_nodes
        self.max_history_len = max_history_len
        self.dt = float(dt)
        self.n_channels = int(eeg_gain.shape[0])
        self.n_controls = n_controls
        self.n_outputs = self.n_channels
        self.n_y = 1
        self.n_u = 1

    @property
    def delay_steps(self) -> np.ndarray:
        """Conduction delay steps matrix as a NumPy array of shape ``(n_nodes, n_nodes)``."""
        return np.asarray(self.delays_tuple, dtype=np.int64)

    @classmethod
    def from_plant_components(  # noqa: PLR0913, PLR0917 -- decoupled domain components
        cls,
        params: JansenRitParams,
        conn: Connectome | None = None,
        stim: StimulationModel | None = None,
        leadfield: FloatArray | None = None,
        dt: float = 1e-4,
        n_nodes: int = 1,
        *,
        x0: FloatArray | None = None,
    ) -> JansenRitModel:
        """Build :class:`JansenRitModel` from modern decoupled domain components."""
        if conn is not None:
            n_nodes = conn.weights.shape[0]
            w_weights = jnp.asarray(conn.weights, dtype=jnp.float64)
            delay_steps = conn.delay_steps(dt)
            k_coupling = jnp.asarray(conn.K, dtype=jnp.float64)
        else:
            w_weights = jnp.zeros((n_nodes, n_nodes), dtype=jnp.float64)
            delay_steps = np.zeros((n_nodes, n_nodes), dtype=np.int64)
            k_coupling = jnp.asarray(0.0, dtype=jnp.float64)

        if stim is not None:
            gamma_arr = getattr(stim, "gamma", None)
            if gamma_arr is not None:
                gamma = jnp.asarray(gamma_arr, dtype=jnp.float64)
            else:
                gamma = jnp.zeros((stim.n_controls, n_nodes), dtype=jnp.float64)
        else:
            gamma = jnp.zeros((1, n_nodes), dtype=jnp.float64)

        eeg_gain = (
            jnp.asarray(leadfield, dtype=jnp.float64) if leadfield is not None else jnp.eye(n_nodes, dtype=jnp.float64)
        )

        a_vec = np.asarray(params.A, dtype=np.float64)
        a_jax = jnp.broadcast_to(jnp.asarray(a_vec, dtype=jnp.float64), (n_nodes,))
        mean_in_vec = np.asarray(params.mean_input, dtype=np.float64)
        mean_in_jax = jnp.broadcast_to(jnp.asarray(mean_in_vec, dtype=jnp.float64), (n_nodes,))

        return cls(
            A=a_jax,
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
            mean_input=mean_in_jax,
            sigma=jnp.asarray(params.sigma, dtype=jnp.float64),
            K=k_coupling,
            w_weights=w_weights,
            eeg_gain=eeg_gain,
            gamma=gamma,
            delay_steps=delay_steps,
            dt=dt,
            x0=x0,
        )

    @classmethod
    def from_dynamics(
        cls,
        dyn: JansenRitDynamics,
        leadfield: FloatArray | None = None,
    ) -> JansenRitModel:
        """Construct a :class:`JansenRitModel` directly from a live Plant instance."""
        n_nodes = dyn.x.shape[1]
        delay_steps = dyn.delay_steps
        w_weights = jnp.asarray(dyn.w_weights, dtype=jnp.float64)
        k_coupling = jnp.asarray(dyn.K, dtype=jnp.float64)

        stim = dyn.stim
        gamma_arr = getattr(stim, "gamma", None)
        if gamma_arr is not None:
            gamma = jnp.asarray(gamma_arr, dtype=jnp.float64)
        else:
            gamma = jnp.zeros((stim.n_controls, n_nodes), dtype=jnp.float64)

        eeg_gain = (
            jnp.asarray(leadfield, dtype=jnp.float64) if leadfield is not None else jnp.eye(n_nodes, dtype=jnp.float64)
        )

        params = dyn.net_params
        a_vec = np.asarray(params.A, dtype=np.float64)
        a_jax = jnp.broadcast_to(jnp.asarray(a_vec, dtype=jnp.float64), (n_nodes,))
        mean_in_vec = np.asarray(params.mean_input, dtype=np.float64)
        mean_in_jax = jnp.broadcast_to(jnp.asarray(mean_in_vec, dtype=jnp.float64), (n_nodes,))

        return cls(
            A=a_jax,
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
            mean_input=mean_in_jax,
            sigma=jnp.asarray(params.sigma, dtype=jnp.float64),
            K=k_coupling,
            w_weights=w_weights,
            eeg_gain=eeg_gain,
            gamma=gamma,
            delay_steps=delay_steps,
            dt=dyn.dt,
            x0=dyn.x,
        )

    @classmethod
    def from_plant(
        cls,
        dyn: JansenRitDynamics,
        leadfield: FloatArray | None = None,
    ) -> JansenRitModel:
        """Alias for :meth:`from_dynamics` for 0 model-Plant mismatch."""
        return cls.from_dynamics(dyn, leadfield=leadfield)

    def rhs(self, x: jax.Array, coupling: jax.Array | float, u_tes: jax.Array | float) -> jax.Array:
        """Continuous-time right-hand side of the Jansen-Rit network."""
        x1, x2, x3, x4, x5, x6 = x

        out = sigmoid_jax(x2 - x3 + u_tes, self.e0, self.v0, self.r)
        exc = sigmoid_jax(self.C1 * x1, self.e0, self.v0, self.r)
        inh = sigmoid_jax(self.C3 * x1, self.e0, self.v0, self.r)

        dx1 = x4
        dx4 = self.A * self.a * out - 2.0 * self.a * x4 - self.a**2 * x1
        dx2 = x5
        dx5 = self.A * self.a * (self.mean_input + self.C2 * exc + coupling) - 2.0 * self.a * x5 - self.a**2 * x2
        dx3 = x6
        dx6 = self.B * self.b * self.C4 * inh - 2.0 * self.b * x6 - self.b**2 * x3

        return jnp.stack([dx1, dx2, dx3, dx4, dx5, dx6])

    def heun_step(
        self,
        x: jax.Array,
        u_tes: jax.Array | float,
        coupling: jax.Array | float,
        dt: float,
    ) -> jax.Array:
        """Advance one deterministic Heun step (no noise)."""
        f0 = self.rhs(x, coupling, u_tes)
        x_pred = x + dt * f0
        f1 = self.rhs(x_pred, coupling, u_tes)
        return x + 0.5 * dt * (f0 + f1)

    def coupling_from_history(self, history: jax.Array, k: jax.Array | int) -> jax.Array:
        """Delayed network coupling from circular history buffer of ``S(y)``."""
        length = self.max_history_len
        delays = jnp.asarray(self.delays_tuple)
        rows = (k - delays) % length
        cols = jnp.arange(self.n_nodes)[None, :]
        s_past = history[rows, cols]
        return self.K * jnp.sum(self.w_weights * s_past, axis=1)

    def seed_history(self, x0: jax.Array) -> jax.Array:
        """Seed a circular history buffer from the initial state, shape ``(max_history_len, N)``."""
        s0 = sigmoid_jax(lfp_jax(x0), self.e0, self.v0, self.r)
        return jnp.broadcast_to(s0, (self.max_history_len, self.n_nodes))

    def update_history_and_coupling(
        self, x: jax.Array, history: jax.Array, k: jax.Array | int
    ) -> tuple[jax.Array, jax.Array]:
        """Write ``S(y)`` for ``x`` into the history buffer at step ``k``, then read delayed coupling."""
        s_y = sigmoid_jax(lfp_jax(x), self.e0, self.v0, self.r)
        history = history.at[k % self.max_history_len].set(s_y)
        coupling = self.coupling_from_history(history, k)
        return coupling, history

    def pack_state(self, x_ode: jax.Array, history: jax.Array, k: jax.Array | float) -> jax.Array:
        """Pack ODE state ``(6, N)``, history ``(L, N)``, and step index ``k`` into 1D state."""
        return jnp.concatenate(
            [
                jnp.asarray(x_ode, dtype=jnp.float64).reshape(-1),
                jnp.asarray(history, dtype=jnp.float64).reshape(-1),
                jnp.atleast_1d(jnp.asarray(k, dtype=jnp.float64)),
            ]
        )

    def unpack_state(self, z: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Unpack 1D state ``z`` into ODE state ``(6, N)``, history ``(L, N)``, and step index ``k``."""
        n_ode = 6 * self.n_nodes
        n_hist = self.max_history_len * self.n_nodes
        x_ode = z[:n_ode].reshape(6, self.n_nodes)
        history = z[n_ode : n_ode + n_hist].reshape(self.max_history_len, self.n_nodes)
        k = z[-1]
        return x_ode, history, k

    def discrete_dynamics(
        self,
        x: jax.Array,
        u: jax.Array,
        t: float | jax.Array,
        dt: float | jax.Array,
    ) -> jax.Array:
        """Advance one deterministic Heun step under Control Current ``u`` -> ``x'``."""
        del t, dt
        x_ode, history, k = self.unpack_state(x)
        k_int = k.astype(jnp.int64)

        u_node = project_control_jax(u, self.gamma)
        coupling, next_history = self.update_history_and_coupling(x_ode, history, k_int)
        next_x_ode = self.heun_step(x_ode, u_node, coupling, self.dt)
        next_k = k + 1.0

        return self.pack_state(next_x_ode, next_history, next_k)

    def initial_state(self) -> FloatArray:
        """Return the initialized state vector."""
        if self._init_x0 is not None:
            x0 = jnp.asarray(self._init_x0, dtype=jnp.float64)
        else:
            x0 = jnp.zeros((6, self.n_nodes), dtype=jnp.float64)
        hist0 = self.seed_history(x0)
        z0 = self.pack_state(x0, hist0, 0.0)
        return np.asarray(z0, dtype=np.float64)

    def is_ready(self, state: FloatArray) -> bool:  # noqa: ARG002 -- state is ready by construction
        """Report whether the Predictor has absorbed enough history to predict."""
        return True

    def absorb(self, state: FloatArray, y: FloatArray, u: FloatArray) -> FloatArray:
        """Absorb state / measurement and last applied control."""
        del u
        z = np.asarray(state, dtype=np.float64).copy()
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)

        # When full 6N state is provided directly by the Plant
        if y_arr.size == 6 * self.n_nodes:
            z[: 6 * self.n_nodes] = y_arr
            x_ode = y_arr.reshape(6, self.n_nodes)
            k_val = round(float(z[-1]))
            s_y = 2.0 * float(self.e0) / (1.0 + np.exp(float(self.r) * (float(self.v0) - (x_ode[1] - x_ode[2]))))
            n_ode = 6 * self.n_nodes
            hist_flat = z[n_ode : n_ode + self.max_history_len * self.n_nodes].reshape(
                self.max_history_len, self.n_nodes
            )
            hist_flat[k_val % self.max_history_len] = s_y
            z[n_ode : n_ode + self.max_history_len * self.n_nodes] = hist_flat.reshape(-1)
        elif y_arr.size == self.n:
            z[:] = y_arr
        return z

    def forward_rollout(self, x0: jax.Array, controls_node: jax.Array, dt: float) -> tuple[jax.Array, jax.Array]:
        """Deterministic, differentiable forward Rollout over control schedule."""

        def body(
            carry: tuple[jax.Array, jax.Array, jax.Array], u_node: jax.Array
        ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
            x, history, k = carry
            coupling, history = self.update_history_and_coupling(x, history, k)
            x_next = self.heun_step(x, u_node, coupling, dt)
            return (x_next, history, k + 1), x_next

        init = (x0, self.seed_history(x0), jnp.array(0, dtype=jnp.int64))
        _, x_seq = jax.lax.scan(body, init, controls_node)
        x_traj = jnp.transpose(x_seq, (1, 2, 0))  # (T, 6, N) -> (6, N, T)
        return x_traj, lfp_jax(x_traj).T

    def free_run(
        self,
        y_hists: FloatArray,
        u_hists: FloatArray,
        u_futures: FloatArray,
    ) -> jax.Array:
        """Stateless free-run rollout in JAX."""
        del y_hists, u_hists
        u_fut = jnp.asarray(u_futures, dtype=jnp.float64)
        x0 = jnp.zeros((6, self.n_nodes), dtype=jnp.float64)
        u_node_fut = u_fut @ self.gamma
        _, y_out = self.forward_rollout(x0, u_node_fut, self.dt)
        return y_out @ self.eeg_gain.T

    def to_checkpoint(self) -> tuple[dict[str, Any], dict[str, FloatArray]]:
        """Serialize model to checkpoint metadata and arrays."""
        meta = {
            "model_type": "jansen_rit",
            "dt": self.dt,
            "n_nodes": self.n_nodes,
            "max_history_len": self.max_history_len,
            "n_channels": self.n_channels,
            "n_controls": self.n_controls,
        }
        arrays = {
            "A": np.asarray(self.A, dtype=np.float64),
            "B": np.asarray(self.B, dtype=np.float64),
            "a": np.asarray(self.a, dtype=np.float64),
            "b": np.asarray(self.b, dtype=np.float64),
            "C1": np.asarray(self.C1, dtype=np.float64),
            "C2": np.asarray(self.C2, dtype=np.float64),
            "C3": np.asarray(self.C3, dtype=np.float64),
            "C4": np.asarray(self.C4, dtype=np.float64),
            "e0": np.asarray(self.e0, dtype=np.float64),
            "v0": np.asarray(self.v0, dtype=np.float64),
            "r": np.asarray(self.r, dtype=np.float64),
            "mean_input": np.asarray(self.mean_input, dtype=np.float64),
            "sigma": np.asarray(self.sigma, dtype=np.float64),
            "K": np.asarray(self.K, dtype=np.float64),
            "w_weights": np.asarray(self.w_weights, dtype=np.float64),
            "eeg_gain": np.asarray(self.eeg_gain, dtype=np.float64),
            "gamma": np.asarray(self.gamma, dtype=np.float64),
            "delay_steps": np.asarray(self.delays_tuple, dtype=np.float64),
        }
        return meta, arrays

    @classmethod
    def from_checkpoint(cls, meta: dict[str, Any], arrays: dict[str, FloatArray]) -> Self:
        """Rebuild model from checkpoint metadata and arrays."""
        return cls(
            A=jnp.asarray(arrays["A"], dtype=jnp.float64),
            B=jnp.asarray(arrays["B"], dtype=jnp.float64),
            a=jnp.asarray(arrays["a"], dtype=jnp.float64),
            b=jnp.asarray(arrays["b"], dtype=jnp.float64),
            C1=jnp.asarray(arrays["C1"], dtype=jnp.float64),
            C2=jnp.asarray(arrays["C2"], dtype=jnp.float64),
            C3=jnp.asarray(arrays["C3"], dtype=jnp.float64),
            C4=jnp.asarray(arrays["C4"], dtype=jnp.float64),
            e0=jnp.asarray(arrays["e0"], dtype=jnp.float64),
            v0=jnp.asarray(arrays["v0"], dtype=jnp.float64),
            r=jnp.asarray(arrays["r"], dtype=jnp.float64),
            mean_input=jnp.asarray(arrays["mean_input"], dtype=jnp.float64),
            sigma=jnp.asarray(arrays["sigma"], dtype=jnp.float64),
            K=jnp.asarray(arrays["K"], dtype=jnp.float64),
            w_weights=jnp.asarray(arrays["w_weights"], dtype=jnp.float64),
            eeg_gain=jnp.asarray(arrays["eeg_gain"], dtype=jnp.float64),
            gamma=jnp.asarray(arrays["gamma"], dtype=jnp.float64),
            delay_steps=np.asarray(arrays["delay_steps"], dtype=np.int64),
            dt=float(meta["dt"]),
        )


def _build_stim_model(stimulation: dict[str, Any], conn: Connectome) -> StimulationModel:
    """Instantiate a StimulationModel from a config dict."""
    model_kind = stimulation.get("model", "none")
    if model_kind == "analytical":
        stim_cfg = _AnalyticalConfig.model_validate(stimulation)
    elif model_kind == "roast_3d":
        stim_cfg = _Roast3DConfig.model_validate(stimulation)
    elif model_kind == "yu_dynamic":
        stim_cfg = _DynamicYuConfig.model_validate(stimulation)
    else:
        stim_cfg = _NullConfig.model_validate(stimulation)
    return build_stimulation(stim_cfg, conn)


def _resolve_model(  # noqa: PLR0913, PLR0917 -- model resolution parameters
    model: JansenRitModel | None,
    artifact: str | Path | None,
    params: JansenRitParams | dict[str, Any] | None,
    connectome: Connectome | dict[str, Any] | None,
    stimulation: StimulationModel | dict[str, Any] | None,
    leadfield: FloatArray | None,
    dt: float,
) -> JansenRitModel:
    """Resolve or construct the JansenRitModel instance."""
    if model is not None:
        return model
    if artifact is not None:
        return JansenRitModel.load(artifact)

    p = (
        JansenRitParams.from_config(params)
        if isinstance(params, dict)
        else (params if params is not None else JansenRitParams())
    )
    conn = Connectome.from_config(connectome) if isinstance(connectome, dict) else connectome

    if isinstance(stimulation, dict):
        if conn is None:
            msg = "connectome must be provided when stimulation is given as a dict"
            raise ValueError(msg)
        stim: StimulationModel | None = _build_stim_model(stimulation, conn)
    else:
        stim = stimulation

    n_nodes = conn.weights.shape[0] if conn is not None else 1
    return JansenRitModel.from_plant_components(
        params=p,
        conn=conn,
        stim=stim,
        leadfield=leadfield,
        dt=dt,
        n_nodes=n_nodes,
    )


def build_jansen_rit_problem(  # noqa: PLR0913 -- problem construction arguments
    model: JansenRitModel | None = None,
    *,
    horizon: int,
    u_max: ArrayLike,
    dt: float = 1e-4,
    params: JansenRitParams | dict[str, Any] | None = None,
    connectome: Connectome | dict[str, Any] | None = None,
    stimulation: StimulationModel | dict[str, Any] | None = None,
    leadfield: FloatArray | None = None,
    artifact: str | Path | None = None,
    w_y: float = 1.0,
    w_u: float = 0.0,
    w_u_l1: float = 0.0,
    kirchhoff: bool = False,
    w_kirchhoff: float = 0.0,
) -> Problem:
    """Assemble a trajopt MPC Problem for the Jansen-Rit model adapter."""
    resolved_model = _resolve_model(model, artifact, params, connectome, stimulation, leadfield, dt)
    n, m = resolved_model.n, resolved_model.m
    N = horizon + 1

    # Stage cost: penalize LFP / state deviation (x2 - x3) and control effort
    n_nodes = resolved_model.n_nodes
    Q_diag = jnp.zeros(n)
    for node in range(n_nodes):
        idx_x2 = 1 * n_nodes + node
        idx_x3 = 2 * n_nodes + node
        Q_diag = Q_diag.at[idx_x2].set(2.0 * w_y / horizon)
        Q_diag = Q_diag.at[idx_x3].set(2.0 * w_y / horizon)

    stage = DiagonalCost.tracking(Q_diag, jnp.full(m, 2.0 * w_u / horizon), jnp.zeros(n), jnp.zeros(m))
    costs: list[CostFunction] = [ExcludeInitialKnotState(stage)]
    if w_u_l1 > 0:
        costs.append(L1ControlCost(n=n, m=m, w_l1=w_u_l1, horizon=horizon))
    if w_kirchhoff > 0:
        costs.append(KirchhoffPenaltyCost(n=n, m=m, w_k=w_kirchhoff, horizon=horizon))

    stage_cost = SumCost(costs) if len(costs) > 1 else costs[0]
    terminal = DiagonalCost.terminal_tracking(Q_diag, jnp.zeros(n), m)
    objective = Objective(stage_cost=stage_cost, terminal_cost=terminal, N=N)

    u_max_arr = np.broadcast_to(np.atleast_1d(np.asarray(u_max, dtype=np.float64)), (m,))
    constraints = ConstraintList(n=n, m=m, N=N)
    constraints.add_constraint(ControlBound(n=n, m=m, u_min=-u_max_arr, u_max=u_max_arr), range(N - 1))
    if kirchhoff:
        constraints.add_constraint(kirchhoff_constraint(n, m), range(N - 1))

    return Problem(model=resolved_model, obj=objective, constraints=constraints, N=N)
