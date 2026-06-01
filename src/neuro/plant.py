"""FitzHugh-Nagumo whole-brain plant components for the simulate framework.

* :class:`NativeFHNDynamics` / :class:`FHNOutput` -- pure NumPy FHN dynamics and EEG output.
* :class:`TVBFHNDynamics` / :class:`TVBFHNOutput` -- backed by The Virtual Brain (TVB).
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Self, cast

import numpy as np
import numpy.typing as npt
from neurolib.utils.loadData import Dataset
from pydantic import BaseModel, ConfigDict
from simulate.dynamics import Dynamics
from simulate.output import Output

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import ArrayLike

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    from tvb.datatypes.connectivity import Connectivity as _Connectivity
    from tvb.datatypes.projections import ProjectionSurfaceEEG as _ProjectionSurfaceEEG
    from tvb.datatypes.region_mapping import RegionMapping as _RegionMapping
    from tvb.simulator import coupling as _tvb_coupling
    from tvb.simulator import integrators as _tvb_integrators
    from tvb.simulator import monitors as _tvb_monitors
    from tvb.simulator import noise as _tvb_noise
    from tvb.simulator import simulator as _tvb_simulator
    from tvb.simulator.models.oscillator import Generic2dOscillator as _Generic2dOscillator

FloatArray = npt.NDArray[np.float64]
_TVBSnapshot = tuple[
    FloatArray,  # current_state
    int,  # current_step
    FloatArray,  # history buffer
    object,  # RNG state (numpy RandomState tuple)
]


def _as_tvb_params(model_params: dict[str, Any] | None) -> dict[str, FloatArray]:
    """Coerce a config-supplied parameter mapping into TVB-ready numpy arrays.

    TVB model attributes are ``NArray`` traits, so each override must be a numpy
    array (a scalar or list from YAML/JSON is wrapped accordingly).
    """
    if not model_params:
        return {}
    return {key: np.array(value, dtype=np.float64) for key, value in model_params.items()}


def _make_leadfield(n_sensors: int, n_nodes: int, seed: int) -> FloatArray:
    """Create a row-normalised random lead-field matrix."""
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((n_sensors, n_nodes))
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix.astype(np.float64)


def _load_leadfield(path: str | Path, n_nodes: int) -> FloatArray:
    """Load a precomputed lead-field matrix and validate its shape."""
    matrix: FloatArray = np.load(path).astype(np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != n_nodes:  # noqa: PLR2004
        msg = (
            f"Lead-field at {path!r} has shape {matrix.shape}; expected a 2-D "
            f"matrix with {n_nodes} columns (one per network node)."
        )
        raise ValueError(msg)
    return matrix


def _tvb_eeg_leadfield(
    n_regions: int,
    projection_file: str = "projection_eeg_65_surface_16k.npy",
    region_mapping_file: str = "regionMapping_16k_76.txt",
) -> FloatArray:
    """Build a region-level EEG gain matrix from TVB's surface projection."""
    proj = _ProjectionSurfaceEEG.from_file(projection_file)
    rmap = _RegionMapping.from_file(region_mapping_file).array_data
    surface_gain = np.asarray(proj.projection_data, dtype=np.float64)
    n_sensors = surface_gain.shape[0]
    gain = np.zeros((n_sensors, n_regions), dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for r in range(n_regions):
            verts = np.where(rmap == r)[0]
            if verts.size:
                gain[:, r] = np.nanmean(surface_gain[:, verts], axis=1)
    return np.nan_to_num(gain, nan=0.0)


# --- Logging schemas --------------------------------------------------------


class FHNDynamicsLog(BaseModel):
    """Pydantic model for internal FHN dynamics state logging."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    v_ou: FloatArray | None = None
    w_ou: FloatArray | None = None


class FHNOutputLog(BaseModel):
    """Pydantic model for FHN EEG output logging."""

    model_config = ConfigDict(arbitrary_types_allowed=True)


# --- Output Classes ---------------------------------------------------------


class FHNOutput(Output[FHNOutputLog]):
    """EEG output component for FitzHugh-Nagumo plants."""

    def __init__(
        self,
        dt: float = 0.1,
        n_sensors: int = 64,
        leadfield_seed: int = 0,
        leadfield_path: str | Path | None = None,
        n_nodes: int = 80,
    ) -> None:
        super().__init__(dt)
        self._leadfield = (
            _load_leadfield(leadfield_path, n_nodes)
            if leadfield_path is not None
            else _make_leadfield(n_sensors, n_nodes, leadfield_seed)
        )
        self._n_sensors = self._leadfield.shape[0]
        self._n_nodes = n_nodes

    @property
    def leadfield(self) -> FloatArray:
        """The EEG lead-field matrix."""
        return self._leadfield

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the output component from config."""
        return cls(
            dt=float(config["dt"]),
            n_sensors=config.get("n_sensors", 64),
            leadfield_seed=config.get("leadfield_seed", 0),
            leadfield_path=config.get("leadfield_path"),
            n_nodes=config.get("n_nodes", 80),
        )

    def update(
        self,
        t: float,  # noqa: ARG002
        x: float | np.ndarray,
        u: float | np.ndarray,  # noqa: ARG002
    ) -> tuple[FloatArray, FHNOutputLog]:
        """Compute the EEG output from current state."""
        x_vec = self.to_col_vec(x)
        # x is [V; W] of shape (2 * n_nodes, 1)
        n_nodes = x_vec.shape[0] // 2
        activity = x_vec[:n_nodes]
        y_vec = self._leadfield @ activity
        return cast("FloatArray", self.from_col_vec(y_vec)), FHNOutputLog()


class TVBFHNOutput(Output[FHNOutputLog]):
    """EEG output component for TVB FitzHugh-Nagumo plants."""

    def __init__(
        self,
        dt: float = 0.1,
        n_nodes: int = 76,
        projection: str = "projection_eeg_65_surface_16k.npy",
        region_mapping: str = "regionMapping_16k_76.txt",
    ) -> None:
        super().__init__(dt)
        self._leadfield = _tvb_eeg_leadfield(n_nodes, projection, region_mapping)
        self._n_sensors = self._leadfield.shape[0]
        self._n_nodes = n_nodes

    @property
    def leadfield(self) -> FloatArray:
        """The region-averaged TVB EEG lead-field matrix."""
        return self._leadfield

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the output component from config."""
        return cls(
            dt=float(config["dt"]),
            n_nodes=config.get("n_nodes", 76),
            projection=config.get("projection", "projection_eeg_65_surface_16k.npy"),
            region_mapping=config.get("region_mapping", "regionMapping_16k_76.txt"),
        )

    def update(
        self,
        t: float,  # noqa: ARG002
        x: float | np.ndarray,
        u: float | np.ndarray,  # noqa: ARG002
    ) -> tuple[FloatArray, FHNOutputLog]:
        """Compute the EEG output from current state using TVB lead-field."""
        x_vec = self.to_col_vec(x)
        n_nodes = x_vec.shape[0] // 2
        activity = x_vec[:n_nodes]
        y_vec = self._leadfield @ activity
        return cast("FloatArray", self.from_col_vec(y_vec)), FHNOutputLog()


# --- Dynamics Classes -------------------------------------------------------


class NativeFHNDynamics(Dynamics[FHNDynamicsLog]):
    """Whole-brain FitzHugh-Nagumo dynamics -- pure NumPy implementation.

    Runs as a discrete-time step transition (``integrator=None``): each call to
    :meth:`dynamics` advances the OU noise and the delay history buffer in place,
    so it must be evaluated exactly once per ``dt``.
    """

    _ALPHA: float = 3.0
    _BETA: float = 4.0
    _GAMMA: float = -1.5
    _DELTA: float = 0.0
    _EPSILON: float = 0.5
    _TAU: float = 20.0
    _K_GL: float = 0.6
    _TAU_OU: float = 5.0
    _SIGNAL_V: float = 20.0
    _X_OU_MEAN: float = 0.0
    _Y_OU_MEAN: float = 0.0
    _X_EXT: float = 1.0
    _Y_EXT: float = 0.0

    def __init__(  # noqa: PLR0913
        self,
        dt: float = 0.1,
        sigma_ou: float = 0.05,
        connectome: str = "hcp",
        seed: int | None = None,
        b: ArrayLike | None = None,
        alpha: float = _ALPHA,
        beta: float = _BETA,
        gamma: float = _GAMMA,
        delta: float = _DELTA,
        epsilon: float = _EPSILON,
        tau: float = _TAU,
        k_gl: float = _K_GL,
        tau_ou: float = _TAU_OU,
        signal_v: float = _SIGNAL_V,
        x_ou_mean: float = _X_OU_MEAN,
        y_ou_mean: float = _Y_OU_MEAN,
        x_ext: float = _X_EXT,
        y_ext: float = _Y_EXT,
    ) -> None:
        super().__init__(dt, integrator=None)
        dataset = Dataset(connectome)
        self._dt = dt
        self._sigma_ou = sigma_ou
        self._seed = seed
        self._n_nodes = int(dataset.Cmat.shape[0])

        # Model parameters (overridable; class constants are the defaults).
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._delta = delta
        self._epsilon = epsilon
        self._tau = tau
        self._k_gl = k_gl
        self._tau_ou = tau_ou
        self._signal_v = signal_v
        self._x_ou_mean = x_ou_mean
        self._y_ou_mean = y_ou_mean
        self._x_ext = x_ext
        self._y_ext = y_ext

        cmat = dataset.Cmat.copy().astype(np.float64)
        np.fill_diagonal(cmat, 0.0)
        self._cmat = cmat

        dmat_ms = (dataset.Dmat / self._signal_v).astype(np.float64)
        dmat_ndt = np.around(dmat_ms / dt).astype(int)
        np.fill_diagonal(dmat_ndt, 0)
        self._dmat_ndt = dmat_ndt

        self._startind = int(dmat_ndt.max()) + 1
        self._src = np.arange(self._n_nodes)

        # Input gain B maps the control vector u onto the rate of change of V.
        self.b = np.atleast_2d(b).astype(np.float64) if b is not None else np.ones((self._n_nodes, 1), dtype=np.float64)

        self.reset()

    @property
    def sigma_ou(self) -> float:
        """OU noise amplitude."""
        return self._sigma_ou

    @property
    def n_nodes(self) -> int:
        """Number of network nodes."""
        return self._n_nodes

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the dynamics component from config."""
        return cls(
            dt=float(config["dt"]),
            sigma_ou=config.get("sigma_ou", 0.05),
            connectome=config.get("connectome", "hcp"),
            seed=config.get("seed"),
            b=config.get("b"),
            alpha=config.get("alpha", cls._ALPHA),
            beta=config.get("beta", cls._BETA),
            gamma=config.get("gamma", cls._GAMMA),
            delta=config.get("delta", cls._DELTA),
            epsilon=config.get("epsilon", cls._EPSILON),
            tau=config.get("tau", cls._TAU),
            k_gl=config.get("k_gl", cls._K_GL),
            tau_ou=config.get("tau_ou", cls._TAU_OU),
            signal_v=config.get("signal_v", cls._SIGNAL_V),
            x_ou_mean=config.get("x_ou_mean", cls._X_OU_MEAN),
            y_ou_mean=config.get("y_ou_mean", cls._Y_OU_MEAN),
            x_ext=config.get("x_ext", cls._X_EXT),
            y_ext=config.get("y_ext", cls._Y_EXT),
        )

    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """Evaluate one step of the native NumPy FHN dynamics."""
        v_prev = x[: self._n_nodes, :1]
        w_prev = x[self._n_nodes :, :1]

        # Sync history buffer with incoming state
        self._history_v[:, -1:] = v_prev
        self._history_w[:, -1:] = w_prev

        # Vectorised diffusive coupling
        delayed_indices = self._startind - self._dmat_ndt - 1
        delayed_v = self._history_v[self._src[np.newaxis, :], delayed_indices]
        vs_input = self._k_gl * (self._cmat * (delayed_v - v_prev)).sum(axis=1)

        # Control input B @ u
        u_vec = self.to_col_vec(u)
        vs_input_ext = (self.b @ u_vec).flatten()

        noise_v_val = self._rng.standard_normal(self._n_nodes)
        noise_w_val = self._rng.standard_normal(self._n_nodes)

        v_rhs = (
            -self._alpha * v_prev.flatten() ** 3
            + self._beta * v_prev.flatten() ** 2
            + self._gamma * v_prev.flatten()
            - w_prev.flatten()
            + vs_input
            + self._v_ou
            + self._x_ext
            + vs_input_ext
        )
        w_rhs = (
            (v_prev.flatten() - self._delta - self._epsilon * w_prev.flatten()) / self._tau + self._w_ou + self._y_ext
        )

        v_next = v_prev.flatten() + self._dt * v_rhs
        w_next = w_prev.flatten() + self._dt * w_rhs

        self._v_ou = (
            self._v_ou
            + (self._x_ou_mean - self._v_ou) * self._dt / self._tau_ou
            + self._sigma_ou * np.sqrt(self._dt) * noise_v_val
        )
        self._w_ou = (
            self._w_ou
            + (self._y_ou_mean - self._w_ou) * self._dt / self._tau_ou
            + self._sigma_ou * np.sqrt(self._dt) * noise_w_val
        )

        self._history_v = np.hstack([self._history_v[:, 1:], v_next.reshape(-1, 1)])
        self._history_w = np.hstack([self._history_w[:, 1:], w_next.reshape(-1, 1)])

        return np.vstack([v_next.reshape(-1, 1), w_next.reshape(-1, 1)])

    def _make_log(self) -> FHNDynamicsLog:
        return FHNDynamicsLog(v_ou=self._v_ou.copy(), w_ou=self._w_ou.copy())

    def reset(self) -> None:
        """Reset the dynamics simulator to its seeded initial state."""
        self.last_output = None
        self.last_log = None
        self.next_update_time = 0.0

        self._rng = np.random.default_rng(self._seed)
        vs_init = 0.05 * self._rng.uniform(0, 1, (self._n_nodes, 1))
        ws_init = 0.05 * self._rng.uniform(0, 1, (self._n_nodes, 1))
        self._v_ou = np.zeros(self._n_nodes, dtype=np.float64)
        self._w_ou = np.zeros(self._n_nodes, dtype=np.float64)
        self._history_v = (vs_init * np.ones((1, self._startind))).astype(np.float64)
        self._history_w = (ws_init * np.ones((1, self._startind))).astype(np.float64)

        self.x = np.vstack([vs_init, ws_init])


class TVBFHNDynamics(Dynamics[FHNDynamicsLog]):
    """Whole-brain FHN-like dynamics backed by The Virtual Brain (TVB).

    Runs as a discrete-time step transition (``integrator=None``): each call to
    :meth:`dynamics` advances the TVB simulator by one ``dt``.
    """

    def __init__(  # noqa: PLR0913
        self,
        dt: float = 0.1,
        nsig: float = 0.0,
        connectome: str = "connectivity_76.zip",
        seed: int | None = None,
        b: ArrayLike | None = None,
        model_params: dict[str, Any] | None = None,
        coupling: str = "Linear",
    ) -> None:
        super().__init__(dt, integrator=None)
        conn = _Connectivity.from_file(connectome)
        conn.configure()
        self._n_nodes = int(conn.number_of_regions)

        model = _Generic2dOscillator(**_as_tvb_params(model_params))
        noise = _tvb_noise.Additive(nsig=np.array([nsig]))
        integrator = _tvb_integrators.HeunStochastic(dt=dt, noise=noise)

        self._sim = _tvb_simulator.Simulator(
            connectivity=conn,
            model=model,
            coupling=getattr(_tvb_coupling, coupling)(),
            integrator=integrator,
            monitors=(_tvb_monitors.Raw(),),
        )
        self._sim.configure()
        self._sim.integrator.noise.random_stream.seed(seed)

        self._dt = dt
        self._seed = seed
        # Baseline of the injected external current, restored on reset().
        self._i_baseline = np.array(getattr(self._sim.model, "I"), dtype=np.float64).copy()  # noqa: B009

        # Input gain B maps the control vector u onto the model's external current I.
        self.b = np.atleast_2d(b).astype(np.float64) if b is not None else np.ones((self._n_nodes, 1), dtype=np.float64)

        self._initial_snapshot = self._snapshot()
        self.reset()

    @property
    def n_nodes(self) -> int:
        """Number of network nodes (regions) in the connectome."""
        return self._n_nodes

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the dynamics component from config."""
        return cls(
            dt=float(config["dt"]),
            nsig=config.get("nsig", 0.0),
            connectome=config.get("connectome", "connectivity_76.zip"),
            seed=config.get("seed"),
            b=config.get("b"),
            model_params=config.get("model_params"),
            coupling=config.get("coupling", "Linear"),
        )

    def _snapshot(self) -> _TVBSnapshot:
        history = self._sim.history
        if history is None:
            msg = "TVB history is None — configure() must be called first"
            raise RuntimeError(msg)
        return (
            self._sim.current_state.copy(),
            int(self._sim.current_step),
            history.buffer.copy(),
            self._sim.integrator.noise.random_stream.get_state(),
        )

    def _restore(self, snap: _TVBSnapshot) -> None:
        state, step, buf, rng_state = snap
        history = self._sim.history
        if history is None:
            msg = "TVB history is None — configure() must be called first"
            raise RuntimeError(msg)
        self._sim.current_state = state.copy()
        self._sim.current_step = step
        history.buffer[:] = buf
        self._sim.integrator.noise.random_stream.set_state(rng_state)

    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """Evaluate one step of the TVB FHN dynamics."""
        u_vec = self.to_col_vec(u)
        setattr(self._sim.model, "I", self._i_baseline + (self.b @ u_vec).flatten())  # noqa: B010

        next(self._sim(simulation_length=self._dt))
        v_new = self._sim.current_state[0, :, -1:]
        w_new = self._sim.current_state[1, :, -1:]
        return np.vstack([v_new, w_new])

    def _make_log(self) -> FHNDynamicsLog:
        return FHNDynamicsLog()

    def reset(self) -> None:
        """Reset the TVB simulator state and the injected external current."""
        self._restore(self._initial_snapshot)
        setattr(self._sim.model, "I", self._i_baseline.copy())  # noqa: B010
        self.last_output = None
        self.last_log = None
        self.next_update_time = 0.0

        v_init = self._sim.current_state[0, :, -1:]
        w_init = self._sim.current_state[1, :, -1:]
        self.x = np.vstack([v_init, w_init])
