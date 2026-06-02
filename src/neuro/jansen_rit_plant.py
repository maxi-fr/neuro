"""Whole-brain Jansen-Rit plant components for the simulate framework.

This module provides :class:`TVBJansenRitDynamics` and :class:`TVBJansenRitOutput`,
neural-mass simulator components built on TVB's :class:`~tvb.simulator.models.jansen_rit.JansenRit` model.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Self, cast

import numpy as np
from pydantic import BaseModel, ConfigDict
from simulate.dynamics import Dynamics
from simulate.output import Output

from neuro.plant import FloatArray, _as_tvb_params, _tvb_eeg_leadfield, _TVBSnapshot

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    from tvb.datatypes.connectivity import Connectivity as _Connectivity
    from tvb.simulator import coupling as _tvb_coupling
    from tvb.simulator import integrators as _tvb_integrators
    from tvb.simulator import monitors as _tvb_monitors
    from tvb.simulator import noise as _tvb_noise
    from tvb.simulator import simulator as _tvb_simulator
    from tvb.simulator.models.jansen_rit import JansenRit as _JansenRit


# --- Logging schemas --------------------------------------------------------


class TVBJansenRitDynamicsLog(BaseModel):
    """Pydantic model for internal TVB Jansen-Rit dynamics logging."""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TVBJansenRitOutputLog(BaseModel):
    """Pydantic model for TVB Jansen-Rit EEG output logging."""

    model_config = ConfigDict(arbitrary_types_allowed=True)


# --- Output Class -----------------------------------------------------------


class TVBJansenRitOutput(Output[TVBJansenRitOutputLog]):
    """EEG output component for TVB Jansen-Rit plants."""

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
        """The region-averaged EEG lead-field matrix."""
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
    ) -> tuple[FloatArray, TVBJansenRitOutputLog]:
        """Compute the EEG projection y from state x."""
        x_vec = self.to_col_vec(x)
        # x contains all 6 state variables for each of the n_nodes regions, shape (6 * n_nodes, 1)
        n_nodes = x_vec.shape[0] // 6
        x_grid = x_vec.reshape((6, n_nodes))
        activity = x_grid[1, :] - x_grid[2, :]  # y1 - y2 pyramidal potential
        y_vec = self._leadfield @ activity.reshape((-1, 1))
        return cast("FloatArray", self.from_col_vec(y_vec)), TVBJansenRitOutputLog()


# --- Dynamics Class ---------------------------------------------------------


class TVBJansenRitDynamics(Dynamics[TVBJansenRitDynamicsLog]):
    """Whole-brain Jansen-Rit dynamics backed by The Virtual Brain (TVB).

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
        coupling_strength: float = 1.0,
    ) -> None:
        super().__init__(dt, integrator=None)
        conn = _Connectivity.from_file(connectome)
        conn.configure()
        self._n_nodes = int(conn.number_of_regions)

        model = _JansenRit(**_as_tvb_params(model_params))
        model.variables_of_interest = ("y1", "y2")

        # Inject additive noise only into the excitatory input pathway (state
        # variable y4, which carries the mean input mu) -- i.e. a "noisy input".
        # Noising all six state variables instead makes the slow potentials
        # (y0, y1, y2) random-walk, swamping the rhythm with low-frequency drift.
        nsig_vec = np.zeros(int(model.nvar), dtype=np.float64)
        nsig_vec[4] = nsig
        noise = _tvb_noise.Additive(nsig=nsig_vec)
        integrator = _tvb_integrators.HeunStochastic(dt=dt, noise=noise)

        # `a` is the global coupling gain: SigmoidalJansenRit.post(gx) = a * gx.
        # Setting it to 0 decouples the nodes (used for single-node bifurcation sweeps).
        self._sim = _tvb_simulator.Simulator(
            connectivity=conn,
            model=model,
            coupling=_tvb_coupling.SigmoidalJansenRit(a=np.array([coupling_strength])),
            integrator=integrator,
            monitors=(_tvb_monitors.Raw(),),
        )
        self._sim.configure()
        self._sim.integrator.noise.random_stream.seed(seed)

        self._dt = dt
        self._seed = seed
        # Baseline mean input firing rate, restored on reset() and used as the
        # set-point that the control input is added to.
        self._mu_baseline = np.array(getattr(self._sim.model, "mu"), dtype=np.float64).copy()  # noqa: B009

        # Input gain B maps the control vector u onto the model's mean input mu.
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
            coupling_strength=config.get("coupling_strength", 1.0),
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
        """Evaluate one step of the Jansen-Rit dynamics."""
        u_vec = self.to_col_vec(u)
        setattr(self._sim.model, "mu", self._mu_baseline + (self.b @ u_vec).flatten())  # noqa: B010

        next(self._sim(simulation_length=self._dt))
        state_new = self._sim.current_state[:, :, -1]
        return state_new.reshape((-1, 1))

    def _make_log(self) -> TVBJansenRitDynamicsLog:
        return TVBJansenRitDynamicsLog()

    def reset(self) -> None:
        """Reset the TVB simulator state and the injected mean input."""
        self._restore(self._initial_snapshot)
        setattr(self._sim.model, "mu", self._mu_baseline.copy())  # noqa: B010
        self.last_output = None
        self.last_log = None
        self.next_update_time = 0.0

        state_init = self._sim.current_state[:, :, -1]
        self.x = state_init.reshape((-1, 1))
