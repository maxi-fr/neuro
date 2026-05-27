"""FitzHugh-Nagumo whole-brain plant models built on (or matching) neurolib.

Two implementations are provided:

* :class:`FHNPlant` -- thin wrapper around :class:`neurolib.models.fhn.FHNModel`.
* :class:`NativeFHNPlant` -- pure NumPy reimplementation of the same dynamics.
  It loads the HCP structural connectome via neurolib's :class:`Dataset` but
  performs all integration internally, matching neurolib's default parameters
  and seeding strategy so both classes produce equivalent outputs when
  constructed with the same ``seed``.

Both classes expose an identical step-by-step API plus a synthetic EEG
lead-field projection.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from neurolib.models.fhn import FHNModel
from neurolib.utils.loadData import Dataset

FloatArray = npt.NDArray[np.float64]


def _make_leadfield(n_sensors: int, n_nodes: int, seed: int) -> FloatArray:
    """Create a row-normalised random lead-field matrix.

    Parameters
    ----------
    n_sensors
        Number of EEG channels (rows).
    n_nodes
        Number of brain nodes (columns).
    seed
        Seed for the deterministic default_rng.
    """
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((n_sensors, n_nodes))
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix.astype(np.float64)


class FHNPlant:
    """Whole-brain FitzHugh-Nagumo plant with step-by-step simulation and EEG output.

    Parameters
    ----------
    dt
        Integration step in milliseconds.
    sigma_ou
        Amplitude of the per-node Ornstein-Uhlenbeck noise driving the FHN
        dynamics. With ``sigma_ou = 0`` the network is deterministic.
    n_sensors
        Number of synthetic EEG channels produced by the lead-field projection.
    leadfield_seed
        Seed for the deterministic random lead-field matrix.
    connectome
        Name of the built-in neurolib dataset providing ``Cmat`` (structural
        connectivity) and ``Dmat`` (fiber lengths). ``"hcp"`` gives 80 nodes.
    seed
        Seed passed to the underlying neurolib RNG for initial conditions and
        OU noise. ``None`` (default) lets neurolib pick from system entropy.

    Notes
    -----
    Each call to :meth:`step` returns only the *most recent* chunk of activity
    (not the accumulated trajectory). Callers that need full histories must
    concatenate the per-step outputs themselves.

    The first underlying neurolib ``run`` must be called without
    ``continue_run`` to initialise the integrator state; subsequent calls use
    ``continue_run=True`` to carry state forward.
    """

    def __init__(  # noqa: PLR0913
        self,
        dt: float = 0.1,
        sigma_ou: float = 0.05,
        n_sensors: int = 64,
        leadfield_seed: int = 0,
        connectome: str = "hcp",
        seed: int | None = None,
    ) -> None:
        dataset = Dataset(connectome)
        self._model = FHNModel(Cmat=dataset.Cmat, Dmat=dataset.Dmat, seed=seed)
        self._model.params["dt"] = dt
        self._model.params["sigma_ou"] = sigma_ou

        self._dt: float = dt
        self._sigma_ou: float = sigma_ou
        self._n_nodes: int = int(dataset.Cmat.shape[0])
        self._leadfield: FloatArray = _make_leadfield(n_sensors, self._n_nodes, leadfield_seed)
        self._initialized: bool = False

    @property
    def dt(self) -> float:
        """Integration step in milliseconds."""
        return self._dt

    @property
    def sigma_ou(self) -> float:
        """OU noise amplitude used by the underlying model."""
        return self._sigma_ou

    @property
    def n_nodes(self) -> int:
        """Number of network nodes (regions) in the connectome."""
        return self._n_nodes

    @property
    def leadfield(self) -> FloatArray:
        """Synthetic EEG lead-field matrix, shape ``(n_sensors, n_nodes)``."""
        return self._leadfield

    def step(self, duration_ms: float) -> tuple[FloatArray, FloatArray]:
        """Advance the simulation by ``duration_ms`` and return the resulting chunk.

        Parameters
        ----------
        duration_ms
            Length of the simulation chunk in milliseconds.

        Returns
        -------
        activity
            Node activity for this chunk, shape ``(n_nodes, n_samples)`` with
            ``n_samples = round(duration_ms / dt)``.
        eeg
            EEG projection ``leadfield @ activity``, shape
            ``(n_sensors, n_samples)``.
        """
        self._model.params["duration"] = duration_ms
        if self._initialized:
            self._model.run(continue_run=True)
        else:
            self._model.run()
            # Propagate the final state into params so the next continue_run
            # starts from the correct history window, not the original N x 1 ICs.
            self._model.setInitialValuesToLastState()
            self._initialized = True
        activity: FloatArray = np.asarray(self._model.x, dtype=np.float64)  # type: ignore
        eeg: FloatArray = self._leadfield @ activity
        return activity, eeg

    def reset(self) -> None:
        """Drop accumulated state so the next :meth:`step` re-initialises the model."""
        self._initialized = False


class NativeFHNPlant:
    """Whole-brain FitzHugh-Nagumo plant -- pure NumPy reimplementation.

    Identical API to :class:`FHNPlant` (plus a ``seed`` parameter shared with
    that class). Uses the same default parameters and seeding strategy as the
    neurolib FHN model so outputs from both classes agree to near machine
    precision when constructed with the same arguments.

    FHN equations (per node *no*)::

        du/dt = -alpha*u^3 + beta*u^2 + gamma*u - w + xs_input[no] + x_ou[no] + x_ext
        dw/dt = (u - delta - epsilon*w) / tau + y_ou[no] + y_ext

    Coupling (diffusive, neurolib default)::

        xs_input[no] = K_gl * sum_l( Cmat[no,l] * (u_l(t-tau_l) - u_no(t-1)) )

    Ornstein-Uhlenbeck noise::

        x_ou += (0 - x_ou) * dt/tau_ou + sigma_ou * sqrt(dt) * xi_x
        y_ou += (0 - y_ou) * dt/tau_ou + sigma_ou * sqrt(dt) * xi_y

    Integration: forward Euler.

    Parameters
    ----------
    dt, sigma_ou, n_sensors, leadfield_seed, connectome
        Same meaning as in :class:`FHNPlant`.
    seed
        Seed for the legacy ``np.random`` global RNG.  Replicates neurolib's
        two-phase seeding: once before generating initial conditions (uniform),
        then reset before pre-generating the OU noise (normal).
        ``None`` (default) draws from system entropy.
    """

    # --- neurolib FHN default parameters (do not change) ---------------------
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
    _X_EXT: float = 1.0  # neurolib default: x_ext = np.ones((N,))
    _Y_EXT: float = 0.0  # neurolib default: y_ext = np.zeros((N,))

    def __init__(  # noqa: PLR0913
        self,
        dt: float = 0.1,
        sigma_ou: float = 0.05,
        n_sensors: int = 64,
        leadfield_seed: int = 0,
        connectome: str = "hcp",
        seed: int | None = None,
    ) -> None:
        dataset = Dataset(connectome)

        self._dt: float = dt
        self._sigma_ou: float = sigma_ou
        self._seed: int | None = seed

        n_nodes = int(dataset.Cmat.shape[0])
        self._n_nodes: int = n_nodes

        # Structural connectivity -- zero self-connections (matches neurolib)
        cmat: FloatArray = dataset.Cmat.copy().astype(np.float64)
        np.fill_diagonal(cmat, 0.0)
        self._cmat: FloatArray = cmat

        # Delay matrix in integer timesteps:
        #   Dmat[i,j] in mm / signalV (mm/ms) -> ms / dt -> timesteps
        dmat_ms: FloatArray = (dataset.Dmat / self._SIGNAL_V).astype(np.float64)
        dmat_ndt = np.around(dmat_ms / dt).astype(int)
        np.fill_diagonal(dmat_ndt, 0)
        self._dmat_ndt: npt.NDArray[np.int_] = dmat_ndt

        self._startind: int = int(dmat_ndt.max()) + 1
        # Source-node index vector for advanced indexing in the coupling step
        self._src: npt.NDArray[np.int_] = np.arange(n_nodes)

        self._leadfield: FloatArray = _make_leadfield(n_sensors, n_nodes, leadfield_seed)

        # Mutable simulation state -- None until the first step()
        self._initialized: bool = False
        self._history_x: FloatArray | None = None  # (n_nodes, startind)
        self._history_y: FloatArray | None = None
        self._x_ou: FloatArray | None = None  # (n_nodes,)
        self._y_ou: FloatArray | None = None

    @property
    def dt(self) -> float:
        """Integration step in milliseconds."""
        return self._dt

    @property
    def sigma_ou(self) -> float:
        """OU noise amplitude."""
        return self._sigma_ou

    @property
    def n_nodes(self) -> int:
        """Number of network nodes."""
        return self._n_nodes

    @property
    def leadfield(self) -> FloatArray:
        """Lead-field matrix, shape ``(n_sensors, n_nodes)``."""
        return self._leadfield

    def _init_chunk(
        self, n_steps: int
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        """Return (history_x, history_y, x_ou, y_ou, noise_x, noise_y) for one chunk.

        On the first call, initial conditions are drawn from
        ``Uniform(0, 0.05)`` (replicating neurolib's ``loadDefaultParams``).
        The OU noise for the chunk is pre-generated by resetting the global RNG
        seed before every call -- exactly as neurolib's ``timeIntegration`` does.
        """
        n_nodes = self._n_nodes
        startind = self._startind

        if not self._initialized:
            # Phase 1: initial conditions
            np.random.seed(self._seed)  # noqa: NPY002
            xs_init = 0.05 * np.random.uniform(0, 1, (n_nodes, 1))  # noqa: NPY002
            ys_init = 0.05 * np.random.uniform(0, 1, (n_nodes, 1))  # noqa: NPY002
            x_ou = np.zeros(n_nodes, dtype=np.float64)
            y_ou = np.zeros(n_nodes, dtype=np.float64)
            # Broadcast IC across history window (matches neurolib's np.dot broadcast)
            history_x: FloatArray = (xs_init * np.ones((1, startind))).astype(np.float64)
            history_y: FloatArray = (ys_init * np.ones((1, startind))).astype(np.float64)
            self._initialized = True
        else:
            if self._history_x is None or self._history_y is None or self._x_ou is None or self._y_ou is None:
                msg = "State is None despite _initialized=True"
                raise RuntimeError(msg)
            history_x = self._history_x
            history_y = self._history_y
            x_ou = self._x_ou
            y_ou = self._y_ou

        # Phase 2: pre-generate noise
        # Seed is reset before every chunk -- replicates neurolib's timeIntegration.
        np.random.seed(self._seed)  # noqa: NPY002
        noise_x: FloatArray = np.random.standard_normal((n_nodes, n_steps))  # noqa: NPY002
        noise_y: FloatArray = np.random.standard_normal((n_nodes, n_steps))  # noqa: NPY002
        return history_x, history_y, x_ou, y_ou, noise_x, noise_y

    def step(self, duration_ms: float) -> tuple[FloatArray, FloatArray]:
        """Advance the simulation by ``duration_ms`` and return the chunk.

        Parameters
        ----------
        duration_ms
            Length of the simulation chunk in milliseconds.

        Returns
        -------
        activity
            Node activity, shape ``(n_nodes, n_samples)``.
        eeg
            EEG projection, shape ``(n_sensors, n_samples)``.
        """
        n_steps = round(duration_ms / self._dt)
        startind = self._startind
        history_x, history_y, x_ou, y_ou, noise_x, noise_y = self._init_chunk(n_steps)

        # Allocate full state buffer: [history | new steps]
        xs = np.empty((self._n_nodes, startind + n_steps), dtype=np.float64)
        ys = np.empty((self._n_nodes, startind + n_steps), dtype=np.float64)
        xs[:, :startind] = history_x
        ys[:, :startind] = history_y

        dt = self._dt
        sqrt_dt = np.sqrt(dt)
        cmat = self._cmat
        dmat_ndt = self._dmat_ndt
        src = self._src

        for i in range(startind, startind + n_steps):
            idx = i - startind

            # Vectorised diffusive coupling.
            # delayed_x shape (n_nodes, n_nodes): source-node activity at each pair's delay.
            delayed_indices = i - dmat_ndt - 1  # (n_nodes, n_nodes)
            delayed_x = xs[src[np.newaxis, :], delayed_indices]  # (n_nodes, n_nodes)
            # Each entry: delayed source minus current target, weighted by Cmat.
            xs_input = self._K_GL * (cmat * (delayed_x - xs[:, i - 1 : i])).sum(axis=1)

            x_prev = xs[:, i - 1]
            y_prev = ys[:, i - 1]

            x_rhs = (
                -self._ALPHA * x_prev**3
                + self._BETA * x_prev**2
                + self._GAMMA * x_prev
                - y_prev
                + xs_input
                + x_ou
                + self._X_EXT
            )
            y_rhs = (x_prev - self._DELTA - self._EPSILON * y_prev) / self._TAU + y_ou + self._Y_EXT

            xs[:, i] = x_prev + dt * x_rhs
            ys[:, i] = y_prev + dt * y_rhs

            # OU update -- uses pre-stored noise (matches neurolib's timing)
            x_ou = x_ou + (self._X_OU_MEAN - x_ou) * dt / self._TAU_OU + self._sigma_ou * sqrt_dt * noise_x[:, idx]
            y_ou = y_ou + (self._Y_OU_MEAN - y_ou) * dt / self._TAU_OU + self._sigma_ou * sqrt_dt * noise_y[:, idx]

        self._history_x = xs[:, -startind:]
        self._history_y = ys[:, -startind:]
        self._x_ou = x_ou
        self._y_ou = y_ou

        activity: FloatArray = xs[:, startind:].copy()
        eeg: FloatArray = self._leadfield @ activity
        return activity, eeg

    def reset(self) -> None:
        """Drop accumulated state so the next :meth:`step` re-initialises."""
        self._initialized = False
        self._history_x = None
        self._history_y = None
        self._x_ou = None
        self._y_ou = None
