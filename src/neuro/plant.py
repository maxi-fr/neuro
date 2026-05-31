"""FitzHugh-Nagumo whole-brain plant models.

Three implementations are provided:

* :class:`FHNPlant` -- thin wrapper around :class:`neurolib.models.fhn.FHNModel`.
* :class:`NativeFHNPlant` -- pure NumPy reimplementation of the same dynamics.
  It loads the HCP structural connectome via neurolib's :class:`Dataset` but
  performs all integration internally, matching neurolib's default parameters
  and seeding strategy so both classes produce equivalent outputs when
  constructed with the same ``seed``.
* :class:`TVBFHNPlant` -- backed by `The Virtual Brain (TVB) <https://www.thevirtualbrain.org>`_.
  Uses TVB's ``Generic2dOscillator`` model (FHN limit-cycle regime) with
  ``HeunStochastic`` integration and additive Gaussian noise.  Shares the same
  HCP structural connectome but is **not** numerically comparable to the
  neurolib plants: the FHN equations, integrators, and coupling functions
  differ.

All three classes expose the same step-by-step API plus a synthetic EEG
lead-field projection.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from neurolib.models.fhn import FHNModel
from neurolib.utils.loadData import Dataset

if TYPE_CHECKING:
    from pathlib import Path

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


def _load_leadfield(path: str | Path, n_nodes: int) -> FloatArray:
    """Load a precomputed lead-field matrix and validate its shape.

    Parameters
    ----------
    path
        Path to a NumPy ``.npy`` file holding a 2-D ``(n_sensors, n_nodes)``
        lead-field matrix, e.g. one produced by
        ``scripts/generate_leadfield.py``.
    n_nodes
        Number of network nodes the plant simulates.  The loaded matrix must
        have exactly this many columns so ``leadfield @ activity`` is defined.

    Returns
    -------
    matrix
        The loaded lead-field, shape ``(n_sensors, n_nodes)``.

    Raises
    ------
    ValueError
        If the loaded array is not 2-D or its column count differs from
        ``n_nodes``.  The neurolib HCP connectome and the AAL2-cortical
        lead-field from ``scripts/generate_leadfield.py`` share the same
        80-region parcellation and node ordering, so an aligned lead-field
        loads directly onto the default ``connectome="hcp"`` plant.
    """
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
    """Build a region-level EEG gain matrix from TVB's surface projection.

    TVB ships a BEM-derived forward solution mapping a 16k-vertex cortical
    surface to 65 EEG sensors, plus a region-mapping file that assigns each
    vertex to one of the AAL76 atlas regions.  This helper averages the
    surface gain across the vertices of each region to obtain a region-level
    leadfield of shape ``(n_sensors, n_regions)``.

    Parameters
    ----------
    n_regions
        Number of regions in the target parcellation (must match the
        ``region_mapping_file``).
    projection_file
        Filename of the TVB surface projection (looked up in
        ``tvb_data.projectionMatrix``).
    region_mapping_file
        Filename of the TVB region mapping (looked up in
        ``tvb_data.regionMapping``).  Region ids must be integers in
        ``[0, n_regions)``.

    Returns
    -------
    gain
        Region-averaged leadfield matrix, shape ``(n_sensors, n_regions)``.

    Notes
    -----
    The default TVB projection file has two sensors (indices 18 and 19) whose
    gain values are NaN for every vertex.  Those rows are zero-filled here so
    the resulting matrix is finite and safe to use in ``leadfield @ activity``.
    """
    proj = _ProjectionSurfaceEEG.from_file(projection_file)
    rmap = _RegionMapping.from_file(region_mapping_file).array_data
    surface_gain = np.asarray(proj.projection_data, dtype=np.float64)
    n_sensors = surface_gain.shape[0]
    gain = np.zeros((n_sensors, n_regions), dtype=np.float64)
    with warnings.catch_warnings():
        # All-NaN sensor rows (18, 19 in the default file) trigger
        # "Mean of empty slice" — they end up NaN and are zero-filled below.
        warnings.simplefilter("ignore", RuntimeWarning)
        for r in range(n_regions):
            verts = np.where(rmap == r)[0]
            if verts.size:
                gain[:, r] = np.nanmean(surface_gain[:, verts], axis=1)
    return np.nan_to_num(gain, nan=0.0)


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
        Seed for the deterministic random lead-field matrix.  Ignored when
        ``leadfield_path`` is given.
    leadfield_path
        Optional path to a precomputed ``.npy`` lead-field (e.g. from
        ``scripts/generate_leadfield.py``).  When set it replaces the random
        matrix and ``n_sensors`` / ``leadfield_seed`` are ignored; the file's
        column count must equal the number of connectome nodes.
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
        leadfield_path: str | Path | None = None,
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
        self._leadfield: FloatArray = (
            _load_leadfield(leadfield_path, self._n_nodes)
            if leadfield_path is not None
            else _make_leadfield(n_sensors, self._n_nodes, leadfield_seed)
        )
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
        """EEG lead-field matrix, shape ``(n_sensors, n_nodes)``.

        A random row-normalised matrix by default.  Pass ``leadfield_path`` at
        construction to load a real anatomical lead-field instead (see
        ``scripts/generate_leadfield.py``, whose AAL2-cortical output is aligned
        to the same 80-region HCP parcellation); its column count must match the
        connectome node count.
        """
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

        dv/dt = -alpha*v^3 + beta*v^2 + gamma*v - w + vs_input[no] + v_ou[no] + v_ext
        dw/dt = (v - delta - epsilon*w) / tau + w_ou[no] + w_ext

    Coupling (diffusive, neurolib default)::

        vs_input[no] = K_gl * sum_l( Cmat[no,l] * (v_l(t-tau_l) - v_no(t-1)) )

    Ornstein-Uhlenbeck noise::

        v_ou += (0 - v_ou) * dt/tau_ou + sigma_ou * sqrt(dt) * xi_v
        w_ou += (0 - w_ou) * dt/tau_ou + sigma_ou * sqrt(dt) * xi_w

    Integration: forward Euler.

    Parameters
    ----------
    dt, sigma_ou, n_sensors, leadfield_seed, leadfield_path, connectome
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
        leadfield_path: str | Path | None = None,
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

        self._leadfield: FloatArray = (
            _load_leadfield(leadfield_path, n_nodes)
            if leadfield_path is not None
            else _make_leadfield(n_sensors, n_nodes, leadfield_seed)
        )

        # Mutable simulation state -- None until the first step()
        self._initialized: bool = False
        self._history_v: FloatArray | None = None  # (n_nodes, startind)
        self._history_w: FloatArray | None = None
        self._v_ou: FloatArray | None = None  # (n_nodes,)
        self._w_ou: FloatArray | None = None

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
        """EEG lead-field matrix, shape ``(n_sensors, n_nodes)``.

        A random row-normalised matrix by default.  Pass ``leadfield_path`` at
        construction to load a real anatomical lead-field instead (see
        ``scripts/generate_leadfield.py``, whose AAL2-cortical output is aligned
        to the same 80-region HCP parcellation); its column count must match the
        connectome node count.
        """
        return self._leadfield

    def _init_chunk(
        self, n_steps: int
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        """Return (history_v, history_w, v_ou, w_ou, noise_v, noise_w) for one chunk.

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
            vs_init = 0.05 * np.random.uniform(0, 1, (n_nodes, 1))  # noqa: NPY002
            ws_init = 0.05 * np.random.uniform(0, 1, (n_nodes, 1))  # noqa: NPY002
            v_ou = np.zeros(n_nodes, dtype=np.float64)
            w_ou = np.zeros(n_nodes, dtype=np.float64)
            # Broadcast IC across history window (matches neurolib's np.dot broadcast)
            history_v: FloatArray = (vs_init * np.ones((1, startind))).astype(np.float64)
            history_w: FloatArray = (ws_init * np.ones((1, startind))).astype(np.float64)
            self._initialized = True
        else:
            if self._history_v is None or self._history_w is None or self._v_ou is None or self._w_ou is None:
                msg = "State is None despite _initialized=True"
                raise RuntimeError(msg)
            history_v = self._history_v
            history_w = self._history_w
            v_ou = self._v_ou
            w_ou = self._w_ou

        # Phase 2: pre-generate noise
        # Seed is reset before every chunk -- replicates neurolib's timeIntegration.
        np.random.seed(self._seed)  # noqa: NPY002
        noise_v: FloatArray = np.random.standard_normal((n_nodes, n_steps))  # noqa: NPY002
        noise_w: FloatArray = np.random.standard_normal((n_nodes, n_steps))  # noqa: NPY002
        return history_v, history_w, v_ou, w_ou, noise_v, noise_w

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
        history_v, history_w, v_ou, w_ou, noise_v, noise_w = self._init_chunk(n_steps)

        # Allocate full state buffer: [history | new steps]
        vs = np.empty((self._n_nodes, startind + n_steps), dtype=np.float64)
        ws = np.empty((self._n_nodes, startind + n_steps), dtype=np.float64)
        vs[:, :startind] = history_v
        ws[:, :startind] = history_w

        dt = self._dt
        sqrt_dt = np.sqrt(dt)
        cmat = self._cmat
        dmat_ndt = self._dmat_ndt
        src = self._src

        for i in range(startind, startind + n_steps):
            idx = i - startind

            # Vectorised diffusive coupling.
            # delayed_v shape (n_nodes, n_nodes): source-node activity at each pair's delay.
            delayed_indices = i - dmat_ndt - 1  # (n_nodes, n_nodes)
            delayed_v = vs[src[np.newaxis, :], delayed_indices]  # (n_nodes, n_nodes)
            # Each entry: delayed source minus current target, weighted by Cmat.
            vs_input = self._K_GL * (cmat * (delayed_v - vs[:, i - 1 : i])).sum(axis=1)

            v_prev = vs[:, i - 1]
            w_prev = ws[:, i - 1]

            v_rhs = (
                -self._ALPHA * v_prev**3
                + self._BETA * v_prev**2
                + self._GAMMA * v_prev
                - w_prev
                + vs_input
                + v_ou
                + self._X_EXT
            )
            w_rhs = (v_prev - self._DELTA - self._EPSILON * w_prev) / self._TAU + w_ou + self._Y_EXT

            vs[:, i] = v_prev + dt * v_rhs
            ws[:, i] = w_prev + dt * w_rhs

            # OU update -- uses pre-stored noise (matches neurolib's timing)
            v_ou = v_ou + (self._X_OU_MEAN - v_ou) * dt / self._TAU_OU + self._sigma_ou * sqrt_dt * noise_v[:, idx]
            w_ou = w_ou + (self._Y_OU_MEAN - w_ou) * dt / self._TAU_OU + self._sigma_ou * sqrt_dt * noise_w[:, idx]

        self._history_v = vs[:, -startind:]
        self._history_w = ws[:, -startind:]
        self._v_ou = v_ou
        self._w_ou = w_ou

        activity: FloatArray = vs[:, startind:].copy()
        eeg: FloatArray = self._leadfield @ activity
        return activity, eeg

    def reset(self) -> None:
        """Drop accumulated state so the next :meth:`step` re-initialises."""
        self._initialized = False
        self._history_v = None
        self._history_w = None
        self._v_ou = None
        self._w_ou = None


_TVBSnapshot = tuple[
    FloatArray,  # current_state
    int,  # current_step
    FloatArray,  # history buffer
    object,  # RNG state (numpy RandomState tuple)
]


class TVBFHNPlant:
    """Whole-brain FHN-like plant backed by The Virtual Brain (TVB).

    Uses TVB's ``Generic2dOscillator`` model (FHN limit-cycle regime with
    default parameters ``a=-2.0``, ``b=-10.0``) integrated with
    ``HeunStochastic`` and additive Gaussian noise of amplitude ``nsig``.
    Connectivity is loaded exclusively from TVB's built-in datasets (requires
    the ``tvb-data`` package).

    The dynamics are **not** numerically comparable to :class:`FHNPlant` or
    :class:`NativeFHNPlant` because the FHN equations, integrators, and
    coupling functions all differ.

    Parameters
    ----------
    dt
        Integration step in milliseconds.
    nsig
        Noise amplitude for TVB's additive Gaussian noise, applied uniformly
        to both state variables (V and W).  ``nsig=0`` gives a fully
        deterministic simulation.  Unlike ``sigma_ou`` in the neurolib plants,
        this is i.i.d. Gaussian — not an Ornstein-Uhlenbeck process.
    connectome
        Filename of a TVB-format connectivity zip (looked up in the installed
        ``tvb_data.connectivity`` package directory).  Defaults to
        ``"connectivity_76.zip"``, TVB's standard 76-node AAL atlas.
        Other built-in options: ``"connectivity_68.zip"`` (68 nodes,
        Desikan-Killiany), ``"connectivity_96.zip"`` (96 nodes).  Must be
        paired with a ``region_mapping`` whose region ids cover the same
        parcellation.
    projection
        Filename of the TVB surface EEG projection matrix (looked up in
        ``tvb_data.projectionMatrix``).  Default ``"projection_eeg_65_surface_16k.npy"``
        is a 65-sensor BEM forward solution on a 16k-vertex cortical surface.
    region_mapping
        Filename of the TVB region mapping (looked up in
        ``tvb_data.regionMapping``) used to average the surface gain to the
        region level.  Default ``"regionMapping_16k_76.txt"`` matches the
        default ``"connectivity_76.zip"`` AAL76 atlas.
    seed
        Seed for the ``numpy.random.RandomState`` used by the HeunStochastic
        integrator.  ``None`` (default) gives non-reproducible noise.

    Notes
    -----
    The lead-field is a real region-averaged BEM forward solution shipped
    with ``tvb-data``: TVB's 65-sensor surface projection averaged across the
    vertices of each region in ``region_mapping``.  The resulting matrix has
    shape ``(65, n_nodes)``.  Callers must ensure ``connectome`` and
    ``region_mapping`` describe the same parcellation; otherwise the gain
    rows for unmapped regions will be zero.

    State is carried forward across :meth:`step` calls through TVB's internal
    history buffer.  :meth:`reset` restores the exact initial conditions and
    RNG state set at construction time, so a reset followed by identically
    parameterised step calls reproduces the original trajectory.
    """

    def __init__(  # noqa: PLR0913
        self,
        dt: float = 0.1,
        nsig: float = 0.0,
        connectome: str = "connectivity_76.zip",
        projection: str = "projection_eeg_65_surface_16k.npy",
        region_mapping: str = "regionMapping_16k_76.txt",
        seed: int | None = None,
    ) -> None:
        conn = _Connectivity.from_file(connectome)
        conn.configure()
        n_nodes = int(conn.number_of_regions)

        model = _Generic2dOscillator()

        # nsig shape (1,) passes through TVB's _configure_integrator_noise
        # without reshaping (it is explicitly allowed as a scalar broadcast).
        noise = _tvb_noise.Additive(nsig=np.array([nsig]))
        integrator = _tvb_integrators.HeunStochastic(dt=dt, noise=noise)

        self._sim = _tvb_simulator.Simulator(
            connectivity=conn,
            model=model,
            coupling=_tvb_coupling.Linear(),
            integrator=integrator,
            monitors=(_tvb_monitors.Raw(),),
        )
        self._sim.configure()

        self._sim.integrator.noise.random_stream.seed(seed)

        self._dt = dt
        self._n_nodes = n_nodes
        self._leadfield: FloatArray = _tvb_eeg_leadfield(n_nodes, projection, region_mapping)

        self._initial_snapshot: _TVBSnapshot = self._snapshot()

    @property
    def dt(self) -> float:
        """Integration step in milliseconds."""
        return self._dt

    @property
    def n_nodes(self) -> int:
        """Number of network nodes (regions) in the connectome."""
        return self._n_nodes

    @property
    def leadfield(self) -> FloatArray:
        """Region-averaged EEG lead-field matrix, shape ``(n_sensors, n_nodes)``.

        Real BEM-derived forward solution shipped with ``tvb-data``: TVB's
        surface EEG projection averaged across the vertices of each region
        in ``region_mapping``.
        """
        return self._leadfield

    def step(self, duration_ms: float) -> tuple[FloatArray, FloatArray]:
        """Advance the simulation by ``duration_ms`` and return the chunk.

        Parameters
        ----------
        duration_ms
            Length of the simulation chunk in milliseconds.

        Returns
        -------
        activity
            Node activity (state variable V), shape ``(n_nodes, n_samples)``
            with ``n_samples = round(duration_ms / dt)``.
        eeg
            EEG projection ``leadfield @ activity``, shape
            ``(n_sensors, n_samples)``.
        """
        batches: list[FloatArray] = []
        for output in self._sim(simulation_length=duration_ms):
            # output = [(time_scalar, data_array)] for one Raw monitor.
            # data_array shape: (1, n_nodes, 1) — first monitored state var
            # (V) across all nodes and one mode.
            _, data = output[0]
            batches.append(data[0, :, 0].astype(np.float64))  # (n_nodes,)
        activity: FloatArray = np.stack(batches, axis=1)  # (n_nodes, n_samples)
        eeg: FloatArray = self._leadfield @ activity
        return activity, eeg

    def reset(self) -> None:
        """Restore the simulator to its post-construction state.

        The next :meth:`step` call will reproduce the same trajectory as the
        very first call after construction (given identical ``nsig``).
        """
        self._restore(self._initial_snapshot)

    def _snapshot(self) -> _TVBSnapshot:
        """Capture current simulator state for later restoration."""
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
        """Restore a previously captured simulator state."""
        state, step, buf, rng_state = snap
        history = self._sim.history
        if history is None:
            msg = "TVB history is None — configure() must be called first"
            raise RuntimeError(msg)
        self._sim.current_state = state.copy()
        self._sim.current_step = step
        history.buffer[:] = buf
        self._sim.integrator.noise.random_stream.set_state(rng_state)
