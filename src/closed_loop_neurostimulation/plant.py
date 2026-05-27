"""FitzHugh-Nagumo whole-brain plant model built on neurolib.

The plant is a network of ~80 FHN oscillators coupled through a built-in HCP
structural connectome, driven by Ornstein-Uhlenbeck noise. It exposes a
step-by-step API so a controller can later inject inputs between steps, plus a
synthetic deterministic EEG lead-field projection.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from neurolib.models.fhn import FHNModel  # type: ignore[import-untyped]
from neurolib.utils.loadData import Dataset  # type: ignore[import-untyped]

FloatArray = npt.NDArray[np.float64]


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

    Notes
    -----
    Each call to :meth:`step` returns only the *most recent* chunk of activity
    (not the accumulated trajectory). Callers that need full histories must
    concatenate the per-step outputs themselves.

    The first underlying neurolib ``run`` must be called without
    ``continue_run`` to initialise the integrator state; subsequent calls use
    ``continue_run=True`` to carry state forward (see plant-model.md §6).
    """

    def __init__(
        self,
        dt: float = 0.1,
        sigma_ou: float = 0.05,
        n_sensors: int = 64,
        leadfield_seed: int = 0,
        connectome: str = "hcp",
    ) -> None:
        dataset = Dataset(connectome)
        self._model = FHNModel(Cmat=dataset.Cmat, Dmat=dataset.Dmat)
        self._model.params["dt"] = dt
        self._model.params["sigma_ou"] = sigma_ou

        self._dt: float = dt
        self._sigma_ou: float = sigma_ou
        self._n_nodes: int = int(dataset.Cmat.shape[0])
        self._leadfield: FloatArray = self._make_leadfield(n_sensors, self._n_nodes, leadfield_seed)
        self._initialized: bool = False

    @staticmethod
    def _make_leadfield(n_sensors: int, n_nodes: int, seed: int) -> FloatArray:
        rng = np.random.default_rng(seed)
        matrix = rng.standard_normal((n_sensors, n_nodes))
        # Row-normalise so every sensor sees the network at comparable gain.
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix.astype(np.float64)

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
            self._initialized = True
        activity: FloatArray = np.asarray(self._model.x, dtype=np.float64)
        eeg: FloatArray = self._leadfield @ activity
        return activity, eeg

    def reset(self) -> None:
        """Drop accumulated state so the next :meth:`step` re-initialises the model."""
        self._initialized = False
