from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

import jax.numpy as jnp
import numpy as np
from simulate.component import NoLog
from simulate.estimator import Estimator
from simulate.sensor import Sensor

from neuro.connectome import Connectome
from neuro.jansen_rit import JansenRitParams
from neuro.predictor.jansen_rit import JansenRitModel, lfp_jax, sigmoid_jax

if TYPE_CHECKING:
    from neuro.types import FloatArray


class FullStateSensor(Sensor[NoLog]):
    """Noiseless Sensor handing the Plant's ``(6, n_nodes)`` state over as a flat vector."""

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary."""
        return cls(dt=float(config["dt"]))

    def update(self, t: float, x: FloatArray, u: FloatArray) -> tuple[FloatArray, NoLog]:
        """Flatten the Plant state; the oracle handover carries no measurement model and no noise.

        Parameters
        ----------
        x : FloatArray
            Plant state of shape ``(6, n_nodes)``.
        """
        del t, u
        return np.asarray(x, dtype=np.float64).reshape(-1), NoLog()


class JansenRitOracleEstimator(Estimator[NoLog]):
    """Pack the Plant's full state into a Jansen-Rit Predictor state, rebuilding the delay buffer on the Predictor grid.

    The Plant carries its delay history at the Plant step while the Predictor integrates at its own
    (coarser) ``dt``, so the buffer cannot be copied across: this Estimator runs at the Predictor's
    ``dt`` and records ``S(y)`` once per Predictor step, which is the spacing
    :meth:`~neuro.predictor.jansen_rit.JansenRitModel.coupling_from_history` reads it back at. The
    handover is an upper bound on state knowledge, not a deployable observer.
    """

    def __init__(self, dt: float, model: JansenRitModel) -> None:
        """Initialize from the Predictor integration step ``dt`` and the model whose state layout it fills."""
        super().__init__(dt)
        self.model = model
        self._history: FloatArray | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate the component from a raw configuration dictionary."""
        dt = float(config["dt"])
        conn = Connectome.from_config(config["connectome"])
        params = JansenRitParams.from_config(config["params"])
        model = JansenRitModel.from_plant_components(params, conn=conn, dt=dt)
        return cls(dt=dt, model=model)

    def update(self, t: float, y_mea: FloatArray, u: FloatArray) -> tuple[FloatArray, NoLog]:
        """Emit the packed Predictor state for the observed Plant state.

        Parameters
        ----------
        y_mea : FloatArray
            Flattened Plant state of shape ``(6 * n_nodes,)``.

        Returns
        -------
        FloatArray
            Packed Predictor state of shape ``(model.n,)``: ODE state, delay buffer, step index.
        """
        del u
        model = self.model
        x_ode = np.asarray(y_mea, dtype=np.float64).reshape(6, model.n_nodes)
        s_y = np.asarray(
            sigmoid_jax(lfp_jax(jnp.asarray(x_ode)), model.e0, model.v0, model.r),
            dtype=np.float64,
        )
        if self._history is None:
            self._history = np.broadcast_to(s_y, (model.max_history_len, model.n_nodes)).copy()

        k = round(t / self.dt)
        self._history[k % model.max_history_len] = s_y
        z = model.pack_state(jnp.asarray(x_ode), jnp.asarray(self._history), float(k))
        return np.asarray(z, dtype=np.float64), NoLog()
