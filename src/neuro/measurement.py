"""Measurement models for the simulate framework.

Currently only :class:`EEGMeasurement`, a :data:`~simulate.sensor.MeasurementModel`
that collapses the Jansen-Rit network state to scalp EEG through the connectome's
forward operator. Compose it with :class:`simulate.sensor.GaussianSensor` (which owns
the sample rate and additive noise) instead of a bespoke sensor.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from neuro.jansen_rit import lfp as regional_lfp

FloatArray = npt.NDArray[np.float64]


class EEGMeasurement:
    """Map the Jansen-Rit network state to scalp EEG via the forward operator.

    The orchestrator passes the plant state (the ``(6, N)`` network array, or its
    flattened ``(6 N,)`` form); this reshapes it to ``(6, N)``, takes the regional
    output ``y = x2 - x3`` (reusing :func:`neuro.jansen_rit.output`), and applies the
    ``(n_channels, N)`` EEG gain ``L`` so the measurement is ``eeg = L @ y`` of shape
    ``(n_channels,)``.

    Instantiated directly with config kwargs by
    :func:`simulate.config.build_measurement`, so it loads the TVB connectome by
    ``speed`` itself (there is no ``from_config``).
    """

    def __init__(
        self,
        speed: float = 50.0,
        n_nodes: int | None = None,
        selected_channels: list[int | str] | None = None,
    ) -> None:
        """Load the ``(n_channels, n_nodes)`` EEG forward operator for ``speed``."""
        from neuro.connectome import load_connectome  # noqa: PLC0415

        connectome = load_connectome(speed=speed)
        gain = connectome.gain
        if n_nodes is not None:
            gain = gain[:, : int(n_nodes)]
        self.gain = np.asarray(gain, dtype=np.float64)
        self.n_nodes = self.gain.shape[1]

        if selected_channels is None:
            self.selected_channels = None
        else:
            resolved = [connectome.channel_index[ch] if isinstance(ch, str) else int(ch) for ch in selected_channels]
            self.selected_channels = np.array(resolved, dtype=np.int64)

    def __call__(
        self,
        _t: float,
        x: float | np.ndarray,
        _u: float | np.ndarray,
    ) -> FloatArray:
        """Collapse the network state ``x`` to an EEG channel vector."""
        x_grid = np.asarray(x, dtype=np.float64).reshape(6, self.n_nodes)
        eeg = self.gain @ regional_lfp(x_grid)
        if self.selected_channels is not None:
            eeg = eeg[self.selected_channels]
        return eeg
