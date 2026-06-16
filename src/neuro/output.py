"""Output components for the simulate framework.

Currently only :class:`EEGOutput`, which collapses the Jansen-Rit network state to
scalp EEG through the connectome's forward operator.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Self

import numpy as np
import numpy.typing as npt
from simulate.output import Output

from neuro.jansen_rit import output as regional_output

FloatArray = npt.NDArray[np.float64]


@dataclasses.dataclass(frozen=True)
class EEGOutputLog:
    """Dataclass log snapshot of the EEG measurement."""


class EEGOutput(Output[EEGOutputLog]):
    """Map the Jansen-Rit network state to scalp EEG via the forward operator.

    The orchestrator passes the flattened ``(6 N,)`` network state; this reshapes it
    to ``(6, N)``, takes the regional output ``y = x2 - x3`` (reusing
    :func:`neuro.jansen_rit.output`), and applies the ``(n_channels, N)`` EEG gain
    ``L`` so the measurement is ``eeg = L @ y`` of shape ``(n_channels,)``.
    """

    def __init__(
        self,
        dt: float,
        gain: FloatArray,
        selected_channels: list[int] | np.ndarray | None = None,
    ) -> None:
        """Initialize with the ``(n_channels, n_nodes)`` EEG forward operator ``gain``."""
        super().__init__(dt)
        self.gain = np.asarray(gain, dtype=np.float64)
        self.n_nodes = self.gain.shape[1]
        self.selected_channels = None if selected_channels is None else np.asarray(selected_channels, dtype=np.int64)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Instantiate from a raw config dict, loading the TVB connectome by ``speed``."""
        from neuro.connectome import load_connectome  # noqa: PLC0415

        connectome = load_connectome(speed=float(config.get("speed", 50.0)))
        gain = connectome.gain
        n_nodes_val = config.get("n_nodes")
        if n_nodes_val is not None:
            gain = gain[:, : int(n_nodes_val)]

        selected_channels_cfg = config.get("selected_channels")
        selected_channels = None
        if selected_channels_cfg is not None:
            selected_channels = []
            for ch in selected_channels_cfg:
                if isinstance(ch, str):
                    selected_channels.append(connectome.channel_index[ch])
                else:
                    selected_channels.append(int(ch))
            selected_channels = np.array(selected_channels, dtype=np.int64)

        return cls(dt=float(config["dt"]), gain=gain, selected_channels=selected_channels)

    def update(
        self,
        t: float,  # noqa: ARG002
        x: float | np.ndarray,
        u: float | np.ndarray,  # noqa: ARG002
    ) -> tuple[float | np.ndarray, EEGOutputLog]:
        """Collapse the network state ``x`` to an EEG channel vector."""
        x_grid = np.asarray(x, dtype=np.float64).reshape(6, self.n_nodes)
        y_region = regional_output(x_grid)
        eeg = self.gain @ y_region
        if self.selected_channels is not None:
            eeg = eeg[self.selected_channels]
        return eeg, EEGOutputLog()
