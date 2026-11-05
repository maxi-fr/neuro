from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from neuro.types import FloatArray

DEFAULT_HOP_S = 0.05
SEIZURE_BAND_HZ = (3.0, 12.0)


@dataclass(frozen=True)
class RolloutIdentity:
    """Explicit identity key for one generated rollout row."""

    plant: str
    parent_seed: int
    branch: str
    replicate: int
    arm: str
    split: str = "eval"
    replicate_seed: int = 0

    @property
    def snapshot_key(self) -> tuple[str, int, str]:
        """Return the parent snapshot identity ``(plant, parent_seed, branch)``."""
        return (self.plant, self.parent_seed, self.branch)

    @property
    def pair_key(self) -> tuple[str, int, str, int]:
        """Return the arm-independent replicate identity ``(plant, parent_seed, branch, replicate)``."""
        return (self.plant, self.parent_seed, self.branch, self.replicate)

    @property
    def full_key(self) -> tuple[str, int, str, int, str]:
        """Return the complete row identity ``(plant, parent_seed, branch, replicate, arm)``."""
        return (self.plant, self.parent_seed, self.branch, self.replicate, self.arm)

    def to_dict(self) -> dict[str, Any]:
        """Serialize identity fields to a JSON-compatible dictionary."""
        return {
            "plant": self.plant,
            "parent_seed": int(self.parent_seed),
            "branch": self.branch,
            "replicate": int(self.replicate),
            "arm": self.arm,
            "split": self.split,
            "replicate_seed": int(self.replicate_seed),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RolloutIdentity:
        """Deserialize an identity record from a dictionary."""
        return cls(
            plant=str(data["plant"]),
            parent_seed=int(data["parent_seed"]),
            branch=str(data["branch"]),
            replicate=int(data["replicate"]),
            arm=str(data["arm"]),
            split=str(data.get("split", "eval")),
            replicate_seed=int(data.get("replicate_seed", 0)),
        )


def validate_identities(identities: Sequence[RolloutIdentity]) -> tuple[RolloutIdentity, ...]:
    """Validate that rollout identities contain no duplicate keys."""
    result = tuple(identities)
    seen: set[tuple[str, int, str, int, str]] = set()
    for ident in result:
        key = ident.full_key
        if key in seen:
            msg = f"Duplicate rollout identity key: {key}"
            raise ValueError(msg)
        seen.add(key)
    return result


def windowed(
    signals: FloatArray,
    fs: float,
    reduce: Callable[[FloatArray, float], FloatArray],
    *,
    window_s: float,
    hop_s: float = DEFAULT_HOP_S,
) -> tuple[FloatArray, FloatArray]:
    """Slide a **trailing** window over ``signals`` and reduce each block per channel.

    Windows are causal: the point reported at time ``t`` is computed from the samples in
    ``(t - window_s, t]``, and ``t`` is the instant the value first becomes available.

    Parameters
    ----------
    signals
        Input signals, shape ``(n_channels, n_samples)``.
    fs
        Sampling rate in Hz.
    reduce
        Maps one ``(n_channels, n_window)`` block and ``fs`` to one value per channel.
    window_s
        Window length in seconds.
    hop_s
        Spacing between consecutive windows in seconds.

    Returns
    -------
    times
        Window-end times in seconds since the start of ``signals``, shape ``(n_windows,)``.
        Empty if ``signals`` is shorter than one window.
    values
        The reduction of each window, shape ``(n_channels, n_windows)``.
    """
    n_window = round(window_s * fs)
    n_hop = round(hop_s * fs)
    starts = np.arange(0, signals.shape[1] - n_window + 1, n_hop)

    times = (starts + n_window) / fs
    columns = [np.asarray(reduce(signals[:, s : s + n_window], fs), dtype=np.float64) for s in starts]
    values = np.stack(columns, axis=-1) if columns else np.empty((signals.shape[0], 0), dtype=np.float64)
    return times.astype(np.float64), values
