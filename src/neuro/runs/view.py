from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from neuro.types import FloatArray


@dataclass
class ViewSettings:
    """Every plot choice needed to reopen a view, with paths relative to the collection."""

    mode: str = "Closed-loop runs"
    runs: list[str] = field(default_factory=list)
    replays: list[str] = field(default_factory=list)
    start: float = 0.0
    width: float = 12.0
    decision: float = 6.0
    neighbors: int = 0
    spacing: int = 5
    eeg_channels: list[int] = field(default_factory=lambda: [0])
    output_channels: list[int] = field(default_factory=lambda: [0])
    electrodes: list[int] = field(default_factory=lambda: [0])
    regions: list[int] = field(default_factory=list)
    offset: float = 0.0
    note: str = ""
    show_spectra: bool = True
    spectral_steps: int = 20
    spectral_frame: int = 1
    spectral_channels: list[int] = field(default_factory=lambda: [0])


def load_bookmarks(path: Path) -> dict[str, ViewSettings]:
    """Load named plot settings, or an empty collection before the first save."""
    if not path.exists():
        return {}
    return {name: ViewSettings(**values) for name, values in json.loads(path.read_text(encoding="utf-8")).items()}


def save_bookmark(path: Path, name: str, settings: ViewSettings) -> None:
    """Persist one named view without losing the collection's other bookmarks."""
    bookmarks = load_bookmarks(path)
    bookmarks[name] = settings
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps({key: asdict(value) for key, value in bookmarks.items()}, indent=2), encoding="utf-8")
    pending.replace(path)


def decision_indices(times: FloatArray, settings: ViewSettings) -> list[int]:
    """Select neighboring saved attempt anchors, including runs with no attempts during Warm-up."""
    if not len(times):
        return []
    center = int(np.argmin(np.abs(times - settings.decision)))
    return [
        i
        for j in range(-settings.neighbors, settings.neighbors + 1)
        if 0 <= (i := center + j * settings.spacing) < len(times)
    ]
