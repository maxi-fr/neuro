from __future__ import annotations

from typing import TYPE_CHECKING

from observable_predictability.features import REPRESENTATIONS

if TYPE_CHECKING:
    from observable_predictability.archive import ObservableEvaluationBundle

REPRESENTATION_LABELS: dict[str, str] = {
    "waveform": "Waveform (50 Hz)",
    "spectral": "Spectral Log-Power Frame",
    "connectivity": "Connectivity iCoh Frame",
}
REPRESENTATION_PENALTY_LABELS: dict[str, str] = {
    "waveform": "Waveform Tracking Penalty",
    "spectral": "Spectral Hinge Penalty",
    "connectivity": "Connectivity Interval Penalty",
}


def representation_shared_history_s(bundle: ObservableEvaluationBundle, representation: str) -> float:
    """Return the post-branch lookahead duration in seconds over which pre-branch history contributes."""
    if representation == "waveform":
        return 0.0
    if representation == "spectral":
        spec_geom = bundle.metadata.spectral_geometry
        fs = float(spec_geom.get("fs", 50.0))
        n_seg = int(spec_geom.get("n_segment", 25))
        k_width = int(spec_geom.get("kernel_width", 1))
        n_hop = int(spec_geom.get("n_hop", 3))
        return float((n_seg + max(0, k_width - 1) * n_hop) / max(fs, 1e-9))
    if representation == "connectivity":
        return float(bundle.connectivity_support.selected_window_s)
    msg = f"Unknown representation {representation!r}; expected one of {REPRESENTATIONS}."
    raise ValueError(msg)
