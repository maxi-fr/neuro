from __future__ import annotations

import re

import numpy as np

from neuro.connectome import Connectome
from neuro.eeg import build_eeg_gain
from neuro.geometry import sensor_positions_mm

_N_REGIONS = 76
_N_CHANNELS = 62


def _channel_side(label: str) -> str:
    """Hemisphere of a 10-20 channel from its number (odd=left, even=right)."""
    match = re.search(r"(\d+)", label)
    if match is None:
        return "z"
    return "L" if int(match.group(1)) % 2 == 1 else "R"


def test_build_eeg_gain_shapes_and_values() -> None:
    """build_eeg_gain returns a (62, 76) gain matrix and 62 channel labels."""
    gain, channel_labels = build_eeg_gain()
    assert gain.shape == (_N_CHANNELS, _N_REGIONS)
    assert channel_labels.shape == (_N_CHANNELS,)
    assert np.isfinite(gain).all()
    assert "CP5" in channel_labels
    assert "PO3" in channel_labels


def test_eeg_gain_is_ipsilateral() -> None:
    """Lateral cortical regions project most strongly to same-side EEG channels."""
    connectome = Connectome.from_config({})
    gain, channel_labels = build_eeg_gain()
    for region, side in (("lTCV", "L"), ("rTCV", "R"), ("lTCI", "L"), ("rTCI", "R")):
        column = np.abs(gain[:, connectome.region_index[region]])
        top_channel = str(channel_labels[int(np.argmax(column))])
        assert _channel_side(top_channel) == side, f"{region} -> {top_channel}"


def test_sensor_positions_land_on_their_own_lead_field() -> None:
    """Placed electrodes sit near the regions their EEG lead field is strongest on."""
    conn = Connectome.from_config({})
    gain, channel_labels = build_eeg_gain()
    channel_index = {label: idx for idx, label in enumerate(channel_labels)}

    labels, positions = sensor_positions_mm()
    np.testing.assert_array_equal(labels, channel_labels)

    dist = np.linalg.norm(conn.centres[None, :, :] - positions[:, None, :], axis=2)
    peak_region = np.abs(gain).argmax(axis=1)
    to_peak = dist[np.arange(len(positions)), peak_region]
    assert to_peak.mean() < 50.0, f"electrodes are {to_peak.mean():.0f} mm from their own lead-field peak"

    left_region = np.array([str(label).startswith("l") for label in conn.region_labels])
    for channel in ("CP5", "T7", "TP9", "O1"):
        assert left_region[dist[channel_index[channel]].argmin()], f"{channel} is not on the left"
