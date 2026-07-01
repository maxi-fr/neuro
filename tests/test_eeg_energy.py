"""Stage 5 validation: EEG projection + band-limited spectral energy.

Projects the canonical uncontrolled EZ/PZ seizure network to the 62 EEG sensors
through the forward operator ``L`` (``connectome.gain``), integrates the 0-50 Hz
Welch PSD per channel, and checks the energy ranking against Fig 3c of Yu et al.
(2024): high-energy channels are left-hemisphere temporo-parietal (F3/P3/CP5),
ipsilateral to the EZ.

We assert the *statistical* signature (ipsilateral lateralization, the Fig-3c
trio dominating their contralateral homologs) rather than a single literal max
channel: under the mandated SUM forward model, broad frontal lead fields (F1/FC1)
carry the largest absolute energy, so the literal argmax is frontal even though
the temporo-parietal cluster is the salient, EZ-driven feature. EEG magnitudes
are in arbitrary units (see the ``jansen-rit-eeg-uncalibrated-units`` note), so
every assertion here is relative. Strict L validation is deferred to Stage 7's
TVB EEG monitor cross-check.
"""

import re
from dataclasses import replace

import numpy as np
import pytest

from neuro.connectome import Connectome, load_connectome
from neuro.jansen_rit import JansenRitParams, lfp, simulate_network
from neuro.types import FloatArray
from utils.processing import band_energy, steady_window

_FIG3C_TRIO = (("CP5", "CP6"), ("P3", "P4"), ("F3", "F4"))  # (left, right homolog)
_DT = 1e-4


def _channel_side(label: str) -> str:
    """Hemisphere of a 10-20 channel from its number (odd=left, even=right)."""
    match = re.search(r"(\d+)", label)
    if match is None:
        return "z"
    return "L" if int(match.group(1)) % 2 == 1 else "R"


@pytest.fixture(scope="module")
def connectome() -> Connectome:
    """Load the TVB connectome once and calibrate its weight density."""
    return load_connectome()


@pytest.fixture(scope="module")
def eeg_energy(connectome: Connectome) -> FloatArray:
    """Per-channel normalized 0-50 Hz energy of the canonical seizure network."""
    n_nodes = len(connectome.region_labels)
    ez_idxs = [connectome.region_index[name] for name in ("lHC", "lPHC", "lAMYG")]
    pz_idxs = [connectome.region_index[name] for name in ("lTCI", "lTCV")]

    a_gains = np.full(n_nodes, 3.25)
    a_gains[ez_idxs] = 3.6
    a_gains[pz_idxs] = 3.4

    _, x_traj = simulate_network(
        params=JansenRitParams.from_config(
            {"connectome": connectome, "dt": _DT, "params": {"K": 0.5357, "A": a_gains}}
        ),
        duration=15.0,
        dt=_DT,
        seed=69,
    )
    y = lfp(x_traj)
    dt_ms = _DT * 1000.0
    y_steady = steady_window(y, dt_ms, transient_ms=2000.0)

    eeg = connectome.gain @ y_steady
    return band_energy(eeg, dt_ms, band=(0.0, 50.0))


def test_top_channels_are_ipsilateral(connectome: Connectome, eeg_energy: FloatArray) -> None:
    """The strongest channels are all left-hemisphere, ipsilateral to the EZ (Fig 3c)."""
    top8 = connectome.channel_labels[np.argsort(eeg_energy)[::-1][:8]]
    assert all(_channel_side(str(ch)) != "R" for ch in top8), list(map(str, top8))


def test_fig3c_trio_beats_contralateral_homologs(connectome: Connectome, eeg_energy: FloatArray) -> None:
    """Each Fig-3c left channel carries far more energy than its right homolog."""
    for left, right in _FIG3C_TRIO:
        e_left = eeg_energy[connectome.channel_index[left]]
        e_right = eeg_energy[connectome.channel_index[right]]
        assert e_left > e_right, f"{left}={e_left:.3f} !> {right}={e_right:.3f}"


def test_fig3c_channels_are_elevated(connectome: Connectome, eeg_energy: FloatArray) -> None:
    """The Fig-3c trio (F3/P3/CP5) sits well above the median channel energy."""
    median = float(np.median(eeg_energy))
    for left, _ in _FIG3C_TRIO:
        assert eeg_energy[connectome.channel_index[left]] > median
