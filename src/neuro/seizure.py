"""Shared seizure-regime constants and helpers for the data-collection experiments.

The EZ/PZ seizure regime -- a seizure originating in ``lHC``/``lPHC``/``lAMYG`` (EZ) and
propagating to ``lTCI``/``lTCV`` (PZ), with all other regions healthy -- underpins the seizure
YAML configs (``configs/simulation/jansen_rit_seizure*.yaml``, whose ``A`` vectors are
:func:`build_seizure_a_gains`) and the smart UKF initialisation in
``scripts/run_ukf_feasibility.py``. Defining the regime once here keeps them in sync.

:func:`spread_profile` measures *when* each region is recruited into that seizure, which is
what the global coupling ``K`` and the noise ``sigma`` are tuned against (see
``scripts/sweep_seizure_spread.py`` and ``docs/seizure_spread_tuning.md``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from neuro.jansen_rit import JansenRitDynamics, lfp, simulate_network

if TYPE_CHECKING:
    import numpy.typing as npt

    from neuro.connectome import Connectome
    from neuro.types import FloatArray


DT = 1e-4
K = 0.54
SPEED = 50.0
SEED = 69

EZ_REGIONS = ("lHC", "lPHC", "lAMYG")
PZ_REGIONS = ("lTCI", "lTCV")

A_HEALTHY = 3.25
A_EZ = 3.6
A_PZ = 3.4

# Recruitment detector. The ictal limit cycle runs at 2-5 Hz with a peak-to-peak swing of
# ~14 mV, so a 1 s window always spans at least two cycles and the 5 mV threshold sits far
# above the noise-driven background (<= 2 mV at the paper's sigma = 500) yet far below the
# ictal amplitude -- the same absolute cut therefore stays valid across the swept sigma range.
# A region must stay above the cut for SPREAD_PERSIST_S to count as recruited: single-window
# excursions are what a noise burst (or the settling transient of a badly-initialised plant)
# produces, and scoring those as onsets makes every region look recruited at t = 0.
SPREAD_WINDOW_S = 1.0
SPREAD_HOP_S = 0.25
SPREAD_PERSIST_S = 1.0
SEIZURE_PTP_MV = 5.0

# Target propagation schedule the K/sigma search is scored against (see
# docs/seizure_spread_tuning.md): the focus ignites promptly, the PZ follows about halfway
# through, and the seizure has taken over half the left hemisphere at ~10 s while the right
# hemisphere stays healthy. Times in seconds.
T_EZ_TARGET = 1.5
T_PZ_TARGET = 5.0
T_LEFT_TARGET = 10.0
LEFT_EXTENT_TARGET = 0.5
CONTRALATERAL_WEIGHT = 2.0


def build_seizure_a_gains(connectome: Connectome) -> FloatArray:
    """Build the per-region excitatory-gain vector for the EZ/PZ seizure regime."""
    a_gains = np.full(len(connectome.region_labels), A_HEALTHY)
    a_gains[[connectome.region_index[name] for name in EZ_REGIONS]] = A_EZ
    a_gains[[connectome.region_index[name] for name in PZ_REGIONS]] = A_PZ
    return a_gains


def focus_indices(connectome: Connectome) -> list[int]:
    """Return the region indices of the EZ then PZ focus nodes."""
    return [connectome.region_index[name] for name in (*EZ_REGIONS, *PZ_REGIONS)]


@dataclass(frozen=True)
class SpreadProfile:
    """When each region is recruited into the seizure, and how loudly it rings.

    Attributes
    ----------
    times
        Window-centre times in s, shape ``(n_windows,)``.
    ptp
        Sliding peak-to-peak amplitude of each region's ``y = x2 - x3`` in mV,
        shape ``(n_nodes, n_windows)``.
    onsets
        Recruitment time of each region in s, shape ``(n_nodes,)``: the centre of the
        first window that starts a sustained run above the threshold, or ``NaN`` for a
        region that never seizes.
    threshold
        The peak-to-peak amplitude in mV that counts as seizing.
    """

    times: FloatArray
    ptp: FloatArray
    onsets: FloatArray
    threshold: float

    @classmethod
    def from_ptp(
        cls,
        times: FloatArray,
        ptp: FloatArray,
        threshold: float = SEIZURE_PTP_MV,
        persist_s: float = SPREAD_PERSIST_S,
    ) -> SpreadProfile:
        """Derive the onsets of a stored amplitude envelope, so it can be re-thresholded.

        A region is recruited at the first window that begins ``persist_s`` of uninterrupted
        supra-threshold amplitude; the trailing ``persist_s`` of the run cannot start such a
        window, so a region that ignites right at the end reads as unrecruited.
        """
        hop = float(times[1] - times[0]) if len(times) > 1 else 1.0
        n_hold = max(1, round(persist_s / hop))

        above = ptp > threshold
        # sustained[:, i] is True when windows i .. i + n_hold - 1 are all above threshold.
        sustained = np.logical_and.reduce([above[:, i : above.shape[1] - n_hold + 1 + i] for i in range(n_hold)])

        onsets = np.full(ptp.shape[0], np.nan, dtype=np.float64)
        ever = sustained.any(axis=1)
        onsets[ever] = times[: sustained.shape[1]][sustained[ever].argmax(axis=1)]
        return cls(times=times, ptp=ptp, onsets=onsets, threshold=threshold)

    def seizing(self) -> npt.NDArray[np.bool_]:
        """Boolean recruitment mask over ``(n_nodes, n_windows)``."""
        return self.ptp > self.threshold

    def n_seizing(self, nodes: npt.NDArray[np.bool_] | list[int] | None = None) -> npt.NDArray[np.int64]:
        """Count the seizing regions in each window, shape ``(n_windows,)``.

        ``nodes`` restricts the count to a subset (boolean mask or index list); the
        default counts the whole network.
        """
        seizing = self.seizing()
        return np.count_nonzero(seizing if nodes is None else seizing[nodes], axis=0)

    def recruited_by(self, t: float, nodes: npt.NDArray[np.bool_] | list[int] | None = None) -> int:
        """Count the regions whose onset is at or before time ``t`` (s)."""
        onsets = self.onsets if nodes is None else self.onsets[nodes]
        return int(np.count_nonzero(onsets <= t))


def spread_profile(
    dyn: JansenRitDynamics,
    duration: float,
    *,
    window_s: float = SPREAD_WINDOW_S,
    hop_s: float = SPREAD_HOP_S,
    threshold: float = SEIZURE_PTP_MV,
) -> SpreadProfile:
    """Run ``dyn`` for ``duration`` seconds and measure when each region starts seizing.

    A region counts as seizing once the peak-to-peak amplitude of its ``y = x2 - x3``
    over a ``window_s`` window exceeds ``threshold``; the windows advance by ``hop_s``
    and are timestamped at their centre, so an onset is unbiased to within ``hop_s``.

    The plant is stepped in ``hop_s`` chunks and only the trailing ``window_s`` of
    states is kept, so a long run costs bounded memory (the full ``(6, N, n_samples)``
    trajectory of a 60 s network run is several GB).

    Parameters
    ----------
    dyn
        The Jansen-Rit plant, stepped on from its current state (it is mutated).
    duration
        Simulated time in seconds.
    window_s
        Length of the amplitude window in seconds.
    hop_s
        Spacing between consecutive windows in seconds.
    threshold
        Peak-to-peak amplitude in mV above which a region counts as seizing.

    Returns
    -------
    SpreadProfile
        Window times, the per-region amplitude envelope, and the onset times.
    """
    chunks_per_window = round(window_s / hop_s)
    n_hops = int(duration / hop_s)

    y_chunks: list[FloatArray] = []
    ptp_cols: list[FloatArray] = []
    times: list[float] = []
    for k in range(n_hops):
        _, x = simulate_network(dyn=dyn, duration=hop_s, t0=k * hop_s)
        y_chunks.append(lfp(x)[:, 1:])  # drop the duplicated carry-over sample
        y_chunks = y_chunks[-chunks_per_window:]
        if len(y_chunks) == chunks_per_window:
            ptp_cols.append(np.ptp(np.concatenate(y_chunks, axis=1), axis=1))
            times.append((k + 1) * hop_s - window_s / 2.0)

    return SpreadProfile.from_ptp(np.asarray(times, dtype=np.float64), np.stack(ptp_cols, axis=1), threshold=threshold)


@dataclass(frozen=True)
class SpreadSummary:
    """The five numbers that describe one seizure's propagation, all onset-based.

    Attributes
    ----------
    t_ez
        Time in s at which the *last* EZ region is recruited (the seizure has fully
        ignited at the focus). ``NaN`` if some EZ region never seizes.
    t_pz
        Time in s at which the last PZ region is recruited; ``NaN`` if one never seizes.
    t_left_half
        Time in s at which half of the left-hemisphere regions have been recruited --
        a fixed-extent stand-in for "the spread across the hemisphere is well under way",
        so it is comparable between runs. ``NaN`` if the run never gets there.
    frac_left
        Fraction of left-hemisphere regions recruited by the end of the run.
    frac_right
        Fraction of right-hemisphere regions recruited by the end of the run; the
        contralateral hemisphere is meant to stay healthy, so this should be ~0.
    """

    t_ez: float
    t_pz: float
    t_left_half: float
    frac_left: float
    frac_right: float

    def score(self, duration: float) -> float:
        """Distance from the target schedule; lower is better, 0 is a perfect match.

        Each term is a relative error against the targets at the top of this module:
        the EZ is penalised only for igniting *late* (early ignition at the focus is the
        point of the regime), the PZ and the hemisphere-wide spread are penalised in both
        directions, a left hemisphere that stays healthy and a right hemisphere that does
        not are penalised by extent. A region that never seizes is scored as if it were
        recruited at ``duration``, so an unrecruited PZ reads as maximally late rather
        than dropping out of the average.
        """

        def observed(t: float) -> float:
            return duration if not np.isfinite(t) else t

        e_ez = max(0.0, observed(self.t_ez) - T_EZ_TARGET) / T_EZ_TARGET
        e_pz = abs(observed(self.t_pz) - T_PZ_TARGET) / T_PZ_TARGET
        e_left = abs(observed(self.t_left_half) - T_LEFT_TARGET) / T_LEFT_TARGET
        e_extent = max(0.0, LEFT_EXTENT_TARGET - self.frac_left) / LEFT_EXTENT_TARGET
        return e_ez + e_pz + e_left + e_extent + CONTRALATERAL_WEIGHT * self.frac_right


def spread_summary(profile: SpreadProfile, connectome: Connectome) -> SpreadSummary:
    """Reduce a :class:`SpreadProfile` to the EZ -> PZ -> hemisphere propagation schedule."""
    onsets = profile.onsets
    left = ~connectome.hemispheres  # TVB marks the right hemisphere True

    def last_onset(regions: tuple[str, ...]) -> float:
        return float(np.max([onsets[connectome.region_index[name]] for name in regions]))

    left_onsets = np.sort(onsets[left])
    half = (int(np.count_nonzero(left)) + 1) // 2

    return SpreadSummary(
        t_ez=last_onset(EZ_REGIONS),
        t_pz=last_onset(PZ_REGIONS),
        t_left_half=float(left_onsets[half - 1]),
        frac_left=float(np.mean(np.isfinite(onsets[left]))),
        frac_right=float(np.mean(np.isfinite(onsets[~left]))),
    )
