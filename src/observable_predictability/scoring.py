from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from neuro.types import FloatArray, IntArray

from observable_predictability.features import (
    DEFAULT_FRAME_HOP_S,
)
from observable_predictability.metrics import RolloutIdentity, validate_identities


@dataclass(frozen=True)
class ReplayMoments:
    """Component means and sample variances over the replays of each snapshot."""

    times: FloatArray
    means: FloatArray
    variances: FloatArray
    n_replays: int


@dataclass(frozen=True)
class PotentialPredictability:
    """Finite-replay corrected signal fraction and native-unit noise floor by lookahead."""

    times: FloatArray
    signal_fraction: FloatArray
    noise_floor: FloatArray
    valid_component_counts: IntArray


def compute_replay_moments(
    values: FloatArray,
    times: FloatArray,
    identities: Sequence[RolloutIdentity],
) -> ReplayMoments:
    """Reduce balanced replay groups to moments with shape (snapshots, lookaheads, components)."""
    grouped = _group_trajectories_by_identities(values, len(times), identities)
    means = grouped.mean(axis=1)
    n_replays = grouped.shape[1]
    variances = grouped.var(axis=1, ddof=1) if n_replays > 1 else np.full_like(means, np.nan)
    return ReplayMoments(times, means, variances, n_replays)


def compute_potential_predictability(blocks: Sequence[ReplayMoments]) -> PotentialPredictability:
    """Combine complete replay groups with the finite-replay correction of Matsumura et al."""
    first = blocks[0]
    if any(block.n_replays != first.n_replays or not np.array_equal(block.times, first.times) for block in blocks):
        msg = "Replay blocks must share the same lookaheads and replay count."
        raise ValueError(msg)

    means = np.concatenate([block.means for block in blocks], axis=0)
    variances = np.concatenate([block.variances for block in blocks], axis=0)
    eligible = np.all(np.isfinite(means) & np.isfinite(variances), axis=0)
    counts = np.count_nonzero(eligible, axis=-1)
    # Select complete components before computing either variance term.
    noise = np.where(eligible, variances, 0.0).mean(axis=0).sum(axis=-1)
    floor = np.sqrt(np.divide(noise, counts, out=np.full_like(noise, np.nan), where=counts > 0))
    if len(means) < 2:  # noqa: PLR2004 -- between-snapshot sample variance needs two snapshots
        undefined = np.full_like(first.times, np.nan, dtype=np.float64)
        return PotentialPredictability(first.times, undefined, floor, counts)
    between = np.where(eligible, means, 0.0).var(axis=0, ddof=1).sum(axis=-1)
    signal = between - noise / first.n_replays
    total = signal + noise
    ratio = np.divide(signal, total, out=np.full_like(total, np.nan), where=total > 0)
    return PotentialPredictability(first.times, ratio, floor, counts)


def _normalize_3d_rollout_components(arr: FloatArray, n_steps: int) -> FloatArray:
    """Validate and reshape rollout trajectories into ``(n_rollouts, n_steps, n_components)``."""
    if arr.ndim < 3:  # noqa: PLR2004 -- (n_rollouts, n_steps, *components)
        msg = f"values must have at least 3 dimensions (n_rollouts, n_steps, *components), got {arr.shape}."
        raise ValueError(msg)
    aligned = np.moveaxis(arr, -1, 1) if (arr.shape[1] != n_steps and arr.shape[-1] == n_steps) else arr
    if aligned.shape[1] != n_steps:
        msg = f"values time dimension ({aligned.shape[1]}) does not match len(times) ({n_steps})."
        raise ValueError(msg)
    return aligned.reshape(aligned.shape[0], n_steps, -1)


def _group_trajectories_by_identities(
    arr: FloatArray,
    n_steps: int,
    identities: Sequence[RolloutIdentity],
) -> FloatArray:
    """Group trajectories by ``(plant, parent_seed, branch, arm, split)`` across stochastic replicates."""
    validated = validate_identities(identities)
    if len(validated) != arr.shape[0]:
        msg = f"identities length ({len(validated)}) does not match values leading dimension ({arr.shape[0]})."
        raise ValueError(msg)
    flat_comp = _normalize_3d_rollout_components(arr, n_steps)

    groups: dict[tuple[str, int, str, str, str], list[tuple[int, int]]] = {}
    for row_idx, ident in enumerate(validated):
        g_key = (ident.plant, int(ident.parent_seed), ident.branch, ident.arm, ident.split)
        groups.setdefault(g_key, []).append((int(ident.replicate), row_idx))

    rep_counts = {len(items) for items in groups.values()}
    if len(rep_counts) != 1:
        msg = f"Unbalanced replicate counts across snapshots: {sorted(rep_counts)}."
        raise ValueError(msg)

    ordered_keys = sorted(groups)
    idx_grid = np.asarray(
        [[idx for _, idx in sorted(groups[g_key], key=lambda pair: pair[0])] for g_key in ordered_keys],
        dtype=np.intp,
    )
    return flat_comp[idx_grid]


def _resolve_aligned_frame_times(
    n_frames: int,
    p_times: FloatArray | None,
    s_times: FloatArray | None,
) -> FloatArray:
    """Validate and return the shared 60 ms Frame timestamps."""
    if p_times is not None and s_times is not None:
        if len(p_times) != n_frames or len(s_times) != n_frames:
            msg = f"Timestamp lengths ({len(p_times)}, {len(s_times)}) do not match Frame count ({n_frames})."
            raise ValueError(msg)
        if not np.allclose(p_times, s_times, rtol=1e-5, atol=1e-6):
            msg = "Penalty Frame timestamps do not align with Seizure State Frame timestamps."
            raise ValueError(msg)
        return s_times
    if s_times is not None:
        return s_times
    if p_times is not None:
        return p_times
    return np.arange(1, n_frames + 1, dtype=np.float64) * DEFAULT_FRAME_HOP_S


def _align_penalty_to_state_frames(
    penalty: FloatArray,
    state: FloatArray,
    *,
    penalty_times: FloatArray | None = None,
    state_times: FloatArray | None = None,
    frame_hop: int | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Align candidate penalty series ``(n_rollouts, n_steps)`` onto the 60 ms Seizure State Frame grid."""
    pen = np.atleast_2d(np.asarray(penalty, dtype=np.float64))
    st = np.atleast_2d(np.asarray(state, dtype=np.float64))
    if pen.shape[0] != st.shape[0]:
        msg = f"Penalty rollout count ({pen.shape[0]}) does not match Seizure State rollout count ({st.shape[0]})."
        raise ValueError(msg)

    hop = frame_hop
    if hop is None and pen.shape[1] != st.shape[1] and st.shape[1] > 0 and pen.shape[1] % st.shape[1] == 0:
        hop = pen.shape[1] // st.shape[1]

    p_times = None if penalty_times is None else np.asarray(penalty_times, dtype=np.float64)
    s_times = None if state_times is None else np.asarray(state_times, dtype=np.float64)

    if hop is not None and hop > 1 and pen.shape[1] != st.shape[1]:
        n_frames = pen.shape[1] // hop
        pen = np.mean(pen[:, : n_frames * hop].reshape(pen.shape[0], n_frames, hop), axis=-1)
        if p_times is not None and len(p_times) >= n_frames * hop:
            p_times = p_times[hop - 1 : n_frames * hop : hop]

    has_t0_prefix = (
        s_times is not None
        and np.isclose(s_times[0], 0.0)
        and (st.shape[1] == len(s_times) + 1 or (st.shape[1] == len(s_times) and pen.shape[1] == st.shape[1] - 1))
    )
    if has_t0_prefix and s_times is not None:
        st = st[:, 1:]
        s_times = s_times[1:]

    if pen.shape[1] != st.shape[1]:
        msg = f"Aligned penalty Frame count ({pen.shape[1]}) does not match Seizure State Frame count ({st.shape[1]})."
        raise ValueError(msg)

    times = _resolve_aligned_frame_times(pen.shape[1], p_times, s_times)
    return times, pen, st


@dataclass(frozen=True)
class SeizureRelevanceResult:
    """Criterion 2 zero-current seizure relevance evaluation on the shared 60 ms Frame grid."""

    times: FloatArray
    penalties_by_branch: dict[str, FloatArray]
    states_by_branch: dict[str, FloatArray]
    identities_by_branch: dict[str, tuple[RolloutIdentity, ...]]

    def finite_pairs(self) -> FloatArray:
        """Return finite seizure-Plant pairs in Seizure State, penalty column order."""
        pairs = []
        for branch, identities in self.identities_by_branch.items():
            seizure = np.array([identity.plant == "seizure" for identity in identities])
            state = self.states_by_branch[branch][seizure]
            penalty = self.penalties_by_branch[branch][seizure]
            pairs.append(np.column_stack((state.ravel(), penalty.ravel())))
        points = np.concatenate(pairs)
        return points[np.isfinite(points).all(axis=1)]

    @staticmethod
    def pearson(points: FloatArray) -> float:
        """Return the pooled Pearson coefficient of the plotted pairs."""
        if not len(points) or np.any(np.ptp(points, axis=0) == 0):
            return float("nan")
        return float(np.corrcoef(points, rowvar=False)[0, 1])


def score_zero_current_seizure_relevance(
    penalties: Mapping[str, FloatArray],
    states: Mapping[str, FloatArray],
    *,
    identities: dict[str, tuple[RolloutIdentity, ...]],
    penalty_times: FloatArray,
    state_times: FloatArray,
) -> SeizureRelevanceResult:
    """Align penalty and Seizure State samples on the common Frame grid."""
    aligned = {
        branch: _align_penalty_to_state_frames(
            penalty, states[branch], penalty_times=penalty_times, state_times=state_times
        )
        for branch, penalty in penalties.items()
    }
    return SeizureRelevanceResult(
        next(iter(aligned.values()))[0],
        {branch: values[1] for branch, values in aligned.items()},
        {branch: values[2] for branch, values in aligned.items()},
        identities,
    )
