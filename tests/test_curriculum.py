"""Unit tests for the horizon-length curriculum schedule (:func:`neuro.nn_training.curriculum_state`)."""

from __future__ import annotations

import itertools

import numpy as np

from neuro.nn_training import curriculum_state


def _length(mask: np.ndarray) -> int:
    return int(mask.sum())


def test_mask_is_prefix_of_ones() -> None:
    """The mask is always a contiguous prefix of ones of length L."""
    horizon = 20
    for epoch in range(0, 120, 7):
        mask = curriculum_state(epoch, horizon, start_epoch=10, end_epoch=90)
        length = _length(mask)
        assert mask.shape == (horizon,)
        assert np.array_equal(mask, np.array([1.0] * length + [0.0] * (horizon - length)))


def test_holds_at_one_before_start_then_grows_to_horizon() -> None:
    """L = 1 until start_epoch, ramps 1 -> horizon by end_epoch, then holds at horizon."""
    horizon, start_epoch, end_epoch = 20, 10, 90

    def length_at(epoch: int) -> int:
        return _length(curriculum_state(epoch, horizon, start_epoch=start_epoch, end_epoch=end_epoch))

    assert all(length_at(e) == 1 for e in range(start_epoch + 1))  # L = 1 up to and incl. start_epoch
    assert length_at(end_epoch) == horizon  # full horizon reached at end_epoch
    assert length_at(end_epoch + 50) == horizon  # and held afterwards
    lengths = [length_at(e) for e in range(end_epoch + 1)]
    assert all(b >= a for a, b in itertools.pairwise(lengths))  # non-decreasing over the ramp


def test_psd_gate_only_at_full_length() -> None:
    """The last mask entry (the PSD gate) is on iff L == horizon."""
    horizon = 20
    for epoch in range(120):
        mask = curriculum_state(epoch, horizon, start_epoch=10, end_epoch=90)
        assert bool(mask[-1] == 1.0) == (_length(mask) == horizon)


def test_zero_width_window_jumps_to_horizon() -> None:
    """With start_epoch == end_epoch the rollout jumps 1 -> horizon (no divide-by-zero)."""
    horizon, pivot = 20, 30
    assert _length(curriculum_state(pivot, horizon, start_epoch=pivot, end_epoch=pivot)) == 1
    assert _length(curriculum_state(pivot + 1, horizon, start_epoch=pivot, end_epoch=pivot)) == horizon
