from __future__ import annotations

import itertools

import numpy as np

from neuro.nn_training import _step_mask_selector, curriculum_state


def _length(mask: np.ndarray) -> int:
    """length."""
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
        """length at."""
        return _length(curriculum_state(epoch, horizon, start_epoch=start_epoch, end_epoch=end_epoch))

    assert all(length_at(e) == 1 for e in range(start_epoch + 1))
    assert length_at(end_epoch) == horizon
    assert length_at(end_epoch + 50) == horizon
    lengths = [length_at(e) for e in range(end_epoch + 1)]
    assert all(b >= a for a, b in itertools.pairwise(lengths))


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


def test_curriculum_max_steps_caps_the_ramp() -> None:
    """The curriculum ramps ``1 -> curriculum_max_steps``, never past it.

    Capping below the horizon is the point: it keeps multi-step training while never entering the
    long-unroll regime that destabilises this fit. Steps beyond the cap stay masked out, so the
    mask keeps its full ``horizon`` width.
    """
    horizon, cap = 20, 10
    selector = _step_mask_selector(horizon, cap, 0, 80)

    lengths = [_length(selector(epoch)) for epoch in (0, 20, 40, 80, 1000)]
    assert lengths[0] == 1
    assert lengths == sorted(lengths)
    assert max(lengths) == cap
    for epoch in (0, 40, 1000):
        mask = selector(epoch)
        assert mask.shape == (horizon,)
        assert mask[cap:].sum() == 0


def test_curriculum_max_steps_of_one_is_one_step_training() -> None:
    """A ramp from 1 to 1 never grows, so ``curriculum_max_steps=1`` is plain one-step training."""
    selector = _step_mask_selector(20, 1, 0, 80)
    assert all(_length(selector(epoch)) == 1 for epoch in (0, 40, 80, 1000))


def test_curriculum_max_steps_none_ramps_to_the_full_horizon() -> None:
    """Leaving the cap unset reproduces :func:`curriculum_state` over the whole horizon."""
    horizon, start, end = 20, 10, 90
    selector = _step_mask_selector(horizon, None, start, end)
    for epoch in (0, 5, 10, 50, 90, 200):
        np.testing.assert_array_equal(
            selector(epoch), curriculum_state(epoch, horizon, start_epoch=start, end_epoch=end)
        )


def test_curriculum_max_steps_above_horizon_is_clamped() -> None:
    """A cap longer than the horizon degrades to the full-horizon ramp, not an error."""
    horizon = 20
    assert _length(_step_mask_selector(horizon, 999, 0, 80)(1000)) == horizon
