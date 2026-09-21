def curriculum_span(steps: int, start: int, end: int, epoch: int | None) -> int:
    """Return the rounded curriculum Span, using the full Span for validation."""
    if epoch is None:
        return steps
    fraction = min(max((epoch - start) / max(end - start, 1), 0.0), 1.0)
    return round(1 + (steps - 1) * fraction)


def curriculum_completion(steps: int, start: int, end: int) -> int:
    """Return the first epoch whose effective curriculum Span is complete."""
    if steps <= 1:
        return 0
    epoch = start
    while curriculum_span(steps, start, end, epoch) < steps:
        epoch += 1
    return epoch
