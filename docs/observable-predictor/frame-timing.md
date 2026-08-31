# Frame and Control Timing

One convention fixes how Observable Frames, Control Currents and the model's step index line up.
Estimator latency, the dataset's target/control pairing and the physical reach of the Control
Horizon are all derived from it rather than chosen separately.

## The convention

On the decimated grid, with `hop` the Frame spacing and

    S = (kernel_width - 1) * hop + segment

the samples one Frame reduces (`StftGeometry.sample_support_steps`):

- **A Frame is stamped at the end of its support.** `o_k` reduces `[t_k - S, t_k)`. It is the
  newest Frame available at `t_k`, so the controller reads it with no reduction latency.
- **A Control Current is stamped at the start of its hold.** `u_k` is held over `[t_k, t_k + hop)`.
- `t_k = k * hop`.

## Why it is well posed

`o_{k+1}` reduces `[t_k + hop - S, t_k + hop)`, which splits into exactly two parts:

| Interval | Status at `t_k` | Governed by |
| :--- | :--- | :--- |
| `[t_k, t_k + hop)` | the only samples new since `o_k` | `u_k` alone |
| `[t_k + hop - S, t_k)` | already realized | `u_{k-1}` and older |

No decision variable beyond `u_k` reaches `o_{k+1}`, so `o_{k+1} = f(o_k, u_k)` is causal. Each
step slides the support by one hop and exactly one Control Current governs the samples that
entered.

## What follows

**Estimator latency is `S`.** The first Frame lands at `t = S` on the decimated grid, i.e.
`S * downsample` plant samples plus the anti-alias filter's group delay; the state is unprimed
until then. Priming Steps are counted in Frames on top of that.

**The Control Horizon's reach is `horizon * hop`.** The predicted Frames `o_{k+1} .. o_{k+H}` are
stamped `t_k + hop .. t_k + H * hop`.

**`min_past_controls()` is the number of Control Currents still inside the support**, that is
`ceil(S / hop) = kernel_width - 1 + ceil(segment / hop)`. Those are past controls, which a true
state would already absorb -- but the Frame state is a power-only, phase-blind summary and cannot
be inverted for them, so an `n_u` below this makes the model reconstruct what it could be handed.

**`frame_supports` indexes by start while this convention stamps by end.** It returns
`(m * hop, (m + kernel_width - 1) * hop + segment)`, and Frame `m`'s timestamp is that span's end.
Pairing a target Frame with the Control Current governing its newest hop therefore goes through
`end - hop`, not through the span's start.

**Only `hop / S` of a predicted Frame is actionable at the step that predicts it.** The first
`ceil(S / hop) - 1` predicted Frames are dominated by signal already realized at `t_k`, so the
control gradient on them is attenuated by roughly `hop / S`. The actionable lookahead is shorter
than the Control Horizon, which matters when sizing the horizon.
