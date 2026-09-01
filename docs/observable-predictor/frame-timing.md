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
`ceil(S / hop) = kernel_width - 1 + ceil(segment / hop)`, and `n_u` must not fall below it.
Causality says only `u_k` reaches the newest hop of `o_{k+1}`; it does not say `u_k` suffices to
predict it. The rest of the support was driven by `u_{k-1}` and older, and dropping those from the
features is safe only if `o_k` screens them off, that is
`E[o_{k+1} | o_k, u_k] = E[o_{k+1} | o_k, u_{k-1}, u_k]`. A true state would. The Frame is a
power-only, phase-blind reduction, and no number of Frame lags recovers which current was applied
inside the last `S` samples.

What the omission costs is bias, not blur. `u_{k-1}` is correlated with `u_k` under any
autocorrelated excitation and more so in closed loop, so its effect loads onto the coefficient of
`u_k`: the fitted sensitivity `d o_{k+1} / d u_k` comes out as the direct effect plus a confounding
term. That derivative is what the MPC differentiates to size `u_k`, and the bias tracks the
correlation structure of whatever excitation generated the training set, so refitting on different
data moves it rather than exposing it. Hence a hard check at config build and again at checkpoint
load, rather than a note about accuracy.

The floor belongs to the observation operator, not the dynamics: the reduction window holds samples
recorded while `u_{k-1}` was applied, so a memoryless plant would still need them, and
`min_past_controls()` takes neither `fs` nor anything about the Plant. The waveform arm carries no
such rule, because there the `n_y` lags form a delay embedding that stands in for the state and
past controls are redundant. Plant memory outliving `S` is a separate reason to raise `n_u`, which
this floor does not cover.

**`frame_supports` indexes by start while this convention stamps by end.** It returns
`(m * hop, (m + kernel_width - 1) * hop + segment)`, and Frame `m`'s timestamp is that span's end.
Pairing a target Frame with the Control Current governing its newest hop therefore goes through
`end - hop`, not through the span's start.

**Only `hop / S` of a predicted Frame is actionable at the step that predicts it.** The first
`ceil(S / hop) - 1` predicted Frames are dominated by signal already realized at `t_k`, so the
control gradient on them is attenuated by roughly `hop / S`. The actionable lookahead is shorter
than the Control Horizon, which matters when sizing the horizon.
