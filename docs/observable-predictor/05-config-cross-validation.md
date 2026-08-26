# 05 — Config cross-validation

**What to build:** The geometry knobs are all config values, and a plausible-looking config can now
be quietly wrong in ways that surface as a silently degraded loop rather than an error: a control
window too short to see the Control Currents that shaped a Frame, a controller ticking off the hop
grid so the same Frame is absorbed twice, an envelope measured at a geometry the model was not
trained at. This ticket makes each of those fail at build time with a message naming the offending
values. It comes last because every rule needs its subject to exist first.

**Blocked by:** 03 — The observable Predictor trains and runs; 04 — The observable closed loop.

## Acceptance criteria

- [ ] The control-support rule holds: `n_u >= kernel_width - 1 + ceil(segment / hop)`, so the past-
      control window covers the whole sample support of the Frame being predicted.
- [ ] The controller's update period equals `hop * downsample * plant_dt`, so exactly one fresh
      Frame is absorbed per tick.
- [ ] The Estimator runs at plant rate, and the sensors feeding it do too.
- [ ] Band, pooling and Frame Kernel agree with the geometry the healthy envelope records.
- [ ] The envelope's channel count and sampling rate agree with the model's.
- [ ] The standardizer's length agrees with the model's output width.
- [ ] The curriculum Span holds at least one Frame at the configured geometry.
- [ ] The Observable geometry in the config agrees with the geometry the checkpoint records, so a
      model is never run against a reduction it was not trained on.
- [ ] Loss terms that carry their own reduction are rejected on the observable kind.
- [ ] A Sweep objective is validated against the observable candidate set when the configured model
      is the observable kind.
- [ ] Every rule has a test that rejects a violating config and accepts a conforming one, and each
      error message names the values that conflict.
