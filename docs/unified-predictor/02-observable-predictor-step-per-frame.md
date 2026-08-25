# 02 — Observable predictor rebuilt as a one-Frame-per-step module

**What to build:** A from-scratch Observable-space predictor that satisfies the protocol directly:
no one-shot `forward([history | future controls]) -> all frames`, no artifact hand-off. Its
position is the **Frame**.

- `prime` builds the opaque history state.
- `step(state, u_bar)` advances exactly one Frame (lifting once if needed, then the shared Frame
  transition + readout) and emits that Frame's raw log-Observable `(n_channels * n_values,)`.
  `u_bar` is one Frame-mean `(n_controls,)`.
- `rollout(state, u_future)` unrolls `step` over the Frame grid, aggregating raw controls into
  Frame means via the shared `control_means` helper; it takes raw samples and returns
  `(n_frames, n_channels * n_values)`.
- `absorb`/`is_ready`/`initial_state` mirror the waveform module's shift-register discipline
  (raw EEG history, NaN readiness).
- Channel/control/log-Observable standardizers are buffers; raw units at the boundary.
- A standardized batched `forward` that unrolls the same step math (BPTT depth = `n_frames`) stays
  for training.

State is opaque and carries both the history register (for `absorb`/`is_ready`) and the lifted
Frame state (for `step`/`rollout`); the module lifts once at `prime`/`absorb`.

This is expand: the incumbent observable module and artifact remain in place for the controller
until the contract ticket. This ticket introduces the new module and its protocol tests.

**Blocked by:** 01

## Acceptance criteria

- [ ] The new observable module satisfies the protocol; `step` advances one Frame, `rollout` returns `(n_frames, n_channels * n_values)` raw log-Observable.
- [ ] `step` consumes a Frame-mean `(n_controls,)`; `rollout`/`absorb` consume raw samples and `rollout` aggregates via `control_means`.
- [ ] `prime`/`absorb`/`is_ready`/`initial_state` behave like the waveform shift register.
- [ ] The standardized batched `forward` unrolls the same Frame recursion; its inverse-standardized output matches `rollout` (float32 tolerance).
- [ ] Frame count follows the geometry across horizons, not a frozen head.
