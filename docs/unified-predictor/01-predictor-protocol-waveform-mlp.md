# 01 — Predictor protocol and the waveform MLP runtime surface

**What to build:** The runtime-only, raw-units **Predictor** protocol becomes a real,
framework-agnostic type (no torch import), and the autoregressive waveform MLP becomes its first
implementor. Callers never see standardized space; **State** is opaque and the module's own
business. The MLP gains channel/control standardizers as buffers and the full runtime surface —
`prime`, `step`, `rollout`, `absorb`, `is_ready`, `initial_state`, plus the batched
`prime_many`/`rollout_many` — while `to_artifact`/`from_artifact` keep producing the existing
artifact, so the controller and evaluation stay green.

The protocol (decision-rich, from the spec):

```text
prime(y_hist, u_hist) -> state
step(state, u) -> (state', output)          # one position: one sample
rollout(state, u_future) -> (n_positions, n_outputs)
prime_many(y_hists, u_hists) -> (B, state)
rollout_many(states, u_futures) -> (B, n_positions, n_outputs)
absorb(state, y, u) -> state                # State Absorption
is_ready(state) -> bool                     # Priming completeness
initial_state() -> state
# identity: n_channels, n_controls, n_outputs, dt, priming_steps, horizon
```

- `horizon` is the native/trained horizon (identity + validation warning), not a hard bound on the
  length of `u_future` accepted by `rollout`.
- The MLP's opaque state is a shift register: standardized EEG window + raw control window.
- `initial_state()` returns a NaN-padded EEG window; `is_ready` is "no NaN in the EEG window".
- `step`/`rollout` emit raw EEG (decode internally); the existing batched `forward` stays in
  standardized space for training.

**Blocked by:** None — can start immediately.

## Acceptance criteria

- [ ] The waveform MLP satisfies the protocol; `rollout_many` returns `(B, n_positions, n_outputs)`.
- [ ] Raw units round-trip at the boundary: `rollout`/`step` decode, `prime`/`absorb` encode, standardizers stay internal.
- [ ] `prime_many` equals a loop of `prime`, `rollout_many` equals a loop of `rollout`, and `rollout` equals a loop of `step` (float32 tolerance).
- [ ] `absorb`/`is_ready`/`initial_state` reproduce the incumbent NaN-padded shift-register behaviour.
- [ ] `to_artifact`/`from_artifact` still round-trip bit-exactly; existing waveform and torch↔CasADi tests stay green.
