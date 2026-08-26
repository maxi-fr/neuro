# 03 — The observable Predictor trains and runs

**What to build:** A Predictor that forecasts the Observable directly, one Frame per step. Training
trajectories reduce to Frames, windows are built on the Frame grid, and the curriculum MSE scores
predicted standardized Frames with no differentiable STFT anywhere in the Loss. The trained model
checkpoints with the Observable geometry it was fitted at, and its jax runtime adapter advances one
Frame per call under the Control Current held over that hop, with State Absorption and Priming Steps
counted in Frames. At the end of this ticket the observable Predictor can be trained from a config
and free-run from a checkpoint; it is not yet wired to a controller.

**Blocked by:** 01 — Canonical Observable reduction and its healthy envelope; 02 — Kind-agnostic
Costs and model core.

## Acceptance criteria

- [ ] A trajectory reduces to Frames and windows onto the Frame grid, giving inputs of `n_y` past
      Frames plus `n_u` past Control Currents and targets of the future Frames the Span holds.
- [ ] The standardizer fits center and scale per output — one pair per (channel, bin) — on the
      training Frames, and is carried as checkpointed buffers.
- [ ] The curriculum MSE scores standardized Frames directly, and a Span declared in seconds
      resolves at the Frame rate rather than the sample rate.
- [ ] Training writes a checkpoint carrying weights, the per-output standardizer and the Observable
      geometry; loading round-trips all three.
- [ ] The runtime adapter advances one Frame per call, absorbs one raw Frame plus the held Control
      Current, reports readiness, exposes an unprimed initial state, and counts Priming Steps in
      Frames as the wider of the two history windows.
- [ ] From the same checkpoint, the adapter's free run equals the torch rollout to floating-point
      tolerance, raw Frames in and raw Frames out.
- [ ] Both the gradient-descent and the Ridge arms produce a checkpoint that satisfies the two
      criteria above.
- [ ] The Trainer reports a validation Frame MSE, and an observable Sweep trial can rank on it.
