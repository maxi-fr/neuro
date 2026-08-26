# 02 — Kind-agnostic Costs and model core

**What to build:** The prefactor that makes the existing waveform stack indifferent to what one
model position means, so the observable Predictor can be built on it rather than beside it. Two
changes, both behaviour-preserving. First, MPC Costs become model-free: a Cost is handed geometry,
the healthy envelope and center/scale arrays at build time, slices the state it is given, and never
holds a model or calls a model method — which also retires the output decode from the Predictor
surface. Second, the autoregressive MLP core stops assuming one position is one EEG sample across
channels, and carries an output width instead. The waveform Predictor comes out the far side
behaving exactly as it went in.

**Blocked by:** None — can start immediately.

## Acceptance criteria

- [x] No MPC Cost holds a model instance or calls a model method; each is constructed from geometry,
      the healthy envelope and center/scale arrays.
- [x] The waveform spectral hinge scores the Frames the stage trajectory already carries, and the
      terminal knot's contribution comes from an explicit terminal Cost rather than from stepping
      the model one extra time.
- [x] On a fixed problem and seed, the waveform MPC reports the same optimal cost and the same first
      Control Current as before the refactor.
- [x] The output decode is absent from the runtime protocol and from every runtime adapter, and no
      caller remains anywhere in the repo, the prediction-plotting script included.
- [x] The MLP core carries an output width independent of channel count, with the residual skip
      fitting the one-step delta over that width; the waveform Predictor is the case where the two
      are equal.
- [x] The Ridge-Fittable design and readout-install path fit a readout of the full output width.
- [x] The waveform checkpoint round-trips weights, standardizers and metadata unchanged, and the
      torch and jax sides still agree on a free run to floating-point tolerance.
