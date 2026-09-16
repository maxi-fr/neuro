# 03 — Train and deploy a time–frequency Observable CNN

**What to build:** The CNN option also predicts spectral Observable Frames, learning local time–frequency patterns and cross-channel interactions without discarding absolute frequency location.

**Blocked by:** 02 — Train and deploy a waveform CNN Predictor.

## Acceptance criteria

- [ ] Structured Observable histories enter a 2D convolution with EEG channels as input features and time/frequency as the sliding axes.
- [ ] Temporal padding is causal; frequency padding preserves bin count. Frequency kernel size is configurable. Stride is one and no pooling discards frequency identity.
- [ ] Features at the latest temporal position retain frequency positions and enter the dense prediction head alongside Control Current history. Output and residual shapes match the next structured Frame.
- [ ] A singleton frequency dimension, including pooled Observable geometry, works without special user action.
- [ ] Training, checkpoint reload, JAX inference, evaluation, plotting, and Observable controller construction work through the same architecture selection used for waveforms.
- [ ] Tests cover multiple EEG channels/frequency bins, singleton frequencies, short histories, residual behavior, action alignment, and PyTorch/JAX Rollout parity. Existing Observable MLP/ridge/DMD coverage stays green.
