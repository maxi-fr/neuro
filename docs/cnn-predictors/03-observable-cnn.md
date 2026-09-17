# 03 — Train and deploy a time–frequency Observable CNN

**What to build:** The CNN option also predicts spectral Observable Frames, learning local time–frequency patterns and cross-channel interactions without discarding absolute frequency location.

**Blocked by:** 02 — Train and deploy a waveform CNN Predictor.

## Acceptance criteria

- [x] Structured Observable histories enter a 2D convolution with EEG channels as input features and time/frequency as the sliding axes.
- [x] Temporal padding is causal; frequency padding preserves bin count. Frequency kernel size is configurable. Stride is one and no pooling discards frequency identity.
- [x] Features at the latest temporal position retain frequency positions and enter the dense prediction head alongside Control Current history. Output and residual shapes match the next structured Frame.
- [x] A singleton frequency dimension, including pooled Observable geometry, works without special user action.
- [x] Training, checkpoint reload, JAX inference, evaluation, plotting, and Observable controller construction work through the same architecture selection used for waveforms.
- [x] Tests cover multiple EEG channels/frequency bins, singleton frequencies, short histories, residual behavior, action alignment, and PyTorch/JAX Rollout parity. Existing Observable MLP/ridge/DMD coverage stays green.

Validation recorded 2026-09-17:

- `uv run pytest tests/test_waveform_cnn.py tests/test_observable_cnn.py tests/test_predictor_protocol.py tests/test_checkpoint_reader.py tests/test_dmd.py tests/test_mpc.py -x` — 90 passed.
- `uv run ruff check src/neuro/config.py src/neuro/control/mpc.py src/neuro/predictor/inference.py src/neuro/predictor/module.py src/neuro/predictor/train.py tests/test_observable_cnn.py` — passed.
- `uv run ty check src/neuro/predictor/inference.py src/neuro/predictor/module.py src/neuro/predictor/train.py` — passed.
