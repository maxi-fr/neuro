# 02 — Train and deploy a waveform CNN Predictor

**What to build:** Users select a temporal CNN alongside the existing MLP and can train it, save it, reload it, evaluate free-running Rollouts, and use it in the controller. Share the autoregressive history machinery between the two architectures instead of copying the training and inference loops.

**Blocked by:** 01 — Preserve structured Observable data.

## Acceptance criteria

- [x] Configuration distinguishes MLP and CNN architecture while existing MLP configurations retain their behavior. CNN settings cover width, depth, and temporal kernel size without unrelated options.
- [x] The waveform CNN mixes EEG channels with learned weights and convolves along time with causal left padding, stride one, and no pooling. The default receptive field covers the default history.
- [x] The latest temporal features and flattened Control Current history feed a small dense prediction head. Residual prediction and the current action's timing follow the existing Predictor convention.
- [x] The existing gradient-descent Trainer and Losses support CNNs. Unsupported ridge/DMD fits fail clearly before training.
- [x] Checkpoints record sufficient architecture metadata and weights. JAX inference, checkpoint dispatch, evaluation, plotting, and controller construction support the CNN.
- [x] Tests verify configuration, training, checkpoint round trips, PyTorch/JAX one-step and multi-step agreement, and controller construction while keeping MLP regressions green.

Validation recorded 2026-09-17:

- Historical ticket-02 run: 745 passed, 7 skipped before the Observable CNN work.
- Post type-check edits: 56 focused tests passed; Ruff and ty passed on the changed source and test files.
