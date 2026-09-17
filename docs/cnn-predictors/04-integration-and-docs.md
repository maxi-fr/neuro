# 04 — Verify and document the CNN options

**What to build:** Provide usable waveform and Observable CNN examples and document the architecture choices, structured dataset contract, and measured verification results. Complete integration checks across both Predictor representations.

**Blocked by:** 03 — Train and deploy a time–frequency Observable CNN.

## Acceptance criteria

- [x] Example configurations for waveform and Observable CNNs load successfully and select the intended architecture through the normal training entry point.
- [x] Training documentation explains dataset dimensions, channel mixing, causal time kernels, frequency preservation, receptive field, residual prediction, and gradient-descent-only fitting.
- [x] Numerical checks verify JAX state/control derivatives against finite differences on small smooth models and establish usable controller gradients for both CNN representations.
- [x] Focused smoke tests cover training through saved checkpoint and controller loading. MLP configurations remain usable.
- [x] Documentation reports parameter counts only where actually computed; no model-quality, speed, or memory improvement is claimed without measurements. No benchmark or hyperparameter Sweep is required by this implementation.
- [x] The complete repository pre-commit gate passes, including tests, type checking, formatting, and documentation checks. Ticket checklists reflect the verified result.

Validation recorded 2026-09-17:

- Example YAMLs load through `neuro.config.load_config` as `architecture: cnn`; both use CPU-capable gradient descent and preserve the existing data path convention.
- Waveform and Observable CNN controller state/control finite-difference Jacobian tests pass with scoped JAX float64 and nonzero predicted-output Control Current derivatives. The focused integration run (`tests/test_waveform_cnn.py tests/test_observable_cnn.py tests/test_output_functions.py tests/test_checkpoint_reader.py tests/test_predictor_protocol.py`) passed 73 tests.
- Generic inference loading now covers CNN and MLP checkpoints in plotting, control-authority, STFT-geometry, and rollout-horizon probes.
- The full pre-commit gate passed cleanly across all repository checks and tests (756 passed, 7 skipped).
