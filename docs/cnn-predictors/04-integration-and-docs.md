# 04 — Verify and document the CNN options

**What to build:** Provide usable waveform and Observable CNN examples and document the architecture choices, structured dataset contract, and measured verification results. Complete integration checks across both Predictor representations.

**Blocked by:** 03 — Train and deploy a time–frequency Observable CNN.

## Acceptance criteria

- [ ] Example configurations for waveform and Observable CNNs load successfully and select the intended architecture through the normal training entry point.
- [ ] Training documentation explains dataset dimensions, channel mixing, causal time kernels, frequency preservation, receptive field, residual prediction, and gradient-descent-only fitting.
- [ ] Numerical checks verify JAX state/control derivatives against finite differences on small smooth models and establish usable controller gradients for both CNN representations.
- [ ] Focused smoke tests cover training through saved checkpoint and controller loading. MLP configurations remain usable.
- [ ] Documentation reports parameter counts only where actually computed; no model-quality, speed, or memory improvement is claimed without measurements. No benchmark or hyperparameter Sweep is required by this implementation.
- [ ] The complete repository pre-commit gate passes, including tests, type checking, formatting, and documentation checks. Ticket checklists reflect the verified result.
