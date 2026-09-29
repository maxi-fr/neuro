# 03: Waveform CNN

**What to build:** Run the continuous waveform controller and its full Cost set with waveform CNN checkpoints.

**Blocked by:** 02: Complete waveform Costs.

## Acceptance criteria

- [ ] A waveform CNN checkpoint works through the same controller configuration and update contract as a waveform MLP checkpoint.
- [ ] CasADi and the current Predictor agree on one-step and Control Horizon outputs for fixed histories and Control Current sequences, including normalization and residual behavior.
- [ ] Each enabled Cost contribution matches the current implementation on fixed CNN predictions.
- [ ] The configured controller produces a feasible plan with amplitude bounds and Kirchhoff balance.
