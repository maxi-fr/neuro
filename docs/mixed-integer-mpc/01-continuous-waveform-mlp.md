# 01: Continuous waveform MLP

**What to build:** A configured CasADi IPOPT controller that uses a waveform MLP Predictor, applies the first planned Control Current, and exposes a comparable result to the existing trajopt controller.

**Blocked by:** None; can start immediately.

## Acceptance criteria

- [ ] Configuration constructs the continuous controller with waveform tracking, quadratic current-effort Cost, per-electrode amplitude bounds, and explicit Kirchhoff balance.
- [ ] State Absorption and the native controller step lead to an applied physical Control Current, predicted outputs, a planned Control Current sequence, decomposed Cost, solver status, success flag, and solve time.
- [ ] A successful solve shifts its Control Current plan into the next initial guess. A failed solve exposes its status and issues no new Control Current.
- [ ] CasADi and the current Predictor agree on one-step and Control Horizon outputs for fixed histories and Control Current sequences.
- [ ] On small, well-behaved continuous problems, CasADi IPOPT and trajopt IPOPT produce feasible plans with close first Control Currents and Costs.
