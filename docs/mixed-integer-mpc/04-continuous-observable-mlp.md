# 04: Continuous Observable MLP

**What to build:** A configured continuous CasADi controller that advances an Observable MLP Predictor on its native Frame step.

**Blocked by:** 01: Continuous waveform MLP.

## Acceptance criteria

- [x] The controller accepts an Observable MLP checkpoint and applies one zero-order-held Control Current per Predictor step after State Absorption.
- [x] The configured Cost supports the Observable hinge, quadratic current effort, and smooth L1 current effort, with the existing reference and terminal scoring.
- [x] CasADi and the current Predictor agree on fixed-sequence outputs, including Observable Frame timing and shape; each Cost contribution agrees on fixed predicted sequences.
- [x] The controller exposes a feasible physical plan, decomposed Cost, status, success flag, solve time, and shifted warm start through the same update contract.
