# 02: Complete waveform Costs

**What to build:** Extend the configured continuous waveform controller to use every Cost option offered by the existing waveform problem builder.

**Blocked by:** 01: Continuous waveform MLP.

## Acceptance criteria

- [x] Waveform tracking supports the distinct terminal weight, and the controller can combine tracking, quadratic effort, smooth L1 effort, and the spectral Observable Frame hinge.
- [x] CasADi and the current implementation agree on every enabled Cost contribution and the total Cost for fixed predicted sequences, including terminal scoring and whole-horizon Frame terms.
- [x] The spectral Cost preserves the existing Frame geometry, reference normalization, and treatment of history preceding the Control Horizon.
- [x] The configured controller returns feasible physical Control Currents and reports each Cost contribution separately.
