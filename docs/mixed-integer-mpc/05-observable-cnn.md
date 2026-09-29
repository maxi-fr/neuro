# 05: Observable CNN

**What to build:** Run the continuous Observable controller and its full Cost set with Observable CNN checkpoints.

**Blocked by:** 04: Continuous Observable MLP.

## Acceptance criteria

- [x] An Observable CNN checkpoint works through the same configuration and update contract as an Observable MLP checkpoint.
- [x] CasADi and the current Predictor agree on fixed-sequence Frame outputs, including convolution geometry, normalization, and residual behavior.
- [x] Each enabled Cost contribution matches the current implementation on fixed CNN predictions.
- [x] The configured controller produces a feasible physical plan on the native Frame step with amplitude bounds and Kirchhoff balance.
