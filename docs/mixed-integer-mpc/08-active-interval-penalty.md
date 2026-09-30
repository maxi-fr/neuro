# 08: Active-interval penalty

**What to build:** Let either Predictor representation penalize enabled intervals through Bonmin while retaining its selected continuous Cost.

**Blocked by:** 06: Waveform active-interval cap; 07: Observable active-interval cap.

## Acceptance criteria

- [x] A configured enabled-step weight adds its weighted count to the Cost for waveform and Observable MLP and CNN Predictors.
- [x] Configuration accepts either a horizon-local cap or an enabled-step weight and rejects a combination of the two. Continuous mode uses neither.
- [x] For fixed feasible plans, the new Cost contribution equals the configured weight times the enabled-step count; all existing Cost contributions remain unchanged.
- [x] Returned plans respect activation coupling, per-electrode amplitude bounds, and Kirchhoff balance. Logs expose enabled-step counts and solver outcomes.
- [x] Shifted warm starts and failure behavior match the capped mode.
