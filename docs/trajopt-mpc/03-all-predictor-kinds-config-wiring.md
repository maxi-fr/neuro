# 03 — All predictor kinds, wired through config

**What to build:** Extend ticket 01's `AbstractModel` pattern to the ESN and Observable predictor
kinds, and wire the resulting controller into the existing config/class-path dispatch pattern.
A `Problem`-building factory function per predictor kind (waveform/ESN/Observable) assembles the
`AbstractModel` + `Objective` + `ConstraintList` from checkpoint + geometry + the existing
cost-weight config fields (`w_y`, `w_u`, `w_u_l1`, `w_psd`, `u_max`, etc.) — no new config tree,
consistent with the unified-predictor spec's "dispatch by config type, not a `model_type` field"
decision. `configs/simulation/*mpc*.yaml` can then point `class_path` at the new controller for
any of the three predictor kinds, the same way it dispatches `neuro.control.nonlinear_mpc.
MPCController` today.

**Blocked by:** 02 — needs the full cost/constraint set to wire per predictor kind, not just the
quadratic/box subset.

## Acceptance criteria

- [ ] ESN and Observable predictor `AbstractModel` adapters exist, each rollout-parity tested
      against the corresponding torch Predictor's `rollout`, same tolerance bar as ticket 01's
      waveform adapter.
- [ ] Both new adapters are drivable through the ticket 01/02 controller with no controller-side
      changes needed.
- [ ] A `Problem`-building factory function exists per predictor kind, assembling `AbstractModel` +
      `Objective` + `ConstraintList` from checkpoint + geometry + existing cost-weight config
      fields.
- [ ] `configs/simulation/*mpc*.yaml` can dispatch `class_path` at the new controller and the
      appropriate factory function for any of the three predictor kinds, with no new config schema.
- [ ] A migrated config produces the same control sequence as the equivalent pre-migration
      CasADi-based config on a fixed scripted trajectory, for at least one predictor kind end to
      end through config loading (not just direct instantiation).
- [ ] Commit the changes. Fix any pre-commit hook errors
