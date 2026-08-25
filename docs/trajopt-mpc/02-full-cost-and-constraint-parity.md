# 02 — Full cost and constraint parity

**What to build:** Bring ticket 01's controller up to today's full cost and constraint set. The
spectral PSD hinge and the Observable log-reference hinge become custom `CostFunction` subclasses
(`evaluate(x, u, t) -> scalar`, JAX-autodiff'd — `jnp.fft` replaces the hand-rolled DFT matrix
directly), added alongside the quadratic tracking term in the controller's `Objective`. The
Observable hinge keeps reading `ObservableGeometry` as the shared source of truth with the
training-time `Loss`, per the unified-predictor spec's Loss↔Cost correspondence — only its host
type changes. The Kirchhoff sum-to-zero constraint is added via `trajopt.constraints.linear`,
alongside the existing box bounds.

**Blocked by:** 01 — needs the controller and its `Objective`/`ConstraintList` wiring to extend.

## Acceptance criteria

- [ ] PSD spectral hinge and Observable log-reference hinge are implemented as custom
      `CostFunction` subclasses and plugged into the controller's `Objective` from ticket 01.
- [ ] Both custom costs evaluate to the same values as today's CasADi cost graph on fixed inputs,
      scoring the same `ObservableGeometry`-derived reference.
- [ ] The Kirchhoff sum-to-zero constraint is added via `trajopt.constraints.linear` to the
      controller's `ConstraintList`, alongside the box bounds from ticket 01.
- [ ] The controller reproduces today's `MPCController.update` control sequence with the full cost
      and constraint set (quadratic + hinge/L1 + box + Kirchhoff) on a fixed scripted trajectory.
- [ ] Whether to keep a smooth L1 surrogate for native-JAX solvers, or restrict L1 sparsity to
      transcription-based solvers, is decided with data from this ticket's testing, not upfront.
- [ ] Commit the changes. Fix any pre-commit hook errors
