# 01 — Waveform controller end-to-end

**What to build:** A working closed-loop MPC controller for the waveform predictor, built on
trajopt's primitives directly rather than on `trajopt.simulate.TrajOptMPC` — the tracer bullet for
the whole refactor. This means a trajopt `AbstractModel` adapter wrapping the waveform MLP
predictor's checkpoint (decision 1 of `spec.md`), and our own `simulate.Controller[L]` subclass
(decision 2) that owns the true absorbed predictor state and `u_last` as instance attributes, calls
`predictor.absorb` before `state.with_measurement`, then `problem.solve` → extract the first
control → `.shift(dt)`, mirroring `TrajOptMPC.update`'s loop shape without instantiating
`TrajOptMPC` itself. The `Objective` for this ticket is quadratic-only (EEG power + control effort
via `QuadraticCostFunction`/`Objective.tracking`) and the `ConstraintList` is box bounds only —
the hinge/L1 costs and the Kirchhoff constraint are out of scope here (ticket 02).

**Blocked by:** None — can start immediately.

## Acceptance criteria

- [ ] A trajopt `AbstractModel` (`DiscreteDynamics`/`EuclideanModel`) wraps the waveform MLP
      predictor, with weights loaded from its numpy-readable checkpoint into Equinox buffers.
- [ ] The adapter's `discrete_dynamics`/rollout matches the torch Predictor's `step`/`rollout` to
      floating-point tolerance on synthetic (non-trained) weights.
- [ ] A new `simulate.Controller[L]` subclass exists (not a wrapper/subclass of `TrajOptMPC`),
      built directly on `Problem`, `Objective`, `ConstraintList`, `MPCState`, and a solver backend.
- [ ] The controller's `update(t, ref, x_hat)` treats `x_hat` as the raw new measurement (today's
      semantics), calls `absorb` itself, and persists the resulting state plus `u_last` as private
      instance attributes — never reading them back from a post-solve trajectory.
- [ ] The controller reproduces today's `MPCController.update` control sequence, restricted to the
      quadratic-cost/box-constraint terms, on a fixed scripted trajectory.
- [ ] Commit the changes. Fix any pre-commit hook errors
