# 08 — Ridge routing, ESN entry, and the `training.fit` build-time check

**What to build:** The unified entry point gains `training.fit` dispatch: `gradient_descent`
routes to the gradient-descent Trainer, `ridge` to the Ridge Trainer. The ESN config enters through
`ridge`. A fit the model does not support fails at build time. ESN candidates
`{rollout_nmse, log_energy}` and checkpoint save join the same entry point.

**Blocked by:** 06, 07

## Acceptance criteria

- [ ] `training.fit: ridge` on a depth-2 MLP fails at build time.
- [ ] `training.fit: gradient_descent` on the ESN fails at build time.
- [ ] An ESN config through `train` produces a checkpoint and candidates.
