# 10 — Sweep seam: Optuna + Grid, objective from candidates

**What to build:** `OptunaSweep` serves the two NN predictors; `GridSweep` serves the ESN (outer
reservoir-size × ridge-λ grid, inner Optuna over the continuous reservoir hyperparameters).
`sweep.objective` is a named string validated against the Trainer's candidates; every candidate is
recorded on every trial so a finished study can be re-ranked; `closed_loop` is available as an
objective for all three predictor kinds.

**Blocked by:** 07, 08, 09

## Acceptance criteria

- [ ] Both sweeps pick the config-named objective from the Trainer's candidates and record every candidate per trial.
- [ ] `closed_loop` works as an objective for the waveform, observable, and ESN predictors.
- [ ] Tests assert the selection wiring, not Optuna internals.
