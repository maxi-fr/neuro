# 10 — Sweep seam: Optuna, objective from candidates

**What to build:** `OptunaSweep` serves the two predictor kinds (waveform and observable).
`sweep.objective` is a named string validated against the Trainer's candidates; every candidate is
recorded on every trial so a finished study can be re-ranked; `closed_loop` is available as an
objective for both predictor kinds.

**Blocked by:** 07, 08, 09

## Acceptance criteria

- [ ] The sweep picks the config-named objective from the Trainer's candidates and records every candidate per trial.
- [ ] `closed_loop` works as an objective for the waveform and observable predictors.
- [ ] Tests assert the selection wiring, not Optuna internals.
