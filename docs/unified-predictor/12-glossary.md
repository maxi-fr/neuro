# 12 — Glossary: Trainer, Ridge-Fittable, Sweep

**What to build:** Add **Trainer**, **Ridge-Fittable**, and **Sweep** to `CONTEXT.md` in the
existing glossary voice, each with an `_Avoid_` line. The rest of the glossary (Predictor, Rollout,
State Absorption, Priming, Observable, Frame) is unchanged.

**Blocked by:** None — can start immediately.

## Acceptance criteria

- [ ] `CONTEXT.md` defines Trainer as a fit-named algorithm that asks the Predictor which fits it supports.
- [ ] `CONTEXT.md` defines Ridge-Fittable as a readout capability checked at build time.
- [ ] `CONTEXT.md` defines Sweep as the hyperparameter search over a Predictor's configuration by one Trainer-reported objective.
