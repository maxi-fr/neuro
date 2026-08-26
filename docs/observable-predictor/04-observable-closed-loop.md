# 04 — The observable closed loop

**What to build:** The loop runs on Frames. A new multi-rate Estimator low-passes the raw EEG at
plant rate, decimates, buffers Segments and emits raw log-power Frames at the hop rate, holding
nothing back that the offline reduction would have produced differently. A one-sided log-power hinge
scores the predicted Frames against the healthy envelope without touching a model, and a problem
builder assembles it with the control penalties, the Control Budget bounds and the Kirchhoff
equality over a Control Horizon counted in Frames. The controller absorbs one Frame per hop, holds
zero control through the Warm-up Period, and then emits a Control Current every hop. Demoable as a
closed-loop run from a config.

**Blocked by:** 01 — Canonical Observable reduction and its healthy envelope; 02 — Kind-agnostic
Costs and model core; 03 — The observable Predictor trains and runs.

## Acceptance criteria

- [ ] The Estimator low-passes at plant rate, decimates, buffers Segments and emits raw log-power
      Frames at the hop rate by calling the canonical reduction, carrying no learned parameters.
- [ ] It returns an unprimed NaN Frame until `segment + (kernel_width - 1) * hop` decimated samples
      have arrived, and holds the last Frame between hops.
- [ ] Fed one raw trajectory, the Frames it emits equal the ones the offline reduction produces for
      training targets, to machine precision, on the same hop grid.
- [ ] A one-sided log-power hinge, built from geometry, the healthy envelope and center/scale with
      no model instance, matches a NumPy reference of the same computation on predicted Frames.
- [ ] The problem builder assembles the hinge with L1 control, quadratic control, Control Budget box
      bounds and the Kirchhoff sum-to-zero equality, over a Control Horizon counted in Frames, and
      the resulting problem solves.
- [ ] A closed-loop run primes the Predictor, emits zero control throughout the Warm-up Period, then
      emits a finite Control Current on every hop thereafter.
- [ ] An example config runs the loop start to finish.
- [ ] CONTEXT.md carries an Estimator entry.
