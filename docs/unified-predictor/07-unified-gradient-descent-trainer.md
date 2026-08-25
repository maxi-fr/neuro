# 07 — Unified gradient-descent Trainer (waveform + observable)

**What to build:** One `train(cfg, data_files)` for the two NN predictors, dispatching on config
type, returning the trained Predictor plus `candidates` and a `save` that round-trips.
Gradient-descent serves any torch module over the protocol. Candidates: waveform
`{log_energy, val_loss, rollout_nmse}`, observable `{val_loss, val_log_mse}`. `save` writes the
numpy-checkpoint plus training stats.

**Blocked by:** 02, 04, 05

## Acceptance criteria

- [ ] `train` returns Predictor + `candidates` + `save`; save round-trips weights, standardizers, and recorded metadata.
- [ ] Waveform and observable paths converge end-to-end on tiny synthetic trajectories.
- [ ] `candidates` contains the config-named objective.
