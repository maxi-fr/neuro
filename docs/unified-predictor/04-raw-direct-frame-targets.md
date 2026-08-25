# 04 — Raw-direct Frame targets in the data pipeline

**What to build:** The Observable projection moves into the data pipeline. Frame targets are
computed raw-direct on the loaded trajectories — no standardize→inverse round trip — on the same
window grid the waveform targets use. `log_observable` becomes the single target path; the
round-trip `build_targets` goes away.

**Blocked by:** None — can start immediately.

## Acceptance criteria

- [ ] Raw-direct Frame targets equal today's round-trip `build_targets` output to floating-point tolerance.
- [ ] The resolved Frame count matches the geometry's own derivation across a range of horizons.
- [ ] Targets are produced for both the training and validation splits on the shared window grid.
