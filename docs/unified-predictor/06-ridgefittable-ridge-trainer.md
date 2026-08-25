# 06 — RidgeFittable capability and the Ridge Trainer

**What to build:** The **Ridge-Fittable** capability protocol and a generic Ridge Trainer. This
amends the spec's decision-5 signature to normal-equation form so the ESN can stream without
materialising a full design matrix:

```text
design_normal_equations(trajectories) -> (G (f, f), P (f, c))   # bias column last
install_readout(A (c, f)) -> None
```

The Ridge Trainer is `G, P = model.design_normal_equations(trajs); A = ridge(G, P, λ);
model.install_readout(A)`, where `ridge` leaves the bias column unregularized. Three arms, all
"fixed feature map plus a linear readout":

- **depth-0 waveform MLP**: features are the one-step inputs, `A` is the single layer (reproduces
  today's warm-start least-squares at `λ = 0`).
- **ESN**: streams `[h; 1]` into `G`/`P` like the incumbent harvest; `A` is `W_out`.
- **depth-0 observable MLP** (`lift_depth = 0`, `transition_depth = 0`): features are the per-Frame
  lifted state `z_m`; `A` is the shared readout.

A non-fittable model handed to the Ridge Trainer fails at build time.

**Blocked by:** 01, 02, 03, 04

## Acceptance criteria

- [ ] Depth-0 MLP ridge fit reproduces the incumbent warm-start least-squares.
- [ ] ESN `design_normal_equations` reproduces the incumbent harvest; `install_readout` writes `W_out`.
- [ ] Depth-0 observable ridge fits the shared readout on harvested `z_m`.
- [ ] A non-fittable model + Ridge Trainer fails at build time.
