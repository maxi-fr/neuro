# 09 — Torch-free checkpoint reader and the CasADi adapter

**What to build:** A torch-free checkpoint reader yields numpy weights + metadata from the
numpy-checkpoint; a thin adapter rebuilds the existing CasADi bridges from those buffers. The
controller and validation read checkpoints, not npz, and the control path stays torch-free (the
torch-import guard test keeps passing with the new modules added to its parametrisation).

Parity is dtype-split: the adapter's symbolic rollout equals the checkpoint's float64 rollout to
~1e-10 across activations and horizons; the float32 module equals the checkpoint to ~1e-5.

**Blocked by:** 07, 08

## Acceptance criteria

- [ ] The adapter's symbolic rollout equals the checkpoint's float64 rollout to ~1e-10 across activations and horizons.
- [ ] Checkpoint save/load round-trips weights, standardizers, and recorded metadata.
- [ ] `test_control_path_never_imports_torch` passes with the new modules in its parametrisation.
- [ ] Controller construction and validation read checkpoints; closed-loop evaluation reads checkpoints.
