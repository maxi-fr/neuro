# 03 — ESN becomes a torch module satisfying the protocol

**What to build:** The ESN predictor becomes a torch module. Reservoir generation stays the
one-time scipy preprocessing step; its outputs (sparse `W_res`, dense `W_in`) are copied into torch
buffers. The runtime — `absorb`, `readout`, `step`, `rollout` — runs in torch, and the ridge
readout solve uses `torch.linalg`. Standardizers are buffers; raw units at the boundary.

The opaque state carries the reservoir vector and a step counter so `is_ready` can compare against
`priming_steps` without an ndarray subclass. The numpy `ESNPredictor`/`ESNArtifact` remain until
the contract ticket.

**Blocked by:** 01

## Acceptance criteria

- [ ] The torch ESN module satisfies the protocol; `rollout_many` returns `(B, n_positions, n_outputs)`.
- [ ] Torch `absorb`/`step`/`rollout` match the numpy ESN runtime to float tolerance (prior art: ESN and batched-rollout tests).
- [ ] The torch ridge solve reproduces the incumbent `solve_ridge`, including the unregularized bias column.
- [ ] `prime`/`initial_state`/`is_ready` respect `priming_steps`.
