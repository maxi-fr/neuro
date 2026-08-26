# 05 — Evaluation moves onto the protocol

**What to build:** Free-run evaluation — `rollout_batches`, `accumulate_rollout_errors`,
`evaluate_rollouts`, `evaluate_log_energy` — operates generically on the Predictor protocol via the
batched `prime_many`/`rollout_many`, instead of on artifacts, and moves to a torch-importing
module. The artifact-dispatch entry point keeps only its loader/dispatch role until the contract
ticket.

**Blocked by:** 01, 03

## Acceptance criteria

- [ ] The batched accumulator reproduces the per-window `prime`/`rollout` loop for both the MLP and observable modules.
- [ ] `evaluate_rollouts`/`evaluate_log_energy` score a module identically to its artifact twin.
