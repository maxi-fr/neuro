# 01 — Preserve structured Observable data

**What to build:** Observable datasets and training retain separate EEG channel and frequency dimensions. Existing MLP, ridge, and DMD workflows retain their numerical behavior. Waveform datasets keep their existing dimensions. This is the prerequisite representation refactor for the CNN options.

**Blocked by:** None — can start immediately.

## Acceptance criteria

- [ ] Observable trajectories have time, channel, and frequency axes; batched histories and targets additionally have a batch axis. Control Currents retain time and control axes.
- [ ] Standardization preserves channel–frequency shape and the existing per-output or global statistics.
- [ ] The MLP flattens measurements internally and returns structured Observable predictions. Losses, evaluation, plotting, ridge, and DMD callers consume the new representation without changing their intended calculations.
- [ ] The controller state remains flat. Reshaping is explicit at the inference interface, with checkpoint standardizers and geometry handled consistently.
- [ ] Regression tests establish numerical equivalence, history/control alignment, and training/evaluation coverage; relevant existing tests pass before and after the refactor.
