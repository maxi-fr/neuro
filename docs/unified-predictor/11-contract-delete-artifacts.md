# 11 — Contract: delete artifacts, numpy runtimes, dispatch entry point, model protocols

**What to build:** Delete the three artifacts and their numpy runtimes, the artifact-dispatch entry
point, and the `SymbolicModel`/`ObservableModel` protocols. Scripts read/write checkpoints;
evaluation reads the torch module. Nothing reads `.npz` model files anymore.

**Blocked by:** 09, 10

## Acceptance criteria

- [ ] No references to the three artifact classes or to npz-model loading remain.
- [ ] The run/train/sweep scripts read and write checkpoints.
- [ ] The full test suite is green.
