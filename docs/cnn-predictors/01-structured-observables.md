# 01 — Preserve structured Observable data

**What to build:** Observable datasets and training retain separate EEG channel and frequency dimensions. Existing MLP, ridge, and DMD workflows retain their numerical behavior. Waveform datasets keep their existing dimensions. This is the prerequisite representation refactor for the CNN options.

**Blocked by:** None — can start immediately.

## Acceptance criteria

- [x] Observable trajectories have time, channel, and frequency axes; batched histories and targets additionally have a batch axis. Control Currents retain time and control axes.
- [x] Standardization preserves channel-frequency shape and the existing per-output or global statistics.
- [x] The MLP flattens measurements internally and returns structured Observable predictions. Losses, evaluation, plotting, ridge, and DMD callers consume the new representation without changing their intended calculations.
- [x] The controller state remains flat. Reshaping is explicit at the inference interface, with checkpoint standardizers and geometry handled consistently.
- [x] Regression tests establish numerical equivalence, history/control alignment, and training/evaluation coverage; relevant existing tests pass before and after the refactor.

## Validation

The focused regression command passed 113 tests:

```text
$env:MPLBACKEND='Agg'; uv run --extra cpu pytest tests/test_transforms.py tests/test_prediction.py tests/test_predictor_protocol.py tests/test_checkpoint_reader.py tests/test_dmd.py tests/test_ridge_trainer.py tests/test_predictor_losses.py tests/test_predictor_train.py tests/test_unified_trainer.py tests/test_predictor_plotting.py -q
113 passed
```

The changed files also pass:

```text
uv run ruff check <changed source and test files>
uv run ty check <changed source files>
```

The broader predictor/controller run reached 123 passed and stopped at the unrelated validation
fixture `tests/test_validation.py::test_jansen_rit_oracle_loop_validates_without_a_checkpoint`,
which requires the absent `data\healthy_lfp_knot20ms_frame500ms.npz`.
