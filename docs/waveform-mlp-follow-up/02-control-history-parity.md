# 02. Align training, offline Rollouts, and MPC input timing

**What to build:** Identical measurement histories and future Control Currents produce equivalent predictions through the training forward pass, offline Rollout evaluation, and MPC dynamics. Offline error reports then describe the Predictor that MPC actually uses.

**Blocked by:** None. Can start immediately.

**Status:** Ready for implementation. Not started.

## Evidence and scope

Training and MPC insert the current candidate Control Current before predicting the next output. The offline Rollout helper predicts first and inserts that Current afterward. The OOD error analysis uses that helper. A constant control sequence can conceal this discrepancy.

Keep the deployed convention: the measurement at a decision incorporates preceding applied inputs, and the newly selected input acts on the next transition. Audit the offline callers as well as the helper so a caller's compensating offset does not become a second shift. Do not retrain checkpoints to conceal an evaluation indexing error.

## Preserve existing experiments

Existing models, checkpoints, sweeps, trials, configs, run logs, metrics, plots, and reports must not be overwritten or deleted. Write new outputs either to a new experiment folder or to a uniquely named addition within an existing experiment. Treat old artifacts as read-only inputs. Appending to an experiment means adding new runs or trial identities and additive index entries; it does not mean resuming into occupied trial IDs, replacing a best-model file, or regenerating an old summary in place. If an output destination already exists, stop before writing and choose a new destination. Retain before-and-after hashes for the existing artifacts used by the experiment.

## Acceptance criteria

- [ ] A deterministic test uses changing, channel-distinct inputs and a Predictor with a known immediate control response. It exposes the current one-step discrepancy.
- [ ] Training and runtime histories refer to the same physical sample times. The candidate input for a decision first affects the next predicted output, without leaking later inputs.
- [ ] Training, offline Rollout, and MPC forecasts agree within documented numerical tolerances at the first step and throughout a multistep Rollout with identical weights and standardization.
- [ ] Callers that build evaluation windows and targets follow the same convention, including the notebook's error analysis and prepared actual-input replay.
- [ ] State Absorption uses the previously applied input. Extending the measurement buffer for spectral Frames does not change the Predictor's trained input window or predictions.
- [ ] Coverage includes the affected MLP and CNN runtime paths, waveform and Observable outputs where they share the changed logic, and a control-history length of one.
- [ ] Previously reported offline metrics are marked as requiring regeneration where their input timing changes. Saved closed-loop logs remain immutable.
- [ ] Targeted parity and replay tests and the repository gate pass.
