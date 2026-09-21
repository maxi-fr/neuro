# 03. Validate Predictor accuracy over the complete Control Horizon

**What to build:** A reproducible offline evaluation reports whether a checkpoint supports the Control Horizon used by MPC, with errors and signal growth resolved by lookahead. A short evaluation cannot be presented as evidence for a longer Control Horizon.

**Blocked by:** 02. Align training, offline Rollouts, and MPC input timing.

**Status:** Complete.

## Evidence and scope

The saved models report evaluation over 0.3 s but MPC recursively predicts 1.5 s. At the final step, actual-input replays have median predicted-to-recorded EEG RMS ratios of approximately 291 and 129. Replaying the actual inputs, rather than comparing the Plant with an abandoned plan, establishes this forecast defect.

Evaluate held-out recordings under their recorded future inputs. Use planned-input replay to check saved forecast reconstruction, and zero-input replay to inspect growth. Only compare a counterfactual forecast with Plant error when a matching Plant recording exists. This ticket builds evaluation and a candidate eligibility result, not a runtime fallback controller.

## Preserve existing experiments

Existing models, checkpoints, sweeps, trials, configs, run logs, metrics, plots, and reports must not be overwritten or deleted. Write new outputs either to a new experiment folder or to a uniquely named addition within an existing experiment. Treat old artifacts as read-only inputs. Appending to an experiment means adding new runs or trial identities and additive index entries; it does not mean resuming into occupied trial IDs, replacing a best-model file, or regenerating an old summary in place. If an output destination already exists, stop before writing and choose a new destination. Retain before-and-after hashes for the existing artifacts used by the experiment.

## Acceptance criteria

- [x] The evaluation uses the requested MPC sample period and complete Control Horizon, including the terminal predicted output. Windows without a full recorded future are excluded and counted.
- [x] It reports per-lookahead waveform RMSE and normalized error, spectral error under the intended Observable geometry, and predicted versus recorded amplitude or energy growth. One-step results remain separately visible.
- [x] A persistence baseline and the existing zero-prediction normalization make error magnitudes interpretable. Near-zero reference energy produces an explicit undefined or rejected result rather than a misleading ratio.
- [x] Checkpoint, data partition, preprocessing, sample rate, horizon, and reference identity accompany the numeric results. Training and evaluation recordings are separated at the trajectory level.
- [x] A stable fixture passes and an unstable fixture fails declared eligibility criteria. The investigated checkpoints are rejected on their saved 75-step actual-input replays without hard-coded checkpoint names.
- [x] Numerical tolerances and scientific acceptance thresholds are separately documented. Scientific thresholds are fixed from held-out calibration before comparing retrained candidates or viewing the closed-loop test-seed outcomes.
- [x] Insufficient horizon coverage or unavailable calibration yields insufficient evidence, never a pass. The comparison workflow can check this result before accepting a candidate for the corrected-model experiment.
- [x] Saved-plan reconstruction and actual-input accuracy have separate labels. The report does not treat later replanning differences as prediction error under the original plan.
- [x] Targeted evaluation tests and the repository gate pass.
