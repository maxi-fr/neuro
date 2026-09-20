# 01. Select checkpoints after the training schedule completes

**What to build:** A training run with delayed Loss terms reaches the intended training phase before early stopping can end it, and exports its best checkpoint from that completed phase. The training summary makes the selected epoch and its eligibility visible.

**Blocked by:** None. Can start immediately.

**Status:** Ready for implementation. Not started.

## Evidence and scope

Both investigated models stopped after epochs 0 through 40, restored epoch 0, and never trained with the STFT Loss scheduled for epoch 80. Validation included STFT from the beginning, while the selected checkpoint had received only one-step MSE training. Increasing patience alone would still allow selection of that early checkpoint.

Derive checkpoint eligibility from the active Loss schedules. Keep a fixed full-Loss validation score for comparison, but count stopping patience and select the final checkpoint only among eligible epochs. This ticket changes training behavior and its reporting; the bounded retraining experiment is ticket 05.

## Preserve existing experiments

Existing models, checkpoints, sweeps, trials, configs, run logs, metrics, plots, and reports must not be overwritten or deleted. Write new outputs either to a new experiment folder or to a uniquely named addition within an existing experiment. Treat old artifacts as read-only inputs. Appending to an experiment means adding new runs or trial identities and additive index entries; it does not mean resuming into occupied trial IDs, replacing a best-model file, or regenerating an old summary in place. If an output destination already exists, stop before writing and choose a new destination. Retain before-and-after hashes for the existing artifacts used by the experiment.

## Acceptance criteria

- [ ] A deterministic regression exercises the real Trainer with a delayed Loss, a curriculum, and a misleading early validation minimum. It fails under the current behavior.
- [ ] The first eligible checkpoint follows an epoch trained with every enabled scheduled Loss active and every curriculum at its full Span. Eligibility uses effective schedules, including rounding and disabled terms.
- [ ] Early-stopping patience starts in the eligible phase. Improvements before that phase cannot consume patience or become the final selected checkpoint.
- [ ] Final selection restores the lowest validation Loss among eligible epochs, even if an earlier ineligible epoch had a lower score.
- [ ] A run whose epoch budget cannot reach eligibility reports the incompatible schedule before training and does not export a checkpoint as ready for evaluation.
- [ ] Training statistics record eligibility start, selected epoch, stopping reason, and per-Loss training and validation components. They distinguish an active zero-valued Loss from a term that never ran.
- [ ] A run with no delayed terms or curriculum retains ordinary best-validation checkpoint selection and patience behavior.
- [ ] Targeted Trainer tests and the repository gate pass. No long retraining run is needed to verify this ticket.
