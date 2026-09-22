# Waveform MLP follow-up tickets

These six tickets expand the recommendations in the [investigation report](../../artifacts/waveform_mlp_diagnosis/report.md), with the requested merges applied. Old tickets 04 and 05 are now ticket 04; old tickets 06 and 07 are now ticket 05; old ticket 08 is now ticket 06. The tickets are ready for implementation. Creating or revising them has not started any fixes, retraining, or Plant comparisons.

| Ticket | Deliverable | Blocked by | Status |
| --- | --- | --- | --- |
| [01. Training-phase checkpoint selection](01-training-phase-checkpoints.md) | Train through scheduled Loss activation and select an eligible checkpoint. | None | Complete |
| [02. Control-history parity](02-control-history-parity.md) | Make training, offline Rollouts, and MPC agree on input timing. | None | Complete |
| [03. Full-Control-Horizon validation](03-control-horizon-validation.md) | Report recursive error and growth through the terminal prediction, with explicit candidate eligibility. | 02 | Complete |
| [04. Spectral Cost coverage and normalization](04-spectral-cost-coverage-and-normalization.md) | Score the full intended Control Horizon and make equivalent spectral weights comparable across Predictor families. | None | Complete |
| [05. Retraining and matched closed-loop comparison](05-retraining-and-closed-loop-comparison.md) | Retrain and qualify replacement checkpoints, then measure suppression and charge in a reproducible comparison. | 01, 02, 03, 04; verified data | Complete |
| [06. Predictor-history OOD](06-predictor-history-ood.md) | Calibrate OOD analysis against verified data using the actual Predictor input history. | 02, 05; verified data | Complete |

## Preserve existing experiments

Existing models, checkpoints, sweeps, trials, configs, run logs, metrics, plots, and reports must not be overwritten or deleted. Write new outputs either to a new experiment folder or to a uniquely named addition within an existing experiment. Treat old artifacts as read-only inputs. Appending to an experiment means adding new runs or trial identities and additive index entries; it does not mean resuming into occupied trial IDs, replacing a best-model file, or regenerating an old summary in place. If an output destination already exists, stop before writing and choose a new destination. Retain before-and-after hashes for the existing artifacts used by the experiment.

This preservation requirement applies to all six tickets, including outputs regenerated after an inference or evaluation fix. It covers old models and sweeps even when they are known to be defective. A fresh comparison can read those artifacts as baselines without replacing them.

## Execution order and limits

Start 01, 02, and 04 independently. Complete 03 after 02, recovering the training data while those fixes proceed. Then run 05 in two phases: retraining and qualification, followed by matched Plant evaluation for candidates that qualify. Complete 06 using the verified data and checkpoint artifacts.

Within 04, coverage and normalization retain separate checks and a captured intermediate revision or patch for controlled comparison. They remain one ticket. Within 05, failed qualification stops the closed-loop candidate phase and produces a negative-result report; it does not trigger an unplanned training search.

The missing original training collection is an external prerequisite. Do not silently substitute the notebook's alternate data. A regenerated dataset must be labeled as new, with the resulting attribution limits documented.

Each implementation ticket starts with a reproducing test and ends with relevant checks and the repository gate. Experiment work requires fixed inputs, retained negative results, and an artifact report. Completion does not depend on showing a favorable scientific outcome.

The tickets live in this documentation folder. The investigation report and experimental evidence remain in the gitignored artifact directory and may be unavailable in a fresh checkout. Their ready status records the revised scope, not completed implementation.
