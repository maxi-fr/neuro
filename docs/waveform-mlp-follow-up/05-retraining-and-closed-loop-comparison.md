# 05. Retrain waveform candidates and compare closed-loop suppression and charge

**What to build:** One bounded experiment retrains both waveform MLP candidates, qualifies their predictions over the full Control Horizon, then evaluates qualified candidates in matched Plant runs. The report compares suppression and delivered charge while separating checkpoint replacement, spectral coverage, and normalization effects.

**Blocked by:** 01. Select checkpoints after the training schedule completes; 02. Align training, offline Rollouts, and MPC input timing; 03. Validate Predictor accuracy over the complete Control Horizon; 04. Correct spectral horizon coverage and Cost normalization. External prerequisite: recover the recorded training data or establish a reproducible replacement dataset.

**Status:** Complete. Report compiled in [`artifacts/waveform_mlp_retraining_comparison/report.md`](../../artifacts/waveform_mlp_retraining_comparison/report.md).

## Evidence and scope

Both historical checkpoints selected epoch 0 before their spectral Loss activated. The recorded training-data directory is absent locally, and the OOD notebook currently reads a different collection whose equivalence is unverified.

Freeze the two existing architectures and intended Loss schedules, repair training selection, and retrain once per candidate with the recorded seed and split where recoverable. Keep old checkpoints immutable as diagnosed baselines. This ticket is a controlled retraining experiment, not a hyperparameter sweep or a search for a different architecture.

The existing collection mixes Warm-up Period lengths and logged Costs that cannot be recovered from the current source, despite exact Predictor forecast replay. Those historical runs establish symptoms but cannot isolate the benefit of a new Cost or training change.

Build a small comparison using freshly generated baselines under captured source and artifacts. Start with seed 7002, which reproduced strong divergence, then evaluate the predetermined finalists on the five original seeds. Hold Plant dynamics, montage, Estimator, Control Budget, sample period, duration, and initialization policy fixed. Resolve any differing required Priming Steps before the runs so stimulation starts at a comparable time.

Run this ticket in two phases: retraining and qualification, then matched closed-loop evaluation. Start the second phase only if at least one replacement checkpoint qualifies. If neither qualifies, finish with the negative result and its diagnostics; do not label an unqualified checkpoint as a corrected candidate or start an unplanned training search.

## Preserve existing experiments

Existing models, checkpoints, sweeps, trials, configs, run logs, metrics, plots, and reports must not be overwritten or deleted. Write new outputs either to a new experiment folder or to a uniquely named addition within an existing experiment. Treat old artifacts as read-only inputs. Appending to an experiment means adding new runs or trial identities and additive index entries; it does not mean resuming into occupied trial IDs, replacing a best-model file, or regenerating an old summary in place. If an output destination already exists, stop before writing and choose a new destination. Retain before-and-after hashes for the existing artifacts used by the experiment.

## Acceptance criteria

### Retraining and qualification

- [ ] A dataset manifest records trajectory identities, preprocessing, sample rate, stimulation montage, and the training, validation, and untouched test partitions. The notebook's alternate dataset is not silently substituted.
- [ ] If the original data cannot be recovered, regenerate only from an adequately specified generation procedure and label the result as a new-data experiment. Document that checkpoint improvements then cannot be attributed solely to the schedule fix.
- [ ] Frozen local configs, checkpoint metadata, source identity, training seed, dependency versions, and dataset hashes accompany the experiment. Any dirty-source changes are captured alongside the revision.
- [ ] Both runs reach the eligible training phase, execute the scheduled spectral Loss, and select final checkpoints from eligible epochs. Full training and validation curves and selection reasons are retained.
- [ ] Each replacement checkpoint receives the full-horizon evaluation from ticket 03 on held-out data and the diagnosed recorded contexts. Evaluation on old closed-loop contexts is labeled separately from an independent test set.
- [ ] A report compares old and new checkpoints on one-step error, lookahead error, spectral error, and amplitude growth using identical data and aligned inference. It records eligibility without asserting closed-loop efficacy.
- [ ] The experiment stops after the declared runs. Failed candidates remain failed findings; another training search requires a separately scoped follow-up.
- [ ] The experiment bundle contains its configs, final models, compact metrics, plots, and report. Temporary Rollouts are cleaned up, and canonical configs and previous artifacts remain unchanged.

### Matched closed-loop evaluation

- [ ] Before execution, record the candidate arms, common warm-up policy, seed sets, weight choices, simulation budget, and rules for advancing candidates. Do not add runs in response to favorable preliminary results.
- [ ] The small screening comparison includes a freshly reproduced legacy failure baseline, a replacement-checkpoint-only arm, a coverage-correction arm, and a normalization-correction arm. Each successive contrast changes one factor. Legacy failed checkpoints are explicitly diagnostic controls, never eligible candidates.
- [ ] Finalists are selected using the declared rule and evaluated on seeds 7000, 7001, 7002, 7004, and 7005 with a matched uncontrolled reference. Selection and final-evaluation evidence are labeled separately where the seed sets overlap.
- [ ] Every run captures exact configs, reference and checkpoint hashes, relevant data identity, source revision plus any dirty patch, dependency versions, and random seeds. Mutable canonical paths do not define the saved experiment.
- [ ] Sampled decisions reproduce both saved forecasts and saved total Cost under the recorded implementation within declared tolerances. A provenance mismatch invalidates that run's comparative attribution.
- [ ] Results report per-seed Seizure Burden, propagation or final recruitment, absolute delivered charge, stimulation duty fraction, saturation, solve success, and runtime. Solver success is not used as a proxy for suppression.
- [ ] Plan-versus-applied comparisons resolve lookahead, verify first-command equality, and distinguish later replanning from incorrect execution. Charge integrates absolute electrode currents over actual holding durations.
- [ ] The report attributes each observed change only to a controlled contrast and includes failures and trade-offs. It recommends a candidate only if it meets the declared suppression-and-charge criteria.
- [ ] Final configs, compact metrics, plots, and a report are packaged together. The result may conclude that none of the candidates improves the trade-off; completing the experiment does not require a positive finding.

### Artifact preservation

- [ ] The experiment declares whether it uses a new folder or uniquely named additions to an existing experiment before any training or simulation writes begin.
- [ ] Checkpoints and trial outputs have new identities. No existing sweep is resumed or altered, and no previous best checkpoint, report, plot, config, or result is replaced.
- [ ] Additions to an existing experiment preserve every previous run and trial record. Any index update is additive; revised comparisons and summaries are saved as new versioned outputs.
- [ ] Existing input artifacts have identical hashes before and after the experiment. A colliding destination is rejected before any existing file is changed.
