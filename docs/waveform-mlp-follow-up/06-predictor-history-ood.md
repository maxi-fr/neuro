# 06. Diagnose OOD using the Predictor's actual input history

**What to build:** The OOD investigation evaluates the state-action histories actually consumed by the selected Predictor against its verified training support. Notebook distances and forecast errors use consistent timing, features, and calibration data.

**Blocked by:** 02. Align training, offline Rollouts, and MPC input timing; 05. Retrain waveform candidates and compare closed-loop suppression and charge, for verified data provenance and replacement checkpoint artifacts. OOD evaluation can still describe a rejected candidate if qualification fails; it cannot make that candidate eligible.

**Status:** Complete.

## Evidence and scope

The existing full-history mode uses one EEG sample and 17 controls, while the investigated MLP consumes 15 EEG samples and 10 controls. Planned queries invent their initial control history by repeating the first planned input. Reference trajectories also come from an unverified alternate collection.

Construct features from the selected checkpoint's actual inputs: its trained measurement history and its control window after insertion of the candidate input. As the Rollout advances, replace measured history with predicted outputs and planned inputs at the same indices used by inference. Do not include extra spectral-history samples that the network itself does not consume. Retain instantaneous features only as an explicitly labeled reduced diagnostic.

## Preserve existing experiments

Existing models, checkpoints, sweeps, trials, configs, run logs, metrics, plots, and reports must not be overwritten or deleted. Write new outputs either to a new experiment folder or to a uniquely named addition within an existing experiment. Treat old artifacts as read-only inputs. Appending to an experiment means adding new runs or trial identities and additive index entries; it does not mean resuming into occupied trial IDs, replacing a best-model file, or regenerating an old summary in place. If an output destination already exists, stop before writing and choose a new destination. Retain before-and-after hashes for the existing artifacts used by the experiment.

## Acceptance criteria

- [x] A deterministic fixture with distinct values at every sample and electrode verifies the query at the first decision and subsequent predicted knots against the actual Predictor input windows.
- [x] The first query uses the preceding recorded controls and measured EEG history; subsequent queries incorporate predictions and planned inputs causally. No repeated-input padding or future measured EEG fills missing history.
- [x] History lengths and flattening order come from the selected Predictor. Insufficient Priming excludes a query with a recorded reason rather than inventing values.
- [x] Training reference and in-distribution calibration use disjoint verified trajectories with matching sample rate, preprocessing, and montage. Feature normalization is fitted only on the reference partition.
- [x] Missing or incompatible source data produces an explicit unavailable result. The notebook displays the dataset and checkpoint identities and the feature definition used for each result.
- [x] Error-versus-distance analysis uses the aligned inference from ticket 02 and matched actual future inputs where recorded prediction error is computed. An abandoned plan is not scored against a different applied sequence as if it were matched ground truth.
- [x] The report separates measured closed-loop queries from recursive planned queries and resolves distance and error by lookahead. Calibration coverage and sample counts accompany percentile summaries.
- [x] Notebook controls, reference construction, and query construction all use the selected history mode and neighbor count consistently.
- [x] Regression tests cover true history, trajectory separation, feature ordering, and input timing. The result treats OOD distance as a diagnostic association, not proof that distribution shift caused the control failure.
- [x] New OOD indices, calibrated distances, replay metrics, plots, and reports use fresh destinations. Original sweep and model artifacts remain unchanged.
