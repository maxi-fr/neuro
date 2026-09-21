# 04. Correct spectral horizon coverage and Cost normalization

**What to build:** Waveform MPC scores its full intended Control Horizon, including the terminal prediction, and applies spectral weights consistently with equivalent Observable Frames. Reports expose spectral and effort contributions so the stimulation trade-off is interpretable.

**Blocked by:** None. Can start immediately.

**Status:** Complete.

## Evidence and scope

The current 75-step problem combines 49 measured past samples with 75 stage outputs. With a 50-sample Segment and hop 25, its Frames end at offsets 0, 25, and 50. Controls 50 through 74 have exactly zero spectral-Cost sensitivity, so effort penalties alone select their values. The terminal output is excluded.

Choose and document a causal Frame grid that reaches the terminal prediction, including when the horizon is not divisible by the hop. Account for the Hann taper and Frame Kernel when establishing coverage. Appending a sample that still does not complete a Frame is insufficient. Preserve the frequency geometry and healthy-envelope meaning. Apply the normalization decision below as a separately verified change within this ticket.

The current waveform hinge sums channel contributions, while the Observable hinge averages them. For 62 otherwise equivalent channels this introduces a factor of 62 before considering Frame selection. Equal numeric weights therefore do not imply equal pressure toward stimulation.

Use the channel-mean convention already established by the Observable Cost, with explicit frequency and scored-Frame reductions. Match Frame selections when checking equivalence; constant measured-history terms and different temporal grids are separate issues. This is a change in effective weighting, so the later experiment must measure it rather than silently reusing old performance claims.

Coverage and normalization belong to this one ticket, but retain separate regression tests and a reproducible intermediate revision or patch so ticket 05 can measure their effects independently. Do not add a permanent compatibility mode solely for that experiment.

## Preserve existing experiments

Existing models, checkpoints, sweeps, trials, configs, run logs, metrics, plots, and reports must not be overwritten or deleted. Write new outputs either to a new experiment folder or to a uniquely named addition within an existing experiment. Treat old artifacts as read-only inputs. Appending to an experiment means adding new runs or trial identities and additive index entries; it does not mean resuming into occupied trial IDs, replacing a best-model file, or regenerating an old summary in place. If an output destination already exists, stop before writing and choose a new destination. Retain before-and-after hashes for the existing artifacts used by the experiment.

## Acceptance criteria

### Horizon coverage

- [x] A deterministic test of the real waveform objective reproduces the blind final 25 controls in the diagnosed 75-step configuration before the fix.
- [x] The corrected objective includes the terminal predicted output in a scored Frame and has no structurally omitted suffix of the intended Control Horizon.
- [x] The Frame grid uses only observed history and predictions through the terminal output. It does not invent a future beyond the Control Horizon or duplicate a terminal Frame's contribution.
- [x] Direct calculation of the same Frames and aggregation matches the objective returned by the production solver path, including its stage and terminal treatment.
- [x] Automatic differentiation agrees with finite differences in an active-hinge, controllable fixture. Every intended control can affect the spectral term, including the final input; tests distinguish structural blindness from a legitimately inactive hinge or zero local sensitivity.
- [x] Coverage includes the diagnosed geometry, a horizon shorter than one Segment with adequate history, a horizon not divisible by the hop, and an enabled Frame Kernel.
- [x] A bounded solve on a deterministic control-sensitive fixture demonstrates that a late input can change the spectral objective and optimized plan.
- [x] Targeted Cost and MPC tests and the repository gate pass. Historical Cost values are not rewritten or claimed to have been produced by this corrected implementation.

### Cost normalization

- [x] A deterministic comparison constructs equivalent excess log-power values for both Predictor families and exposes the current channel-reduction discrepancy.
- [x] For matched scored Frames, reference envelopes, and weights, both spectral Costs agree within numerical tolerance.
- [x] Duplicating identical channels and their matching reference values leaves the normalized spectral Cost unchanged. The frequency and temporal reduction conventions are documented and tested separately.
- [x] Evaluation reports separate spectral, quadratic-effort, and sparse-effort Cost contributions, and identify the normalization convention used.
- [x] Existing saved runs retain their original recorded Cost and metadata. No compatibility mode is added solely to make historical totals resemble new totals.
- [x] Experimental weight choices explicitly account for the changed reduction. Claims about improved suppression or reduced charge are deferred to ticket 05.
- [x] Targeted Cost tests and the repository gate pass. The normalization change does not alter the corrected Frame placement or the Control Budget.

- [x] Historical artifacts remain unchanged. New diagnostic outputs use a new experiment folder or uniquely named additions, and output collisions are detected before writing.
