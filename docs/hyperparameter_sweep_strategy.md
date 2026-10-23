# Hyperparameter sweep strategy

This is the protocol for the October 2026 replacement Sweep campaign. It supersedes the previous strategy. Rerun Observable MLP, Waveform MLP and Hankel-DMDc selection under the current implementation; carry forward the completed CNN architecture searches under `sweep_2026_10/cnn/`. Re-evaluate downstream controller choices using the selected Predictors. This document specifies planned work; results and execution status belong in each experiment's `report.md`.

## 1. Scope, carryover and storage

Freeze these conventions for the campaign:

- Per-channel mean/std standardization for Waveform EEG and Control Currents; per-channel-frequency mean/std for Observables. Fit statistics on training recordings only. Neural recursion uses standardized coordinates; every Loss uses original output units.
- Waveform sampling rate 100 Hz. Curriculum MSE Span 0.25 s; STFT Span 1.0 s, symmetric Hann Segments of 50 samples, hop 25, 1–25 Hz band, no Frame smoothing. Joint Loss is MSE + 15 × STFT for both Waveform families.
- Baseline Observable geometry: 50 Hz EEG, Segment 25 samples, hop 3, 1–25 Hz band, linear Frame Kernel of width 1, asymmetric Hann, no frequency pooling. Curriculum MSE Span is the nominal one-second recursive Rollout.
- Capacity comparisons keep Loss, geometry, split, optimizer and schedule fixed within a family. Geometry and Loss–Cost comparisons are ranked by closed-loop performance, never by differently scaled validation Losses.
- Controller arms use the same Plant, sensors, montage, Control Budget and solver settings except for the factor explicitly varied. Snapshot resolved settings before execution.

| Existing work | Treatment |
| --- | --- |
| [CNN architecture searches](../artifacts/sweep_2026_10/cnn/report.md), backed up under `artifacts_gdrive/sweep_2026_10/cnn/` | Carry forward complete studies, checkpoints, manifests and source snapshots. Confirm finalists below; do not restart architecture searches. |
| Excitation and healthy-reference datasets | Reuse. Audit recording identities and Plant seeds; derive geometry-specific references and standardizers where needed. |
| `artifacts_gdrive/output_scaling_audit/` and its dated companion | Retain distribution and conditioning evidence. |
| [Shared uncertainty calibration](../artifacts_gdrive/shared_uncertainty_10000/report.md) | Retain the frozen 1:15 objective. Calibration is a scaling choice, not evidence of optimal closed-loop performance. |
| [Historical output-gradient calibration](../artifacts_gdrive/gradient_weight_calibration/report.md) | Retain the initial Waveform tracking/hinge Cost ratio, 1:1000. This does not calibrate effort through a trained Predictor. |
| Earlier calibration packages and dataset-sensitivity investigation | Preserve as provenance; superseded training ratios do not define this campaign. |
| Old MLP/DMDc searches and downstream controller Sweeps | Archive intact under `artifacts_gdrive/trash/`; rerun affected comparisons. |

The previous strategy and relocation manifest are under `artifacts_gdrive/trash/sweep_strategy_2026_10_23/`. Historical configs retain original paths and objectives; use the relocation manifest to locate dependencies. Never resume archived Optuna databases or mix their scores into new rankings. Unrelated montage, mixed-integer and JAX diagnostic experiments remain in place.

The campaign lives under `artifacts/sweep_2026_10/`, with backup storage under `artifacts_gdrive/sweep_2026_10/`. The completed CNN searches are part of this layout at `cnn/{observable,waveform}/`, with shared provenance at `cnn/reproducibility/`. Local storage is the execution copy; Drive stores the backup, not an independent study. Each experiment snapshots its base model, base simulation, healthy-reference specification and resolved candidates locally. Comparison manifests reference these snapshots, never mutable production configs. CNN migration records preserve original paths, config snapshots and trial identities while operational configs use the campaign paths. Copy required configs, checkpoints and provenance into downstream consuming experiments before execution.

Record source revision, working-tree patch, dependency lock, hardware, data identities, effective random seeds, parameter counts, timing, failures and selected checkpoints. Each experiment has a `report.md` with Motivation, Experimental Setup, Results, and Actionable Recommendation / Decision sections; mark unrun stages as pending. Stage high-volume intermediate Rollouts in `results/sweep_2026_10/`; package required evidence before clearing that staging output at experiment completion. Archiving never deletes historical evidence.

## 2. Data partitions and random seeds

### 2.1 Predictor training and validation

Use existing 200-recording excitation datasets: `experiment_excited_60ms/train` for baseline Observable fitting and `experiment_excited_20ms/train` for Waveform fitting. Freeze sorted file manifests with the existing first-160/last-40 training/validation split. Keep recordings together; overlapping windows never cross the split. MLP/CNN comparisons within a representation use identical files and preprocessing. DMDc uses matching Observable data.

Compare histories at common validation decision times: use the longest candidate's Priming Steps and require the entire future Rollout to exist. Freeze this scoring manifest before ranking. The existing Trainer can produce history-dependent window populations; when its metrics differ from this manifest, record them as search metrics and rescore saved candidates before shortlisting. Apply the same principle across geometries and initialization seeds. This document specifies requirements, not newly implemented runtime behavior.

Broad neural searches use `training.seed: 69`. The current runner adds trial number, so the effective fit seed is `69 + trial.number`; record it. Confirmation fits run standalone at effective seeds `[69, 70, 71]`, without a trial offset. Initialization seeds are distinct in role from Plant/data-generation seeds even when numeric values coincide. DMDc fits are deterministic and need no initialization replicates.

### 2.2 Closed-loop development and final evaluation

| Partition | Explicit Plant seeds | Permitted use |
| --- | --- | --- |
| Development, five seeds | `[7000, 7001, 7002, 7004, 7005]` | Effort, geometry, Loss–Cost, terminal weighting and Control Horizon selection; debugging and qualification. |
| Final evaluation, ten seeds | `[7100, 7101, 7102, 7103, 7104, 7105, 7106, 7107, 7108, 7109]` | One frozen final comparison, including uncontrolled and threshold baselines. |

Use explicit lists in manifests; runtime named seed tiers are not this partition protocol. Before launch, audit all excitation, healthy-reference and historical evaluation manifests for overlap with the final list and save the audit. Do not claim untouched status solely because numbers differ from development seeds. If prior use is found, resolve the partition before final execution; never silently replace difficult seeds.

Final seeds never guide checkpoints, references, normalization, qualification thresholds or controller settings. Freeze model identities, configurations, operating points, analyses and exclusion rules before running them. A substantive change after inspecting final outcomes requires a newly declared evaluation partition. Report all ten seeds, including those without appreciable uncontrolled seizure activity; never screen by observed difficulty.

## 3. Shared training and qualification protocol

### 3.1 Baselines, budgets and resource pilots

| Family | Baseline model | Optimizer baseline |
| --- | --- | --- |
| Observable MLP | Residual, `n_y=1`, `n_u=9`, width 256, depth 1, tanh | AdamW, learning rate 1e-4, weight decay 1e-5, batch 256 |
| Waveform MLP | Residual, `n_y=15`, `n_u=10`, width 256, depth 1, softplus | Same as Observable MLP |
| Observable CNN | Completed `cnn/observable/` candidate configs and weights | Architecture search used AdamW, 2e-4, 1e-5, batch 512 |
| Waveform CNN | Completed `cnn/waveform/` configs and weights, shared 1:15 Loss | Same as Observable CNN |
| Observable DMDc | `n_y=1`, geometry-dependent minimum `n_u`, energy cutoff 0.99, `dmd_lambda=1e-6` | DMDc fit; no gradient optimizer |

New neural fits use 150 epochs, patience 20, curriculum expansion from epoch 10 to epoch 80, and `max_training_hours: 7.0`. Patience counts consecutive eligible epochs without validation improvement; it does not stop training before curriculum completion. Imported and archived runs retain their recorded patience. Qualify a full-curriculum baseline before each new broad architecture search; short resource or uncertainty pilots do not replace that fit. Select checkpoints only when the resolved schedule reports every enabled Loss active and the curriculum at full Span. Record `eligibility_start` and `selected_epoch`; do not infer eligibility from a rounded epoch label. Nominal offline evaluation duration is one second, rounded to the representation's time grid; record requested and actual durations.

Before each MLP search, pilot the baseline, maximum history at baseline width, and maximum history/width/depth. Measure host RAM, GPU peak memory, full-Span epoch time and projected fit time on the 4 GB GPU. Warm-up-only fitting does not establish full-Span feasibility. Pilot controller derivatives and solver construction for large models too. Freeze any reduced batch size across architecture candidates before launch; never change it selectively for a candidate. Keep the baseline learning rate unless a matched pilot supports changing it.

Run one GPU fit at a time. CPU Plant runs may overlap only when measured capacity permits; record concurrency for interpretable runtime comparisons. Current runtime pruning projects the full epoch budget after three epochs using the latest two epoch durations, including elapsed compilation cost. It is an epoch-boundary forecast, not a hard seven-hour timeout. Report memory/runtime pruning separately from accuracy rejection; conclusions apply to configurations feasible on this hardware.

New architecture and optimizer searches each have 25 trials per study, using the current seeded multivariate TPE sampler with 15 startup trials. An invocation adds trials to the database; check existing counts before resuming. Preserve failures and pruning reasons. Changes to budgets or ranges after seeing results require a recorded protocol amendment based on development evidence only.

### 3.2 Objectives and three-seed confirmation

| Representation | Search and confirmation ranking | Additional diagnostics |
| --- | --- | --- |
| Observable | `val_loss`: validation recursive MSE in recovered log-power units | `val_log_mse`, per-lookahead Frame error, Control Current sensitivity, growth and terminal normalized error |
| Waveform | Fixed `val_loss` = original-unit curriculum MSE + 15 × STFT | One-second `rollout_nmse`, log-energy error, spectral Frame error, growth, terminal NMSE and Control Current sensitivity |

All Tier 1 architecture, history, optimizer and DMDc searches use `sweep.objective: val_loss`, also the runtime default. For neural fits this aligns checkpoint selection, early stopping and configuration ranking after schedule eligibility. Other forecast metrics remain diagnostics and qualification evidence. Preserve imported CNN study objectives and scores; form current shortlists using recorded or rescored `val_loss` without relabeling historical study values.

Waveform checkpoint selection and architecture ranking minimize the same fixed joint Loss. MSE scores its 0.25 s Span; spectral supervision extends to one second. Never describe this aggregate as one-second waveform MSE. Aggregate Losses from pure-MSE, pure-STFT and joint arms are incomparable; rank their closed-loop comparisons using common Plant metrics and report common unweighted forecast diagnostics.

After each architecture search, rescore on common validation windows, shortlist the best three distinct configurations and fit each at all three confirmation seeds. Rank by arithmetic mean of the declared objective; report standard deviation and every individual value. Break exact ties by lower parameter count. Use the confirmed winning architecture for the family's optimizer search, then apply the same three-configuration, three-seed confirmation to its best optimizer candidates. Reuse a fit only if its effective seed, config, data, source and scoring protocol match exactly. Report limitations if fewer than three eligible configurations exist.

For CNNs confirm the top three rescored configurations from each imported architecture study, then run fresh optimizer searches. Replicated confirmation may select another configuration without invalidating the completed study. The small temporal-kernel ablation and memory-limited architecture trends are suggestive evidence, not established mechanisms or universal conclusions.

Require a saved `evaluate_control_horizon` report for every finalist using actual recorded future Control Currents and common held-out windows. Freeze the qualification manifest, stride, reference identity and scientific-threshold identity before screening. Retain prior gates: maximum amplitude-growth ratio at most 3.0 and terminal NMSE at most 1.25; undefined required metrics do not pass. Record the threshold calibration identity required by the evaluator and supporting held-out evidence. These are operational screening criteria, not proof of closed-loop stability. Planned/zero-input diagnostics alone never qualify a model. The Sweep runner does not automatically enforce these gates.

Every initialization replicate must qualify before its configuration advances. Check current JAX/CasADi inference and derivative agreement, derivative construction cost, nonzero Control Current sensitivity and development solver behavior. Sensitivity magnitude is a diagnostic, not a reward to maximize. CNN report claims do not substitute for saved qualification evidence.

Downstream tuning and final evaluation use the qualified seed-69 checkpoint of the aggregate-winning configuration. This fixes checkpoint identity without choosing the luckiest initialization. Report confirmation variance separately; ten-seed final Plant results characterize this frozen checkpoint, not all possible training initializations.

## 4. Predictor Sweeps

### 4.1 Observable MLP: joint capacity and history

Freeze baseline geometry and residual prediction. Search dimensions jointly:

| Parameter | Candidates |
| --- | --- |
| `model.n_y` | Integer 1–9 Frames |
| `model.hidden_size` | `[128, 256, 512, 768, 1024]` |
| `model.depth` | `[1, 2, 3]` |
| `model.activation` | `[tanh, softplus]` |

Keep `n_u=9` at baseline geometry. Cap the architecture search at one second of underlying EEG history: nine Frames contain 0.48 s between first and last endpoints and 0.98 s of EEG support, computed as `0.5 + (n_y - 1) * 0.06`. Ten Frames would exceed the cap. The old MLP Sweeps used one Frame (0.5 s of EEG support). Frame count, endpoint span, EEG support and future Control Horizon are different quantities.

The MLP flattens history into its first dense layer. At 62 channels × 12 frequency bins, maximum history requires about 1.72 million first-layer weights at width 256 and 6.88 million at width 1024. These counts motivate pilots, not automatic exclusion. Retain the range unless pilots establish a limitation, then document it before the search.

After architecture confirmation, optimize learning rate log-uniform `[1e-4, 3e-3]`, weight decay log-uniform `[1e-6, 1e-3]`, and batch size `[128, 256, 512]` on the frozen architecture. Confirm optimizer finalists under Section 3.2.

### 4.2 Waveform MLP: joint capacity and history

Use residual prediction, 100 Hz, `n_u=10`, and the frozen 1:15 joint Loss. Search jointly:

| Parameter | Candidates |
| --- | --- |
| `model.n_y` | Integer 5–100 samples, approximately 0.05–1.0 s |
| `model.hidden_size` | `[64, 128, 256]` |
| `model.depth` | `[1, 2, 3]` |
| `model.activation` | `[tanh, softplus]` |

This caps the architecture search at one second of history, replacing fixed 150 ms history and the optional 250 ms extension. The maximum is about 6.7 times the old 15-sample history. At maximum history and width 256, the first layer has about 1.59 million weights for 62 EEG channels and three Control Current channels. Recursive training and controller derivatives still require pilots.

After architecture confirmation, optimize learning rate log-uniform `[1e-4, 2e-3]` and weight decay log-uniform `[1e-6, 1e-3]`; keep the pilot-qualified batch size fixed. Confirm optimizer finalists under Section 3.2.

### 4.3 CNN carryover and optimizer Sweeps

Both CNN architecture studies are completed work under `cnn/{observable,waveform}/architecture/`. They use direct output, valid temporal convolutions, activation after every convolution, full-history flattening and optional hidden heads. Older newest-position checkpoints do not qualify under this runtime. Preserve complete studies, not only reported winning weights.

The Observable search covered 5–17 Frames, 8–32 initial filters, depth 1–2, temporal kernel 3, frequency kernel `[1,3,5]`, tapering 0.5–1.0, tanh/softplus and hidden-head depth 0–1 with widths 32–744. The Waveform search covered 25–150 samples, 8–32 initial filters, depth 1–3, temporal kernels 3–9, the same tapering/activation/head-depth choices, and `trunk`/`geometric_mean` hidden widths. Original manifests define exact executed ranges and budgets.

After common-window rescoring and three-seed confirmation, optimize learning rate log-uniform: Observable `[1e-4,3e-3]`, Waveform `[1e-4,2e-3]`; weight decay `[1e-6,1e-3]` log-uniform for both. Keep batch 512 unless a matched pilot requires a declared family-wide change. Confirm optimizer finalists under Section 3.2. No new CNN architecture study is planned.

### 4.4 Hankel-DMDc: fresh fitting campaign

Rerun DMDc. At each hop in `[3,5]`, fix Segment 25, band 1–25 Hz, Kernel width 1 and asymmetric Hann. Use the same 60 ms excitation recordings and split at both hops, so hop changes do not also change training excitation. Recompute `observable.min_past_controls()` and use that minimum as `n_u`; old nominal control-delay candidates below this minimum are invalid and removed.

Evaluate the full Cartesian grid of `n_y=[1,2,4,8]`, energy cutoff `[0.80,0.85,0.90,0.95,0.99,0.999,1.0]`, and `dmd_lambda=[0,1e-8,1e-6,1e-4,1e-2]`: 140 fits per hop. Never set an explicit rank, which overrides energy truncation. Record retained rank and numerical conditioning; energy 1.0 does not establish good conditioning. Use the current DMDc Trainer and record its resolved preprocessing rather than imposing neural-specific scaling assumptions.

Select within each geometry by common-window recursive validation MSE (`val_loss`) and qualification; retain `val_log_mse` as a diagnostic and compare hop finalists by development closed-loop scores. Deterministic fits need no initialization confirmation. Derive geometry-dependent references and calibrate DMDc effort independently.

## 5. Closed-loop protocol and selection

Every development comparison uses all five development seeds; every final arm uses all ten final seeds. Runs last 12.0 s from the same Plant start, with paired noise seeds and initial conditions within each seed. Freeze regime, connectome, dynamics, leadfield, Field Projection, montage `[TP9, CP5, Ex8]`, Kirchhoff constraint, per-electrode limit 2 mA, any total-current/rate limits, solver tolerances, iteration limits and fallback policy in local snapshots. Record omitted limits explicitly.

History and geometry affect Priming Steps. Use one common Warm-up Period per comparison, sufficient for every candidate Estimator and Predictor, and apply zero Control Current until then in all controlled arms. Freeze it from geometry/history requirements before execution. Report Seizure Burden and Delivered Charge over the full 12 s, plus post-Warm-up values. This avoids an unreported intervention-timing advantage for short histories. Use a common Warm-up Period across all final arms, including the threshold controller.

Primary metric is Seizure Burden. Report Delivered Charge `integral ||u(t)||_1 dt` with explicit current/time units, final seizing-region count, PZ recruitment/containment, IPOPT success rate, fallback count, iteration count, and solve-time median/p95/max. Report end-to-end controller time separately from IPOPT time, construction/compilation separately from repeated decisions, and the fraction of decisions exceeding the controller period. State whether execution is simulated-time or wall-clock constrained; slow offline solves do not establish real-time feasibility.

Use this frozen advancement rule throughout development:

1. Require offline qualification, all five Plant runs completed, and finite constraint-respecting applied Control Currents. A failed Plant run makes the arm ineligible; preserve partial evidence and its failure reason.
2. Keep solver failures in results. A run completed using the declared fallback remains in its arm; report actual suppression, charge and solver success. Never average only successful solves or remove difficult seeds.
3. Report the nondominated set in mean Seizure Burden and mean Delivered Charge, weighting Plant seeds equally. Advance the eligible candidate with smallest mean Seizure Burden; break exact ties by lower mean charge, higher solver success rate, then lower p95 solve time. Never invent a weighted burden/charge score after seeing results.
4. Backbone screening advances the best two by the same ordered rule. Preserve every candidate's results, including dominated candidates. Charge-saving alternatives remain visible even when they are not the suppression-first champion.

Report means, standard deviations, individual seed values and paired differences against uncontrolled and threshold baselines. The final comparison adds 95% paired-bootstrap intervals for mean differences, resampling the ten Plant seeds together across arms. Training windows and solver decisions are not independent evaluation replicates. Uncertainty with ten seeds remains limited; avoid post hoc significance claims from repeated comparisons.

## 6. Controller experiments

### 6.1 Effort calibration for every selected Predictor

Start Observable controllers at `w_hinge=10`, `w_u=1`, `w_u_l1=0`; start Waveform controllers at `w_y=1`, `w_hinge=1000`, `w_u=10`, `w_u_l1=0`. The historical Waveform output-gradient calibration supplies only the initial tracking/hinge ratio; Observable Costs have different units and sensitivities.

For every selected Predictor, compare effort multipliers `[0.1,0.3,1,3,10,30,100]` relative to its initial `w_u`, with state Cost fixed. Use five development seeds, report the suppression–charge frontier and select by Section 5. Recalibrate after selecting a changed Predictor/checkpoint or Cost; within controlled geometry or Loss–Cost ablations, hold calibrated effort fixed instead.

Retain the previous optional sparse-effort comparison: after quadratic calibration, compare `w_u_l1=[0,0.01,0.1]` with the selected quadratic weight and state Cost fixed. Declare whether this extension is included before execution; the main campaign uses `w_u_l1=0`. Archived L1 results remain historical evidence, not current operating points.

### 6.2 Observable geometry for MLP and CNN

Run the sequence independently for the selected Observable MLP and CNN. Freeze architecture, optimizer, schedule and calibrated controller Cost weights. Screen geometries with fresh seed-69 fits, qualifying each before paired Plant comparisons. Confirm the final geometry and original baseline at all three matched training seeds. Reuse exact repeated candidates rather than duplicate baselines.

1. **Backbone:** Segment `[25,50]` × hop `[3,5]` at 50 Hz; band 1–25 Hz, Kernel width 1, asymmetric Hann. Keep the top two backbones.
2. **Band:** On both retained backbones compare `[0,25]`, `[1,25]`, `[3,12]` Hz; advance one backbone/band pair.
3. **Frame Kernel:** On that pair compare linear widths `[1,4,7]`.
4. **Taper:** Compare asymmetric and symmetric Hann on the selected configuration.
5. **Confirmation:** Compare final geometry directly with the original baseline using identical fixed settings and all five development seeds. Report initialization variability and seed-level Plant outcomes; the staged search does not exhaust interactions.

Preserve approximately the architecture-selected Frame endpoint history duration when changing hop: `n_y = max(1, 1 + round(D / dt))`, where `D=(original_n_y-1)*original_dt`. Record actual endpoint duration and total EEG support, including the Segment and Frame Kernel; Segment and smoothing changes intentionally alter support. Recompute minimum `n_u`, standardizers, checkpoints, references and controller artifacts. Keep excitation recordings and splits fixed across geometries.

Nominal Control Horizon is one second: `H=ceil(1.0/dt)`, giving 17 steps/1.02 s at 60 ms or 10 steps/1.00 s at 100 ms. Qualify offline Rollouts to those actual durations. Segment length, bin spacing, hop and smoothing delay are distinct; report them separately. Recruitment detection latency requires a frozen source-space recruitment definition and sensor-space detection rule; it does not override the advancement rule.

### 6.3 Waveform Loss–Cost alignment

Run this comparison for each selected Waveform family, freezing architecture, history, optimizer and schedule. Fit pure curriculum MSE, pure STFT, and joint 1:15 configurations at all three confirmation seeds. Use the declared active Loss for eligible checkpoint selection; qualify with common unweighted diagnostics. Use each seed-69 checkpoint for Plant runs and report validation variability across initializations.

Calibrate effort once on that family's joint-Loss checkpoint under combined Cost, then hold it fixed across these arms:

| Arm | Training Loss | MPC Cost |
| --- | --- | --- |
| A | Pure MSE, weight 1 | Voltage tracking, `w_y=1`, hinge off |
| B | Pure STFT, weight 15 | Spectral hinge, `w_hinge=1000`, tracking off |
| C | Joint 1:15 | Voltage tracking only |
| D | Joint 1:15 | Spectral hinge only |
| E | Joint 1:15 | Combined tracking + hinge |

The joint checkpoint is shared across C–E. This retains the old controlled fixed-effort comparison under current Loss units. Changes in total Cost scale relative to effort are part of this comparison, not proof of an intrinsically superior objective. Before final family benchmarking, recalibrate effort for the selected checkpoint/Cost and report the tuned operating point separately from this ablation.

### 6.4 Terminal weighting and Control Horizon

On each selected controller compare terminal multipliers `[1,2,5,10]` for the active state penalty. For combined Waveform Cost, screen tracking and hinge terminal multipliers separately with the other fixed at 1, then confirm the proposed combination against uniform weighting. Keep effort fixed within this comparison.

Next compare requested Control Horizons `[0.6,0.8,1.0,1.2,1.5]` s with `H=ceil(T/dt)`; record `H*dt` and deduplicate identical resolved horizons. First evaluate Predictor error and qualification over every actual horizon, including 1.2/1.5 s extrapolation beyond the one-second training Rollout. Never assume one-second qualification transfers. Rank eligible controllers using development outcomes and report solve cost and deadline misses.

If the selected horizon changes the state/effort balance, repeat the same effort grid once on development seeds, freeze the operating point and report the sequential tuning dependency. No final-seed feedback enters these decisions.

## 7. Final comparison and execution order

Qualification and pilots precede broad fitting. MLP capacity/history searches and DMDc fitting can proceed while CNN finalists are rescored and confirmed, subject to resource limits. Architecture confirmation precedes optimizer selection and confirmation. Controller work starts only on qualified selected Predictors; each family's development experiments may proceed while another family trains.

After geometry/Loss–Cost selection, terminal weighting, Control Horizon and final development effort calibration, freeze the final manifest with seven arms:

1. Uncontrolled Plant.
2. Threshold stimulation with the same Plant, montage, Control Budget and common Warm-up Period; tune any heuristic settings on development seeds only.
3. Observable MLP.
4. Observable CNN.
5. Observable Hankel-DMDc.
6. Waveform MLP.
7. Waveform CNN.

Run 7 × 10 = 70 final Plant simulations, each 12 s, with the explicit final seed list. Record execution order before observing results; measure timing with consistent hardware contention. Report every arm even if a final run fails. Partial paired summaries identify missing pairs and never replace the planned ten-seed comparison.

Before launch verify seed separation; local config/checkpoint/reference hashes; common Warm-up Period and scoring manifest; saved qualification reports at selected Control Horizons; three-seed confirmation records; frozen effort and solver fallback; baseline parity; and archive relocation records. This strategy and archival do not claim experiments have executed: their reports must record that evidence separately.
