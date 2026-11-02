# Hyperparameter sweep strategy — November 2026

This protocol defines Predictor selection, controller experiments and final evaluation for the November 2026 campaign in `artifacts/sweep_2026_11/`. It investigates how the prediction target, stimulation resolution, training Loss and MPC Cost affect Seizure Suppression, Delivered Charge and computation.

All Observable Predictors use **dt=0.06 s: 50 Hz EEG and hop 3**. Waveform prediction and State Absorption remain at 100 Hz; investigate both distinct-current and blocked-current execution with 60 ms replanning. Select controller operating points using the **20–30 mAs mean Delivered Charge target** in Section 5.

## Campaign execution checklist

Status as of 2026-11-02. Completed studies are reused; pending items describe the remaining experimental work. Optimizer Sweeps and three-seed confirmation are outside this campaign's scope.

### Phase 1: Predictor selection and qualification

- [x] Observable MLP architecture search: 25 trials; Trial 23 (`n_y=1`, width 1024, depth 1, `softplus`, residual; `val_loss=1.6541`).
- [x] Observable MLP saved qualification at H=17 / 1.02 s: growth 0.9288, terminal NMSE 0.3030; passed recorded gates.
- [x] Observable CNN architecture search: Trial 22 (`n_y=16`, temporal/frequency kernels 3/1, head 78, `softplus`; historical `val_log_mse=1.5996`), baseline, pilots and temporal-kernel ablation.
- [ ] Observable CNN actual-current Control Horizon qualification.
- [x] Hankel-DMDc hop-3 grid: 140 fits; Trial 89 (`n_y=4`, energy 0.95, lambda 0.01, rank 1339; `val_loss=1.6197`).
- [x] DMDc qualifications at H=10/14/17/20/25; saved reports record passes at actual durations 0.60/0.84/1.02/1.20/1.50 s.
- [ ] Observable MLP residual-connection ablation (`true` versus `false`).
- [x] Waveform MLP architecture search: Trial 20 (`n_y=23`, width 256, depth 1, `tanh`, residual; joint `val_loss=56.7467`).
- [x] Waveform CNN architecture search: Trial 24 (`n_y=29`, temporal kernel 5, head 212, `tanh`; joint `val_loss=60.0482`).
- [ ] Qualify both Waveform checkpoints at the actual matched Control Horizon and under intended current sequences.

### Phase 2: Development controller studies

- [x] Observable MLP/CNN effort grids: 35 runs per family, seven weights × five development seeds, dt=60 ms, H=17.
- [x] CNN target-aligned baseline effort: `w_u=0.3` gives mean charge about 23.00 mAs and burden 0.0544.
- [ ] Refine MLP effort: `w_u=0.3` gives 30.94 mAs and `w_u=1` gives 14.81 mAs; no original candidate meets the target.
- [x] DMDc hop-3 effort comparison: 45 runs including baselines. At H=17, `w_u=0.3` gives 25.51 mAs and burden 0.0492; `w_u=0.1` gives 42.75 mAs and is outside target at that horizon.
- [x] Hop-3 Segment-25/50 screening: ten runs per neural family. All four family/Segment means exceed 30 mAs at the original `w_u=0.1`.
- [ ] Reconfirm Segment candidates at a common target-calibrated effort per family; then complete band, Frame Kernel and taper studies at hop 3.
- [x] DMDc Control Horizon, terminal and exact-L1 studies. Selected point: H=10 / 0.60 s, `w_u=0.1`, terminal multiplier 1, `w_u_l1=0`.
- [ ] Prepare and verify both Waveform execution policies for the timing study in Section 6.1.
- [ ] Waveform timing study for both families: same checkpoint and fixed weights across native 10 ms, distinct-current 60 ms and blocked-current 60 ms execution.
- [ ] Independent Waveform effort calibration for each 60 ms policy.
- [ ] Waveform Loss–Cost arms A–E at effective training seed 69; alternate-policy ablation on the selected checkpoint/Cost.
- [ ] Outstanding neural terminal/horizon studies and sequential effort recalibration where needed.
- [ ] Optional declared follow-up: fixed-architecture Waveform training on 60 ms hold data, retaining 100 Hz EEG / 10 ms Predictor steps, compared with 20 ms hold training under blocked execution.

### Phase 3: Final evaluation and closure

- [ ] Audit excitation, references and historical manifests for overlap with reserved final seeds `[7100–7109]`.
- [ ] Freeze checkpoints, actual horizons, effort, timing policies, references, common Warm-up Period, solver/fallback and scoring.
- [ ] Primary benchmark: 70 simulations, seven arms × ten reserved seeds, 12 s each; distinct-current 60 ms Waveform replanning.
- [ ] Paired differences and 95% paired-bootstrap intervals for Burden/Charge, plus seed values, timing and failures.
- [ ] If blocking is also evaluated on final seeds, predeclare two extra arms: nine arms / 90 runs. Otherwise blocking remains a five-development-seed ablation.
- [ ] Complete experiment reports and package reproducibility artifacts.

## 1. Scope and experiment organization

Reuse the completed Predictor searches and development studies listed below. No new architecture or optimizer search is planned. Observable geometry studies keep hop 3 throughout. Training-initialization replication is limited to single-seed ablations; report this limitation separately from variability across Plant seeds.

| Location under `artifacts/sweep_2026_11/` | Experiment |
| --- | --- |
| `observable_mlp/`, `cnn/observable/` | Observable neural Predictor selection. |
| `observable_effort_calibration/` | Neural effort calibration. |
| `observable_geometry/` | Segment, band, Frame Kernel and taper comparisons. |
| `observable_dmd/` | Hankel-DMDc identification and controller studies. |

Each new experiment snapshots its model, simulation, reference and resolved settings locally. Comparison manifests reference these snapshots, never mutable production configs. Record source revision/patch, dependency lock, hardware/concurrency, data identities, effective seeds, timing and failures. Permanent outputs and `report.md` live in `artifacts/sweep_2026_11/<experiment>/`; temporary large Rollouts use `results/sweep_2026_11/`. Reports contain Motivation, Experimental Setup, Results, and Actionable Recommendation / Decision; mark unrun stages pending. Completed study databases remain frozen; new ablations use separate experiment directories.

## 2. Data and seeds

Reuse the existing 200-recording datasets: `data/experiment_excited_60ms/train` for Observable fitting and `data/experiment_excited_20ms/train` for initial Waveform fitting. Freeze sorted manifests and the existing first-160/last-40 recording split; overlapping windows remain within recording partitions. Shared excitation data stay under `data/`.

The execution-policy comparison uses the same Waveform checkpoint across all three policies. Training on 60 ms holds is a separate ablation, not a prerequisite folded into that comparison. Retain native 10 ms sampling and match architecture, Loss, fit budget and recording scenarios/splits as closely as possible. Document unmatched data differences and assess transfer to repeated-current blocks and rapidly changing sequences.

Completed neural searches used effective seeds `69 + trial.number`; retain those identities for selected checkpoints. New standalone ablations use effective seed 69 without a trial offset. Final results characterize frozen checkpoints rather than all training initializations. Compare histories/geometries on common held-out decision times with sufficient Priming and complete future Rollouts; distinguish search metrics from rescored diagnostics.

| Partition | Explicit Plant seeds | Use |
| --- | --- | --- |
| Development | `[7000,7001,7002,7004,7005]` | Calibration, timing, geometry, Loss–Cost, terminal/horizon selection and debugging. |
| Reserved final | `[7100,7101,7102,7103,7104,7105,7106,7107,7108,7109]` | Frozen primary comparison and predeclared extra arms only. |

Earlier DMDc evaluations include development seeds and do not replace the reserved final comparison. Audit excitation/reference/evaluation identities for final-seed overlap before launch. Final outcomes never select settings; substantive redesign informed by them requires a new partition. Never screen seeds by difficulty or silently replace failures.

## 3. Training conventions and qualification

Use training-only per-channel mean/std for Waveform EEG and currents, and per-channel-frequency scaling for Observables. Neural recursion uses standardized coordinates; Losses use original output units.

- **Observable:** 50 Hz, hop 3 / 60 ms; baseline Segment 25, 1–25 Hz, linear Frame Kernel width 1, asymmetric Hann, no bin pooling. Baseline minimum `n_u=9`; recompute after Segment/Kernel changes. Curriculum MSE uses nominal one-second recursive Span.
- **Waveform:** 100 Hz, baseline `n_u=10`; MSE Span 0.25 s plus 15 × STFT over 1.0 s, symmetric Hann Segment 50, hop 25, 1–25 Hz, no smoothing. This training-Loss hop is distinct from Observable hop 3. Slower replanning does not change the training Loss.
- **New neural ablations:** selected architecture and frozen optimizer/batch settings; 150 epochs, patience 20, curriculum epochs 10–80 where applicable and projected maximum fit time seven hours. Imported runs retain original schedules. Checkpoint selection/early stopping requires full curriculum eligibility.

Retain historical objectives: Observable CNN ranked `val_log_mse`; MLP/DMDc use recorded `val_loss`. Do not relabel scores or rank differently scaled pure-MSE, STFT and joint Losses together. Advance controller variants using common Plant metrics.

Save actual-recorded-current `evaluate_control_horizon` evidence at every resolved duration: maximum amplitude growth ≤3.0 and terminal NMSE ≤1.25, all required metrics finite. Record checkpoint hash, windows, preprocessing, reference/threshold identities and actual horizon. These gates are operational qualification, not stability proofs. Check inference/derivative agreement for supported solver paths, nonzero current sensitivity and full-horizon construction cost. Qualify upward-rounded or extrapolated horizons explicitly.

Use one GPU fit at a time and record contention for controller timing. Resource pilots must reach the full Loss Span.

## 4. Predictor selection

### 4.1 Observable MLP

Retain Trial 23 and all 25 trials: search `n_y=1–9`, widths `[128,256,512,768,1024]`, depths `[1,2,3]`, activations `[tanh,softplus]`, residual enabled, `n_u=9`. No new architecture/optimizer search. Fit the residual-off counterpart only as the declared ablation, with otherwise identical settings/data.

### 4.2 Observable CNN

Retain Trial 22, all trials and temporal-kernel ablation, preserving direct output, valid temporal convolutions, full-history readout and historical objective. No new architecture/optimizer search. Geometry fits preserve the selected architecture where dimensionally possible and record required shape changes.

### 4.3 Hankel-DMDc

Retain only the hop-3 grid for advancement: `n_y=[1,2,4,8]`, energy `[0.80,0.85,0.90,0.95,0.99,0.999,1.0]`, lambda `[0,1e-8,1e-6,1e-4,1e-2]`. Trial 89 is the recorded champion. Keep rank, conditioning and qualification evidence; no grid rerun, hop-finalist comparison or initialization replication is required.

### 4.4 Waveform MLP/CNN

Use Trial 20 MLP and Trial 24 CNN from the completed searches at `artifacts/sweep_2026_10/waveform_mlp/` and `artifacts/sweep_2026_10/cnn/waveform/`. Snapshot the selected weights/config/provenance into each consuming experiment. Complete actual-horizon qualification, then use fixed checkpoints for timing studies. Fresh pure-MSE/STFT fits belong to Section 6.3.

## 5. Closed-loop protocol and selection

### 5.1 Independent timing quantities

| Arm/policy | Predictor / State Absorption | Replanning | Current updates |
| --- | --- | --- | --- |
| Observable MLP/CNN/DMDc | 60 ms | 60 ms | 60 ms |
| Waveform native diagnostic | 10 ms | 10 ms | 10 ms |
| Waveform distinct-current, primary | 10 ms | 60 ms | Six distinct planned 10 ms moves |
| Waveform blocked ablation | 10 ms | 60 ms | One current per six-step block |

Record native step, replanning period, input block size, physical Control Horizon and phase separately. Keep Plant/sensor/filter rates unchanged. Absorb fresh Waveform measurements and the preceding actually applied current at every native tick. Advance plans/warm starts by consumed knots, not clock changes alone. Freeze fallback and feed actual fallback currents into subsequent history. Blocking constrains every future block, not only the executed first block.

For common lookahead use `B=ceil(T_requested/0.06)`, handling exact integer-grid boundaries without floating-point overshoot; `H_observable=B`, `H_waveform=6*B`, `T_actual=0.06*B`. Nominal one second means 17 Frames or 102 Waveform samples, both 1.02 s. Native-rate Waveform diagnostics use that same duration. Preserve native-grid Cost normalization and terminal treatment; fewer solves do not justify scaling weights by six.

Distinct-current versus Observable compares complete formulations at common feedback cadence, retaining finer Waveform actuation. Blocked versus distinct-current isolates actuation freedom within a fixed checkpoint. A target-focused comparison with matched admissible currents additionally uses blocking, common physical horizon and aligned spectral Cost/reference semantics. Data and Estimator-latency differences remain limitations.

### 5.2 Shared setup and measurements

Use all five development seeds per arm, 12 s runs and paired initial conditions/noise. Freeze Plant regime/connectome, Leadfield, Field Projection, montage `[TP9,CP5,Ex8]`, Kirchhoff, 2 mA/electrode, any total-current/slew limits, solver settings and fallback. Report omitted limits and express rate bounds in physical units.

Use one common Warm-up Period per comparison, sufficient for every Estimator/Predictor and released on a shared 60 ms boundary; all controlled arms, including threshold, emit zero until then. Report full-run and post-Warm-up Burden/Charge. Reused runs must match intervention timing, scoring and effective settings; otherwise use them as screening evidence and run the matched comparison.

Report Burden, actual-current Delivered Charge `integral ||u(t)||_1 dt` in mAs, final seizing regions, PZ recruitment/containment, solve failures/fallbacks, success and iterations. Compute solve statistics only from actual attempts; playback is neither a successful solve nor a zero-time solve. Report latency median/p95/max, solves and controller time per simulated second, total simulation runtime and separate construction/compilation. Compare end-to-end deadlines with replanning period. Offline simulated-time execution does not establish real-time feasibility; wall-clock execution needs a declared delay/fallback policy.

### 5.3 Charge-target advancement

1. Require qualification, all five Plant runs completed, and finite constraint-respecting applied currents. Preserve failures/partial evidence. A completed fallback run remains in its arm.
2. An eligible operating point has **mean full-run Delivered Charge in [20,30] mAs**, equally weighting development seeds. This is an operating target, not a clinical safety assertion or a per-seed current constraint. Report every seed and the complete Burden/Charge frontier, including out-of-target points.
3. Within the target, minimize mean Burden; break exact ties by lower mean charge, higher solve success and lower p95 time. If none qualifies, refine calibration on development seeds; do not silently select an out-of-target winner.
4. Fixed-weight ablations keep all declared arms and label target attainment. Recalibrate the selected final checkpoint/Cost/policy separately; do not tune each arm and call the result a fixed-weight comparison.

Report means, standard deviations, individual seeds and paired differences. Final intervals resample the ten Plant seeds together across arms; windows and solver decisions are not independent replicates. Report final outcomes even if they leave the development charge target; do not retune or exclude based on reserved results.

## 6. Development studies

### 6.1 Timing and effort

For each Waveform family compare the three policies using one checkpoint, physical horizon, common Warm-up Period and fixed weights. Use five development seeds and a matched Observable baseline. Preserve out-of-target outcomes in this policy ablation; measure suppression, Charge and runtime without assuming sixfold speedup. Both 60 ms policies are explicit arms; native 10 ms is a diagnostic.

Calibrate the two 60 ms policies independently afterward. Start Waveform at `w_y=1,w_hinge=1000,w_u=10,w_u_l1=0`; effort multipliers `[0.1,0.3,1,3,10,30,100]`. Observable baseline uses `w_hinge=10,w_u_l1=0`; executed effort weights `[0.1,0.3,1,3,10,30,100]`. Refine where the target lacks coverage and declare added grids before execution. MLP refinement starts with `w_u=[0.4,0.5,0.6,0.8]` between existing bracketing points; CNN's provisional baseline is 0.3. Retain DMDc's horizon-specific calibrated evidence.

Recalibrate after selected checkpoint/Cost/horizon/policy changes where needed. Hold weights fixed within ablations and report tuned points separately. Optional L1 work must be declared before execution; primary settings retain `w_u_l1=0` unless a documented development decision changes it. Historical DMDc L1 results remain evidence.

### 6.2 Observable geometry, fixed hop 3

Run MLP/CNN independently. Keep 50 Hz/hop 3 and architecture-selected Frame endpoint history. Recompute `n_u` support, standardizers, references/checkpoints after geometry changes. Segment/Kernel intentionally change EEG support. Freeze data/splits and optimizer/schedule; fresh fits use effective seed 69.

1. **Segment:** preserve existing 25/50 screening. Reconfirm both at the same target-calibrated effort per family and common Warm-up Period. Advance one eligible backbone under Section 5.3; if neither meets target, refine the family's common effort before confirmation.
2. **Band:** `[0,25]`, `[1,25]`, `[3,12]` Hz on the retained backbone.
3. **Frame Kernel:** linear widths `[1,4,7]` on selected backbone/band.
4. **Taper:** asymmetric versus symmetric Hann.
5. **Confirmation:** final geometry versus baseline on all five development seeds at effective training seed 69 and declared fixed weights; independently recalibrate the selected final operating point. Three-seed confirmation stays descoped.

Report Segment support, bin spacing, hop, smoothing latency and history separately. The staged search does not exhaust interactions. Use H=17 / 1.02 s until horizon selection.

### 6.3 Waveform Loss–Cost alignment

Freeze family architecture/history, optimizer/schedule, dataset and execution policy. Fit pure MSE, pure STFT and joint 1:15 at effective seed 69; select using each declared eligible Loss and qualify using common diagnostics. Use distinct-current 60 ms execution for primary A–E; investigate blocking on selected checkpoint/Cost separately rather than implicitly adding a full factorial.

Calibrate effort on the joint checkpoint/combined Cost, then hold it fixed:

| Arm | Training Loss | MPC Cost |
| --- | --- | --- |
| A | MSE, weight 1 | Tracking, `w_y=1`, hinge off |
| B | STFT, weight 15 | Hinge, `w_hinge=1000`, tracking off |
| C | Joint 1:15 | Tracking only |
| D | Joint 1:15 | Hinge only |
| E | Joint 1:15 | Combined tracking/hinge |

C–E share the joint checkpoint. Changes in state-Cost scale relative to effort are part of this comparison. Rank with common Plant outcomes, then recalibrate selected final points. Spectral target comparisons must align physical Segment support, band, taper, smoothing, reference semantics and normalization; wording alone does not make objectives equal.

### 6.4 Terminal weighting and Control Horizon

Compare terminal multipliers `[1,2,5,10]` for selected neural controllers, effort fixed. For combined Waveform Cost, screen tracking and hinge multipliers separately, then confirm any combination. Reuse completed DMDc candidates and target selection where the effective setup passes audit.

| Requested horizon (s) | Actual (s) | Observable H | Waveform H |
| --- | --- | --- | --- |
| 0.6 | 0.60 | 10 | 60 |
| 0.8 | 0.84 | 14 | 84 |
| 1.0 | 1.02 | 17 | 102 |
| 1.2 | 1.20 | 20 | 120 |
| 1.5 | 1.50 | 25 | 150 |

Qualify every actual duration, including extrapolation. Report deadline misses and Burden/Charge/runtime. Distinguish common-horizon controlled comparisons from independently tuned final controller-system comparisons. If horizon selection changes the operating point, recalibrate once on development seeds, then freeze. DMDc's historical H=10 point is a candidate, not a horizon imposed on other families.

## 7. Final benchmark and execution order

Prepare and verify timing separation, qualify the actual Control Horizons and complete the five-seed timing study before broad Waveform controller comparisons. Observable calibration and geometry studies can proceed independently. Complete Loss–Cost, terminal/horizon and final effort choices on development seeds only.

The primary final manifest has seven arms: uncontrolled, threshold, Observable MLP, Observable CNN, Observable DMDc, distinct-current Waveform MLP and distinct-current Waveform CNN. All Predictors replan at 60 ms. Threshold shares Plant, montage, Control Budget and common Warm-up Period; tune heuristic settings on development seeds only.

Run 70 × 12 s simulations on all ten reserved seeds. Blocking is a required development ablation; adding both blocked controllers to final evaluation means 90 runs and must be declared before reserved outcomes. Substituting a primary policy also requires a development-only amendment before final evaluation.

Freeze run/scoring manifests, execution order, checkpoint/config/reference hashes, effective fit seeds, actual horizons, Warm-up Period, charge-target selection, qualification, source/dependencies and fallback. Keep timing contention consistent. Preserve every arm/failure; partial paired summaries identify missing pairs and never replace the planned benchmark. Package results and reproducibility records with each experiment.
