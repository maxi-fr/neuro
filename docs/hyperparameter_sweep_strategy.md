# Hyperparameter sweep strategy — November 2026

This protocol defines Predictor selection, controller experiments and final evaluation for the November 2026 campaign in `artifacts/sweep_2026_11/`. It investigates how the prediction target, stimulation resolution, training Loss and MPC Cost affect Seizure Suppression, Delivered Charge and computation.

Use one spectral geometry for the baseline Observable training targets, Waveform spectral Losses, offline scoring, and spectral MPC Costs. The common geometry uses 50 Hz EEG, a 25-sample Segment, a 3-sample hop (60 ms), asymmetric Hann, a 1–25 Hz band, and no Frame smoothing or bin pooling. The Observable geometry ablations in Section 6.2 vary this baseline deliberately.

Keep the training excitation holds at **60 ms for Observable** and **20 ms for Waveform**. Waveform Predictors and State Absorption advance at 20 ms. Observable Predictors advance at 60 ms. Both controllers replan every 60 ms, with the applied-current policies in Section 5.1. The [EEG check](../artifacts/sweep_2026_11/timing_rationale/report.md) supports 50 Hz sampling. The 60 ms replanning period follows the better-performing Observable configuration in preliminary experiments. Select controller operating points using the **20–30 mAs mean Delivered Charge target** in Section 5.

## Campaign execution checklist

Status as of 2026-11-08. Reuse the Observable studies listed below and keep the legacy Waveform results as historical evidence. The [matched-geometry reruns](../artifacts/sweep_2026_11/waveform_matched_geometry/report.md) have a separate artifact directory. Optimizer Sweeps and three-seed confirmation are outside this campaign's scope.

### Phase 1: Predictor selection

- [x] Observable MLP architecture search: 25 trials; Trial 23 (`n_y=1`, width 1024, depth 1, `softplus`, residual; `val_loss=1.6541`).
- [x] Observable CNN architecture search: Trial 22 (`n_y=16`, temporal/frequency kernels 3/1, head 78, `softplus`; historical `val_log_mse=1.5996`), baseline, pilots and temporal-kernel ablation.
- [x] Hankel-DMDc hop-3 grid: 140 fits; Trial 89 (`n_y=4`, energy 0.95, lambda 0.01, rank 1339; `val_loss=1.6197`).
- [x] Observable MLP residual-connection ablation (`true` versus `false`): completed at effective seed 69; `val_loss=1.6340` (`residual: false`, epoch 81) versus `1.6541` (Trial 23 champion `residual: true`, epoch 79).
- [ ] Rerun both Waveform architecture searches from scratch with the shared spectral geometry, the existing 20 ms excitation dataset and the joint 1:15 objective. Search current history along with EEG history and architecture. Select new checkpoints under that Loss.
- [ ] Rebuild the offline lookahead comparison on the frozen validation recordings. Start at Frame 16 or the first later origin that satisfies every selected checkpoint's Priming. Use the same origins for all Predictors and persistence, and score all 17 future Frames without refreshing histories during the Rollout.

### Phase 2: Development controller studies

- [x] Observable MLP/CNN effort grids: 35 runs per family, seven weights × five development seeds, dt=60 ms, H=17.
- [x] CNN target-aligned baseline effort: `w_u=0.3` gives mean charge about 23.00 mAs and burden 0.0544.
- [x] Refine MLP effort: `w_u=0.4` gives 26.98 mAs and burden 0.0494; target-aligned operating point confirmed.
- [x] DMDc hop-3 effort comparison: 45 runs including baselines. At H=17, `w_u=0.3` gives 25.51 mAs and burden 0.0492; `w_u=0.1` gives 42.75 mAs and is outside target at that horizon.
- [x] Hop-3 Segment-25/50 screening: ten runs per neural family. All four family/Segment means exceed 30 mAs at the original `w_u=0.1`.
- [x] Observable geometry: completed all five stages. Segment 25 selected over 50 (22.27 vs 13.99 mAs); Frequency Band [1.0, 25.0] Hz confirmed (26.85 mAs, burden 0.0486; [0,25] and [3,12] Hz exceed 30 mAs); Frame Kernel width 1 confirmed (wider kernels kw4/kw7 over-stimulate at 33.2/37.1 mAs); Asymmetric Hann confirmed (symmetric Hann suffers 25-region seizure breakouts on seeds 7000/7002, burden 0.152). Champion geometry locked: Segment 25, hop 3, [1, 25] Hz, kw1, asymmetric Hann.
- [x] DMDc Control Horizon, terminal and exact-L1 studies. Selected point: H=10 / 0.60 s, `w_u=0.1`, terminal multiplier 1, `w_u_l1=0`.
- [x] Verify 20 ms measurement/current alignment and three-move playback at 60 ms replanning for both Waveform families (`artifacts/sweep_2026_11/waveform_timing/report.md`).
- [ ] Qualify the new Waveform checkpoints and recheck readiness, State Absorption, playback, and fallback with the matched healthy reference.
- [ ] Refit Waveform MSE-only, STFT-only, and joint models at effective seed 69.
- [ ] Recalibrate Waveform effort for each family on its seed-69 joint checkpoint and matched spectral MPC Cost, then rerun Loss–Cost arms A–E.
- [x] Complete outstanding Observable terminal and Control Horizon studies independently (Observable MLP champion: H=17, `term_m1`, burden 0.0494, charge 26.98 mAs; Observable CNN champion: H=17, `term_m5`, burden 0.0523, charge 22.64 mAs; both strictly in [20, 30] mAs Delivered Charge without effort recalibration). See `artifacts/sweep_2026_11/neural_terminal_horizon/report.md`.
- [ ] Complete Waveform terminal and Control Horizon Sweeps with the selected matched-geometry checkpoint and Cost, then recalibrate final effort as needed.
- [ ] Compare L1 effort, Bonmin, block candidates, and homotopy on a shared development protocol. Report active time and bout count alongside suppression, Charge, and solve latency.

### Phase 3: Final evaluation and closure

- [x] Audit excitation, references and historical manifests for overlap with reserved final seeds `[7100–7109]`: confirmed zero overlap with excitation (seeds 2000–2199), healthy references (seeds 69–78), development seeds (7000–7005), and historical DMDc runs (7000–7010).
- [x] Freeze checkpoints, actual horizons, effort, timing policies, references, common Warm-up Period, solver/fallback and scoring for Observable predictors. Shared release verified at `control_start_s = 1.38` s for Observable MPC and threshold stimulation.
- [ ] Primary benchmark: 70 simulations, seven arms × ten reserved seeds, 12 s each with the frozen timing in Section 5.1 (Observable 5-arm confirmation completed across all 10 reserved seeds `[7100–7109]`: `obs_mlp`, `obs_cnn`, `obs_dmdc`, `threshold`, `uncontrolled`; full 7-arm benchmark awaits pending Waveform controllers). See `artifacts/sweep_2026_11/observable_benchmark/report.md`.
- [x] Paired differences and 95% paired-bootstrap intervals for Burden/Charge on Observable champions, plus seed values, timing and failures. All three champions achieve decisive burden reduction vs uncontrolled (CIs strictly below 0) and significant burden & charge reduction vs threshold.
- [ ] Complete experiment reports and package reproducibility artifacts. Preserve timestamped applied currents and run end times for later sparsity scoring.

## 1. Scope and experiment organization

Reuse the completed Predictor searches and development studies listed below. Rerun the two Waveform architecture searches and their downstream comparisons with the shared spectral geometry. Reuse the completed Observable architecture and identification searches. Optimizer settings remain fixed. Observable geometry studies keep hop 3 throughout. Training-initialization replication is limited to single-seed ablations; report this limitation separately from variability across Plant seeds.

| Location under `artifacts/sweep_2026_11/` | Experiment |
| --- | --- |
| `observable_mlp/`, `cnn/observable/` | Observable neural Predictor selection. |
| `observable_effort_calibration/` | Neural effort calibration. |
| `observable_geometry/` | Segment, band, Frame Kernel and taper comparisons. |
| `observable_dmd/` | Hankel-DMDc identification and controller studies. |
| `waveform_matched_geometry/` | Fresh Waveform Sweeps, common-window evaluation, and dependent controller reruns. |
| `waveform_mlp/`, `cnn/waveform/`, `waveform_effort_calibration/`, `waveform_loss_cost/` | Historical Waveform checkpoints and controller runs from the previous protocol. |

Each new experiment snapshots its model, simulation, reference and resolved settings locally. Comparison manifests reference these snapshots, never mutable production configs. Record source revision/patch, dependency lock, hardware/concurrency, data identities, effective seeds, timing and failures. Permanent outputs and `report.md` live in `artifacts/sweep_2026_11/<experiment>/`; temporary large Rollouts use `results/sweep_2026_11/`. Reports contain Motivation, Experimental Setup, Results, and Actionable Recommendation / Decision; mark unrun stages pending. Completed study databases remain frozen; new ablations use separate experiment directories.

## 2. Data and seeds

Reuse the existing 200-recording datasets: `data/experiment_excited_60ms/train` for Observable fitting and `data/experiment_excited_20ms/train` for Waveform fitting. Freeze sorted manifests and the existing first-160/last-40 recording split; overlapping windows remain within recording partitions. Shared excitation data stay under `data/`.

Completed neural searches used effective seeds `69 + trial.number`; retain those identities for selected checkpoints. New standalone ablations use effective seed 69 without a trial offset. Final results characterize frozen checkpoints rather than all training initializations. Compare histories/geometries on common held-out decision times with sufficient Priming and complete future Rollouts; distinguish search metrics from rescored diagnostics.

| Partition | Explicit Plant seeds | Use |
| --- | --- | --- |
| Development | `[7000,7001,7002,7004,7005]` | Calibration, timing, geometry, Loss–Cost, terminal/horizon selection and debugging. |
| Reserved final | `[7100,7101,7102,7103,7104,7105,7106,7107,7108,7109]` | Frozen primary comparison and predeclared extra arms only. |

Earlier DMDc evaluations include development seeds and do not replace the reserved final comparison. Audit excitation/reference/evaluation identities for final-seed overlap before launch. Final outcomes never select settings; substantive redesign informed by them requires a new partition. Never screen seeds by difficulty or silently replace failures.

## 3. Training conventions

Use training-only per-channel mean/std for Waveform EEG and currents, and per-channel-frequency scaling for Observables. Neural recursion uses standardized coordinates; Losses use original output units.

- **Observable:** 50 Hz, hop 3 / 60 ms; baseline Segment 25, 1–25 Hz, linear Frame Kernel width 1, asymmetric Hann, no bin pooling. Baseline minimum `n_u=9`; recompute after Segment/Kernel changes. Curriculum MSE uses nominal one-second recursive Span.
- **Waveform:** 50 Hz. Keep the joint objective (waveform MSE plus 15 × STFT) and the 12-step MSE Span (0.24 s). Use the Observable baseline geometry for the STFT Loss, with `n_span=51` to score 17 forecast Frames at 0.06–1.02 s. The 25 Hz endpoint is at Nyquist and lies in the default filter roll-off.
- **New neural ablations:** selected architecture and frozen optimizer/batch settings; 150 epochs, patience 20, curriculum epochs 10–80 where applicable and projected maximum fit time seven hours. Imported runs retain original schedules. Checkpoint selection/early stopping requires full curriculum eligibility.

Rank new campaign architecture sweeps by `val_loss`. Preserve the historical Observable CNN ranking by `val_log_mse` and the recorded MLP/DMDc rankings by `val_loss`. Compare differently trained checkpoints using a shared evaluation score or common Plant outcomes.

Waveform ridge and DMD are standalone fits. New neural Waveform runs use their validation Loss without automatically scoring log-energy error or Rollout NMSE.

Use one GPU fit at a time and record contention for controller timing. Resource pilots must reach the full Loss Span.

## 4. Predictor selection

### 4.1 Observable MLP

Retain Trial 23 and all 25 trials: search `n_y=1–9`, widths `[128,256,512,768,1024]`, depths `[1,2,3]`, activations `[tanh,softplus]`, residual enabled, `n_u=9`. No new architecture/optimizer search. Fit the residual-off counterpart only as the declared ablation, with otherwise identical settings/data.

### 4.2 Observable CNN

Retain Trial 22, all trials and temporal-kernel ablation, preserving direct output, valid temporal convolutions, full-history readout and historical objective. No new architecture/optimizer search. Geometry fits preserve the selected architecture where dimensionally possible and record required shape changes.

### 4.3 Hankel-DMDc

Retain only the hop-3 grid for advancement: `n_y=[1,2,4,8]`, energy `[0.80,0.85,0.90,0.95,0.99,0.999,1.0]`, lambda `[0,1e-8,1e-6,1e-4,1e-2]`. Trial 89 is the recorded champion. Keep rank and conditioning evidence; no grid rerun, hop-finalist comparison or initialization replication is required.

### 4.4 Waveform MLP/CNN

Run fresh MLP and CNN searches under `waveform_matched_geometry/` with 25 trials per family. Keep the existing 20 ms architecture ranges and add `n_u ∈ {5, 13, 25}` to each search. After inserting the upcoming candidate Current, these settings supply 80, 240, or 480 ms of past currents per prediction. The shortest retains the historical setting, and the longest matches the baseline Observable Predictor's past-current duration. Use the same joint `val_loss`, data split, and scoring settings across the two families. Snapshot the selected weights, config, and provenance into each consuming experiment. Fresh pure-MSE, pure-STFT, and joint fits for the Loss–Cost ablation belong to Section 6.3.

October Trial 20 MLP and Trial 24 CNN are historical 10 ms checkpoints. October Waveform effort calibration, partial Loss–Cost work, and the archived [timing pilot](../artifacts/sweep_2026_10/waveform_timing/report.md) all depend on that grid. The archived pilot's Observable arms belong to that mixed-protocol pilot. Standalone November Observable evidence retains its existing reference and Warm-up audits.

November Trial 8 MLP and Trial 23 CNN are historical 20 ms checkpoints trained with the symmetric Hann and hop-12 Loss. Keep both generations at their recorded paths. The [matched-geometry rerun report](../artifacts/sweep_2026_11/waveform_matched_geometry/report.md) records which downstream results require replacement.

## 5. Closed-loop protocol and selection

### 5.1 Independent timing quantities

| Arm/policy | Predictor / State Absorption | Replanning | Current updates |
| --- | --- | --- | --- |
| Observable MLP/CNN/DMDc | 60 ms | 60 ms | 60 ms |
| Waveform MLP/CNN | 20 ms | 60 ms | Three independently optimized 20 ms moves |

Record native step, replanning period, input block size, physical Control Horizon and phase separately. Keep raw Plant and sensor rates unchanged. Use the frozen causal filter for each Predictor input. Absorb fresh Waveform measurements and the preceding actually applied current at every native tick. Advance plans/warm starts by consumed knots, not clock changes alone. Freeze fallback and feed actual fallback currents into subsequent history.

For common lookahead use `B=ceil(T_requested/0.06)`, handling exact integer-grid boundaries without floating-point overshoot; `H_observable=B`, `H_waveform=3*B`, `T_actual=0.06*B`. Nominal one second means 17 Frames or 51 Waveform steps, both 1.02 s. Preserve native-grid Cost normalization and terminal treatment; calibrate effort on the new grid.

The reserved benchmark compares complete controller systems at a common replanning cadence. Waveform MPC has three admissible moves per replan and Observable MPC has one. Report the actuation, data, and Estimator differences when interpreting results.

### 5.2 Shared setup and measurements

Use all five development seeds per arm, 12 s runs and paired initial conditions/noise. Freeze Plant regime/connectome, Leadfield, Field Projection, montage `[TP9,CP5,Ex8]`, Kirchhoff, 2 mA/electrode, any total-current/slew limits, solver settings and fallback. Report omitted limits and express rate bounds in physical units.

Use one common Warm-up Period per comparison, sufficient for every Estimator/Predictor and released on a shared 60 ms boundary; all controlled arms, including threshold, emit zero until then. Report full-run Burden and Charge. Reused runs must match intervention timing, scoring and effective settings; otherwise use them as screening evidence and run the matched comparison.

Report Burden, actual-current Delivered Charge `integral ||u(t)||_1 dt` in mAs, final seizing region count, solve failures/fallbacks, success and iterations. Compute solve statistics only from actual attempts; playback is neither a successful solve nor a zero-time solve. Report latency median/p95/max, solves and controller time per simulated second, total simulation runtime and separate construction/compilation. Compare end-to-end deadlines with replanning period. Offline simulated-time execution does not establish real-time feasibility; wall-clock execution needs a declared delay/fallback policy.

### 5.3 Charge-target advancement

1. Require all five Plant runs to complete with finite, constraint-respecting applied currents. Preserve failures/partial evidence. A completed fallback run remains in its arm.
2. An eligible operating point has **mean full-run Delivered Charge in [20,30] mAs**, equally weighting development seeds. This is an operating target, not a clinical safety assertion or a per-seed current constraint. Report every seed and the complete Burden/Charge frontier, including out-of-target points.
3. Within the target, minimize mean Burden; break exact ties by lower mean charge, higher solve success and lower p95 time. If no point meets the target, refine calibration on development seeds; do not silently select an out-of-target winner.
4. Fixed-weight ablations keep all declared arms and label target attainment. Recalibrate the selected final checkpoint/Cost/policy separately; do not tune each arm and call the result a fixed-weight comparison.

Report means, standard deviations, individual seeds and paired differences. Final intervals resample the ten Plant seeds together across arms; windows and solver decisions are not independent replicates. Report final outcomes even if they leave the development charge target; do not retune or exclude based on reserved results.

## 6. Development studies

### 6.1 Timing and effort

At H=51, verify that each Waveform controller solves once every 60 ms and applies three independently optimized 20 ms moves per plan. Use the new checkpoint and the shared asymmetric Hann, hop-3 healthy envelope. Check actual currents, State Absorption, readiness, plan advancement, constraints, and fallback before running the development effort grid.

Calibrate each Waveform family on its seed-69 joint checkpoint from Section 6.3. Start Waveform at `w_y=1,w_hinge=1000,w_u=10,w_u_l1=0`; effort multipliers `[0.1,0.3,1,3,10,30,100]`. Observable baseline uses `w_hinge=10,w_u_l1=0`; executed effort weights `[0.1,0.3,1,3,10,30,100]`. Refine where the target lacks coverage and declare added grids before execution. MLP refinement starts with `w_u=[0.4,0.5,0.6,0.8]` between existing bracketing points; CNN's provisional baseline is 0.3. Retain DMDc's horizon-specific calibrated evidence.

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

Freeze family architecture/history, optimizer/schedule, dataset and execution policy. Refit pure MSE, pure STFT and joint 1:15 at effective seed 69 from scratch and select using each declared eligible Loss. Use the shared spectral geometry for every active spectral Loss and hinge Cost. Use 20 ms steps and holds with 60 ms replanning across A–E.

Use the effort calibrated in Section 6.1 on the seed-69 joint checkpoint with the matched combined Cost. Hold it fixed across A–E.

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
| 0.6 | 0.60 | 10 | 30 |
| 0.8 | 0.84 | 14 | 42 |
| 1.0 | 1.02 | 17 | 51 |
| 1.2 | 1.20 | 20 | 60 |
| 1.5 | 1.50 | 25 | 75 |

Report Seizure Burden, Delivered Charge, final seizing region count, and solve latency. Distinguish common-horizon controlled comparisons from independently tuned final controller-system comparisons. If horizon selection changes the operating point, recalibrate once on development seeds, then freeze. DMDc's historical H=10 point is a candidate, not a horizon imposed on other families.

## 7. Final benchmark and execution order

Rerun the Waveform architecture searches and offline comparison. Refit the Loss–Cost variants, qualify the checkpoints, and verify the existing playback policy with the matched reference. Recalibrate effort on the new joint fits, rerun A–E, then perform terminal/Control Horizon selection and final effort calibration on development seeds. Observable calibration and geometry studies can proceed independently.

The primary final manifest has seven arms: uncontrolled, threshold, Observable MLP, Observable CNN, Observable DMDc, Waveform MLP and Waveform CNN. All Predictors replan at 60 ms. Threshold shares Plant, montage, Control Budget and common Warm-up Period; tune heuristic settings on development seeds only.

Run 70 × 12 s simulations on all ten reserved seeds.

Freeze run/scoring manifests, execution order, checkpoint/config/reference hashes, effective fit seeds, actual horizons, Warm-up Period, charge-target selection, source/dependencies and fallback. Keep timing contention consistent. Preserve every arm/failure; partial paired summaries identify missing pairs and never replace the planned benchmark. Package results and reproducibility records with each experiment.
