# Hyperparameter sweep strategy — November 2026

This protocol defines Predictor selection, controller experiments and final evaluation for the November 2026 campaign in `artifacts/sweep_2026_11/`. It investigates how the prediction target, stimulation resolution, training Loss and MPC Cost affect Seizure Suppression, Delivered Charge and computation.

All Observable Predictors use **dt=0.06 s: 50 Hz EEG and hop 3**. Waveform Predictors and State Absorption use **20 ms steps (50 Hz EEG)**, with **20 ms current holds** and **60 ms replanning**. The 20 ms excitation recordings match the actuation grid. The [EEG check](../artifacts/sweep_2026_11/timing_rationale/report.md) supports 50 Hz sampling. The 60 ms replanning period follows the better-performing Observable configuration in preliminary experiments. Select controller operating points using the **20–30 mAs mean Delivered Charge target** in Section 5.

## Campaign execution checklist

Status as of 2026-11-04. Completed studies are reused; pending items describe the remaining experimental work. Optimizer Sweeps and three-seed confirmation are outside this campaign's scope.

### Phase 1: Predictor selection

- [x] Observable MLP architecture search: 25 trials; Trial 23 (`n_y=1`, width 1024, depth 1, `softplus`, residual; `val_loss=1.6541`).
- [x] Observable CNN architecture search: Trial 22 (`n_y=16`, temporal/frequency kernels 3/1, head 78, `softplus`; historical `val_log_mse=1.5996`), baseline, pilots and temporal-kernel ablation.
- [x] Hankel-DMDc hop-3 grid: 140 fits; Trial 89 (`n_y=4`, energy 0.95, lambda 0.01, rank 1339; `val_loss=1.6197`).
- [ ] Observable MLP residual-connection ablation (`true` versus `false`).
- [ ] Rerun Waveform MLP and CNN architecture searches at 20 ms using the same joint validation objective and recording split. Archive trials and selected checkpoints in November.
- [ ] Retain checkpoints and recording identities so we can later rescore selected Predictors on common validation windows within each target and use common spectral error for Waveform-derived versus direct Observable forecasts.

### Phase 2: Development controller studies

- [x] Observable MLP/CNN effort grids: 35 runs per family, seven weights × five development seeds, dt=60 ms, H=17.
- [x] CNN target-aligned baseline effort: `w_u=0.3` gives mean charge about 23.00 mAs and burden 0.0544.
- [ ] Refine MLP effort: `w_u=0.3` gives 30.94 mAs and `w_u=1` gives 14.81 mAs; no original candidate meets the target.
- [x] DMDc hop-3 effort comparison: 45 runs including baselines. At H=17, `w_u=0.3` gives 25.51 mAs and burden 0.0492; `w_u=0.1` gives 42.75 mAs and is outside target at that horizon.
- [x] Hop-3 Segment-25/50 screening: ten runs per neural family. All four family/Segment means exceed 30 mAs at the original `w_u=0.1`.
- [ ] Reconfirm Segment candidates at a common target-calibrated effort per family; then complete band, Frame Kernel and taper studies at hop 3.
- [x] DMDc Control Horizon, terminal and exact-L1 studies. Selected point: H=10 / 0.60 s, `w_u=0.1`, terminal multiplier 1, `w_u_l1=0`.
- [ ] Verify 20 ms measurement/current alignment and three-move playback at 60 ms replanning for both Waveform families.
- [ ] Calibrate Waveform effort for each family on the new timing grid.
- [ ] Run Waveform Loss–Cost arms A–E at effective training seed 69.
- [ ] Outstanding neural terminal/horizon studies and sequential effort recalibration where needed.
- [ ] Compare L1 effort, Bonmin, block candidates, and homotopy on a shared development protocol. Report active time and bout count alongside suppression, Charge, and solve latency.

### Phase 3: Final evaluation and closure

- [ ] Audit excitation, references and historical manifests for overlap with reserved final seeds `[7100–7109]`.
- [ ] Freeze checkpoints, actual horizons, effort, timing policies, references, common Warm-up Period, solver/fallback and scoring. Verify that threshold stimulation uses the shared release time.
- [ ] Primary benchmark: 70 simulations, seven arms × ten reserved seeds, 12 s each with the frozen timing in Section 5.1.
- [ ] Paired differences and 95% paired-bootstrap intervals for Burden/Charge, plus seed values, timing and failures.
- [ ] Complete experiment reports and package reproducibility artifacts. Preserve timestamped applied currents and run end times for later sparsity scoring.

## 1. Scope and experiment organization

Reuse the completed Predictor searches and development studies listed below. Rerun the two Waveform architecture searches for the 20 ms grid. Reuse the completed Observable architecture and identification searches. Optimizer settings remain fixed. Observable geometry studies keep hop 3 throughout. Training-initialization replication is limited to single-seed ablations; report this limitation separately from variability across Plant seeds.

| Location under `artifacts/sweep_2026_11/` | Experiment |
| --- | --- |
| `observable_mlp/`, `cnn/observable/` | Observable neural Predictor selection. |
| `observable_effort_calibration/` | Neural effort calibration. |
| `observable_geometry/` | Segment, band, Frame Kernel and taper comparisons. |
| `observable_dmd/` | Hankel-DMDc identification and controller studies. |

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
- **Waveform:** 50 Hz, baseline `n_u=5` for 100 ms of current history. Preserve the joint objective (waveform MSE plus 15 × STFT). Use a 12-step MSE Span and STFT hop (0.24 s, matching the existing rounding of 0.25 s at 50 Hz), a 50-step STFT Span (1.0 s), and a symmetric Hann Segment of 25 samples (0.50 s). Retain the historical 1–25 Hz range with no smoothing. The 25 Hz endpoint is at Nyquist and lies in the default filter roll-off.
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

Run fresh MLP and CNN searches at 20 ms under `waveform_mlp/` and `cnn/waveform/`. Convert history and temporal kernel candidates by physical duration and record the resolved integer search space. Use the same joint `val_loss`, data split, and scoring settings across the two families. Snapshot the selected weights, config, and provenance into each consuming experiment. Fresh pure-MSE/STFT fits belong to Section 6.3.

October Trial 20 MLP and Trial 24 CNN are historical 10 ms checkpoints. October Waveform effort calibration, partial Loss–Cost work, and the archived [timing pilot](../artifacts/sweep_2026_10/waveform_timing/report.md) all depend on that grid. Refit and recalibrate for November. Rebuild or verify every Waveform reference, including timing and spectral geometry, before reuse. The archived pilot's Observable arms belong to that mixed-protocol pilot. Standalone November Observable evidence retains its existing reference and Warm-up audits.

## 5. Closed-loop protocol and selection

### 5.1 Independent timing quantities

| Arm/policy | Predictor / State Absorption | Replanning | Current updates |
| --- | --- | --- | --- |
| Observable MLP/CNN/DMDc | 60 ms | 60 ms | 60 ms |
| Waveform MLP/CNN | 20 ms | 60 ms | Three planned 20 ms moves |

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

At H=51, verify that each Waveform controller solves once every 60 ms and applies three 20 ms moves per plan. Check actual currents, State Absorption, plan advancement, constraints, and fallback before running the development effort grid.

Calibrate each Waveform family afterward. Start Waveform at `w_y=1,w_hinge=1000,w_u=10,w_u_l1=0`; effort multipliers `[0.1,0.3,1,3,10,30,100]`. Observable baseline uses `w_hinge=10,w_u_l1=0`; executed effort weights `[0.1,0.3,1,3,10,30,100]`. Refine where the target lacks coverage and declare added grids before execution. MLP refinement starts with `w_u=[0.4,0.5,0.6,0.8]` between existing bracketing points; CNN's provisional baseline is 0.3. Retain DMDc's horizon-specific calibrated evidence.

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

Freeze family architecture/history, optimizer/schedule, dataset and execution policy. Fit pure MSE, pure STFT and joint 1:15 at effective seed 69; select using each declared eligible Loss. Use 20 ms steps and holds with 60 ms replanning across A–E.

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
| 0.6 | 0.60 | 10 | 30 |
| 0.8 | 0.84 | 14 | 42 |
| 1.0 | 1.02 | 17 | 51 |
| 1.2 | 1.20 | 20 | 60 |
| 1.5 | 1.50 | 25 | 75 |

Report Seizure Burden, Delivered Charge, final seizing region count, and solve latency. Distinguish common-horizon controlled comparisons from independently tuned final controller-system comparisons. If horizon selection changes the operating point, recalibrate once on development seeds, then freeze. DMDc's historical H=10 point is a candidate, not a horizon imposed on other families.

## 7. Final benchmark and execution order

Rerun the Waveform architecture searches and verify the new timing before Waveform controller comparisons. Observable calibration and geometry studies can proceed independently. Complete Loss–Cost, terminal/horizon and final effort choices on development seeds only.

The primary final manifest has seven arms: uncontrolled, threshold, Observable MLP, Observable CNN, Observable DMDc, Waveform MLP and Waveform CNN. All Predictors replan at 60 ms. Threshold shares Plant, montage, Control Budget and common Warm-up Period; tune heuristic settings on development seeds only.

Run 70 × 12 s simulations on all ten reserved seeds.

Freeze run/scoring manifests, execution order, checkpoint/config/reference hashes, effective fit seeds, actual horizons, Warm-up Period, charge-target selection, source/dependencies and fallback. Keep timing contention consistent. Preserve every arm/failure; partial paired summaries identify missing pairs and never replace the planned benchmark. Package results and reproducibility records with each experiment.
