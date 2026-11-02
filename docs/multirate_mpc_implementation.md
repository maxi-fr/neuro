# Multirate MPC implementation specification

Implement the controller timing and input parameterization required by the [Sweep strategy](hyperparameter_sweep_strategy.md). Both the trajopt and CasADi paths must support distinct-current execution and blocked-current execution. This document specifies proposed interfaces and behavior; it does not claim that the changes are implemented.

## 1. Required behavior

Separate three quantities: the Predictor step, the period between MPC solves, and the interval over which a planned Control Current is constant.

| Configuration | Predictor / State Absorption | Replanning | Input block size |
| --- | --- | --- | --- |
| Observable MLP/CNN/DMDc | 60 ms | 60 ms | One native step |
| Waveform native diagnostic | 10 ms | 10 ms | One native step |
| Waveform distinct-current | 10 ms | 60 ms | One native step |
| Waveform blocked-current | 10 ms | 60 ms | Six native steps |

For distinct-current execution, solve once, then apply the first six planned currents in order. For blocked execution, optimize one current per six-step block throughout the whole Control Horizon and apply the first block. Continue absorbing measurements and the preceding actually applied current every 10 ms in both cases. Intermediate measurements update the Predictor state but do not change the queued commands before the next solve.

Retain the learned one-step transition, filtering, sensor rates, spectral geometry and training Loss. A 60 ms current hold does not make the Waveform Predictor a 60 ms Predictor. Training on a different hold dataset is a separate experiment.

Initial required solver support is continuous single shooting with IPOPT in trajopt and CasADi, for Waveform MLP/CNN. Observable behavior at a one-step prefix must remain unchanged. Native solver, multiple-shooting and mixed-integer blocking are outside the initial implementation; unsupported combinations must fail at construction rather than silently ignore blocking. Existing one-step configurations retain their existing solver support.

## 2. Timing and configuration contract

Keep `controller.dt` as the native service interval: it determines when the simulation calls `update` and supplies the next measurement. Add `controller.replan_dt`, defaulting to the native interval. Add structural `problem.control_block_steps`, defaulting to one, to the relevant problem builders and both backends. The controller and solver must read the same problem setting; do not maintain independent block-size settings.

Proposed configuration fragments, with checkpoint/reference/Cost settings omitted:

```yaml
# Waveform: distinct-current execution
controller:
  class_path: neuro.control.mpc.TrajOptMPCController
  dt: 0.01
  replan_dt: 0.06
  control_start_s: 1.2  # illustrative; derive the common Warm-up Period from all arms
  problem:
    class_path: neuro.control.problems.build_waveform_problem
    horizon: 102
    control_block_steps: 1
```

```yaml
# Waveform: blocked-current execution; all other settings are identical
controller:
  dt: 0.01
  replan_dt: 0.06
  problem:
    horizon: 102
    control_block_steps: 6
```

The CasADi configuration uses `neuro.control.casadi.CasADiMPCController` and its corresponding Waveform problem builder, with the same timing/block fields and native-grid horizon semantics. These are fragments, not runnable experiment configs.

Let `d` be native dt, `r=replan_dt/d` the execution prefix, `b=control_block_steps`, and `H` the native Control Horizon. For this implementation:

- Require controller dt to equal checkpoint dt and `r` to be a positive integer.
- Support `b=1` and `b=r`; require `H >= r` and `H % b == 0`.
- Express the common Warm-up release as `control_start_s`, default zero, while retaining readiness gating. Require it to be on the replanning grid and long enough for every compared arm's Estimator/Priming requirements.
- Use integer native indices for boundary decisions, with phase anchored to simulation time zero. Never reset the grid to the instant at which a particular Predictor first becomes ready.
- Default `r=b=1` reproduces the existing successful one-step execution sequence. An explicitly specified common start delays intervention while continuing Priming.

`horizon` always counts native prediction transitions; trajopt retains `N=H+1` Knot Points. For nominal one-second comparisons use H=102 Waveform samples and H=17 Observable Frames, both 1.02 s. Do not pass replanning dt into learned dynamics, Fourier frequency calculations, predicted-output timestamps or Cost decomposition.

The 20–30 mAs selection target belongs to experiment scoring. It is not a new hard OCP constraint or an input to the scheduling API.

## 3. neuro: controller execution

### 3.1 Split the native tick into three operations

Restructure the touched `update` methods into State Absorption, planning, and command playback. Keep solver-specific construction and numerical work in each backend. Share the small timing/playback contract where this prevents the two backends from developing different scheduling semantics; a general event scheduler or controller framework is unnecessary.

The native update at index `k` has this order:

1. Absorb the measurement at time `k*d` with the current applied over the preceding native interval. At the first tick, that current is zero. Never absorb the command about to be issued as though it caused the current measurement.
2. During the common Warm-up Period, emit zero without solving. Predictor readiness alone must not release one arm earlier than another. Reject an inadequate experimental start configuration rather than quietly treating a later release as a matched comparison.
3. At an active replanning boundary, advance the old warm start by exactly the prefix consumed since its solve, then measure/pin the current absorbed state and solve. Skip this advancement before the first solve.
4. Capture the unshifted native-grid prediction, complete physical-current plan, Cost components and solver diagnostics. Store an execution prefix independently of the mutable driver warm start.
5. Emit the prefix's current indexed by `k-k_plan`. Between boundaries, only State Absorption, playback and logging run; no solver, planned Rollout or Cost reevaluation is needed.
6. Record the emitted current as the previous applied current for the next absorption.

Example with the common release at `t0`: at `t0`, solve and emit U[0]; at `t0+10,...,50 ms`, absorb fresh EEG and emit U[1],...,U[5]; at `t0+60 ms`, absorb EEG with applied U[5], advance six knots, and solve from the new measured state. Blocked execution has U[0]=...=U[5], but the six absorptions still occur.

### 3.2 Failure and coordinate handling

Preserve the current zero-current fallback for the continuous campaign's existing current constraints. If a solve fails, invalidate its execution prefix and issue fallback at every tick until the next scheduled solve. Do not execute a failed optimizer iterate, retry on every native tick, or reuse a stale successful prefix without an explicitly different experimental policy. Preserve attempted-solve diagnostics separately from the fallback plan.

In `TrajOptMPCController.update`, fix the existing failure path that stores `u_solved` in `_u_last` even when `_extract_plan` emitted zero. State Absorption must receive the applied current in the coordinates expected by the inference adapter. Handle Nullspace reduction and any supported epigraph adapter explicitly; physical-electrode logs must never contain reduced decisions or slack coordinates.

The first implementation adds no new slew constraints. If they are introduced later, an instantaneous zero fallback must be reconsidered against those constraints.

### 3.3 Files and responsibilities

| File | Required work |
| --- | --- |
| `src/neuro/control/mpc.py` | Add timing config; split `update`; store/play a prefix; shift at the correct boundary; use native dt for outputs/Costs; correct applied-current history. Forward block structure through `NeuroProblem`, builders and wrappers. |
| `src/neuro/control/casadi.py` | The same execution contract, CasADi block variables, native plan expansion and prefix-length warm-start shift. |
| `src/neuro/validation.py` | Validate native rate, replanning ratio, supported block size, horizon and common-start alignment independently. Retain native-rate Estimator/filter/reference checks. |
| `src/neuro/comparison.py` | Separate native current records from solve attempts and use replanning dt for deadlines. |
| `src/neuro/run_view.py`, `src/neuro/run_plot.py` | Display plans only at attempt timestamps and preserve native prediction spacing. |
| `src/neuro/predictor/replay.py`, `src/neuro/ood.py` | Preserve native observations/applied-current replay; distinguish actual solver plans from diagnostic replays at other ticks. |
| `src/neuro/control/benchmark.py` | Benchmark both fixed-state solves and actual replanning/playback execution. |

The current `simulate.Controller.evaluate` already applies zero-order hold between native ticks. No `simulate` scheduler change is required: changing its service interval to 60 ms would lose intermediate Waveform observations. The equality between `controller.dt` and native Predictor dt remains valid; it is the solve frequency that becomes independent. Reference/filter checks must not be redirected to `replan_dt`.

## 4. trajopt: consume several Knot Points

The installed `MPC.shift(dt)` always advances one Knot Point; its dt argument only advances the clock. Calling `shift(0.06)` on a 10 ms problem therefore misaligns the warm start and reference.

Provide an explicit integer-count operation, for example `MPC.advance(steps)`, that advances the existing warm start and reference by that many Knot Points and advances time by the corresponding native intervals. Its semantics must be distinct from supplying elapsed time to the current one-knot method. Repeated existing one-knot shifts are a correct initial implementation; batch the operation only if useful.

Required behavior:

- Shift native X/U, canonical constraint and bound multipliers, and augmented-Lagrangian state consistently with the existing one-step terminal-fill/mask rules.
- Advance a reference window by the same count, holding its end for the constant-reference campaign. Keep a separately specified terminal goal unchanged. Do not implicitly invent a sequence of new reference points from a single pending push.
- At the next solve, replace the initial state with the measured/State-Absorbed state. The driver prediction is a warm start, not a substitute for measured history.
- Keep Problem structure, Program identity and compiled numerical work stable across ticks. Advancing or measuring must not reconstruct the solver.
- Consume each prefix once. Remove the current unconditional post-solve one-knot shift when adopting the boundary-advance contract.

Primary upstream files: `src/trajopt/mpc.py` and `src/trajopt/program.py`. Define count-based tests in the existing MPC/Program unit tests. An initially repeated one-step implementation inherits one-step multiplier behavior; it does not require introducing new multiplier-shift conventions.

## 5. trajopt: blocked single-shooting transcription

### 5.1 Smaller decisions, full native dynamics

Add uniform `control_block_steps` as static Problem structure, default one. Preserve it through problem construction, retargeting and any supported wrapping. `trajopt.transcription.single_shooting.SingleShooting` reads it from the Problem; model dynamics and native time grid remain unchanged.

For b steps per block, optimize V with shape `(H/b, m)` and expand to native U with shape `(H, m)`:

```text
U[j] = V[j // b]
X[j+1] = f(X[j], U[j])
J_block(V) = J_native(X(expand(V)), expand(V))
```

For H=102 and three electrode decisions, distinct-current planning has 306 scalar controls; six-step blocking has 51. The Rollout still has 102 transitions and 103 state knots. With a two-coordinate Nullspace representation, these counts are 204 and 34.

Parameterize V directly. Adding equality constraints between all original native controls would enforce the policy but retain the large decision vector; it is not the required blocked transcription.

### 5.2 Constraints and derivatives

Evaluate state-dependent constraints on every native knot supported by the existing transcription. Preserve its existing rejection of unsupported state bounds; blocking must not silently drop them.

Collapse identical input-only constraints to one evaluation per block. In particular, do not impose the same Kirchhoff equality six times on one block variable: redundant rows can cause an equality-rank/constraint-count failure after dimension reduction. Preserve physical-electrode bounds when decisions are reduced coordinates. If supported stage bounds differ within a block, the block bound is their intersection; an empty intersection is infeasible. Never discard a tighter native bound.

Build objective/constraint derivatives with respect to V through expansion. For the linear expansion map S, the gradient is `S.T @ grad_U`; exact Hessians, where exposed, transform as `S.T @ Hessian_U @ S`. Rebuild Jacobian/Hessian dimensions and sparsity for reduced decisions and the retained constraint rows. Retain the current limited-memory Hessian option for the campaign.

Primary upstream files: `src/trajopt/problem.py`, `src/trajopt/transcription/single_shooting.py`, and relevant construction/retarget helpers. All solver entry points that cannot interpret non-unit blocking must reject it explicitly. Do not add arbitrary block schedules, variable replanning periods or support for every transcription in this change.

### 5.3 Public result and warm-start contract

Keep `SolverResult.trajectory`, the canonical Primal Vector and `MPC.controls/states` on the full native grid. Expand optimized V before reconstructing X and the result. Existing native-grid Cost decomposition and prediction readers can then retain their mathematical meaning.

Accept a native-grid control guess and reduce it deterministically by taking each block's first current. Successful blocked plans already satisfy equality within each block. Advancing r=b native knots preserves block alignment; append a constant terminal block when filling the tail.

Reduced backend multipliers must not be mislabeled as canonical full-grid multipliers. Specify and test the mapping for collapsed input rows and block bounds, including complementarity for any stage-dependent bounds supported. Alternatively return no unsupported multiplier components and deliberately reset them in the driver. The initial continuous single-shooting path may use this explicit reset, as long as it does not retain stale full-grid multipliers or rely on `canonicalize_duals` to guess reduced dimensions. Preserve multipliers normally for b=1. Compare primal-only versus retained-dual behavior as an optimization later, not a prerequisite for policy correctness.

Generic dimensions must distinguish native controls, independent block controls, retained NLP rows and canonical result rows. Audit `single_shooting_dimensions`, primal/constraint bounds, callback construction, constraint-violation reporting and `canonicalize_duals` together. Verify feasibility on the expanded physical plan.

## 6. CasADi: native graph with blocked decisions

Implement the same problem setting and public native-grid plan semantics without requiring trajopt's numerical solver.

1. Add `control_block_steps` to `CasADiWaveformProblem` and its builder. Forward `replan_dt` and common start through `CasADiMPCController.from_config`. Native `dt == model.dt` validation remains.
2. Refactor `_build_solver` so the native Rollout, tracking, effort, spectral and terminal expressions remain available for a full native U. Construct reduced symbolic V and use `U[j]=V[j//b]` for the NLP. A native-U CasADi function reused by the reduced NLP can preserve the evaluation interface without duplicating Cost logic.
3. Set NLP dimension and current bounds to `(H/b)*m`. Add identical input-only Kirchhoff constraints once per block, while retaining native state/output-dependent evaluations. Do not simply repeat the existing per-native equality loop after reducing decisions.
4. In `solve_state`, reduce the initial native guess, solve over V, expand to U, and return `planned_u` as `(H,m)` and `predicted_y` as `(H+1,p)`. Keep the method independent of controller history and playback.
5. Keep `_predicted_y_fn`, `_evaluate_costs` and public `decompose_cost` native-plan evaluators, so parity tests can compare the same explicit U between implementations. Their argument shapes must not silently change to block-grid shapes.
6. Shift `_u_guess` by the consumed native prefix before the next solve, with terminal fill; stop shifting it unconditionally by one after each solve. Retain one persistent compiled CasADi solver per structural problem.
7. Apply the same failure-prefix invalidation and native applied-current history contract as trajopt.

Initially reject blocked mixed-integer/activation configurations. Do not reinterpret activation counts, pulse durations or `max_active_intervals` as block counts. Existing b=1 integer paths retain their semantics; extending blocking to them requires a separate explicit design.

## 7. Costs, logging and downstream metrics

Keep all tracking/effort/spectral/terminal calculations on their original native prediction and Frame grids, including causal history and the currently excluded initial state term. Blocking changes admissible U, not Cost units. Charge must account for all six native intervals in a block. Neither replanning less often nor reducing decisions warrants a factor-of-six weight adjustment.

Extend both controller logs with the same fields:

| Field | Meaning |
| --- | --- |
| `solve_attempted` | True only when optimization was invoked on this tick. |
| `plan_index` | Native index of the emitted command within the execution prefix; -1 during Warm-up. |
| `fallback` | Whether this tick's applied command comes from fallback. |
| `controller_time` | Elapsed native update time, including absorption, planning when due, and playback preparation. |

Retain `u` as the physical current actually delivered at every native tick. Retain `warmup` as the common pre-control phase; playback is not Warm-up. `success`, solve status, `solve_time`, predicted plans and Cost components describe optimization only where `solve_attempted` is true. On other ticks use explicit playback/Warm-up status, false as the unused success value, NaN solve diagnostics/plans and the actual current; readers must use the attempt mask. Fixed array shapes remain compatible with the logger. Save solve-time predictions/plans before warm-start advancement.

Update these consumers together:

- `comparison.solver_metrics` and `cost_metrics`: mask attempts, count failures as attempts, compute solve latency/success without playback or Warm-up records, and use `replan_dt` as the deadline denominator. Aggregate native `controller_time` separately to include absorption/playback work.
- `score_run`: retain actual command timestamps and truncate interval durations at the simulation end for Delivered Charge. A logged endpoint command has zero delivered duration.
- `Run.planned_predictions` and plotting: select attempt rows/timestamps while preserving native spacing inside each Rollout. Align observed outputs explicitly; filtering plans must not shift their observation targets.
- `predictor/replay.py`: continue absorbing all native observations and applied currents. Diagnostic actual-current Rollouts may be evaluated at every native tick; label them independently from solver attempts and solver plans.
- `ood.py`: replace `~warmup` as the definition of solved decisions wherever plans, timing or success are read. Replay/bootstrap sampling at decision times must use the correct population.
- `mixed_integer_comparison.py`: preserve existing b=1 behavior and ensure any shared solve-time/failure readers do not treat replay ticks as attempts.
- `control/benchmark.py`: report fixed-OCP latency separately from complete controller runtime/solves per simulated second. A one-step driver benchmark alone does not exercise the new policy.

Construction/compilation remain separate from repeated runtime. Offline simulations still apply commands in simulated time; deadline reporting does not create wall-clock feasibility or a delay-compensation algorithm.

## 8. Verification and implementation order

Make each change verifiable at its public boundary before implementation. Use the existing tiny Predictor checkpoints and deterministic small OCPs for the fast loop; full Plant runs belong to the subsequent experiment, not the unit suite.

| Boundary | Required evidence |
| --- | --- |
| Controller `update`/`evaluate`, both backends | Prefix one retains successful control/Cost/Priming parity. Six-step execution emits a solved prefix in order and replans only on boundaries. Changing intermediate measurements does not alter the queued prefix, but affects the next solve through updated history. |
| State Absorption through controller execution | The next solved initial state contains every native measurement and preceding applied current, including fallback zeros and reduced-coordinate conversions. No future observation enters a planned Rollout. |
| Warm-up/start | Two different Priming requirements share one aligned release; both emit zero until then. Invalid ratios, inadequate start times and misaligned block/horizon settings are rejected. |
| trajopt MPC/Program | Six-count advance matches six one-count advances for time, reference and all supported warm-start state; measurement pins the next initial state; Program identity remains stable. |
| Blocked solve, both backends | At H=12,b=6, solve over two independent control vectors, return twelve native currents and thirteen states. Every returned block is constant; native predictions and Cost match explicit expansion. |
| Constraint/derivative callbacks | Kirchhoff works without duplicate equalities per block; expanded current bounds hold; finite differences agree with reduced derivatives. Unsupported constraint/backend combinations are rejected. |
| Backend parity | Same checkpoint/state/native U gives the same predictions and Cost components in JAX/CasADi. Compare optimized feasibility/Cost and, for a well-conditioned deterministic case, controls within solver tolerances. |
| Results and metrics | Mixed attempt/playback/failure records produce correct attempt counts, success, p95/deadline values, plan timestamps and actual-duration Charge. Playback does not dilute failures or latency. |

Suggested sequence:

1. Lock down existing one-step controller behavior; implement timing/playback and failure history with b=1 in both backends, plus trajopt count-based advancement.
2. Add attempt-aware logs/readers and config validation so the distinct-current controller is usable end to end.
3. Implement trajopt blocked single shooting and CasADi block variables; verify expanded-plan/Cost/constraint parity before controller integration.
4. Exercise blocked playback with the same measured histories and horizon as distinct-current execution. Confirm no new solver construction/recompilation per tick.
5. Run focused checks in `tests/test_mpc.py`, `tests/test_casadi_mpc.py`, CNN parity tests, `tests/test_validation.py`, `tests/test_comparison.py`, and relevant replay/view tests. Upstream use `test/unit/test_single_shooting.py`, `test/unit/test_program.py` and MPC characterization/recompilation tests. Use `uv` for commands and complete each repository's required checks.
6. Prepare separate experiment-local configurations for native, distinct-current and blocked-current policies. Then execute the strategy's development study as a separate task, holding checkpoint/weights fixed before independent effort calibration.

Delivery is complete when both continuous solver paths can run both policies under the same timing and result contract, existing supported one-step behavior passes regression checks, and metrics faithfully distinguish prediction, actuation and optimization rates. No sixfold wall-time improvement or preserved suppression is assumed by implementation acceptance.
