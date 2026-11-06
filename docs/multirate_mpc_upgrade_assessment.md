# Multirate MPC upgrade assessment

Assessment date: 2026-11-02. This is an implementation review and migration plan;
no controller changes or campaign runs were made.

## Verdict and review boundary

The upgraded trajopt supplies the numerical foundation required by our
[implementation plan](multirate_mpc_implementation.md). No upstream blocker was
found for the initial continuous IPOPT single-shooting campaign. Neuro still
needs the execution policy, CasADi blocking and attempt-aware reporting. The
upgrade currently breaks existing neuro execution, so compatibility migration
comes before multirate experiments.

Reviewed trajopt range: `9a8346e6ea3c674c7f23a79742ec2e3b02083d4c..e7ff2ee08344e52ef9401de717ce772b9044fca4`
(`HEAD~15..HEAD`). Both neuro's lockfile and the installed distribution's
`direct_url.json` identify the ending revision. Sources are the
[upstream spec](../../trajopt/docs/multi-rate_mpc/spec.md), the actual code and
tests at that revision, and our
[campaign strategy](hyperparameter_sweep_strategy.md). Source links to trajopt
assume the sibling checkout used for this review.

## Fit to our requirements

| Requirement | Upgraded package | Remaining neuro work |
| --- | --- | --- |
| Consume six native Knot Points | `MPC.advance(6)` moves primal/dual warm starts, reference and clock; `measure` pins absorbed state | Advance at the next solve boundary, skip before first solve, remove post-solve advancement |
| Optimize one current per six native steps | Structural `control_block_steps`; reduced decisions in both IPOPT shooting formulations | Forward the field through builders and problem reconstruction; implement CasADi reduced variables |
| Preserve native prediction and Cost geometry | Expanded native X/U, canonical Z, original native Objective | Keep native dt in inference, Fourier geometry, output timestamps and Cost decomposition |
| Kirchhoff and physical bounds | Eligible input rows deduplicated; block bounds intersected; expanded-plan feasibility checked | Existing standard LinearConstraint is suitable; no new constraint-lowering layer needed |
| Stable compiled problem | Persistent Program-owned IPOPT handle and compiled callbacks | Retain each MPC/Program; do not reconstruct per tick |
| Correct dual representation | Canonical duals for unit blocks; explicit canonical zero arrays for blocked solves | Delete obsolete dual adapter and its assumptions |
| Six native absorptions per solve | Intentionally outside package scope | Implement scheduling, playback, previous applied current and common Warm-up release |
| Failure playback and reliable metrics | Failed expanded iterate and diagnostics retained honestly | Invalidate prefix, emit zero until next scheduled attempt, log attempts separately |

Evidence: [MPC advancement](../../trajopt/src/trajopt/mpc.py#L284),
[Program pairing guards](../../trajopt/src/trajopt/program.py#L230),
[single-shooting solve and persistent handle](../../trajopt/src/trajopt/transcription/single_shooting.py#L690),
[native result and dual construction](../../trajopt/src/trajopt/transcription/single_shooting.py#L762).

The upstream scope deliberately exceeds our initial requirements: blocked
multiple shooting, binary continuation and linear Horizon constraints also work.
Those extensions justify removing local duplicate solver code, but do not require
adding blocked integer controllers or multiple shooting to this campaign.
Single shooting still rejects state bounds; unsupported solver/blocking pairings
fail at construction. Blocked canonical dual reset is intentional. Binary
continuation remains local optimization without a global certificate. Custom
input constraints need truthful time-independence metadata to be deduplicated;
our built-in Kirchhoff LinearConstraint already has it.

## Immediate compatibility problems

1. [Both controller driver paths](../src/neuro/control/mpc.py#L227) still call
   removed `MPC.shift(dt)` (also line 252). Migrate to integer-count `advance`.
   Do not pass elapsed seconds to `advance`, and do not combine a retained
   post-solve advance with the eventual boundary advance.
2. [The default binary/Horizon solver selection](../src/neuro/control/solvers.py#L38)
   imports local homotopy, whose
   [private upstream import](../src/neuro/control/homotopy.py#L17) no longer exists.
   Replace the local implementation with stock `SingleShooting(Ipopt(...))`.
3. [CanonicalDuals](../src/neuro/control/solvers.py#L124) is obsolete and hides
   the solver's concrete type from upstream pairing validation. It will make
   valid blocked or binary/Horizon combinations appear unsupported. Delete the
   wrapper and its call sites rather than retaining a no-op wrapper.

The controller also retains the known applied-current bug: after a failed solve
emits fallback, [it stores `u_solved` as `_u_last`](../src/neuro/control/mpc.py#L253).
Fix this while separating State Absorption from playback, including conversions
for Nullspace and epigraph coordinates. The stored inference input must represent
the delivered physical current in the adapter's expected coordinates.

## Simplifications worth taking

- **Delete local homotopy and dual repair.** The stock solver now handles binary
  continuation, canonical results and linear Horizon constraints. Our
  [activation cap already uses LinearHorizonConstraint](../src/neuro/control/problems.py#L232).
  Preserve application activation/amplitude construction and physical-current
  extraction. Map continuation options explicitly: local near-binary warm-start
  shortcuts and upstream continuation need not yield identical iterates or times.
- **Let trajopt own warm starts.** Remove write-only `_u_guess`/`_active_guess`
  bookkeeping in TrajOptMPCController where confirmed unused; retain CasADi's
  actively consumed guesses. Use `advance(r)` without another driver abstraction.
- **Start with physical-electrode decisions.** Our standard Kirchhoff constraint
  can be deduplicated upstream. For H=102 and three electrodes, six-step blocking
  reduces 306 decisions to 51 already. Nullspace reduction to 34 is optional
  subsequent optimization, not a prerequisite for the initial blocked campaign.
- **Keep native evaluators and result shapes.** Reuse full-plan Cost decomposition
  and prediction readers. There is no need for block-grid logs or a second Cost
  normalization. Keep neuro's whole-horizon Objective/solver compatibility guard.
- **Share only timing/playback and attempt selection.** A small contract across
  both backends prevents schedule drift; a general scheduler is unnecessary.
  Consolidate the metric logic already identified in the
  [simplification checklist](repo_simplification_checklist.md).

Do not replace CandidateMPCController's activation schedule blocks with uniform
current blocking: enabled native currents there can differ, and exhaustive
candidate enumeration is not equivalent to local continuation. Its direct private
transcription helper calls need a separate migration or explicit rejection of
non-unit blocking. Keep that outside initial campaign scope.

## Work still required locally

No `replan_dt`, `control_start_s` or `control_block_steps` integration exists in
neuro source today. Implement the accepted contract from the existing plan:

- At every native tick absorb the measurement and preceding applied current.
  Use integer native indices anchored at time zero for common start and solve
  boundaries. Preserve readiness gating and reject inadequate common starts.
- At a solve boundary advance the consumed prefix, pin the current absorbed state,
  solve, and copy the unshifted physical native plan for playback. Intermediate
  ticks only absorb, play and log. Failed attempts invalidate the prefix until the
  next boundary; preserve attempted-solve diagnostics separately from fallback.
- Forward block structure through NeuroProblem, `_assemble_problem`, builders and
  epigraph reconstruction in [problems.py](../src/neuro/control/problems.py#L145).
  Keep supported wrappers honest about physical-current bounds and coordinates.
- In [CasADi](../src/neuro/control/casadi.py#L660), separate native-U evaluation
  from the reduced-V NLP, deduplicate input equalities, expand public results and
  shift the native guess by the consumed prefix before the next solve. Keep one
  solver per structural problem. Initially reject blocked integer configurations.
- Extend validation independently for native rate, integral replanning ratio,
  b=1 or b=r, H>=r, H divisible by b, and aligned sufficient common start.
- Add `solve_attempted`, `plan_index`, `fallback` and `controller_time` consistently
  across both logs. Use attempt masks in comparison and OOD. Preserve native
  observations and applied currents for OOD analysis. Solve deadlines
  use replan_dt; within-plan spacing still uses native dt.

The current [solver/Cost metrics](../src/experiments/comparison.py#L240) and
[OOD readers](../src/experiments/ood.py#L396) use `~warmup` as the solve population;
that becomes wrong with playback. Update benchmark decision counts and report
fixed-state solve latency separately from complete controller runtime and solves
per simulated second.

Delivered Charge is partly done already:
[score_run](../src/experiments/comparison.py#L325) uses logged command durations and gives
the endpoint command zero duration; OOD has similar handling. Preserve that work
and test irregular/end timestamps rather than replacing it with constant-dt sums.

## Recommended implementation sequence and acceptance gates

1. **Restore existing unit-step operation and remove duplicate solver code.**
   Remove dual wrappers, adopt stock binary/Horizon single shooting, migrate
   advancement, and update obsolete tests/config paths. Verify successful unit-step
   control/Cost behavior, cap feasibility, original Cost diagnostics, canonical
   result dimensions, two solves through one Program and failure-current history.
2. **Deliver distinct-current execution end to end in both backends (b=1,r=6).**
   Add timing, common start, immutable prefix playback, failure invalidation and
   attempt-aware consumers together. A deterministic public-boundary test must
   show U[0..5] emitted in order, no intermediate solves, all six absorptions
   reaching the next initial state, and no command changes from intermediate
   measurements. Test shared release, invalid configurations and mixed
   attempt/playback/failure metric records.
3. **Add blocked continuous execution (b=r=6).** Forward structure, implement
   CasADi V, then test H=12,b=6: two control vectors, twelve expanded currents,
   thirteen states, constant blocks, physical bounds/Kirchhoff and matching native
   predictions/Cost. Check checkpoint-level JAX/CasADi parity and retained solver
   identity. Keep H=102 for the actual matched 1.02 s campaign horizon.
4. **Qualify and run the development timing study.** Qualify both frozen Waveform
   checkpoints at the actual horizon/current sequences. Compare native, distinct
   and blocked policies with fixed checkpoints/weights, common Warm-up Period and
   all five development seeds. Measure Burden, actual Delivered Charge, failures,
   latency and complete controller time; then calibrate each 60 ms policy's effort
   independently. Leave reserved final seeds untouched until the protocol freezes.

Steps 1–3 are implementation work; step 4 is a separate experiment. Slower
replanning does not change Predictor dt, training Loss or spectral geometry,
justify sixfold Cost reweighting, or establish real-time feasibility. The 20–30 mAs
selection target remains experiment scoring, not a new hard constraint.

## Verification performed

Upstream, real IPOPT available, **70 passed in 79.97 s, no skips**:

```powershell
uv run pytest test/unit/test_mpc_advance.py test/unit/test_problem_blocking.py test/unit/test_blocked_single_shooting.py test/unit/test_blocked_multiple_shooting.py test/unit/test_input_constraint_dedup.py test/unit/test_program_lifetime.py test/unit/test_blocked_binary_continuation.py test/unit/test_single_shooting.py -q
```

This covers numerical blocked solves, finite-difference derivatives, constraint
deduplication, canonical stationarity, failed blocked dual reset and handle/trace
reuse. It does not establish neuro checkpoint compatibility or closed-loop benefit.

Neuro, installed upgraded dependency, **73 passed / 13 failed in 44.57 s**:

```powershell
uv run --no-sync pytest tests/test_mpc.py tests/test_casadi_mpc.py tests/test_validation.py -n 2 --tb=short
```

Ten failures call removed `shift`, two import the removed private homotopy helper,
and one asserts the old incorrect single-shooting dual dimensions. These are a
reproducible migration baseline, not evidence of upstream numerical failure.
Full suites and actual campaign runs were not executed. Existing user edits to
the thesis and dependency lock were preserved.
