# Observable-Space Prediction for the Spectral MPC

Exploration this spec resolves: [`docs/direct_observable_prediction.md`](../direct_observable_prediction.md).
Section references below (§3.1, §4.2, ...) point into that document.

## Problem Statement

The spectral MPC costs a one-sided log hinge against a healthy envelope, but it computes that cost
by rolling the autoregressive raw-EEG predictor 75 steps and running a symbolic DFT over the
predicted waveform. Three consequences, all measured rather than estimated:

1. **The rollout dominates the solve.** At `horizon: 75`, single shooting, MX: the objective costs
   19.5 ms and its Jacobian 71.0 ms. The symbolic DFT is 3.3 ms of that Jacobian — under 5%. The
   75 sequential evaluations of a 148k-parameter map over a 960-dimensional state are the other
   95%. Eliminating the DFT alone buys nothing; eliminating the rollout is worth roughly 10x.

2. **The Predictor optimises a different quantity than the Cost consumes.** Training minimises a
   curriculum MSE on the waveform plus an auxiliary STFT Loss; the controller scores a spectral
   hinge. To get the spectrum right the rollout has to stay waveform-plausible 1.5 s ahead, which
   is well past where free-run NMSE saturates.

3. **The geometry is coupled but unverified.** `PsdEnvelope` carries the segment length and hop
   that define the Cost, `neuro.validation` checks the YAML against the npz, and nothing checks
   either against what the Predictor was fit on. `mse02_psd_mpc_spectral.yaml` already runs
   `horizon: 75` against an artifact whose `native_horizon` is 50.

The load-bearing risk is separate from all three, and it is why this spec is gated rather than
merely implemented (§3.6). Stimulation has no direct feedthrough to the sensors: electrode currents
project through the Field Projection into Stimulation Drives inside the pyramidal firing-rate
sigmoid, so suppression works by shifting a region's operating point. Measured on the training set
that Dose-response is real and monotone, but small — `corr(mean TP9 current, log power) = -0.17`,
and the full ±2.4 mA range moves log power by ~0.28 nats against a std of 0.53, about half a
standard deviation across the entire admissible Control Budget. A ridge predicting the 3100-dim
log-power target from the history state alone scores 0.131 held-out R²; adding the 225-dim control
block scores 0.132. **A model that is affine in the Control Current has already been measured to
fail**, and any observable Predictor that ends up Control-blind hands the solver no gradient to
descend regardless of how fast it evaluates.

## Solution

Forecast the Observable itself, and recurse on the **frame grid** rather than the sample grid.

Most Observables reduce along the time axis: the STFT Loss reduces a length-`L` Segment to one
Frame per hop `R`, and `eeg_ms` reduces a trailing segment to one scalar per channel. So a Rollout
covering the same Control Horizon needs far fewer nodes. At the deployed geometry (`fs = 50 Hz`,
`H = 75`, `L = 50`, `R = 25`) the frame grid holds **2 nodes instead of 75**, and each node carries
a lifted state of order 10¹ rather than 960. That is the whole of the 95% identified above, removed
without giving up recursion.

```text
[Raw EEG & u history]  ──lift E──►  z_0            (raw history in, so no phase is discarded)
                                     │
        u_bar_0 ─────────────────────┤ f_theta      (one shared transition per Frame)
                                     ▼
                                    z_1  ──► l_hat_1 = C z_1     log-power Frame
        u_bar_1 ─────────────────────┤ f_theta
                                     ▼
                                    z_2  ──► l_hat_2 = C z_2
                                                     │
                                                     ▼
                                    [elementwise hinge on l_hat vs log P_ref]  ──► Cost
```

Keeping recursion is what makes this affordable and what makes it defensible:

- **Weight sharing returns.** One transition map is reused at every Frame, so the parameter count
  drops by an order of magnitude against the horizon-wide direct map's ~565k and §3.1's data-budget
  objection shrinks with it.
- **The geometry stays a config knob.** More Frames simply means more recursion steps, so `H`, `L`
  and `R` are not frozen into an output layer. §3.4 dissolves; `native_horizon` keeps the meaning
  it already has, and the existing "horizon exceeds trained horizon" warning keeps working.
- **Causality is structural.** Frame `m` can only depend on Control Currents that landed before its
  Segment ended, because a recursion cannot see forward. §3.3 dissolves; no causal masking is built.
- **The literature stops arguing against us.** §2.2's usual citation (Marcellino, Stock & Watson)
  concludes iterated forecasts beat direct ones and that the gap grows with horizon. Recursion in
  Observable space *is* the iterated option, applied at a coarser clock.

The cost of keeping recursion is §3.2's hard case: the lifted state must carry enough phase and
velocity to be Markovian on the frame grid. This spec buys that off by lifting from the **raw** EEG
and control history at `m = 0` — the same history state the incumbent Predictor already primes —
so nothing is discarded at the start of the Rollout, and only the propagation between Frames lives
in Observable space.

## Implementation Decisions

### 1. Observable geometry is defined once and shared three ways

The geometry that defines the Observable (segment length, hop, scored bin range, bin pooling, Frame
Kernel width) is the single source of truth for the offline target, the training Loss, and the MPC
Cost. It is defined once in `neuro.config` as a geometry model per Observable kind, and the existing
`StftSpec` / `EegMsSpec` compose it rather than redeclaring it — their weight and epoch-gate fields
stay on the Loss side, their geometry fields and validators move down. No geometry field, and no
`n_frames()`-style derivation, exists in two places after this change.

Two kinds are supported, selected by config in exactly the way `LossSpecs.active()` already
dispatches on which spec is non-`None`:

| Kind | Frame value | Shape at `H = 75`, deployed geometry |
| :--- | :--- | :--- |
| `stft` | pooled log power per `(channel, bin)`, DC unscored | `(2, 62, 25)` |
| `eeg_ms` | log mean-square power per channel | `(2, 62)` at `window_s: 1.0`, `hop_s: 0.5` |

`eeg_ms` at its `neuro.metrics` defaults (`window_s = 0.1`, `hop_s = 0.05`) instead yields ~36
Frames of 62 values. That is still 36 recursion nodes on a state of order 10¹ against the
incumbent's 75 nodes on 960, so the frame-grid argument holds; but the geometry that matches the
deployed spectral Cost is the 1.0 s / 0.5 s one, and that is what the shipped config uses.

### 2. The Predictor: lift, shared frame transition, log readout

- **Lift.** `z_0 = E(x_0)`, an MLP over the existing history state layout — the standardized EEG
  window `(n_y, C)` concatenated with the raw control window `(n_u, m)`, byte-identical to what
  `MLPArtifact.prime` produces today. Reusing the layout is what lets State Absorption, Priming,
  `is_ready` and `initial_state` carry over unchanged, which §4.2 correctly identified as the part
  of the interface that survives.
- **Transition.** `z_{m+1} = f_theta(z_m, u_bar_m)`, one MLP shared across all Frames. Not the
  linear/bilinear Koopman form §7 sketches: a model affine in `u` is precisely the model §3.6
  measured at 0.000 added held-out R², and the Dose-response table itself saturates
  (`+0.263, +0.265, +0.219, +0.175, +0.058` across current bins), so an affine-in-`u` structure
  cannot represent the effect it is being built to capture.
- **Readout.** `l_hat_m = C z_m + c`, emitting **standardized log power** directly. There is no
  exponential in the CasADi graph, no positivity constraint on the network output, and no
  `LOG_FLOOR` applied to a predicted quantity — the floor stays where it belongs, on a *measured*
  power (§3.5). The artifact carries an `l_std` `Standardizer` fitted on the training log-power
  targets, mirroring `y_std` / `u_std`; the symbolic bridge un-standardizes before the hinge.
- **Sizes are config knobs**: lifted state dimension, and hidden width / depth for the lift and the
  transition separately.

### 3. Control aggregation onto the frame grid

`u_bar_m` is the unweighted mean of the Control Currents over Frame `m`'s Segment support. The
Segments overlap when `R < L`, so the map from the `(H, m)` control block to the `(M, m)` frame
means is a fixed linear operator and gradients flow back through it unchanged.

This is the sharpest assumption in the spec and it is stated as one: the Predictor cannot see
intra-segment control shape. The justification is mechanistic — currents act by shifting a regional
operating point across the bifurcation rather than by cancelling cycles (§3.6), so the mean is the
right first-order summary of a Segment's Stimulation Drive. It is also the first thing to revisit
if the Control-blindness gate comes back marginal. No taper-weighted average is built; the Hann
taper belongs to the spectral estimator, not to the plant's response.

### 4. Artifact

A new artifact type with `model_type: "observable"`, in the repo's existing single-`.npz`-plus-JSON-
meta convention, dispatched by `load_any_artifact` and bridged by `build_symbolic_model` exactly as
`"mlp"` and `"esn"` are. It carries the lift / transition / readout weights and activation, `n_y`,
`n_u`, `n_channels`, `n_controls`, `dt`, `downsample`, `y_std`, `u_std`, `l_std`, `TrainingProvenance`,
and — this is the part that does not exist on any current artifact — **the resolved Observable
geometry it was trained against**, including the kind, the segment length, the hop, `fs`, the scored
bin range, the bin pooling and the Frame Kernel width.

It records geometry, not a frozen horizon. An artifact trained at 2 Frames can be deployed at 3;
`native_horizon` continues to mean "the Rollout depth it was fit at" and the existing extrapolation
warning in `validation._check_predictor` continues to apply unchanged.

### 5. Protocol: a sibling, not an optional-member `SymbolicModel`

`SymbolicModel` in `neuro.types` is left alone. A sibling protocol is added for models that forecast
an Observable over the Control Horizon in one shot:

- carried over verbatim: `state_shape`, `n_controls`, `n_channels`, `native_horizon`,
  `initial_state`, `absorb`, `is_ready`
- new: the resolved Observable geometry, and a symbolic `(x_0, u_seq) -> l_hat` member plus its
  cached `ca.Function`
- deliberately absent: `f_step`, `f_out`, `step`, `output`

Making the stepping members optional on the existing protocol was considered and rejected: every
current consumer would then have to guard, and a model that silently lacks `f_step` degrades the
incumbent path rather than failing at build time. `build_symbolic_model` returns the union; the MPC
branches on which it received.

### 6. MPC cost path

`MPCNlp.build` gains a branch for observable models. On that branch there is no `_rollout_cost`, no
per-step `y_nodes`, no defect constraints and no shooting roots — so `shooting_depth`, `phi_vars`
and `get_phi` are not merely unused, they are not constructed. What is identical to the incumbent
path: the decision variable `u` over `(H, m)`, the box bounds `-u_max <= u <= u_max`, the Kirchhoff
`_sum_to_zero` equality, `w_u`, and the `w_u_l1` epigraph and its slacks.

The Cost becomes an elementwise hinge on the forecast:

```text
J_PSD = mean over (m, c, f) of [ max(0, l_hat[m,c,f] - log P_ref[c,f]) ]^2
```

reduced over Frames, channels and bins together — never over Frames alone — so `w_psd` keeps
exactly the meaning it has in `_spectral_hinge_cost` and a hot Frame cannot be cancelled by a cold
one. `_spectral_hinge_cost` itself is untouched and stays in service for the incumbent path.

`w_y` and `w_y_terminal` have no meaning without per-step outputs. They are **rejected with an
error**, not silently ignored: dropping a weight the YAML asked for is exactly the class of failure
`neuro.validation` exists to prevent. An explicit `shooting_depth` below the horizon is rejected on
the same grounds.

### 7. Reference envelope for both kinds

`stft` uses the existing `PsdEnvelope` and `data/healthy_psd.npz` unchanged. `eeg_ms` needs a
per-channel healthy mean-square reference; it is written into **the same npz** by
`scripts/build_healthy_psd.py` as an additional array, computed as the time-domain windowed
mean-square power of the healthy trajectories on the same segment/hop grid. Deriving it from the PSD
envelope by Parseval was rejected — the spectral cost leaves DC unscored while a time-domain mean
square includes the offset, and reconciling the two would make the reference something other than
what the Predictor forecasts. `neuro.spectral` grows an envelope type per kind; `PsdEnvelope` keeps
its current shape and loader.

### 8. Controller

`MPCController._solve` currently seeds its warm start by rolling `model.f_step` forward to produce
`phi_guess`. With no shooting roots that seed collapses to the shifted `u_guess` plus, when `w_u_l1
> 0`, its L1 slacks. The branch lives in`_solve`.`update`, State Absorption, the Warm-up Period
behaviour and`MPCControllerLog` are unchanged.

### 9. Validation

`neuro.validation` gains the check §4.2 names as the one thing standing between a geometry edit and
a silently wrong Cost: on an observable artifact, the geometry **recorded in the artifact** must
match the reference envelope's — kind, segment length, hop, `fs`, channel count, scored bin range,
pooling and Frame Kernel width. Mismatch raises `ConfigConsistencyError`. The existing YAML-vs-npz
check (`psd_window_s` / `psd_hop_s`) stays and is now the weaker of the two.

The predictor training config gains an `observable` block naming the kind and its geometry, plus the
architecture knobs from decision 2.

### 10. Training

Supervised regression through the `M`-Frame recursion. Targets are the true future's log-Observable
on the frame grid, built by the same geometry object the Loss uses, from the same sliding-window
`(y_past, u_past, u_future)` construction the incumbent dataset builder already produces. The
training Loss is a plain MSE in standardized log-Observable space over `(frame, channel, bin)` — the
hinge stays in the controller, because training a hinge would discard every gradient from Frames
already under the envelope.

BPTT depth is `M` (2–3 Frames at the deployed geometry), not 75, so the backward pass is cheap
enough that the Frame count is not a training-cost constraint.

§3.1's data-budget objection is reduced, not eliminated, and the spec says so: weight sharing across
Frames returns and the parameter count drops roughly an order of magnitude, but each target still
spans `H` samples, so the honest independent-sample count remains "non-overlapping targets per
trajectory", on the order of 10³ for the current 200-trajectory training set. The reported sample
count is the non-overlapping one.

The existing `du_sensitivity` diagnostic on `TrainingResult` carries over as the Frobenius norm of
`d(l_hat)/d(u_future)` — the first half of §6 stage 4, computed during training rather than only in
a probe.

### 11. Gates, with binding kill criteria

These are deliverables, not aspirations, and they run in the stated order. Each has a kill criterion;
if it trips, the next stage does not run.

1. **Control-blindness gate.** Train the observable Predictor twice at identical architecture,
   schedule and seed set (≥ 3 seeds): once on the real control block, once with the future controls
   zeroed. *Kill:* if the full model does not beat the Control-blind one on held-out trajectories by
   more than the seed-to-seed spread, stop. No amount of solver speed fixes a Predictor the solver
   cannot push on.
2. **Direct-vs-iterated arm.** The same probe trains the horizon-wide direct map `g(x_0, u)` of
   §1–6 as a third arm, so §2.2's iterated-beats-direct question is measured on this data rather
   than assumed in either direction. This map exists **only inside the probe script** — it gets no
   artifact type, no symbolic bridge and no MPC path.
3. **Incumbent baseline.** Score the observable Predictor against `artifacts/nonlinear_mse02_psd`
   pushed through the identical geometry. The baseline is the incumbent, not the training mean.
4. **Gradient sanity.** Sweep a constant TP9 Control Current from −2 to +2 mA and check the
   forecast log power decreases monotonically, reproducing §3.6's table; check that
   `‖grad_u l_hat‖` is not orders of magnitude below `‖grad_x0 l_hat‖`, since the solver only ever
   sees the former.
5. **Solve-time measurement.** Extend `scripts/probe_solve_time.py` to time objective and Jacobian
   on the *built* model. §4.1's 10x was measured on an assumed architecture; the claim is only
   earned once the trained one is timed.
6. **Closed-loop benchmark.** Seizure Burden against `mse02_psd_mpc_spectral.yaml` and
   `threshold_control.yaml` over the same seed set. Seizure State is scored on source-space LFP,
   which no Observable model forecasts, so no amount of offline Observable accuracy substitutes for
   this run.

## Testing Decisions

A good test here asserts behaviour at a seam a caller actually uses, on a tiny synthetic artifact
with random weights, and never reaches into private state or asserts on a trained model's accuracy.
The four seams below were chosen deliberately; each has prior art in this repo to match.

### Seam 1 — Observable target vs training Loss

The offline target builder and the training Loss must produce the same numbers from the same
trajectory and geometry, or the Predictor is trained against one quantity and scored on another.
Tests: the builder's output equals `StftLoss.log_spectrogram` and `log(EegMsLoss.windowed_power)`
to floating-point tolerance; the resolved Frame count matches the geometry's own derivation across
a range of `(H, L, R)`.

Prior art: `tests/test_predictor_losses.py::test_spectrogram_matches_scipy`,
`::test_stft_frame_kernel_pools_before_the_log`, `tests/test_metric_losses.py`.

### Seam 2 — Torch ↔ CasADi parity

The control path never runs torch, so the artifact's NumPy forecast and the CasADi symbolic
forecast are two independent implementations of one function and must agree. Tests: a tiny synthetic
observable artifact forecast through both paths agrees to ~1e-10 across activations and Frame
counts; artifact save/load round-trips weights, standardizers and the recorded geometry exactly.

Prior art: `tests/test_predictor_module.py::test_torch_rollout_matches_casadi`,
`tests/test_nn_predictor_casadi.py::test_multistep_rollout_matches_artifact_rollout`,
`::test_mlp_artifact_round_trip_preserves_exact_weights_and_meta`. The existing
`test_predictor_module.py::test_control_path_never_imports_torch` must keep passing with the new
modules added to its parametrisation.

### Seam 3 — Cost equivalence at the MPC seam

The new Cost must be the same functional as the incumbent one, so that `w_psd` transfers and a
comparison between the two controllers is a comparison of Predictors rather than of objectives.
Tests: fed the *true* log-spectrogram of a trajectory in place of a forecast, the observable hinge
equals `_spectral_hinge_cost` evaluated on that trajectory's per-step outputs, and both equal
`neuro.spectral.hinge_penalty` over `compute_periodograms`; the hinge is exactly zero when the
spectrum is under the envelope everywhere; `w_y > 0` and `shooting_depth < horizon` raise on the
observable path.

Prior art: `tests/test_mpc_controller.py::test_spectral_cost_matches_numpy_periodogram`,
`::test_spectral_cost_is_zero_when_under_envelope`.

### Seam 4 — Controller through `simulate.Simulation`

The highest seam available: a tiny synthetic observable artifact wired into `MPCController` and run.
Tests: the Warm-up Period emits zero Control Current until `is_ready`; solved controls respect the
per-electrode bounds; controls satisfy Kirchhoff's sum-to-zero at every step; a short closed-loop
run completes and produces logs; `from_config` loads an observable artifact and defaults the horizon
from it.

Prior art: `tests/test_mpc_controller.py::test_closed_loop_simulation_runs`,
`::test_warmup_emits_zero_until_window_filled`, `::test_update_respects_bounds`,
`::test_control_obeys_kirchhoff_current_law`, `::test_from_config_loads_artifact_and_defaults_horizon`.

### Additional guards

- **Causality as a property.** Frame `m`'s forecast is invariant to Control Currents landing after
  its Segment ends. Structurally guaranteed by the recursion, so the test is cheap and catches an
  indexing error in the control aggregation of decision 3. No prior art; a direct property test.
- **Geometry mismatch raises.** An artifact whose recorded geometry disagrees with the envelope's
  raises `ConfigConsistencyError`. Prior art: `tests/test_validation.py`.

### What pytest must not assert

Predictive accuracy, Control-sensitivity magnitude, and closed-loop Seizure Burden are **not** unit
tests. They depend on trained weights and on the data budget, they are the subject of the decision 11
gates, and encoding a threshold for them in the test suite would convert a measurement into a
tautology. Tests assert wiring, equivalence and invariants; scripts with kill criteria assert
performance.

## Out of Scope

- **Replacing the incumbent path.** The autoregressive raw-EEG MPC, `_spectral_hinge_cost`, and
  `mse02_psd_mpc_spectral.yaml` all stay runnable and unchanged. This adds a second path; the
  closed-loop benchmark is what would later justify retiring the first.
- **The horizon-wide direct map as a product.** It is built only as an arm of the decision 11 probe.
  No artifact type, no symbolic bridge, no MPC path, no config surface.
- **Linear / bilinear Koopman structure.** The state-independent `B` and the resulting
  near-QP subproblem §7 sketches are recorded as a structural ablation to try if the shared
  transition MLP overfits — not built now, for the reason in decision 2.
- **Causal masking architecture.** Causal convolutions and block-lower-triangular heads (§3.3) are
  unnecessary once the model recurses.
- **Paired trajectories.** §6's data gap — trajectories sharing a plant seed and differing only in
  the Control Current, which would measure `dP/du` directly instead of inferring it from a variance
  decomposition — is not generated here. It is the highest-value follow-up if the decision 11 gate
  comes back marginal rather than clear.
- **An ESN observable variant.** One implementation, MLP-based.
- **Source-space forecasting.** No Observable model predicts Seizure State; that stays a post-hoc
  score on region LFP.

## Further Notes

- **The numbers to beat**, all measured and all from the exploration document: Jacobian 71.0 ms at
  `H = 75`, objective 19.5 ms; the DFT is 3.3 ms of the Jacobian; the direct-map benchmark reached
  6.6 ms on the Jacobian at the same width. The observable recursion should land in that
  neighbourhood, and decision 11 stage 5 is what confirms it rather than assuming it.
- **The failure mode this spec is organised around** is Control-blindness, not accuracy. A model can
  score well on log power while being useless for control, because absolute log power is dominated
  by where in the seizure the Segment happens to sit. That is why the gate compares full against
  Control-blind rather than comparing test Loss against a threshold.
- **Two caveats on the speed claim carry forward.** A single objective-plus-Jacobian evaluation is
  not a solve — SQP iteration counts and QP conditioning may differ — and the observable model's
  Jacobian with respect to the control block is dense, which can move cost into the QP subproblem
  rather than removing it. Decision 11 stage 5 times the built model precisely because of this.
- **`docs/spectral_objectives.md` and `docs/spectrogram_loss_guide.md`** remain the reference for
  why the geometry is what it is; nothing in this spec changes the choice of `L`, `R` or the Frame
  Kernel, only who computes them and when.
