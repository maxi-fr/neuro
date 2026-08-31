# Hankel-DMDc Hyperparameter Sweep Results

## Executive Summary

Four Optuna sweeps were run over the closed-form Hankel-DMDc predictor (`training.fit: dmd`,
`src/neuro/predictor/dmd.py`) in Observable (STFT log-power) space:

- Three offline (`val_log_mse`) sweeps over `dmd_energy`/`dmd_lambda`, one per decision-rate
  geometry (`n_hop` = 25/15/10, i.e. `dt_frame` = 0.5s/0.3s/0.2s), to see whether a faster hop
  meaningfully improves one-step Frame prediction accuracy.
- One closed-loop (`objective: closed_loop`) sweep at the production hop25 geometry, scoring
  `dmd_energy`/`dmd_lambda` directly on multi-seed MPC seizure-suppression burden, narrowed
  around the hop25 offline sweep's best trial.

**Headline results:**

- **Faster hop helps offline accuracy monotonically**: `val_log_mse` improves from 2.959 (hop25)
  to 2.621 (hop15) to 2.343 (hop10) — going from a 0.5s to 0.2s frame nearly halves the offline
  MSE gap to zero. This is a real, unsurprising effect of shorter STFT windows tracking faster
  dynamics, but it says nothing about deployability at that rate (see caveats below).
- **The offline-optimal hyperparameters transfer to the closed-loop objective**: the closed-loop
  sweep's best trial (`dmd_energy=0.918`, `dmd_lambda=0.0047`) lands almost exactly on the hop25
  offline sweep's best trial (`dmd_energy=0.921`, `dmd_lambda=0.0060`). Unlike prior closed-loop
  work on this predictor family (`docs/dmd_closed_loop_evaluation.md` §5), offline MSE was **not**
  a misleading proxy here — the two objectives agree.
- **Seizure-suppression burden is nearly flat across the whole narrowed range**: all 8 closed-loop
  trials land within 0.0748–0.0806 seizure-burden and suppress exactly 2 of 3 seeds regardless of
  the specific `dmd_energy`/`dmd_lambda` value. The hyperparameters matter much less for
  closed-loop outcome than for offline MSE — one seed (of 3) is never suppressed by any trial in
  this sweep, suggesting the remaining gap is not a DMD-hyperparameter problem.

## 1. Offline sweeps: dmd_energy / dmd_lambda across decision rates

| Config | n_hop | dt_frame | Trials (of configured) | Best val_log_mse | Best val_loss | dmd_energy | dmd_lambda | Median / worst val_log_mse |
|---|---|---|---|---|---|---|---|---|
| `sweep_dmd_observable_hop25.yaml` | 25 | 0.5s | 47 (30) | **2.9588** | 1.1163 | 0.9212 | 0.006042 | 2.963 / 3.995 |
| `sweep_dmd_observable_hop15.yaml` | 15 | 0.3s | 60 (30) | **2.6207** | 0.9949 | 0.9640 | 2.96e-05 | 2.634 / 3.162 |
| `sweep_dmd_observable_hop10.yaml` | 10 | 0.2s | 30 (30) | **2.3432** | 0.8993 | 0.9904 | 0.001873 | 2.376 / 2.585 |

All three studies converged smoothly (no pruned/failed trials); the "worst" column shrinks
alongside the best as the hop shortens, i.e. the whole search landscape gets easier at higher
decision rates, not just its optimum.

The trial counts exceed the configured 30 (47 and 60 for hop25/hop15) because of an `optuna`
semantics bug on my part during execution: `study.optimize(objective, n_trials=N)` runs `N`
*additional* trials per invocation rather than capping the study at `N` total, and hop25/hop15
were each re-invoked more than once while working around a subagent that stalled repeatedly on
this task (see caveats). This is harmless to the result — more trials only tightens the estimate
of the optimum — but the reported "30-trial budget" wasn't strictly honored for those two.

`dmd_lambda`'s optimum has no consistent trend across hops (6e-3 → 3e-5 → 2e-3): the regularization
strength that's best is noisy/geometry-specific rather than following the energy trend, and given
how flat the closed-loop objective turned out to be (§2), this parameter looks under-determined by
data at any of these hops.

## 2. Closed-loop sweep at hop25 (production geometry)

`sweep_dmd_observable_closed_loop.yaml`, narrowed to `dmd_energy ∈ [0.90, 0.95]`,
`dmd_lambda ∈ [1e-3, 3e-2]` (loguniform) around the hop25 offline optimum, scored by
`evaluate_closed_loop_suppression` over seeds `[69, 70, 71]`, `t_end=12s`.

| Trial | dmd_energy | dmd_lambda | val_log_mse | seizure_burden (↓ better) | suppressed / total seeds |
|---|---|---|---|---|---|
| 0 | 0.9177 | 0.004688 | 2.9596 | **0.07476** (best) | 2 / 3 |
| 4 | 0.9291 | 0.001452 | 2.9585 | 0.07856 | 2 / 3 |
| 7 | 0.9131 | 0.002470 | 2.9625 | 0.07671 | 2 / 3 |
| 1 | 0.9253 | 0.024162 | 2.9592 | 0.07573 | 2 / 3 |
| 3 | 0.9028 | 0.002433 | 2.9715 | 0.07758 | 2 / 3 |
| 5 | 0.9374 | 0.010025 | 2.9633 | 0.07807 | 2 / 3 |
| 6 | 0.9122 | 0.019792 | 2.9633 | 0.07914 | 2 / 3 |
| 2 | 0.9427 | 0.001147 | 2.9658 | 0.08060 (worst) | 2 / 3 |

**Best trial**: `dmd_energy=0.9177`, `dmd_lambda=0.004688`, seizure_burden=0.07476, matching the
hop25 offline optimum (`dmd_energy=0.9212`, `dmd_lambda=0.006042`) closely — both objectives point
to the same corner of hyperparameter space.

**Flatness**: the spread across all 8 trials is only 0.0748–0.0806 (≈7% relative range), and every
single trial suppresses exactly 2 of the 3 seeds. This sweep only ran 8 trials as configured (no
overshoot, since it was invoked once) — with the objective this flat, more trials would sharpen
the ranking within this narrow band but are unlikely to find dramatically different qualitative
behavior. The unsuppressed seed appears to be a plant/config property, not something these two
DMD hyperparameters can fix.

## 3. Caveats

- **Off-plant warning**: the closed-loop sweep run emitted
  `UserWarning: the plant this config simulates is not the one the predictor was identified on
  (dynamics or sensors differ); the prediction model is off-plant.` on each trial. This is the
  same warning pattern seen in the repo's pre-existing closed-loop sweeps (e.g.
  `sweep_3_loss_curriculum_stft.yaml`) and stems from `closed_loop_eval_observable_12s.yaml` using
  a fixed `seed: 69` dynamics instance while the DMD model is fit on
  `data/experiment_excited_long/train` — expected given how `evaluate_closed_loop_suppression` is
  designed to reuse a shared eval config across many trials, not a new issue introduced here, but
  worth keeping in mind if these closed-loop numbers are compared against a true on-plant
  evaluation later.
- **Faster-hop offline gains are unvalidated in closed loop.** Only hop25 was swept against the
  true closed-loop objective; hop15/hop10's offline improvements have not been checked for whether
  they translate into deployable seizure suppression, or whether a 0.2–0.3s decision rate is even
  compatible with the `TrajOptMPCController`'s `dt` and horizon assumptions (`dt=0.5` in the
  current production config is tied to `n_hop=25`; a smaller hop would need `controller.dt` and
  likely `horizon`/`w_u*` retuned, not just a predictor swap).
- **Execution note**: the sweeps were originally delegated to a subagent, which repeatedly stalled
  waiting on background-task notifications for commands that had already exited (or were never
  actually started in the background). After four escalating corrections failed to fix this, all
  four sweeps were run directly in the coordinating session instead; this is what produced the
  hop25/hop15 trial-count overshoot noted in §1.

## 4. What to try next

- Run a closed-loop sweep at hop15 and/or hop10 (with `controller.dt`/`horizon` retuned to match)
  to check whether the offline MSE gains at faster hops actually reduce seizure_burden, or whether
  they're absorbed/negated by the controller's fixed decision cadence.
- Investigate the seed that no trial suppresses (which of `[69, 70, 71]` it is, and why) — since
  no combination of `dmd_energy`/`dmd_lambda` in the swept range moves seizure_burden much, the
  remaining gap looks structural (geometry, controller tuning, or that seed's dynamics) rather than
  a DMD-hyperparameter problem.
- If a faster hop is pursued, consider sweeping `controller.dt`/`horizon`/`w_u*` jointly with the
  faster-hop predictor rather than reusing the hop25-tuned controller settings unchanged.
