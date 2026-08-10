# Why the MPC does not suppress, and what to change

**Date:** 2026-08-10
**Context:** closes the last open item on the `roast_3d` closed loop.
Evidence and full result tables: [`tes_field_geometry.md`](tes_field_geometry.md) §2.
Artifacts, commands and traps: [`roast_operations.md`](roast_operations.md).

## Findings

**The nonlinear MPC is tractable.** At `shooting_depth: 20` (single shooting) a 20 s seed costs
~20 min and a 7-seed ensemble ~1 h at 3 workers, ~400–500 MB each. The earlier
">8.4 h, computationally impractical" write-up was measuring `shooting_depth: 1`, i.e. full
multiple shooting, and is retracted. No fallback (`expand`, lower `max_iter`, thread pinning, PCA)
was needed.

**It still does not suppress**, but it fails differently from the linear one (7-seed medians):

| run | suppressed | seizing L | eegMS | mean \|u\| |
| --- | --- | --- | --- | --- |
| no stimulation | 0/7 | 30 | 139.3 | 0 |
| threshold controller | 6/7 | 4 | 6.0 | 1.27 |
| linear MPC, `w_u` = 10 | 0/7 | 32 | 228.5 | 1.32 |
| nonlinear MPC, `w_u` = 10 | 0/7 | 31 | **59.2** | 0.67 |
| nonlinear MPC, `w_u` = 0 | 1/7 | 29 | 184.4 | 1.32 |

The linear MPC drove scalp power *above* the unstimulated baseline (228 vs 139) — §2.2 reads that
as an identification failure, correctly. The nonlinear MPC drives it **57 % below** baseline on
half the command effort, and 31/34 regions still seize. It is minimising its objective and
succeeding. The lone 1/7 at `w_u = 0` is one draw into the bimodal suppressed basin, not a result.

**The objective is anti-correlated with the outcome at its own horizon.** Five seeds, settled 12 s
into the seizure, comparing `u = 0` against the known-good sustained `[+2, 0, -2]` mA:

| what | median good/zero | improves in |
| --- | --- | --- |
| model's 20-step cost (what the MPC minimises) | 1.019 | 1/5 |
| true EEG power over 0.2 s (= the horizon) | 1.032 | 1/5 |
| true EEG power over 2.0 s | 0.968 | 3/5 |

Per-seed rows matter more than the medians. Seed 69: the good command makes EEG power *worse* over
200 ms (4.68 → 4.83) and **20× better** over 2 s (112.5 → 5.4); seed 1026 is the same shape
(5.83 → 7.23, then 46.7 → 6.4). In exactly the states where stimulating is hugely beneficial, the
200 ms cost reports it as slightly harmful, so an MPC minimising that cost is *correct* to refuse.

**The predictor is not the weak link.** The model's ratio (1.019) tracks the plant's true 200 ms
ratio (1.032): it accurately forecasts a quantity that does not matter.

**The window to act closes.** Seeds 1023–1025, already fully seizing at t = 12 s (EEG MS ~210–260),
do not benefit even over 2 s (227 → 231, 227 → 237, 259 → 251). That is §2.1's "trigger early,
then hold" measured directly — the controller must value an action whose payoff is seconds away.

## Next steps

**1. Extend the effective horizon from 200 ms to ~2 s.** Not by raising `horizon` to 200 — the NLP
grows 10× and the predictor is trained to a 20-step rollout. Retrain at `downsample: 1000`
(100 ms effective dt) keeping `horizon: 20`: same NLP size, 2 s of lookahead.

**2. Forecast an envelope, not the waveform.** Raw EEG at 100 ms sampling aliases — seizure rhythms
are 3–10 Hz — so step 1 forces the output representation to change anyway. Predict band power or
sliding PTP. That is also the quantity that should be small, so `sumsqr(y)` stops being a proxy and
becomes the target. Weight it toward the channels loading on `lTCI` rather than uniformly across
62, since suppression here is a propagation block at that region (§1.2).

**3. Add sustained holds to the excitation set.** `hold_ms: [10, 50, 200]` means the longest
constant command in training is 200 ms, while the working policy holds for seconds at 95 % duty.
Regenerate with 1000/2000/4000 ms added. This is a *prerequisite for step 1*, not the root cause:
at a 200 ms horizon the gap does not bind; at 2 s the model would extrapolate on every solve.
Changing the data alone would not have moved the closed loop.

**Skip:** solver tuning, `w_du` / `w_u_l1`, bigger networks, PCA. The `w_u` sweep already showed the
controller side is not a lever, and the probe above shows why.

**Success criterion:** the MPC rediscovers "trigger early, then hold" on its own, and beats the
threshold controller on **duty cycle** (95 % is poor) rather than on suppression count — 6/7 is
probably near the ceiling given the bimodality.
