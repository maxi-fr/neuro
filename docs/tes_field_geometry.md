# tES Field Geometry & Closed-Loop Control (`roast_3d`)

**Status:** `roast_3d` is the canonical stimulation model across all configs (`data/roast_leadfield_3d.npz`). Under `roast_3d`, mesial focus polarization is physically unreachable; stimulation functions as a network propagation block at `lTCI`. The predictor-free threshold controller achieves 6/7 seed suppression (95% duty cycle), while both linear and nonlinear MPC fail because 62-channel EEG power variance is not a proxy for single-region propagation control.

---

## 1. Field Physics & Plant Dynamics (`roast_3d`)

### 1.1 Unreachability of the Mesial Focus

The FEM leadfield (`data/roast_leadfield_3d.npz`, referenced to Ex8 return) gives scalp electrode TP9:

- **1.36 V/m per mA** at the superficial temporal hub (`lTCI`)
- **0.04–0.17 V/m per mA** at the mesial focus (`lHC`, `lPHC`, `lAMYG`)

At a polarization length of 0.35 mm, maximum stimulation (±4 mA across reachable 2-DOF montages) delivers at most **−0.087 mV** somatic drive to `lHC` — far below the ~0.5 mV threshold required for direct focus silencing. Intracranial human measurements corroborate that scalp tES fields decay rapidly with depth (<0.1 V/m in mesial structures). `roast_3d` evaluates electric field normal components ($E \cdot n$), which are gauge-invariant.

### 1.2 Network Propagation Block

While direct focus silencing is impossible, TP9 drives `lTCI` at −0.47 mV/mA. Because `lTCI` acts as the primary propagation hub for seizure spread, cathodal stimulation at TP9 silences `lTCI`, locking seizure activity within the focus while protecting the surrounding cortical network. Clinical tDCS benefits in temporal lobe epilepsy are similarly attributed to network gatekeeper effects rather than deep mesial polarization.

### 1.3 Seed Bimodality & Ensemble Evaluation

The plant is bimodal across seeds: outcomes split into either ~4–5 seizing regions (suppressed) or ~27–35 seizing regions (unsuppressed). Open-loop DC stimulation shifts the probability of landing in the suppressed basin:

| Current | Suppressed Seeds | Per-Seed Seizing Left | Median EEG MS |
| :--- | :--- | :--- | :--- |
| 0.00 mA | 0/7 | 27, 27, 30, 30, 31, 31, 34 | 139.3 |
| 0.10 mA | 2/7 | 4, 4, 28, 29, 32, 32, 32 | 131.9 |
| 0.25 mA | 3/7 | 4, 5, 5, 28, 29, 31, 35 | 19.8 |
| 0.50 mA | 2/7 | 4, 5, 28, 30, 30, 31, 33 | 69.7 |
| 1.00 mA | 4/7 | 4, 4, 5, 5, 29, 29, 31 | 5.8 |
| 2.00 mA | **5/7** | 4, 4, 4, 4, 4, 28, 29 | 6.0 |
| 3.00 mA | 4/7 | 4, 4, 4, 5, 27, 29, 30 | 7.1 |

*Note:* Open-loop DC fails to suppress 2/7 seeds even at 3.0 mA, highlighting the necessity of closed-loop feedback control. All evaluations are reported across 7-seed ensembles (seeds 69, 1023–1028).

---

## 2. Closed-Loop Control Benchmarks

### 2.1 Amplitude-Threshold Controller (Working Baseline)

The predictor-free `AmplitudeThresholdController` monitors feedback channel TP9 (channel 27). Because the propagation block relies on keeping `lTCI`'s gate shut, triggering early is critical:

- A 20 mV trigger threshold is too late (suppresses 1/7 seeds).
- Lowering the trigger to **10 mV** with 2.0 mA bursts (`configs/simulation/threshold_control.yaml`) yields **6/7 seeds suppressed** (median 4 seizing regions, 95% duty cycle).

### 2.2 Linear MPC Failure (Under-Identified Control Map)

Linear MPC fails (0/7 seeds suppressed across all penalty weights $w_u$). Under `roast_3d`, control input explains only **$1.4 \times 10^{-4}$** of single-step EEG variance and **$5.0 \times 10^{-3}$** of 20-step variance. The control signature on scalp EEG is too weak relative to autoregressive dynamics, leaving the linear input map $B$ under-identified.

### 2.3 Nonlinear MPC Failure (Objective Mismatch)

The nonlinear predictor (`hidden_size: 128`, depth 1 softplus) fits the multi-step rollout cleanly (validation loss 0.1796). Under MPC ($w_u = 10$), it successfully reduces 62-channel scalp EEG power to **59.2** (57% below the unstimulated 139.3 baseline) using half the control effort (0.67 mA).

However, **regional seizure count remains at 31/34 (0/7 seeds suppressed)**. Minimizing global 62-channel scalp EEG power is not a proxy for regional propagation control at `lTCI`.

---

## 3. Summary & Takeaways for Closed-Loop Control

1. **Always evaluate on 7-seed ensembles:** Single-seed evaluations are noise due to plant bimodality.
2. **Target state-space gatekeeping, not global scalp power:** Suppression is a single-region propagation block at `lTCI`. Controllers minimizing total scalp EEG variance achieve lower cost without stopping the seizure.
3. **Check control variance share before training:** Low input authority on scalp EEG ($<10^{-3}$) predicts linear MPC identification failure.
4. **Trigger early and hold:** Intermittent bursts after seizure spread fail; effective control requires early intervention to keep `lTCI` closed.
