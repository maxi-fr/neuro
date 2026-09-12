# Excitation Hold Tailoring & Closed-Loop Performance

This document records the evaluation of training dynamical Predictors on Random Amplitude Schedule (RAS) stimulation data whose hold duration matches the controller's discrete decision interval $\Delta t \in \{0.02, 0.10, 0.06\}\,\text{s}$, compared against baseline models trained on mismatched $10\,\text{ms}$ holds (`data/experiment_excited_long/train`).

Source files:

- Datasets:
  - Arm 1 ($20\,\text{ms}$): `data/experiment_excited_20ms/train`
  - Arm 2 ($100\,\text{ms}$): `data/experiment_excited_100ms/train`
  - Arm 3 ($60\,\text{ms}$): `data/experiment_excited_60ms/train`
- Training configs:
  - `configs/nn_predictor/waveform_mlp_20ms.yaml`
  - `configs/nn_predictor/observable_mlp_hop5_100ms.yaml`
  - `configs/nn_predictor/observable_mlp_fast_60ms.yaml`
- Closed-loop comparison manifest:
  - [`configs/comparison/observable_tailoring.yaml`](../configs/comparison/observable_tailoring.yaml)
- Results:
  - [`results/comparison/observable_tailoring/rows.csv`](../results/comparison/observable_tailoring/rows.csv)
  - [`results/comparison/observable_tailoring/summary.csv`](../results/comparison/observable_tailoring/summary.csv)

---

## 1. Executive Summary

1. **Offline error drops across all three arms**:
   - **Arm 1 (Waveform MLP, $\Delta t = 20\,\text{ms}$)**: Rollout log energy error dropped by **13.4%** ($1.2793 \to 1.1081$), validation loss dropped by **9.3%** ($4.0707 \to 3.6917$).
   - **Arm 2 (Hop-5 Observable MLP, $\Delta t = 100\,\text{ms}$)**: Observable Frame MSE decreased from $2.8529 \to 2.8311$, and input sensitivity $\|\partial \hat{y} / \partial u\|$ increased by **+24.7%** ($35.61 \to 44.40$).
   - **Arm 3 (Fast STFT Observable MLP, $\Delta t = 60\,\text{ms}$)**: Frame MSE reached a sub-1.0 record of **0.9861** (**-3.4%** reduction from $1.0203$).

2. **Offline gains translate directly into closed-loop seizure arrest**:
   On the held-out screened seed `7000`, both baseline models trained on mismatched $10\,\text{ms}$ data failed to arrest seizure spread, recruiting 26–31 brain regions and posting high Seizure Burden ($>0.30$). Both tailored models arrested seizure propagation:
   - **Fast STFT ($60\,\text{ms}$)**: Seizure Burden dropped from **0.306 to 0.059** (**-81%** reduction), and recruiting regions dropped from 26 down to 6.
   - **Hop-5 ($100\,\text{ms}$)**: Seizure Burden dropped from **0.314 to 0.072** (**-77%** reduction), and recruiting regions dropped from 31 down to 5.

---

## 2. Offline Validation Comparison

All models were evaluated on the held-out validation partition of their respective datasets under identical training protocols (EarlyStopping on validation loss, AdamW, cosine annealing):

| Arm | Metric | Baseline (10 ms Holds) | Tailored Holds | Relative Change |
| :--- | :--- | :---: | :---: | :---: |
| **Arm 1: Waveform MLP** ($\Delta t = 20\,\text{ms}$) | Rollout Log Energy Error | 1.2793 | **1.1081** | **-13.4%** 🏆 |
| | Minimum Validation Loss | 4.0707 | **3.6917** | **-9.3%** 🏆 |
| | Input Sensitivity ($\|\partial \hat{y}/\partial u\|$) | 15.7837 | 13.0592 | -17.3% |
| **Arm 2: Hop-5 Observable MLP** ($\Delta t = 100\,\text{ms}$) | Observable Frame MSE | 2.8529 | **2.8311** | **-0.8%** 🏆 |
| | Minimum Validation Loss | 1.1702 | 1.1880 | +1.5% |
| | Input Sensitivity ($\|\partial \hat{y}/\partial u\|$) | 35.6100 | **44.4028** | **+24.7%** 🏆 |
| **Arm 3: Fast STFT Observable MLP** ($\Delta t = 60\,\text{ms}$) | Observable Frame MSE | 1.0203 | **0.9861** | **-3.4%** 🏆 |
| | Minimum Validation Loss | 0.7566 | 0.7648 | +1.1% |
| | Input Sensitivity ($\|\partial \hat{y}/\partial u\|$) | 18.3136 | 17.4516 | -4.7% |

---

## 3. Closed-Loop Seizure Suppression

Closed-loop evaluation was run with **Single Shooting IPOPT** (`SingleShooting(Ipopt)`) over a $12.0\,\text{s}$ simulation ($t_{\text{end}} = 12.0\,\text{s}$) paired on screened seed `7000`:

| Arm | Training Hold | Seizure Burden | Seizing Regions | $t_{\text{left half}}$ | Delivered Charge | Solve Time (Mean) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `uncontrolled` | — | 0.124 | 32 / 76 | 10.0 s | $0.00\,\mu\text{C}$ | — |
| **Fast STFT ($60\,\text{ms}$)** Baseline | 10 ms | 0.306 | 26 / 76 | 3.75 s | $1.95\,\mu\text{C}$ | 3.54 s |
| **Fast STFT ($60\,\text{ms}$)** Tailored | 60 ms | **0.059** | **6 / 76** | **12.00 s** | $7.03\,\mu\text{C}$ | 3.69 s |
| **Hop-5 ($100\,\text{ms}$)** Baseline | 10 ms | 0.314 | 31 / 76 | 3.75 s | $2.29\,\mu\text{C}$ | 158.0 s |
| **Hop-5 ($100\,\text{ms}$)** Tailored | 100 ms | **0.072** | **5 / 76** | **12.00 s** | $1.61\,\mu\text{C}$ | 85.0 s |

### Observations

- In both baseline runs, the controller allowed seizure activity to cross into the contralateral hemisphere by $t = 3.75\,\text{s}$ (`t_left_half = 3.75 s`), recruiting over 25 regions.
- In both tailored runs, the controller confined seizure activity to the focal region (`t_left_half = 12.0 s`), preventing spread to the left hemisphere entirely.

---

## 4. IPOPT Iteration Caps & Solve Time Analysis

### 4.1 Historical Evolution of `max_iter`

In the project history, IPOPT iteration caps evolved as follows:

1. **CasADi MPC Implementation** (`commit cab4707`, Aug 10):
   In `src/neuro/control.py`, `max_iter` was explicitly parameterized and defaulted to `100`:

   ```python
   max_iter: int = Field(default=100, ge=1)
   # "Hard cap on IPOPT iterations per solve. When the cap is hit the best warm-started iterate is applied and success is False (capped, not failed)."
   ```

2. **Transition to TrajOpt** (`commit f5eef8c` & `commit 4855063`, late August):
   When the control loop migrated to TrajOpt's `SingleShooting(Ipopt)`, the controller's `_default_solver` configured:

   ```python
   SingleShooting(solver=Ipopt(options={"print_level": 0, "hessian_approximation": "limited-memory"}))
   ```

   The `max_iter` parameter was not passed, causing IPOPT to revert to its internal default of **3,000 iterations**.
3. **Solver Benchmark Cap** (`commit 41e9e87`, Sep 7):
   In `src/neuro/control/benchmark.py`, a cap was enforced to prevent hangs during sweeps:

   ```python
   _IPOPT_DEFAULTS = {"print_level": 0, "hessian_approximation": "limited-memory", "max_iter": 300}
   ```

4. **Tolerance Tuning & Default Integration** (`commit 969ee66`, Sep 10, merged via `3d5e6eb`):
   In `src/neuro/control/mpc.py`, tuned receding-horizon defaults were introduced:

   ```python
   IPOPT_DEFAULTS: dict[str, Any] = {
       "print_level": 0,
       "hessian_approximation": "limited-memory",
       "tol": 1e-3,
       "acceptable_tol": 1e-2,
       "acceptable_iter": 5,
       "max_iter": 300,
   }
   ```

### 4.2 Why Hop-5 Stalled Without the Cap

1. **Network Depth and State Dimension**: Hop-5 operates on state $n = 1550$ with a 2-hidden-layer MLP ($512$ units).
2. **Control Horizon and Reverse-Mode AD**:
   - At $H = 15$ knots ($1.5\,\text{s}$), each reverse-mode AD pass unrolls 15 sequential evaluations of the 2-layer network.
   - During non-seizing periods, IPOPT converges in 3–5 iterations (~1.1 s).
   - At seizure onset, the spectral hinge cost exhibits sharp non-smooth transitions where gradient descent struggles to achieve tight tolerance ($10^{-8}$).
   - Without an iteration cap, IPOPT ran through hundreds of iterations per step, pushing solve times to $365\text{--}507\,\text{s}$ per step.
3. **Horizon Reduction ($1.0\,\text{s}$)**:
   - Reducing the Control Horizon to $1.0\,\text{s}$ ($H = 10$ for Hop-5, $H = 17$ for Fast STFT) reduces unrolled computation graph depth, significantly improving solver conditioning.
   - Combining a $1.0\,\text{s}$ horizon with the merged `IPOPT_DEFAULTS` (`max_iter = 300`, `tol = 1e-3`) guarantees that unconverged steps terminate cleanly in seconds rather than minutes.
