# SQP Solver & IPOPT Fallback Benchmark for Nonlinear MPC

## 1. Executive Summary

This document evaluates the performance, stability, and feasibility of using Sequential Quadratic Programming (**SQP**) with an **IPOPT fallback** mechanism for nonlinear Receding-Horizon Model Predictive Control (MPC) in closed-loop neurostimulation.

The investigation tested the core hypothesis:
> **SQP** provides fast solves close to local minima when warm-started, but can suffer from non-convergence or QP infeasibility during abrupt state transitions; **IPOPT** serves as a robust, globally convergent fallback to guarantee controller reliability.

---

## 2. Experimental Setup

### A. Predictor Model & NLP Problem Size

* **Predictor Artifact**: [`artifacts/nonlinear_mse02_eeg_ms/model`](file:///C:/Users/frank/closed-loop-neurostimulation/artifacts/nonlinear_mse02_eeg_ms/model) (Unrolled 48-node Multi-Layer Perceptron).
* **Sampling Rate & Native Time Step**: $f_s = 50\,\text{Hz}$ ($\Delta t = 0.02\,\text{s} = 20\,\text{ms}$).
* **Prediction Horizon**: $H = 50$ steps ($1.0\,\text{s}$ preview).
* **Control Dimension**: $n_{\text{controls}} = 3$ electrodes with Kirchhoff sum-to-zero constraint $\sum_{i=1}^3 u_k^{(i)} = 0$.
* **State Buffer Dimension**: $n_{\text{state}} = 960$ dimensions (autoregressive state representation).

### B. Formulations Evaluated

1. **Single Shooting ($D = 50$)**:
   * States condensed out via direct rollout in the cost graph.
   * **Variables**: 150 optimization variables ($50 \times 3$ controls).
   * **Constraints**: 50 linear equality constraints ($\sum u = 0$) + box constraints $-u_{\text{max}} \le u \le u_{\text{max}}$.
2. **Partial Multiple Shooting ($D = 25$)**:
   * 2 shooting segments with 1 intermediate shooting node $\phi_1$.
   * **Variables**: 1,110 optimization variables (150 controls + 960 state variables).
   * **Constraints**: 960 nonlinear state defect equality constraints ($\phi_1 - f^{25}(x_0, u) = 0$) + 50 sum-to-zero equalities.

### C. Solvers & Configurations Evaluated

* **IPOPT Baseline**: `ipopt` with `limited-memory` Hessian approximation (`max_iter=100`).
* **SQP (`sqpmethod`)**:
  * **QP Subsolvers**:
    * `qpoases`: Dense active-set QP solver.
    * `qrqp`: CasADi built-in dense QR active-set QP solver.
    * `osqp`: ADMM first-order QP solver.
  * **Hessian Approximations**:
    * `limited-memory`: L-BFGS quasi-Newton update (memory $M = 10$).
    * `exact`: Second-order algorithmic differentiation (AD) of the unrolled computational graph.
  * **Iteration Caps**: $K_{\text{max}} \in \{10, 15\}$.
* **Hybrid Fallback Controller**:
  * Attempts SQP first. If status $\neq \text{"Solve\_Succeeded"}$ or an exception occurs, instantly invokes IPOPT warm-started from the SQP iterate.

---

## 3. Offline Open-Loop Benchmark Results

Evaluated across primed initial state snapshots $x_0$ sampled from held-out test trajectories in [`data/experiment_excited_roast/test/`](file:///C:/Users/frank/closed-loop-neurostimulation/data/experiment_excited_roast/test).

### A. Single Shooting ($D = 50$)

| Solver | Hessian Approx | QP Subsolver | Max Iters | Median Solve Time | P90 Solve Time | Standalone Convergence | Fallback Trigger Rate | Total Success Rate | vs IPOPT Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **IPOPT** | `limited-memory` | *N/A* | 100 | **1,702.3 ms** | **2,113.0 ms** | 100% | *N/A* | 100% | *Baseline* |
| **SQP** | `limited-memory` (L-BFGS) | `qpoases` | 15 | **989.4 ms** | **1,089.4 ms** | 100% | 0% | 100% | **42% faster** |
| **SQP** | `limited-memory` (L-BFGS) | `osqp` | 15 | **1,091.0 ms** | **1,175.9 ms** | 100% | 0% | 100% | **36% faster** |
| **SQP** | `limited-memory` (L-BFGS) | `qrqp` | 15 | **1,432.1 ms** | **1,627.2 ms** | 100% | 0% | 100% | **16% faster** |
| **SQP** | `limited-memory` (L-BFGS) | `osqp` | 10 | **704.4 ms** | **711.2 ms** | 30% | 70% | 100% (via fallback) | *Fallback active* |
| **SQP** | `exact` (Full AD) | `qrqp` | 15 | **12,500.1 ms** | **15,609.8 ms** | 100% | 0% | 100% | **7.3x slower** |
| **SQP** | `exact` (Full AD) | `qpoases` | 15 | **12,927.6 ms** | **14,282.5 ms** | 100% | 0% | 100% | **7.6x slower** |
| **SQP** | `exact` (Full AD) | `osqp` | 15 | **18,329.0 ms** | **26,448.9 ms** | 100% | 0% | 100% | **10.8x slower** |

### B. Multiple Shooting ($D = 25$)

| Solver | Hessian Approx | QP Subsolver | Max Iters | Median Solve Time | P90 Solve Time | Convergence | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **IPOPT** | `limited-memory` | *N/A* | 100 | **5,385.4 ms** | **5,820.9 ms** | 100% | Slower due to 960 defect constraints |
| **SQP** | `limited-memory` | `osqp` | 15 | **10,612.0 ms** | **10,828.0 ms** | 10% | 90% fallback trigger rate |
| **SQP** | `limited-memory` | `qpoases` | 15 | **60,380.8 ms** | **87,590.7 ms** | 10% | Active-set struggle with dense 960 equalities |

---

## 4. Online Closed-Loop Simulation Results

Simulated in closed loop with the Jansen-Rit connectome model on [`configs/simulation/mse02_eeg_ms_mpc.yaml`](file:///C:/Users/frank/closed-loop-neurostimulation/configs/simulation/mse02_eeg_ms_mpc.yaml) ($t_{\text{end}} = 3.0\,\text{s}$, 137 controller steps):

| Controller Mode | Median Solve Time | P90 Solve Time | Max Solve Time | Fallback Rate | Total Success Rate | Seizure Burden | Delivered Charge |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Pure IPOPT Baseline** | 1,073.0 ms | 1,257.5 ms | 1,385.2 ms | 0% | 100% | 100.0% | 10.370 |
| **Hybrid SQP (`qpoases`) + IPOPT** | **861.8 ms** | **919.3 ms** | **983.7 ms** | 0% | 100% | 100.0% | 10.370 |
| **Pure SQP (`qpoases`) Standalone** | **861.2 ms** | **917.9 ms** | **983.7 ms** | 0% | 100% | 100.0% | 10.370 |

* **Latency Reduction**: SQP provides a **~20% reduction in per-step solve time** in warm-started closed-loop control.
* **Control Fidelity**: The control trajectory $u(t)$ and cost between SQP and IPOPT are virtually identical (mean $u_0$ difference $< 10^{-5}$).
* **Real-time Feasibility**: While SQP is significantly faster, solving full $H=50$ nonlinear MPC at ~860 ms remains above the $20\,\text{ms}$ hard deadline ($f_s = 50\,\text{Hz}$), highlighting the need for Real-Time Iteration (RTI) or shorter horizons if strict real-time execution is required.

---

## 5. Technical Insights & Discussion

### A. Why Exact Hessian is Impractical for Neural Predictors

* Evaluating an **exact Lagrangian Hessian** $\nabla^2 \mathcal{L}$ through 50 unrolled steps of a recurrent/MLP network requires second-order automatic differentiation (forward-over-reverse AD).
* Building and evaluating the $150 \times 150$ exact Hessian at every iteration adds 10–18 seconds of overhead per step.
* **L-BFGS (`limited-memory`)** only requires first-order gradients (standard backpropagation) and builds low-rank curvature updates, making it ~10x faster.

### B. Comparison of QP Subsolvers

1. **`qpoases`**: Best dense QP solver for single shooting. Active-set strategy enables fast warm-starts on box-constrained control vectors ($150 \times 150$).
2. **`osqp`**: Robust first-order ADMM QP solver. Highly effective for sparse structures, but slightly slower on dense condensations due to ADMM tolerance checks on equality constraints.
3. **`qrqp`**: CasADi's built-in QR active-set solver. Works reliably without third-party dependencies, but is ~30% slower than `qpoases`.

### C. Single Shooting vs. Multiple Shooting

* In this architecture, the state vector is 960-dimensional. Multiple shooting ($D < 50$) introduces $960 \times \text{segments}$ nonlinear equality constraints, causing both QP and interior-point solvers to slow down dramatically.
* **Single shooting ($D = 50$)** is the superior formulation for this history-buffered neural network structure.

---

## 6. Reproducibility & Benchmark Scripts

The experiment scripts are available in `scratch/`:

1. **Open-Loop Benchmark Script**: [`scratch/benchmark_sqp_ipopt.py`](file:///C:/Users/frank/closed-loop-neurostimulation/scratch/benchmark_sqp_ipopt.py)

   ```powershell
   uv run python scratch/benchmark_sqp_ipopt.py --n-solves 20 --shooting-depths 50 --sqp-qpsols qpoases qrqp osqp --sqp-hessians limited-memory
   ```

2. **Closed-Loop Simulation Script**: [`scratch/sim_closed_loop_sqp.py`](file:///C:/Users/frank/closed-loop-neurostimulation/scratch/sim_closed_loop_sqp.py)

   ```powershell
   uv run python scratch/sim_closed_loop_sqp.py --t-end 5.0 --modes pure_ipopt hybrid pure_sqp --qpsol qpoases
   ```
