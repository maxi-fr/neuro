---
title: "Deep Learning Model Predictive Control for Deep Brain Stimulation in Parkinson's Disease"
type: summary
tags: [model-predictive-control, deep-brain-stimulation, parkinsons-disease, neural-networks, convex-optimization]
sources: [Literature/Steffen_Cannon_2025.pdf]
created: 2026-05-14
updated: 2026-05-14
---

# Deep Learning Model Predictive Control for Deep Brain Stimulation in Parkinson's Disease

**Source:** `Literature/Steffen_Cannon_2025.pdf`
**Ingested:** 2026-05-14

## Summary

This paper presents a nonlinear data-driven Model Predictive Control (MPC) algorithm for closed-loop deep brain stimulation (CLDBS) in Parkinson's disease (PD). The authors address the limitations of existing CLDBS algorithms, which are often limited to simple proportional or on/off control, by employing a multi-step predictor based on **Difference of Convex (DC) functions**. The model uses **Input-Convex Neural Networks (ICNN)** to represent the nonlinear dynamics of beta-band oscillations (13-30 Hz) in response to stimulation. The resulting control law is formulated as a robust tube MPC strategy that accounts for linearization errors and modeling uncertainty.

## Key Takeaways

- **Nonlinear Multi-step Predictor:** The model uses a separate neural network for each step of the prediction horizon, represented as a difference of two ICNNs. This simplifies robust tube construction by avoiding recursive error propagation [[Literature/Steffen_Cannon_2025.pdf | Steffen & Cannon 2025 (III. CONTROL LAW)]].
- **Superior Performance:** In simulations using patient LFP data, the DCNN TMPC achieved >20% reduction in tracking error and control activity compared to linear MPC, PI, and on-off controllers [[Literature/Steffen_Cannon_2025.pdf | Steffen & Cannon 2025 (IV. NUMERICAL SIMULATIONS)]].
- **Generalizability:** The pre-trained model (trained on other patients) outperformed linear MPC even when the latter was tuned to the specific test patient, demonstrating significant robustness and generalization capability [[Literature/Steffen_Cannon_2025.pdf | Steffen & Cannon 2025 (IV-C. Comparison with Alternative Control Strategies)]].
- **Mathematical Formulation:** The system is modeled as $\dot{y}(t) = f(y(t), u(t), t)$ where $y(t)$ is the beta-band envelope. The cost function minimizes pathological activity $[y(t) - y_0]_{\geq 0}$ and stimulation energy $Ru(t)^2$ [[Literature/Steffen_Cannon_2025.pdf | Steffen & Cannon 2025 (II. PROBLEM FORMULATION)]].

## Related Concepts

- [[concepts/closed-loop-brain-stimulation.md|Closed-loop Brain Stimulation]] — Application to Parkinson's Disease via beta-band suppression.
- [[concepts/model-predictive-control.md|Model Predictive Control]] — Nonlinear implementation using DCNN and Tube MPC.
- [[concepts/difference-of-convex-functions.md|Difference of Convex Functions]] — DC functions and ICNN architecture used to build the nonlinear predictor.
- [[concepts/parkinsons-disease.md|Parkinson's Disease]] — Clinical target for the proposed neurostimulation strategy.
- [[concepts/beta-band-oscillations.md|Beta-band Oscillations]] — The primary biomarker used for feedback in PD CLDBS.

## Raw Notes

- Stimulation response model (synthetic): $y(t) = y_{\beta}(t) \cdot e^{-\eta(u(t))}$.
- Multi-step predictor avoids "recursive feasibility" issues but provides a simpler construction for robust tubes.
- Uses `cvxpy` and `MOSEK` for online optimization.
