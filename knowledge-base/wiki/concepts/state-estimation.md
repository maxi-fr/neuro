---
title: State Estimation
type: concept
tags: [control-theory, observers, mhe, neurostimulation]
sources: [thesis-proposal.md]
created: 2026-05-18
updated: 2026-05-18
---

# State Estimation

State estimation is the process of inferring the internal states of a dynamical system from noisy or incomplete measurements. In neurostimulation, it is essential because many critical variables (e.g., individual ion channel states or internal population dynamics) cannot be directly measured in real-world settings.

## Key Points

- **Necessity for Feedback:** Since the states of neural dynamics models (like [[concepts/hodgkin-huxley-model.md|Hodgkin-Huxley]] or [[concepts/neural-mass-model.md|Neural Mass Models]]) are typically not directly measurable, a state observer is required for feedback control [[Notes/Thesis.md#Challenges | Thesis]].
- **Moving Horizon Estimation (MHE):** A common approach mentioned for the MPC framework, which solves an optimization problem over a sliding window of past measurements to estimate current states [[Notes/Thesis.md#Prediction model | Thesis]].
- **Parameter Estimation:** Beyond states, estimating time-varying parameters (e.g., synaptic weights, disease progression coefficients) is necessary for long-term adaptation and plasticity-aware control [[Notes/Thesis.md#Prediction model | Thesis]].
- **Observability:** A key challenge in designing observers for large-scale neural population models, where it may be unclear if the measured signals provide enough information to reconstruct the full state [[Notes/Thesis.md#Prediction model | Thesis]].

## Related Concepts

- [[concepts/model-predictive-control.md|Model Predictive Control]] — Relies on estimated states for its prediction horizon.
- [[concepts/neural-mass-model.md|Neural Mass Model]] — One of the models requiring state estimation for real-time control.

## Sources

- [[sources/thesis-proposal.md|Master's Thesis Proposal: MPC for Neurostimulation]] — Highlights state and parameter estimation as critical components of the neurostimulation framework.
