---
title: Difference of Convex Functions and ICNN
type: concept
tags: [mathematics, optimization, modeling, machine-learning, neural-networks]
sources: [steffen-cannon-2025.md]
created: 2026-05-14
updated: 2026-05-16
---

# Difference of Convex Functions and ICNN

A Difference of Convex (DC) function is a function that can be expressed as the difference of two convex functions: $f(x) = g(x) - h(x)$, where both $g$ and $h$ are convex. In data-driven control, DC functions are implemented using **Input-Convex Neural Networks (ICNN)**, yielding a **Difference of Convex Neural Network (DCNN)** architecture.

## DC Functions

- **Universal Approximation:** Any continuous twice-differentiable function can be represented with arbitrary accuracy as a DC function [[raw/Steffen_Cannon_2025/Steffen_Cannon_2025.md#II. PROBLEM FORMULATION | Steffen & Cannon 2025]].
- **Sequential Convex Programming:** Optimization proceeds by linearizing the concave part ($-h(x)$), reducing the problem to a sequence of convex subproblems [[raw/Steffen_Cannon_2025/Steffen_Cannon_2025.md#I. INTRODUCTION | Steffen & Cannon 2025]].
- **Tight Linearization Bounds:** The DC structure provides tight error bounds on linearization, critical for robust strategies like Tube MPC [[raw/Steffen_Cannon_2025/Steffen_Cannon_2025.md#I. INTRODUCTION | Steffen & Cannon 2025]].

## ICNN Architecture

Input-Convex Neural Networks (ICNN) are the building block for each convex component $g$ or $h$:

- **Structural Constraints:** Convexity is enforced by non-negative inter-hidden-layer weights and convex, non-decreasing activations (e.g., ReLU), plus skip connections from the input to preserve expressiveness [[raw/Steffen_Cannon_2025/Steffen_Cannon_2025.md#III-A. Difference of Convex Functions Neural Network Model | Steffen & Cannon 2025]].
- **DCNN:** Two ICNNs combined as $f(x) = f_1(x) - f_2(x)$ can represent complex nonlinearities while keeping inference tractable via convex programming [[raw/Steffen_Cannon_2025/Steffen_Cannon_2025.md#III-A. Difference of Convex Functions Neural Network Model | Steffen & Cannon 2025]].

## Related Concepts

- [[concepts/model-predictive-control.md|Model Predictive Control]] — Optimization framework using DCNN models via sequential convex programming.
- [[concepts/closed-loop-brain-stimulation.md|Closed-loop Brain Stimulation]] — Application domain where DCNN models the nonlinear beta-band response to DBS.

## Sources

- [[sources/steffen-cannon-2025.md|Deep Learning Model Predictive Control for Deep Brain Stimulation in Parkinson's Disease]] — Employs DCNN for nonlinear Tube MPC in closed-loop DBS.
