---
title: Beta-band Oscillations
type: concept
tags: [neuroscience, biomarker, parkinsons-disease]
sources: [steffen-cannon-2025.md]
created: 2026-05-14
updated: 2026-05-14
---

# Beta-band Oscillations

Beta-band oscillations are neural rhythms in the frequency range of approximately 13-30 Hz. They are a prominent feature of population-level neural activity (Local Field Potentials, LFP) in the basal ganglia and motor cortex.

## Key Points

- **Parkinson's Disease Biomarker:** In Parkinson's Disease (PD), the envelope of beta-band oscillations in the subthalamic nucleus (STN) is a key biomarker of the disease state. Pathological motor symptoms are associated with prolonged bursts of high-amplitude beta activity [[Literature/Steffen_Cannon_2025.pdf | Steffen & Cannon 2025 (I. INTRODUCTION)]].
- **Feedback Signal:** In [[concepts/closed-loop-brain-stimulation.md|Closed-loop Brain Stimulation]], the beta-band envelope is used as a feedback signal. The controller modulates stimulation to suppress these oscillations when they exceed a certain threshold $\beta_0$ [[Literature/Steffen_Cannon_2025.pdf | Steffen & Cannon 2025 (II. PROBLEM FORMULATION)]].
- **Nonlinear Dynamics:** The response of beta-band oscillations to stimulation is inherently nonlinear and time-varying, necessitating advanced control strategies like [[concepts/model-predictive-control.md|Model Predictive Control]] [[Literature/Steffen_Cannon_2025.pdf | Steffen & Cannon 2025 (I. INTRODUCTION)]].

## Related Concepts

- [[concepts/parkinsons-disease.md|Parkinson's Disease]] — Clinical context for beta-band pathology.
- [[concepts/closed-loop-brain-stimulation.md|Closed-loop Brain Stimulation]] — Control framework using beta-band feedback.

## Summaries

- [[summaries/steffen-cannon-2025.md|Deep Learning Model Predictive Control for Deep Brain Stimulation in Parkinson's Disease]] — Focuses on suppressing beta-band activity using nonlinear MPC.
