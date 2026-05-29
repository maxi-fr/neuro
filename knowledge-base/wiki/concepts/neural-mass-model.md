---
title: Neural Mass Model
type: concept
tags: [modeling, neuroscience, eeg]
sources: [chang-et-al-2020.md, salfenmose-obermayer-2022.md, thesis-proposal.md]
created: 2026-05-14
updated: 2026-05-18
---

# Neural Mass Model

A Neural Mass Model (NMM) is a macroscopic neurophysiological model that simulates the collective activity of large populations of neurons. It is widely used to analyze the mechanisms of brain disorders like epilepsy and to simulate EEG signals.

## Key Points

- **Macroscopic Scale:** Instead of modeling individual neurons, NMMs represent the average behavior of neuronal populations (e.g., pyramidal cells, excitatory interneurons, inhibitory interneurons).
- **Reduced Complexity:** NMMs (and the closely related **Wilson-Cowan model**) provide a computationally tractable alternative to complex population models built from diffusively coupled single-cell models. This makes them suitable for real-time applications like [[concepts/model-predictive-control.md|Model Predictive Control]] [[Notes/Thesis.md#Prediction model | Thesis]].
- **EEG Simulation:** NMMs can produce normal EEG waves or epileptiform discharges by adjusting parameters governing excitatory and inhibitory interactions.
- **Biophysically Realistic Variants:** Newer models, such as the **EI EIF model**, are derived from mean-field approximations of exponential integrate-and-fire (EIF) neurons, ensuring all parameters and variables are biophysically grounded. These models are often described by sets of first-order ordinary differential equations (ODEs) for each population $i$:
    $$ \tau_i \dot{r}_i = -r_i + \Phi_i(I_{ext,i} + \sum_j W_{ij} r_j) $$
    where $r_i$ is the population firing rate, $\tau_i$ is the time constant, and $\Phi_i$ is the nonlinear transfer function [[Literature/Salfenmose_Obermayer_2022.pdf | Salfenmose & Obermayer 2022 (2.1.)]].
- **Dynamics:** These models can exhibit complex behaviors including bistability (coexistence of "up" and "down" states), oscillations, and limit cycles. Specific models are tailored to describe different pathological behaviors such as Parkinson's disease or epilepsy [[Notes/Thesis.md#Prediction model | Thesis]].

## Related Concepts

- [[concepts/seizure-suppression.md|Seizure Suppression]] — NMMs are often used as "virtual brains" to test suppression strategies.
- [[concepts/model-predictive-control.md|Model Predictive Control]] — NMMs serve as the prediction models for real-time control.
- [[concepts/optimal-nonlinear-control.md|Optimal Nonlinear Control]] — Used to find efficient ways to steer NMMs between different activity states.

## Sources

- [[sources/chang-et-al-2020.md|Model Predictive Control for Seizure Suppression Based on Nonlinear Auto-Regressive Moving-Average Volterra Model]] — Uses an NMM as a black-box model.
- [[sources/salfenmose-obermayer-2022.md|Nonlinear optimal control of a mean-field model of neural population dynamics]] — Applies optimal control to a biophysically grounded EI EIF neural mass model.
- [[sources/thesis-proposal.md|Master's Thesis Proposal: MPC for Neurostimulation]] — Discusses NMMs and Wilson-Cowan models as tractable prediction models for MPC.
