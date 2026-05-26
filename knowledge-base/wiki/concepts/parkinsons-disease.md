---
title: Parkinson's Disease
type: concept
tags: [neurological-disorder, movement-disorder, deep-brain-stimulation]
sources: [steffen-cannon-2025.md, thesis-proposal.md]
created: 2026-05-14
updated: 2026-05-18
---

# Parkinson's Disease

Parkinson's Disease (PD) is a progressive neurological disorder that primarily affects movement. It is characterized by the loss of dopamine-producing neurons in the brain, leading to symptoms such as tremors, stiffness, and slow movement.

## Key Points

- **Biomarkers:** PD is associated with pathological bursts in the amplitude of **beta-band oscillations (13-30 Hz)** in population-level neural activity.
- **Control Objective:** In an MPC framework, the goal for PD is typically **desynchronization** of the pathological neural population [[raw/Thesis.md#Cost function | Thesis]].
- **Treatment via DBS:** Deep Brain Stimulation (DBS) is a standard surgical treatment involving electrodes that deliver electrical pulses to disrupt pathological activity.
- **Closed-loop DBS (CLDBS):** Adaptive or closed-loop DBS modulates stimulation amplitude in real-time based on disease biomarkers (like beta oscillations) to improve outcomes and reduce side effects [[raw/Steffen_Cannon_2025/Steffen_Cannon_2025.md#I. INTRODUCTION | Steffen & Cannon 2025]].

## Related Concepts

- [[concepts/beta-band-oscillations.md|Beta-band Oscillations]] — The primary feedback signal for PD CLDBS.
- [[concepts/model-predictive-control.md|Model Predictive Control]] — Used in advanced CLDBS to suppress pathological activity.

## Sources

- [[sources/steffen-cannon-2025.md|Deep Learning Model Predictive Control for Deep Brain Stimulation in Parkinson's Disease]] — Discusses nonlinear control strategies for PD treatment.
- [[sources/thesis-proposal.md|Master's Thesis Proposal: MPC for Neurostimulation]] — Identifies desynchronization as a key control objective for PD.
