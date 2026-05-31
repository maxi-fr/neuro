---
title: Seizure Suppression
type: concept
tags: [epilepsy, neurology, treatment]
sources: [chang-et-al-2020.md, thesis-proposal.md]
created: 2026-05-14
updated: 2026-05-18
---

# Seizure Suppression

Seizure suppression refers to the methods and strategies used to eliminate or reduce the occurrence of epileptic seizures, characterized by epileptiform spikes and high-frequency oscillations in EEG signals.

## Key Points

- **Epileptiform Activity:** Characterized by abnormal discharges that can be simulated using macroscopic models like the [[concepts/neural-mass-model.md|Neural Mass Model]].
- **Control Objective:** In an MPC framework, the goal for epilepsy is often **synchronization** (to stabilize activity) or desynchronization, depending on the specific seizure type and mechanism [[Notes/Thesis.md#Cost function | Thesis]].
- **Brain Stimulation:** Techniques like Deep Brain Stimulation (DBS) and Transcranial Electrical Stimulation (TES) are used to modulate neural activity and suppress seizures.
- **Closed-loop Approach:** Modern research focuses on [[concepts/closed-loop-brain-stimulation.md|Closed-loop Brain Stimulation]] which adjusts stimulation in real-time to match the patient's state.

## Related Concepts

- [[concepts/neural-mass-model.md|Neural Mass Model]] — Used to simulate epileptiform EEG signals for research.
- [[concepts/model-predictive-control.md|Model Predictive Control]] — Applied to generate optimal stimulation waveforms for seizure suppression.

## Summaries

- [[summaries/chang-et-al-2020.md|Model Predictive Control for Seizure Suppression Based on Nonlinear Auto-Regressive Moving-Average Volterra Model]] — Investigates using MPC for seizure suppression.
- [[summaries/thesis-proposal.md|Master's Thesis Proposal: MPC for Neurostimulation]] — Identifies synchronization/desynchronization as control objectives for epilepsy.
