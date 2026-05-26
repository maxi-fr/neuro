---
title: "Master's Thesis Proposal: MPC for Neurostimulation"
type: source
tags: [model-predictive-control, neurostimulation, state-estimation, large-language-models, neural-dynamics]
sources: [raw/Thesis.md]
created: 2026-05-18
updated: 2026-05-18
---

# Master's Thesis Proposal: MPC for Neurostimulation

**Source:** `raw/Thesis.md`
**Ingested:** 2026-05-18

## Summary

This document outlines a Master's thesis project focused on using Model Predictive Control (MPC) to steer neural populations from pathological to physiological states. The framework incorporates neural dynamics models, state and parameter estimation, and the integration of Large Language Models (LLMs) for automated design and real-time adaptation. The proposal highlights the need for patient-specific models and multi-scale adaptation (milliseconds to days) to address research gaps in neurostimulation.

## Key Takeaways

- **MPC Framework:** Uses a prediction model and cost function to achieve desired behaviors while respecting safety constraints to avoid tissue damage [[raw/Thesis.md#Model Predictive Control for Neurostimulation | Thesis]].
- **Hierarchical Control:** Suggests a hierarchical structure where a faster low-level controller tracks MPC-generated references to achieve millisecond sampling times [[raw/Thesis.md#Master's project | Thesis]].
- **LLM Integration:** Proposes using LLMs to design treatment protocols, integrate multimodal feedback (EEG, patient/physician assessments), and propose stimulation parameters [[raw/Thesis.md#Master's project | Thesis]].
- **Modeling Hierarchy:** Distinguishes between complex population models (diffusively coupled single-cells) for simulation and simpler models (Wilson-Cowan, Neural Mass Models) for real-time MPC [[raw/Thesis.md#Prediction model | Thesis]].
- **Adaptive Control:** Emphasizes the need for parameter estimation and model adaptation to account for long-term effects like plasticity and disease progression [[raw/Thesis.md#Prediction model | Thesis]].

## Related Concepts

- [[concepts/model-predictive-control.md|Model Predictive Control]] — Core control methodology for neurostimulation.
- [[concepts/neural-mass-model.md|Neural Mass Model]] — Used for reduced-order population modeling.
- [[concepts/parkinsons-disease.md|Parkinson's Disease]] — One of the target pathologies (desynchronization objective).
- [[concepts/seizure-suppression.md|Seizure Suppression]] — Objective for epilepsy (synchronization/desynchronization).
- [[concepts/fitzhugh-nagumo-model.md|FitzHugh-Nagumo Model]] — Single-cell dynamics model mentioned.
- [[concepts/hodgkin-huxley-model.md|Hodgkin-Huxley Model]] — Single-cell dynamics model mentioned.
- [[concepts/llms-in-neurostimulation.md|LLMs in Neurostimulation]] — Integration of LLMs for protocol design and feedback.
- [[concepts/state-estimation.md|State Estimation]] — Required for feedback control using unmeasurable states.

## Raw Notes

- Objectives: Synchronization (epilepsy), Desynchronization (Parkinson's), Amplitude reduction (depression).
- Sampling time challenge: Pure MPC may struggle with ms requirements.
- Constraints: Must be robustly satisfied to avoid overstimulation.
