---
title: LLMs in Neurostimulation
type: concept
tags: [artificial-intelligence, llms, neurostimulation, personalized-medicine, hierarchical-control]
sources: [thesis-proposal.md]
created: 2026-05-18
updated: 2026-05-18
---

# LLMs in Neurostimulation

The integration of Large Language Models (LLMs) into neurostimulation frameworks represents a novel approach to automate treatment protocol design and incorporate multimodal, patient-specific data into real-time control loops.

## Key Points

- **Protocol Design:** LLMs can be used to automatically design or shape [[concepts/model-predictive-control.md|Model Predictive Control]] designs and treatment protocols based on clinical guidelines and patient history [[raw/Thesis.md#Master's project | Thesis]].
- **Multimodal Feedback Integration:** LLMs can integrate diverse feedback signals online, including:
  - Electrophysiological data (e.g., EEG).
  - Medical signals from other sensors.
  - Qualitative assessments from the patient (subjective reports).
  - Expert assessments from the physician [[raw/Thesis.md#Master's project | Thesis]].
- **Parameter Proposal:** The LLM can propose high-level stimulation parameters such as frequency, amplitude, or electrode configuration. These proposals can be refined by local optimizers or repeated queries [[raw/Thesis.md#Master's project | Thesis]].
- **Hierarchical Role:** In a hierarchical control architecture, the LLM may function at the highest level, making contextual decisions that guide the mid-level MPC and low-level trackers [[raw/Thesis.md#Master's project | Thesis]].

## Related Concepts

- [[concepts/model-predictive-control.md|Model Predictive Control]] — The control framework that LLMs help design and adapt.
- [[concepts/closed-loop-brain-stimulation.md|Closed-loop Brain Stimulation]] — The clinical application for LLM-integrated control.

## Sources

- [[sources/thesis-proposal.md|Master's Thesis Proposal: MPC for Neurostimulation]] — Proposes the integration of LLMs for treatment protocol design and multimodal feedback.
