---
title: Volterra Model
type: concept
tags: [mathematics, system-identification, modeling]
sources: [chang-et-al-2020.md]
created: 2026-05-14
updated: 2026-05-14
---

# Volterra Model

The Volterra model is a functional series representation used to describe the input-output relationship of nonlinear dynamic systems. It is particularly effective for modeling uncertain and complex nonlinear processes.

## Key Points

- **Structure:** Generally less time-consuming and more compact than neural networks for certain implementations. [[raw/Chang_et_al_2020/Chang_et_al_2020.md#I. Introduction | Chang et al. 2020]]
- **NARMA Volterra:** Nonlinear Auto-Regressive Moving-Average Volterra models incorporate past inputs and outputs to predict future states. A second-order discrete-time Volterra series mapping stimulation $u(n)$ to EEG output $y(n)$ can be expressed as:
    $$ y(n) = h_0 + \sum_{i=0}^M h_1(i) u(n-i) + \sum_{i=0}^M \sum_{j=0}^M h_2(i, j) u(n-i) u(n-j) + \dots $$
    where $h_k$ are the Volterra kernels of order $k$. [[raw/Chang_et_al_2020/Chang_et_al_2020.md#I. Introduction | Chang et al. 2020]]
- **System Identification:** Used to learn the dynamics of a system from experimental data without needing to know its internal physical structure (black-box modeling). [[raw/Chang_et_al_2020/Chang_et_al_2020.md#I. Introduction | Chang et al. 2020]]
- **Application in MPC:** Often serves as the predictive model in [[concepts/model-predictive-control.md|Model Predictive Control]] schemes. [[raw/Chang_et_al_2020/Chang_et_al_2020.md#I. Introduction | Chang et al. 2020]]

## Related Concepts

- [[concepts/model-predictive-control.md|Model Predictive Control]] — Uses Volterra models for future state prediction.
- [[concepts/neural-mass-model.md|Neural Mass Model]] — The dynamics of an NMM can be captured by a Volterra model via system identification.
- [[concepts/seizure-suppression.md|Seizure Suppression]] — Primary application domain for Volterra-based MPC.
- [[concepts/closed-loop-brain-stimulation.md|Closed-loop Brain Stimulation]] — Volterra models enable data-driven closed-loop control of neural dynamics.

## Sources

- [[sources/chang-et-al-2020.md|Model Predictive Control for Seizure Suppression Based on Nonlinear Auto-Regressive Moving-Average Volterra Model]] — Utilizes a NARMA Volterra model as the predictive component of an MPC controller. [[raw/Chang_et_al_2020/Chang_et_al_2020.md#I. Introduction | Chang et al. 2020]]
