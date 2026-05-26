---
title: "Model Predictive Control for Seizure Suppression Based on Nonlinear Auto-Regressive Moving-Average Volterra Model"
type: source
tags: [mpc, seizure-suppression, nmm, volterra, closed-loop]
sources: [raw/Chang_et_al_2020/Chang_et_al_2020.md]
created: 2026-05-14
updated: 2026-05-14
---

# Model Predictive Control for Seizure Suppression Based on Nonlinear Auto-Regressive Moving-Average Volterra Model

**Source:** `raw/Chang_et_al_2020/Chang_et_al_2020.md`
**Ingested:** 2026-05-14

## Summary

This article investigates a closed-loop brain stimulation method using Model Predictive Control (MPC) to suppress epileptic seizures. The researchers used a Neural Mass Model (NMM) as a black-box representation of the brain. They identified a nonlinear auto-regressive moving-average (NARMA) Volterra model:
$$ y(k) = F[y(k-1), \dots, y(k-n), u(k-1), \dots, u(k-m)] $$
to characterize the relationship between stimulation input $u$ and neuronal responses $y$. The MPC strategy then uses this Volterra model to generate optimal stimulation waveforms to eliminate epileptiform activity.

## Key Takeaways

- **Black-box Approach:** The control strategy does not require detailed prior knowledge of the brain's physiological parameters, treating it as a black box for system identification. [[raw/Chang_et_al_2020/Chang_et_al_2020.md#II. METHODS | Chang et al. 2020]]
- **NARMA Volterra Model:** This model is used for its ability to capture complex nonlinear dynamics with a more compact structure than neural networks. [[raw/Chang_et_al_2020/Chang_et_al_2020.md#I. Introduction | Chang et al. 2020]]
- **Robustness:** The proposed MPC strategy shows robustness to system disturbances, making it suitable for clinical applications where modeling uncertainty is high. [[raw/Chang_et_al_2020/Chang_et_al_2020.md#I. Introduction | Chang et al. 2020]]
- **Optimal Stimulation:** Unlike traditional open-loop stimulation, this closed-loop approach automatically adjusts waveforms in real-time based on state feedback, potentially reducing energy consumption and side effects. [[raw/Chang_et_al_2020/Chang_et_al_2020.md#I. Introduction | Chang et al. 2020]]

## Related Concepts

- [[concepts/model-predictive-control.md|Model Predictive Control]] — The core control strategy used to optimize stimulation.
- [[concepts/seizure-suppression.md|Seizure Suppression]] — The primary therapeutic goal of the study.
- [[concepts/neural-mass-model.md|Neural Mass Model]] — Used as the virtual brain/black-box model for simulation.
- [[concepts/volterra-model.md|Volterra Model]] — The mathematical framework used for system identification and prediction.
- [[concepts/closed-loop-brain-stimulation.md|Closed-loop Brain Stimulation]] — The broader technological framework for the research.

## Raw Notes

- EEG signals are the summation of post-synaptic membrane potentials.
- NMM can produce normal or epileptiform discharges by adjusting excitatory/inhibitory interactions.
- Traditional DBS/TES are often open-loop and constant, which can lead to energy waste and side effects.
- MPC solves an optimization problem over a future time horizon.
- NARMA Volterra model: $y(k) = F[y(k-1), ..., y(k-n), u(k-1), ..., u(k-m)]$.
