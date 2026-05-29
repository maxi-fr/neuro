---
title: "Nonlinear optimal control of a mean-field model of neural population dynamics"
type: source
tags: [nonlinear-optimal-control, neural-mass-model, bistability, mean-field]
sources: [Literature/Salfenmose_Obermayer_2022.pdf]
created: 2026-05-14
updated: 2026-05-14
---

# Nonlinear optimal control of a mean-field model of neural population dynamics

**Source:** `Literature/Salfenmose_Obermayer_2022.pdf`
**Ingested:** 2026-05-14

## Summary

This paper explores the application of [[concepts/optimal-nonlinear-control.md|nonlinear optimal control]] to a biophysically grounded [[concepts/neural-mass-model.md|neural mass model]] (EI EIF model) consisting of recurrently coupled excitatory and inhibitory populations. The study focuses on the [[concepts/closed-loop-brain-stimulation.md#Bistability | bistable regime]] where low-activity ("down state") and high-activity ("up state") stable fixed points coexist. The authors use a gradient descent algorithm to minimize cost functions $\mathcal{J}$ that trade off target accuracy against control strength:
$$ \mathcal{J} = \int_{t_0}^{T} |r_E(t) - r_{target}|^2 dt + \alpha \int_{0}^{T} |u(t)|^k dt $$
where $k=1$ for sparsity and $k=2$ for energy. The population dynamics follow mean-field firing rate equations $\tau \dot{r} = -r + \Phi(I)$.

## Key Takeaways

- **Minimal Intervention Strategy:** In the noiseless case, optimal control strategies for state switching consist of a finite pulse that steers the system only minimally across the boundary of the target state's basin of attraction. Once the boundary is passed, the system naturally relaxes to the target state without further input [[Literature/Salfenmose_Obermayer_2022.pdf | Salfenmose & Obermayer 2022 (3.2.)]].
- **Cost Function Effects:**
  - **L1-norm (Sparsity):** Penalizing the 1-norm leads to "one-dimensional" control solutions, targeting only one of the populations (excitatory or inhibitory).
  - **L2-norm (Energy):** Penalizing the 2-norm leads to "two-dimensional" solutions, where both populations may receive input to keep absolute values low [[Literature/Salfenmose_Obermayer_2022.pdf | Salfenmose & Obermayer 2022 (3.3.)]].
- **Target Population Efficiency:** For the down-to-up transition, the choice of target population (E or I) depends on the system's location in state space relative to bifurcation lines. For the up-to-down transition, stimulating the excitatory population is consistently more efficient due to the geometry of the regime boundary [[Literature/Salfenmose_Obermayer_2022.pdf | Salfenmose & Obermayer 2022 (3.3.)]].
- **Linear Scaling:** Despite the model's nonlinearity, the required control amplitude scales linearly with the distance to the target regime boundary in the dominating input channel [[Literature/Salfenmose_Obermayer_2022.pdf | Salfenmose & Obermayer 2022 (3.3.)]].

## Related Concepts

- [[concepts/optimal-nonlinear-control.md|Optimal Nonlinear Control]] — Main framework applied.
- [[concepts/neural-mass-model.md|Neural Mass Model]] — Specifically the EI EIF model.
- [[concepts/closed-loop-brain-stimulation.md|Closed-Loop Brain Stimulation]] — Application context.
- [[concepts/fitzhugh-nagumo-model.md|FitzHugh-Nagumo Model]] — Mentioned as a point of comparison for whole-brain network models.

## Raw Notes

- The model is a DDAE (delay differential-algebraic equations) system.
- Focus on "Point a" ($\mu_E^{ext}=0.45, \mu_I^{ext}=0.475$) and "Point b" ($\mu_E^{ext}=0.475, \mu_I^{ext}=0.6$) in the bistable regime.
- Precision measurement onset $t_0$ is used to define the control time window.
- Gradient descent via adjoint method.
