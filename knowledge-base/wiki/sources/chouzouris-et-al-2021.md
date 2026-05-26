---
title: "Applications of optimal nonlinear control to a whole-brain network of FitzHugh-Nagumo oscillators"
type: source
tags: [optimal-control, nonlinear-dynamics, fitzhugh-nagumo, whole-brain-network, connectome]
sources: [raw/Chouzouris_et_al_2021/Chouzouris_et_al_2021.md]
created: 2026-05-14
updated: 2026-05-14
---

# Applications of optimal nonlinear control to a whole-brain network of FitzHugh-Nagumo oscillators

**Source:** `raw/Chouzouris_et_al_2021/Chouzouris_et_al_2021.md`
**Ingested:** 2026-05-14

## Summary

This paper applies the framework of optimal nonlinear control to steer the dynamics of a large-scale brain network. The network consists of nodes representing cortical areas, with local dynamics modeled by FitzHugh-Nagumo (FHN) oscillators and internode coupling strengths derived from human diffusion tensor imaging (DTI) data. The global dynamics are governed by:
$$ \dot{\mathbf{x}}_i = \mathbf{f}(\mathbf{x}_i) + \sigma \sum_{j} A_{ij} (\mathbf{x}_j - \mathbf{x}_i) + \mathbf{B}_i \mathbf{u}_i $$
The study focuses on two main tasks: targeted attractor switching between multistable network states and increasing network synchronization.

The authors find that optimal control strategies for nonlinear brain networks are highly task-dependent and state-dependent. Crucially, they demonstrate that intuitions from linear control theory (such as average and modal controllability) do not necessarily carry over to nonlinear systems, as the role of specific nodes in steering the network dynamics varies based on the specified task and the system's current location in state space.

## Key Takeaways

- **Optimal Nonlinear Control vs. Linear Theory:** Intuitions from linear control theory about node roles (based solely on connectome features) do not generally apply to nonlinear systems. [[raw/Chouzouris_et_al_2021/Chouzouris_et_al_2021.md#Abstract | Chouzouris et al. 2021]]
- **Task-Dependent Node Roles:** The importance of a node for controlling the network depends critically on the specific task (e.g., state-switching vs. synchronization) and the state-space location. [[raw/Chouzouris_et_al_2021/Chouzouris_et_al_2021.md#Abstract | Chouzouris et al. 2021]]
- **FitzHugh-Nagumo Model:** Used to represent average neural activity in each brain region, capturing transitions between bifurcations which linear models cannot. [[raw/Chouzouris_et_al_2021/Chouzouris_et_al_2021.md#I. Introduction | Chouzouris et al. 2021]]
- **Whole-Brain Connectivity:** internode coupling is based on an atlas-based segmentation and DTI-derived connectome of the human brain. [[raw/Chouzouris_et_al_2021/Chouzouris_et_al_2021.md#Abstract | Chouzouris et al. 2021]]
- **Sparse Control:** The framework allows for finding optimal control signals that affect only a small number of control sites. [[raw/Chouzouris_et_al_2021/Chouzouris_et_al_2021.md#I. Introduction | Chouzouris et al. 2021]]

## Related Concepts

- [[concepts/optimal-nonlinear-control.md|Optimal Nonlinear Control]] — The primary control framework used in the study.
- [[concepts/fitzhugh-nagumo-model.md|FitzHugh-Nagumo Model]] — The mathematical model for node dynamics.
- [[concepts/human-connectome.md|Human Connectome]] — The structural foundation for the brain network.
- [[concepts/closed-loop-brain-stimulation.md|Closed-loop Brain Stimulation]] — The broader application area for these control paradigms.

## Raw Notes

- The network state-space includes low- and high-activity fixed points separated by a high-amplitude limit cycle.
- Cost functional minimizes deviations from desired dynamics, control energy, and spatial nonsparsity.
- The study uses a 4th order Runge-Kutta method for numerical integration.
- The adjacency matrix was symmetrized and enforced a sparsity of 20%.
