---
title: Human Connectome
type: concept
tags: [neuroanatomy, connectivity, DTI]
sources: [chouzouris-et-al-2021.md, chang-et-al-2020.md]
created: 2026-05-14
updated: 2026-05-14
---

# Human Connectome

The human connectome is the comprehensive map of neural connections in the human brain. In the context of large-scale brain modeling, it usually refers to the structural connectivity (white matter tracts) between different brain regions.

## Key Points

- **Data Acquisition:** Structural connectivity is often estimated using Diffusion Tensor Imaging (DTI) and tractography algorithms. [[Literature/Chouzouris_et_al_2021.pdf | Chouzouris et al. 2021 (Appendix D)]]
- **Network Structure:** The connectome defines the edges and weights in a brain network model, where nodes represent regions of interest (defined by an atlas). Mathematically, it is represented by an adjacency matrix $A$, where $A_{ij}$ denotes the connection strength between node $i$ and node $j$. The degree of a node $k$ is given by $d_k = \sum_{i} A_{ik}$. [[Literature/Chouzouris_et_al_2021.pdf | Chouzouris et al. 2021 (I. Introduction)]]
- **Role in Dynamics:** The topology of the connectome (e.g., hubness, modularity) significantly influences the global dynamics and controllability of the brain. [[Literature/Chouzouris_et_al_2021.pdf | Chouzouris et al. 2021 (I. Introduction)]]
- **Symmetry and Sparsity:** Structural connectivity matrices are often symmetrized ($A = A^T$, as tractography is usually undirected) and thresholded to enforce sparsity, typically by keeping only the top percentage of connections (e.g., 20% sparsity threshold). [[Literature/Chouzouris_et_al_2021.pdf | Chouzouris et al. 2021 (Appendix D)]]

## Related Concepts

- [[concepts/neural-mass-model.md|Neural Mass Model]] — The structural framework on which neural mass models are built.
- [[concepts/optimal-nonlinear-control.md|Optimal Nonlinear Control]] — Strategies for control often depend on connectome features.
- [[concepts/fitzhugh-nagumo-model.md|FitzHugh-Nagumo Model]] — Node dynamics model coupled via connectome weights in whole-brain simulations.

## Sources

- [[sources/chouzouris-et-al-2021.md|Applications of optimal nonlinear control to a whole-brain network of FitzHugh-Nagumo oscillators]] — Uses a DTI-derived connectome from the Human Connectome Project.
- [[sources/chang-et-al-2020.md|Model Predictive Control for Seizure Suppression Based on Nonlinear Auto-Regressive Moving-Average Volterra Model]] — Also utilizes brain connectivity models for its control framework.
