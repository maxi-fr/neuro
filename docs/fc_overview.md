# Functional Connectivity in the Literature Knowledge Base

This document provides a comprehensive overview of research papers in the knowledge base that investigate, utilize, or reference **functional connectivity (FC)** and **functional connectivity dynamics (FCD)**.

---

## 1. Primary Subject & Model Fitting Targets

These papers place functional connectivity at the core of their investigation—either measuring stimulation-induced changes in FC or using static FC and time-resolved FCD matrices as empirical targets in whole-brain optimization.

### [Kunze et al. (2016)](file:///C:/Users/frank/closed-loop-neurostimulation/knowledge-base/Literature/extracted/neural-dynamics/kunze-tdcs-functional-connectivity-2016/kunze-tdcs-functional-connectivity-2016.md)

* **Title:** *Transcranial direct current stimulation changes resting state functional connectivity: A large-scale brain network modeling study*
* **Wiki Summary:** [kunze-tdcs-functional-connectivity-2016.md](file:///C:/Users/frank/closed-loop-neurostimulation/knowledge-base/Literature/wiki/summaries/kunze-tdcs-functional-connectivity-2016.md)
* **Raw PDF:** [kunze-tdcs-functional-connectivity-2016.pdf](file:///C:/Users/frank/closed-loop-neurostimulation/knowledge-base/Literature/raw/neural-dynamics/kunze-tdcs-functional-connectivity-2016.pdf)
* **Mentions:** 28
* **Key Role & Findings:**
  * Uses a large-scale structural connectivity connectome with local Jansen-Rit and Kuramoto dynamics to investigate how tDCS alters resting-state functional connectivity.
  * Demonstrates that tDCS induces broad increases in FC both across cerebral regions and between cortical areas and subcortical structures (e.g., thalamus).

### [Pille et al. (2025)](file:///C:/Users/frank/closed-loop-neurostimulation/knowledge-base/Literature/extracted/neural-dynamics/pille-parameter-estimation-autodiff-2025/pille-parameter-estimation-autodiff-2025.md)

* **Title:** *Parameter estimation in whole-brain models using automatic differentiation*
* **Wiki Summary:** [pille-parameter-estimation-autodiff-2025.md](file:///C:/Users/frank/closed-loop-neurostimulation/knowledge-base/Literature/wiki/summaries/pille-parameter-estimation-autodiff-2025.md)
* **Raw PDF:** [pille-parameter-estimation-autodiff-2025.pdf](file:///C:/Users/frank/closed-loop-neurostimulation/knowledge-base/Literature/raw/neural-dynamics/pille-parameter-estimation-autodiff-2025.pdf)
* **Mentions:** 14
* **Key Role & Findings:**
  * Uses static functional connectivity (Pearson correlation / RMSE) and functional connectivity dynamics (FCD via Kolmogorov–Smirnov distance over kernel density smoothed histograms) as the loss function for gradient-based parameter estimation in reduced Wong–Wang models.
  * Shows that static FC loss forms an extended ridge in parameter space, while adding FCD regularizes the fit and prevents unphysiological over-synchronization.

### [Roffet et al. (2026)](file:///C:/Users/frank/closed-loop-neurostimulation/knowledge-base/Literature/extracted/neural-dynamics/roffet-bifurcation-biomarkers-brain-states/roffet-bifurcation-biomarkers-brain-states.md)

* **Title:** *Bifurcation-based biomarkers of brain states*
* **Wiki Summary:** [roffet-bifurcation-biomarkers-brain-states.md](file:///C:/Users/frank/closed-loop-neurostimulation/knowledge-base/Literature/wiki/summaries/roffet-bifurcation-biomarkers-brain-states.md)
* **Raw PDF:** [roffet-bifurcation-biomarkers-brain-states.pdf](file:///C:/Users/frank/closed-loop-neurostimulation/knowledge-base/Literature/raw/neural-dynamics/roffet-bifurcation-biomarkers-brain-states.pdf)
* **Mentions:** 12
* **Key Role & Findings:**
  * Calibrates whole-brain network models against empirical phase functional connectivity dynamics (phFCD) and static FC matrices.
  * Tracks dynamical transitions and bifurcation biomarkers distinguishing wakefulness, propofol anesthesia, and sleep stages.

### [Deco et al. (2017)](file:///C:/Users/frank/closed-loop-neurostimulation/knowledge-base/Literature/extracted/neural-dynamics/deco-resting-fluctuations-2017/deco-resting-fluctuations-2017.md)

* **Title:** *The dynamics of resting fluctuations in the brain: metastability and its dynamical origins*
* **Wiki Summary:** [deco-resting-fluctuations-2017.md](file:///C:/Users/frank/closed-loop-neurostimulation/knowledge-base/Literature/wiki/summaries/deco-resting-fluctuations-2017.md)
* **Raw PDF:** [deco-resting-fluctuations-2017.pdf](file:///C:/Users/frank/closed-loop-neurostimulation/knowledge-base/Literature/raw/neural-dynamics/deco-resting-fluctuations-2017.pdf)
* **Mentions:** 1 (Foundational paper)
* **Key Role & Findings:**
  * Fits a whole-brain Hopf bifurcation model simultaneously to grand-average static FC and functional connectivity dynamics (FCD).
  * Demonstrates that maximum metastability occurs at the edge of the Hopf bifurcation, matching empirical resting-state fMRI fluctuations.

---

## 2. Control Target Selection & Biomarker Feature Extraction

These papers integrate functional connectivity into control strategies or feature extraction pipelines for neurological disorders:

### [Liang et al. (2024)](file:///C:/Users/frank/closed-loop-neurostimulation/knowledge-base/Literature/extracted/model-predictive-control/liang-dmd-mpc-seizure-hub-2024/liang-dmd-mpc-seizure-hub-2024.md)

* **Title:** *Dynamic Mode Decomposition and Model Predictive Control for Seizure Suppression Targeting Network Hubs*
* **Wiki Summary:** [liang-dmd-mpc-seizure-hub-2024.md](file:///C:/Users/frank/closed-loop-neurostimulation/knowledge-base/Literature/wiki/summaries/liang-dmd-mpc-seizure-hub-2024.md)
* **Raw PDF:** [liang-dmd-mpc-seizure-hub-2024.pdf](file:///C:/Users/frank/closed-loop-neurostimulation/knowledge-base/Literature/raw/model-predictive-control/liang-dmd-mpc-seizure-hub-2024.pdf)
* **Mentions:** 2
* **Key Role & Findings:**
  * Uses covariance-based functional connectivity estimated during the first 2 seconds of ictal iEEG to determine "FC hubs" (nodes with the highest functional connection strength to the seizure onset zone).
  * Demonstrates that targeting FC hubs with DMD-based MPC achieves effective seizure termination with significantly less control effort than brute-force stimulation.

### [Chourdaki et al. (2026)](file:///C:/Users/frank/closed-loop-neurostimulation/knowledge-base/Literature/extracted/neural-dynamics/chourdaki-tkeo-feature-extraction-2026/chourdaki-tkeo-feature-extraction-2026.md)

* **Title:** *Nonlinear Teager-Kaiser energy operator feature extraction for epilepsy detection*
* **Wiki Summary:** [chourdaki-tkeo-feature-extraction-2026.md](file:///C:/Users/frank/closed-loop-neurostimulation/knowledge-base/Literature/wiki/summaries/chourdaki-tkeo-feature-extraction-2026.md)
* **Raw PDF:** [chourdaki-tkeo-feature-extraction-2026.pdf](file:///C:/Users/frank/closed-loop-neurostimulation/knowledge-base/Literature/raw/neural-dynamics/chourdaki-tkeo-feature-extraction-2026.pdf)
* **Mentions:** 1
* **Key Role & Findings:**
  * Outlines future directions for combining local instantaneous energy measures (TKEO/DESA-1) with functional connectivity to map network propagation pathways during seizure onset.

### [Fang & Yang (2023)](file:///C:/Users/frank/closed-loop-neurostimulation/knowledge-base/Literature/extracted/model-predictive-control/fang-predictive-neuromodulation-mdd-2023/fang-predictive-neuromodulation-mdd-2023.md)

* **Title:** *Predictive neuromodulation and model predictive control for major depressive disorder*
* **Wiki Summary:** [fang-predictive-neuromodulation-mdd-2023.md](file:///C:/Users/frank/closed-loop-neurostimulation/knowledge-base/Literature/wiki/summaries/fang-predictive-neuromodulation-mdd-2023.md)
* **Raw PDF:** [fang-predictive-neuromodulation-mdd-2023.pdf](file:///C:/Users/frank/closed-loop-neurostimulation/knowledge-base/Literature/raw/model-predictive-control/fang-predictive-neuromodulation-mdd-2023.pdf)
* **Mentions:** 1
* **Key Role & Findings:**
  * Synthesizes clinical transcranial magnetic stimulation (TMS) literature highlighting that antidepressant efficacy strongly correlates with intrinsic functional connectivity between the subgenual cingulate and the targeted dlPFC sites.

---

## 3. Network Modeling Context & Literature Citations

* **[Chouzouris et al. (2021)](file:///C:/Users/frank/closed-loop-neurostimulation/knowledge-base/Literature/extracted/neural-dynamics/chouzouris-optimal-nonlinear-control-2021/chouzouris-optimal-nonlinear-control-2021.md)** ([Summary](file:///C:/Users/frank/closed-loop-neurostimulation/knowledge-base/Literature/wiki/summaries/chouzouris-optimal-nonlinear-control-2021.md) | [PDF](file:///C:/Users/frank/closed-loop-neurostimulation/knowledge-base/Literature/raw/neural-dynamics/chouzouris-optimal-nonlinear-control-2021.pdf)) — 4 mentions discussing how spontaneous fluctuations in whole-brain FitzHugh-Nagumo networks near criticality reproduce empirical resting-state functional connectivity patterns.
* **[Ye et al. (2024)](file:///C:/Users/frank/closed-loop-neurostimulation/knowledge-base/Literature/extracted/neural-dynamics/ye-brain-network-models-2024/ye-brain-network-models-2024.md)** ([Summary](file:///C:/Users/frank/closed-loop-neurostimulation/knowledge-base/Literature/wiki/summaries/ye-brain-network-models-2024.md) | [PDF](file:///C:/Users/frank/closed-loop-neurostimulation/knowledge-base/Literature/raw/neural-dynamics/ye-brain-network-models-2024.pdf)) — 1 mention reviewing graph representations in brain network models, distinguishing structural connectivity (anatomical tracts) from functional connectivity (statistical dependencies).
* **[Sun et al. (2020)](file:///C:/Users/frank/closed-loop-neurostimulation/knowledge-base/Literature/extracted/learned-dynamics-models/sun-echo-state-networks-review-2020/sun-echo-state-networks-review-2020.md)** ([Summary](file:///C:/Users/frank/closed-loop-neurostimulation/knowledge-base/Literature/wiki/summaries/sun-echo-state-networks-review-2020.md) | [PDF](file:///C:/Users/frank/closed-loop-neurostimulation/knowledge-base/Literature/raw/learned-dynamics-models/sun-echo-state-networks-review-2020.pdf)) — 1 mention citing applications of reservoir computing to recover resting-state fMRI functional connectivity.
* **[Yu et al. (2024)](file:///C:/Users/frank/closed-loop-neurostimulation/knowledge-base/Literature/extracted/neural-dynamics/yu-closed-loop-tes-epileptic-propagation-2024/yu-closed-loop-tes-epileptic-propagation-2024.md)** ([Summary](file:///C:/Users/frank/closed-loop-neurostimulation/knowledge-base/Literature/wiki/summaries/yu-closed-loop-tes-epileptic-propagation-2024.md) | [PDF](file:///C:/Users/frank/closed-loop-neurostimulation/knowledge-base/Literature/raw/neural-dynamics/yu-closed-loop-tes-epileptic-propagation-2024.pdf)) — 1 mention citing Kunze et al. (2016) regarding tES-induced changes in resting-state functional networks.
