# Thesis Review Feedback & Action Items (09/11 Review by Max Pfefferkorn)

> [!NOTE]
> Extracted from `Thesis_Max_Frankenberg_2026_11_06__MP.pdf` in `thesis/thesis-feedback/09_11_MP/`.

## Overview & Statistics

| Metric | Count |
| :--- | :--- |
| **Total Review Comments & Action Items** | **25** |
| **Status: Open (`[ ]`)** | **9** |
| **Status: Resolved (`[x]`)** | **16** |
| **Primary Chapters Reviewed** | Chapter 5 (`05_mpc_framework.tex`), pp. 38–48 |

---

## 1. General MPC Formulation & Theoretical Framing (4 comments)

| ID | Status | Page (PDF / Doc) | Source File & Line(s) | Marked Text / Element | Reviewer Feedback (Max Pfefferkorn) |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **GEN-01** | `[ ]` | 52 / 38 | 05_mpc_framework.tex:L26-28 | `interval` in `The controller plans over H \in \mathbb{N}_{>0} sampling intervals of duration \Delta t > 0... \mathbf{u}_{j\vert k} is the input planned at step k for interval j = 0, \dots, H-1` | Refers to the discretization, but is this mentioned explicitly? If this is left implicit, the sentence might be unclear to the reader. |
| **GEN-02** | `[ ]` | 53 / 39 | 05_mpc_framework.tex:L87 | `Replanning alone does not ensure that a feasible plan exists at every subsequent update or that the closed-loop state converges to the target equilibrium [13], [60].` | I would add this to the previous paragraph, and instead of making it a stand-alone observation with no conlusion, consider something like "A suitably chosen terminal constraint, regiona, and controller can be used to ensure that ..., which is not naturally the case". Oder irgendwie sowas. |
| **GEN-03** | `[x]` | 53 / 39 | 05_mpc_framework.tex:L88-89 | `For feedback at the chosen sampling rate, the controller must also solve each finite-horizon problem within one sampling interval. The next section compares two ways to formulate that problem for a numerical solver.` | Das passt vom Kontext her vielleicht besser zum vorletzten Absatz |
| **GEN-04** | `[x]` | 54 / 40 | 05_mpc_framework.tex:L75-81 | Figure 5.1 caption: `The plan and realized entries in the legend apply to both panels.` | Can be omitted, this is standard |

---

## 2. Numerical Transcription & Section Structure (3 comments)

| ID | Status | Page (PDF / Doc) | Source File & Line(s) | Marked Text / Element | Reviewer Feedback (Max Pfefferkorn) |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **TRAN-01** | `[x]` | 54 / 40 | 05_mpc_framework.tex:L91-155 | Section 5.2 (`\section{Numerical Transcription and Solution}`) | This section tries to do the comparison for each "property" of the two methods, which leads to jumping back and forth.<br><br>I would try to first describe one of the methods, and then the other. For the second one, you could refer back to the first one to highlight the differences, but this would bring some more structure into the section. But you'll have to test this and see how if it works out. |
| **TRAN-02** | `[x]` | 54 / 40 | 05_mpc_framework.tex:L126-128 | `In multiple shooting, the state constraints instead act directly on the state decision variables. The input constraints have the same form in both transcriptions.` | I would move this to the multiple shooting description and make it clear there; Here, you should use a parallel structure to only describe single shooting; try to avoid jumping |
| **TRAN-03** | `[x]` | 54 / 40 | 05_mpc_framework.tex:L130 | `This difference` in `This difference also determines which derivatives a gradient-based solver needs to evaluate the objective and constraint sensitivities.` | in the optimization variables<br>or something like this, to make it clear. |

---

## 3. Model State Transition, References & Notation (8 comments)

| ID | Status | Page (PDF / Doc) | Source File & Line(s) | Marked Text / Element | Reviewer Feedback (Max Pfefferkorn) |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **NOT-01** | `[x]` | 56 / 42 | 05_mpc_framework.tex:L191 | `\mathbf{x} = \mathbf{h}\text{, }\mathbf{f} = \mathbf{F}_{\boldsymbol{\theta}}\text{ and }\hat{\mathbf{x}}_k = \mathbf{h}_k` | use `\equiv` here instead of `=` |
| **NOT-02** | `[ ]` | 56 / 42 | 05_mpc_framework.tex:L237-238 | `Regional recruitment cannot be assessed directly from the scalp EEG observations available to the controller. So instead we extract features of predicted EEG to serve as proxies for seizure activity and` | [Highlighted text across page break; paired with NOT-03] |
| **NOT-03** | `[ ]` | 57 / 43 | 05_mpc_framework.tex:L238-239 | `compare that to features extracted from EEG signals of healthy simulated brains.` | I'd reformulate this sentence to make its meaning clearer. Extracting features and comparing thos derived under different conditions could be misunderstood and then seems unrealted to this section |
| **NOT-04** | `[x]` | 57 / 43 | 05_mpc_framework.tex:L241 | `Panel (a) of Figure 5.3` in `Panel (a) of Figure~\ref{fig:mpc_suppression_references} shows that healthy fluctuations...` | Figure 5.3(a) |
| **NOT-05** | `[x]` | 57 / 43 | 05_mpc_framework.tex:L244 | `squared tracking cost` (highlighted `squared`) | quadratic tracking cost |
| **NOT-06** | `[x]` | 57 / 43 | 05_mpc_framework.tex:L243 | `healthy reference trajectories` in `mean of the waveform samples pooled across time and across the healthy reference trajectories.` | Were they introduced? |
| **NOT-07** | `[x]` | 57 / 43 | 05_mpc_framework.tex:L277 | `Panel (b) of Figure 5.3` in `Panel (b) of Figure~\ref{fig:mpc_suppression_references} illustrates where seizure log power...` | [Highlighted text; paired with NOT-04 for figure reference style `Figure 5.3(b)`] |
| **NOT-08** | `[ ]` | 58 / 44 | 05_mpc_framework.tex:L285 | Figure 5.3 caption: `(a)` in `(a) Waveform excerpts from a separate healthy run...` | This is also something that you don't do consistently |

---

## 4. Sparsity Formulations & Objective Structure (7 comments)

| ID | Status | Page (PDF / Doc) | Source File & Line(s) | Marked Text / Element | Reviewer Feedback (Max Pfefferkorn) |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **SPAR-01** | `[x]` | 59 / 45 | 05_mpc_framework.tex:L339-341 | `Nagahara et al.,` in `problem studied by Nagahara et al., the L0 and L1 formulations...` | Don't use this citation style |
| **SPAR-02** | `[ ]` | 59 / 45 | 05_mpc_framework.tex:L339-342 | `For the discrete-time maximum hands-off problem studied by Nagahara et al., the L0 and L1 formulations have the same minimizers if the L1 problem is normal and has a solution [64]. In that setting, normality excludes switching-function values at which the minimizing input is ambiguous.` | Relevance for your work?<br>What is the main message that the reader needs to understand from these two sentences? This is unclear to me right now. |
| **SPAR-03** | `[ ]` | 59 / 45 | 05_mpc_framework.tex:L354 | `continuous-current` in `The continuous-current objective can then be written as` | Why is it called continuous-current? |
| **SPAR-04** | `[ ]` | 59 / 45 | 05_mpc_framework.tex:L366-370 | `Near zero, a small feasible increase in current can improve predicted suppression to first order while increasing the L2 current cost only to second order... If the first-order decrease in weighted suppression cost is smaller than the first-order increase in weighted L1 cost, a sufficiently small change in that direction increases the objective.` | Please check if you can make the phrasing more intuitive, which would make this easier to get at first read. Maybe state first that L1 prefers currents at zero and bring this as the reasoning. |
| **SPAR-05** | `[x]` | 59 / 45 | 05_mpc_framework.tex:L375-381 | Margin note next to Eq. (5.12) (`\frac{w_1}{H}\sum_{j=0}^{H-1}\sum_{p=1}^{\ndim{u}}z_{j\vert k,p}`) | J_L1 = ? |
| **SPAR-06** | `[ ]` | 59 / 45 | 05_mpc_framework.tex:L354-363 | Margin note next to Eq. (5.11) (`\mathcal{J}_{\mathrm{cont}} = \mathcal{J}_{\mathrm{supp}} + w_{L_2}\mathcal{J}_{L_2} + w_1\mathcal{J}_{L_1}`) | If you aren't do so already, you could look into the literature that we originally provided. I think they also used a similar cost structure. So you could relate this to existing work as well. |
| **SPAR-07** | `[x]` | 60 / 46 | 05_mpc_framework.tex:L416-424 | `initial condition.` preceding Eq. (5.15) (`\min \mathcal{J}_{\mathrm{cont}} + w_\delta \mathcal{J}_\delta`) | (5.15) doesn't belong to any sentence |

---

## 5. Activation Solvers & Chapter Summary (3 comments)

| ID | Status | Page (PDF / Doc) | Source File & Line(s) | Marked Text / Element | Reviewer Feedback (Max Pfefferkorn) |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **ACT-01** | `[x]` | 61 / 47 | 05_mpc_framework.tex:L470-478 | Margin note next to `Candidate MPC with Grouped Activation Decisions` / Eq. (5.17) (`The controller partitions the horizon into B contiguous, nonempty blocks...`) | Just to make this clear: The controller selects the partitioning; this is not given a priori? And the blocks do not need to have the same lengths, correct?<br><br>Could you state this explicit here to make it unambiguous, please? |
| **SUMM-01** | `[x]` | 62 / 48 | 05_mpc_framework.tex:L527-529 | `The waveform cost gives us a different criterion for suppression,` in Section 5.5 (`\section{Chapter Summary}`) | This is very vague.<br>The summary should clearly summarize what options we have to design the controller and what these options do. Please rework the summary. |
| **SUMM-02** | `[x]` | 62 / 48 | 05_mpc_framework.tex:L529-533 | Margin note on Section 5.5 (`\section{Chapter Summary}`) | As said, generally rework. Also, the writing style is a little too casual here and could be more formal for ascientific text. |
