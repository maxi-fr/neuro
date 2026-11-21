# Thesis Review Feedback & Action Items (20/11 Review by Max Pfefferkorn)

> [!NOTE]
> Extracted from `Thesis_Max_Frankenberg_2026_11_20__MP.pdf` in `thesis/thesis-feedback/20_11_MP/`.
> The feedback was given to the draft at commit 4d795194c136d583d0d72b749b0b1c623959b4e0

## Overview & Statistics

| Metric | Count |
| :--- | :--- |
| **Total Review Comments & Action Items** | **11** |
| **Status: Open (`[ ]`)** | **6** |
| **Status: Resolved (`[x]`)** | **5** |
| **Primary Sections Reviewed** | Chapter 6 (`06_experiment_and_discussion.tex`, pp. 62–65), Chapter 7 (`07_conclusion_and_future_outlook.tex`, p. 66) |

---

## 1. Chapter 6: Controller Comparison & Sparse Stimulation (5 comments)

| ID | Status | Page (PDF / Doc) | Source File & Line(s) | Marked Text / Element | Reviewer Feedback (Max Pfefferkorn) |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **CL-01** | `[x]` | 78 / 62 | `chapters/06_experiment_and_discussion.tex:L469-470, L478-479` | Margin note next to `Both spectral controllers contained seizure spread on all ten reserved seeds and used \SIrange{43}{48}{\percent} less mean charge than threshold stimulation (Table~\ref{tab:controller_comparison}).` | Compared to the previous version, you do note mentino the uncontrolled and threshold stimulation cases anymore explicitly. Is there a reason? Have they been mentioned earlier and I onmly overlooked it? If not, I'd add one sentence each again to make the reader understand all the approaches listed in Table 6.4. But please check that again. As said, I'm right now scanning through it only very briefly. |
| **SPAR-01** | `[x]` | 80 / 64 | `chapters/06_experiment_and_discussion.tex:L550` | `Activation` in Table 6.5 `(b) Stimulation penalties` | Doe this mean that you use the grouped search method only without any additional sparsity-promoting cost on the input? So which is the underlying MPC formulation in (b)? |
| **SPAR-02** | `[x]` | 81 / 65 | `chapters/06_experiment_and_discussion.tex:L564` | Margin note next to Figure 6.8 caption (`Sum of absolute applied electrode currents on seed 7104 for each stimulation penalty.`) | Which controller? |
| **SPAR-03** | `[x]` | 81 / 65 | `chapters/06_experiment_and_discussion.tex:L556-558` | Margin note next to `penalty (w_{L_1}=0.60) was close to that with the quadratic penalty (\SI{4.95}{\percent} and \SI{4.91}{\percent}, respectively). The controller with the activation penalty alone contained spread in nine of the ten simulations...` | For which controller? Or do I get something wrong here? |
| **SPAR-04** | `[x]` | 81 / 65 | `chapters/06_experiment_and_discussion.tex:L570` | `(Figure~\ref{fig:sparse_current_traces}, bottom)` in `These pauses were interspersed between around 26 stimulation bouts per run (Figure~\ref{fig:sparse_current_traces}, bottom).` | I think you only explciitly reference the top and the bottom panels of Figure 6.8, but the two middle panels are left "implicit". |

---

## 2. Chapter 7: Conclusions and Outlook (6 comments)

| ID | Status | Page (PDF / Doc) | Source File & Line(s) | Marked Text / Element | Reviewer Feedback (Max Pfefferkorn) |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **CONC-01** | `[ ]` | 82 / 66 | `chapters/07_conclusion_and_future_outlook.tex:L3-4` | `Focal seizures can spread from the epileptogenic zone and recruit other brain regions. We investigated` | Too fast. The conmclusion should start again with one sentence introducing the problem/field of research. This is then already the particular problem that you are looking it, and you later describe with which methods. |
| **CONC-02** | `[ ]` | 82 / 66 | `chapters/07_conclusion_and_future_outlook.tex:L5-6` | `model of left-temporal seizure propagation with scalp measurement and stimulation models. We used this plant to generate predictor training data and to evaluate the \ac{mpc} in closed-loop simulations.` | The first paragraph is generally good and well on point, but it is a little too short. You should add about one sentence specifying the models again. Otherwise, the following paragrpah can be hard to understand since you start talking about waveform and spectral formulations. |
| **CONC-03** | `[ ]` | 82 / 66 | `chapters/07_conclusion_and_future_outlook.tex:L8-15` | Margin note next to summary paragraphs 2–3 | As said, the summary parts reads well already, but is a little short. Furthermore, the achievements of your thesis are a little "hidden". That is that you say along all the other things "that worked". This is technically not wrong, but you should try to highlight that a bit more. That does not mean to overstate what you did or overload the summary with all the numbers you discussed earlier, but just to rephrase a little bit such it gets immediately clear what the main outcomes/main messages are. So this is not a content problem but a phrasing flaw. |
| **CONC-04** | `[ ]` | 82 / 66 | `chapters/07_conclusion_and_future_outlook.tex:L13` | `The affine \ac{dmdc}` in `The affine \ac{dmdc} predictor achieved the lowest mean log-power prediction error among the spectral predictors tested.` | Here as well: It sounds like the reader should know it, but it is good practice to "introduce" it again in the conclusions (see earlier comment in the first apragraph) |
| **CONC-05** | `[ ]` | 82 / 66 | `chapters/07_conclusion_and_future_outlook.tex:L14-15` | `and delivered less charge than the baseline threshold-triggered stimulation controller. Finally, we showed that we could promote sparse stimulation by penalizing absolute electrode currents or by explicitly penalizing the intervals in which stimulation was enabled.` | extend a little bit to include a few more details (at least what you proposed should be explciti again). |
| **CONC-06** | `[ ]` | 82 / 66 | `chapters/07_conclusion_and_future_outlook.tex:L23-24` | Note at end of Chapter 7 (below `electrode montages.`) | I'd suggest to add one more point here (the elephant in the room): the computational burden. You could discuss if more efficient formulations are possible, more efficient solvers exist, or if one should think about approximating the controller, e.g., by learning the resulting feedback map again with an easy-to-evaluate approach (e.g., a NN policy). |
