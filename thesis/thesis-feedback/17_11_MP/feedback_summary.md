# Thesis Review Feedback & Action Items (17/11 Review by Max Pfefferkorn)

> [!NOTE]
> Extracted from `Thesis_Max_Frankenberg_2026_11_17__MP.pdf` in `thesis/thesis-feedback/17_11_MP/`.

## Overview & Statistics

| Metric | Count |
| :--- | :--- |
| **Total Review Comments & Action Items** | **39** |
| **Status: Open (`[ ]`)** | **10** |
| **Status: Resolved (`[x]`)** | **29** |
| **Primary Chapters Reviewed** | Chapter 6 (`06_experiment_and_discussion.tex`), pp. 50–66 |

---

## 1. General Remarks, Narrative Flow & Section Structure (14 comments)

| ID | Status | Page (PDF / Doc) | Source File & Line(s) | Marked Text / Element | Reviewer Feedback (Max Pfefferkorn) |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **STR-02** | `[x]` | 64 / 50 | `06_experiment_and_discussion.tex:L3` | `which scalp \ac{eeg} representations` | specify to remind the reader |
| **STR-03** | `[x]` | 64 / 50 | `06_experiment_and_discussion.tex:L11` | `each representation` in `how far ahead each representation remained predictable and how closely its suppression cost tracked regional recruitment.` | be more specific |
| **STR-04** | `[ ]` | 64 / 50 | `06_experiment_and_discussion.tex:L10-11` | Section 6.1 intro paragraph | This short paragraph is hghly repetitive with the one before. Keep both, but rearrange the content and level of details such that they build on each other rather than repeat each other |
| **STR-05** | `[ ]` | 64 / 50 | `06_experiment_and_discussion.tex:L13` | `\subsection{Potential Predictability}` | This section needs a clearer structure.<br>First tell the reader what you are doing and why. Then describe how you are doing it in a way that you smoothly connect anything that you introduce to the context. Do not only mention numbers, but tell the reader explicitly what that means. |
| **STR-06** | `[x]` | 64 / 50 | `06_experiment_and_discussion.tex:L17-18` | margin comment at the beginning of "Potential Predictability" subsection  | Clearly state the objective in the beginning. What do you want to test / investigate? This is a little hidden here and comes only relatively late |
| **STR-07** | `[x]` | 66 / 52 | `06_experiment_and_discussion.tex:L101-105` | `\subsection{Suppression Cost and Seizure State}` | One sentence: What is your goal here? |
| **STR-08** | `[x]` | 71 / 57 | `06_experiment_and_discussion.tex:L266` | `\subsection{Prediction Target}` | This section is better in the sense that you communicate early your objective. But I feel that it could be included more smoothly. You write what you do and how. But coming from the models for the two predictors already introduced in the earlier chapters, the "why" could be more explicit. |
| **STR-09** | `[ ]` | 72 / 58 | `06_experiment_and_discussion.tex:L300` | `\subsection{Predictor Architectures}` | I'm a little confused here. 6.2.1 already discussed the different architectures, provided an overview of tested settings, and concluded with the best models, right?<br>6.2.2 then slightly shifts the focus.<br>And what is the purpose of this section now? It feels like a summary of 6.2.1.<br>Honest question: What information is provided here that (i) is not already provided in 6.2.1 or (ii) can't be provided along the discussion in 6.2.1? If there is a reason for this section, make the reader understand it. |
| **STR-10** | `[ ]` | 73 / 59 | `06_experiment_and_discussion.tex:L317-347` | `\section{Solver and Transcription Benchmark}` (entire section) | While you write "benchmarked", it actually is a pretty isolated test, using a single setting. This does not imply a general trend. This entire section is therefore blown up and overstates a little bit in my opinion. I would condense this to one sentence "Initial tests showed that single-shooting is beneficial over multiple-shooting and NARX transcription in terms of computational demnd." or something like this and not make it an independent subsection. |
| **STR-11** | `[ ]` | 73 / 59 | `06_experiment_and_discussion.tex:L349` | `\section{Closed-Loop Seizure Suppression}` | try to tighten the phrasing a little |
| **STR-12** | `[ ]` | 75 / 61 | `06_experiment_and_discussion.tex:L426` | `\subsection{Stimulation Cost Weight}` | Also here, try to tighten the phrasing and streamline the presentation to what is necessary to get the story line. What is the main message that you want to communicate? Any presented results must support this main message clearly. |
| **STR-13** | `[ ]` | 77 / 63 | `06_experiment_and_discussion.tex:L477` | `\subsection{Effect of Spectral Geometry on Seizure Suppression}` | This is also a pretty long section on actually preliminary results. SO here as well: Do not present any comparison you have performed in detail. Highlight the important aspects; summarize less important ones concisely. Keep in mind that there is a story you want to tell: YOu want to show the reader in the end that and which MPC is working best for the task. So any result presented here and earlier must streamline the story toward this climax. If you add too many too unimportant details, trying to cover literally everything, you only distract the reader from what's really necessary. Present less important facts more concise or omit them from the main body; if necessary, add table with the numerical values in the appendix for the interested reader.<br><br>So generally in these sections here:<br>What is the objective of this section?<br>What are you doing here to investigate what you want to investigate? (Concise(!) description of setting, baselines, metrics, ...)<br>What is the outcome? (highlight msot important facts, summarize less important ones concisely) |
| **STR-14** | `[ ]` | 80 / 66 | `06_experiment_and_discussion.tex:L612` | `\subsection{Waveform Prediction Target}` | Here again: What is the intended key message and how does it support your story line? Build the section around it, highlight the important facts and connect them explicitly. Reduce the presentation of unimportant details. |

---

## 2. Section 6.1: Observable Predictability, Notation & Interpretation (13 comments)

| ID | Status | Page (PDF / Doc) | Source File & Line(s) | Marked Text / Element | Reviewer Feedback (Max Pfefferkorn) |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **OBS-01** | `[x]` | 64 / 50 | `06_experiment_and_discussion.tex:L21` | `the other \ac{eeg} representations` | the spectrum-based ones? |
| **OBS-02** | `[x]` | 64 / 50 | `06_experiment_and_discussion.tex:L32` | `1, 3, 7, and \SI{14}{\second}` | any time point should have a unit |
| **OBS-03** | `[x]` | 64 / 50 | `06_experiment_and_discussion.tex:L33` | `30 branches` in `Each of the 120 snapshots served as the starting point for 30 branches that ran for \SI{3}{\second} without stimulation.` | Why 30? What's the rationale behind that number? DOes this mean 30 branches per snapshot or 30 branches in total to which the snapshots are distributed somehow? |
| **OBS-04** | `[x]` | 64 / 50 | `06_experiment_and_discussion.tex:L39-45` | `For the segment indexed by q ending at t_q \ge 0, let \widetilde y_{\beta,c} be the standardized sample... The two 24-element vectors \mathbf a_c, \mathbf b_c \in \mathbb{R}^{24} used to compute connectivity differ by an offset of one sample (20 ms)...` | So far, it is not clear that/how they are used tom compute the connectivity. The symbols are defined without proper context / before they are actually used |
| **OBS-05** | `[x]` | 65 / 51 | `06_experiment_and_discussion.tex:L55-56` | `component a in branch b \in \{1,\ldots,n_{\mathrm{branch}}\} from snapshot \mu \in \{1,\ldots,n_{\mathrm{snap}}\}, and let \bar o_a^{(\mu)}(h) be its mean across branches. For n_{\mathrm{snap}} \in \mathbb{N} snapshots and n_{\mathrm{branch}} \in \mathbb{N} branches per snapshot, the estimates of noise variance within snapshots and signal variance between snapshots are` | Is there a better way to describe what these quantities are? This sentence is a little hard to read. |
| **OBS-06** | `[x]` | 65 / 51 | `06_experiment_and_discussion.tex:L58` | `\sum_\mu` in Eq. (6.2) | `\sum_{\mu=1}^{n_\mathrm{snap}}` |
| **OBS-07** | `[x]` | 65 / 51 | `06_experiment_and_discussion.tex:L59` | `\operatorname{Var}_b` in Eq. (6.2) | You should define what Var_b(.) means or avoid this symbol completely by writing out the variance |
| **OBS-08** | `[x]` | 65 / 51 | `06_experiment_and_discussion.tex:L60` | `\operatorname{Var}_\mu` in Eq. (6.3) | same here |
| **OBS-09** | `[x]` | 65 / 51 | `06_experiment_and_discussion.tex:L83` | `At the first spectral frame endpoint, \SI{60}{\milli\second} after the snapshot, potential predictability was 0.90 for spectral log-power and approximately 0.99 for both the waveform and functional connectivity.` | Ans what should it tell me? Good or bad? |
| **OBS-10** | `[x]` | 66 / 52 | `06_experiment_and_discussion.tex:L88` | `potential predictability was 0.46 for the functional connectivity feature, 0.64 for spectral log-power, and 0.70 for the waveform` | interpret/discuss these values |
| **OBS-11** | `[x]` | 66 / 52 | `06_experiment_and_discussion.tex:L92` | `The residual` in `The residual after subtracting this mean lost predictability much faster than either the trailing mean or spectral log-power` | Which residual? |
| **OBS-12** | `[x]` | 67 / 53 | `06_experiment_and_discussion.tex:L123` | `\bar R_{\mathrm{healthy},c,c'}` in `let \bar R_{\mathrm{healthy},c,c'} be the mean of (6.1)` | Was it introduced before? It seems to fall from the sky, like the reader should know that this is the measure central to this section. It is also left implicit why this measure is suitable. But keep in mind that a reader was not involved in your work. You need to tell this explicitly. That's related to what I comented on earlier already a few times: You start explaining details and only in the very end you tell the reader why and how. And this is not ideal for someone who has no prior knowledge. |
| **OBS-13** | `[x]` | 67 / 53 | `06_experiment_and_discussion.tex:L125-129` | `\mathcal J_{\mathrm{conn}}(q) = \frac{1}{\|\mathcal E\|}\sum_{(c,c')\in\mathcal E} (R_{q,c,c'} - \bar R_{\mathrm{healthy},c,c'})^2` | This cost was not introduced in the MPC chapter. The cost terms introduced there are not referred to here before the final evaluation. Make that explicit in the text as well as in the new first sentence telling the reader what this section is about. At the moment, it is hard to relate the presented results properly to your work. |

---

## 3. Sections 6.2 & 6.3: Offline Predictor Evaluation & Solver Benchmark (5 comments)

| ID | Status | Page (PDF / Doc) | Source File & Line(s) | Marked Text / Element | Reviewer Feedback (Max Pfefferkorn) |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **OFF-01** | `[x]` | 68 / 54 | `06_experiment_and_discussion.tex:L150` | `The predictors for the two remaining targets, waveform and spectral log-power, were compared at matched lookaheads using the same log-power representation that enters the suppression cost.` | I read this sentence several times but I think I still haven't understood it. |
| **OFF-02** | `[x]` | 68 / 54 | `06_experiment_and_discussion.tex:L150` | Same sentence (reply to **OFF-01**) | Is this relevant here when you anyway start with the prediction models? |
| **OFF-03** | `[x]` | 69 / 55 | `06_experiment_and_discussion.tex:L185-187` | Table 6.1 caption: `Predictor settings evaluated for each prediction target... For the \acp{cnn}, the number of feature channels in each successive convolutional layer was the previous layer's count multiplied by a factor between 0.5 and 1.` | Very unspecific, not mentioned in the table -> doesn't belong into the heading<br>Is it important? Yes -> include it properly; no: mention it in the main body |
| **OFF-04** | `[x]` | 72 / 58 | `06_experiment_and_discussion.tex:L293` | `The gap in mean spectral error persisted under spectral-only training, suggesting that using a waveform forecast as an intermediate step reduces the accuracy of the resulting log-power forecast.` | There must be a one-sentence-conclusion. What's the consequence of these investigations now? |
| **OFF-05** | `[x]` | 73 / 59 | `06_experiment_and_discussion.tex:L345` | [Struck out] `Multiple shooting can be preferable when long rollouts make single shooting poorly conditioned, since its dynamics constraints use only local derivatives (Section 5.2).` | Before you wrote that with multiple-shooting the solver didn't converge |

---

## 4. Section 6.4: Closed-Loop Seizure Suppression (7 comments)

| ID | Status | Page (PDF / Doc) | Source File & Line(s) | Marked Text / Element | Reviewer Feedback (Max Pfefferkorn) |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **CL-01** | `[x]` | 73 / 59 | `06_experiment_and_discussion.tex:L353-354` | `The closed-loop experiments test whether feedback from scalp \ac{eeg} can contain seizure spread while reducing the delivered charge. Each simulation ran for \SI{12}{\second} on the calibrated 76-region plant.` | phrasing |
| **CL-02** | `[x]` | 75 / 61 | `06_experiment_and_discussion.tex:L417-418` | `whether stimulation began at time zero or at \SI{1.38}{\second}. With the strongest \ac{ez}-directed current starting at \SI{1.38}{\second}, mean burden fell to \SI{4.46}{\percent}...` | Were tge different start times mentioned earlier? Were do they come from and why? |
| **CL-03** | `[x]` | 76 / 62 | `06_experiment_and_discussion.tex:L436-443` | Figure `fig:stimulation_cost_weight` (`closed_loop_effort_sweeps`) | Why do small weights lead to better suppression? Shouldn't a high weight prioritize the suppression objective over the input-related terms? |
| **CL-04** | `[x]` | 79 / 65 | `06_experiment_and_discussion.tex:L590-592` | `Their mean burdens were within the range associated with containment and residual focal activity in this plant. The spectral \ac{dmdc} controller reached this range with the linear predictor and the shortest tested horizon. The spectral \ac{cnn} controller used...` | This is the key result. Highlight. |
| **CL-05** | `[ ]` | 79 / 65 | `06_experiment_and_discussion.tex:L601-603` | `On seed 7104, recruitment accelerated shortly after \SI{2}{\second} under the waveform \ac{mlp} controller. Before that point, it supplied weaker negative drive to both \ac{pz} regions than either spectral controller, which allowed the seizure to spread. Once recruitment becomes widespread, the available stimulation is insufficient to restore containment in this run.` | I find this explanation interesting and important. Is there the chance to show this in a well-designed plot? |
| **CL-06** | `[x]` | 79 / 65 | `06_experiment_and_discussion.tex:L555` | `\subsection{Controller Comparison}` | earlier you stated somewhere, if I remember correctly, that you only carry the MLP waveform predictor forward to the closed-loop. But here you have considere also the spectral controllers. This is fine, and they should be here, but please check tht evalidity of your earlier statement again. |
| **CL-07** | `[x]` | 80 / 66 | `06_experiment_and_discussion.tex:L614-616` | `The controller comparison evaluated the selected waveform and spectral controllers as complete systems. To isolate choices within the waveform controller...` | Why only this one? |

---

## 5. Overarching Chapter 6 Feedback (1 comment)

| ID | Status | Page (PDF / Doc) | Source File & Line(s) | Marked Text / Element | Reviewer Feedback (Max Pfefferkorn) |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **OVR-01** | `[ ]` | - | `06_experiment_and_discussion.tex` (Chapter 6 overall) | Overarching comment | I have one general comment that summarizes many of the individual comments I gave: Chapter 6 feels a little lengthy. You present many results in relatively great detail. However, your story line and key messages get lost in all of these details. You need to build that chapter and the corresponding sections around the key message that you want to communicate. Aspects that directly support these key messages are important and should be presented in detail, but also explicitly connected to the story. Aspects that do not directly contribute can be presented less comprehensively as concise comments. I had the feeling you wanted to provide a complete overview of all the things you did. However, you should select the important things instead. Presenting everything just for the sake of completeness in the end only distracts the reader from the important findings. |
