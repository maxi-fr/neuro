# Thesis Review Feedback & Action Items (29/10 Review by Max Pfefferkorn)

> [!NOTE]
> Extracted from `Thesis_Max_Frankenberg_2026_10_27__MP.pdf` and page visual annotations (`page-15.png` to `page-23.png`) in `thesis/thesis-feedback/29_10_MP/`.

## Overview & Statistics

| Metric | Count |
| :--- | :--- |
| **Total Review Comments & Annotations** | **36** |
| **Status: Open (`[ ] Open`)** | **36** |
| **Status: Resolved (`[x] Done`)** | **0** |
| **Primary Chapters Reviewed** | Chapter 1 (`01_introduction.tex`), pp. 1–2<br>Chapter 2 (`02_problem_formulation.tex`), pp. 3–6<br>Chapter 3 (`03_biological_plant_modeling.tex`), pp. 7–9 |

## 1. Introduction Structure & Narrative Flow (8 comments)

> [!TIP]
> **Key Structural Advice from Reviewer (INT-06):**
> Sharpen the storyline of the introduction into three progressive stages:
>
> 1. Neuropsychiatric diseases, prevalences, available drug-based treatment strategies and their limitations, motivating neurostimulation.
> 2. Neurostimulation introduction: approaches, mechanisms of action, intended effects, and literature overview of open-loop vs. closed-loop settings.
> 3. MPC as a promising closed-loop approach: high-level concept, literature review on MPC for brain stimulation, and clear motivation for the thesis questions.

| ID | Status | Page (PDF / Doc) | Source File & Line(s) | Marked Text / Element | Reviewer Feedback (Max Pfefferkorn) |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **INT-01** | `[ ] Open` | PDF p. 15 (Doc p. 1) | [`01_introduction.tex:L3`](file:///C:/Users/frank/closed-loop-neurostimulation/thesis/chapters/01_introduction.tex#L3) | `Some patients with epilepsy continue to experience seizures despite medication [1], [2]. Electrical` | This is too harsh for a first sentence. You start right away with a problem/an observation without introducing the field/area of research before. There could be a concise paragraph telling about neuropsychiatric disorders and their prevalence, that there exist drug-based treatments that, however, need to be supported. Epilepsy is one such case, and particularly the example that you consider here. And then you can start telling the reader facts about epilepsy and motivating brain stimulation as an alternative/additional treatment strategy (so also the following sentence should be smoothed a little). |
| **INT-02** | `[ ] Open` | PDF p. 15 (Doc p. 1) | [`01_introduction.tex:L4-6`](file:///C:/Users/frank/closed-loop-neurostimulation/thesis/chapters/01_introduction.tex#L4-L6) | `stimulation offers an additional approach to treatment by modulating neural activity...` | Smoothen the phrasing/presentation. You state an observation and only afterwards you make the essential point. I suggest connecting this better, e.g., by switching the order. First tell the reader about the setting, then about its outcomes and limitations. |
| **INT-03** | `[ ] Open` | PDF p. 15 (Doc p. 1) | [`01_introduction.tex:L7-9`](file:///C:/Users/frank/closed-loop-neurostimulation/thesis/chapters/01_introduction.tex#L7-L9) | `stimulation to ongoing neural activity during treatment [4], [5]. Closed-loop approaches instead use...` | Could be better connected. You need to establish the difference between responsive stimulation, closed-loop tES and feedback (mentioned as the third concept in the subsequent paragraph). |
| **INT-04** | `[ ] Open` | PDF p. 15 (Doc p. 1) | [`01_introduction.tex:L12`](file:///C:/Users/frank/closed-loop-neurostimulation/thesis/chapters/01_introduction.tex#L12) | `predict the response` | of ? *(Clarify what response is predicted, e.g., the brain network's dynamical response / neural activity response).* |
| **INT-05** | `[ ] Open` | PDF p. 15 (Doc p. 1) | [`01_introduction.tex:L12`](file:///C:/Users/frank/closed-loop-neurostimulation/thesis/chapters/01_introduction.tex#L12) | `candidate stimulation` | of ? *(Clarify what is stimulated / candidate stimulation sequences applied to what target).* |
| **INT-06** | `[ ] Open` | PDF p. 15 (Doc p. 1) | [`01_introduction.tex:L1-26`](file:///C:/Users/frank/closed-loop-neurostimulation/thesis/chapters/01_introduction.tex#L1-L26) | `Margin note across introductory paragraphs` | Generally, please sharpen the structure of the introduction a bit.<br>1. Neuropsychiatric diseases, prevalences, treatment strategies, Why neurostimulation?<br>2. Short introduction of neurostimulation: approaches, how does it work, what's the intended effect? Literature overview: "open-loop" vs. closed-loop settings.<br>3. MPC as a promising approach, what is MPC (high-level), literature review on MPC for brain stimulation -> this is already the main theme for these three paragraphs. It is totally fine to structure this a little. But try to make the connections more explicit and the transitions smoother. There must be a clear story line. |
| **INT-07** | `[ ] Open` | PDF p. 16 (Doc p. 2) | [`01_introduction.tex:L46-49`](file:///C:/Users/frank/closed-loop-neurostimulation/thesis/chapters/01_introduction.tex#L46-L49) | `The three contributions are: 1. Evaluation of MPC...` | Literature research to identify suitable MPC formulations and its components.<br>Set-up of a brain simulation representing neuronal dynamics as a connected graph, where each node embeds a Jansen-Rit dynamics. |
| **INT-08** | `[ ] Open` | PDF p. 16 (Doc p. 2) | [`01_introduction.tex:L50`](file:///C:/Users/frank/closed-loop-neurostimulation/thesis/chapters/01_introduction.tex#L50) | `2. Development of data-driven models that directly predict sequences of STFT-derived scalp EEG...` | Extend this a bit to make it more concrete. |

---

## 2. Problem Formulation & Control Engineering Setting (11 comments)

| ID | Status | Page (PDF / Doc) | Source File & Line(s) | Marked Text / Element | Reviewer Feedback (Max Pfefferkorn) |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **PF-01** | `[ ] Open` | PDF p. 17 (Doc p. 3) | [`02_problem_formulation.tex:L14`](file:///C:/Users/frank/closed-loop-neurostimulation/thesis/chapters/02_problem_formulation.tex#L14) | `At a chemical synapse, activity in the sending neuron changes the input received by another neuron.` | It'd be good to say that synapses are the things connecting cells before you explain how they work. |
| **PF-02** | `[ ] Open` | PDF p. 17 (Doc p. 3) | [`02_problem_formulation.tex:L27`](file:///C:/Users/frank/closed-loop-neurostimulation/thesis/chapters/02_problem_formulation.tex#L27) | `Using a neural mass` | model missing? *(Should read: "Using a neural mass **model** to describe an entire brain region...")* |
| **PF-03** | `[ ] Open` | PDF p. 18 (Doc p. 4) | [`02_problem_formulation.tex:L64`](file:///C:/Users/frank/closed-loop-neurostimulation/thesis/chapters/02_problem_formulation.tex#L64) | `The primary control objective is to keep recruitment outside the EZ as low as possible.` | Falls from the sky. First say that we want control and for what. -> smoothen the transition / sharpen the story line. |
| **PF-04** | `[ ] Open` | PDF p. 19 (Doc p. 5) | [`02_problem_formulation.tex:L82`](file:///C:/Users/frank/closed-loop-neurostimulation/thesis/chapters/02_problem_formulation.tex#L82) | `\section{Control Engineering Setting}` | Please think about a better title. |
| **PF-05** | `[ ] Open` | PDF p. 19 (Doc p. 5) | [`02_problem_formulation.tex:L68-81`](file:///C:/Users/frank/closed-loop-neurostimulation/thesis/chapters/02_problem_formulation.tex#L68-L81) | `Section 2.3 (Measurement and Stimulation)` | Can we have two schematics here, one for the EEG setting and the measurement, and one for the stimulation setting (no signal required)?<br>Maybe you could add 2-3 sentences about EEG signals then, how they look like, how to interpret them (on a high level). |
| **PF-06** | `[ ] Open` | PDF p. 19 (Doc p. 5) | [`02_problem_formulation.tex:L91-93`](file:///C:/Users/frank/closed-loop-neurostimulation/thesis/chapters/02_problem_formulation.tex#L91-L93) | `...according to a control objective and constraints [7]. A healthy reference specifies the desired non-seizure activity...` | Without (necessarily) introducing any maths, please be a little more specific in describing what the control objective is and how it is expressed. Do the details later in the respective chapter. |
| **PF-07** | `[ ] Open` | PDF p. 19 (Doc p. 5) | [`02_problem_formulation.tex:L90-101`](file:///C:/Users/frank/closed-loop-neurostimulation/thesis/chapters/02_problem_formulation.tex#L90-L101) | `The feedback controller uses MPC to select electrode currents by predicting the effects of candidate...` | I'd suggest to restructure this. Sharpen the description of MPC: It uses a dynamics model of the brain network to predict the brain's response to currents; the "quality" of stimulation signals is assessed through a cost, subject to constraints. I find it more naturally to then start with the prediction model, mention the cost function next, and conclude with the constraints (nothing is said about them in the current version). |
| **PF-08** | `[ ] Open` | PDF p. 19 (Doc p. 5) | [`02_problem_formulation.tex:L96-101`](file:///C:/Users/frank/closed-loop-neurostimulation/thesis/chapters/02_problem_formulation.tex#L96-L101) | `To obtain a prediction model for the MPC, simulations of the plant generate data consisting of applied currents...` | You miss to "introduce" (on a high level) the simulation environment. This is what links this part to the next chapter. |
| **PF-09** | `[ ] Open` | PDF p. 19 (Doc p. 5) | [`02_problem_formulation.tex:L103`](file:///C:/Users/frank/closed-loop-neurostimulation/thesis/chapters/02_problem_formulation.tex#L103) | `Recruitment is assessed from regional activity in the plant, whereas control decisions rely on...` | I'd mention this earlier. |
| **PF-10** | `[ ] Open` | PDF p. 19 (Doc p. 5) | [`02_problem_formulation.tex:L101-104`](file:///C:/Users/frank/closed-loop-neurostimulation/thesis/chapters/02_problem_formulation.tex#L101-L104) | `Recruitment is assessed from regional activity in the plant, whereas control decisions rely on the available measurements and the identified prediction model.` | Include this earlier, where it fits and make the connections explicit. |
| **PF-11** | `[ ] Open` | PDF p. 19 (Doc p. 5) | [`02_problem_formulation.tex:L103-105`](file:///C:/Users/frank/closed-loop-neurostimulation/thesis/chapters/02_problem_formulation.tex#L103-L105) | `measurements and the identified prediction model. Figure 2.2 shows the resulting feedback loop and the offline identification procedure. The subsequent chapters specify the biophysical plant, prediction model identification and the predictive control formulation.` | Too much. Use a dedicated transition to the next chapter. The "linker" is the brief introduction of the plant model already in this section. |

---

## 3. Biophysical Plant Modeling (3 comments)

| ID | Status | Page (PDF / Doc) | Source File & Line(s) | Marked Text / Element | Reviewer Feedback (Max Pfefferkorn) |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **BIO-01** | `[ ] Open` | PDF p. 21 (Doc p. 7) | [`03_biological_plant_modeling.tex:L8-11`](file:///C:/Users/frank/closed-loop-neurostimulation/thesis/chapters/03_biological_plant_modeling.tex#L8-L11) | `In Section 3.1... Section 3.2... Section 3.3... Section 3.4.` | In Chapter 2, you didn't use section numbers. Here, you do. This is inconsistent. Use one variant throughout the entire thesis. |
| **BIO-02** | `[ ] Open` | PDF p. 21 (Doc p. 7) | [`03_biological_plant_modeling.tex:L4-6`](file:///C:/Users/frank/closed-loop-neurostimulation/thesis/chapters/03_biological_plant_modeling.tex#L4-L6) | `The simulation setting established in Chapter 2 requires a plant that represents seizure recruitment... The chapter is organized into four sections.` | Consider extending this by 2-3 sentences, making the connection and the purpose of this chapter more explicit. |
| **BIO-03** | `[ ] Open` | PDF p. 21 (Doc p. 7) | [`03_biological_plant_modeling.tex:L28-30`](file:///C:/Users/frank/closed-loop-neurostimulation/thesis/chapters/03_biological_plant_modeling.tex#L28-L30) | `Each regional node i contains three interacting populations...` | Reference to a figure would be good to make this unambiguous and immediately understandable, because you "jump" between the terms "node", "region", and "cortical macro-column". You could also be more consistent with the terminology instead to make this immediately clear. |

---

## 4. Figures, Plots & Layout (3 comments)

| ID | Status | Page (PDF / Doc) | Source File & Line(s) | Marked Text / Element | Reviewer Feedback (Max Pfefferkorn) |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **FIG-01** | `[ ] Open` | PDF p. 22 (Doc p. 8) | [`03_biological_plant_modeling.tex:L12-20`](file:///C:/Users/frank/closed-loop-neurostimulation/thesis/chapters/03_biological_plant_modeling.tex#L12-L20) | `Figure 3.1 (plant_overview)` | Put it on the same page where it is referenced. The image should then be understandable from the introductory paragraph to Chapter 3 as well. |
| **FIG-02** | `[ ] Open` | PDF p. 23 (Doc p. 9) | [`jansen_rit_microcircuit.tex`](file:///C:/Users/frank/closed-loop-neurostimulation/thesis/figures/03_biological_plant_modeling/jansen_rit_microcircuit.tex) | `Figure 3.2 TikZ diagram font size` | Enlarge font size please. |
| **FIG-03** | `[ ] Open` | PDF p. 23 (Doc p. 9) | [`03_biological_plant_modeling.tex:L39`](file:///C:/Users/frank/closed-loop-neurostimulation/thesis/chapters/03_biological_plant_modeling.tex#L39) | `...potential summation and sigmoid conversion lie inside the receiving population.` | Add symbols. *(Add corresponding mathematical symbols to the caption of Figure 3.2, e.g. $\sum$, $S(\cdot)$, $h(t)$, etc.).* |

---

## 5. Citation Style & Reference Redundancy (10 comments)

> [!IMPORTANT]
> **Thesis-Wide Citation Policy (from CIT-01 & CIT-09):**
>
> 1. **Numeric citation format only:** Avoid "Author et al. [X]" in running text unless grammatically unavoidable. Do not start sentences with author names or citations. Use pure bracket references `[X]` integrated naturally into the sentence.
> 2. **No consecutive duplicate citations:** Do not cite the same reference in two adjacent sentences without another reference in between. Cite the reference once at the end or integrate the thoughts into a single sentence.

| ID | Status | Page (PDF / Doc) | Source File & Line(s) | Marked Text / Element | Reviewer Feedback (Max Pfefferkorn) |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **CIT-01** | `[ ] Open` | PDF p. 15 (Doc p. 1) | [`01_introduction.tex:L14`](file:///C:/Users/frank/closed-loop-neurostimulation/thesis/chapters/01_introduction.tex#L14) | `by Fröhlich and Jezernik [9]` | Don't use this citation style. [X] only. If necessary, rework the sentence to make this sound more natural. |
| **CIT-02** | `[ ] Open` | PDF p. 15 (Doc p. 1) | [`01_introduction.tex:L15`](file:///C:/Users/frank/closed-loop-neurostimulation/thesis/chapters/01_introduction.tex#L15) | `Chang et al. [8]` | [Struck out author name; paired with CIT-01: use bracket reference only]. |
| **CIT-03** | `[ ] Open` | PDF p. 15 (Doc p. 1) | [`01_introduction.tex:L16`](file:///C:/Users/frank/closed-loop-neurostimulation/thesis/chapters/01_introduction.tex#L16) | `Liang et al. [10] use` | Rework sentence, don't start with a reference. |
| **CIT-04** | `[ ] Open` | PDF p. 15 (Doc p. 1) | [`01_introduction.tex:L19`](file:///C:/Users/frank/closed-loop-neurostimulation/thesis/chapters/01_introduction.tex#L19) | `Truong et al. [11]` | [Struck out author name; paired with CIT-01: use bracket reference only]. |
| **CIT-05** | `[ ] Open` | PDF p. 15 (Doc p. 1) | [`01_introduction.tex:L21`](file:///C:/Users/frank/closed-loop-neurostimulation/thesis/chapters/01_introduction.tex#L21) | `Fang and Yang [12]` | [Struck out author name; paired with CIT-01: use bracket reference only]. |
| **CIT-06** | `[ ] Open` | PDF p. 15 (Doc p. 1) | [`01_introduction.tex:L22`](file:///C:/Users/frank/closed-loop-neurostimulation/thesis/chapters/01_introduction.tex#L22) | `Steffen and Cannon [13]` | [Struck out author name; paired with CIT-01: use bracket reference only]. |
| **CIT-07** | `[ ] Open` | PDF p. 16 (Doc p. 2) | [`01_introduction.tex:L35`](file:///C:/Users/frank/closed-loop-neurostimulation/thesis/chapters/01_introduction.tex#L35) | `Yu et al. [19]` | [Struck out author name; paired with CIT-01: use bracket reference only]. |
| **CIT-08** | `[ ] Open` | PDF p. 18 (Doc p. 4) | [`02_problem_formulation.tex:L52-53`](file:///C:/Users/frank/closed-loop-neurostimulation/thesis/chapters/02_problem_formulation.tex#L52-L53) | `...its initial territory [28]. The resulting manifestations depend on the neural systems involved [28].` | Don't put the same reference in two subsequent sentences; citing the reference once is sufficient. |
| **CIT-09** | `[ ] Open` | PDF p. 19 (Doc p. 5) | [`02_problem_formulation.tex:L73-75`](file:///C:/Users/frank/closed-loop-neurostimulation/thesis/chapters/02_problem_formulation.tex#L73-L75) | `...tissues of the head [18]... available measurements is therefore indirect [18].` | Please check this again throughout the whole thesis: If you have the same reference in subsequent sentences, without another reference in between, this is redundant. Cite the reference only once. It'd be fine if you had Sentence 1 [18], Sentence 2 [19], Sentence 3 [18] or first_part_sentence_1 [18], second_part_sentence_1 [19]. Sentence 2 [18]. |
| **CIT-10** | `[ ] Open` | PDF p. 19 (Doc p. 5) | [`02_problem_formulation.tex:L91-94`](file:///C:/Users/frank/closed-loop-neurostimulation/thesis/chapters/02_problem_formulation.tex#L91-L94) | `...objective and constraints [7]... become available [7].` | [Highlighted duplicate citation [7] in subsequent sentences within the same paragraph; paired with CIT-09]. |

---

## 6. Mathematical Typography & Capitalization (1 comment)

| ID | Status | Page (PDF / Doc) | Source File & Line(s) | Marked Text / Element | Reviewer Feedback (Max Pfefferkorn) |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **TYPO-01** | `[ ] Open` | PDF p. 15 (Doc p. 1) | [`01_introduction.tex:L12`](file:///C:/Users/frank/closed-loop-neurostimulation/thesis/chapters/01_introduction.tex#L12) | `Model Predictive Control (MPC)` | Model predictive control -> lowercase *(use lowercase in running text, reserving uppercase for acronym MPC)*. |
