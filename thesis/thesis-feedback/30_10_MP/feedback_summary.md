# Thesis Review Feedback & Action Items (30/10 Review by Max Pfefferkorn)

> [!NOTE]
> Extracted from `Thesis_Max_Frankenberg_2026_10_27__MP-1.pdf` and page visual annotations (`page-26.png` to `page-44.png`) in `thesis/thesis-feedback/30_10_MP/` excluding all that were covered by `thesis/thesis-feedback/29_10_MP/feedback_summary.md`.

## Overview & Statistics

| Metric | Count |
| :--- | :--- |
| **Total Review Comments & Action Items** | **29** |
| **Status: Open (`[ ]`)** | **27** |
| **Status: Resolved (`[x]`)** | **2** |
| **Primary Chapters Reviewed** | Chapter 3 (03_biological_plant_modeling.tex), pp. 12–18<br>Chapter 4 (04_data_driven_predictor_design.tex), pp. 19–30 |

---

## 1. Chapter 4 Narrative Flow & Top-Down Structure (5 comments)

| ID | Status | Page (PDF / Doc) | Source File & Line(s) | Marked Text / Element | Reviewer Feedback (Max Pfefferkorn) |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **STR-01** | `[x]` | 33 / 19 | 04_data_driven_predictor_design.tex:L15-22 | `Three families of data-driven models are investigated for these prediction tasks. Multilayer perceptrons (MLPs)... Convolutional neural networks (CNNs)... delay-embedded dynamic mode decomposition with control (DMDc)...` | After reading the introductory text here, I only had a glimpse on what comes next. But it seems like the learning is only dealt with in Section 4.2, which is why I find it too early to mention this here. |
| **STR-02** | `[x]` | 33 / 19 | 04_data_driven_predictor_design.tex:L10-14 | introduction paragraph 2 | What is missing here is how this relates to the task. Your overall objective is still to control the neuronal activity. Now, you introduce these two different prediction targets. But you don't close the loop why both of them could be relevant for the considered problem. You don't have to do this in great detail, but it'd help the reader to establish that connection. |
| **STR-03** | `[ ]` | 33 / 19 | 04_data_driven_predictor_design.tex:L29-31 | `with the spectral transformation specified in Section 4.1.2.` | I would either mention both or none explicitly with the section number |
| **STR-04** | `[ ]` | 38 / 24 | 04_data_driven_predictor_design.tex:L286-302 | `Section 4.1.4: Trade-Offs between Waveform and Spectral Prediction (and final MPC paragraph)` | It feels like this discussion is a little misplaced here. With some slight restructuring, it might better fit the beginning of the chapter. The problem is that it discusses differences between the two approaches while they have (in my opinion) not been sufficiently motivated in the beginning. Furthermore, the last paragraph jumps to MPC for a few sentences, but disrupts the actual story line. Right now, you go from the definition of the regression problem to MPC to how you solve the regression problem. Why is the MPC in between? |
| **STR-05** | `[ ]` | 40 / 26 | 04_data_driven_predictor_design.tex:L324-345 | `Section 4.2.1 (Input-Output Regression and Optimization) & Section 4.2.2 (Training Data Generation)` | I think 4.2.1 and 4.2.2 could be shortened and probably merged:<br>Define the regression problem and specify explicitly the use of NNs.<br>The training procedure is standard and could be tightened.<br>To train the network, you rely on a dataset constructed ...<br>Then you have the setting at hand and you can focus on the less standard / more tailored parts of the following sections. |

---

## 2. Mathematical Rigor, Observable Definitions & Notation (16 comments)

| ID | Status | Page (PDF / Doc) | Source File & Line(s) | Marked Text / Element | Reviewer Feedback (Max Pfefferkorn) |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **MATH-01** | `[ ]` | 34 / 20 | 04_data_driven_predictor_design.tex:L34 | `Let \mathbf{o}_k \in \mathbb{R}^{\ndim{o}} denote the observable neural activity at discrete time step k, where \ndim{o} is its dimension.` | Introduced before it is defined, which leaves it unclear here. |
| **MATH-02** | `[ ]` | 34 / 20 | 04_data_driven_predictor_design.tex:L34-36 | `...where \ndim{o} is its dimension... where \ndim{u} is the dimension... where \ndim{y} is its coordinate dimension...` | Avoid repetitions |
| **MATH-03** | `[ ]` | 34 / 20 | 04_data_driven_predictor_design.tex:L36 | `sample index r` in `multichannel EEG waveform sample at sample index r` | not introduced, what do they represent/mean? |
| **MATH-04** | `[ ]` | 34 / 20 | 04_data_driven_predictor_design.tex:L36 | `frame index q.` in `spectral power feature vector at frame index q.` | [Highlighted; paired with MATH-03] |
| **MATH-05** | `[ ]` | 34 / 20 | 04_data_driven_predictor_design.tex:L41-44 | `\mathbf{y}_r\text{, for waveform prediction, with } k=r` in `\eqref{eq:observable_cases}` | Why introduce 2 symbols? You could first define o_k and then explain the two possibilities, which would help you to avoid them, I think |
| **MATH-06** | `[ ]` | 34 / 20 | 04_data_driven_predictor_design.tex:L46 | `downsampled` in `...one sample of the downsampled waveform or one hop of the STFT.` | hasn't been mentioned before |
| **MATH-07** | `[ ]` | 34 / 20 | 04_data_driven_predictor_design.tex:L64 | `For \nhist{u}=0, the past-current block is empty.` | [Struck out text] |
| **MATH-08** | `[ ]` | 34 / 20 | 04_data_driven_predictor_design.tex:L76 | `step k` in `predicted observable one step ahead of step k, given the information available at that step.` | time |
| **MATH-09** | `[ ]` | 35 / 21 | 04_data_driven_predictor_design.tex:L126 | `acquired at frequency f_\mathrm{s}>0` | sampling freq.? |
| **MATH-10** | `[ ]` | 35 / 21 | 04_data_driven_predictor_design.tex:L126 | `with \nchan{}=\ndim{y} channels in total.` | Don't use two symbols to express the same quantity |
| **MATH-11** | `[ ]` | 35 / 21 | 04_data_driven_predictor_design.tex:L127 | `both expressed in samples, let` | [Highlighted text; paired with MATH-12] |
| **MATH-12** | `[ ]` | 35 / 21 | 04_data_driven_predictor_design.tex:L127 | `let q\in\mathbb N_0 index successive spectral frames` | a single index q can't index multiple (-> successive) frames |
| **MATH-13** | `[ ]` | 35 / 21 | 04_data_driven_predictor_design.tex:L136 | `The coefficients form the complex vector \mathbf{Y}_{q,c}\in\mathbb{C}^{\lfloor \nseg{}/2\rfloor+1}. For real-valued signals, these coefficients correspond to m=0,\ldots,\lfloor \nseg{}/2\rfloor, with frequencies` | ? |
| **MATH-14** | `[ ]` | 35 / 21 | 04_data_driven_predictor_design.tex:L131 | `w_\beta` in Eq. (4.5): `\sum_{\beta=0}^{\nseg{}-1}w_\beta y_{q\nhop{}+\beta,c}\exp\left(-\frac{2\pi\mathrm{i}m\beta}{\nseg{}}\right)` | Is this meant to be an entry of boldface w? If so, it isn't defined. If not, it isn't defined at all. |
| **MATH-15** | `[ ]` | 35 / 21 | 04_data_driven_predictor_design.tex:L124-142 | Section 4.1.2 margin note | This section has the problem that almost all symbols are introduced before even the first one is used. This makes it hard to follow what the different symbols mean. They should be better described and introduced only once needed.<br>Especially the high number of index/number variables (n_...), which are revisited also quite often in the subsequent sections, don't help. Please check which of them are really necessary, and if some of them are only duplicates. Try to improve the presentation in a way that helps the reader remember the meaning of these variables better. I know that you will need a lot of these variables, but especially in this case, you should assist the reader as much as possible. |
| **MATH-16** | `[ ]` | 37 / 23 | 04_data_driven_predictor_design.tex:L240-248 | Section 4.1.3 margin note | Very technical. It might be helpful to slightly reformulate the beginnings of the paragraph to make connections explicit and explain early on what follows next and why. |

---

## 3. Training Pipeline, Dataset Structure & Regularization (6 comments)

| ID | Status | Page (PDF / Doc) | Source File & Line(s) | Marked Text / Element | Reviewer Feedback (Max Pfefferkorn) |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **TRN-01** | `[ ]` | 40 / 26 | 04_data_driven_predictor_design.tex:L326 | `In gradient descent,` | gradient descent is way more general than this |
| **TRN-02** | `[ ]` | 40 / 26 | 04_data_driven_predictor_design.tex:L326 | `the neural-network parameters are updated` | sudden jump from parametrized mapping to neural networks |
| **TRN-03** | `[ ]` | 40 / 26 | 04_data_driven_predictor_design.tex:L340-344 | `During control, the predictor encounters histories and stimulation sequences beyond the training examples, so fitting those examples is useful only if the learned mapping generalizes to new data. Overfitting occurs when improved agreement with the training data comes at the expense of this ability [43]. Performance on separate validation data helps identify overfitting and guides model selection [43]. The next subsection describes how the trajectories for training and validation are generated.` | Auf einen Satz mit stärkerem Bezug zum relevanten Inhalt kürzen |
| **TRN-04** | `[ ]` | 41 / 27 | 04_data_driven_predictor_design.tex:L386 | `initial` in `Together with the initial history state from~\eqref{eq:predictor_history}, these form the dataset` | initial is misleading here; it might be interpreted as h_0 |
| **TRN-05** | `[ ]` | 41 / 27 | 04_data_driven_predictor_design.tex:L386-390 | Eq. (4.9): `\mathcal{D}_\mathrm{tr} = \{ ((\mathbf{h}_k^{(\nu)}, \mathbf{U}_k^{(\nu)}), \mathbf{O}_k^{(\nu)}) : \nu = 1, \dots, n_\mathrm{train} \}` | Might it, for later references, be better to introduce the data set immediately for the normalized quantities instead of for the original ones? Or to have a second set containing the standardized quantities? Please check this. |
| **TRN-06** | `[ ]` | 43 / 29 | 04_data_driven_predictor_design.tex:L470-472 | `The neural networks are trained with AdamW, which uses running estimates of the gradients and their squared values to adapt parameter updates and applies decoupled weight decay for regularization [45]. The networks are trained on shuffled mini-batches, with an initial learning-rate warm-up followed by cosine decay.` | This was, at least in parts, already mentioned earlier. It fits here better. But avoid repetitions |

---

## 4. Model Architectures, Activations & Solver Compatibility (2 comments)

| ID | Status | Page (PDF / Doc) | Source File & Line(s) | Marked Text / Element | Reviewer Feedback (Max Pfefferkorn) |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **ARCH-01** | `[ ]` | 44 / 30 | 04_data_driven_predictor_design.tex:L528-532 | Section 4.3 opening paragraph: `ReLU is widely used for neural-network training, but is not differentiable at zero. When a trained ReLU network is embedded in an optimization problem, its nonsmoothness can cause convergence difficulties for nonlinear solvers that assume differentiable problem functions [47]. In MPC, the predictor is evaluated within the optimization over future stimulation currents, so this issue concerns the control optimization even when network training succeeds. Tanh and softplus provide smooth mappings for this subsequent optimization.` | Shorten this paragraph and keep only a side comment. This is not in the focus of the work |
| **ARCH-02** | `[ ]` | 44 / 30 | 04_data_driven_predictor_design.tex:L533 | `the standardized history from~\eqref{eq:predictor_history} is concatenated` | (4.1) is not the standardized history |
