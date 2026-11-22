# Pre-Submission Thesis Audit Checklist

Mark the items you want fixed with `[x]` (and add inline notes where a choice is needed). All already fixed items get `[o]` and items marked for clarification `[c]`.

---

## CRITICAL — Must Fix Before Submission

- [o] **C1 `[TYPO + LOGIC]`** `chapters/06_experiment_and_discussion.tex:518-519`
  - **Issue:** `"The bonmin sovler failed on seven updates, possibly contributing to this worse performance."` Two typos (`bonmin` uncapitalized, `sovler` -> `solver`), and `"this worse performance"` follows a sentence about **homotopy** (which had the lowest burden).
  - **Fix:** State Bonmin's higher mean burden (`5.06%`), final count (`4.2`), and slower solve time (`1.64 s`) before noting its seven failed updates.

- [o] **C2 `[INCONSISTENCY]`** `chapters/03_biological_plant_modeling.tex:308-309`
  - **Issue:** Promises `"with its detailed motivation and sensitivity reported in the appendix"` for the 5 mV recruitment criterion, but `08_appendix.tex` has no such section.
  - **Fix:** Remove `, with its detailed motivation and sensitivity reported in the appendix` (or add the appendix section).
  - **Choice:** *(remove promise / add appendix)*

- [o] **C3 `[INCONSISTENCY]`** `chapters/03_biological_plant_modeling.tex:427-428`
  - **Issue:** Promises `"The search procedure and montage comparison are reported in the appendix."` No such section exists in `08_appendix.tex`.
  - **Fix:** Delete that sentence and summarize the selection criterion inline (or add the appendix section).
  - **Choice:** *(remove promise / add appendix)*

- [ ] **C4 `[CITATION / FACTUAL]`** `chapters/03_biological_plant_modeling.tex:304-305`
  - **Issue:** Claims that in `Yu2024`, `"recruitment spreads through most of the left hemisphere while not extending to the right hemisphere."` In `Yu2024` (Sec. 3.1, Fig. 3a), uncontrolled left-temporal discharges spread across the left hemisphere **and extend into the right hemisphere**.
  - **Fix:** Accurately describe `Yu2024` and clarify that $(K,\sigma)$ is calibrated here to target left-hemisphere spread within $\approx\SI{10}{\second}$.

- [o] **C5 `[CITATION / PDF RENDER]`** `literature.bib:720`
  - **Issue:** `Buzs{'a}ki, Gy{"o}rgy` is missing backslashes and renders literally as `G. Buzs’aki` in the PDF (`[20]`).
  - **Fix:** Change to `Buzs{\'a}ki, Gy{\"o}rgy`.

- [o] **C6 `[CITATION / PDF RENDER]`** `literature.bib:635-641`
  - **Issue:** `Loshchilov2019` (ICLR 2019) is typed as `@Online`, so `biblatex-ieee` suppresses `year = {2019}` completely in `[50]`.
  - **Fix:** Change to `@InProceedings` with `booktitle = {International Conference on Learning Representations (ICLR)}` and `year = {2019}`.

---

## HIGH — Strongly Recommended Before Submission

### Logic, Math, Figures & Tables

- [o] **H1 `[FIGURE]`** `figures/03_biological_plant_modeling/plant_overview.tex:68-74` & `plant_overview_network.tex:3-178`
  - **Issue:** The 76-region connectome in `plant_overview_network.tex` is rotated 90° relative to `pics/brain` (hemispheres split along $y=0$ instead of $x=0$, so the interhemispheric fissure runs horizontally across the vertically split brain outline).
  - **Fix:** Rotate/swap coordinates in `plant_overview_network.tex` so hemispheres align with $\pm x$, and re-verify the `r69` callout.

- [ ] **H2 `[FIGURE / SYMBOL COLLISION]`** `figures/03_biological_plant_modeling/plant_overview.tex:62-63`
  - **Issue:** Scalp currents are labeled `$+I$` / `$-I$`, colliding with the constant excitatory background drive $I = \SI{90}{\per\second}$ throughout Chapter 3.
  - **Fix:** Replace `$+I$` / `$-I$` with `$+u$` / `$-u$` (or `$u_1$` / `$u_2$`).

- [ ] **H3 `[FIGURE / CAPTION MISMATCH]`** `chapters/03_biological_plant_modeling.tex:274` & `figures/03_biological_plant_modeling/excitability_regimes.py:86,90`
  - **Issue:** Caption says Welch PSD uses `16,000` samples, but the script uses `nperseg=2**14` (`16,384`). The script also labels y-axes `$y_\mathrm{LFP}$` instead of `$y_\mathrm{reg}$`.
  - **Fix:** Update caption to `16,384` (or `$2^{14}$`) and update y-axis label in `excitability_regimes.py` to `$y_\mathrm{reg}$` (regenerate PDF).

- [ ] **H4 `[LOGIC / MATH]`** `chapters/05_mpc_framework.tex:472,494,498` vs. `chapters/08_appendix.tex:476-477`
  - **Issue:** Ch. 5 assumes $B$ equal blocks of $H/B$ intervals and scores $\frac{w_\delta}{B}\sum_{b=1}^B d_b$. Appendix D uses $H=17$ split into unequal blocks `(4, 4, 4, 5)` and charges $w_\delta/17$ per enabled hold ($\frac{w_\delta}{H}\sum_b |\mathcal B_b|d_b$).
  - **Fix:** Generalize Ch. 5 to arbitrary contiguous blocks $\mathcal B_1,\ldots,\mathcal B_B$ with penalty $\frac{w_\delta}{H}\sum_{b=1}^B |\mathcal B_b|d_b$ (noting equality with $\frac{w_\delta}{B}\sum d_b$ when all blocks have length $H/B$).

- [ ] **H5 `[INCONSISTENCY]`** `chapters/06_experiment_and_discussion.tex:471` vs. `chapters/06_experiment_and_discussion.tex:549`
  - **Issue:** Table 6.3 (`5.04%` burden, `23.98 mAs`, `3.9` final count) and Table 6.4(b) (`4.91%`, `25.40 mAs`, `3.5`) disagree for the same Quadratic ($L_2$) spectral DMDc controller on the same 10 reserved seeds, because Sec. 6.3.5 starts control at `\SI{0.66}{\second}` (only mentioned in Appendix D:441) instead of `\SI{1.38}{\second}`.
  - **Fix:** Explicitly state the `\SI{0.66}{\second}` start time in Sec. 6.3.5 and/or Table 6.4's caption.

- [x] **H6 `[INCONSISTENCY]`** `chapters/08_appendix.tex:193-199` vs. `chapters/08_appendix.tex:202-230`
  - **Issue:** Two consecutive paragraphs in Appendix B.3 report different metrics for 0.60 s-horizon IPOPT spectral DMDc on the same 10 reserved seeds (`0.967 s`, `5.00%`, `25.61 mAs` vs. Table B.2's `618.0 ms`, `5.01%`, `25.74 mAs`) with no explanation.
  - **Fix:** State what differed between the two runs.
  - **Note:** pick whatever number fits the storyline better and update the other table.

- [o] **H7 `[LOGIC / MATH]`** `chapters/08_appendix.tex:444`
  - **Issue:** `"Off time was measured after release from applied holds whose electrode currents were all at or below \SI{1}{\micro\ampere}"` — missing `"absolute"` (negative return currents up to `-2 mA` are $\le 1\,\mu\mathrm{A}$), and `"after release"` is undefined codebase jargon.
  - **Fix:** `"Off time was measured after control began from applied holds whose absolute electrode currents were all at or below \SI{1}{\micro\ampere}, excluding fallback holds."`

- [ ] **H8 `[INCONSISTENCY]`** `chapters/05_mpc_framework.tex:230`
  - **Issue:** Table 5.1 counts $(H+1)\ndim{h}+H\ndim{u}$ decision variables for multiple shooting (including $\mathbf{h}_{0|k}$) but only $H\ndim{h}$ equalities, omitting the $\ndim{h}$ initial-condition equalities $\mathbf{h}_{0|k}=\mathbf{h}_k$.
  - **Fix:** List $(H+1)\ndim{h}$ equalities (renaming column to "Equality constraints") or exclude $\mathbf{h}_{0|k}$ as a fixed parameter ($H(\ndim{h}+\ndim{u})$ variables).

- [o] **H9 `[INCONSISTENCY]`** `chapters/03_biological_plant_modeling.tex:299`
  - **Issue:** Says conduction speed $c$ determines tract delays `through~\eqref{eq:delay_quantisation}`, but $c$ appears in `\eqref{eq:physical_delay}`.
  - **Fix:** Change to `through~\eqref{eq:physical_delay} and~\eqref{eq:delay_quantisation}`.

- [o] **H10 `[INCONSISTENCY]`** `chapters/07_conclusion_and_future_outlook.tex:11`
  - **Issue:** Says waveform MPC tracks `"a healthy reference waveform"` (it tracks constant healthy channel means $\bar{\mathbf y}_{\mathrm{healthy}}$) and spectral MPC penalizes excess log-power relative to `"a healthy reference spectrum"` (a 90th-percentile envelope).
  - **Fix:** `"We formulated waveform-based \ac{mpc} to track the healthy channel means and spectral \ac{mpc} to penalize excess log-power above a healthy spectral envelope."`

- [c] **H11 `[INCONSISTENCY]`** `chapters/07_conclusion_and_future_outlook.tex:23`
  - **Issue:** Frames specialized QP solvers for linear DMDc purely as future work, ignoring Appendix B's OSQP benchmark (`34.7 ms` median, `93.2%` within `60 ms`).
  - **Fix:** Acknowledge the Appendix B OSQP result for linear spectral DMDc and focus future work on Koopman-extended predictors and specialized MIQP solvers for sparse stimulation.

- [o] **H12 `[INCONSISTENCY]`** `chapters/08_appendix.tex:32-36`
  - **Issue:** Table A.1 omits current history (`80, 240, 480 ms`) from the Waveform MLP and Waveform CNN search spaces, even though Tables A.4 and A.5 report it.
  - **Fix:** Add `current history 80, 240, or \SI{480}{\milli\second},` to both rows in Table A.1.

### Typos, Grammar & Unresolved Review Comments

- [o] **H13 `[TYPO + CITATION]`** `chapters/03_biological_plant_modeling.tex:176`
  - **Issue:** `"necessary avoid"` (missing `"to"`). Also `Ableidinger2017` studies Euler–Maruyama accuracy/distortion rather than Heun instability.
  - **Fix:** `"Such small steps are necessary to preserve numerical accuracy and stability of the stochastic Jansen-Rit integration~\cite{Ableidinger2017}."`

- [x] **H14 `[TYPO]`** `chapters/05_mpc_framework.tex:219`
  - **Issue:** `"They constitue fixed parameters which supply..."`
  - **Fix:** `"They constitute fixed parameters that supply..."`

- [o] **H15 `[GRAMMAR]`** `chapters/06_experiment_and_discussion.tex:75`
  - **Issue:** `"from the snapshot, while 0 means that knowing them..."` (singular antecedent, plural pronoun).
  - **Fix:** Replace `knowing them` with `knowing it` (or `knowing the snapshot`).
  - Note: that sentence feeld a bit AI-like. Can you suggest another sentence entirely?

- [o] **H16 `[GRAMMAR]`** `chapters/06_experiment_and_discussion.tex:483`
  - **Issue:** `"remained widespread until end of the run."` (missing `"the"`).
  - **Fix:** `"remained widespread until the end of the run."`

- [o] **H17 `[GRAMMAR]`** `chapters/06_experiment_and_discussion.tex:18`
  - **Issue:** `"even when the initial plant state would be known."` (Germanism).
  - **Fix:** `"even if the initial plant state were known."` (or `"even when the initial plant state is known."`).

- [x] **H18 `[UNRESOLVED COMMENT]`** `chapters/02_problem_formulation.tex:54`
  - **Issue:** Live review comment `% what is "it"? the version before was clearer` on ambiguous `"It is a change..."`.
  - **Fix:** Replace `It` with `Recruitment` and delete the comment. (Also remove leftover draft comments `% PROSE-11?` at line 39 and dead `[!ht]` comment at line 80.)
  - Note: take the comment seriously. "PROSE-11" comes from the /thesis-review skill

- [o] **H19 `[UNRESOLVED COMMENT]`** `chapters/06_experiment_and_discussion.tex:449`
  - **Issue:** Live review comment `% following comparisons. "reserved-seed" is overused...`.
  - **Fix:** Change `for the reserved-seed comparison` to `for the controller comparison` (or `for the following comparisons`) and delete the comment.

- [o] **H20 `[UNRESOLVED COMMENT]`** `chapters/08_appendix.tex:241`
  - **Issue:** Live review comment `% what stack? We implemented the mpc formulation in jax using IPOPT for optimization` on `"on the same JAX and IPOPT solver stack"`.
  - **Fix:** Replace with `"implemented in JAX and solved with IPOPT"` and delete the comment.

### LaTeX, Acronyms, Symbols & Bibliography

- [x] **H21 `[LATEX ERROR]`** `chapters/00_symbols_and_acronyms.tex:5-197` (`out/thesis.log:2641-2646`)
  - **Issue:** Four TeX errors (`ignored error: Infinite glue shrinkage found in box being split [9..12]`) on page breaks of the symbols `longtable`.
  - **Fix:** Fix vertical glue around `longtable` and recompile to verify zero errors in `out/thesis.log`.

- [o] **H22 `[LATEX / BABEL]`** `thesis.tex:228-230` & `chapters/task_description.tex:16-18`
  - **Issue:** German `task_description.tex` is compiled under English hyphenation (`babel` main language) and uses `\\` instead of blank lines for paragraph breaks.
  - **Fix:** Wrap `\input{chapters/task_description.tex}` in `\begin{otherlanguage}{ngerman}...\end{otherlanguage}` and replace `\\` paragraph breaks with blank lines.

- [o] **H23 `[ACRONYM]`** `chapters/05_mpc_framework.tex:198-203`
  - **Issue:** `narx` is never expanded anywhere in the thesis (all 9 uses are `\acs{narx}`).
  - **Fix:** Keep `\acs{narx}` in the heading (line 198) and use `\ac{narx}` at its first body mention (line 200/202).
  - note apart from the heading it should just use \ac everywhere. Thats the point of the package no?

- [o] **H24 `[ACRONYM]`** `chapters/01_introduction.tex:24` & `chapters/04_data_driven_predictor_design.tex:643`
  - **Issue:** `dmd` is written out manually in Ch. 1:24, causing its first `\ac{dmd}` expansion to occur inside a subsection heading (`4.3.3 Linear dynamic mode decomposition (DMD)-Based Predictor`).
  - **Fix:** Use `\ac{dmd}` in `01_introduction.tex:24` and `\acs{dmd}` in the heading at `04_data_driven_predictor_design.tex:643`.

- [o] **H25 `[ACRONYM]`** `chapters/04_data_driven_predictor_design.tex:647`, `chapters/08_appendix.tex:147,184`, `chapters/07_conclusion_and_future_outlook.tex:23`, `thesis.tex:124-127,179-182`
  - **Issue:** `qp` is written out manually 4 times before a single `\acf{qp}` in Ch. 7 (which renders `"quadratic program (QP) solvers"`). Also `nmpc` and `miqp` are declared in `thesis.tex` with 0 uses.
  - **Fix:** Use `\ac{qp}` starting at `04:647` (and fix `"quadratic programming"` in Ch. 7:23); remove unused `nmpc` and `miqp` declarations from `thesis.tex` (unless `miqp` is used in H11).

- [o] **H26 `[SYMBOLS]`** `chapters/00_symbols_and_acronyms.tex`
  - **Issue:** Missing prominent symbols used across chapters/appendices: $\mathcal{L}_{\mathrm{joint}}^{(\nu)}$, $w_{\mathrm{wave}}, w_{\mathrm{spec}}$, and Appendix B's $\spectral^{(0)}_{j|q}, \mathbf{D}_{j,\ell}, \widetilde{\mathcal{J}}_{\mathrm{spec}}, z_{j|q,c,m}$.
  - **Fix:** Add these entries to the appropriate blocks in `00_symbols_and_acronyms.tex`.

- [o] **H27 `[CITATION / BIB]`** `literature.bib:698-705` (`Pille2025`)
  - **Issue:** `@Misc` silently drops `journal = {bioRxiv}` in `biblatex-ieee`, so `[43]` omits the preprint venue.
  - **Fix:** Replace `journal = {bioRxiv}` with `howpublished = {bioRxiv preprint}`.
  - Note: for arxiv my reviwer wanted: use @article and use the field journal for this, i.e, journal = {arXiv:2406.09989}

- [ ] **H28 `[CITATIONS / ACCURACY]`** Five citation-accuracy issues across chapters:
  - [ ] **H28a** `chapters/02_problem_formulation.tex:18`: `Buzsaki2012` does not establish that individual neurons need not fire on every oscillation cycle (rephrase to match `Buzsaki2012` or cite Buzsáki & Draguhn 2004).
  - [o] **H28b** `chapters/03_biological_plant_modeling.tex:298,322`: Attribute $I=\SI{90}{\per\second}$ to `Yu2024` / `Wendling2002`, not `Jansen1995` (which used random $[120, 320]\,\mathrm{s}^{-1}$).
  - [x] **H28c** `chapters/03_biological_plant_modeling.tex:397`: Remove `Kunze2016` (it converts constant tDCS into an equivalent firing-rate input, the opposite of the sentence's contrast).
  - [o] **H28d** `chapters/03_biological_plant_modeling.tex:376`: Cite `\cite{SanzLeon2013,SanzLeon2015}` (62-channel EEG cap is in `SanzLeon2013`, not `SanzLeon2015`).
  - [ ] **H28e** `chapters/06_experiment_and_discussion.tex:74`: Clarify that Eq. (6.3) extends the 1D error-variance ratio of `DelSole2004` componentwise (`DelSole2004` argues against multivariate MSE ratios).

- [o] **H29 `[CODEBASE LEAK]`** `chapters/08_appendix.tex:439`
  - **Issue:** Stray Title-Case `Control Horizon` (lowercase everywhere else).
  - **Fix:** Change `Control Horizon` -> `control horizon`.

---

## AI-Language & Style Rewrites (Prioritized Top 7)

- [ ] **AI1 `[AI-LANGUAGE]`** `chapters/01_introduction.tex:18`
  - **Current:** `"In recent years, \ac{mpc} has emerged as a promising approach to closed-loop neurostimulation, leveraging dynamical models of neural activity to guide the selection of stimulation inputs~\cite{Chang2020,Fang2023,Liang2024,Steffen2025}."`
  - **Fix:** `"\Ac{mpc} uses dynamical models of neural activity to select closed-loop stimulation inputs~\cite{Chang2020,Fang2023,Liang2024,Steffen2025}."`

- [ ] **AI2 `[AI-LANGUAGE]`** `chapters/06_experiment_and_discussion.tex:403`
  - **Current:** `"Taken together, the results suggest that predictor training was robust enough to learn the features needed for containment from a representation that preserved short-term changes in spectral power across a broad frequency band without the need for the stabilizing effect of smoothing across frames."`
  - **Fix:** `"These results show that the predictor can learn containment-relevant features from unsmoothed, broad-band frames that preserve short-term changes in spectral power."`

- [ ] **AI3 `[AI-LANGUAGE]`** `chapters/01_introduction.tex:26`
  - **Current:** `"Rather than predicting neural activity itself, prediction models can instead target quantities derived from it."`
  - **Fix:** `"Prediction models can also forecast quantities derived from neural activity rather than the signal itself."`

- [ ] **AI4 `[AI-LANGUAGE]`** `chapters/04_data_driven_predictor_design.tex:804-805`
  - **Current:** Back-to-back `"forecasts over longer horizons ... may therefore"` repetition.
  - **Fix:** `"Direct spectral prediction avoids tracking individual oscillation phases and can remain accurate over longer horizons, letting the controller account for delayed stimulation effects."`

- [ ] **AI5 `[AI-LANGUAGE]`** `chapters/04_data_driven_predictor_design.tex:605-606`
  - **Current:** Two consecutive sentences ending in `, allowing ... to ...`.
  - **Fix:** `"Because $\mathbf{z}_k^{\mathrm{CNN}}$ preserves feature positions, the head can weight different time lags and frequency bins separately. Feeding the stimulation currents directly into this head lets their predicted effect depend on the extracted features."`

- [ ] **AI6 `[AI-LANGUAGE]`** `chapters/06_experiment_and_discussion.tex:480`
  - **Current:** `"Notably, the linear spectral \ac{dmdc} predictor was sufficient for containment on these seeds: ..."`
  - **Fix:** Drop `"Notably,"` and start directly with `"The linear spectral \ac{dmdc} predictor was sufficient..."`.

- [ ] **AI7 `[AI-LANGUAGE]`** Redundant section-ending bridges (`chapters/04_data_driven_predictor_design.tex:257,270,360,641`; `chapters/05_mpc_framework.tex:86,237,331`)
  - **Issue:** Mechanical `"The next [sub]section..."` / `"With X defined, the next..."` handoffs right above the next heading.
  - **Fix:** Delete these 7 redundant bridge sentences (and optionally vary the `"first ... then ... finally"` chapter openings in Ch. 3–5).

---

## MEDIUM — Worth Polishing

### Narrative, Logic & Math Precision

- [x] **M1 `[INCONSISTENCY]`** `chapters/05_mpc_framework.tex:432-441,534` vs. `chapters/06_experiment_and_discussion.tex:507-581`
  - **Issue:** Activation cap ($\sum \delta_{j|k} \le K$) is introduced in Ch. 5 but silently omitted in Ch. 6.
  - **Fix:** Add one sentence in Sec. 6.3.5 explaining why experiments focus on the activation penalty rather than the cap.
  - Note: I'd rather clarify this in the appendix calibration section.

- [ ] **M2 `[INCONSISTENCY]`** `chapters/01_introduction.tex:48,59-60` vs. `chapters/06_experiment_and_discussion.tex:448-476`
  - **Issue:** Ch. 1 frames the main closed-loop comparison around a shared spectral objective, whereas Table 6.3 compares spectral MPC against waveform-tracking MPC (`\mathcal{J}_{\mathrm{wave}}`).
  - **Fix:** Refine Ch. 1:48,59-60 to mention that waveform predictors are evaluated under both spectral and waveform-tracking objectives.

- [ ] **M3 `[INCONSISTENCY]`** `thesis.tex:246-247,260,263-264`
  - **Issue:** `"Both controllers"` / `"Beide Regler"` conflates 2 targets with 3 MPC controllers (DMDc, CNN, MLP). Also German line 260 has dangling definite article `"in den gemischt-ganzzahligen Formulierungen"`.
  - **Fix:** Use `"Controllers with either target"` / `"Die Regler beider Zielgrößen"` and drop `"den"` on line 260.

- [ ] **M4 `[INCONSISTENCY]`** `chapters/07_conclusion_and_future_outlook.tex:13`
  - **Issue:** Claims spectral loss `"made closed-loop suppression possible"` for waveform MPC, but waveform-only contained 2/5 dev seeds (App. D:396).
  - **Fix:** Change to `"enabled consistent closed-loop containment across development seeds"`.

- [ ] **M5 `[INCONSISTENCY]`** `chapters/07_conclusion_and_future_outlook.tex:14`
  - **Issue:** Unqualified claim that DMDc had the lowest spectral prediction error (true on Ch. 6's matched 6,400-origin benchmark: `1.5818` vs `1.5902` CNN, but CNN won in App. A's sweep: `1.5996` vs `1.6197`).
  - **Fix:** Qualify with `"on the matched offline benchmark, closely followed by the \ac{cnn} and \ac{mlp}"`.

- [o] **M6 `[LOGIC / MATH]`** `chapters/05_mpc_framework.tex:427`
  - **Issue:** Binary restriction $\delta_{j|k}\in\{0,1\}$ is omitted from the constraint list of `\eqref{eq:mpc_activation_penalty_problem}`.
  - **Fix:** Add `$\delta_{j|k}\in\{0,1\}$ for $j=0,\ldots,H-1$,` to line 427.

- [o] **M7 `[LOGIC / MATH]`** `chapters/05_mpc_framework.tex:462`
  - **Issue:** `"The minimum of a relaxed subproblem bounds from below..."` — only the *global* minimum is a valid lower bound.
  - **Fix:** Change `The minimum` -> `The global minimum`.

- [x] **M8 `[LOGIC / MATH]`** `chapters/05_mpc_framework.tex:506-524`
  - **Issue:** Homotopy section never writes the relaxed homotopy objective explicitly, nor states what happens if $\delta_{j|k}$ is still fractional when the continuation limit is reached.
  - **Fix:** State the relaxed objective and the fallback/post-processing rule briefly.

- [x] **M9 `[LOGIC / MATH]`** `chapters/05_mpc_framework.tex:178-193`
  - **Issue:** Nonpositive offsets ($s\le 0$) appear in `\eqref{eq:mpc_history_transition}` before being defined at line 217; also $\nhist{u}=0$ makes the input-history block empty.
  - **Fix:** Define the nonpositive-offset convention before `\eqref{eq:mpc_history_transition}` and note $\nhist{u}=0$.

- [ ] **M10 `[LOGIC / MATH]`** `chapters/03_biological_plant_modeling.tex:150-160`
  - **Issue:** Long-range coupling applies $S$ to unstimulated $y_{\mathrm{reg},i'}$, whereas local pyramidal firing uses $S(y_{\mathrm{reg},i}+s_i)$.
  - **Fix:** Add one clause noting that long-range coupling follows the standard TVB convention on unstimulated $y_{\mathrm{reg},i'}$.

- [x] **M11 `[LOGIC / MATH]`** `chapters/03_biological_plant_modeling.tex:403,418-425`
  - **Issue:** Units of $\mathbf{E}_{p,i}$ ($\si{\volt\per\metre\per\milli\ampere}$) are omitted, and $L_{\mathrm{stim},ip}=\lambda_{\mathrm{p}}\mathbf{E}_{p,i}^{\mathsf{T}}\mathbf{n}_i$ is written next to a sum over all $p=1,\ldots,\ndim{u}$ before stating Ex8 ($p=3$) has a zero column.
  - **Fix:** State the units of $\mathbf{E}_{p,i}$ and qualify the formula for $p\in\{1,2\}$.

- [ ] **M12 `[LOGIC / MATH]`** `chapters/03_biological_plant_modeling.tex:443-446`
  - **Issue:** Understates why $\mathbf{L}_{\mathrm{EEG}}$ and $\mathbf{L}_{\mathrm{stim}}$ are not transposes (mentions only Ex8, omitting different head models, spatial aggregations, and scaling).
  - **Fix:** Briefly mention the different head models (TVB surface vs. ROAST MNI152 FEM) and spatial aggregations in addition to Ex8.

- [x] **M13 `[INCONSISTENCY]`** `chapters/06_experiment_and_discussion.tex:222,224`
  - **Issue:** Discusses both spectral and waveform sweeps but only cites the waveform appendix tables, omitting `Table~\ref{tab:search_spectral_mlp}` and `Table~\ref{tab:search_spectral_cnn}`.
  - **Fix:** Cite both spectral and waveform tables in each sentence.
  - Note: how about citing the section instead of the tables?

- [o] **M14 `[LOGIC / WRITING]`** `chapters/06_experiment_and_discussion.tex:399`
  - **Issue:** Uses `"significantly"` without a statistical hypothesis test.
  - **Fix:** Replace `"while not reducing seizure burden significantly"` with `"without substantially reducing seizure burden"`.

- [o] **M15 `[LATEX]`** `chapters/06_experiment_and_discussion.tex:514`
  - **Issue:** Forward reference to Appendix B's `\eqref{eq:dmdc_current_rollout}` with redundant `"Equation~\eqref{...}"` (`latex-lint` `TYPO-11`, `MATH-03`).
  - **Fix:** Change `(Equation~\eqref{eq:dmdc_current_rollout})` to `(see Appendix~\ref{app:dmdc_comparison}, \eqref{eq:dmdc_current_rollout})` or `via~\eqref{eq:dmdc_current_rollout}`.

- [o] **M16 `[INCONSISTENCY]`** `chapters/08_appendix.tex:80`
  - **Issue:** `Last conv.` appears in Trial 25 of Table A.3 (`observable_cnn_trials.tex:31`) but is only defined later in Table A.5 (`08_appendix.tex:101`).
  - **Fix:** Define `Last conv.` in Table A.3's caption as well.

### Figures, Notation, LaTeX & Bibliography

- [ ] **M17 `[ORPHANED FIGURE]`** `figures/02_problem_formulation/stimulation_field.tex`
  - **Issue:** Never included or referenced anywhere in the thesis.
  - **Choice:** *(delete file / include in Ch. 2)*

- [ ] **M18 `[FIGURE]`** `figures/04_data_driven_predictor_design/cnn_architecture.tex:22-37`
  - **Issue:** Draws 8 time cells on both input and output tensors despite valid convolution reducing the number of time positions.
  - **Fix:** Draw fewer time cells (e.g., 6) on the output tensor to match valid-convolution shrinkage.

- [o] **M19 `[SYMBOL COLLISION]`** `chapters/06_experiment_and_discussion.tex:107,284` vs. `329`
  - **Issue:** Seizure state $s(t)$ / $s(t_i)$ collides with stimulation drive $s_i$ (reused on line 329).
  - **Fix:** Rename seizure state (e.g., to $\rho(t)$ or $z(t)$) or explicitly disambiguate.
  - Note: z is already used. maybe rho works. Check again

- [o] **M20 `[SYMBOL COLLISION]`** `chapters/03_biological_plant_modeling.tex:377-380`
  - **Issue:** EEG channel index $c$ collides with conduction speed $c$ in the same chapter.
  - **Fix:** Use another channel index (e.g., $m$ or $j$) in `$L^{\mathrm{surf}}_{mv}$`, `$L_{\mathrm{EEG},mi}$`.
  - Note: are you sure that m and j arent used otherwise?

- [o] **M21 `[NOTATION]`** `chapters/06_experiment_and_discussion.tex:43,45,50`
  - **Issue:** Uses `^\top` instead of `^{\mathsf{T}}` used everywhere else.
  - **Fix:** Replace `^\top` with `^{\mathsf{T}}`.

- [o] **M22 `[LATEX]`** `chapters/05_mpc_framework.tex:310,350`
  - **Issue:** Numbered equations `eq:mpc_current_effort` and `eq:mpc_l1_cost` are never referenced (`latex-lint` `MATH-02`).
  - **Fix:** Reference them in text or switch to `equation*`.

- [o] **M23 `[NOTATION / ACRONYM]`** `chapters/00_symbols_and_acronyms.tex:28,82`
  - **Issue:** Line 28 uses singular `\acs*{psp}` (`"three regional PSP"`); line 82 writes `"local field potential"` manually instead of `\acs*{lfp}`.
  - **Fix:** Change to `\acsp*{psp}` and `\acs*{lfp}`.

- [o] **M24 `[NOTATION]`** `chapters/02_problem_formulation.tex:88`, `04:102,440,535`, `05:11,172`
  - **Issue:** Section headings mix `\ac{...}` and `\acs{...}`.
  - **Fix:** Standardize on `\acs{...}` in all headings.

- [o] **M25 `[NOTATION]`** `notation.md:14,58,66` & `chapters/06_experiment_and_discussion.tex:284-288`
  - **Issue:** Minor drift between `notation.md` and thesis (`n_{\mathrm{past}\,o}` vs `n_{\mathrm{past},o}`, DMDc `prev/next` vs `\mathbf{X},\mathbf{X}_+`, uppercase `N_{\mathrm{win}}` vs lowercase `n_{\mathrm{win}}`).
  - **Fix:** Reconcile `notation.md` (and optionally rename `N_{\mathrm{win}}` -> `n_{\mathrm{win}}`).
  - Note: `n_{\mathrm{win}}` seems like it makes sense. lets try it

- [o] **M26 `[BIB FORMATTING]`** `literature.bib:32,1002,540,907-913,3-17`
  - **Issue:** Unbraced `{Teager--Kaiser}` (`Chourdaki2026`:32) and `{Part I}` (`DelSole2004`:1002) render lowercase in PDF; `Liang2024`:907 is an arXiv preprint typed as `@Article`; WHO `@Online` entries (3–17) hide `2024` in IEEE style; unused `Rowley2009`:540 has broken `Mezi'{c}`.
  - **Fix:** Brace `{Teager--Kaiser}` and `{Part I}`, fix `Mezi{\'c}`, retype `Liang2024` as `@Misc`/`@Online` with `eprint`, and optionally expose `2024` on the WHO entries.

- [ ] **M27 `[CITATIONS]`** Minor citation-precision tweaks:
  - [ ] **M27a** `chapters/01_introduction.tex:4-5`: Add `\cite{WHOEpilepsy2024}` to sentence 4 (`50 million people worldwide`) or merge with sentence 5.
  - [ ] **M27b** `chapters/01_introduction.tex:16`: Note `Berenyi2012` used skull-mounted electrodes (not scalp).
  - [ ] **M27c** `chapters/03_biological_plant_modeling.tex:122-128`: Note 76-region symmetric connectome is from the TVB software distribution (papers document 74-region asymmetric).
  - [ ] **M27d** `chapters/04_data_driven_predictor_design.tex:728`: Pair `Boyd2004` with `Proctor2016`/`Hoerl1970` for SVD truncation & ridge bias.
  - [ ] **M27e** `chapters/05_mpc_framework.tex:365`: Soften parallel to `Chouzouris2021,Salfenmoser2022,Salfenmoser2023` (spatial vs. temporal $L_1$).

### Writing & Style Polish

- [o] **M28 `[SPELLING]`** `chapters/03_biological_plant_modeling.tex:267,274`, `04:274`, `00_symbols_and_acronyms.tex:56,136`
  - **Issue:** British stragglers (`behaviour`, `Fibre`, `Centred`) in an American-English thesis.
  - **Fix:** Change to `behavior`, `Fiber`, `Centered`.

- [o] **M29 `[PUNCTUATION]`** `chapters/03_biological_plant_modeling.tex:128,269`, `05_mpc_framework.tex:57`, `06_experiment_and_discussion.tex:3`
  - **Issue:** Missing commas after `$w_{ii'}=w_{i'i}$`, `Within this model`, and `respectively`; missing hyphen in `scalp \ac{eeg} based`.
  - **Fix:** Add the missing commas and hyphen.

- [ ] **M30 `[WRITING]`** `chapters/01_introduction.tex:52`
  - **Issue:** `"The literature review informs..."` implies a dedicated Literature Review chapter that doesn't exist.
  - **Fix:** Change to `"The prior work reviewed above informs..."`.

- [o] **M31 `[WRITING]`** `chapters/03_biological_plant_modeling.tex:316` & `chapters/08_appendix.tex:243`
  - **Issue:** Implementation jargon `"saved uncontrolled seed-7000 run"` (`03:316`) and unexpanded `"JIT"` (`08:243`).
  - **Fix:** `"representative uncontrolled run (random seed 7000)"` and `"just-in-time compilation warmup solve"`.
  - Note: just remove the mention of the specific seed

- [c] **M32 `[WRITING]`** `chapters/06_experiment_and_discussion.tex:371,485`
  - **Issue:** Informal `robust` / `robustness` used for hyperparameter sensitivity while `robust MPC` appears technically in Ch. 7:27.
  - **Fix:** Replace with `least sensitive` / `low sensitivity`.

- [o] **M33 `[WRITING]`** `chapters/02_problem_formulation.tex:32,69-72,106,118`
  - **Issue:** Caption colon (`scales: Zoom`) vs. period in other figures; redundant `Consequently ... therefore`; `show` repeated 3× in lines 71–72; `or` -> `and` on line 118; `"dynamical model of the brain network"` -> `"dynamical prediction model"` on line 106.
  - **Fix:** Clean up these five small polish items in Ch. 2.
  - Note: i am very confused. Clarify

---

## LOW — Optional Final Polish

- [o] **L1 `[BIB]`** `literature.bib:148-157` — Remove template placeholder `Duden` (and optionally the other 18 unused entries).
- [ ] **L2 `[MATH CHECK]`** `chapters/08_appendix.tex:254` — Verify `71,601` vs. `71,583` variables for spectral CNN multiple shooting (differs by 18 = 9 vs. 8 past currents).
- [ ] **L3 `[INCONSISTENCY]`** `chapters/05_mpc_framework.tex:288` vs. `06:121` — State how many healthy runs were used for the closed-loop controller references (Fig. 5.3 says 9; Sec. 6.1.2 says 2).
- [o] **L4 `[LATEX]`** `figures/04_data_driven_predictor_design/mlp_architecture.tex:63` & `cnn_architecture.tex:159` — Replace `\mid` with `|` in `\widetilde{\mathbf{o}}_{1|k}` to match chapter equations.
- [o] **L5 `[LATEX]`** `figures/02_problem_formulation/scalp_interfaces.tex:34,46` & `tables/06_experiment_and_discussion/offline_predictor_selection.tex:4,6-10` — Replace bare `EEG`/`MLP`/`CNN`/`DMDc` with `\acs{...}` (and unify `One` vs. `2` in Table 6.1).
- [o] **L6 `[LATEX]`** `chapters/04_data_driven_predictor_design.tex:474,510,638-639` — Change `e^x` -> `\mathrm{e}^x`, move period inside `\end{cases}`, use `\num{37678}` and `\num{1611838}`.
- [o] **L7 `[WRITING]`** `chapters/05_mpc_framework.tex:452` (`a \ac{minlp}` -> `an \ac{minlp}`), `06:373` (`from 29 ... to four` -> `to 4`), `08:394-395` (`from 30 to three` -> `to 3`), `08:510` (`Off (s)` -> `Off time (s)`).
- [o] **L8 `[LATEX]`** `chapters/06_experiment_and_discussion.tex:22` — Remove stray space in `\cite{Matsumura2010, Schubert2002}`.
- [o] **L9 `[LATEX]`** `chapters/03_biological_plant_modeling.tex:332-335,341` — Replace ASCII `-` with `\textendash{}` in Table 3.1's Unit column.
