# Supplementary Pre-Submission Thesis Audit Checklist

Mark the items you want fixed with `[x]`, completed items with `[o]`, and items needing clarification with `[c]`. Item numbers continue from the [pre-submission audit checklist](pre_submission_audit_checklist.md).

---

## CRITICAL — Must Fix Before Submission

- [ ] **C7 `[MODEL / REPRODUCIBILITY]`** `chapters/03_biological_plant_modeling.tex:81,213,313,348` & `chapters/00_symbols_and_acronyms.tex:47-48`
  - **Issue:** The stated diffusion coefficient $A_i a\sigma$ gives 279.5, 292.4 and 309.6 for background, propagation-zone and epileptogenic-zone regions. The benchmark applies 280 directly to the fifth state in every region. The printed equation therefore adds about 11% more noise in the epileptogenic zone.
  - **Fix:** Give the common state-level diffusion coefficient and its units in the equations, table and symbol list. Check which convention earlier simulation campaigns used before extending the correction to them.
  - **Evidence:** [Benchmark configuration](../artifacts/dmdc_closed_loop_timing/base_simulation.yaml), revision `4de695c`, `src/neuro/plant/jansen_rit.py:165-169`.

- [c] **C8 `[FIGURE / TIMING]`** `chapters/03_biological_plant_modeling.tex:314,323` & `chapters/06_experiment_and_discussion.tex:284-290`
  - **Issue:** The caption says onset was detected at 0.5 s. That timestamp is the centre of the first 1 s window in a run of four above-threshold windows spaced 0.25 s apart. Establishing persistence requires later samples.
  - **Fix:** Label the plotted onset as retrospective. Distinguish centre timestamps in the recruitment trace from endpoint timestamps used to score burden, and check intervention times read from those traces.
  - **Evidence:** [Figure generator](../figures/03_biological_plant_modeling/plant_assignment_and_spread.py), revision `4de695c`, `spread_profile_from_lfp` and `SpreadProfile.from_ptp`.

- [ ] **C9 `[OUTCOME DEFINITION]`** `chapters/02_problem_formulation.tex:61` & `chapters/06_experiment_and_discussion.tex:284-290,345,478-482` & `chapters/08_appendix.tex:546`
  - **Issue:** The 10/10 spectral and 9/10 waveform containment claims require a per-run classification. Mean burden and final count do not identify transient spread or which regions were active. Some contained runs ended with four or five seizing regions despite the three-region epileptogenic zone.
  - **Fix:** State how each run was classified, including any threshold, region set and observation window. If classification came from inspecting traces, say so and give the paired per-seed outcomes. Six of the ten uncontrolled runs spread widely.

---

## HIGH — Strongly Recommended Before Submission

- [ ] **H30 `[MATH / DIMENSIONS]`** `chapters/04_data_driven_predictor_design.tex:124` & `chapters/05_mpc_framework.tex:271-275` & `chapters/06_experiment_and_discussion.tex:157` & `chapters/08_appendix.tex:303`
  - **Issue:** The one-sided 25-sample FFT has 13 bins including DC. The selected 1–25 Hz band contains 12 bin centres at 2, 4, …, 24 Hz. Summing over the full grid gives 806 coordinates across 62 channels, while the experiment reports 744.
  - **Fix:** Define the retained frequency-index set and use its cardinality in the observable dimension, spectral-cost normalization, QP and symbol list.

- [ ] **H31 `[EXPERIMENT DESIGN]`** `chapters/06_experiment_and_discussion.tex:153-160,237-246,261` & `thesis.tex:246,263`
  - **Issue:** The common 6,400 forecast origins come from the 60 ms-hold validation set used to select the spectral models. Waveform models were trained and selected on 20 ms holds. The benchmark therefore combines the effects of prediction target, current schedule and model selection.
  - **Fix:** Qualify the ranking in Chapter 6 and both abstracts as a comparison of mean log-power error under this protocol. The waveform MLP led at 60 ms, and the spectral predictors led from 120 ms onward.

- [ ] **H32 `[SOLVER / REPRODUCIBILITY]`** `chapters/06_experiment_and_discussion.tex:437,531` & `chapters/08_appendix.tex:440,488`
  - **Issue:** Three waveform-only runs had 48–61 failed solves each. The manuscript gives no applied-current rule for these failures. The sparse table also leaves readers unable to tell that zero-current fallback intervals are excluded from its off-time total.
  - **Fix:** State the executed fallback for each applicable controller and how each metric treats those intervals. In the benchmark MPC playback, failure discards the planned prefix and produces zero current until a successful replan. Make the table caption explicit about the off-time exclusion.

- [ ] **H33 `[BENCHMARK COUNTS]`** `chapters/08_appendix.tex:298-304`
  - **Issue:** The 69 comparisons comprise 55 NARX–single-shooting pairs (36 waveform, 19 spectral) and 14 multiple–single-shooting pairs. “Jointly converged” reads as three-way convergence on 69 cases. The following universal single-shooting claim also conflicts with the closed-loop OSQP experiment.
  - **Fix:** Say “69 pairwise comparisons against single shooting” and name the experiments that retained single shooting.

- [ ] **H34 `[MEASUREMENT LIMITATION]`** `chapters/03_biological_plant_modeling.tex:379` & `chapters/06_experiment_and_discussion.tex:298-299` & `chapters/07_conclusion_and_future_outlook.tex:32`
  - **Issue:** TP9 is used for sensing and stimulation. The observation equation includes only projected neural activity, and the benchmark sets independent sensor-noise standard deviation to zero.
  - **Fix:** State in the outlook that the simulations assumed artifact-free EEG during stimulation and that performance with stimulation artifacts and sensor noise remains untested.
  - **Evidence:** [Sensor configuration](../artifacts/dmdc_closed_loop_timing/base_simulation.yaml).

- [ ] **H35 `[STUDY SCOPE]`** `chapters/task_description.tex:19` & `chapters/07_conclusion_and_future_outlook.tex:6-9`
  - **Issue:** The official assignment calls for comparing phenomenological and data-driven predictors. The thesis identifies a biophysical simulation plant and data-driven control predictors, leaving the reader to infer how the phenomenological predictor requirement was addressed.
  - **Fix:** In the introduction or conclusion, identify the predictor that fulfilled this role, if applicable, or explain the scope decision.

- [ ] **H36 `[RETROSPECTIVE ANALYSIS]`** `chapters/06_experiment_and_discussion.tex:486,500-504,576-580`
  - **Issue:** The 1.68 s snapshot was selected after looking for a recorded replay with at least 20 seizing regions under the spectral plan. In the separate delayed-start example, 6.90 s was also selected retrospectively from one uncontrolled trajectory.
  - **Fix:** Disclose the 1.68 s selection rule in the figure caption. For the delayed start, report that waiting until 6.90 s worked on that trajectory and leave prospective identification of a safe delay open.
  - **Evidence:** [Snapshot selection record](../artifacts/seed_7104_snapshot_branches/report.md).

---

## AI-Language & Style Rewrites

- [ ] **AI8 `[AI-LANGUAGE]`** `chapters/06_experiment_and_discussion.tex:263-264`
  - **Current:** “A further observation can be made when the signs of the forecast errors are considered.”
  - **Fix:** Delete the lead-in. The next sentence already gives the finding that the waveform-only fit underestimated log-power at every lookahead.

---

## MEDIUM — Worth Polishing

- [ ] **M34 `[MODEL INITIALIZATION]`** `chapters/03_biological_plant_modeling.tex:170`
  - **Issue:** The model requires a regional state and a history spanning the longest delay. The text does not say how these were initialized.
  - **Fix:** Record the benchmark convention: initialize at the healthy resting state and fill the delay buffer with its unshifted firing rate.
  - **Evidence:** [Benchmark configuration](../artifacts/dmdc_closed_loop_timing/base_simulation.yaml), revision `4de695c`, `src/neuro/plant/jansen_rit.py`.

- [ ] **M35 `[OPTIMIZATION COMPARISON]`** `chapters/06_experiment_and_discussion.tex:511,526` & `chapters/08_appendix.tex:520-521`
  - **Issue:** The comparison calls the three approaches solvers for the “same problem.” Grouped search allows only 16 activation schedules, a restriction on the feasible set.
  - **Fix:** Say the approaches share a per-hold objective and identify the schedule restriction when describing the comparison.

- [ ] **M36 `[MONTAGE SELECTION]`** `chapters/03_biological_plant_modeling.tex:435` & `chapters/08_appendix.tex:21-25,39`
  - **Issue:** The text attributes the selected montage to sparse search, while the appendix presents TP9/CP5/Ex8 alongside the screened supports as a separate candidate. Its static refit reports approximately zero Ex8 current.
  - **Fix:** Explain how TP9/CP5/Ex8 entered the candidate set and distinguish electrodes available to the controller from electrodes carrying current in the static refit.

- [ ] **M37 `[RESULT ATTRIBUTION]`** `chapters/08_appendix.tex:246,276`
  - **Issue:** The 17.8-fold median solve-time improvement is attributed to condensing alone. Both the transcription and solver changed.
  - **Fix:** Attribute the measured improvement to condensed OSQP relative to single-shooting IPOPT.
  - **Evidence:** [Timing protocol](../artifacts/dmdc_closed_loop_timing/report.md).

- [ ] **M38 `[STATISTICAL INTERPRETATION]`** `chapters/06_experiment_and_discussion.tex:133-144`
  - **Issue:** The pooled correlations use 180,000 overlapping frame pairs from branches that share snapshots. The resulting coefficient describes a pooled association under the tested conditions.
  - **Fix:** Keep that scope explicit when interpreting the coefficient. Within-seed discrimination and performance during stimulation would require separate analyses.

- [ ] **M39 `[MODEL ASSUMPTION]`** `chapters/03_biological_plant_modeling.tex:437`
  - **Issue:** The 2 mA limit is described as satisfying safety and hardware limits without identifying a device or protocol.
  - **Fix:** Describe it as the per-electrode simulation bound, or cite the device or protocol that justifies the stronger wording.

- [ ] **M40 `[RESULT DENOMINATOR]`** `chapters/07_conclusion_and_future_outlook.tex:20` & `chapters/06_experiment_and_discussion.tex:573` & `chapters/08_appendix.tex:485`
  - **Issue:** “About half the time” leaves the denominator ambiguous because each simulation begins with a period before control starts.
  - **Fix:** Report the mean off time as 5.45 s of the 11.34 s after control began (48%).

- [ ] **M41 `[MODEL INTERPRETATION]`** `chapters/03_biological_plant_modeling.tex:69`
  - **Issue:** The sigmoid midpoint is described as the potential at which half the population fires. The equation defines a population-mean firing rate.
  - **Fix:** Define the midpoint as the potential at which the mean firing rate reaches half its maximum.

- [ ] **M42 `[UNITS]`** `chapters/06_experiment_and_discussion.tex:228`
  - **Issue:** Mean squared waveform error is given in “the original output units.”
  - **Fix:** Use “squared output units” for the reported MSE values.

- [ ] **M43 `[SIGNED ERROR]`** `chapters/06_experiment_and_discussion.tex:266`
  - **Issue:** The signed log-power error rises from −0.40 to −0.07. The current text says that it fell.
  - **Fix:** Write “reduced the magnitude of the mean log-power bias from 0.40 to 0.07,” or retain the signed values and say that the bias moved toward zero.

- [ ] **M44 `[TERMINOLOGY]`** `chapters/02_problem_formulation.tex:53` & `chapters/06_experiment_and_discussion.tex:325`
  - **Issue:** Chapter 2 reserves recruitment for regions outside the initiating zone. Chapter 6 calls the initiating regions themselves recruited.
  - **Fix:** Write “all three EZ regions met the seizure criterion during the run.”

---

## LOW — Optional Final Polish

- [ ] **L10 `[PDF NAVIGATION]`** `thesis.tex:223`
  - **Issue:** Resetting the page counter after the title pages generated duplicate destinations for pages i and ii in the reviewed build.
  - **Fix:** Give the title pages distinct PDF anchors and verify that links to the numbered front matter land on the intended pages.

- [ ] **L11 `[PAGE LAYOUT]`** `chapters/08_appendix.tex:27-43,276-279`
  - **Issue:** In the reviewed 112-page PDF, printed p. 69 contains only the four-row montage table, and printed p. 80 opens mid-sentence with the last two lines of a paragraph.
  - **Fix:** After the text edits, check these passages in the final PDF. Keep the montage table near its discussion and bring the trailing lines back onto the preceding page if they still spill over.
