# Chapter 4 review — 26 October 2026

Reviewed `chapters/04_data_driven_predictor_design.tex` against all 11 thesis-writing rules and all 11 thesis-review criteria. Locations refer to the unchanged source reviewed on this date. Findings are grouped by rule; related rules are noted where useful.

**Scope:** source-level review and compiler diagnostics. Visual checks were excluded at the user's request. The cited publications were not independently audited against their full texts, and the implementation was not inspected. Consequently, implementation-dependent questions below are clarification requests, not assertions that the implementation is wrong.

**Severity:** P2 = methodological or mathematical clarification needed; P3 = style, notation, or documentation improvement.

## Mechanical checks

- `uv run latex-lint check chapters/04_data_driven_predictor_design.tex` completed successfully with no diagnostics.
- `uv run latex-lint compile thesis.tex -o out` completed successfully. No undefined-reference, undefined-citation, or overfull-box warnings were reported for Chapter 4. The full-document log contains package warnings and box warnings elsewhere; these are outside this chapter review.
- These results do not establish compliance with the qualitative writing rules or confirm rendered formula wrapping.

## CITE-01 — Substantiate technical claims and parameter choices

### 1. P3 — State the signal-band rationale for downsampling

**Location:** `chapters/04_data_driven_predictor_design.tex:366`

**Excerpt:** “A causal fourth-order Butterworth filter with a cutoff frequency of 25 Hz attenuates high-frequency components before downsampling to reduce aliasing.”

The plant integration period is already specified in Chapter 3 (`chapters/03_biological_plant_modeling.tex:173`), so the original review's request to supply it again was unnecessary. The author clarified that downsampling removes the excessive temporal resolution needed for plant integration because the Jansen–Rit signal content relevant to this study lies below 25 Hz. This supplies the missing rationale for choosing a 50 Hz waveform representation. The finding is therefore reduced to a prose clarification, not a demonstrated filtering defect. A cutoff at the new Nyquist frequency does not itself establish rejection of all aliasing, but adequacy depends on the actual out-of-band signal power; the original sentence only claims to reduce aliasing.

**Concrete fix:** connect the integration time scale in Chapter 3 to the prediction sampling rate, and explicitly state that the signal content relevant to this study lies below 25 Hz. Support that signal-band premise with a relevant source or the simulated spectra. Keep the existing qualified claim that filtering reduces aliasing; do not imply that it eliminates aliasing. No filter redesign is requested by this finding.

### 2. P2 — Make the training protocol reproducible

**Locations:** `chapters/04_data_driven_predictor_design.tex:350`, `:391`, `:465`, `:468`, `:475`, `:841`

**Excerpts:** “200 trajectories of 12 s”; “assigned to training or validation”; “an initial learning-rate warm-up followed by cosine decay”; “over a prescribed range of training epochs”; “once activated”; “a specified fraction of the sum of squared singular values.”

The chapter specifies some data-generation values but leaves the split sizes, training and checkpoint settings, loss-activation timing, and rank-selection settings unspecified. These are experimental choices rather than claims that necessarily need literature citations, but their actual values and selection basis must be documented. Chapter 6 currently contains only an outline, so it does not yet supply the missing protocol.

**Concrete fix:** add one experiment-settings table, here or in Chapter 6, and reference it from this chapter. Include split counts and seed policy; actual histories and horizons; spectral window, segment, hop, smoothing, retained band, and floor; network sizes; optimizer and schedule settings; loss weights and activation epochs; curriculum endpoints; early-stopping settings; and DMD rank/regularization choices. Explain whether values were fixed by practical constraints or selected using validation data. A methods chapter can defer these values, but it needs an explicit destination.

## MATH-10 — Define quantities, domains, and conventions

### 3. P2 — Define a dimensionless logarithmic power feature

**Location:** `chapters/04_data_driven_predictor_design.tex:239` (formula at `:243`, floor definition at `:249`)

**Excerpt:** `z_{q,c,m} = \ln(\overline{S}_{q,c,m}+\varepsilon)`; “a small floor.”

The argument is a power density plus a floor, but neither a reference power density nor a convention that treats the argument as a numerical value in fixed units is stated. The feature offset and the numerical floor consequently depend on the units chosen for the waveform. The floor also lacks a value or a reference to its specification.

**Concrete fix:** define a fixed reference density `S_ref > 0` in the same units as `S`, and write `z = ln((S_bar + epsilon)/S_ref)`, with the units and value of `epsilon` stated. Alternatively, explicitly define `S_bar` in this expression as its numerical value in specified fixed units. Explain that changing the reference shifts log features by a constant; the floor also affects behavior near zero.

### 4. P2 — Specify stimulation/frame synchronization

**Locations:** `chapters/04_data_driven_predictor_design.tex:184`, `:352`, `:368`

**Excerpts:** `n_q=qH_s+L_s-1`; holding intervals “matching the two spectral prediction-step durations”; “u_k is the current applied over the interval from the timestamp of o_k to that of o_{k+1}.”

Matching the stimulation holding duration to the spectral hop does not by itself align their update boundaries. Spectral availability times have the segment-dependent offset `(L_s-1)/f_s`; stimulation updates may use a different origin. The formulation assumes one constant current across each prediction interval, so this alignment affects the meaning of the input-target pairs.

**Concrete fix:** state the update origin or frame-selection convention that makes stimulation changes coincide with spectral availability times. If changes can occur within a spectral prediction interval, explain how those intervals are represented instead. This is an unresolved timing convention in the text, not a demonstrated data-processing bug.

### 5. P2 — Interpret current sensitivity on the admissible current subspace

**Locations:** `chapters/04_data_driven_predictor_design.tex:347`, `:355`, `:850`

**Excerpts:** “variation to the zero-sum current subspace”; `partial z_hat / partial u_tilde = B_y`.

The derivative is correct for the fitted affine map, but its individual electrode columns include directions not independently excited by the training data. With zero-sum currents, only P−1 instantaneous current directions are physically admissible. In physical coordinates, adding a matrix of the form `v 1^T` to the current coefficient matrix leaves its action on every zero-sum current unchanged. Standardization transforms the corresponding null direction. Ridge or a pseudoinverse can select a coefficient representative, but cannot create evidence for an independently varied electrode response outside that subspace.

**Concrete fix:** qualify the interpretation of `B_y` as a sensitivity of the fitted extension and state that physical response comparisons use admissible zero-sum perturbations. Optionally express those perturbations through a basis of the zero-sum subspace. The conclusion that the affine model's admissible current sensitivity is history independent remains valid.

### 6. P3 — Complete mathematical domains and index ranges

**Locations and excerpts:**

- `chapters/04_data_driven_predictor_design.tex:63`: “history lengths” — give `d_o` a positive-integer domain and state whether `d_u` may be zero. If zero input history is allowed, say that its block is empty.
- `:125`–`:136`: sampling frequency, segment/hop lengths, frame index, and channel index — specify `f_s > 0`, positive-integer segment/hop lengths, and valid frame/channel ranges. State the nonzero-window condition needed by the power-density denominator.
- `:207`: “kernel of length L_k” — specify a positive-integer length; the weight constraints are already supplied.
- `:491`–`:512`: layer counts/widths and scalar activation argument — specify integer counts and `x in R` rather than leaving these domains implicit.
- `:595`–`:653`: convolution lengths, layer count, and position indices — give positive-integer domains and valid output ranges, notably `t=0,...,T_l-1` and a consistent retained-bin index range for the padded frequency operation.

**Concrete fix:** add these domains where each group is introduced. This is a strict MATH-10 compliance issue; most intended domains are inferable, but the rule explicitly requires them.

## MATH-08 — Distinguish running indices from labels

### 7. P3 — Set naming subscripts upright

**Locations:** `chapters/04_data_driven_predictor_design.tex:34`, `:36`, `:658`, `:723`

**Excerpts:** `n_o`, `n_y`, `n_v`, `n_x`.

These subscripts identify the observable, waveform, flattened feature vector, and state; they do not count elements. They are currently italic, unlike the upright naming subscripts already used in `d_\mathrm{o}` and `d_\mathrm{u}`.

**Concrete fix:** use `n_\mathrm{o}`, `n_\mathrm{y}`, `n_\mathrm{v}`, and `n_\mathrm{x}` consistently, including formulas and diagram labels. Retain italic counting subscripts such as `n_\ell` and `C_\ell`. Apply any eventual notation change consistently with the symbol list.

## MATH-07 — Match symbols to mathematical categories

### 8. P3 — Distinguish the Fourier coefficient vector from a matrix

**Location:** `chapters/04_data_driven_predictor_design.tex:138`

**Excerpt:** `\mathbf{Y}_{q,c} = \mathcal{F}_{+}(...)`.

For fixed frame and channel this object is a complex vector, but bold uppercase is reserved for matrices by the stated rule. Its space is also not stated explicitly.

**Concrete fix:** choose a lowercase bold symbol for the coefficient vector and state that it belongs to `C^{floor(L_s/2)+1}`; identify its scalar bin coefficients consistently. Avoid introducing a collision with the waveform vector.

### 9. P3 — Resolve the strict lowercase-scalar convention

**Locations:** `chapters/04_data_driven_predictor_design.tex:35`, `:96`, `:126`, `:309`, `:491`, `:618`, `:751`

**Excerpts:** scalar counts `P`, `H`, `L_s`, `H_s`, `N_tr`, `L`, `C_l`, and rank `R`.

These conventional uppercase scalar symbols conflict with the skill's literal requirement that scalars be lowercase. The same conflict already exists in the thesis symbol list, so this is a notation-policy inconsistency rather than evidence of incorrect mathematics.

**Concrete fix:** decide once whether uppercase counts and lengths are an explicit exception. Record that exception in the notation convention, or consistently rename the affected quantities throughout the document. A chapter-only rename would create a larger consistency problem.

## PROSE-05 — Use consistent terminology and notation

### 10. P3 — Reconcile the chapter with the symbol list

**Locations:** `chapters/04_data_driven_predictor_design.tex:34`, `:36`, `:63`; comparison: `chapters/00_symbols_and_acronyms.tex`, MPC symbol table.

**Excerpts:** Chapter 4 uses `n_o` for the general observable dimension, `n_y` for waveform channels, and `d_o` for observation history. The symbol list calls `n_y` the “Observable output dimension” and lists `d_y` as the output-history length.

The distinction between waveform channels and spectral-feature dimension is central to this chapter, so the old glossary wording can mislead a reader checking dimensions.

**Concrete fix:** add the general observable dimension and observation-history symbols to the list, identify `n_y` as the waveform channel count, and reconcile `d_y` with `d_o`. Also reconcile the symbol list's EEG channel count `M` with `n_y`, either by explicit equivalence or a single notation. This is a related-document correction; no symbol-list edits were made in this review.

## STRUC-05 — Supply opening and closing transitions

### 11. P3 — Bridge the data-generation and preprocessing subsections

**Locations:** `chapters/04_data_driven_predictor_design.tex:359`, `:364`, `:409`, `:413`

**Excerpts:** “The predictors are tested on separately generated trajectories with held-out seeds”; “The plant integration resolves faster dynamics”; “The fitted transformations remain fixed during validation and testing”; “The mean squared error is widely used for regression.”

The data-generation subsection ends without leading into preprocessing. Preprocessing begins with a new fact and ends without leading into the loss definition; the next opening then starts as a general textbook statement. The sequence is logical, but the bidirectional bridges required by the rule are incomplete.

**Concrete fix:** connect the generated trajectories to causal resampling and sample extraction, then connect the constructed standardized samples to the original-scale prediction errors that will be minimized. For example, replace the final preprocessing sentence with a sentence that both fixes the validation transformations and introduces the loss evaluated after inverse transformation.

### 12. P3 — Close the spectral construction with the representation comparison

**Location:** `chapters/04_data_driven_predictor_design.tex:268` (followed by the figure and the subsection at `:283`).

**Excerpt:** “Figure ... illustrates how recruitment ... produces changes in both the waveform and its time-varying spectrum.”

This leads into the illustration, but does not close the spectral-representation argument or explain why the following waveform-versus-spectral comparison is needed.

**Concrete fix:** add a brief bridge before the next subsection: the constructed features discard phase and aggregate measurements, so the next comparison considers what that means for prediction and stimulation-response discrimination. Keep it short to avoid repeating the comparison itself.

## PROSE-11 — Prefer precise, impersonal reporting of procedures

### 13. P3 — Reserve “we” for reasoning rather than data-processing actions

**Locations:** `chapters/04_data_driven_predictor_design.tex:301`, `:732`, `:840`

**Excerpts:** “we use supervised learning”; “we collect ... one-step transitions”; “We apply ... and recover the offset.”

These report procedures rather than guide the reader through a derivation. They are understandable, but the skill prefers impersonal wording for this use.

**Concrete fix:** write “The prediction models are fitted to simulated trajectories using supervised learning”; “The general fit is constructed from ... one-step transitions”; and “Equation ... is evaluated with ...; the offset is then recovered as ...”. Keep reader-guiding phrases such as “We first introduce” where they organize the derivation.

## PROSE-09 — Remove conceptual repetition

### 14. P3 — Consolidate the inverse-transformation explanation

**Locations:** `chapters/04_data_driven_predictor_design.tex:408`, `:558`

**Excerpts:** “Predictions are transformed back before loss evaluation”; “The prediction ... is transformed back to the original observable coordinates before loss evaluation.”

The second statement repeats the general procedure without adding information. The following sentence about cancellation of the mean for residual prediction does add information and should remain.

**Concrete fix:** remove the repeated sentence at line 558 and retain the residual-specific explanation with a short reference to the standardization equation. The figure caption may remain self-contained.

## PROSE-10 — Connect adjacent ideas into a continuous explanation

### 15. P3 — Integrate the isolated variance statement with its remedy

**Locations:** `chapters/04_data_driven_predictor_design.tex:204`, `:206`

**Excerpts:** “These power-density estimates remain variable even for stationary signals”; “To reduce fluctuations, power is averaged across consecutive spectral frames.”

The one-sentence paragraph introduces the problem, and the next paragraph immediately supplies its remedy. Their causal connection is already clear, so separating them makes the exposition more fragmented than necessary.

**Concrete fix:** combine them into one paragraph that first explains why longer segments alone do not solve the variance problem, then introduces causal averaging. Preserve the supporting citation.

## Remaining criteria and limits

- **STRUC-04:** the overall problem-to-representation-to-training-to-architecture arc is coherent; no separate finding.
- **PROSE-01 / PROSE-08:** no distinct actionable finding beyond the local cohesion and repetition edits above. No promotional rhetoric or conspicuous chatbot boilerplate identified.
- **CITE-05:** no separate common-knowledge-boundary finding. Citation presence was assessed; source support was not independently verified.
- **MATH-05:** displayed formulas are complete relations; no bare-expression finding.
- **PROSE-06:** no failed experiment or underperformance result is reported here. The affine predictor's history-independent current sensitivity is derived rather than presented as an unexplained negative result.
- **TAB-04:** no tables occur in the reviewed chapter.
- **STRUC-07:** there is no conclusion section in this chapter, and no conclusion-scope violation was identified.
- **MATH-11:** not visually assessed, as requested. Successful compilation cannot establish that inline mathematics never wraps awkwardly.
- **FIG-02 / FIG-04 / FIG-05 / FIG-09:** visual compliance was not assessed, as requested. No rendered-figure claims are made.

As source-level arithmetic checks, the CNN/MLP parameter counts at lines 702–703 agree with the stated architectures. The temporal receptive-field relation, spectral sample-support relation, and truncated-ridge solution are internally consistent as written. These checks do not validate the implementation or empirical performance.
