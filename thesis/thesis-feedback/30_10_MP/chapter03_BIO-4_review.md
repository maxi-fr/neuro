# Chapter 3 review — 30 October 2026

Scope: `chapters/03_biological_plant_modeling.tex`, reviewed against `thesis-review` and its referenced `thesis-writing` rules, with particular attention to PROSE-08–11. Line numbers refer to the unchanged chapter at review time. Supporting inspection stayed within the thesis workspace, apart from the requested skills and external literature. No chapter, bibliography, or figure was changed.

The chapter's overall order works: regional dynamics → delayed network → pathological configuration → electrical interfaces. The main editorial problem is repeated explanation at several levels, combined with paragraphs that introduce implementation details before explaining the modeling question those details answer. Preserve this order; revise paragraph structure locally.

Findings below distinguish definite changes, editorial judgments, and verification questions. Repetition between a caption and the body is not automatically a problem: captions must remain independently interpretable. Likewise, mathematical operations are legitimate subjects under PROSE-11.

## PROSE-08: Boilerplate, low-information prose, and unhelpful rhetoric

### 08-A — Definite: generic opening and closing phrases

Locations: chapter lines 3 and 448.

Excerpts: “This chapter specifies ...”; “This completes the specification ...”.

The opening sentence is serviceable, but the closing declaration adds little before another generic handoff. Prefer a concrete statement of what has become available for the next task: stimulation-current and EEG trajectories from the specified plant. Do not repeat the whole closed-loop workflow at both ends.

Fix at the end: link the completed electrical interfaces to learning the plant's current-to-EEG response, then name Chapter 4. The closing transition remains mandatory; replace boilerplate with a functional connection.

### 08-B — Definite: mechanical narration of an algorithm

Location: chapter line 251.

Excerpt: “This integration advances the coupled regional dynamics; the next section specifies ...”.

The first clause merely says what an integrator does. The transition should identify what remains unresolved: which regions generate pathological activity and which can be recruited. For example: “With the network dynamics specified, the regional gains Ai determine where pathological oscillations originate and which regions can be recruited.”

### 08-C — Optional: explaining a self-evident proportionality

Location: chapter line 139.

Excerpt: “Thus, longer tracts transmit activity later even when their connection weights are identical.”

The delay equation already establishes proportionality to length. Retain this only if the intended teaching point is the independence of connection strength and transmission time. In that case state that distinction explicitly, then use the numerical example at line 140.

### 08-D — Definite: vague figure introduction

Location: chapter line 266.

Excerpt: “illustrates the behaviour of nodes for these different excitability values.”

Name the comparison that supports the argument: the PZ remains at low amplitude in isolation but develops a large-amplitude oscillation under network coupling. This prepares the figure's interpretation and the definition of propagation.

### 08-E — Definite: code provenance in scientific exposition

Location: chapter line 315.

Excerpt: “the saved uncontrolled seed-7000 run.”

“Saved” describes file handling rather than the experiment. Prefer “an uncontrolled simulation (random seed 7000).” Retain the seed for reproducibility.

## PROSE-09: Conceptual repetition

### 09-A — Definite: the introduction repeats the feedback loop

Location: chapter lines 4–6.

Excerpts: “training and validating the data-driven predictors”; “select currents applied to the plant”; “provide feedback to the controller”.

These sentences retell the surrounding workflow in small pieces. Consolidate the two uses of the plant—data generation and closed-loop evaluation—then explain the distinct evaluation roles of EEG and regional activity. The regional activity sentence should identify seizure suppression or recruitment as the assessment quantity, rather than “its ability”.

### 09-B — Definite: regional coarse-graining is introduced twice

Location: chapter lines 24–30.

Excerpts: “describes a cortical macro-column ... whole populations”; “Although originally formulated for a cortical macro-column ... regional activity”.

The second paragraph restarts the scale explanation. Develop one progression: population averaging → Jansen–Rit microcircuit → one such circuit per atlas region. Keep the regional-scale modeling assumption, which is scientifically relevant, but integrate it with node introduction.

### 09-C — Definite: weights are defined twice

Location: chapter lines 126 and 129.

Excerpts: “assigns a weight ... sending region ... receiving region”; “specify how strongly each sending region contributes”.

Line 129 repeats the definition. Move its citation to the initial definition or the nontrivial normalization claim, then delete the repeated explanation. Preserve the sender/receiver convention.

### 09-D — Definite: the effect of K is explained again during calibration

Locations: chapter lines 162–165 and 301.

Excerpts: “increasing K strengthens the incoming drive”; “Increasing K strengthens the delayed firing-rate input”.

Explain K's mathematical operation beside the coupling equation. In calibration, state the selection objective and what was varied. If recruitment is the reason for adjusting K, connect it directly to the calibration target instead of repeating the earlier definition.

### 09-E — Definite: the discrete coupling explanation ends with a second summary

Location: chapter lines 194–204.

Excerpt: “Thus, ... the discrete approximation of the same network input, using the quantized delay and numerical output history.”

The introductory sentence, equation, lag explanation, and concluding sentence all describe the same construction. Retain the equation and the meaning of the delayed index; remove the concluding paraphrase. Under the ground-truth convention, introduce this as the coupling used by the discrete plant.

### 09-F — Definite: linearity is re-established in the measurement subsection

Locations: chapter lines 352–354 and 364–365.

Excerpts: “depend linearly on their sources”; “regional contributions superpose linearly”.

Establish the quasi-static linearity assumption once in the Electrical Interfaces introduction. In the measurement subsection, apply it: the assumed regional sources give the stated matrix observation map. Retain the explanation connecting membrane potentials to source density; that is new information.

### 09-G — Definite: the stimulation entry point is justified twice

Locations: chapter lines 101–102 and 392–396.

Excerpts: “Stimulation shifts ... pyramidal sigmoid”; “This motivates the additive perturbation si ...”.

Keep the mathematical meaning and local sign effect beside the node equations. Reserve the physical polarization argument and its citations for the stimulation subsection. There, link the projected field to the already-defined si without again contrasting it with a firing-rate input unless that distinction is needed to explain the modeling choice.

### 09-H — Optional: the vertex aggregation is interpreted three times

Location: chapter lines 381–383.

Excerpts: “all vertices ... share its neural-mass time course”; “spatially uniform source-density amplitude”; “This aggregation gives ...”.

The first two statements overlap. Combine them while preserving the important consequence that summing rather than averaging introduces a dependence on represented regional extent. Put the matrix dimensions with the dataset description or original matrix definition.

### 09-I — Optional: a basic background-gain point follows its definition

Location: chapter lines 262 and 265.

Excerpt: “Background nodes have a lower gain and are less susceptible ... although sufficiently strong network activity can recruit them too.”

“Lower gain” is already given numerically. Keep the new information about possible background recruitment, because the later calibration targets spread beyond the PZ. Connect it to that modeling objective rather than repeating the gain comparison.

## PROSE-10: Cohesion and narrative development

### 10-A — Definite: the two transformations are attached to the wrong subject

Location: chapter lines 50–63.

Excerpt: “Each synaptic pathway consists of two transformations. First ... convolution ... Then ... sigmoid”.

The filter maps incoming firing rate to postsynaptic potential; the sigmoid maps a population's net potential to output firing rate. Present these as the two transformations in the population model, rather than suggesting that the sigmoid is itself a synaptic filter. Introduce their composition through the receiving population.

### 10-B — Definite: state definitions do not fully exploit the preceding circuit explanation

Location: chapter lines 71–75.

Excerpt: “The states xi,2 and xi,3 are the excitatory and inhibitory PSPs, respectively.”

Unlike the description of xi,1, this omits the receiving population. Say explicitly that these are the excitatory and inhibitory potentials at the pyramidal population. Then group the potentials and their derivatives before introducing the six-state equations.

### 10-C — Optional: an administrative aside interrupts the network introduction

Location: chapter lines 122–124.

Excerpt: “In this work, TVB supplies the anatomical data; the numerical integration is specified separately below.”

The distinction is useful for provenance, but currently interrupts the progression from dataset to weights. Integrate the provenance with the dataset sentence or place it when introducing the numerical method. Do not imply that TVB supplied a simulation if only its data were used.

### 10-D — Definite: a required closing bridge is missing before numerical integration

Location: chapter lines 167–174.

Excerpt: “An initial history ... is therefore required ...” followed by the Numerical Integration heading.

This is a good explanation of why the current state is insufficient, but it does not finish the connection to the grid. Extend it with the computational need: storing the required output history motivates a discrete representation of the delays and the integration grid. This adds a causal bridge rather than a generic “next subsection” sentence.

### 10-E — Definite: the anatomical assignment is not explicitly motivated by the preceding regimes

Location: chapter lines 268–279.

Excerpt: “seizure propagation ... recruitment ...” followed by “The assignment ... follows ...”.

Explain that the three dynamical regimes must now be assigned to anatomical regions to define a spatial propagation scenario. End the preceding discussion with this need, then identify the left mesial temporal configuration. Keep the model-specific definition of propagation.

### 10-F — Definite: calibration begins with a table inventory

Location: chapter lines 284–306.

Excerpt: “Table ... summarizes ...”; “The shared ...”; “The regional values ...”; “The remaining ...”.

The reader reaches the calibration purpose only at line 303. Close Anatomical Assignment by explaining that fixed regional susceptibilities leave the timing and extent of recruitment to be set by coupling and noise. Open Network Calibration with the target propagation pattern, followed by the recruitment criterion, the adjusted parameters, the selected values, and the representative result. Move the parameter-table overview after the calibration account or into its caption.

### 10-G — Definite: the calibration metric follows the selected values

Location: chapter lines 303–310.

Excerpt: “The resulting values are ...” precedes “A region is classified as recruited if ...”.

Introduce the classification before reporting the outcome it measures. Distinguish the qualitative reference pattern, the operational target, the detection rule, and the variability across seeds. Specify whether “recruited” means currently above threshold or ever above threshold; the figure reports cumulative recruitment.

### 10-H — Definite / BIO-03: preserve and improve the closing section transition

Location: chapter lines 347–348.

Excerpt: “autonomous and network-driven evolution”; “Closed-loop control additionally requires ...”.

“Autonomous” is a distracting description of a system that also includes stochastic and stimulation inputs. The bridge should grow from the completed propagation scenario and identify the remaining connection to the controller.

Suggested replacement, kept at the end of the section:

> The calibrated network defines the seizure propagation scenario used to evaluate control. The controller observes this activity through scalp EEG and acts on it through electrode currents. The next section specifies how these measurements and currents are related to the regional neural dynamics.

The Electrical Interfaces opening can then explain the common volume-conduction physics without repeating the controller workflow.

### 10-I — Definite: the measurement subsection ends without a forward link

Location: chapter lines 385–391.

Excerpt: “rather than an absolute prediction in clinical microvolts” followed by “The actuation proceeds in the opposite physical direction.”

The limitation is important, but the last paragraph ends on calibration while the next begins with directionality. Add a short closing link from the specified observation map to the still-required actuation map. The stimulation opening should then introduce field-induced polarization rather than repeat the bridge.

### 10-J — Definite: the stimulation derivation mixes dependent and independent quantities

Location: chapter lines 398–425.

Excerpt: “For each non-reference electrode ...” followed later by a sum over all electrodes and then the zero-column explanation.

Define the return-electrode parameterization before presenting formulas over all electrodes. State that each basis solution injects at one active electrode and returns through Ex8, and that Ex8's column is zero by construction. Then define the projected regional coefficient and matrix map. This makes the all-electrode notation immediately interpretable.

### 10-K — Definite: montage choice, constraints, and reciprocity interrupt one another

Location: chapter lines 427–446.

Excerpt: “The montage was selected ...”; “safety and hardware limits”; “This difference does not violate Helmholtz reciprocity.”

Describe the montage's purpose alongside its introduction at line 399. Keep the admissible-current constraints together. Place the lead-field comparison after both maps have been fully defined, in a separate paragraph whose opening explains why these separately constructed maps need not be transposes. The reciprocity clarification is relevant; reduce the defensive standalone sentence rather than discarding the explanation.

## PROSE-11: Precise methodological agency

### 11-A — Definite: unclear assessment subject and object

Location: chapter line 6.

Excerpt: “regional activity allows its ability to suppress the seizures to be assessed.”

“Its” has competing antecedents, and the sentence does not identify the assessment. Prefer: “Seizure suppression is assessed from the simulated regional activity.” If the actual endpoint is propagation beyond the EZ, name that endpoint instead of the broader suppression claim.

### 11-B — Definite: abstract subjects obscure assignments

Location: chapter lines 262–264.

Excerpts: “Healthy background regions use ...”; “The EZ takes ...”; “the PZ takes ...”.

These describe a methodological assignment, not an operation performed by tissue. Use a parallel construction: “The gains ... are assigned to background, PZ and EZ regions, respectively.” Then develop the distinct dynamical consequences. Avoid a mechanical three-sentence list of “takes” and “uses”.

### 11-C — Definite: calibration language is vague about the actual criterion

Location: chapter lines 303–306.

Excerpt: “was calibrated to obtain ...”; “Inspired by this sequence ... targets ...”; “The resulting values ...”.

The first impersonal construction is acceptable. The weakness is missing operational information: how the pair was selected, which runs were evaluated, and what counted as meeting the target. Replace “Inspired by” with the actual selection criterion and a precise appendix reference once the procedure exists. Do not invent a sweep, optimization objective, or robustness result.

### 11-D — Definite: a geometric operation is attributed vaguely to a projection

Location: chapter line 381.

Excerpt: “the surface projection supplies their different sensor weights.”

Prefer the actual construction: “Each vertex's time course is weighted by its column of the supplied surface lead field.” Combine with the shared regional source-density assumption where appropriate.

### 11-E — Definite: constraints are attributed to a design intention

Location: chapter line 429.

Excerpt: “By design, the individual electrode currents are constrained to respect safety and hardware limits”.

State the imposed bound directly: “Each electrode current is constrained to ...”. If retaining the safety/hardware rationale, identify its supporting guideline or actual hardware specification. A numerical bound alone does not substantiate both claims.

### Constructions to retain

“The sigmoid converts ...”, “the corrector averages ...”, “the factor K scales ...”, and “the lead-field matrix maps ...” are precise descriptions of mathematical operations. They do not violate PROSE-11. Impersonal prose is preferable for calibration, parameter assignment, and FEM computation; it is not a reason to replace every active sentence with passive voice.

## Other core and specialized rules

### PROSE-01 — Definite: overloaded sentences

- Chapter line 27, excerpt “is used here because, unlike ... while remaining ...”: combines model comparison, cell populations, filtering, dimensionality, and pathological dynamics. Split the mechanistic rationale from the useful dynamical repertoire; remove the vague comparison with unspecified simpler models unless it matters.
- Chapter line 69, excerpt “where 2e0 ... v0 ... and r ... reflecting ...”: three parameter definitions and a physiological interpretation. Keep definitions adjacent to the sigmoid, but split the physiological interpretation into a separate sentence if retained.
- Chapter line 264, excerpt “leaving it just below ... sufficiently excitable ...”: separate isolated-node behavior from recruitment by EZ coupling.
- Chapter line 360, excerpt “synaptic currents flowing along ... together with their return currents ...”: split source formation from orientation. The next sentence can then explain coherent summation.
- Chapter line 372: define the vector and matrix spaces without turning the sentence into a long inventory; a compact equation-aligned definition is sufficient.
- Chapter line 273: the caption includes a long interpretive explanation and a hypothetical background-region trace. Remove the hypothetical trace unless the underlying result is actually shown or documented.

### PROSE-05 — Definite: terminology and symbols

- Chapter line 273 calls the EZ response a “limit cycle” while describing simulations with noise. Clarify whether this characterizes the underlying deterministic regime or the noisy trajectory; do not equate a stochastic trace with an exact periodic orbit.
- Chapter lines 262–265 alternate “healthy background regions”, “background nodes”, and region classes. Prefer one designation after introduction, such as “background regions”.
- Chapter line 315 and lines 307–310 should distinguish instantaneous detection from cumulative recruitment. These are related but different quantities.
- Chapter line 273 labels the output `y_reg`, while `figures/03_biological_plant_modeling/excitability_regimes.py:85` labels it `y_LFP`. Align the figure with the chapter's defined regional-output symbol.
- Chapter lines 401 and 424–425 use “reference/return” for Ex8, whereas line 384 discusses the EEG reference. Explicitly distinguish the stimulation basis return from the EEG referencing convention.

### CITE-01 — Definite: unresolved supporting material and parameter claims

- Chapter line 308, excerpt “detailed motivation and sensitivity reported in the appendix”, and line 428, excerpt “search procedure and montage comparison ... appendix”: `chapters/08_appendix.tex` currently contains only a comment about calibration. These are unfulfilled references. Populate the supporting material or remove/qualify the promises; do not fabricate analyses.
- Chapter line 140 / table line 341: the chosen conduction speed is numerically illustrated but not justified there. Identify whether it is inherited from the reference model or selected for this study, and cite or document that choice. The equation converting length to delay does not justify the numerical speed.
- Chapter line 127: “normalized” does not identify the normalization. Specify the actual convention if needed to reproduce the weights and interpret K; the later statement that K absorbs normalization does not identify the supplied matrix.
- Chapter lines 404–405: the orientation vector's source and registration to the ROAST field are not specified. Buzsaki2012 supports the physiological orientation motivation, but does not establish the construction of this study's regional normals. This needs a factual clarification before rewriting.
- Chapter line 374: the precise TVB projection dimensions and version are dataset provenance facts. Check the cited source or cite a dataset/version if the general framework paper does not document this exact file.
- Chapter line 429: substantiate “safety and hardware limits” if retaining those descriptions.

This was not a full claim-by-claim validation of every cited paper; existing `cite-checked` comments were not treated as independently verified evidence.

### MATH-10 — Definite: local definitions need correction or completion

- Chapter line 69, excerpt “the potential at which half the population fires”: the equation establishes half the maximum mean firing rate, not that half the individual neurons fire. Use “the potential at which the mean firing rate reaches half its maximum”.
- Chapter lines 34 and 87: several newly introduced coefficients and inputs lack explicit domains under the skill's strict rule. Add domains compactly where first introduced, without scattering separate definition sentences throughout the derivation.
- Chapter line 91: the drift map's signature is implicit. If enforcing the rule strictly, state its state/input domains and six-dimensional output alongside its definition.
- Chapter line 375: `c` as a channel index and `v` as a vertex index are inferable but should be introduced explicitly; `c` also denotes conduction speed elsewhere. Consider a distinct channel index if it improves local clarity.
- Chapter lines 402–421: the electric-field basis is defined only for non-reference electrodes, while the coefficient formula is presented for all electrodes. Explicitly define the reference coefficient to be zero and restrict the projection formula to non-reference electrodes.
- Chapter lines 170–200: an initial history is declared necessary but its actual initialization is not specified. A fully defined ground-truth simulation requires the initial regional states and prehistory convention somewhere in the thesis. Report the actual procedure; do not assume a constant history or a burn-in.

### STRUC-04 and STRUC-05

The principal narrative arc should be retained. STRUC-05 weaknesses occur at lines 170/172, 268/277, 284/293, and 385/387 as described under PROSE-10. Do not satisfy the transition rule by adding a repeated summary at every heading: each bridge should identify an unresolved question that the following section answers. This reconciles mandatory transitions with PROSE-09.

### Figure and caption consistency — definite and provisional findings

- Chapter line 273 says Welch segments contain 16,000 samples. The script at `figures/03_biological_plant_modeling/excitability_regimes.py:89` sets `nperseg=2**14`, i.e. 16,384. Align the caption with the actual figure generation. Three segments can still result from Welch's default overlap; the segment count itself is not established as wrong.
- The same caption says all runs use the table parameters, but `excitability_regimes.py:24` sets `SIGMA = 280.0`, whereas chapter table line 339 reports 0.86 s^(-1/2). This is a verification question: it may reflect a legacy parameter convention or a unit conversion. No implementation outside the thesis was inspected, so these values have not been shown to be directly comparable. Check provenance before asserting consistency or regenerating the figure.
- FIG-02: the inspected Python plotting sources explicitly enable grids. No missing-grid finding is established from source inspection.
- FIG-04: physical units are present in the inspected plot-source labels, but strict quantity/unit notation could be standardized, e.g. `t / s`, `f / Hz`, and the regional-output symbol over mV. Dimensionless region indices and recruitment counts do not need invented physical units.
- FIG-05: the propagation plot and assignment plot include legends in source. The single-trace excitability panels need no multi-curve legend. Verify the rendered artifacts before claiming full compliance.
- FIG-09: figures are included as PDF or TikZ, and the shared plotting style uses thesis TeX fonts. This supports the intended format/font policy; actual embedded fonts and final figure scaling were not visually audited. Heatmap cells being raster content is not by itself evidence that the surrounding PDF labels are rasterized.

### Rules with no substantive issue found in the inspected text

- MATH-05: displayed expressions form relations.
- MATH-07 and MATH-08: no clear general scalar/vector/matrix or index-typography violation found in the chapter. Keep existing semantic distinctions.
- TAB-04: the parameter table already has a dedicated unit column; only its dimensionless convention produces warnings.
- PROSE-06: no failed competing method is analyzed here, so a negative-results analysis is not required.
- STRUC-07: the chapter handoff introduces no new result; the thesis conclusion was outside scope.
- MATH-11: actual awkward formula linebreaks cannot be determined from source alone; inspect the compiled pages after revision.

## Recommended editing sequence

1. Establish the ground-truth simulation convention and literature-supported step-size rationale.
2. Restructure the calibration paragraphs around target → metric → selection → result, without inventing missing procedure details.
3. Remove repeated conceptual explanations while preserving source definitions and standalone figure captions.
4. Repair the closing/opening bridges, especially the transition kept at the end of Regional Excitability and Seizure Propagation.
5. Rewrite the stimulation-basis exposition so the return electrode is defined before the all-electrode formulas.
6. Resolve factual questions about figure parameters, orientation provenance, initialization, and missing appendix material before making claims that depend on them.
7. Run chapter lint again, compile the thesis into `out/`, and inspect the affected pages for floats, captions, and formula wrapping.
