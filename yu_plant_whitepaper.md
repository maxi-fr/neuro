# Implementation Plan — Closed-loop tES Whole-Brain Epilepsy Model (Yu et al. 2024)

A staged build with explicit checkpoints and verification
steps. TVB is the source of
truth for connectivity, delays, sensors, and the EEG projection.

**Guiding principle for verification:** because the model is stochastic and (later) uses a different
integrator than the TVB reference, compare *statistics* — power spectra, energy rankings, seizure-node
counts, phase-plane topology, recruitment order — not pointwise traces.

---

## Stage 0 — Environment & data (prerequisite)

**Goal.** Load everything the network needs from TVB and build the region $\to$ sensor EEG gain matrix.

**Build.**

- `from tvb.simulator.lab import *`.
- `conn = connectivity.Connectivity.from_file()` (default dataset). Pull `conn.weights`,
  `conn.tract_lengths`, `conn.centres`, `conn.region_labels`, `conn.hemispheres`; set `conn.speed = 50`.
- Load EEG side: `SensorsEEG.from_file('eeg_unitvector_62.txt.bz2')`,
  `ProjectionSurfaceEEG.from_file('projection_eeg_62_surface_16k.mat', matlab_data_name="ProjectionMatrix")`,
  `RegionMapping.from_file('regionMapping_16k_76.txt')`.
- Collapse the surface projection (vertices $\to$ 62 sensors) through the region mapping (vertices $\to$ regions)
  into a single **$L$ of shape $(62, N_{\text{regions}})$** — your hand-rolled EEG forward operator.
- Build two dictionaries: `region_label` $\to$ `index` and `channel_label` $\to$ `index`.

**Checkpoint.** A loaded `(weights, tract_lengths, centres, labels)` set, a delay matrix
$D = \text{tract\_lengths} / \text{speed}$, and a $L$ matrix.

**Verify.**

- Reconcile region count: the paper uses **74**, TVB's default is **76**. Confirm which two regions are
  dropped (or whether a different dataset is intended) and that labels include the EZ/PZ regions you need:
  `lHC, lPHC, lAMYG, lTCI, lTCV` (and the right-hemisphere counterparts for patients 2/3/5/6).
- Confirm the 62 channel labels include every channel the paper names: `CP5, CP6, PO3, P1, P3, F3, F5, AF3, O1`.
- `weights` plausibility: nonnegative, expected hemispheric block structure; `hemispheres` splits L/R correctly.
- Delays sane: $D$ off-diagonal in the ~ms–tens-of-ms range, not seconds.

---

## Stage 1 — Single-node Jansen–Rit

**Goal.** One isolated node; reproduce the fixed-point $\leftrightarrow$ limit-cycle transition as $A$ varies.

**Build.** The 6-ODE system + sigmoid $S(v) = \frac{2e_0}{1 + \exp(r(v_0 - v))}$. Deterministic RK4 first; then add the
noise term $\zeta$ *only* to the $\dot{x}_5$ (excitatory-interneuron) equation and switch to a stochastic scheme
(stochastic Heun is the simplest correct choice; the paper uses stochastic RK4). Output $y = x_2 - x_3$.

**Checkpoint.** Time series of $y$ for a single node at $A = 3.25$ and $A = 3.6$, plus a phase-plane plot.

**Verify.**

- $A = 3.25$: low-amplitude, noise-driven background (target amplitude $\approx 2$, matching the paper's resting state).
- $A = 3.6$: sustained limit-cycle oscillation; phase plane shows a closed orbit, not a fixed point (cf. Fig 8b).
- Sweeping $A$ upward shows a clear onset of oscillation; dominant frequency lands in a plausible
  Jansen–Rit band (alpha-ish, ~8–12 Hz with standard parameters).
- **Unit check:** confirm $a = 100$, $b = 50$ are consistent with your $dt$ and time unit (seconds);
  integration is stable. Lock $dt$ here — it is otherwise unspecified by the paper.

---

## Stage 2 — Network: instantaneous coupling, then delays

**Goal.** Couple $N$ nodes; reproduce seizure initiation in the EZ and recruitment of the rest of the network.

**Build.**

- **2a (instantaneous):** add $K \sum_{j} w_{ij} S(y_j(t))$ to the $\dot{x}_5$ equation.
  ⚠️ **Use node $j$'s output $S(x_{j,2} - x_{j,3})$, not the node's own** — the paper's printed Eq. (4)
  shows $x_{i,\dots}$, which is almost certainly a typo. Getting this wrong gives uncoupled oscillators.
- **2b (delayed):** maintain a ring buffer of past $y$; read $y_j(t - D_{ij})$ with $D_{ij}$ rounded to steps.

**Checkpoint.** Full $N$-node run with $\text{EZ} = \{\text{lHC}, \text{lPHC}, \text{lAMYG}\}$ ($A=3.6$) and $\text{PZ} = \{\text{lTCI}, \text{lTCV}\}$ ($A=3.4$),
others $A=3.25$, $K=0.75$. Spatiotemporal raster of all nodes (target: Fig 3a).

**Verify.**

- All-healthy control: with every $A = 3.25$ and $K = 0.75$, the network stays in background (no spontaneous seizure).
- With EZ/PZ set, oscillation **starts in the EZ and recruits neighbors over time**, spreading across the
  left hemisphere and then to the right (qualitative match to Fig 3a).
- **Coupling-correctness test:** zero a downstream node's incoming weights — it must *stop* seizing while
  its neighbors still do (proves recruitment is network-driven, not a local bug). An isolated EZ node must
  still oscillate on its own.
- 2a vs 2b: delays change recruitment *timing/order* (onset time rises with tract distance from the EZ)
  but not *whether* the network can seize.

---

## Stage 3 — Immediate tES via the $\gamma$ vector

**Goal.** Inject $U_{\text{tES}} = \hat{u}_{\text{tES}} \cdot \gamma$ *inside the pyramidal sigmoid* (Eq. 2). Use the simplified
Gaussian-falloff $\gamma$ (from the weight-generator) or loaded values; $\hat{u}_{\text{tES}} = 1.5$.

**Build.** Per-node $U_{\text{tES}}$ vector; cathode placed near the left-temporal target ($\approx$ CP5); apply only during
a stimulation window (e.g. first 25 s). **No $Z$ dynamics yet** — this isolates the immediate effect.

**Checkpoint.** Reproduce Fig 4a: during cathodic tES, oscillation is confined to EZ/PZ nodes; on removal,
it re-expands across the hemisphere.

**Verify.**

- Stimulation ON $\to$ number of oscillating nodes collapses to $\approx$ EZ/PZ only (immediate suppression).
- Stimulation OFF $\to$ oscillation re-expands quickly (**no lasting effect without $Z$** — this is the intended
  contrast and confirms persistence must come from Stage 4, not Stage 3).
- $\gamma$ is **heterogeneous**, with the largest $|U_{\text{tES}}|$ on the target/EZ/PZ nodes (cf. Fig 4c/d). If $\gamma$ is nearly
  flat, your falloff $\sigma$ is too large and later results won't separate.
- Polarity sanity: cathodic (negative) suppresses; flipping the sign should *excite*.

---

## Stage 4 — $Z$-dynamics lasting effect

**Goal.** Add the slow per-node synaptic state and modulate $C_1, C_2$.

**Build.** Extend the state with $Z, \dot{Z}$ per node:
$\ddot{Z} = a|u_{\text{tES}}| - a_1\dot{Z} - b_1Z$, $a_1 = 8 + \beta\left(\int |u_{\text{tES}}| dt\right)^2$, $a=10, b_1=10$;
$C_1 = 135(1 + \text{sig} \cdot Z)$, $C_2 = 135(0.8 + \text{sig} \cdot Z)$, with $\text{sig} = -1$ for cathodic nodes ($\gamma < 0$), else $0$.
Accumulate $\int |u_{\text{tES}}| dt$ per node. $C_1, C_2$ are now per-node, time-varying inputs to Stage 2's equations.

**Checkpoint.** Reproduce Fig 4b/d/e: suppression persists ~25 s after stimulation stops; per-node $Z(t)$
curves show the Fig 2 shape (rise during stim, slow decay after).

**Verify.**

- $Z(t)$ trends (Fig 2): saturates with longer stimulation; higher intensity $\to$ larger and longer-lasting $Z$.
- **Prolonged-transient behavior (Fig 8):** post-stim suppression lasts a *finite* time, then the system
  reverts to seizure. $Z$ is a transient, not a new stable fixed point — if your suppression is permanent,
  the $Z \to C_1/C_2$ coupling or decay is wrong.
- Node-dependence: higher-intensity nodes (e.g. node 3) show longer post-effects (Fig 4e).
- **Internal-consistency sweep** (TVB can't check this stage): post-effect duration increases monotonically
  with stimulation intensity and duration (Fig 2 / Fig S1).

---

## Stage 5 — EEG projection + energy

**Goal.** Project node outputs to 62 channels; compute band-limited energy; identify the strongest channel.

**Build.** $\text{EEG} = LY$; detrend; PSD via Welch or multitaper (substitute for Chronux); integrate
0–50 Hz over the 50 s window; normalize so $\text{max} = 1$.

**Checkpoint.** Reproduce Fig 3c (left channels F3/P3/CP5 high-amplitude) and Fig 5a (energy ranking).

**Verify.**

- For the canonical lHC/lPHC/lAMYG setup, the **max-energy channel is CP5**. If not, suspect parcellation
  misalignment or a transposed $L$.
- Background channel amplitude $\approx 2$; seizing channels reach the tens (validates the closed-loop threshold $= 5$).
- High-energy channels are ipsilateral (left, temporo-parietal) to the EZ.
- This stage's $L$ is independently checkable in Stage 7 via TVB's EEG monitor.

---

## Stage 6 — Closed-loop controller + sweeps

**Goal.** Amplitude-triggered tES; reproduce the feedback-signal, open-vs-closed, and $\tau$-$\beta$ results.

**Build.** Online amplitude monitor on the feedback channel; when amplitude $>$ **threshold $5$**, turn tES on
for **$\tau$** seconds (drives the $Z$ dynamics). Track duty cycle and count oscillating nodes in $[0,25]$ vs $[25,50]$.
Sweep **$\tau$** and **$\beta$**. Then open-loop control matched in total energy for comparison.

**Checkpoint.** Reproduce Fig 5 (feedback dependence), Fig 6c (open vs closed lasting effect),
Fig 7a/b ($\tau$-$\beta$ heatmaps).

**Verify.**

- **Feedback matters:** CP5/PO3 (near EZ/PZ) trigger early, fire $\ge 2$ stims in the first 25 s, and achieve
  lasting suppression; O1 (far) triggers late or never and fails (Fig 5d–g).
- Lasting effect improves monotonically with $\tau$ and $\beta$ (Fig 7a vs 7b).
- Duty cycle stays below ~45 % even at $\tau = 5$ (paper reports $\approx 40.13\%$ for CP5).
- Closed-loop matches open-loop's immediate effect but gives **better lasting** suppression at equal energy (Fig 6c).
- Patient sweeps: remap EZ/PZ per Table 2; for Patient 6 (bihemispheric PZ), single-target fails and
  **dual-cathode** CP5+CP6 (split $1\text{ mA} \to 0.5 + 0.5\text{ mA}$) recovers suppression (Fig 10).
- **Lock the amplitude measure as config** (instantaneous vs. windowed peak vs. envelope) — the paper is
  vague and this drives trigger timing. Average over $\ge 10$ seeds, as the paper does.

---

## Stage 7 — Full TVB reference implementation (validation)

**Goal.** Rebuild the **uncontrolled** network in TVB and use it to validate your hand-rolled backbone.

**Build.**

- `model = models.JansenRit(...)` with a **spatialized $A$** array (EZ/PZ/other per region), $B=22$,
  $v_0=6$, etc. Reconcile TVB's parameter names and time-unit conventions against the paper's $a=100, b=50$.
- `connectivity = conn` (Stage 0); `conn.speed = 50`; `coupling = coupling.SigmoidalJansenRit(r=0.56, midpoint=6, ...)`
  with `cmin/cmax` tuned so the coupling scale matches your $K = 0.75$.
- `integrator = integrators.HeunStochastic(dt=…, noise=noise.Additive(nsig=…))` matched to your noise level.
- Monitors: a raw/temporal-average region monitor **and** `monitors.EEG(projection=…, sensors=…, region_mapping=…)`.

**Checkpoint.** A TVB run of the canonical EZ/PZ network producing region time series + 62-channel EEG.

**Verify (the comparison).**

- **Scope.** TVB cleanly validates Stages 1, 2, 5 (single-node dynamics, coupled + delayed network, EEG
  forward) — i.e. the *uncontrolled backbone*. It does **not** natively validate Stages 3/4/6: TVB's
  stimulus is added to the *coupling variables*, whereas this paper's tES is a polarization *inside the
  pyramidal sigmoid* plus slow $Z$-modulation of $C_1/C_2$. Replicating those in TVB requires subclassing
  `JansenRit` (the route TVB maintainers recommend) — treat that as an optional stretch goal.
- **Match on the uncontrolled run** (statistics, not pointwise — integrators and seeds differ): per-node
  dominant frequency, which nodes seize, recruitment order, and the EEG energy ranking (top = CP5).
- **Validate $L$ directly:** feed identical region time series through both your $L$ and TVB's EEG monitor;
  the 62-channel outputs should agree up to scaling/reference.
- **Triage rule:** if TVB and your engine disagree on *which* nodes seize or on the energy ranking, the bug
  is in Stages 1/2/5 (coupling sign/indexing, delay buffer, or $L$ orientation) — not in the control logic.

---

## Cross-cutting

**Parameters the paper leaves unspecified — fix these as config and freeze them after Stage 1–2:**
integration step $dt$; noise std for $\zeta$; the closed-loop amplitude measure; the "oscillating node"
criterion (the paper uses $<10$ remaining nodes as the effectiveness cutoff in Fig 9). Match the
qualitative backbone (Fig 3/4) *before* trusting any control result.

**Regression harness.** After each stage passes, snapshot its key statistic (e.g. Stage 2: set of seizing
nodes + recruitment times; Stage 5: energy ranking; Stage 6: oscillating-node counts per $(\tau,\beta)$) into a
small test so later refactors can't silently break an earlier stage.

**Validation map.**

| Stage | Validated by |
|-------|--------------|
| 1 Single node | Bifurcation behavior + TVB single-node (Stage 7) |
| 2 Network + delays | Coupling/disconnect tests + TVB region series (Stage 7) |
| 3 Immediate tES | ON/OFF contrast, $\gamma$ heterogeneity (internal) |
| 4 Z lasting effect | Fig 2 monotonic trends, prolonged-transient revert (internal) |
| 5 EEG + energy | CP5 = max channel + TVB EEG monitor (Stage 7) |
| 6 Closed loop | Feedback/duty-cycle/$\tau$-$\beta$ reproductions + patient sweeps (internal) |
| 7 TVB reference | Backbone cross-check (validates 1, 2, 5) |
