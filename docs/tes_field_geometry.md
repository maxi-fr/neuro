# tES field geometry — why suppression collapsed, and the fix

**Status:** root cause found and fixed in `src/neuro/connectome.py`; open-loop suppression
verified end-to-end. The modelling decisions the note left open are now made and shipped
(§9). The linear MPC's harmful failure (§9.6) is **resolved**: it was an under-identified input
map, not the cost function. With the identification fixed the linear MPC suppresses the seizure
network-wide (31 → 1 seizing regions, focus silenced) and beats the threshold-controller
baseline — see §9.8. Open: the 99 % stimulation duty cycle, and a hyperparameter sweep on the
corrected data.
**Date:** 2026-07-31, updated 2026-08-01
**Supersedes the root-cause section of** [`kcl_control_authority_investigation.md`](kcl_control_authority_investigation.md).

## TL;DR

`gamma` was never a physically meaningless map — it was a map built on **electrodes that were
all sitting at the centre of the head**. TVB's `eeg_unitvector_62` sensor file stores *unit
vectors* (radius exactly 1.0) in a *permuted axis frame*, and `_compute_gamma` was subtracting
those raw rows from region centroids measured in millimetres. Every electrode therefore landed
within ~1 mm of the origin, so every `gamma` row was the same radial function of a region's
distance from the head centre. That is exactly the pathology the previous note measured:
pairwise row cosine 0.9994–0.9999, `gammaᵀ` singular values `[3.13, 0.045, 0.011]`, and 98.5 %
of the current→brain authority parked in the common-mode direction KCL forbids.

With the geometry corrected, a KCL-legal montage recovers **81–99 %** of its authority in the
zero-sum subspace, and the plant turns out to be *comfortably* controllable:

- **1 mA cathodal at TP9 (extracephalic return) abolishes the seizure network-wide** — 28 → 0
  seizing regions, sustained.
- The dose–response reproduces Yu 2024 Fig. 4a: 0.1–0.35 mA confines oscillation to the EZ/PZ,
  ≥0.5 mA silences the focus too.
- Reversing polarity spreads the seizure bilaterally (61–64 regions), matching the paper's
  cathodal-inhibits / anodal-excites claim.

So neither KCL nor the MPC nor the Jansen–Rit parameters were the blocker. Both of the
hypotheses in the prompt turn out to be partly right for the same underlying reason (below),
and the previous note's conclusion — "the fix is a physically-valid `gamma` (lead field,
dipolar sources)" — was aiming at a much bigger rebuild than was needed.

## 1. The two coordinate bugs

### 1a. Unit vectors differenced against millimetres (primary)

```text
sensor locations (eeg_unitvector_62): |x| = 1.000 for all 62 rows
region centres (connectivity):        |x| = 13.6 .. 82.8 mm
```

`_compute_gamma` computed `norm(centres - sensors.locations[i])`. Since `|electrode| ≈ 1` and
`|centre| ≈ 10–83`, that distance is `≈ |centre|` **regardless of which electrode is named**.
Consequences:

- Electrode choice was a near-no-op. The measured pairwise electrode distances of the
  configured `CP5/T7/F9` montage were **0.5, 1.2 and 0.8 mm**.
- `gamma` became a radial kernel centred on the head's centroid, i.e. *deep* regions got the
  most drive and cortex the least — backwards for tES.
- All rows near-identical → their differences (the only thing a zero-sum montage can command)
  ≈ 0 → the KCL collapse.

### 1b. Permuted axis frame (secondary, but fatal on its own)

The two datasets do not share an axis convention:

| frame | axis 0 | axis 1 | axis 2 |
| --- | --- | --- | --- |
| sensors (`eeg_unitvector_62`) | left + | posterior + | superior + |
| connectome (`centres`) | anterior + | left + | superior + |

Evidence: `CP5 = [+0.888, 0.341, 0.309]` vs `CP6 = [-0.888, …]` (they differ on sensor axis 0),
while left-hemisphere region centroids average `y = +28.2` and right-hemisphere `y = −28.1`
(they differ on connectome axis 1). `Fp1/Fp2` sit at sensor `y = −0.95` and `O1/O2` at
`y = +0.95`, fixing sensor axis 1 as anterior–posterior.

The transform is therefore `x_conn = −y_sens`, `y_conn = +x_sens`, `z_conn = +z_sens`.

**Independent validation.** The EEG gain `L` is a witness that was built without any of this
geometry: a scalp channel's largest lead-field entry must belong to a region physically near
that electrode. Scoring all 48 axis permutations × sign flips by mean electrode-to-peak-region
distance picks exactly one winner:

| transform | mean distance to own lead-field peak |
| --- | --- |
| `perm=(1,0,2)`, `signs=(−1,1,1)`, R = 80 mm | **28.6 mm** |
| next best | 64.3 mm |
| worst | 160.7 mm |

and every spot-check lands where a neurologist would put it:

```text
CP5 -> lPCI (19 mm)      CP6 -> rPCI (17 mm)     T7  -> lA2 (22 mm)
Fp1 -> lPFCPOL (17 mm)   O1  -> lV2 (11 mm)      Cz  -> lPMCM / rPMCM (42 mm)
```

Note this also *confirms* the existing `_mirror_partner_permutation` hack in `_build_eeg_gain`:
the sensor labels and locations are mutually consistent, so it was correct to conclude that the
**projection matrix** was the mirrored one.

## 2. Authority recovered

Zero-sum authority = `σ₀(P·gammaᵀ) / σ₀(gammaᵀ)` where `P = I − 11ᵀ/n` projects onto the
KCL-legal current subspace. This is the fraction of the current→brain map that survives
`sum(u) = 0`.

| montage | model | zero-sum authority |
| --- | --- | --- |
| `CP5/T7/F9` (buggy geometry) | analytical, s=15 | **0.005** |
| `CP5/T7/F9` (buggy geometry) | gaussian, s=20 | **0.014** |
| `CP5/T7/F9` (fixed) | analytical, s=15 | 0.221 |
| `CP5/CP6` (fixed) | gaussian, s=40 | 0.898 |
| `CP5/T8/Fp2` (fixed) | gaussian, s=20 | 0.996 |
| `TP9/CP6` (fixed) | analytical, s=15 | 0.94 |

(The `gaussian` rows are a historical record; that model has since been removed — see §9.2.)

## 3. Electrode choice — the prompt's second hypothesis: **confirmed**

Under the corrected geometry, the configured montage is a poor choice on two counts.

**It is not near the focus.** Distances from electrode to the EZ/PZ centroids
(`lHC, lPHC, lAMYG, lTCI, lTCV`), in mm:

| electrode | lHC | lPHC | lAMYG | lTCI | lTCV |
| --- | --- | --- | --- | --- | --- |
| **TP9** | 60 | 49 | 58 | **13** | 35 |
| F9 | 73 | 63 | 52 | 66 | 71 |
| T7 | 71 | 68 | 70 | 51 | 65 |
| CP5 | 75 | 79 | 86 | 68 | 76 |
| CP6 | 94 | 108 | 110 | 140 | 120 |

`TP9` dominates — and this agrees with [`eeg_montage_selection.md`](eeg_montage_selection.md),
which independently found `TP9` to be the strongest *lead-field* electrode for `lPHC`/`lTCI`.
The paper uses CP5 because its `γ` came from a ROAST head model with properly MNI-registered
region coordinates; in this connectome's geometry the mesial-temporal electrode is TP9.

**Its three electrodes are mutually adjacent.** `CP5`, `T7` and `F9` are all left
temporo-frontal. A zero-sum montage can only command *differences* between electrode kernels,
so clustering them minimises exactly the quantity the controller needs. Any montage with a
distant return does better.

A knock-on: the configs used `gaussian` with `gamma_spread: 15–20`. With real distances, the
focus is 50–90 mm from the nearest electrode and `exp(−75²/(2·20²)) ≈ 9e-4` — the Gaussian
kernel delivers essentially **zero** drive to a deep focus. This is one of the reasons the
Gaussian model was removed outright in favour of `analytical` (§9.2).

## 4. The Jansen–Rit parameters — the prompt's first hypothesis: **not the blocker**

Every experiment in this note ran on the *current* slow-propagation parameters
(`K = 0.60`, `sigma = 280`, `initial_state: rest`, EZ/PZ `A = 3.6/3.4`). Those parameters give
the paper-like slow march — 4 seizing regions at t = 2.5 s, 28 by t = 15 s, all left hemisphere
— and they are fully suppressible. So the change away from the old instant-spread setup did not
cause the regression.

It is still *relevant*, just not causal: the paper is explicit that once oscillation is
established network-wide, stimulation can no longer abort it (Fig. 5g), so slow propagation is
what makes a closed-loop trigger meaningful at all. It widens the window rather than closing it.

## 5. The real cost of KCL: the return electrode

This is the one genuine physical constraint that survives the fix, and it is worth stating
clearly because it changes montage design.

`sum(u) = 0` means current injected at the cathode must return somewhere. Where it returns, the
drive is **anodal** — depolarising, seizure-*promoting*. With a scalp return you trade one
hemisphere for the other:

| montage (t = 15 s, 16 s run, stim from t = 2 s) | seizing left | seizing right | focus |
| --- | --- | --- | --- |
| no stimulation | 28 | 0 | 5/5 |
| `TP9(−1) / TP10(+1)` | **0** | **30** | 0/5 |
| `TP9(−1)` / return split over `Fp2,F10,P8,O2` | 4 | 27 | 0/5 |
| `TP9(−1)` / 4×1 HD ring `T7,P7,F9,O1` | 31 | 32 | 3/5 |
| `TP9(−1)` / return spread over all 62 channels | 18 | 26 | 2/5 |
| **`TP9(−1)` / extracephalic return** | **0** | **0** | **0/5** |

The cathodal side works perfectly every time — the left hemisphere including the EZ goes silent.
What differs is the damage done by the anode. The local HD ring is the *worst* option here: its
surround electrodes sit over left cortex and ignite it directly.

The paper avoids this entirely, and says so: the anode is placed at **Ex7/Ex8**, "additional
electrode position for ROAST, non-epileptic area" (§2.3). An extracephalic return is far enough
that its potential contribution to every region is small and nearly uniform, so **no region ever
sees a positive drive** (max drive `−0.15 mV` at 1 mA, i.e. cathodal everywhere).

Those Ex7/Ex8 positions have no counterpart in TVB's 62-channel set, which is why every montage
reachable from the config *at the time of writing* was forced into an intracephalic return. That
modelling decision has since been made: a virtual extracephalic electrode, see §9.1.

## 6. Open-loop dose–response (the paper's simple protocol)

Constant current, cathode `TP9`, extracephalic return, on from t = 2 s. Regions counted as
seizing when the 1 s sliding peak-to-peak of `y = x₂ − x₃` exceeds 5 mV (`neuro.seizure`
criterion), measured at t = 15 s:

| current | focus drive (mV) | seizing regions | focus seizing | mean ptp (mV) |
| --- | --- | --- | --- | --- |
| 0 (baseline) | 0 | 28 | 5/5 | 4.01 |
| 0.10 mA | −0.15 | 3 | 3/5 | 1.35 |
| 0.20 mA | −0.29 | 2 | 2/5 | 1.13 |
| 0.35 mA | −0.51 | 1 | 1/5 | 0.96 |
| 0.50 mA | −0.73 | **0** | 0/5 | 0.78 |
| 1.00 mA | −1.47 | **0** | 0/5 | 0.76 |
| 1.00 mA **reversed** | +1.47 | **61** | 5/5 | 7.11 |

The 0.1–0.35 mA rows are the paper's Fig. 4a result — propagation blocked, oscillation confined
to EZ/PZ. The reversed-polarity row is the control that proves the effect is the modelled
polarity mechanism and not generic sigmoid saturation. (For contrast, in the *buggy* geometry a
`u = [−3, 0, +3]` command also reduced the node count — but via saturation, in the
seizure-promoting direction, which is what made the old results so hard to read.)

`CP5` as cathode also works, at roughly 2× the current of `TP9` (0.5 mA → 2 regions,
1.0 mA → 0), consistent with it being ~60 mm further from the focus.

## 7. What was changed, and what is left

### Changed (the geometry fix)

- `src/neuro/connectome.py`: new public `sensor_positions_mm()` returning electrode positions
  in the connectome frame, in millimetres, on a `SCALP_RADIUS_MM = 90` sphere. `_compute_gamma`
  now uses it for the distance-based models.
- `tests/test_connectome.py`: two regression tests — electrodes must land near their own
  lead-field peak and on the correct side of the head; and a KCL-legal two-electrode montage
  must keep distinguishable rows (pairwise cosine < 0.95) and deliver *negative* drive to the
  EZ under cathodal current. Either test fails loudly if the frame or the scaling regresses.

### Still open

1. **Lasting effects (`Z`, `C₁`/`C₂` modulation) are not implemented.** The paper's §2.4
   synaptic mechanism drives all of its *post*-stimulation results. Everything above is
   immediate-effect only, which is sufficient for suppression during stimulation (the paper's
   Fig. 4a is likewise immediate-effect only) but cannot reproduce its lasting-effect figures.
2. **`SCALP_RADIUS_MM = 90` is a modelling choice, not a measurement.** The lead-field fit in
   §1b marginally prefers 80 mm, but 80 mm puts electrodes only ~0 mm outside the outermost
   region centroid (82.8 mm). 90 mm keeps every electrode outside the cortex and is in the adult
   range. It scales absolute currents but not any conclusion here.

## 8. Reproduction

Geometry check (fails on the pre-fix code):

```python
import numpy as np
from neuro.connectome import Connectome, sensor_positions_mm

conn = Connectome.from_config({})
labels, pos = sensor_positions_mm()
dist = np.linalg.norm(conn.centres[None] - pos[:, None], axis=2)
print(dist[np.arange(62), np.abs(conn.gain).argmax(axis=1)].mean())  # 36 mm (was ~160)
```

Zero-sum authority of a montage:

```python
G = np.atleast_2d(Connectome.from_config({
    "target_electrode": ["TP9", "CP6"], "gamma_model": "analytical", "gamma_spread": 15.0,
}).gamma)
P = np.eye(len(G)) - np.ones((len(G),) * 2) / len(G)
print(np.linalg.svd((P @ G).T, compute_uv=False)[0] / np.linalg.svd(G.T, compute_uv=False)[0])
```

Open-loop suppression through the shipped API (16 s, seed 69, `K = 0.60`, `sigma = 280`):

```python
from neuro.jansen_rit import JansenRitDynamics, JansenRitParams, lfp, resting_state, simulate_network
from neuro.seizure import SEIZURE_PTP_MV, build_seizure_a_gains

conn = Connectome.from_config({"speed": 50.0, "K": 0.60, "target_electrode": ["TP9", "CP6"],
                               "gamma_model": "analytical", "gamma_spread": 15.0})
left = np.array([str(r).startswith("l") for r in conn.region_labels])
for u in ([0.0, 0.0], [-1.0, 1.0]):
    dyn = JansenRitDynamics(dt=1e-4, conn=conn, seed=69,
                            params=JansenRitParams(A=build_seizure_a_gains(conn), sigma=280.0),
                            initial_state=resting_state(conn, 1e-4))
    chunks = []
    for k in range(64):
        _, x = simulate_network(dyn=dyn, duration=0.25, u_hat_tES=np.array(u),
                                stim_window=(2.0, 16.0), t0=k * 0.25)
        chunks = [*chunks, lfp(x)[:, 1:]][-4:]
    s = np.ptp(np.concatenate(chunks, axis=1), axis=1) > SEIZURE_PTP_MV
    print(u, "left", s[left].sum(), "right", s[~left].sum())
# [0.0, 0.0]   left 28  right 0
# [-1.0, 1.0]  left  0  right 30   <- cathodal side silenced, anodal side ignited (see section 5)
```

The extracephalic-return rows in §5 and §6 are reachable from the shipped config as of §9:
`target_electrode: [TP9, CP5, EX_NECK]`.

## 9. The stimulation model, decided

### 9.1 A virtual extracephalic return, and why it is not a KCL loophole

`EXTRACEPHALIC_ELECTRODES_MM` adds one off-head electrode, `EX_NECK` at connectome coordinate
`[−60, 0, −180]` mm, usable only with `gamma_model: analytical`. With a montage
`u = [u₁ … uₙ, u_E]` and `u_E = −Σuᵢ`, the node drive is

```text
d = Σ_scalp uᵢ (γᵢ − γ_E)
```

so the question of whether a distant return is just "KCL switched off" is entirely a question
about `γ_E` — and the answer differs per field model:

| model | `γ_E` over the 76 regions (145–243 mm away) | consequence |
| --- | --- | --- |
| gaussian, `s = 40` | `≤ 1.3e-3` | `d = Σ uᵢ γᵢ` exactly — a KCL loophole |
| gaussian, `s = 15` | `≤ 4e-21` | same, more so |
| **analytical, `s = 15`** | **0.41 – 0.68** (vs TP9's 0.67 – 3.79) | a real, same-signed, near-uniform term |

Under the Coulomb kernel the return is a genuine sink whose contribution is comparable to the
far field of the scalp cathode. It is small near the focus and, crucially, never changes sign,
which is exactly the physics of a distant return and exactly what buys out the §5 cost.

There is a sharper reason to keep KCL rather than delete it. `analytical` γ is an electric
*potential*, and a potential has no absolute zero: adding a constant `C` to every `γᵢ` changes
the drive by `C·Σuᵢ`. **The model is only well-defined when `Σu = 0`.** Deleting the constraint
would not merely make control easier, it would make the plant depend on an arbitrary reference.
A virtual electrode obeying KCL with a `1/r` kernel preserves that; the Gaussian version
destroys it.

The gauge-invariant alternative — driving on `E = −∇φ` projected on a pyramidal axis rather
than on `φ` — is what a FEM leadfield would give for free; see §9.4.

Caveat worth stating in any write-up: a real neck/shoulder return does **not** produce a flat
brain potential. It drives a superior→inferior current, so inferior structures see *more* field,
and the EZ here (`lHC/lPHC/lAMYG/lTCI/lTCV`) is inferior mesial temporal. The homogeneous `1/r`
monopole understates that. Extracephalic returns are used in real tES — cerebellar tDCS with a
deltoid reference, tsDCS — precisely to avoid a second *active* cortical electrode, at the cost
of a more diffuse path, a higher current for the same cortical field, and brainstem current flow
(a raised safety question that empirical HR/BP studies have not borne out).

### 9.2 `gaussian` removed

`gaussian` is gone. It is not a potential: it does not superpose, so a montage's fields cannot
be added; it has no far field, so it cannot represent a distant return; and at the spreads the
configs used (15–20 mm) it delivered `exp(−75²/2·20²) ≈ 9e-4` to a focus 75 mm away, i.e.
nothing. The default `gamma_model` is now `analytical` and the default `gamma_spread` 15.0.

### 9.3 Montage in the configs

`[CP5, T7, F9]` → **`[TP9, CP5, EX_NECK]`**, `gamma_spread: 15.0`, `gamma_model: analytical`.
Two left-temporal cathodes plus the off-head return: 3 controls (kept, so the L1-sparsity work
still has `n_controls ≥ 3`), zero-sum authority 0.334, zero-sum condition number 1.7, and every
cathodal command tested drives no region anodally.

Verified end-to-end through the shipped API (16 s, seed 69, stim from t = 2 s, seizing regions
at t = 15 s):

| `u` (mA) | seizing left | seizing right | focus |
| --- | --- | --- | --- |
| `[0, 0, 0]` | 29 | 0 | 5/5 |
| `[−0.5, −0.5, +1.0]` | **0** | **0** | 0/5 |
| `[−1.0, −1.0, +2.0]` | **0** | **0** | 0/5 |

### 9.4 Why not SimNIBS (yet)

A FEM leadfield is a real upgrade on two axes `analytical` cannot reach: conductivity structure
(skull and CSF shunting are first-order, and the homogeneous `1/r` gets relative depth
dependence wrong), and the output quantity (E-field vector, so the cortical-normal component
can drive the model — gauge-invariant, and physiologically the right thing). SimNIBS's
`TDCSLEADFIELD` precomputes exactly the `(n_electrodes, …)` object `gamma` wants, KCL is built
into its reference convention, and it models sponge/gel electrode geometry.

It is deferred, not rejected, because:

- The cost is not the FEM (hours, once, then a static file) but the **registration**: SimNIBS
  lives in its head model's space and the TVB centres in the frame §1b had to pin down. Mapping
  element E-fields to 76 regions introduces an error that is invisible without a check — the
  EEG-gain witness of §1b is the check to reuse.
- Standard SimNIBS head meshes stop at the neck, so a genuine extracephalic pad needs an
  extended mesh.
- At 76 regions the parcellation is coarser than the field variation either model produces, so
  much of the added fidelity averages away.
- Nothing structural changes downstream: it drops in as a fourth `gamma_model` reading a
  precomputed matrix, leaving the MPC and identification untouched.

So: do the closed-loop work on `analytical`, and treat SimNIBS as the validity check for the
write-up. (Yu 2024 used ROAST, the same class of tool.)

**Update.** FEM leadfield generation is performed via ROAST in MATLAB (`matlab/generate_roast_gamma.m`), outputting `roast_gamma.npz` for the precomputed matrix model.

### 9.5 The amplitude-threshold controller

`neuro.control.AmplitudeThresholdController` implements Yu 2024 §3.2: a fixed-length constant
burst triggered whenever the trailing-window peak-to-peak of a feedback EEG channel crosses a
threshold, with no predictor in the loop — so unlike the MPC it is unaffected by how well the
tES→EEG response is identified. `configs/simulation/threshold_control.yaml` monitors TP9
(channel 27; ~12 mV ptp before spread, ~45 mV once established), threshold 20 mV, 1 s window,
1 s bursts of `[−0.5, −0.5, +1.0]` mA.

Result over a 20 s run (seed 69): **4 bursts, 4 s of stimulation total (20 % duty cycle)**,
first trigger at t = 10.5 s. The network never reaches the untreated run's 29-region spread —
it is fully silent at t = 12 s and falls back to the 4–5 focus regions between bursts. This is
the reference point the MPC has to beat, and the closest thing here to the paper's Fig. 4a.

### 9.6 Identification and MPC re-run — the linear MPC now fails *harmfully*

`data/experiment_excited` was regenerated on the corrected plant (25 × 20 s, montage
`[TP9, CP5, EX_NECK]`); every other dataset was deleted as stale (§9.7). `linear_full` was refitted
on it and `meeting_seven/full_linear_mpc.yaml` re-run. Seizing regions at t = 20 s, seed 69:

| run | control | seizing L/R | focus |
| --- | --- | --- | --- |
| no stimulation | — | 29 / 0 | 5/5 |
| threshold controller | 20 % duty, ≤ 1 mA | **5 / 0** | 4/5 |
| linear MPC, `u_max = 3` | 99 % duty, mean \|u\| 1.93 | 38 / 35 | 5/5 |
| linear MPC, `u_max = 1` | 99 % duty, mean \|u\| 0.66 | 36 / 32 | 5/5 |

The MPC does not merely fail to suppress — it drives a **bilateral** seizure worse than no
stimulation at all. Its command chatters between saturated corners of the KCL-legal box at the
100 Hz control rate (`[3, −3, 0] → [−3, 3, 0] → [−3, 0, 3] → …`; TP9 is cathodal only 55 % of
the time). Halving the bound does not help, so this is not an amplitude problem: the linear
predictor evidently believes alternating saturated currents cancel EEG power, while the plant's
firing-rate sigmoid rectifies them into net excitation — which §6 already showed is the
seizure-*promoting* direction.

So the gamma fix does not by itself rescue the MPC results; it changes the failure from "no
authority" to "authority used in the wrong direction". Worth trying next, roughly in order of
cost: a Δu (rate) penalty to stop the chattering, which the current cost has no term for; a much
larger `w_u` now that ~0.5 mA is known to suffice; and checking the sign and magnitude of the
identified `B` against a direct open-loop step response before trusting any closed-loop number.
The threshold controller is a working baseline in the meantime, and being predictor-free it
isolates whether a given failure is the plant or the model.

### 9.7 Stale data and configs deleted

Everything identified against the collapsed `gamma` or the old `CP5/T7/F9` montage was removed
(≈83 GB).

| deleted | why |
| --- | --- |
| `data/experiment_excited_pre_tes_fix` | the pre-fix identification set |
| `data/experiment_excited_analytical` | duplicate of `experiment_excited` once `analytical` became the default |
| `artifacts/pre_kirchhoff_wrong_stim` | archived pre-Kirchhoff runs |
| `configs/simulation/experiment_excited_analytical.yaml` | the gamma-model comparison track |
| `configs/simulation/closed_loop_analytical.yaml` | pointed at timestamped artifacts that no longer exist |
| `configs/simulation/l1_{before,after}.yaml` | same dangling artifact, and the old montage |
| `configs/nn_predictor/{linear,mlp3}_analytical.yaml` | their datasets are gone; `meeting_seven/{linear,nonlinear}_full.yaml` already cover the canonical set |

What survives: `data/experiment_excited` (regenerated), `configs/simulation/experiment_excited.yaml`
as the single identification config, `threshold_control.yaml`, the `meeting_seven/` MPC set, and
the fresh `artifacts/{meeting_seven_experiment,threshold_control,mpc_linear_full_tes_fix}`.

Two knock-ons: the reproduction snippets in
[`kcl_control_authority_investigation.md`](kcl_control_authority_investigation.md) reference
`l1_before.yaml` and the archived artifacts, so that (already superseded) note is now a
historical record only; and rebuilding the L1-sparsity experiment means re-deriving its configs
from `meeting_seven/full_linear_mpc.yaml`.

### 9.8 Why the MPC failed: what forecast MSE cannot see

§9.6 left the linear MPC driving a bilateral seizure and guessed at three fixes. Measuring first
settles it: the QP is not malfunctioning, it is correctly minimising a cost built on a model whose
input map is wrong. Method — fork the plant into two copies sharing state *and* noise stream, drive
one with a command and one with zero, and compare the difference against the same CasADi unroll the
MPC uses (`NNSymbolicModel`). Three findings, in causal order.

**1. `B` is under-identified by construction.** In `data/experiment_excited`, the control explains
**5.9e-4** of the one-step EEG variance (one-step R²: 0.9766 AR-only → 0.9772 with `u`; over 20
steps, 0.256 → 0.272). More than 99.9 % of the training loss is autoregression, so any `B`
consistent with that residual is equally optimal to the optimiser. The excitation is not at fault:
RAS covers the 2-D Kirchhoff plane isotropically (in-plane covariance condition number 1.003).

**2. The identified `B` is uncorrelated with the truth in the direction that suppresses.** At k=1
(pure `B`, no compounding):

| command | cos(model, plant) | gain ratio |
| --- | --- | --- |
| `[+3,-3,0]` | 0.75 | 0.35 |
| `[-0.5,-0.5,+1.0]` (the suppressing one) | −0.05 … +0.12 | 1.1–1.6 |

By k=10 the TP9 sign is inverted: plant −4.6, model +5.7.

**3. The model rewards chatter.** Under a 50 Hz alternating `[+3,-3,0]/[-3,+3,0]` command the model
predicts `|Δy|` = 3.4–13 (gain **0.05** — it believes the currents cancel), while the plant produces
`|Δy|` = 70–140, the largest deviation of any command tested, TP9 +41 against a baseline RMS of 2.5.
That is §6's rectification. The QP prefers chatter because its model tells it chatter is free.

**A linear predictor cannot be fixed here.** Cancellation of alternating input is a property of
*any* linear model; rectification by the firing-rate sigmoid is an even-order nonlinear effect. No
hyperparameter setting closes that gap — only bandwidth-limiting the controller keeps it inside the
regime where the linear model is valid.

#### What changed

- **A slew-rate penalty.** `w_du` on every MPC controller adds
  `w_du * Σ ||u_k − u_{k−1}||²`, with `u_{−1}` read off the window-state parameter so the penalty is
  continuous across solves rather than only within a horizon.
- **Broadband excitation.** `hold_ms` now accepts a list; the identification configs use
  `[10, 50, 200]` ms. A single 10 ms hold excites only around the control rate, while the MPC
  commands near-constant currents over a whole horizon — so the low-frequency gain it relies on was
  identified from almost no data.
- **A paired baseline dataset.** `configs/simulation/experiment_baseline.yaml` is the same plant and
  seeds with `amp: 0.0`. The plant's noise draw does not depend on `u`, so trial *k* of each set
  shares a noise realisation (verified: pre-stimulus EEG differs by exactly 0.0).
- **A response metric, since removed.** A forked-probe score (fork the plant sharing state *and*
  noise, drive one copy and not the other, compare the model's incremental prediction against the
  difference) was built to measure the input map directly, on a scale where `0` is exact and `1` is
  what predicting no response scores. It did its diagnostic job — the pre-fix `linear_full` scored
  **1.155**, worse than silence, corroborating findings 2 and 3 at 880 operating points. It was
  then **deleted**, because it proved non-monotone against closed-loop outcome (below) and so could
  not be trusted to rank anything. The numbers it produced are retained here as evidence.
- **`train_and_save_predictor` returns the *normalized* MSE** (raw MSE still recorded in
  `training_stats.json`), which is comparable across trials whose window offsets differ.
  `configs/nn_predictor/sweep_nn_predictor.yaml` is the sweep on the corrected data.
- **A truncation bug, found on the way.** Every predictor config carried
  `n_steps: 500` — 5 s — with a comment dating from the old 100 × 5 s dataset. Since the
  2026-07-30 switch to 25 × 20 s trials this silently discarded 75 % of every trajectory, including
  nearly all of the seizure the MPC has to suppress. All predictor configs now use `n_steps: null`.
  Every model trained between 2026-07-30 and this note saw only the first 5 s.

#### Results

Identification was re-run on the corrected data (`linear_full`'s architecture unchanged, so this
isolates the *data* fixes from any architecture search). The response score below is the
forked-probe metric described above, on 880 train / 120 test probes; `1.0` is what predicting no
response scores.

| model | train | test |
| --- | --- | --- |
| `linear_full`, pre-fix data | 1.155 | 1.207 |
| `linear_full`, corrected data | **0.968** | **0.968** |

Closed loop, 20 s, seed 69, seizing regions from the trailing 1 s peak-to-peak at `t_end`:

| run | seizing L/R | focus | mean \|u\| | duty |
| --- | --- | --- | --- | --- |
| no stimulation | 31 / 0 | 5/5 | 0 | 0 % |
| threshold controller (§9.5) | 5 / 0 | 4/5 | ≤ 1 | 20 % |
| linear MPC, pre-fix (§9.6) | 38 / 35 | 5/5 | 1.93 | 99 % |
| **linear MPC, refit, `w_du = 0`** | **1 / 0** | **0/5** | 1.57 | 99 % |
| linear MPC, refit, `w_du = 10` | 4 / 0 | 2/5 | 1.22 | 99 % |
| linear MPC, refit, `w_du = 100` | 36 / 35 | 5/5 | 1.68 | 99 % |
| linear MPC, refit, `w_du = 1000` | 26 / 30 | 5/5 | 1.88 | 99 % |

**The identification fixes repaired the MPC on their own**, from worse-than-untreated to better
than the threshold baseline, with the focus silenced. No change to the cost function was required.
A 2x2 ablation (below) separates which fix did what.

**The slew-rate penalty is not the fix, and at scale it is harmful.** §9.6 proposed it first;
that was wrong. Raising `w_du` monotonically degrades suppression and past ~100 the run goes
bilateral again — a heavy rate penalty pins the controller to a near-constant command, and a
sustained command in the anodal direction is precisely §6's seizure-promoting case. The chatter
was a *symptom* of the wrong input map, not an independent defect. `w_du` stays in the API and
defaults to `0`; `w_du = 10` is a real trade (22 % less mean amplitude for 3 more seizing regions)
but nothing above that is usable.

The open weakness is the **99 % duty cycle**: the MPC suppresses harder than the threshold
controller but stimulates continuously, where the threshold controller uses 20 %. That is what
`w_u` / `w_u_l1` are for and wants its own sweep.

#### The curriculum

The horizon-length curriculum destabilises this fit. On the corrected (longer, seizure-heavy)
data the linear refit's training MSE went 0.38 → 3e7 the moment `L` reached the full 20-step
unroll; only early stopping rescued it, so the shipped artifact is the epoch-161 model selected
mid-ramp. Validation is always scored at the full horizon (`val_mask_jax = jnp.ones(horizon)`),
so that selection is sound.

Turning the curriculum off entirely was tested head-to-head anyway
(`TrainingConfig.curriculum_max_steps` caps the rollout length the ramp approaches; `1` is plain
one-step training, since a ramp from 1 to 1 never grows). It is decisively worse:

| training | forecast nmse | response score | closed loop |
| --- | --- | --- | --- |
| curriculum | **0.64** | **0.968** | **1 / 0, focus 0/5** |
| one-step (`curriculum_max_steps: 1`) | 1.08 | 0.994 | 37 / 33, focus 5/5 |

One-step training reproduces the original harmful failure exactly. With no multi-step term in the
loss nothing penalises rollout drift, so the 20-step forecast the MPC actually consumes ends up
worse than predicting the mean (`nmse > 1`) and the response score returns to ~1.0. The earlier
impression that the curriculum was not helping closed-loop performance dates from the pre-fix data,
where every model was broken for the independent reason above; once the input map is identified the
curriculum is load-bearing. `sweep_nn_predictor.yaml` therefore keeps it on.

The instability at `L = horizon` is still unresolved and is worth revisiting. Since the curriculum
is what helps and only its *tail* diverges, the natural test is a lower cap —
`curriculum_max_steps: 10` ramps `1 -> 10` and never enters the regime that blows up, which is a
different (and more promising) experiment than either one-step training or an earlier ramp to the
full horizon.

#### Which data fix did it — the 2x2

The refit changed two things at once (the dataset's excitation and `n_steps`), so they were
separated by training the two missing cells with byte-identical architecture and hyperparameters
and judging them by closed loop:

| cell | excitation | `n_steps` | nmse | response score | seizing L/R | focus |
| --- | --- | --- | --- | --- | --- | --- |
| A | uniform 10 ms | 500 | (val 3.05) | 1.155 | 38 / 35 | 5/5 |
| B | uniform 10 ms | `null` | 0.81 | 1.013 | **0 / 26** | **0/5** |
| C | mixed 10/50/200 | 500 | 3.98 | 1.177 | 33 / 1 | 5/5 |
| D | mixed 10/50/200 | `null` | **0.64** | **0.968** | **1 / 0** | **0/5** |

**Both were necessary; neither alone sufficient.**

- **`n_steps` decides whether the focus can be silenced at all.** Both truncated cells leave it
  seizing 5/5; both untruncated cells silence it 0/5. A model trained on the first 5 s has never
  seen a seizing network, so its 20-step forecast at t = 10–20 s is pure extrapolation — and
  better excitation cannot compensate (cell C, mixed excitation on truncated data, is the worst
  nmse of the four).
- **Excitation bandwidth decides whether the effect stays lateralised.** Cell B silences the whole
  left hemisphere *and* the focus, then ignites 26 regions on the right — exactly §5's anode
  trade-one-hemisphere-for-the-other signature. Mixed holds are what turn that into cell D's 1/0.

#### The response metric does not survive this

Ranked by response score the order is D (0.968) < one-step (0.994) < B (1.013) < A (1.155) <
C (1.177). By closed-loop total seizing it is D (1) < B (26) < C (34) < one-step (70) < A (73).
The metric places one-step **second best** when it is fourth of five and drives a bilateral
seizure — it is non-monotone against ground truth, not merely narrow-margined. Plain forecast
nmse orders the same five better (only C and one-step swap).

The metric was therefore **removed from the repo** (`neuro.response_probe`,
`scripts/generate_response_probes.py`, its test and its dataset). It had real diagnostic value —
a score above 1.0 flagged every model that failed, and it is what exposed the original input-map
defect — but a measurement that cannot be trusted to rank is a liability sitting in a sweep
objective, and keeping it invited exactly that. The findings it produced are recorded above; the
apparatus is not worth maintaining for them.

What remains: forecast NMSE as the cheap proxy, and **closed-loop suppression as the only
criterion shown to be reliable** — ~7-10 min per run against ~50 min to train, i.e. a ~20 %
surcharge. Verify sweep winners in closed loop rather than trusting the proxy.
